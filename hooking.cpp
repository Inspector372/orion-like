/*
	hooking.cpp

	There are APIs which need to be hooked in device-level,
	and APIs which is okay to be hooked in runtime-level.
	* NO HOOK * Device Management: cuFlushGPUDirectRDMAWrites() is a blocking API, otherwise no hooking is needed.
	* NO HOOK * Context Management(Driver): The hooking itself is not a big deal, but launching in OS thread will struggle to give its error state, etc... to user thread.
											For now just leave it as no hooking.
	* DEVICE HOOK * Module Management(Driver): 	If a tenant loads a module, then the OS must be able to use it(because hooking is ultimately coming to OS).
											This might need hooking the whole API.
											Some module functions collide with cuLaunchKernel(). Inspect this and hook if needed.
	* NO HOOK * Library Management(Driver): Almost asme as module management part.
											One thing is there are some functions that feeds configuration to kernels,
											but need to think if OS should ignore such user-level configurations.
											For now just leave it as no hooking.
	* NO HOOK * Error Handling:				This will be implemented later with error queue for clients.
	* DEVICE HOOK * Memory Management:		Need to be hooked, and need to be distinguished.
											The top design is differtiantiate between 'pure-memory' operation and 'kernel-may-involved' operations.
											I want to make a separate work queue dealing with only memory operations because that is how CUDA works.
											As mentioned earlier, some operations like cudaMemset() or cudaMemcpy(..., DtoD) involves kernel launch, so they need to be treated like kernel launch.
	* NO HOOK * Stream & Event Management: 	Currently, we ignore the stream and put everything to our stream,
											However, when events and syncrhonizations are involved, we need to hook events.
											Will be implemented later after all basic operations are done.
	
	Cooperative Launch: 					Later.

	Graph Management:						Later.

	
*/

#include <dlfcn.h>
#include <stdio.h>
#include <sys/types.h>
#include <syscall.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <queue>
#include <pthread.h>
#include <assert.h>
#include <string.h>

#include "hooking.h"

/* Just error handling of c++ name mangling. */
typedef cudaError_t (*PFN_cudaLaunchKernel)(
    const void*, dim3, dim3, void**, size_t, cudaStream_t
);

using namespace std;

/* switch variable for no-hook launches(for initial launches) */
bool no_hook = true;

pthread_t* thread_ids;

/* 
	Work queue, which corresponds to the 'stream'.
    Each time the client creates a stream, a work queue is newly created.
    Non-stream operations go to the default queue.
*/
client_streams* work_queue;
pthread_mutex_t** work_queue_mutex;


/*
	Real instances of cuda APIs.
*/
cudaError_t (*real_cudaMemset)(void* devPtr, int value, size_t count) = NULL;
cudaError_t (*real_cudaMemsetAsync)(void* devPtr, int value, size_t count, cudaStream_t stream) = NULL;
cudaError_t (*real_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = NULL;
CUresult (*real_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) = NULL;
CUresult (*real_cuLaunchKernelEx)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) = NULL;
CUresult (*real_cuLaunchKernel_ptsz)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) = NULL;
CUresult (*real_cuLaunchKernelEx_ptsz)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) = NULL;
CUresult (*real_cuGraphLaunch)(CUgraphExec hGraphExec, CUstream hStream) = NULL;
CUresult (*real_cuLaunchCooperativeKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**) = NULL;

/*
	Main cuda address resolving API.
*/
typedef CUresult (CUDAAPI *cuGetProcAddress_t)(const char*, void**, int, cuuint64_t);
typedef CUresult (CUDAAPI *cuGetProcAddress_v2_t)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

/*
	Real instance of dlsym().
*/
static void* real_dlsym(void* handle, const char* symbol) {
    static void* (*loader)(void*, const char*) =
        (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    return loader ? loader(handle, symbol) : NULL;
}
void* cu_handle = NULL;


/*
	variable for identifying if current launch is from cudaLaunchKernel(), or separate mechanism
*/
bool cudaLaunchKernel_launched[THREAD_NUM];


// Slot zero always represents this client's default stream. Explicit streams
// can also be shared by clients; route them to the creator's FIFO.
static stream_queue_ptr find_stream(int client, CUstream stream) {
    bool use_default = !stream || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD;
    for (int n = 0; n < THREAD_NUM; ++n) {
        int owner = (client + n) % THREAD_NUM;
        pthread_mutex_lock(work_queue_mutex[owner]);
        stream_queue_ptr found;
        if (use_default) found = work_queue[client].front();
        else for (const stream_queue_ptr& queue : work_queue[owner]) {
            if (queue->gpu == stream) { found = queue; break; }
        }
        pthread_mutex_unlock(work_queue_mutex[owner]);
        if (found || use_default) return found;
    }
    // Existing streams may come from CUDA libraries or initialization before
    // client registration. Use the client control FIFO, retaining the GPU handle.
    pthread_mutex_lock(work_queue_mutex[client]);
    stream_queue_ptr fallback = work_queue[client].front();
    pthread_mutex_unlock(work_queue_mutex[client]);
    return fallback;
}

static uint64_t submit(const stream_queue_ptr& queue, queue_record record) {
    pthread_mutex_t* lock = work_queue_mutex[queue->owner];
    pthread_mutex_lock(lock);
    uint64_t ticket = 0;
    if (!queue->closing) {
        queue->records.push(record);
        ticket = ++queue->issued;
        if (record.type == RECORD_STREAM_DESTROY && queue != work_queue[queue->owner].front())
            queue->closing = true;
    }
    pthread_mutex_unlock(lock);
    return ticket;
}

static void wait_for_submission(const stream_queue_ptr& queue, uint64_t ticket) {
    while (queue->retired.load(std::memory_order_acquire) < ticket) sched_yield();
}

static CUresult submit_request(const stream_queue_ptr& queue, record_type type,
                               driver_request& request) {
    record_data data;
    data.request = &request;
    if (!submit(queue, {type, data})) return CUDA_ERROR_INVALID_HANDLE;
    while (!request.done.load(std::memory_order_acquire)) sched_yield();
    return request.status;
}

struct launch_ticket { stream_queue_ptr queue; uint64_t number; };
static thread_local std::vector<launch_ticket> runtime_submissions;

/*
	** intercept part **

	Reference:
	curtSCHED: Architecture-Independent Real-Time GPU Scheduling via Statistical Deferrable Servers by Hao Zhang, Frank Muelle(https://github.com/zhanghao5/curtSCHED/blob/main/cusched.cpp)

	cuda resolves names dynamically using this path:
	1. dlsym() on cuGetProcAddress_v2
	2. cuGetProcAddress_v2 on cuGetProcAddress
	3. cuGetProcAddress on cuLaunchKernel(and a lot of other cuda driver functions)

	So what we do is:
	1. hooked dlsym() and return our own (hook)cuGetProcAddress_v2 address.
	2. cuGetProcAddress_v2() is invoked and we hook it.
	3. hooked cuGetProcAddress_v2() and return our own (hook)cuGetProcAddress address.
	4. cuGetProcAddress() is invoked and we hook it.
	5. hooked cuGetProcAddress() and return our own (hook)cuLaunchKernel address.
	6. further all cuLaunchKernel is hooked.

*/
// Resolve driver symbols without passing through our dlsym hook.
static void* memory_driver_symbol(const char* symbol) {
    static void* (*loader)(void*, const char*) =
        (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    static void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    return loader && driver ? loader(driver, symbol) : NULL;
}

static int memory_client_idx() {
    // Startup, scheduler and unregistered threads must not self-enqueue.
    if (__atomic_load_n(&no_hook, __ATOMIC_ACQUIRE) || !thread_ids || !work_queue || !work_queue_mutex) return -1;
    for (int i = 0; i < THREAD_NUM; ++i)
        if (pthread_equal(pthread_self(), thread_ids[i])) return i;
    return -1;
}

extern "C" {

CUresult CUDAAPI cuStreamCreate(CUstream* phStream, unsigned int flags) {
    static CUresult (CUDAAPI *real)(CUstream*, unsigned int) =
        (CUresult (CUDAAPI *)(CUstream*, unsigned int))memory_driver_symbol("cuStreamCreate");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(phStream, flags);
    driver_request request;
    request.stream_out = phStream;
    request.flags = flags;
    return submit_request(find_stream(idx, NULL), RECORD_STREAM_CREATE, request);
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) {
    static CUresult (CUDAAPI *real)(CUstream*, unsigned int, int) =
        (CUresult (CUDAAPI *)(CUstream*, unsigned int, int))memory_driver_symbol("cuStreamCreateWithPriority");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(phStream, flags, priority);
    driver_request request;
    request.stream_out = phStream;
    request.flags = flags;
    return submit_request(find_stream(idx, NULL), RECORD_STREAM_CREATE, request);
}

CUresult CUDAAPI cuStreamDestroy_v2(CUstream stream) {
    static CUresult (CUDAAPI *real)(CUstream) =
        (CUresult (CUDAAPI *)(CUstream))memory_driver_symbol("cuStreamDestroy_v2");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(stream);
    if (!stream || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD)
        return CUDA_ERROR_INVALID_HANDLE;
    driver_request request;
    request.destroy_stream = real;
    request.stream = stream;
    return submit_request(find_stream(idx, stream), RECORD_STREAM_DESTROY, request);
}

CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
    static CUresult (CUDAAPI *real)(CUstream) =
        (CUresult (CUDAAPI *)(CUstream))memory_driver_symbol("cuStreamSynchronize");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(stream);
    driver_request request;
    request.stream = stream;
    return submit_request(find_stream(idx, stream), RECORD_STREAM_SYNC, request);
}

CUresult CUDAAPI cuEventCreate(CUevent* phEvent, unsigned int flags) {
    static CUresult (CUDAAPI *real)(CUevent*, unsigned int) =
        (CUresult (CUDAAPI *)(CUevent*, unsigned int))memory_driver_symbol("cuEventCreate");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(phEvent, flags);
    driver_request request;
    request.event_out = phEvent;
    request.flags = flags;
    return submit_request(find_stream(idx, NULL), RECORD_EVENT_CREATE, request);
}

CUresult CUDAAPI cuEventDestroy_v2(CUevent event) {
    static CUresult (CUDAAPI *real)(CUevent) =
        (CUresult (CUDAAPI *)(CUevent))memory_driver_symbol("cuEventDestroy_v2");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(event);
    driver_request request;
    request.event = event;
    request.destroy_event = real;
    return submit_request(find_stream(idx, NULL), RECORD_EVENT_DESTROY, request);
}

CUresult CUDAAPI cuEventQuery(CUevent event) {
    static CUresult (CUDAAPI *real)(CUevent) =
        (CUresult (CUDAAPI *)(CUevent))memory_driver_symbol("cuEventQuery");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(event);
    driver_request request;
    request.event = event;
    return submit_request(find_stream(idx, NULL), RECORD_EVENT_QUERY, request);
}

CUresult CUDAAPI cuEventRecord(CUevent event, CUstream stream) {
    static CUresult (CUDAAPI *real)(CUevent, CUstream) =
        (CUresult (CUDAAPI *)(CUevent, CUstream))memory_driver_symbol("cuEventRecord");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(event, stream);
    driver_request request;
    request.event = event;
    request.stream = stream;
    return submit_request(find_stream(idx, stream), RECORD_EVENT_RECORD, request);
}

CUresult CUDAAPI cuEventSynchronize(CUevent event) {
    static CUresult (CUDAAPI *real)(CUevent) =
        (CUresult (CUDAAPI *)(CUevent))memory_driver_symbol("cuEventSynchronize");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(event);
    driver_request request;
    request.event = event;
    return submit_request(find_stream(idx, NULL), RECORD_EVENT_SYNC, request);
}

CUresult CUDAAPI legacy_cuStreamDestroy(CUstream stream) asm("cuStreamDestroy");
CUresult CUDAAPI legacy_cuStreamDestroy(CUstream stream) {
    static CUresult (CUDAAPI *real)(CUstream) =
        (CUresult (CUDAAPI *)(CUstream))memory_driver_symbol("cuStreamDestroy");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(stream);
    if (!stream || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD)
        return CUDA_ERROR_INVALID_HANDLE;
    driver_request request;
    request.destroy_stream = real;
    request.stream = stream;
    return submit_request(find_stream(idx, stream), RECORD_STREAM_DESTROY, request);
}

CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream stream) {
    static CUresult (CUDAAPI *real)(CUstream) =
        (CUresult (CUDAAPI *)(CUstream))memory_driver_symbol("cuStreamSynchronize_ptsz");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(stream);
    driver_request request;
    request.stream = stream;
    return submit_request(find_stream(idx, stream), RECORD_STREAM_SYNC, request);
}

CUresult CUDAAPI legacy_cuEventDestroy(CUevent event) asm("cuEventDestroy");
CUresult CUDAAPI legacy_cuEventDestroy(CUevent event) {
    static CUresult (CUDAAPI *real)(CUevent) =
        (CUresult (CUDAAPI *)(CUevent))memory_driver_symbol("cuEventDestroy");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(event);
    driver_request request;
    request.event = event;
    request.destroy_event = real;
    return submit_request(find_stream(idx, NULL), RECORD_EVENT_DESTROY, request);
}

CUresult CUDAAPI cuEventRecord_ptsz(CUevent event, CUstream stream) {
    static CUresult (CUDAAPI *real)(CUevent, CUstream) =
        (CUresult (CUDAAPI *)(CUevent, CUstream))memory_driver_symbol("cuEventRecord_ptsz");
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    int idx = memory_client_idx();
    if (idx < 0) return real(event, stream);
    driver_request request;
    request.event = event;
    request.stream = stream;
    return submit_request(find_stream(idx, stream), RECORD_EVENT_RECORD, request);
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    static CUresult (CUDAAPI *real_cuMemAlloc)(CUdeviceptr*, size_t) =
        (CUresult (CUDAAPI *)(CUdeviceptr*, size_t))memory_driver_symbol("cuMemAlloc_v2");
    if (real_cuMemAlloc == NULL) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real_cuMemAlloc(dptr, bytesize);

    CUresult status = CUDA_SUCCESS;
    record_cuMemAlloc new_record = {dptr, bytesize, &status};
    union record_data new_record_data;
    new_record_data.r_cuMemAlloc = new_record;
    queue_record new_qrecord = {RECORD_CUMEMALLOC, new_record_data};

    stream_queue_ptr queue = find_stream(idx, NULL);
    wait_for_submission(queue, submit(queue, new_qrecord));
    return status;
}


CUresult CUDAAPI cuMemAllocHost(void** pp, size_t bytesize) {
    static CUresult (CUDAAPI *real_cuMemAllocHost)(void**, size_t) =
        (CUresult (CUDAAPI *)(void**, size_t))memory_driver_symbol("cuMemAllocHost_v2");
    if (real_cuMemAllocHost == NULL) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real_cuMemAllocHost(pp, bytesize);

    CUresult status = CUDA_SUCCESS;
    record_cuMemAllocHost new_record = {pp, bytesize, &status};
    union record_data new_record_data;
    new_record_data.r_cuMemAllocHost = new_record;
    queue_record new_qrecord = {RECORD_CUMEMALLOCHOST, new_record_data};

    stream_queue_ptr queue = find_stream(idx, NULL);
    wait_for_submission(queue, submit(queue, new_qrecord));
    return status;
}


CUresult CUDAAPI cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
    static CUresult (CUDAAPI *real_cuMemHostAlloc)(void**, size_t, unsigned int) =
        (CUresult (CUDAAPI *)(void**, size_t, unsigned int))memory_driver_symbol("cuMemHostAlloc");
    if (real_cuMemHostAlloc == NULL) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real_cuMemHostAlloc(pp, bytesize, flags);

    CUresult status = CUDA_SUCCESS;
    record_cuMemHostAlloc new_record = {pp, bytesize, flags, &status};
    union record_data new_record_data;
    new_record_data.r_cuMemHostAlloc = new_record;
    queue_record new_qrecord = {RECORD_CUMEMHOSTALLOC, new_record_data};

    stream_queue_ptr queue = find_stream(idx, NULL);
    wait_for_submission(queue, submit(queue, new_qrecord));
    return status;
}


CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    static CUresult (CUDAAPI *real_cuMemFree)(CUdeviceptr) =
        (CUresult (CUDAAPI *)(CUdeviceptr))memory_driver_symbol("cuMemFree_v2");
    if (real_cuMemFree == NULL) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real_cuMemFree(dptr);

    CUresult status = CUDA_SUCCESS;
    record_cuMemFree new_record = {dptr, &status};
    union record_data new_record_data;
    new_record_data.r_cuMemFree = new_record;
    queue_record new_qrecord = {RECORD_CUMEMFREE, new_record_data};

    stream_queue_ptr queue = find_stream(idx, NULL);
    wait_for_submission(queue, submit(queue, new_qrecord));
    return status;
}


CUresult CUDAAPI cuMemFreeHost(void* p) {
    static CUresult (CUDAAPI *real_cuMemFreeHost)(void*) =
        (CUresult (CUDAAPI *)(void*))memory_driver_symbol("cuMemFreeHost");
    if (real_cuMemFreeHost == NULL) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real_cuMemFreeHost(p);

    CUresult status = CUDA_SUCCESS;
    record_cuMemFreeHost new_record = {p, &status};
    union record_data new_record_data;
    new_record_data.r_cuMemFreeHost = new_record;
    queue_record new_qrecord = {RECORD_CUMEMFREEHOST, new_record_data};

    stream_queue_ptr queue = find_stream(idx, NULL);
    wait_for_submission(queue, submit(queue, new_qrecord));
    return status;
}


cudaError_t cudaDeviceSynchronize(void) {
    int idx = memory_client_idx();
    if (idx < 0) {
        void* (*loader)(void*, const char*) =
            (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        static cudaError_t (*real_sync)(void) =
            (cudaError_t (*)(void))loader(RTLD_NEXT, "cudaDeviceSynchronize");
        return real_sync ? real_sync() : cudaErrorInitializationError;
    }
    driver_request request;
    return (cudaError_t)submit_request(find_stream(idx, NULL), RECORD_DEVICE_SYNC, request);
}

/*
	** Kernel-like APIs **
	Hook for cuLaunchKernel, ... etc
	They are mainly modifying GPU conternts in allocated memory.
	Kernel Queue will handle this.
*/

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
	if (real_cudaLaunchKernel == NULL) {
        real_cudaLaunchKernel = (cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t))real_dlsym(RTLD_NEXT, "cudaLaunchKernel");
        if(real_cudaLaunchKernel == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cudaLaunchKernel == NULL\n");
        }
    }
	int idx = memory_client_idx();
	if(idx < 0) {
		return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
	}
	// fprintf(stderr, "[cudaHook] caught call from %d!\n", idx);
	cudaLaunchKernel_launched[idx] = true;
	cudaError_t err = real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
	cudaLaunchKernel_launched[idx] = false;
	// fprintf(stderr, "[cudaHook] block-start from %d!\n", idx);
	for (const launch_ticket& t : runtime_submissions) wait_for_submission(t.queue, t.number);
	runtime_submissions.clear();
	// fprintf(stderr, "[cudaHook] block-end from %d!\n", idx);
	
	return err;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    if (real_cuLaunchKernel == NULL) {
        if(cu_handle == NULL) cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        real_cuLaunchKernel = (CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**))real_dlsym(cu_handle, "cuLaunchKernel");
        if(real_cuLaunchKernel == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernel == NULL\n");
        }
        if(real_cuLaunchKernel == cuLaunchKernel) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernel == cuLaunchKernel\n");
        }
    }

	int idx = memory_client_idx();
	if(idx < 0) {
		// immediately run the kernel.
		fprintf(stderr, "[cuHook] no-hook launch of %p\n", f);
		return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
	}

	fprintf(stderr, "[cuHook] caught call from %d!\n", idx);

	// inspect kernel size and setup atomization info.

	// Change: now we atomize it in block boundary always.
	// This is more logical.
	int total_grid_dim = gridDimX * gridDimY * gridDimZ;

	// atom_size is now defined as number of 'block'.
	// TODO: dynamically adjust atom_size.
	int atom_num = 8;
	int atom_size = (total_grid_dim % atom_num == 0) ? (total_grid_dim / atom_num) : (total_grid_dim / atom_num + 1);
	

	CUresult err = CUDA_SUCCESS;
	record_cuLaunchKernel new_record;
	
	assert(work_queue_mutex != NULL);
	assert(work_queue != NULL);

	new_record = {f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra, 0, (uint32_t)atom_size};
	union record_data new_record_data;
	new_record_data.r_cuLaunchKernel = new_record;
	queue_record new_qrecord = {RECORD_CULAUNCHKERNEL, new_record_data};

    stream_queue_ptr queue = find_stream(idx, hStream);
    pthread_mutex_t* lock = work_queue_mutex[queue->owner];
    pthread_mutex_lock(lock);
    if (queue->closing) {
        pthread_mutex_unlock(lock);
        return CUDA_ERROR_INVALID_HANDLE;
    }
    for (int i = 0; i < atom_num; ++i) {
        queue->records.push(new_qrecord);
        ++queue->issued;
        new_qrecord.data.r_cuLaunchKernel.lidx += atom_size;
        new_qrecord.data.r_cuLaunchKernel.hidx += atom_size;
    }
    uint64_t ticket = queue->issued;
    pthread_mutex_unlock(lock);

    if (cudaLaunchKernel_launched[idx]) runtime_submissions.push_back({queue, ticket});
    else wait_for_submission(queue, ticket);

    return err;

}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
	fprintf(stderr, "Warning: (non-implemented) cuLaunchKernelEx is captured!\n");
	if (real_cuLaunchKernelEx == NULL) {
        if(cu_handle == NULL) cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        real_cuLaunchKernelEx = (CUresult (*)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra))real_dlsym(cu_handle, "cuLaunchKernelEx");
        if(real_cuLaunchKernelEx == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernelEx == NULL\n");
        }
        if(real_cuLaunchKernelEx == cuLaunchKernelEx) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernelEx == cuLaunchKernelEx\n");
        }
    }
	return real_cuLaunchKernelEx(config, f, kernelParams, extra);

	
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    if (memory_client_idx() >= 0)
        return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                              sharedMemBytes, hStream, kernelParams, extra);
	fprintf(stderr, "Warning: (non-implemented) cuLaunchKernel_ptsz is captured!\n");
	if (real_cuLaunchKernel_ptsz == NULL) {
        if(cu_handle == NULL) cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        real_cuLaunchKernel_ptsz = (CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**))real_dlsym(cu_handle, "cuLaunchKernel_ptsz");
        if(real_cuLaunchKernel_ptsz == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernel_ptsz == NULL\n");
        }
        if(real_cuLaunchKernel_ptsz == cuLaunchKernel_ptsz) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernel_ptsz == cuLaunchKernel_ptsz\n");
        }
    }
	return real_cuLaunchKernel_ptsz(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

	
}

CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
	fprintf(stderr, "Warning: (non-implemented) cuLaunchKernelEx_ptsz is captured!\n");
	if (real_cuLaunchKernelEx_ptsz == NULL) {
        if(cu_handle == NULL) cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        real_cuLaunchKernelEx_ptsz = (CUresult (*)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra))real_dlsym(cu_handle, "cuLaunchKernelEx_ptsz");
        if(real_cuLaunchKernelEx_ptsz == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernelEx_ptsz == NULL\n");
        }
        if(real_cuLaunchKernelEx_ptsz == cuLaunchKernelEx_ptsz) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernelEx_ptsz == cuLaunchKernelEx_ptsz\n");
        }
    }
	return real_cuLaunchKernelEx_ptsz(config, f, kernelParams, extra);

}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
	fprintf(stderr, "Warning: (non-implemented) cuGraphLaunch is captured!\n");
	if (real_cuGraphLaunch == NULL) {
        if(cu_handle == NULL) cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        real_cuGraphLaunch = (CUresult (*)(CUgraphExec, CUstream))real_dlsym(cu_handle, "cuGraphLaunch");
        if(real_cuGraphLaunch == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cuGraphLaunch == NULL\n");
        }
        if(real_cuGraphLaunch == cuGraphLaunch) {
            fprintf(stderr, "FATAL ERROR: real_cuGraphLaunch == cuGraphLaunch\n");
        }
    }
	return real_cuGraphLaunch(hGraphExec, hStream);
	
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                        		unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                        		unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) {
	fprintf(stderr, "Warning: (non-implemented) cuLaunchCooperativeKernel is captured!\n");
	if (real_cuLaunchCooperativeKernel == NULL) {
        if(cu_handle == NULL) cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        real_cuLaunchCooperativeKernel = (CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**))real_dlsym(cu_handle, "cuLaunchCooperativeKernel");
        if(real_cuLaunchCooperativeKernel == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchCooperativeKernel == NULL\n");
        }
        if(real_cuLaunchCooperativeKernel == cuLaunchCooperativeKernel) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchCooperativeKernel == cuLaunchCooperativeKernel\n");
        }
    }
	return real_cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}


/*
	Hook for cudaMemset, cudaMemsetAsync, ...
	cudaMemcpy DtoD, ...
	which internally calls a kernel silently.
	They *must* be launched with the scheduler, otherwise it will break libsmctrl logic!
	
*/
cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
	if (real_cudaMemset == NULL) {
        real_cudaMemset = (cudaError_t (*)(void*, int, size_t))real_dlsym(RTLD_NEXT, "cudaMemset");
        if(real_cudaMemset == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cudaMemset == NULL\n");
        }
    }
	int idx = memory_client_idx();
	if (idx < 0) return real_cudaMemset(devPtr, value, count);
	fprintf(stderr, "[cudaMemsetHook] hooked from %d!\n", idx);

	assert(work_queue_mutex != NULL);
	assert(work_queue != NULL);

	record_cudaMemset new_record;
	cudaError_t status = cudaSuccess;
	new_record = {devPtr, value, count, &status};
	union record_data new_record_data;
	new_record_data.r_cudaMemset = new_record;
	queue_record new_qrecord = {RECORD_CUDAMEMSET, new_record_data};

    stream_queue_ptr queue = find_stream(idx, NULL);
    if (!queue) return cudaErrorInvalidResourceHandle;
    uint64_t ticket = submit(queue, new_qrecord);
    if (!ticket) return cudaErrorInvalidResourceHandle;
    wait_for_submission(queue, ticket);
    if (status != cudaSuccess) return status;
    driver_request checkpoint;
    return (cudaError_t)submit_request(queue, RECORD_STREAM_SYNC, checkpoint);

}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
	if (real_cudaMemsetAsync == NULL) {
        real_cudaMemsetAsync = (cudaError_t (*)(void*, int, size_t, cudaStream_t))real_dlsym(RTLD_NEXT, "cudaMemsetAsync");
        if(real_cudaMemsetAsync == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cudaMemsetAsync == NULL\n");
        }
    }
	int idx = memory_client_idx();
	if (idx < 0) return real_cudaMemsetAsync(devPtr, value, count, stream);
	fprintf(stderr, "[cudaMemsetAsyncHook] hooked from %d!\n", idx);

	assert(work_queue_mutex != NULL);
	assert(work_queue != NULL);

	record_cudaMemsetAsync new_record;
	cudaError_t status = cudaSuccess;
	new_record = {devPtr, value, count, stream, &status};
	union record_data new_record_data;
	new_record_data.r_cudaMemsetAsync = new_record;
	queue_record new_qrecord = {RECORD_CUDAMEMSETASYNC, new_record_data};

    stream_queue_ptr queue = find_stream(idx, stream);
    if (!queue) return cudaErrorInvalidResourceHandle;
    uint64_t ticket = submit(queue, new_qrecord);
    if (!ticket) return cudaErrorInvalidResourceHandle;
    wait_for_submission(queue, ticket);
	return status;
}

/*
	cuGetProcAddress is ACTUALLY where we redirect driver APIs.
	for convenience(and maybe existing just name mangling way of linking), we set the exact same name for driver API hooks,
	but this is where the hook actaully proceed.
*/

struct intercepted_symbol {
    const char* base;
    const char* exported;
    void* hook;
};

static const intercepted_symbol stream_event_symbols[] = {
    {"cuStreamCreate", "cuStreamCreate", (void*)cuStreamCreate},
    {"cuStreamCreateWithPriority", "cuStreamCreateWithPriority", (void*)cuStreamCreateWithPriority},
    {"cuStreamDestroy", "cuStreamDestroy_v2", (void*)cuStreamDestroy_v2},
    {"cuStreamSynchronize", "cuStreamSynchronize", (void*)cuStreamSynchronize},
    {"cuEventCreate", "cuEventCreate", (void*)cuEventCreate},
    {"cuEventDestroy", "cuEventDestroy_v2", (void*)cuEventDestroy_v2},
    {"cuEventQuery", "cuEventQuery", (void*)cuEventQuery},
    {"cuEventRecord", "cuEventRecord", (void*)cuEventRecord},
    {"cuEventSynchronize", "cuEventSynchronize", (void*)cuEventSynchronize},
    {"cuStreamDestroy", "cuStreamDestroy", (void*)legacy_cuStreamDestroy},
    {"cuStreamSynchronize", "cuStreamSynchronize_ptsz", (void*)cuStreamSynchronize_ptsz},
    {"cuEventDestroy", "cuEventDestroy", (void*)legacy_cuEventDestroy},
    {"cuEventRecord", "cuEventRecord_ptsz", (void*)cuEventRecord_ptsz},
};

static void* stream_event_address(const char* name, void* address) {
    if (!address || !name) return address;
    for (const intercepted_symbol& entry : stream_event_symbols) {
        if ((strcmp(name, entry.base) == 0 || strcmp(name, entry.exported) == 0) &&
            address == memory_driver_symbol(entry.exported)) return entry.hook;
    }
    return address;
}

CUresult my_cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) {

    if (symbol && (strcmp(symbol, "cuMemAlloc") == 0 || strcmp(symbol, "cuMemAlloc_v2") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t))
                memory_driver_symbol("cuGetProcAddress");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemAlloc_v2"))
            *pfn = (void*)cuMemAlloc;
        return status;
    }

    if (symbol && (strcmp(symbol, "cuMemAllocHost") == 0 || strcmp(symbol, "cuMemAllocHost_v2") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t))
                memory_driver_symbol("cuGetProcAddress");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemAllocHost_v2"))
            *pfn = (void*)cuMemAllocHost;
        return status;
    }

    if (symbol && (strcmp(symbol, "cuMemHostAlloc") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t))
                memory_driver_symbol("cuGetProcAddress");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemHostAlloc"))
            *pfn = (void*)cuMemHostAlloc;
        return status;
    }

    if (symbol && (strcmp(symbol, "cuMemFree") == 0 || strcmp(symbol, "cuMemFree_v2") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t))
                memory_driver_symbol("cuGetProcAddress");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemFree_v2"))
            *pfn = (void*)cuMemFree;
        return status;
    }

    if (symbol && (strcmp(symbol, "cuMemFreeHost") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t))
                memory_driver_symbol("cuGetProcAddress");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemFreeHost"))
            *pfn = (void*)cuMemFreeHost;
        return status;
    }
    // fprintf(stderr, "[HOOK v1] Inside cuGetProcAddress looking for: %s\n", symbol);

    if (symbol && strcmp(symbol, "cuLaunchKernel") == 0) {
        fprintf(stderr, "[HOOK v1] Hijacking cuLaunchKernel pointer assignment!\n");
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    }

	if (symbol && strcmp(symbol, "cuLaunchKernelEx") == 0) {
        fprintf(stderr, "[HOOK v1] Hijacking cuLaunchKernelEx pointer assignment!\n");
        *pfn = (void*)cuLaunchKernelEx;
        return CUDA_SUCCESS;
    }

	if (symbol && strcmp(symbol, "cuGraphLaunch") == 0) {
        fprintf(stderr, "[HOOK v1] Hijacking cuGraphLaunch pointer assignment!\n");
        *pfn = (void*)cuGraphLaunch;
        return CUDA_SUCCESS;
    }

	if (symbol && strcmp(symbol, "cuLaunchKernel_ptsz") == 0) {
        fprintf(stderr, "[HOOK v1] Hijacking cuLaunchKernel_ptsz pointer assignment!\n");
        *pfn = (void*)cuLaunchKernel_ptsz;
        return CUDA_SUCCESS;
    }

	if (symbol && strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0) {
        fprintf(stderr, "[HOOK v1] Hijacking cuLaunchKernelEx_ptsz pointer assignment!\n");
        *pfn = (void*)cuLaunchKernelEx_ptsz;
        return CUDA_SUCCESS;
    }

	if (symbol && strcmp(symbol, "cuLaunchCooperativeKernel") == 0) {
        fprintf(stderr, "[HOOK v1] Hijacking cuLaunchCooperativeKernel pointer assignment!\n");
        *pfn = (void*)cuLaunchCooperativeKernel;
        return CUDA_SUCCESS;
    }

    static cuGetProcAddress_t resolve =
        (cuGetProcAddress_t)memory_driver_symbol("cuGetProcAddress");
    if (!resolve) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult status = resolve(symbol, pfn, cudaVersion, flags);
    if (status == CUDA_SUCCESS && pfn) *pfn = stream_event_address(symbol, *pfn);
    return status;
}

CUresult my_cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {

    if (symbol && (strcmp(symbol, "cuMemAlloc") == 0 || strcmp(symbol, "cuMemAlloc_v2") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*))
                memory_driver_symbol("cuGetProcAddress_v2");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags, (CUdriverProcAddressQueryResult*)symbolStatus);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemAlloc_v2"))
            *pfn = (void*)cuMemAlloc;
        return status;
    }

    if (symbol && (strcmp(symbol, "cuMemAllocHost") == 0 || strcmp(symbol, "cuMemAllocHost_v2") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*))
                memory_driver_symbol("cuGetProcAddress_v2");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags, (CUdriverProcAddressQueryResult*)symbolStatus);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemAllocHost_v2"))
            *pfn = (void*)cuMemAllocHost;
        return status;
    }

    if (symbol && (strcmp(symbol, "cuMemHostAlloc") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*))
                memory_driver_symbol("cuGetProcAddress_v2");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags, (CUdriverProcAddressQueryResult*)symbolStatus);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemHostAlloc"))
            *pfn = (void*)cuMemHostAlloc;
        return status;
    }

    if (symbol && (strcmp(symbol, "cuMemFree") == 0 || strcmp(symbol, "cuMemFree_v2") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*))
                memory_driver_symbol("cuGetProcAddress_v2");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags, (CUdriverProcAddressQueryResult*)symbolStatus);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemFree_v2"))
            *pfn = (void*)cuMemFree;
        return status;
    }

    if (symbol && (strcmp(symbol, "cuMemFreeHost") == 0)) {
        // Let the driver validate the lookup and select its ABI first.
        static CUresult (CUDAAPI *resolve)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) =
            (CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*))
                memory_driver_symbol("cuGetProcAddress_v2");
        if (resolve == NULL) return CUDA_ERROR_NOT_INITIALIZED;
        CUresult status = resolve(symbol, pfn, cudaVersion, flags, (CUdriverProcAddressQueryResult*)symbolStatus);
        if (status == CUDA_SUCCESS && pfn && *pfn &&
            *pfn == memory_driver_symbol("cuMemFreeHost"))
            *pfn = (void*)cuMemFreeHost;
        return status;
    }
    // fprintf(stderr, "[HOOK v2] Inside cuGetProcAddress_v2 looking for: %s\n", symbol);

    if (symbol && strcmp(symbol, "cuLaunchKernel") == 0) {
        fprintf(stderr, "[HOOK v2] Hijacking cuLaunchKernel pointer assignment!\n");
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    }

    static cuGetProcAddress_v2_t resolve =
        (cuGetProcAddress_v2_t)memory_driver_symbol("cuGetProcAddress_v2");
    if (!resolve) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult status = resolve(symbol, pfn, cudaVersion, flags, symbolStatus);
    if (status == CUDA_SUCCESS && pfn && *pfn && symbol && strcmp(symbol, "cuGetProcAddress") == 0) {
        if (*pfn == memory_driver_symbol("cuGetProcAddress")) *pfn = (void*)my_cuGetProcAddress;
        else if (*pfn == memory_driver_symbol("cuGetProcAddress_v2")) *pfn = (void*)my_cuGetProcAddress_v2;
        return status;
    }
    if (status == CUDA_SUCCESS && pfn) *pfn = stream_event_address(symbol, *pfn);
    return status;
}

// The same wrappers serve ELF-linked callers and explicit driver lookups.
CUresult CUDAAPI linked_get_proc(const char*, void**, int, cuuint64_t) asm("cuGetProcAddress");
CUresult CUDAAPI linked_get_proc(const char* name, void** address, int version, cuuint64_t flags) {
    return my_cuGetProcAddress(name, address, version, flags);
}
CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** address, int version, cuuint64_t flags,
                                    CUdriverProcAddressQueryResult* query) {
    return my_cuGetProcAddress_v2(name, address, version, flags, query);
}

void* dlsym(void* handle, const char* symbol) {
    // fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s.\n", symbol);
    if (symbol && strcmp(symbol, "cuMemAlloc_v2") == 0) {
        void* resolved = real_dlsym(handle, symbol);
        if (resolved && resolved == memory_driver_symbol("cuMemAlloc_v2"))
            return (void*)cuMemAlloc;
        return resolved;
    }

    if (symbol && strcmp(symbol, "cuMemAllocHost_v2") == 0) {
        void* resolved = real_dlsym(handle, symbol);
        if (resolved && resolved == memory_driver_symbol("cuMemAllocHost_v2"))
            return (void*)cuMemAllocHost;
        return resolved;
    }

    if (symbol && strcmp(symbol, "cuMemHostAlloc") == 0) {
        void* resolved = real_dlsym(handle, symbol);
        if (resolved && resolved == memory_driver_symbol("cuMemHostAlloc"))
            return (void*)cuMemHostAlloc;
        return resolved;
    }

    if (symbol && strcmp(symbol, "cuMemFree_v2") == 0) {
        void* resolved = real_dlsym(handle, symbol);
        if (resolved && resolved == memory_driver_symbol("cuMemFree_v2"))
            return (void*)cuMemFree;
        return resolved;
    }

    if (symbol && strcmp(symbol, "cuMemFreeHost") == 0) {
        void* resolved = real_dlsym(handle, symbol);
        if (resolved && resolved == memory_driver_symbol("cuMemFreeHost"))
            return (void*)cuMemFreeHost;
        return resolved;
    }

    // CRUCIAL: Intercept libcudart trying to look up cuGetProcAddress
    if (symbol && (strcmp(symbol, "cuGetProcAddress") == 0)) {
        fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s! Returning our hook.\n", symbol);
        
        
        return (void*)my_cuGetProcAddress;
    }

    if (symbol && (strcmp(symbol, "cuGetProcAddress_v2") == 0)) {
        fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s! Returning our hook.\n", symbol);
        
        
        return (void*)my_cuGetProcAddress_v2;
    }

	/* 
		cuLaunchKernel-style dlsym() is needed because cublas finds cuLaunchKernel() using dlsym().
	*/
	if (symbol && (strcmp(symbol, "cuLaunchKernel") == 0)) {
        fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s! Returning our hook.\n", symbol);
        return (void*)cuLaunchKernel;
    }
	if (symbol && (strcmp(symbol, "cudaLaunchKernel") == 0)) {
        fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s! Returning our hook.\n", symbol);
		PFN_cudaLaunchKernel target_func = &::cudaLaunchKernel; 
        return (void*)target_func;
    }

    return stream_event_address(symbol, real_dlsym(handle, symbol));
}

}
