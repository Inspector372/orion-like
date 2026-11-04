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
	Kernel Queue. This queues cuLaunchKernel, and its derivatives.
	cudaMemset/Async() can be a kernel, so it goes to kernel queue.
	For now, memory operations also go here.
*/
queue<queue_record>** work_queue;
pthread_mutex_t** work_queue_mutex;

/*
	TODO: create Memory Queue, which contains HtoD/DtoH memory operations.
*/

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
cuGetProcAddress_t real_cuGetProcAddress = NULL;
cuGetProcAddress_v2_t real_cuGetProcAddress_v2 = NULL;

/*
	Real instance of dlsym().
*/
void* (*real_dlsym)(void*, const char*) = NULL;
void* cu_handle = NULL;


/*
	variable for identifying if current launch is from cudaLaunchKernel(), or separate mechanism
*/
bool cudaLaunchKernel_launched[THREAD_NUM];


int get_idx() {
	assert(thread_ids != NULL);
	pthread_t tid = pthread_self();
	// fprintf(stderr, "tid = %ld\n", tid);
	// pid_t tid = syscall(SYS_gettid);
	int idx = -1;
	for (int i = 0; i < THREAD_NUM; i++) {
		if (pthread_equal(tid, thread_ids[i])) {
			idx = i;
			break;
		}
	}
	assert(idx != -1);
	// fprintf(stderr, "idx = %d\n", idx);
	return idx;
}

void block(int idx, pthread_mutex_t** mutexes, queue<queue_record>** kqueues) {
	while (1) {
		pthread_mutex_lock(mutexes[idx]);
		volatile int sz = kqueues[idx]->size();
		pthread_mutex_unlock(mutexes[idx]);
		if (sz==0)
			break;
	}
}

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

// Use the real loader, never our dlsym hook, for driver entry points.
// Function-local statics provide thread-safe initialization for these new hooks.
static void* memory_driver_symbol(const char* symbol) {
    static auto loader = reinterpret_cast<void* (*)(void*, const char*)>(
        dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
    static void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    return loader && driver ? loader(driver, symbol) : nullptr;
}

static int memory_client_idx() {
    // Startup, scheduler and other unregistered threads must not self-enqueue.
    // Unregistered framework workers are outside this harness's queue coverage.
    if (no_hook || !thread_ids || !work_queue || !work_queue_mutex) return -1;
    for (int i = 0; i < THREAD_NUM; ++i)
        if (pthread_equal(pthread_self(), thread_ids[i])) return i;
    return -1;
}

static CUresult queue_driver_memory(int idx, record_type type,
                                    record_driver_memory record) {
    static auto get_context = reinterpret_cast<decltype(&cuCtxGetCurrent)>(
        memory_driver_symbol("cuCtxGetCurrent"));
    if (!get_context) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult status = get_context(&record.context);
    if (status != CUDA_SUCCESS) return status;

    driver_memory_completion completion;
    record.completion = &completion;
    record_data data;
    data.r_driver_memory = record;
    queue_record request = {type, data};
    pthread_mutex_lock(work_queue_mutex[idx]);
    try {
        work_queue[idx]->push(request);
    } catch (...) {
        pthread_mutex_unlock(work_queue_mutex[idx]);
        pthread_cond_destroy(&completion.condition);
        pthread_mutex_destroy(&completion.mutex);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    pthread_mutex_unlock(work_queue_mutex[idx]);

    pthread_mutex_lock(&completion.mutex);
    while (!completion.done)
        pthread_cond_wait(&completion.condition, &completion.mutex);
    status = completion.status;
    pthread_mutex_unlock(&completion.mutex);
    pthread_cond_destroy(&completion.condition);
    pthread_mutex_destroy(&completion.mutex);
    return status;
}

extern "C" {

CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    static auto real = reinterpret_cast<decltype(&cuMemAlloc)>(
        memory_driver_symbol("cuMemAlloc_v2"));
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real(dptr, bytesize);
    record_driver_memory record{};
    record.device_out = dptr; record.bytesize = bytesize;
    return queue_driver_memory(idx, RECORD_CUMEMALLOC, record);
}


CUresult CUDAAPI cuMemAllocHost(void** pp, size_t bytesize) {
    static auto real = reinterpret_cast<decltype(&cuMemAllocHost)>(
        memory_driver_symbol("cuMemAllocHost_v2"));
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real(pp, bytesize);
    record_driver_memory record{};
    record.host_out = pp; record.bytesize = bytesize;
    return queue_driver_memory(idx, RECORD_CUMEMALLOCHOST, record);
}


CUresult CUDAAPI cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
    static auto real = reinterpret_cast<decltype(&cuMemHostAlloc)>(
        memory_driver_symbol("cuMemHostAlloc"));
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real(pp, bytesize, flags);
    record_driver_memory record{};
    record.host_out = pp; record.bytesize = bytesize; record.flags = flags;
    return queue_driver_memory(idx, RECORD_CUMEMHOSTALLOC, record);
}


CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    static auto real = reinterpret_cast<decltype(&cuMemFree)>(
        memory_driver_symbol("cuMemFree_v2"));
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real(dptr);
    record_driver_memory record{};
    record.device_ptr = dptr;
    return queue_driver_memory(idx, RECORD_CUMEMFREE, record);
}


CUresult CUDAAPI cuMemFreeHost(void* p) {
    static auto real = reinterpret_cast<decltype(&cuMemFreeHost)>(
        memory_driver_symbol("cuMemFreeHost"));
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    int idx = memory_client_idx();
    if (idx < 0) return real(p);
    record_driver_memory record{};
    record.host_ptr = p;
    return queue_driver_memory(idx, RECORD_CUMEMFREEHOST, record);
}

/*
	*exact* behavior of cudaDeviceSynchronize() need to be inspected.

	For now, it creates a cudaevent, attaches in work_queue.
	and wait until it gets scheduled and eventually finished.
*/
cudaError_t cudaDeviceSynchronize(void) {
	
	cudaEvent_t event;
	cudaEventCreate(&event);
	int idx = get_idx();

	record_cudaEvent new_record;
	new_record = {event};
	union record_data new_record_data;
	new_record_data.r_cudaEvent = new_record;
	queue_record new_qrecord = {RECORD_CUDAEVENT, new_record_data};

	pthread_mutex_lock(work_queue_mutex[idx]);
	work_queue[idx]->push(new_qrecord);
	pthread_mutex_unlock(work_queue_mutex[idx]);
	
	block(idx, work_queue_mutex, work_queue);
	cudaEventSynchronize(event);

	cudaEventDestroy(event);
	return cudaSuccess;
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
	if(no_hook) {
		return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
	}
	int idx = get_idx();
	fprintf(stderr, "[cudaHook] caught call from %d!\n", idx);
	cudaLaunchKernel_launched[idx] = true;
	cudaError_t err = real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
	cudaLaunchKernel_launched[idx] = false;
	fprintf(stderr, "[cudaHook] block-start from %d!\n", idx);
	block(idx, work_queue_mutex, work_queue);
	fprintf(stderr, "[cudaHook] block-end from %d!\n", idx);
	
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

	if(no_hook) {
		// immediately run the kernel.
		fprintf(stderr, "[cuHook] no-hook launch of %p\n", f);
		return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
	}

	int idx = get_idx();
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

	new_record = {f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra, 0, atom_size};
	union record_data new_record_data;
	new_record_data.r_cuLaunchKernel = new_record;
	queue_record new_qrecord = {RECORD_CULAUNCHKERNEL, new_record_data};

	pthread_mutex_lock(work_queue_mutex[idx]);

	// Atomization.
	for(int i=0; i<atom_num; i++) {
		work_queue[idx]->push(new_qrecord);
		new_qrecord.data.r_cuLaunchKernel.lidx += atom_size;
		new_qrecord.data.r_cuLaunchKernel.hidx += atom_size;
	}
	

	pthread_mutex_unlock(work_queue_mutex[idx]);
	
	if(!cudaLaunchKernel_launched[idx]) {
		fprintf(stderr, "[cuHook] block-start from %d!\n", idx);
		block(idx, work_queue_mutex, work_queue);
		fprintf(stderr, "[cuHook] block-end from %d!\n", idx);
	}

    return err;

}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
	fprintf(stderr, "cuLaunchKernelEx is captured!\n");
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
	fprintf(stderr, "cuLaunchKernel_ptsz is captured!\n");
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
	fprintf(stderr, "cuLaunchKernelEx_ptsz is captured!\n");
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
	fprintf(stderr, "cuGraphLaunch is captured!\n");
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
	fprintf(stderr, "cuLaunchCooperativeKernel is captured!\n");
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
	int idx = get_idx();
	fprintf(stderr, "[cudaMemsetHook] hooked from %d!\n", idx);

	assert(work_queue_mutex != NULL);
	assert(work_queue != NULL);

	record_cudaMemset new_record;
	new_record = {devPtr, value, count};
	union record_data new_record_data;
	new_record_data.r_cudaMemset = new_record;
	queue_record new_qrecord = {RECORD_CUDAMEMSET, new_record_data};

	pthread_mutex_lock(work_queue_mutex[idx]);
	work_queue[idx]->push(new_qrecord);
	pthread_mutex_unlock(work_queue_mutex[idx]);
	
	fprintf(stderr, "[cudaMemsetHook] blocking from %d!\n", idx);
	block(idx, work_queue_mutex, work_queue);
	fprintf(stderr, "[cudaMemsetHook] unblocking from %d!\n", idx);

	return cudaSuccess;

}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
	if (real_cudaMemsetAsync == NULL) {
        real_cudaMemsetAsync = (cudaError_t (*)(void*, int, size_t, cudaStream_t))real_dlsym(RTLD_NEXT, "cudaMemsetAsync");
        if(real_cudaMemsetAsync == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cudaMemsetAsync == NULL\n");
        }
    }
	int idx = get_idx();
	fprintf(stderr, "[cudaMemsetAsyncHook] hooked from %d!\n", idx);

	assert(work_queue_mutex != NULL);
	assert(work_queue != NULL);

	record_cudaMemsetAsync new_record;
	new_record = {devPtr, value, count, stream};
	union record_data new_record_data;
	new_record_data.r_cudaMemsetAsync = new_record;
	queue_record new_qrecord = {RECORD_CUDAMEMSETASYNC, new_record_data};

	pthread_mutex_lock(work_queue_mutex[idx]);
	work_queue[idx]->push(new_qrecord);
	pthread_mutex_unlock(work_queue_mutex[idx]);
	
	// Async is not blocking 'at user level', so no block here.
	// but it's certainly blocking 'at OS level'.

	return cudaSuccess;
}

/*
	cuGetProcAddress is ACTUALLY where we redirect driver APIs.
	for convenience(and maybe existing just name mangling way of linking), we set the exact same name for driver API hooks,
	but this is where the hook actaully proceed.
*/

// Only redirect the ABI that our CUDA 12.8 signatures implement. In particular,
// legacy cuMemAlloc/cuMemFree exports use different pointer/size widths.
static void* memory_hook_address(const char* symbol, void* resolved) {
    if (!symbol || !resolved) return nullptr;
    if ((strcmp(symbol, "cuMemAlloc") == 0 || strcmp(symbol, "cuMemAlloc_v2") == 0) &&
        resolved == memory_driver_symbol("cuMemAlloc_v2"))
        return reinterpret_cast<void*>(&cuMemAlloc);
    if ((strcmp(symbol, "cuMemAllocHost") == 0 || strcmp(symbol, "cuMemAllocHost_v2") == 0) &&
        resolved == memory_driver_symbol("cuMemAllocHost_v2"))
        return reinterpret_cast<void*>(&cuMemAllocHost);
    if ((strcmp(symbol, "cuMemHostAlloc") == 0 || strcmp(symbol, "cuMemHostAlloc") == 0) &&
        resolved == memory_driver_symbol("cuMemHostAlloc"))
        return reinterpret_cast<void*>(&cuMemHostAlloc);
    if ((strcmp(symbol, "cuMemFree") == 0 || strcmp(symbol, "cuMemFree_v2") == 0) &&
        resolved == memory_driver_symbol("cuMemFree_v2"))
        return reinterpret_cast<void*>(&cuMemFree);
    if ((strcmp(symbol, "cuMemFreeHost") == 0 || strcmp(symbol, "cuMemFreeHost") == 0) &&
        resolved == memory_driver_symbol("cuMemFreeHost"))
        return reinterpret_cast<void*>(&cuMemFreeHost);
    return nullptr;
}

static void redirect_driver_address(const char* symbol, void** pfn) {
    if (!symbol || !pfn || !*pfn) return;
    if (void* hook = memory_hook_address(symbol, *pfn)) {
        *pfn = hook;
        return;
    }
    // Preserve the existing launch interception, with the correct ptsz targets.
    if (strcmp(symbol, "cuLaunchKernel") == 0)
        *pfn = reinterpret_cast<void*>(&cuLaunchKernel);
    else if (strcmp(symbol, "cuLaunchKernelEx") == 0)
        *pfn = reinterpret_cast<void*>(&cuLaunchKernelEx);
    else if (strcmp(symbol, "cuGraphLaunch") == 0)
        *pfn = reinterpret_cast<void*>(&cuGraphLaunch);
    else if (strcmp(symbol, "cuLaunchKernel_ptsz") == 0)
        *pfn = reinterpret_cast<void*>(&cuLaunchKernel_ptsz);
    else if (strcmp(symbol, "cuLaunchKernelEx_ptsz") == 0)
        *pfn = reinterpret_cast<void*>(&cuLaunchKernelEx_ptsz);
    else if (strcmp(symbol, "cuLaunchCooperativeKernel") == 0)
        *pfn = reinterpret_cast<void*>(&cuLaunchCooperativeKernel);
}

CUresult CUDAAPI my_cuGetProcAddress(const char* symbol, void** pfn,
                                    int cudaVersion, cuuint64_t flags) {
    static auto real = reinterpret_cast<cuGetProcAddress_t>(
        memory_driver_symbol("cuGetProcAddress"));
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult status = real(symbol, pfn, cudaVersion, flags);
    if (status == CUDA_SUCCESS) redirect_driver_address(symbol, pfn);
    return status;
}

CUresult CUDAAPI my_cuGetProcAddress_v2(const char* symbol, void** pfn,
        int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
    static auto real = reinterpret_cast<cuGetProcAddress_v2_t>(
        memory_driver_symbol("cuGetProcAddress_v2"));
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult status = real(symbol, pfn, cudaVersion, flags, symbolStatus);
    if (status != CUDA_SUCCESS || !pfn || !*pfn) return status;
    // cuGetProcAddress itself has two ABIs; honor the one the driver selected.
    if (symbol && strcmp(symbol, "cuGetProcAddress") == 0) {
        if (*pfn == memory_driver_symbol("cuGetProcAddress"))
            *pfn = reinterpret_cast<void*>(&my_cuGetProcAddress);
        else if (*pfn == memory_driver_symbol("cuGetProcAddress_v2"))
            *pfn = reinterpret_cast<void*>(&my_cuGetProcAddress_v2);
    } else {
        redirect_driver_address(symbol, pfn);
    }
    return status;
}

void* dlsym(void* handle, const char* symbol) {
    // fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s.\n", symbol);
    // Bootstrap the real dlsym using dlvsym to avoid recursion
    if (real_dlsym == NULL) {
        real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (real_dlsym == NULL) {
            real_dlsym = (void* (*)(void*, const char*))dlopen("libdl.so.2", RTLD_NOW); 
        }
    }

    // CRUCIAL: Intercept libcudart trying to look up cuGetProcAddress
    if (symbol && (strcmp(symbol, "cuGetProcAddress") == 0)) {
        fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s! Returning our hook.\n", symbol);
        
        // Save the real function pointer from the requested handle before we fake the return
        real_cuGetProcAddress = (cuGetProcAddress_t)real_dlsym(handle, symbol);
        
        return (void*)my_cuGetProcAddress;
    }

    if (symbol && (strcmp(symbol, "cuGetProcAddress_v2") == 0)) {
        fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s! Returning our hook.\n", symbol);
        
        // Save the real function pointer from the requested handle before we fake the return
        real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_t)real_dlsym(handle, symbol);
        
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

    void* resolved = real_dlsym(handle, symbol);
    if (void* hook = memory_hook_address(symbol, resolved)) return hook;
    return resolved;
}

}
