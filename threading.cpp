/*
	threading.cpp

	Spawns scheduler(OS) and client(tanent) threads.
	main branch of the code.
*/

#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>
#include <iostream>
#include <queue>
#include <algorithm>
#include <cstring>
#include <time.h>
#include <assert.h>
#include <atomic>
#include <cstdlib>
#include <exception>
#include "testcase/testcase.h"

#include <cuda_runtime.h>
#include <cuda.h>

#include "hooking.h"
#include "wrapper.h"
#include "libsmctrl.h"

using namespace std;

/*
	Variables used for checking testcases.
*/
struct arg_t {
    testcase::Selection selection;
    testcase::Result result;
};
std::atomic<int> clients_done{0};

/*
	Actual instances for cuda APIs. we use "actual_" in front of threading.cpp,
	while "real_" is used in hooking.cpp.
*/
cudaError_t (*actual_cudaDeviceSynchronize)(void) = nullptr;
cudaError_t (*actual_cudaMemsetAsync)(void*, int, size_t, cudaStream_t) = nullptr;

CUresult (*actual_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
CUresult (*actual_cuFuncGetParamInfo)(CUfunction, size_t, size_t*, size_t*);

/*
	hooking.cpp.
*/
void* klib;


/*
	Kernel Queue.
*/
client_streams* work_queue;
pthread_mutex_t** work_queue_mutex;

/*	
	Clients wait until registration and scheduler setup are complete.
*/
pthread_mutex_t start_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t start_cond = PTHREAD_COND_INITIALIZER;
bool ready = false;


/*
	No hook switch. no_hook at hooking.cpp.
*/
bool* no_hook_thr;


/*
	Stream for metadata passing / launch blocking APIs.
*/
cudaStream_t metadata_pass_stream;
cudaStream_t launch_block_stream;

/*
	Main's initialized CUDA context, shared by scheduler streams and metadata.
*/
typedef struct scheduler_arg {
	CUcontext context;
} scheduler_arg;


void hash_insert(uint64_t key, AtomMetaData value) {
	if(key == 0) {
		fprintf(stderr, "hash_insert: trying to insert a null key. returning.\n");
	}
	/* Problem: This can be INSIDE work_queue_mutex critical section, which will cause deadlock */
	table_insert(key, value);
}

void hash_delete(uint64_t key) {
	if(key == 0) {
		fprintf(stderr, "hash_delete: trying to delete a null key. returning.\n");
	}
	table_delete(key);
}

// The harness owns startup and completion; workloads contain ordinary CUDA code.
void* thread_wrapper(void* arg) {
    auto* args = static_cast<arg_t*>(arg);
    pthread_mutex_lock(&start_mutex);
    while (!ready) pthread_cond_wait(&start_cond, &start_mutex);
    pthread_mutex_unlock(&start_mutex);
    try {
        args->result = args->selection.entry->run(args->selection.config);
    } catch (const std::exception& e) {
        args->result = {false, e.what()};
    } catch (...) {
        args->result = {false, "Unknown workload exception"};
    }
    ++clients_done;
    return nullptr;
}

/*
	imported from Orion.
	When handling libcuda(not libcudart), RTLD_DEFAULT didnt work,
	so we open it directly and load it into handle.
*/
void register_functions() {
	void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);

    // for kernel
	void* (*actual_dlsym)(void*, const char*) = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");;
	*(void **)(&actual_cuLaunchKernel) = actual_dlsym(handle, "cuLaunchKernel");
	assert(actual_cuLaunchKernel != NULL);

	*(void **)(&actual_cuFuncGetParamInfo) = actual_dlsym(handle, "cuFuncGetParamInfo");
	assert(actual_cuFuncGetParamInfo != NULL);

	// for wrapper, initial wrapper run.
	*(void**)(&actual_cudaDeviceSynchronize) = dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    assert(actual_cudaDeviceSynchronize != nullptr);

	*(void**)(&actual_cudaMemsetAsync) = dlsym(RTLD_NEXT, "cudaMemsetAsync");
    assert(actual_cudaMemsetAsync != nullptr);

    // assign hash_insert_callback of libsmctrl.
	assign_hash_insert((void*)hash_insert);

}

void variables_setup() {
	klib = dlopen("./hooking.so", RTLD_NOW | RTLD_GLOBAL);

	// 1. queue for each thread.
    client_streams** work_queue_ptr = (client_streams**)dlsym(klib, "work_queue");
    work_queue = new client_streams[THREAD_NUM];
    *work_queue_ptr = work_queue;

	// 2. mutexes for queues.
	pthread_mutex_t*** mutex_ptr = (pthread_mutex_t***)dlsym(klib, "work_queue_mutex");
	*mutex_ptr = (pthread_mutex_t**)malloc(THREAD_NUM * sizeof(pthread_mutex_t*));
	work_queue_mutex = *mutex_ptr;
	for (int i = 0; i < THREAD_NUM; i++) {
		work_queue_mutex[i] = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
		pthread_mutex_init(work_queue_mutex[i], NULL);
	}


	// 4. no-hook switch.
	no_hook_thr = (bool*)dlsym(klib, "no_hook");

	// 5. Setup metadata table.
	setup_metadata();
	

	// for now, those are just all. now we can use those variables in hooking.cpp.
}

/*
	create THREAD_NUM streams,
	where the last stream is high priority. (curerntly same as Orion.)

	beside the last stream, we need stream for fake launch.
	this one got the highest priority.
	
*/
void create_streams() {
    int low_priority, high_priority;
    cudaError_t status = cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);
    if (status != cudaSuccess) {
        fprintf(stderr, "Cannot get stream priorities: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < THREAD_NUM; ++i) {
        cudaStream_t stream;
        int priority = low_priority;
        if (i == THREAD_NUM - 1 && low_priority != high_priority) priority = high_priority - 1;
        status = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
        if (status != cudaSuccess) {
            fprintf(stderr, "Cannot create default stream for client %d: %s\n", i, cudaGetErrorString(status));
            std::exit(EXIT_FAILURE);
        }
        work_queue[i].push_back(std::make_shared<stream_queue>(i, stream));
    }

    status = cudaStreamCreateWithPriority(&metadata_pass_stream, cudaStreamNonBlocking, high_priority);
    if (status == cudaSuccess)
        status = cudaStreamCreateWithPriority(&launch_block_stream, cudaStreamNonBlocking, high_priority);
    if (status != cudaSuccess) {
        fprintf(stderr, "Cannot create scheduler stream: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }

}

/*
	call initial_wrapper_run() and initial_nothing_run() to assign
	wrapper256() (and more later!), do_nothing to variables at libsmctrl.c.

	+ call assign_hash_insert() to assign hash insert function to libsmctrl.
*/
void assign_launch() {
	libsmctrl_false_launch_callback();
	callback_mode = 0;
	initial_wrapper_run();
	callback_mode = 1;
}



// The caller owns the request until done is published. The scheduler alone owns
// each private marker, so user event destruction/re-recording cannot retarget a wait.
struct pending_checkpoint {
    CUevent marker;
    driver_request* request;
};

static void finish_request(driver_request* request, CUresult status) {
    request->status = status;
    request->done.store(true, std::memory_order_release);
}

static CUresult record_checkpoint(CUstream stream, CUevent* marker) {
    CUresult status = cuEventCreate(marker, CU_EVENT_DISABLE_TIMING);
    if (status != CUDA_SUCCESS) return status;
    status = cuEventRecord(*marker, stream);
    if (status != CUDA_SUCCESS) {
        cuEventDestroy(*marker);
        *marker = NULL;
    }
    return status;
}

// Take a snapshot of the event's current recording. Give a pending wait its
// own nonblocking stream so an unrelated wait cannot hold it up.
static CUresult event_checkpoint(CUevent event, CUevent* marker) {
    CUresult status = cuEventQuery(event);
    if (status != CUDA_ERROR_NOT_READY) return status;
    CUstream stream;
    status = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    if (status != CUDA_SUCCESS) return status;
    status = cuStreamWaitEvent(stream, event, 0);
    if (status == CUDA_SUCCESS) status = record_checkpoint(stream, marker);
    CUresult destroy_status = cuStreamDestroy(stream);
    if (status == CUDA_SUCCESS && destroy_status != CUDA_SUCCESS) {
        cuEventDestroy(*marker);
        *marker = NULL;
        return destroy_status;
    }
    return status;
}

/*
	for now, the scheduler runs in round-robin fashion.
	no priority, no streams, just running.
*/
void* scheduler(void* scarg) {
    scheduler_arg* args = (scheduler_arg*)scarg;
    CUresult context_status = cuCtxSetCurrent(args->context);
    if (context_status != CUDA_SUCCESS) {
        fprintf(stderr, "Cannot bind scheduler CUDA context: %d\n", (int)context_status);
        std::exit(EXIT_FAILURE);
    }

    void* memory_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    void* (*memory_dlsym)(void*, const char*) =
        (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    assert(memory_handle != NULL && memory_dlsym != NULL);
    CUresult (CUDAAPI *actual_cuMemAlloc)(CUdeviceptr*, size_t) =
        (CUresult (CUDAAPI *)(CUdeviceptr*, size_t))memory_dlsym(memory_handle, "cuMemAlloc_v2");
    assert(actual_cuMemAlloc != NULL);
    CUresult (CUDAAPI *actual_cuMemAllocHost)(void**, size_t) =
        (CUresult (CUDAAPI *)(void**, size_t))memory_dlsym(memory_handle, "cuMemAllocHost_v2");
    assert(actual_cuMemAllocHost != NULL);
    CUresult (CUDAAPI *actual_cuMemHostAlloc)(void**, size_t, unsigned int) =
        (CUresult (CUDAAPI *)(void**, size_t, unsigned int))memory_dlsym(memory_handle, "cuMemHostAlloc");
    assert(actual_cuMemHostAlloc != NULL);
    CUresult (CUDAAPI *actual_cuMemFree)(CUdeviceptr) =
        (CUresult (CUDAAPI *)(CUdeviceptr))memory_dlsym(memory_handle, "cuMemFree_v2");
    assert(actual_cuMemFree != NULL);
    CUresult (CUDAAPI *actual_cuMemFreeHost)(void*) =
        (CUresult (CUDAAPI *)(void*))memory_dlsym(memory_handle, "cuMemFreeHost");
    assert(actual_cuMemFreeHost != NULL);

    std::vector<pending_checkpoint> checkpoints;
    int turn = 0;
    pthread_mutex_lock(&start_mutex);
    ready = true;
    pthread_cond_broadcast(&start_cond);
    pthread_mutex_unlock(&start_mutex);
    fprintf(stderr, "scheduler init...\n");

    while (true) {
        for (size_t i = 0; i < checkpoints.size();) {
            pending_checkpoint item = checkpoints[i];
            CUresult status = cuEventQuery(item.marker);
            if (status == CUDA_ERROR_NOT_READY) { ++i; continue; }
            CUresult cleanup = cuEventDestroy(item.marker);
            checkpoints[i] = checkpoints.back();
            checkpoints.pop_back();
            finish_request(item.request, status == CUDA_SUCCESS ? cleanup : status);
        }
		// reap every possible event in the release queue, and call hash_delete.
		bool release_queue_empty = true;
		for(int i = 0; i < RELEASE_QUEUE_LENGTH; i++) {
			if(release_queue[i].key != 0) {
				cudaError_t query = cudaEventQuery(release_queue[i].event);
                if (query != cudaSuccess && query != cudaErrorNotReady) {
                    fprintf(stderr, "Cannot reap atom: %s\n", cudaGetErrorString(query));
                    std::exit(EXIT_FAILURE);
                }
                if (query == cudaSuccess) {
					if(!release_queue[i].reaped) hash_delete(release_queue[i].key);
					cudaEventDestroy(release_queue[i].event);
					cuMemFree(release_queue[i].launchblock);
					release_queue[i].key = 0;
            		release_queue[i].event = 0;
            		release_queue[i].reaped = false;
					
				}
				release_queue_empty = false;
			}
			
		}


        // Clients synchronize before returning. Once every producer has finished,
        // drain all records before stopping. Use an external timeout for hangs.
        if (clients_done.load() == THREAD_NUM) {
            bool empty = true;
            for (int i = 0; i < THREAD_NUM; ++i) {
                pthread_mutex_lock(work_queue_mutex[i]);
                for (const stream_queue_ptr& queue : work_queue[i])
                    empty &= queue->records.empty();
                pthread_mutex_unlock(work_queue_mutex[i]);
            }
            if (empty && checkpoints.empty() && release_queue_empty) return nullptr;
        }
		// pop one from queue, and assign.
		pthread_mutex_lock(work_queue_mutex[turn]);
        stream_queue_ptr queue;
        for (const stream_queue_ptr& candidate : work_queue[turn]) {
            if (!candidate->records.empty()) { queue = candidate; break; }
        }
        if (queue) {
            queue_record qrecord = queue->records.front();
            pthread_mutex_unlock(work_queue_mutex[turn]);
            CUstream stream = queue->gpu;
            driver_request* request = NULL;
            CUresult status = CUDA_SUCCESS;
            CUevent marker = NULL;
            stream_queue_ptr created;
            CUstream rollback_stream = NULL;

			switch(qrecord.type) {

                case RECORD_DEVICE_SYNC:
                    request = qrecord.data.request;
                    status = cuEventCreate(&marker, CU_EVENT_DISABLE_TIMING);
                    if (status == CUDA_SUCCESS) {
                        status = cuCtxRecordEvent(args->context, marker);
                        if (status != CUDA_SUCCESS) {
                            cuEventDestroy(marker);
                            marker = NULL;
                        }
                    }
                    break;
                case RECORD_STREAM_CREATE: {
                    request = qrecord.data.request;
                    if (!request->stream_out) { status = CUDA_ERROR_INVALID_VALUE; break; }
                    try {
                        created = std::make_shared<stream_queue>(turn, nullptr);
                    } catch (const std::bad_alloc&) {
                        status = CUDA_ERROR_OUT_OF_MEMORY;
                        break;
                    }
                    status = cuStreamCreate(&created->gpu, request->flags);
                    if (status != CUDA_SUCCESS) created.reset();
                    break;
                }
                case RECORD_STREAM_DESTROY:
                    request = qrecord.data.request;
                    status = request->destroy_stream(request->stream);
                    break;
                case RECORD_STREAM_SYNC:
                    request = qrecord.data.request;
                    status = record_checkpoint(dispatch_stream(request->stream, stream), &marker);
                    break;
                case RECORD_EVENT_CREATE:
                    request = qrecord.data.request;
                    status = cuEventCreate(request->event_out, request->flags);
                    break;
                case RECORD_EVENT_DESTROY:
                    request = qrecord.data.request;
                    status = request->destroy_event(request->event);
                    break;
                case RECORD_EVENT_QUERY:
                    request = qrecord.data.request;
                    status = cuEventQuery(request->event);
                    break;
                case RECORD_EVENT_RECORD:
                    request = qrecord.data.request;
                    status = cuEventRecord(request->event, dispatch_stream(request->stream, stream));
                    break;
                case RECORD_EVENT_SYNC:
                    request = qrecord.data.request;
                    status = event_checkpoint(request->event, &marker);
                    break;

                case RECORD_CUMEMALLOC: {
                    record_cuMemAlloc record = qrecord.data.r_cuMemAlloc;
                    *record.status = actual_cuMemAlloc(record.dptr, record.bytesize);
                }
                break;

                case RECORD_CUMEMALLOCHOST: {
                    record_cuMemAllocHost record = qrecord.data.r_cuMemAllocHost;
                    *record.status = actual_cuMemAllocHost(record.pp, record.bytesize);
                }
                break;

                case RECORD_CUMEMHOSTALLOC: {
                    record_cuMemHostAlloc record = qrecord.data.r_cuMemHostAlloc;
                    *record.status = actual_cuMemHostAlloc(record.pp, record.bytesize, record.flags);
                }
                break;

                case RECORD_CUMEMFREE: {
                    record_cuMemFree record = qrecord.data.r_cuMemFree;
                    *record.status = actual_cuMemFree(record.dptr);
                }
                break;

                case RECORD_CUMEMFREEHOST: {
                    record_cuMemFreeHost record = qrecord.data.r_cuMemFreeHost;
                    *record.status = actual_cuMemFreeHost(record.p);
                }
                break;

				case RECORD_CULAUNCHKERNEL: {
					record_cuLaunchKernel record = qrecord.data.r_cuLaunchKernel;
                    stream = dispatch_stream(record.hStream, stream);
					// fprintf(stderr, "scheduler found pending job of #%d: function = %ld, lidx = %d\n", turn, (uint64_t)record.f, record.lidx);
					// TODO: how to pass status?
					launch_lidx = record.lidx;
					launch_hidx = record.hidx;
					launch_signal = 1;

					cudaEvent_t launch_finish_event;
					cudaEventCreateWithFlags(&launch_finish_event, cudaEventDisableTiming);

					CUdeviceptr launch_block_memory = 0;
					cuMemAlloc(&launch_block_memory, sizeof(uint32_t));
    				cuMemsetD32Async(launch_block_memory, 0, 1, (CUstream)launch_block_stream);
					cuStreamSynchronize(launch_block_stream);
				
					cuStreamWaitValue32((CUstream)stream, launch_block_memory, 1, CU_STREAM_WAIT_VALUE_EQ);

					(*actual_cuLaunchKernel)(record.f, record.gridDimX, record.gridDimY, record.gridDimZ, record.blockDimX, record.blockDimY, record.blockDimZ, record.sharedMemBytes, stream, record.kernelParams, record.extra);

					hash_insert(last_key, last_metadata);

					cuStreamWriteValue32((CUstream)launch_block_stream, launch_block_memory, 1, 0);
					
					cudaEventRecord(launch_finish_event, stream);
					release_queue_insert(last_key, launch_finish_event, launch_block_memory);
					// fprintf(stderr, "scheduler finish submitting job of #%d: function = %ld, lidx = %d\n", turn, (uint64_t)record.f, record.lidx);

				}
				break;

                case RECORD_CUDAMEMSET: {
                    record_cudaMemset record = qrecord.data.r_cudaMemset;
                    *record.status = actual_cudaMemsetAsync(record.devPtr, record.value, record.count, stream);
                    break;
                }
                case RECORD_CUDAMEMSETASYNC: {
                    record_cudaMemsetAsync record = qrecord.data.r_cudaMemsetAsync;
                    stream = dispatch_stream(record.stream, stream);
                    *record.status = actual_cudaMemsetAsync(record.devPtr, record.value, record.count, stream);
                    break;
                }

				default:
				fprintf(stderr, "Error: unknown record type\n");
                    std::exit(EXIT_FAILURE);
			}
			pthread_mutex_lock(work_queue_mutex[turn]);
            queue->records.pop();
            if (created) {
                try {
                    work_queue[turn].push_back(created);
                    *request->stream_out = created->gpu;
                } catch (const std::bad_alloc&) {
                    rollback_stream = created->gpu;
                    status = CUDA_ERROR_OUT_OF_MEMORY;
                }
            }
            if (qrecord.type == RECORD_STREAM_DESTROY && queue->closing) {
                if (status == CUDA_SUCCESS) {
                    client_streams& streams = work_queue[turn];
                    streams.erase(std::find(streams.begin(), streams.end(), queue));
                } else {
                    queue->closing = false;
                }
            }
            queue->retired.fetch_add(1, std::memory_order_release);
            pthread_mutex_unlock(work_queue_mutex[turn]);
            if (rollback_stream) cuStreamDestroy(rollback_stream);
            if (marker) {
                try {
                    checkpoints.push_back({marker, request});
                } catch (const std::bad_alloc&) {
                    cuEventDestroy(marker);
                    finish_request(request, CUDA_ERROR_OUT_OF_MEMORY);
                }
            } else if (request) {
                finish_request(request, status);
            }

		} else {
			pthread_mutex_unlock(work_queue_mutex[turn]);
		}
		turn = (turn + 1) % THREAD_NUM;
	}

	return nullptr;
	
}

int main(int argc, char** argv) {

	// create N client threads and 1 scheduler thread.
	pthread_t threads[THREAD_NUM + 1];

	// data structure used for N client threads.
	arg_t args[THREAD_NUM];
    if (argc == 2 && std::string(argv[1]) == "--list") { testcase::list(); return 0; }
    if (argc != 1 && argc != 2 && argc != THREAD_NUM + 1) {
        fprintf(stderr, "Usage: %s [name[:size[:iterations[:work[:seed]]]]]\n"
                        "Supply one workload for all clients, or exactly %d specs.\n", argv[0], THREAD_NUM);
        return 2;
    }
    try {
        for (int i = 0; i < THREAD_NUM; ++i) {
            const char* spec = argc == 1 ? "coverage" : argv[argc == 2 ? 1 : i + 1];
            args[i].selection = testcase::parse(spec);
            args[i].result = {false, "Not run"};
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what()); return 2;
    }

	size_t scheduler_idx = THREAD_NUM;

	printf("starting...\n");

	// register real kernel functions.
	register_functions();

	printf("register_functions done.\n");

	// setup variables that is from hooking.cpp.
	variables_setup();

	printf("variables_setup done.\n");

	// create THREAD_NUM streams.
	create_streams();
	
	printf("create_streams done.\n");

	// Kernel Launch of wrapper and idle kernel, to assign wrapper kernel and idle kernel.
	// These launches should not be hooked.
	assign_launch();

	printf("assign_launch done.\n");

    // Capture the context that owns the streams and metadata before starting clients.
    scheduler_arg scarg = {};
    CUresult context_status = cuCtxGetCurrent(&scarg.context);
    if (context_status != CUDA_SUCCESS || scarg.context == NULL) {
        fprintf(stderr, "Cannot obtain main CUDA context: status=%d, context=%p\n",
                (int)context_status, (void*)scarg.context);
        return EXIT_FAILURE;
    }



	// create [num] threads to run kernel.
	// each thread gets arguments.
	printf("creating clients...\n");
	for(int i = 0; i < THREAD_NUM; i++) {
		if (pthread_create(&threads[i], NULL, thread_wrapper, &args[i]) != 0) {
            fprintf(stderr, "Client creation failed\n"); std::exit(EXIT_FAILURE);
        }
		printf("created thread %d: id %ld\n", i, threads[i]);
	}

	// setup of thread_ids is done here.
	pthread_t** tids = (pthread_t**)dlsym(klib, "thread_ids");
	*tids = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));
	for (int i = 0; i < THREAD_NUM; i++) {
		(*tids)[i] = threads[i];
	}

    // Publish fully initialized queues and thread IDs to the interposer.
    __atomic_store_n(no_hook_thr, false, __ATOMIC_RELEASE);

	// create scheduler.
	printf("creating scheduler...\n");
	if (pthread_create(&threads[scheduler_idx], NULL, scheduler, &scarg) != 0) {
        fprintf(stderr, "Scheduler creation failed\n"); std::exit(EXIT_FAILURE);
    }
	printf("created scheduler: id %ld\n", threads[scheduler_idx]);

	// join everything.
	for(int i = 0; i < THREAD_NUM + 1; i++) {
		pthread_join(threads[i], NULL);
	}
	printf("launch complete.\n");

    bool passed = true;
    for (int i = 0; i < THREAD_NUM; ++i) {
        const auto& result = args[i].result;
        printf("client %d %s: %s: %s\n", i, args[i].selection.entry->name,
               result.passed ? "PASS" : "FAIL", result.message.c_str());
        passed &= result.passed;
    }
    return passed ? 0 : 1;
}
