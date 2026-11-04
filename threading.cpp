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
cudaError_t (*actual_cudaMemset)(void*, int, size_t) = nullptr;
cudaError_t (*actual_cudaMemsetAsync)(void*, int, size_t, cudaStream_t) = nullptr;

decltype(&cuMemAlloc) actual_cuMemAlloc = nullptr;
decltype(&cuMemAllocHost) actual_cuMemAllocHost = nullptr;
decltype(&cuMemHostAlloc) actual_cuMemHostAlloc = nullptr;
decltype(&cuMemFree) actual_cuMemFree = nullptr;
decltype(&cuMemFreeHost) actual_cuMemFreeHost = nullptr;
decltype(&cuCtxGetCurrent) actual_cuCtxGetCurrent = nullptr;
decltype(&cuCtxPushCurrent) actual_cuCtxPushCurrent = nullptr;
decltype(&cuCtxPopCurrent) actual_cuCtxPopCurrent = nullptr;

CUresult (*actual_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
CUresult (*actual_cuFuncGetParamInfo)(CUfunction, size_t, size_t*, size_t*);

/*
	hooking.cpp.
*/
void* klib;


/*
	Kernel Queue.
*/
queue<queue_record>** work_queue;
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
	Streams for each clients.
*/
cudaStream_t** sched_streams;


/*
	Stream for metadata passing / launch blocking APIs.
*/
cudaStream_t metadata_pass_stream;
cudaStream_t launch_block_stream;

/*
	Argument for scheduler (currently none.)
*/
typedef struct scheduler_arg {
	int PLACEHOLDER;
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

    actual_cuMemAlloc = reinterpret_cast<decltype(actual_cuMemAlloc)>(
        actual_dlsym(handle, "cuMemAlloc_v2"));
    assert(actual_cuMemAlloc != nullptr);
    actual_cuMemAllocHost = reinterpret_cast<decltype(actual_cuMemAllocHost)>(
        actual_dlsym(handle, "cuMemAllocHost_v2"));
    assert(actual_cuMemAllocHost != nullptr);
    actual_cuMemHostAlloc = reinterpret_cast<decltype(actual_cuMemHostAlloc)>(
        actual_dlsym(handle, "cuMemHostAlloc"));
    assert(actual_cuMemHostAlloc != nullptr);
    actual_cuMemFree = reinterpret_cast<decltype(actual_cuMemFree)>(
        actual_dlsym(handle, "cuMemFree_v2"));
    assert(actual_cuMemFree != nullptr);
    actual_cuMemFreeHost = reinterpret_cast<decltype(actual_cuMemFreeHost)>(
        actual_dlsym(handle, "cuMemFreeHost"));
    assert(actual_cuMemFreeHost != nullptr);
    actual_cuCtxGetCurrent = reinterpret_cast<decltype(actual_cuCtxGetCurrent)>(
        actual_dlsym(handle, "cuCtxGetCurrent"));
    assert(actual_cuCtxGetCurrent != nullptr);
    actual_cuCtxPushCurrent = reinterpret_cast<decltype(actual_cuCtxPushCurrent)>(
        actual_dlsym(handle, "cuCtxPushCurrent_v2"));
    assert(actual_cuCtxPushCurrent != nullptr);
    actual_cuCtxPopCurrent = reinterpret_cast<decltype(actual_cuCtxPopCurrent)>(
        actual_dlsym(handle, "cuCtxPopCurrent_v2"));
    assert(actual_cuCtxPopCurrent != nullptr);

	// for wrapper, initial wrapper run.
	*(void**)(&actual_cudaDeviceSynchronize) = dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    assert(actual_cudaDeviceSynchronize != nullptr);

	*(void**)(&actual_cudaMemset) = dlsym(RTLD_NEXT, "cudaMemset");
    assert(actual_cudaMemset != nullptr);

	*(void**)(&actual_cudaMemsetAsync) = dlsym(RTLD_NEXT, "cudaMemsetAsync");
    assert(actual_cudaMemsetAsync != nullptr);

    // assign hash_insert_callback of libsmctrl.
	assign_hash_insert((void*)hash_insert);

}

void variables_setup() {
	klib = dlopen("./hooking.so", RTLD_NOW | RTLD_GLOBAL);

	// 1. queue for each thread.
	queue<queue_record>*** work_queue_ptr = (queue<queue_record>***)dlsym(klib, "work_queue");
	*work_queue_ptr = (queue<queue_record>**)malloc(THREAD_NUM * sizeof(queue<queue_record>*));
	work_queue = *work_queue_ptr;
	for (int i = 0; i < THREAD_NUM; i++) {
		(*work_queue_ptr)[i] = new queue<queue_record>();
	}

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
	int* lp = (int*)malloc(sizeof(int));
	int* hp = (int*)malloc(sizeof(int));

	cudaDeviceGetStreamPriorityRange(lp, hp);

	sched_streams = (cudaStream_t**)malloc((THREAD_NUM) * sizeof(cudaStream_t*));
	for(int i = 0; i < THREAD_NUM - 1; i++) {
		sched_streams[i] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[i], cudaStreamNonBlocking, *lp);
	}
	sched_streams[THREAD_NUM - 1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
	if(*lp == *hp)
		cudaStreamCreateWithPriority(sched_streams[THREAD_NUM - 1], cudaStreamNonBlocking, *hp);
	else
		cudaStreamCreateWithPriority(sched_streams[THREAD_NUM - 1], cudaStreamNonBlocking, *hp - 1);

 
	cudaStreamCreateWithPriority(&metadata_pass_stream, cudaStreamNonBlocking, *hp);
	cudaStreamCreateWithPriority(&launch_block_stream, cudaStreamNonBlocking, *hp);

	free(lp);
	free(hp);

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
	*no_hook_thr = false;
}



/*
	for now, the scheduler runs in round-robin fashion.
	no priority, no streams, just running.
*/

static CUresult execute_driver_memory_raw(record_type type, const record_driver_memory& record) {
    switch (type) {
        case RECORD_CUMEMALLOC:
            return actual_cuMemAlloc(record.device_out, record.bytesize);
        case RECORD_CUMEMALLOCHOST:
            return actual_cuMemAllocHost(record.host_out, record.bytesize);
        case RECORD_CUMEMHOSTALLOC:
            return actual_cuMemHostAlloc(record.host_out, record.bytesize, record.flags);
        case RECORD_CUMEMFREE:
            return actual_cuMemFree(record.device_ptr);
        case RECORD_CUMEMFREEHOST:
            return actual_cuMemFreeHost(record.host_ptr);
        default: return CUDA_ERROR_INVALID_VALUE;
    }
}

// A null caller context must not accidentally inherit the scheduler context.
// Temporarily unwind the context stack, then restore it in the same order.
static CUresult execute_memory_without_context(record_type type,
                                               const record_driver_memory& record) {
    CUcontext current = nullptr;
    CUresult status = actual_cuCtxGetCurrent(&current);
    if (status != CUDA_SUCCESS) return status;
    if (!current) return execute_driver_memory_raw(type, record);
    status = actual_cuCtxPopCurrent(&current);
    if (status != CUDA_SUCCESS) return status;
    status = execute_memory_without_context(type, record);
    CUresult restore = actual_cuCtxPushCurrent(current);
    return status == CUDA_SUCCESS ? restore : status;
}

static CUresult execute_driver_memory(record_type type, const record_driver_memory& record) {
    if (!record.context) return execute_memory_without_context(type, record);
    CUresult status = actual_cuCtxPushCurrent(record.context);
    if (status != CUDA_SUCCESS) return status;
    status = execute_driver_memory_raw(type, record);
    CUcontext popped = nullptr;
    CUresult restore = actual_cuCtxPopCurrent(&popped);
    return status == CUDA_SUCCESS ? restore : status;
}

void* scheduler(void* scarg) {
	int turn = 0;
	(void)scarg;
    pthread_mutex_lock(&start_mutex);
    ready = true;
    pthread_cond_broadcast(&start_cond);
    pthread_mutex_unlock(&start_mutex);
    fprintf(stderr, "scheduler init...\n");

    while (true) {
		// reap every possible event in the release queue, and call hash_delete.
		bool release_queue_empty = true;
		for(int i = 0; i < RELEASE_QUEUE_LENGTH; i++) {
			if(release_queue[i].key != 0) {
				if(cudaEventQuery(release_queue[i].event) == cudaSuccess) {
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
                empty &= work_queue[i]->empty();
                pthread_mutex_unlock(work_queue_mutex[i]);
            }
            if (empty) return nullptr;
        }
		// pop one from queue, and assign.
		pthread_mutex_lock(work_queue_mutex[turn]);
		if(!(*work_queue[turn]).empty()) {
			// We should delay popping until the end of the execution,
			// otherwise the client will proceed and may invaildate some arguments.
			queue_record qrecord = (*work_queue[turn]).front();
			pthread_mutex_unlock(work_queue_mutex[turn]);

            driver_memory_completion* memory_completion = nullptr;
            CUresult memory_status = CUDA_SUCCESS;
			switch(qrecord.type) {
                case RECORD_CUMEMALLOC:
                case RECORD_CUMEMALLOCHOST:
                case RECORD_CUMEMHOSTALLOC:
                case RECORD_CUMEMFREE:
                case RECORD_CUMEMFREEHOST:
                {
                    // This runs only between complete launch/gate-release transactions.
                    // Never call a potentially blocking free while a gate still needs
                    // to be opened by this scheduler.
                    const auto& record = qrecord.data.r_driver_memory;
                    memory_completion = record.completion;
                    memory_status = execute_driver_memory(qrecord.type, record);
                    break;
                }
				case RECORD_CULAUNCHKERNEL: {
					record_cuLaunchKernel record = qrecord.data.r_cuLaunchKernel;
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
				
					cuStreamWaitValue32((CUstream)*sched_streams[turn], launch_block_memory, 1, CU_STREAM_WAIT_VALUE_EQ);

					(*actual_cuLaunchKernel)(record.f, record.gridDimX, record.gridDimY, record.gridDimZ, record.blockDimX, record.blockDimY, record.blockDimZ, record.sharedMemBytes, *sched_streams[turn], record.kernelParams, record.extra);

					hash_insert(last_key, last_metadata);

					cuStreamWriteValue32((CUstream)launch_block_stream, launch_block_memory, 1, 0);
					
					cudaEventRecord(launch_finish_event, *sched_streams[turn]);
					release_queue_insert(last_key, launch_finish_event, launch_block_memory);
					// fprintf(stderr, "scheduler finish submitting job of #%d: function = %ld, lidx = %d\n", turn, (uint64_t)record.f, record.lidx);

				}
				break;

				case RECORD_CUDAEVENT: {
					record_cudaEvent record_event = qrecord.data.r_cudaEvent;
					// fprintf(stderr, "event recorded for #%d\n", turn);
					if (cudaEventRecord(record_event.event, *sched_streams[turn]) != cudaSuccess) {
                        // fprintf(stderr, "Scheduler event record failed\n");
                        std::exit(EXIT_FAILURE);
                    }
				}
				break;

				case RECORD_CUDAMEMSET: {
					record_cudaMemset record_event = qrecord.data.r_cudaMemset;

					if (actual_cudaMemset(record_event.devPtr, record_event.value, record_event.count) != cudaSuccess) {
                        fprintf(stderr, "Scheduler memset record failed\n");
                        std::exit(EXIT_FAILURE);
                    }
				}
				break;

				case RECORD_CUDAMEMSETASYNC: {
					// We use synchronous version... for now.
					record_cudaMemsetAsync record_event = qrecord.data.r_cudaMemsetAsync;

					if (actual_cudaMemset(record_event.devPtr, record_event.value, record_event.count) != cudaSuccess) {
                        fprintf(stderr, "Scheduler memsetAsync record failed\n");
                        std::exit(EXIT_FAILURE);
                    }
				}
				break;

				default:
				fprintf(stderr, "Error: unknown record type\n");
                    std::exit(EXIT_FAILURE);
			}
			pthread_mutex_lock(work_queue_mutex[turn]);
			(*work_queue[turn]).pop();
			pthread_mutex_unlock(work_queue_mutex[turn]);

            if (memory_completion) {
                // Publish output-pointer writes and status only after removing the
                // request. Do not access this caller-owned object after unlocking.
                pthread_mutex_lock(&memory_completion->mutex);
                memory_completion->status = memory_status;
                memory_completion->done = true;
                pthread_cond_signal(&memory_completion->condition);
                pthread_mutex_unlock(&memory_completion->mutex);
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

	// create scheduler.
	printf("creating scheduler...\n");
	scheduler_arg scarg;
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
