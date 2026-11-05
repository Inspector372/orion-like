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
	Stream for uploading atom metadata.
*/
cudaStream_t metadata_pass_stream;

/*
	Main's initialized CUDA context, shared by scheduler streams and metadata.
*/
typedef struct scheduler_arg {
	CUcontext context;
} scheduler_arg;


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

	*(void**)(&actual_cudaMemset) = dlsym(RTLD_NEXT, "cudaMemset");
    assert(actual_cudaMemset != nullptr);

	*(void**)(&actual_cudaMemsetAsync) = dlsym(RTLD_NEXT, "cudaMemsetAsync");
    assert(actual_cudaMemsetAsync != nullptr);
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

 
	if (cudaStreamCreateWithPriority(&metadata_pass_stream, cudaStreamNonBlocking, *hp) != cudaSuccess) {
		fprintf(stderr, "Cannot create metadata stream\n");
		std::exit(EXIT_FAILURE);
	}

	free(lp);
	free(hp);

}

void assign_launch() {
	libsmctrl_false_launch_callback();
	initial_wrapper_run();
	*no_hook_thr = false;
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

	CUfunction batch_function[THREAD_NUM] = {};
	uint64_t batch_pc[THREAD_NUM] = {};
	int turn = 0;
    pthread_mutex_lock(&start_mutex);
    ready = true;
    pthread_cond_broadcast(&start_cond);
    pthread_mutex_unlock(&start_mutex);
    fprintf(stderr, "scheduler init...\n");

    while (true) {
        // Clients synchronize before returning. Once every producer has finished,
        // drain all records before stopping. Use an external timeout for hangs.
        if (clients_done.load() == THREAD_NUM) {
            bool empty = true;
            for (int i = 0; i < THREAD_NUM; ++i) {
                pthread_mutex_lock(work_queue_mutex[i]);
                empty &= work_queue[i]->empty();
                pthread_mutex_unlock(work_queue_mutex[i]);
            }
            if (empty) {
                shutdown_metadata();
                return nullptr;
            }
        }
		// pop one from queue, and assign.
		pthread_mutex_lock(work_queue_mutex[turn]);
		if(!(*work_queue[turn]).empty()) {
			// We should delay popping until the end of the execution,
			// otherwise the client will proceed and may invaildate some arguments.
			queue_record qrecord = (*work_queue[turn]).front();
			pthread_mutex_unlock(work_queue_mutex[turn]);

			switch(qrecord.type) {

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
					int slot = acquire_atom_slot();
					if (slot < 0) {
						// Keep this record at the head while other clients can progress.
						turn = (turn + 1) % THREAD_NUM;
						continue;
					}
					CUstream stream = *sched_streams[turn];
					auto launch = [&]() {
						return actual_cuLaunchKernel(record.f, record.gridDimX, record.gridDimY,
							record.gridDimZ, record.blockDimX, record.blockDimY, record.blockDimZ,
							record.sharedMemBytes, stream, record.kernelParams, record.extra);
					};
					// Resolve the target before uploading the first atom of this client launch.
					if (record.lidx == 0 || batch_function[turn] != record.f) {
						atom_probe_entry();
						check_atom_driver(launch(), "probe original kernel");
						batch_pc[turn] = atom_finish_entry();
						batch_function[turn] = record.f;
					}
					prepare_atom_slot(slot, batch_pc[turn], record.lidx, record.hidx, stream);
					atom_wrap_entry(slot, batch_pc[turn]);
					check_atom_driver(launch(), "launch atom");
					atom_finish_entry();
					finish_atom_slot(slot, stream);
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

	// Capture the program descriptors for the slot wrappers and the probe.
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
