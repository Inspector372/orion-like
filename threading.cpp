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

struct arg_t {
    testcase::Selection selection;
    testcase::Result result;
};
std::atomic<int> clients_done{0};

cudaError_t (*actual_cudaDeviceSynchronize)(void) = nullptr;

CUresult (*actual_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);

// hooking.cpp
void* klib;


// Work queue for N client threads.
// Need mutex lock for those.
queue<queue_record>** work_queue;
pthread_mutex_t** work_queue_mutex;

// Clients wait until registration and scheduler setup are complete.
pthread_mutex_t start_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t start_cond = PTHREAD_COND_INITIALIZER;
bool ready = false;

// This is used for insertion of atomMetaDataTable.
// only one thread can access this table, otherwise it will mess up things.
pthread_mutex_t table_mutex;

// the variable that prevents hooking.
bool* no_hook_thr;

// Streams for each clients.
cudaStream_t** sched_streams;

// Stream for fake launch, gets the highest priority.
cudaStream_t* fake_launch_stream;

// Stream for metadata passing.
cudaStream_t metadata_pass_stream;



typedef struct scheduler_arg {
	int PLACEHOLDER;
} scheduler_arg;


void hash_insert(uint64_t key, AtomMetaData value) {
	pthread_mutex_lock(&table_mutex);
	table_insert(key, value);
	pthread_mutex_unlock(&table_mutex);
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
	*(void **)(&actual_cuLaunchKernel) = dlsym(handle, "cuLaunchKernel");
	assert(actual_cuLaunchKernel != NULL);

	// for wrapper, initial wrapper run.
	*(void**)(&actual_cudaDeviceSynchronize) = dlsym(RTLD_DEFAULT, "cudaDeviceSynchronize");
    assert(actual_cudaDeviceSynchronize != nullptr);

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

	// 3. mutex for table.
	pthread_mutex_t** table_mutex_ptr = (pthread_mutex_t**)dlsym(klib, "table_mutex");
	*table_mutex_ptr = &table_mutex;
	pthread_mutex_init(&table_mutex, NULL);


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

	cudaStream_t* fake_launch_stream_ptr = (cudaStream_t*)dlsym(klib, "fl_stream");
	cudaStreamCreateWithPriority(fake_launch_stream_ptr, cudaStreamNonBlocking, *hp);
 
	cudaStreamCreateWithPriority(&metadata_pass_stream, cudaStreamNonBlocking, *hp);

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
void* scheduler(void* scarg) {
	int turn = 0;
	(void)scarg;
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
            if (empty) return nullptr;
        }
		// pop one from queue, and assign.
		pthread_mutex_lock(work_queue_mutex[turn]);
		if(!(*work_queue[turn]).empty()) {
			queue_record qrecord = (*work_queue[turn]).front();

			switch(qrecord.type) {
				case RECORD_CULAUNCHKERNEL: {
					fprintf(stderr, "scheduler found job of #%d\n", turn);
					record_cuLaunchKernel record = qrecord.data.r_cuLaunchKernel;
					// TODO: how to pass status?
					launch_lidx = record.lidx;
					launch_hidx = record.hidx;
					launch_signal = 1;
					(*actual_cuLaunchKernel)(record.f, record.gridDimX, record.gridDimY, record.gridDimZ, record.blockDimX, record.blockDimY, record.blockDimZ, record.sharedMemBytes, *sched_streams[turn], record.kernelParams, record.extra);
					(*work_queue[turn]).pop();
					fprintf(stderr, "scheduler finish assigning job of #%d\n", turn);

				}
				break;

				case RECORD_CUDAEVENT: {
					record_cudaEvent record_event = qrecord.data.r_cudaEvent;
					fprintf(stderr, "event recorded for #%d\n", turn);
					if (cudaEventRecord(record_event.event, *sched_streams[turn]) != cudaSuccess) {
                        fprintf(stderr, "Scheduler event record failed\n");
                        std::exit(EXIT_FAILURE);
                    }
					(*work_queue[turn]).pop();
				}
				break;

				default:
				fprintf(stderr, "Error: unknown record type\n");
                    std::exit(EXIT_FAILURE);
			}

		}
		pthread_mutex_unlock(work_queue_mutex[turn]);
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
