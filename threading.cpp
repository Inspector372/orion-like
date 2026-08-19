/*
	threading.cpp

*/

#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>
#include <iostream>
#include <queue>
#include <cstring>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <assert.h>

#include "kernel_example.h"
#include "hooking.h"
#include "wrapper.h"
#include "libsmctrl.h"

#define THREAD_NUM 4
#define LEN 4532

using namespace std;

CUresult (*actual_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);

CUresult (*actual_cuFuncGetParamInfo)(CUfunction, size_t, size_t*, size_t*);


void* klib;


// Work queue for N client threads.
// Need mutex lock for those.
queue<queue_record>** work_queue;
pthread_mutex_t** work_queue_mutex;

// This is for letting threads stall before all setups are done.
// it's possible to do this because this is a toy experiment,
// but need to wrap functions with mutex, or remove it later.
pthread_mutex_t start_mutex;

// This is for global variable kernel_ptrs_index in hooking.cpp.
// we may need this index in threading.cpp,
// but for now it has no use.
pthread_mutex_t* kernel_ptrs_mutex_thr;

// the variable that prevents hooking.
bool* no_hook_thr;

// Streams for each clients.
cudaStream_t** sched_streams;

// Stream for fake launch, gets the highest priority.
cudaStream_t* fake_launch_stream;

time_t start_os;

typedef struct scheduler_arg {
	int PLACEHOLDER;
} scheduler_arg;


void hash_insert(uint64_t key, AtomMetaData value) {
	table_insert(key, value);
}

/* imported from Orion, RTLD_DEFAULT -> handle */
void register_functions() {
	void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);

    // for kernel
	*(void **)(&actual_cuLaunchKernel) = dlsym(handle, "cuLaunchKernel");
	assert(actual_cuLaunchKernel != NULL);


	// for inspections, we need to load those functions too.
	*(void **)(&actual_cuFuncGetParamInfo) = dlsym (handle, "cuFuncGetParamInfo");
	assert (actual_cuFuncGetParamInfo != NULL);

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
		(*mutex_ptr)[i] = new pthread_mutex_t();
	}

	// 3. mutex for global variable kernel_ptrs_index.
	kernel_ptrs_mutex_thr = (pthread_mutex_t*)dlsym(klib, "kernel_ptrs_mutex");

	// 4. no-hook switch.
	no_hook_thr = (bool*)dlsym(klib, "no_hook");

	// 5. launch metadata address sharing to hooking.cpp
	LaunchMetaData_hooking_t** launchMeta = (LaunchMetaData_hooking_t**)dlsym(klib, "launchMetaData_hooking");
	*launchMeta = (LaunchMetaData_hooking_t*)launchMetaData;

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
	if(lp == hp)
		cudaStreamCreateWithPriority(sched_streams[THREAD_NUM - 1], cudaStreamNonBlocking, *hp);
	else
		cudaStreamCreateWithPriority(sched_streams[THREAD_NUM - 1], cudaStreamNonBlocking, *hp - 1);

	cudaStream_t* fake_launch_stream_ptr = (cudaStream_t*)dlsym(klib, "fl_stream");
	cudaStreamCreateWithPriority(fake_launch_stream_ptr, cudaStreamNonBlocking, *hp);

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
	initial_nothing_run();
	callback_mode = 2;
	*no_hook_thr = false;
}



/*
	for now, the scheduler runs in round-robin fashion.
	no priority, no streams, just running.
*/
void* scheduler(void* scarg) {
	int turn = 0;
	int job_count = 0;
	int total_job = THREAD_NUM * 10000; // currently only 1 kernel is launched per thread.

	pthread_mutex_lock(&start_mutex);
    pthread_mutex_unlock(&start_mutex);
	fprintf(stderr, "scheduler init...\n");

	while(1) {
		// return after (JOB_NUM) number of jobs.
		if (job_count == total_job) {
			fprintf(stderr, "scheduler return - expected %d jobs completed\n", job_count);
			return nullptr;
		}
		else if (time(NULL) - start_os > 25) {
			fprintf(stderr, "scheduler return - total %d jobs completed, timeout of 25 seconds\n", job_count);
			return nullptr;
		}
		// pop one from queue, and assign.
		pthread_mutex_lock(work_queue_mutex[turn]);
		if(!(*work_queue[turn]).empty()) {
			// this routine should be something like assign_job(),
			// and changed when we intercept other cuda calls.

			// TODO: add branch for other record types.
			// currently all types are r_cuLaunchKernel.
			queue_record qrecord = (*work_queue[turn]).front();
			record_cuLaunchKernel record = qrecord.data.r_cuLaunchKernel;

			// TODO: how to pass status?
			// fprintf(stderr, "job %d running\n", turn);
			// fprintf(stderr, "sched_streams[turn]: %p\n", *sched_streams[turn]);

    		(*actual_cuLaunchKernel)(record.f, record.gridDimX, record.gridDimY, record.gridDimZ, record.blockDimX, record.blockDimY, record.blockDimZ, record.sharedMemBytes, *sched_streams[turn], record.kernelParams, record.extra);

			(*work_queue[turn]).pop();
			// fprintf(stderr, "scheduler finish job of #%d\n", turn);
			job_count++;

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
	int* h_As[THREAD_NUM];
	int* h_Bs[THREAD_NUM];
	int* h_outs[THREAD_NUM];
	addKernel_arg args[THREAD_NUM];

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

	// before spawning threads, acquire start mutex.
	pthread_mutex_init(&start_mutex, NULL);
	pthread_mutex_lock(&start_mutex);

	// create [num] threads to run kernel.
	// each thread gets arguments.
	printf("creating clients...\n");
	for(int i = 0; i < THREAD_NUM; i++) {
		h_As[i] = (int*)malloc(sizeof(int) * LEN);
		h_Bs[i] = (int*)malloc(sizeof(int) * LEN);
		h_outs[i] = (int*)malloc(sizeof(int) * LEN);
		for(int j = 0; j < LEN; j++) {
			h_As[i][j] = j;
			h_Bs[i][j] = j;
			h_outs[i][j] = 0;
		}
		args[i] = {LEN, h_As[i], h_Bs[i], h_outs[i], &start_mutex};
		// pthread_create(&threads[i], NULL, addKernel_wrap, (void *)&args[i]);
		pthread_create(&threads[i], NULL, softmaxKernel_wrap, (void *)&args[i]);
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
	pthread_create(&threads[scheduler_idx], NULL, scheduler, (void *)&scarg);
	printf("created scheduler: id %ld\n", threads[scheduler_idx]);

	// **unblock** every threads and start launching.
	start_os = time(NULL);
	printf("launching...\n");
	pthread_mutex_unlock(&start_mutex);

	// join everything.
	for(int i = 0; i < THREAD_NUM + 1; i++) {
		pthread_join(threads[i], NULL);
	}
	printf("launch complete.\n");


	// cleanup.
	for(int i = 0; i < THREAD_NUM; i++) {
		free(h_As[i]);
		free(h_Bs[i]);
		free(h_outs[i]);
	}
    return 0;
}