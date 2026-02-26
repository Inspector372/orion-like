
// Threading logics prototype.
// Current goal: make queue for each thread, and schedule them in round-robin fashion.
// 1. register real cuda functions to global pointers.
// 2. create N threads that runs kernels 3 times, and make queue for each thread.
// 3. create 1 scheduler() thread that can see all queues from threads.
// 4. let the scheduler round-robin jobs from threads.

/*
# Compile the .cu file into an object file
nvcc -c cuda_functions.cu -o cuda_functions.o

# Compile the .cpp file (using g++)
g++ -c main.cpp -o main.o

# Link the object files and the CUDA runtime library
g++ main.o cuda_functions.o -L/usr/local/cuda/lib64 -lcudart -o my_program
# (Adjust the -L path based on your CUDA Toolkit installation location)
*/

#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>
#include <iostream>
#include <queue>

#include <cuda_runtime.h>
#include <cuda.h>

#include <assert.h>

#include "kernel_example.h"
#include "hooking.h"

#define THREAD_NUM 4
#define LEN 1024

using namespace std;

cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t (*malloc_function)(void** devPtr, size_t size);
cudaError_t (*free_function)(void* devPtr);
cudaError_t (*memset_function)(void* devPtr, int  value, size_t count);
cudaError_t (*memset_async_function)(void* devPtr, int  value, size_t count, cudaStream_t stream);

void* klib;


// Work queue for N client threads.
// Need mutex lock for those.
queue<func_record>** work_queue;
pthread_mutex_t** work_queue_mutex;

// This is for letting threads stall before all setups are done.
// it's possible to do this because this is a toy experiment,
// but need to wrap functions with mutex, or remove it later.
pthread_mutex_t start_mutex;

// Streams.
cudaStream_t** sched_streams;



typedef struct scheduler_arg {
	int PLACEHOLDER;
} scheduler_arg;

/* imported from Orion, RTLD_DEFAULT -> handle */
void register_functions() {
	void* handle = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);

    // for kernel
	*(void **)(&kernel_function) = dlsym(handle, "cudaLaunchKernel");
	assert(kernel_function != NULL);

	// for memcpy
	*(void **)(&memcpy_function) = dlsym (handle, "cudaMemcpy");
	assert(memcpy_function != NULL);

	// for memcpy_async
	*(void **)(&memcpy_async_function) = dlsym (handle, "cudaMemcpyAsync");
	assert(memcpy_async_function != NULL);

	// for malloc
	*(void **)(&malloc_function) = dlsym (handle, "cudaMalloc");
	assert(malloc_function != NULL);

	// for free
	*(void **)(&free_function) = dlsym (handle, "cudaFree");
	assert(free_function != NULL);

	// for memset
	*(void **)(&memset_function) = dlsym (handle, "cudaMemset");
	assert (memset_function != NULL);

	// for memset_async
	*(void **)(&memset_async_function) = dlsym (handle, "cudaMemsetAsync");
	assert (memset_async_function != NULL);

}

void variables_setup() {
	klib = dlopen("./hooking.so", RTLD_NOW | RTLD_GLOBAL);

	// 1. queue for each thread.
	queue<func_record>*** work_queue_ptr = (queue<func_record>***)dlsym(klib, "work_queue");
	*work_queue_ptr = (queue<func_record>**)malloc(THREAD_NUM * sizeof(queue<func_record>*));
	work_queue = *work_queue_ptr;
	for (int i = 0; i < THREAD_NUM; i++) {
		(*work_queue_ptr)[i] = new queue<func_record>();
	}

	// 2. mutexes for queues.
	pthread_mutex_t*** mutex_ptr = (pthread_mutex_t***)dlsym(klib, "work_queue_mutex");
	*mutex_ptr = (pthread_mutex_t**)malloc(THREAD_NUM * sizeof(pthread_mutex_t*));
	work_queue_mutex = *mutex_ptr;
	for (int i = 0; i < THREAD_NUM; i++) {
		(*mutex_ptr)[i] = new pthread_mutex_t();
	}

	// for now, those are just all. now we can use those variables in hooking.cpp.
}

/*
	create THREAD_NUM streams,
	where the last stream is high priority. (curerntly same as Orion.)
	
*/
void create_streams() {
	int* lp = (int*)malloc(sizeof(int));
	int* hp = (int*)malloc(sizeof(int));

	cudaDeviceGetStreamPriorityRange(lp, hp);

	sched_streams = (cudaStream_t**)malloc(THREAD_NUM * sizeof(cudaStream_t*));
	for(int i = 0; i < THREAD_NUM - 1; i++) {
		sched_streams[i] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
		cudaStreamCreateWithPriority(sched_streams[i], cudaStreamNonBlocking, *lp);
	}
	sched_streams[THREAD_NUM - 1] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
	cudaStreamCreateWithPriority(sched_streams[THREAD_NUM - 1], cudaStreamNonBlocking, *hp);

	free(lp);
	free(hp);

}

/*
	for now, the scheduler runs in round-robin fashion.
	no priority, no streams, just running.
*/
void* scheduler(void* scarg) {
	int turn = 0;
	int job_count = 0;
	int total_job = THREAD_NUM; // currently only 1 kernel is launched per thread.

	pthread_mutex_lock(&start_mutex);
    pthread_mutex_unlock(&start_mutex);
	fprintf(stderr, "scheduler init...\n");

	while(1) {
		if (job_count == THREAD_NUM) {
			fprintf(stderr, "scheduler return\n");
			return nullptr;
		}
		// pop one from queue, and assign.
		pthread_mutex_lock(work_queue_mutex[turn]);
		if(!(*work_queue[turn]).empty()) {
			// this routine should be something like assign_job(),
			// and changed when we intercept other cuda calls.
			func_record frecord = (*work_queue[turn]).front();
			kernel_record record = frecord.data.krecord;

			// kernel wrapper needed.
			// TODO: how to pass status?
			cudaError_t status = (*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, *sched_streams[turn]);
			(*work_queue[turn]).pop();
			fprintf(stderr, "scheduler finish job #%d\n", turn);
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
		args[i] = {LEN, h_As[i], h_Bs[i], h_outs[i], &start_mutex};
		pthread_create(&threads[i], NULL, addKernel_wrap, (void *)&args[i]);
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