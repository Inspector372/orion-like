
// Hooking library - hooks CUDA calls
// Hooking logics are mostly directly imported from Orion.
// env CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=./hooking.so ./threading


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

#include "hooking.h"

#define THREAD_NUM 4

using namespace std;

pthread_t* thread_ids;
queue<func_record>** work_queue;
pthread_mutex_t** work_queue_mutex;

extern cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);

// orion uses thread ids to inspect 'what is this thread's thread number'.
// this is a trick, but let's use it.
int get_idx() {
	assert(thread_ids != NULL);
	pthread_t tid = pthread_self();
	fprintf(stderr, "tid = %ld\n", tid);
	// pid_t tid = syscall(SYS_gettid);
	int idx = -1;
	for (int i = 0; i < THREAD_NUM; i++) {
		if (pthread_equal(tid, thread_ids[i])) {
			idx = i;
			break;
		}
	}
	assert(idx != -1);
	fprintf(stderr, "idx = %d\n", idx);
	return idx;
}

// directly adapted from Orion.
void block(int idx, pthread_mutex_t** mutexes, queue<func_record>** kqueues) {
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
	most of those codes are just ctrl CVed from Orion.
	for now, only cudaLaunchKernel is intercepted.
*/
extern "C" {

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream)
{
    // go into work_queue[idx].
    // need to inspect 'who am I'
	fprintf(stderr, "caught call from someone!\n");
	int idx = get_idx();
	fprintf(stderr, "caught call from %d!\n", idx);

	cudaError_t err = cudaSuccess;
	kernel_record new_kernel_record;
	
	assert(work_queue_mutex != NULL);
	assert(work_queue != NULL);

	pthread_mutex_lock(work_queue_mutex[idx]);

	// TODO: inspect kernel size and setup atomization info
	// queue multiple kernels of same instance
	new_kernel_record = {func, gridDim, blockDim, args, sharedMem, stream, false, 0};
	union func_data new_func_data;
	new_func_data.krecord = new_kernel_record;
	func_record new_record = {KERNEL_RECORD, new_func_data};
	work_queue[idx]->push(new_record);

	pthread_mutex_unlock(work_queue_mutex[idx]);

	// wait until kernel is resolved.
	block(idx, work_queue_mutex, work_queue);

    return err;
}

}
