
// Hooking library - hooks CUDA calls
// Hooking logics are mostly directly imported from Orion.
// env CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=./hooking.so ./threading
// ..

/* 
	goal
		1. fake launch configuration
		2. atomization by 1024 (for now fixed number, but will be changed later .. )

	main problems
		1. where do I put fake launch?
			- looks like work_queue need to be filled with 'real' launches.
			- separate 'highest priority' stream need to be there.
			- submit 'directly' to highest stream at hooker-level, with threadDim = K.
			- append that number K to queue.

		2. clean way to 'actually' link launch
			- TODO

		3. So what should be in queue??
			- cudaLaunchKernel() args (except stream)
			- actual stream to be scheduled, or some hint to this...?
				(for now no need to care)
			- lidx, hidx

		4. when to construct arguments in wrapper()?
			- arg -> at scheduler.
			- func -> (just pass)
			- lidx, hidx -> need to be calculated at hooker and send to scheduler.


	Total capture logic
	1. cudaLaunchKernel() capture
	2. submit func and attribute to per-client work_queue[]
	3. (block until scheduler fetches) -> for other 'blocking' cuda calls!!
		need to modify !!!
	4. scheduler fetches work_queue, and launch (func, attribute) in the queue.

	Modified logic
	1. cudaLaunchKernel() capture
	+. calculate blockDim * threadDim / (NUM_PARTITION)
	+. calculate (lidx, hidx) ranges
	+. 'directly launch' original kernel using highest stream, with threadDIm=K(fake-launch).
	2. submit func and attribute to per-client work_queue[]
	+. submit lidx, hidx, K too.
	3. (block until scheduler fetches) -> for other 'blocking' cuda calls!!
		need to modify !!!
	4. scheduler fetches work_queue (func, attribute, K, lidx, hidx)
	+. checks if kernel_ptrs[K] contains address (if fake launch has finished)
		- policy design -> if not, fetch others? or stall?
	+. fetch kernel_ptrs[K]
	+. construct parameter pack 'arg' from original arguments.
	+. cudaKernelLaunch EQUIVALENT TO wrapper<<<gridDim, blockDim, sharedMem, dedicated_stream>>>(arg, func, lidx, hidx)
		- we shouldn't call this because wrapper is getting hooked!!

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

#include "hooking.h"

#define THREAD_NUM 4

using namespace std;

bool no_hook = 1;

pthread_t* thread_ids;
queue<func_record>** work_queue;
pthread_mutex_t** work_queue_mutex;

cudaError_t (*kernel_func)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
cudaError_t (*paraminfo_func)(const void* func, size_t paramIndex, size_t* paramOffset, size_t* paramSize);



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
	if (kernel_func == NULL) {
		*(void **)(&kernel_func) = dlsym (RTLD_NEXT, "cudaLaunchKernel");
		assert (kernel_func != NULL);
	}

	if (paraminfo_func == NULL) {
		*(void **)(&paraminfo_func) = dlsym (RTLD_NEXT, "cudaFuncGetParamInfo");
		assert (paraminfo_func != NULL);
	}

	if(no_hook) {
		// immediately run the kernel.

	}

	fprintf(stderr, "caught call from someone!\n");
	int idx = get_idx();
	fprintf(stderr, "caught call from %d!\n", idx);

	// TODO: inspect kernel size and setup atomization info
	
	// We postpone false launch to scheduler, because it need to wait until kernel PROGRAM_ADDRESS is fetched.


	cudaError_t err = cudaSuccess;
	kernel_record new_kernel_record;
	
	assert(work_queue_mutex != NULL);
	assert(work_queue != NULL);

	pthread_mutex_lock(work_queue_mutex[idx]);
	
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
