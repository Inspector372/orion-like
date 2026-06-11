
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
			- submit 'directly' to highest stream at hooker-level, with blockDim.x = K.
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
	+. calculate gridDim * blockDim / (NUM_PARTITION)
	+. calculate (lidx, hidx) ranges
	+. 'directly launch' original kernel using highest stream, with blockDim.x=K(fake-launch).
	2. submit func and attribute to per-client work_queue[]
	+. submit lidx, hidx, K too.
	3. (block until scheduler fetches) -> for other 'blocking' cuda calls!!
		need to modify !!!
	4. scheduler fetches work_queue (func, attribute, K, lidx, hidx)
	+. checks if kernel_ptrs[K] contains address (if fake launch has finished)
		- policy design -> if not, fetch others? or stall?
	+. fetch kernel_ptrs[K]
	+. construct parameter pack 'arg' from original arguments.
	+. cudaLaunchKernel EQUIVALENT TO wrapper<<<gridDim, blockDim, sharedMem, dedicated_stream>>>(arg, func, lidx, hidx)
		- we shouldn't call this because wrapper is getting hooked!!

*/

#define _GNU_SOURCE

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
#define MAX_KERNEL_PTRS 256

using namespace std;

bool no_hook = true;
uint32_t kernel_ptrs_index = 1;
pthread_mutex_t kernel_ptrs_mutex;

pthread_t* thread_ids;
queue<func_record>** work_queue;
pthread_mutex_t** work_queue_mutex;

cudaError_t (*kernel_func)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
cudaError_t (*paraminfo_func)(const void* func, size_t paramIndex, size_t* paramOffset, size_t* paramSize);

cudaStream_t fl_stream;


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
		void* cudart_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
		*(void **)(&kernel_func) = dlsym (cudart_handle, "cudaLaunchKernel");
		assert (kernel_func != NULL);
	}

	if (paraminfo_func == NULL) {
		void* cudart_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
		*(void **)(&paraminfo_func) = dlsym (cudart_handle, "cudaFuncGetParamInfo");
		assert (paraminfo_func != NULL);
	}

	if(no_hook) {
		// immediately run the kernel.
		fprintf(stderr, "no-hook launch of %p\n", func);
		return (*kernel_func)(func, gridDim, blockDim, args, sharedMem, stream);
	}

	fprintf(stderr, "caught call from someone!\n");
	int idx = get_idx();
	fprintf(stderr, "caught call from %d!\n", idx);

	// inspect kernel size and setup atomization info.
	// for now, we assume they only have 1 dimension.

	// TODO: dynamically adjust atom_size(need to look at lithOS paper.)
	// TODO: expand them to 3 dimensions.
	int atom_size = 1024;
	int atom_num = ((gridDim.x * blockDim.x) % atom_size == 0) ? (gridDim.x * blockDim.x / atom_size) : (gridDim.x * blockDim.x / atom_size + 1);

	pthread_mutex_lock(&kernel_ptrs_mutex);
	int kptr_idx = kernel_ptrs_index;
	kernel_ptrs_index = (kernel_ptrs_index + 1) % MAX_KERNEL_PTRS;
	if(kernel_ptrs_index == 0) kernel_ptrs_index = 1;
	pthread_mutex_unlock(&kernel_ptrs_mutex);

	// Fake launch.
	dim3 dim_of_idx = dim3(kptr_idx, 1, 1);
	(*kernel_func)(func, gridDim, dim_of_idx, args, sharedMem, fl_stream);

	cudaError_t err = cudaSuccess;
	kernel_record new_kernel_record;
	
	assert(work_queue_mutex != NULL);
	assert(work_queue != NULL);

	// queue multiple kernels of same instance
	new_kernel_record = {func, gridDim, blockDim, args, sharedMem, stream, kptr_idx, 0, atom_size - 1};
	union func_data new_func_data;
	new_func_data.krecord = new_kernel_record;
	func_record new_record = {KERNEL_RECORD, new_func_data};

	pthread_mutex_lock(work_queue_mutex[idx]);

	// Atomization.
	for(int i=0; i<atom_num; i++) {
		work_queue[idx]->push(new_record);
		new_record.data.krecord.lidx += atom_size;
		new_record.data.krecord.hidx += atom_size;
	}
	

	pthread_mutex_unlock(work_queue_mutex[idx]);

	block(idx, work_queue_mutex, work_queue);

    return err;
}

}
