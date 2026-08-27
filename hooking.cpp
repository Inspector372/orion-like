/*
	hooking.cpp
	How to run: env CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=./hooking.so ./threading
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

using namespace std;

bool no_hook = true;

pthread_t* thread_ids;
queue<queue_record>** work_queue;
pthread_mutex_t** work_queue_mutex;
cudaStream_t fl_stream;

CUresult (*real_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) = NULL;

typedef CUresult (*cuGetProcAddress_t)(const char*, void**, int, unsigned int, void*);
cuGetProcAddress_t real_cuGetProcAddress = NULL;
cuGetProcAddress_t real_cuGetProcAddress_v2 = NULL;
void* (*real_dlsym)(void*, const char*) = NULL;
void* cu_handle = NULL;




// orion uses thread ids to inspect 'what is this thread's thread number'.
// this is a trick, but let's use it.
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

// directly adapted from Orion.
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
	most of those codes are just ctrl CVed from Orion.
	for now, only cudaLaunchKernel is intercepted.

	Now I can intercept cuLaunchKernel too, thanks to 
	curtSCHED: Architecture-Independent Real-Time GPU Scheduling via Statistical Deferrable Servers by Hao Zhang, Frank Muelle(https://github.com/zhanghao5/curtSCHED/blob/main/cusched.cpp)

	cuda c++ resolves names dynamically using this path:
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
extern "C" {

/*
	We need some synchronization behavior for this one.
	for that, *exact* behavior of cudaDeviceSynchronize() need to be inspected.

	For now, it creates a cudaevent, attaches in work_queue.
	and wait until it gets scheduled and eventually finished.
	this works because one thread(user) got only one thread. 
*/
/* cudaError_t cudaDeviceSynchronize(void) {
	
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
	

	fprintf(stderr, "thread %d, waiting until finish\n", idx);
	block(idx, work_queue_mutex, work_queue);
	cudaEventSynchronize(event);
	fprintf(stderr, "thread %d, finished waiting\n", idx);

	cudaEventDestroy(event);
}*/

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

	fprintf(stderr, "[cuHook] caught call from someone!\n");
	int idx = get_idx();
	fprintf(stderr, "[cuHook] caught call from %d!\n", idx);

	// inspect kernel size and setup atomization info.
	// for now, we assume they only have 1 dimension.

	// TODO: dynamically adjust atom_size(need to look at lithOS paper.)
	// TODO: expand them to 3 dimensions.
	int atom_size = 1024;
	int atom_num = ((gridDimX * blockDimX) % atom_size == 0) ? (gridDimX * blockDimX / atom_size) : (gridDimX * blockDimX / atom_size + 1);

	// (Fake launch)
	// real_cuLaunchKernel(f, 1, 1, 1, kptr_idx, 1, 1, sharedMemBytes, (CUstream)fl_stream, kernelParams, extra);

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
	fprintf(stderr, "[cuHook] block-start from %d!\n", idx);
	block(idx, work_queue_mutex, work_queue);
	fprintf(stderr, "[cuHook] block-end from %d!\n", idx);

    return err;

    // return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

CUresult my_cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, unsigned int flags, void* symbolStatus) {
    fprintf(stderr, "[HOOK v1] Inside cuGetProcAddress looking for: %s\n", symbol);

    if (symbol && strcmp(symbol, "cuLaunchKernel") == 0) {
        fprintf(stderr, "[HOOK v1] Hijacking cuLaunchKernel pointer assignment!\n");
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    }

    // Call the real cuGetProcAddress if it's anything else
    if (real_cuGetProcAddress == NULL) {
        if(cu_handle == NULL) cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        real_cuGetProcAddress = (cuGetProcAddress_t)dlvsym(RTLD_NEXT, "cuGetProcAddress", "RTLD_NEXT"); 
        if(!real_cuGetProcAddress) {
             real_cuGetProcAddress = (cuGetProcAddress_t)real_dlsym(cu_handle, "cuGetProcAddress");
        }
    }
    return real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}

CUresult my_cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, unsigned int flags, void* symbolStatus) {
    fprintf(stderr, "[HOOK v2] Inside cuGetProcAddress_v2 looking for: %s\n", symbol);

    if (symbol && strcmp(symbol, "cuLaunchKernel") == 0) {
        fprintf(stderr, "[HOOK v2] Hijacking cuLaunchKernel pointer assignment!\n");
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    }

    if (symbol && strcmp(symbol, "cuGetProcAddress") == 0) {
        fprintf(stderr, "[HOOK v2] Hijacking cuGetProcAddress pointer assignment!\n");
        *pfn = (void*)my_cuGetProcAddress;
        return CUDA_SUCCESS;
    }

    // Call the real cuGetProcAddress if it's anything else
    if (real_cuGetProcAddress_v2 == NULL) {
        if(cu_handle == NULL) {
            cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if(cu_handle == NULL) {
                fprintf(stderr, "[HOOK v2] WARNING: cu_handle == NULL\n");
            }
        }
        real_cuGetProcAddress_v2 = (cuGetProcAddress_t)dlvsym(RTLD_NEXT, "cuGetProcAddress_v2", "RTLD_NEXT"); 
        if(!real_cuGetProcAddress_v2) {
             real_cuGetProcAddress_v2 = (cuGetProcAddress_t)real_dlsym(cu_handle, "cuGetProcAddress_v2");
        }
        if(real_cuGetProcAddress_v2 == NULL) {
            fprintf(stderr, "[HOOK v2] WARNING: real_cuGetProcAddress_v2 == NULL\n");
        }
    }
    return real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}

void* dlsym(void* handle, const char* symbol) {
    fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s.\n", symbol);
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
        real_cuGetProcAddress = (cuGetProcAddress_t)real_dlsym(handle, symbol);
        
        return (void*)my_cuGetProcAddress_v2;
    }

    return real_dlsym(handle, symbol);
}

}
