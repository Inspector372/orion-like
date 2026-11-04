#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <cstdint>

#define THREAD_NUM 9

enum record_type {
	RECORD_CULAUNCHKERNEL,
	RECORD_CUDAEVENT,
	RECORD_CUDAMEMSET,
	RECORD_CUDAMEMSETASYNC,
	RECORD_CUMEMALLOC,
	RECORD_CUMEMALLOCHOST,
	RECORD_CUMEMHOSTALLOC,
	RECORD_CUMEMFREE,
	RECORD_CUMEMFREEHOST,
	OTHERS
};

typedef struct record_cuLaunchKernel {
	CUfunction f;
	unsigned int gridDimX;
	unsigned int gridDimY;
	unsigned int gridDimZ; 
	unsigned int blockDimX;
	unsigned int blockDimY;
	unsigned int blockDimZ; 
	unsigned int sharedMemBytes;
	CUstream hStream;
	void** kernelParams;
	void** extra;

	uint32_t lidx;
	uint32_t hidx;
} record_cuLaunchKernel;

typedef struct record_cudaEvent {
	cudaEvent_t event;
} record_cudaEvent;

typedef struct record_cudaMemset {
	void* devPtr;
	int value;
	size_t count;
} record_cudaMemset;

typedef struct record_cudaMemsetAsync {
	void* devPtr;
	int value;
	size_t count;
	cudaStream_t stream;
} record_cudaMemsetAsync;

// Owned by the blocked caller; scheduler publishes completion after queue pop.
struct driver_memory_completion {
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
    bool done = false;
    CUresult status = CUDA_SUCCESS;
};

struct record_driver_memory {
    CUcontext context;
    CUdeviceptr* device_out;
    void** host_out;
    size_t bytesize;
    unsigned int flags;
    CUdeviceptr device_ptr;
    void* host_ptr;
    driver_memory_completion* completion;
};

union record_data {
    record_driver_memory r_driver_memory;
	record_cuLaunchKernel r_cuLaunchKernel;
	record_cudaEvent r_cudaEvent;
	record_cudaMemset r_cudaMemset;
	record_cudaMemsetAsync r_cudaMemsetAsync;

	record_data() {}
	~record_data(){};
};

typedef struct queue_record {
	enum record_type type;
	union record_data data;
} queue_record;
