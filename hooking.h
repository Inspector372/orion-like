#pragma once
#include <atomic>
#include <memory>
#include <queue>
#include <vector>

#define THREAD_NUM 9

enum record_type {
	RECORD_CULAUNCHKERNEL,
	RECORD_DEVICE_SYNC,
	RECORD_STREAM_CREATE,
	RECORD_STREAM_DESTROY,
	RECORD_STREAM_SYNC,
	RECORD_EVENT_CREATE,
	RECORD_EVENT_DESTROY,
	RECORD_EVENT_QUERY,
	RECORD_EVENT_RECORD,
	RECORD_EVENT_SYNC,
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

// Borrowed until the scheduler publishes done. Only sync requests wait for GPU completion.
struct driver_request {
    CUstream* stream_out = NULL;
    CUstream stream = NULL;
    CUevent* event_out = NULL;
    CUevent event = NULL;
    CUresult (CUDAAPI *destroy_event)(CUevent) = NULL;
    CUresult (CUDAAPI *destroy_stream)(CUstream) = NULL;
    unsigned int flags = 0;
    CUresult status = CUDA_SUCCESS;
    std::atomic<bool> done{false};
};

typedef struct record_cudaMemset {
	void* devPtr;
	int value;
	size_t count;
	cudaError_t* status;
} record_cudaMemset;

typedef struct record_cudaMemsetAsync {
	void* devPtr;
	int value;
	size_t count;
	cudaStream_t stream;
	cudaError_t* status;
} record_cudaMemsetAsync;

typedef struct record_cuMemAlloc {
	CUdeviceptr* dptr;
	size_t bytesize;
	CUresult* status; // Valid until the scheduler retires this entry.
} record_cuMemAlloc;

typedef struct record_cuMemAllocHost {
	void** pp;
	size_t bytesize;
	CUresult* status; // Valid until the scheduler retires this entry.
} record_cuMemAllocHost;

typedef struct record_cuMemHostAlloc {
	void** pp;
	size_t bytesize;
	unsigned int flags;
	CUresult* status; // Valid until the scheduler retires this entry.
} record_cuMemHostAlloc;

typedef struct record_cuMemFree {
	CUdeviceptr dptr;
	CUresult* status; // Valid until the scheduler retires this entry.
} record_cuMemFree;

typedef struct record_cuMemFreeHost {
	void* p;
	CUresult* status; // Valid until the scheduler retires this entry.
} record_cuMemFreeHost;

union record_data {
	record_cuMemAlloc r_cuMemAlloc;
	record_cuMemAllocHost r_cuMemAllocHost;
	record_cuMemHostAlloc r_cuMemHostAlloc;
	record_cuMemFree r_cuMemFree;
	record_cuMemFreeHost r_cuMemFreeHost;
	record_cuLaunchKernel r_cuLaunchKernel;
	driver_request* request;
	record_cudaMemset r_cudaMemset;
	record_cudaMemsetAsync r_cudaMemsetAsync;

	record_data() {}
	~record_data(){};
};

typedef struct queue_record {
	enum record_type type;
	union record_data data;
} queue_record;

// Registry membership, closing, issued and records are protected by the owner's
// work_queue_mutex. A producer may still hold a ticket after removal from the vector.
struct stream_queue {
    stream_queue(int client, CUstream stream) : owner(client), gpu(stream) {}
    int owner;
    CUstream gpu;
    bool closing = false;
    std::queue<queue_record> records;
    uint64_t issued = 0;
    std::atomic<uint64_t> retired{0};
};
typedef std::shared_ptr<stream_queue> stream_queue_ptr;
typedef std::vector<stream_queue_ptr> client_streams;

// The harness virtualizes default-stream handles on a per-client GPU stream.
// A stream created outside the harness retains its native handle when dispatched.
inline CUstream dispatch_stream(CUstream requested, CUstream fallback) {
    if (!requested || requested == CU_STREAM_LEGACY || requested == CU_STREAM_PER_THREAD)
        return fallback;
    return requested;
}
