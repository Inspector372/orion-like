#define NO_WRAPPER 0

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

typedef struct record_cuMemAlloc {
	CUdeviceptr* dptr;
	size_t bytesize;
	CUresult* status; // Caller stays in block() until the scheduler pops this record.
} record_cuMemAlloc;

typedef struct record_cuMemAllocHost {
	void** pp;
	size_t bytesize;
	CUresult* status; // Caller stays in block() until the scheduler pops this record.
} record_cuMemAllocHost;

typedef struct record_cuMemHostAlloc {
	void** pp;
	size_t bytesize;
	unsigned int flags;
	CUresult* status; // Caller stays in block() until the scheduler pops this record.
} record_cuMemHostAlloc;

typedef struct record_cuMemFree {
	CUdeviceptr dptr;
	CUresult* status; // Caller stays in block() until the scheduler pops this record.
} record_cuMemFree;

typedef struct record_cuMemFreeHost {
	void* p;
	CUresult* status; // Caller stays in block() until the scheduler pops this record.
} record_cuMemFreeHost;

union record_data {
	record_cuMemAlloc r_cuMemAlloc;
	record_cuMemAllocHost r_cuMemAllocHost;
	record_cuMemHostAlloc r_cuMemHostAlloc;
	record_cuMemFree r_cuMemFree;
	record_cuMemFreeHost r_cuMemFreeHost;
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
