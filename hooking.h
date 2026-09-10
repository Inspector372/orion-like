
#define THREAD_NUM 4

enum record_type {
	RECORD_CULAUNCHKERNEL,
	RECORD_CUDAEVENT,
	RECORD_CUDAMEMSET,
	RECORD_CUDAMEMSETASYNC,
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

union record_data {
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
