
enum record_type {
	RECORD_CULAUNCHKERNEL,
	RECORD_CUDAEVENT,
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

union record_data {
	record_cuLaunchKernel r_cuLaunchKernel;
	record_cudaEvent r_cudaEvent;

	record_data() {}
	~record_data(){};
};

typedef struct queue_record {
	enum record_type type;
	union record_data data;
} queue_record;
