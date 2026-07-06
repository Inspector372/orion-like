
enum record_type {
	RECORD_CULAUNCHKERNEL,
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

	size_t kptr_index;
	size_t lidx;
	size_t hidx;
} record_cuLaunchKernel;

union record_data {
	record_cuLaunchKernel r_cuLaunchKernel;

	record_data() {}
	~record_data(){};
};

typedef struct queue_record {
	enum record_type type;
	union record_data data;
} queue_record;
