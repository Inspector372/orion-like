// Same thing from orion.
enum func_type {
	KERNEL_RECORD,
	MEMCPY_RECORD,
	MALLOC_RECORD,
	FREE_RECORD,
	MEMSET_RECORD
};

typedef struct kernel_record
{

	const void *func;
	dim3 gridDim;
	dim3 blockDim;
	void **args;
	size_t sharedMem;
	cudaStream_t stream;
	volatile bool run;
	volatile cudaStream_t sched_stream;
} kernel_record;

typedef struct memcpy_record
{

	void *dst;
	const void *src;
	size_t count;
	enum cudaMemcpyKind kind;
	cudaStream_t stream;
	bool async;
} memcpy_record;

typedef struct malloc_record
{

	void **devPtr;
	size_t size;

} malloc_record;

typedef struct free_record
{
	void *devPtr;

} free_record;

typedef struct memset_record
{

	void *devPtr;
	int value;
	size_t count;
	cudaStream_t stream;
	bool async;

} memset_record;

union func_data {
	kernel_record krecord;
	memcpy_record mrecord;
	malloc_record malrecord;
	free_record frecord;
	memset_record msetrecord;

	func_data() {}
	~func_data(){};
};

// Same thing from orion.
typedef struct func_record {
	enum func_type type;
	union func_data data;
} func_record;