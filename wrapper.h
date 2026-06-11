typedef struct{
    unsigned char data[256];  
} box256;

typedef __global__ void (*func_ptr_t)();

typedef cudaError_t (*kernel_func_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*paraminfo_func_t)(CUfunction, size_t, size_t*, size_t*);

__global__ void wrapper256(box256 arg, void* func, uint32_t lidx, uint32_t hidx);

void initial_wrapper_run();
void initial_nothing_run();