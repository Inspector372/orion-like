typedef struct{
    unsigned char data[256];  
} box256;

typedef void (*func_ptr_t)();

typedef cudaError_t (*kernel_func_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
typedef cudaError_t (*paraminfo_func_t)(CUfunction, size_t, size_t*, size_t*);

extern "C" void run_wrapper(void* kernel_func, void* paraminfo_func, void* target_kernel, void* target_kernel_program_addr, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream, size_t lidx, size_t hidx);