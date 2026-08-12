typedef struct{
    void* kernel; 
    uint32_t lidx;
    uint32_t hidx; 
} AtomMetaData;

typedef void (*func_ptr_t)();

__global__ void wrapper(const __grid_constant__ uint32_t argu);

extern CUfunction wrapper256_handle;

void initial_wrapper_run();
void initial_nothing_run();