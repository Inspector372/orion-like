typedef struct{
    unsigned char data[256];  
} box256;

typedef void (*func_ptr_t)();

__global__ void wrapper256(box256 arg, void* func, uint32_t lidx, uint32_t hidx);

extern CUfunction wrapper256_handle;

void initial_wrapper_run();
void initial_nothing_run();