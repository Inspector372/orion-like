/*
    defines wrapper and idle kernels.
    currently there is only wrapper256(), but size of box need to vary(to reduce overhead...?),
    so multiple box size need to be supported later.
    current kernel will raise error if total size of parameter is bigger than 256 bytes.
*/ 

#include "wrapper.h"

__global__ void wrapper256(box256 arg, void* func, size_t lidx, size_t hidx) {
    // Index filtering.
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < lidx || workIndex >= hidx) return;

    // theory: "move" the context to actual kernel.
    (func_ptr_t(func))();
}

__global__ void do_nothing() {
    return;
}

/* Runs when callback_mode=0. assigns wrapper256 to wrapper_ptr. */
void initial_wrapper_run() {
    box256 fakearg;
    fakearg.data[0] = 0;
    wrapper256<<<1, 1>>>(fakearg, nullptr, 0, 0);
    cudaDeviceSynchronize();
}

/* Runs when callback_mode=1. assigns do_nothing to nothing_ptr_upper/lower. */
void initial_nothing_run() {
    do_nothing<<<1, 1>>>();
    cudaDeviceSynchronize();
}

/*
    invoke paraminfo_func to target_kernel, to get information about **args.
    read arguments, put in wrapper box in arranged manner.
    then invoke kernel_func, with wrapper as its func arguments,
    and further arguments filled with following arguments.

    TODO: support multiple box size.
*/
void run_wrapper(void* kernel_func, void* paraminfo_func, void* target_kernel, void* target_kernel_program_addr, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream, size_t lidx, size_t hidx) {
    box256 argbox;
    size_t func_param_count = 0;
    size_t func_param_offset;
	size_t func_param_size;
	cudaError_t param_err;
    
    while ((param_err = (*((paraminfo_func_t*)paraminfo_func))((CUfunction)target_kernel, func_param_count, &func_param_offset, &func_param_size)) == cudaSuccess) {
        memcpy(&(argbox[func_param_offset]), args[func_param_count], func_param_size);
        func_param_count++;
    }

    void* func = target_kernel_program_addr;
    size_t lidx_arg = lidx;
    size_t hidx_arg = hidx;

    void* kernel_args[] = {
        &argbox,
        &func,
        &lidx_arg,
        &hidx_arg
    }

    (*((kernel_func_t*)kernel_func))((const void*)wrapper256, gridDim, blockDim, kernel_args, sharedMem, stream);
}