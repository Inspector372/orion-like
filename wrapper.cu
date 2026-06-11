/*
    defines wrapper and idle kernels.
    currently there is only wrapper256(), but size of box need to vary(to reduce overhead...?),
    so multiple box size need to be supported later.
    current kernel will raise error if total size of parameter is bigger than 256 bytes.
*/ 
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "wrapper.h"

__global__ void wrapper256(box256 arg, void* func, uint32_t lidx, uint32_t hidx) {
    // Index filtering.
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex == 3) {
        printf("sanity check, func = %p, lidx = %d, hidx = %d\n", func, lidx, hidx);
    }
    if (workIndex < lidx || workIndex >= hidx) return;

    // theory: "move" the context to actual kernel.
    ((func_ptr_t)func)();
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

