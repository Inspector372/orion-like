/*
    wrapper.cu

    Defines wrapper.

*/ 
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "wrapper.h"


__global__ void wrapper(const __grid_constant__ uint32_t argu) {
    AtomMetaData* metadata = atomMetaDataTable.find(&argu);
    void* kernel = metadata->kernel;
    uint32_t lidx = metadata->lidx;
    uint32_t hidx = metadata->hidx;

    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < lidx || workIndex >= hidx) return;

    ((func_ptr_t)kernel)();
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

