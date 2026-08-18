/*
    wrapper.cu

    Defines wrapper.

*/ 
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "wrapper.h"

using StaticMapRefType = decltype(std::declval<cuco::static_map<uint64_t, AtomMetaData>>().ref(cuco::op::insert, cuco::op::find));
__device__ StaticMapRefType* atomMetaDataTable_ref = nullptr;


__global__ void wrapper(const __grid_constant__ uint32_t argu) {
    auto iter = atomMetaDataTable_ref->find(&argu);
    if(iter != atomMetaDataTable_ref->end()) {
        AtomMetaData metadata = iter->second;
        void* kernel = metadata->kernel;
        uint32_t lidx = metadata->lidx;
        uint32_t hidx = metadata->hidx;

        int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
        if (workIndex < lidx || workIndex >= hidx) return;
        ((func_ptr_t)kernel)();
    }
    
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

/* Runs when atomMetaDataTable is setup in threading.cpp. Sets up atomMetaDataTable_ref. */
void setup_metadata_ref() {
    auto host_ref = atomMetaDataTable->ref(cuco::op::insert, cuco::op::find);
    StaticMapRefType* d_ref_ptr;
    cudaMalloc(&d_ref_ptr, sizeof(StaticMapRefType));
    cudaMemcpy(d_ref_ptr, &host_ref, sizeof(StaticMapRefType), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(atomMetaDataTable_ref, &d_ref_ptr, sizeof(StaticMapRefType*));
    cudaFree(d_ref_ptr);
}
