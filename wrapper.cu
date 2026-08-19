/*
    wrapper.cu

    Defines wrapper.

*/ 
#define MAP_LENGTH 1024

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "wrapper.h"

__device__ AtomMetaData atomMetaDataTable[MAP_LENGTH];

void table_insert(uint64_t key, AtomMetaData value) {
    uint64_t idx = (key * 11400714819323198485ULL) % MAP_LENGTH;
    AtomMetaData metadata;
    
    for(int i = 0; i < MAP_LENGTH; i++) {
        fprintf(stderr, "table_insert, mem_copy\n");
        cudaMemcpyFromSymbol(&metadata, atomMetaDataTable, sizeof(AtomMetaData), sizeof(AtomMetaData) * idx);
        if(metadata.key == 0) {
            fprintf(stderr, "table_insert correct, mem_copy\n");
            cudaMemcpyToSymbol(atomMetaDataTable, &value, sizeof(AtomMetaData), sizeof(AtomMetaData) * idx);
            return;
        }
        fprintf(stderr, "hello? %d\n", i);
        idx = (idx + 1) % MAP_LENGTH;
    }

}

__device__ AtomMetaData table_find(uint64_t key) {
    uint64_t idx = (key * 11400714819323198485ULL) % MAP_LENGTH;
    
    for(int i = 0; i < MAP_LENGTH; i++) {
        if(atomMetaDataTable[idx].key == key) {
            atomMetaDataTable[idx].key = 0;
            return atomMetaDataTable[idx];
        }
        idx = (idx + 1) % MAP_LENGTH;
    }

    AtomMetaData emptydata;
    emptydata.key = 0;
    
    return emptydata;

}

__global__ void wrapper(const __grid_constant__ uint32_t argu) {
    AtomMetaData metadata = table_find((uint64_t)&argu);
    void* kernel = (void*)metadata.kernel;
    uint32_t lidx = metadata.lidx;
    uint32_t hidx = metadata.hidx;

    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < lidx || workIndex >= hidx) return;
    ((func_ptr_t)kernel)();
} 
    
void setup_metadata() {
    AtomMetaData zero_ptrs[MAP_LENGTH];
    for(int i = 0; i < MAP_LENGTH; i++) {
        zero_ptrs[i].key = 0;
    }
    cudaMemcpyToSymbol(atomMetaDataTable, &zero_ptrs, sizeof(zero_ptrs));
}



__global__ void do_nothing() {
    return;
}

/* Runs when callback_mode=0. assigns wrapper256 to wrapper_ptr. */
void initial_wrapper_run() {
    uint32_t fakearg = 0;
    wrapper<<<1, 1>>>(fakearg);
    cudaDeviceSynchronize();
}

/* Runs when callback_mode=1. assigns do_nothing to nothing_ptr_upper/lower. */
void initial_nothing_run() {
    do_nothing<<<1, 1>>>();
    cudaDeviceSynchronize();
}



