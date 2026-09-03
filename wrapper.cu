/*
    wrapper.cu

    Defines wrapper.

*/ 
#define MAP_LENGTH 1024
#define MAGIC 0x2020064020200640ULL

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "wrapper.h"

__device__ AtomMetaData atomMetaDataTable[MAP_LENGTH];

void table_insert(uint64_t key, AtomMetaData value) {
    uint64_t idx = (key * 11400714819323198485ULL) % MAP_LENGTH;
    AtomMetaData metadata;

    cudaEvent_t copy_to, copy_from;
    // TODO: when debugging/profiling is done, change it with cudaEventDisableTiming for better performance.
    cudaEventCreateWithFlags(&copy_to, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&copy_from, cudaEventBlockingSync);

    fprintf(stderr, "trying to insert key: %lx, value: %lx, %lx, %d, %d inside table...\n", key, value.key, value.kernel, value.lidx, value.hidx);
    
    for(int i = 0; i < MAP_LENGTH; i++) {
        fprintf(stderr, "table_insert starting, mem_copy to host 1\n");
        cudaMemcpyFromSymbolAsync(&metadata, atomMetaDataTable, sizeof(AtomMetaData), sizeof(AtomMetaData) * idx, cudaMemcpyDeviceToHost, metadata_pass_stream);
        fprintf(stderr, "table_insert starting, mem_copy to host 2\n");
        cudaEventRecord(copy_to, metadata_pass_stream);

        fprintf(stderr, "Synchronizing to copy_to...\n");
        cudaEventSynchronize(copy_to);
        if(metadata.key == 0) {
            cudaMemcpyToSymbolAsync(atomMetaDataTable, &value, sizeof(AtomMetaData), sizeof(AtomMetaData) * idx, cudaMemcpyHostToDevice, metadata_pass_stream);
            cudaEventRecord(copy_from, metadata_pass_stream);
            fprintf(stderr, "Synchronizing to copy_from...\n");
            cudaEventSynchronize(copy_from);


            /*
            cudaMemcpyFromSymbolAsync(&test, atomMetaDataTable, sizeof(AtomMetaData), sizeof(AtomMetaData) * idx, cudaMemcpyDeviceToHost, metadata_pass_stream);
            cudaEventRecord(copy_to, metadata_pass_stream);
            cudaEventSynchronize(copy_to);
            err = cudaGetLastError();
            fprintf(stderr, "testing, test.key = %lx, test.kernel = %lx, test.lidx = %d, test.hidx = %d\n", test.key, test.kernel, test.lidx, test.hidx);
            */

            cudaEventDestroy(copy_to);
            cudaEventDestroy(copy_from);
            return;
        }
        idx = (idx + 1) % MAP_LENGTH;
    }

}

/*
    Kernel wrapper.
    
    

*/
__global__ void wrapper(const __grid_constant__ uint64_t argu) {
    if(argu == MAGIC) return; // For initial_wrapper_run().

    uint64_t ptr = (uint64_t)&argu;
    uint64_t idx = (ptr * 11400714819323198485ULL) % MAP_LENGTH;

    int i;
    for(i = 0; i < MAP_LENGTH; i++) {
        if(atomMetaDataTable[idx].key == ptr) {
            atomMetaDataTable[idx].key = 0;
            break;
        }
        idx = (idx + 1) % MAP_LENGTH;
    }

    void* kernel = (void*)atomMetaDataTable[idx].kernel;
    uint32_t lidx = atomMetaDataTable[idx].lidx;
    uint32_t hidx = atomMetaDataTable[idx].hidx;

    size_t workIndex = threadIdx.x + blockDim.x * blockIdx.x;
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


/*
    Runs only once when callback_mode = 0.
*/
void initial_wrapper_run() {
    uint64_t fakearg = MAGIC;
    wrapper<<<1, 1>>>(fakearg);
    (*actual_cudaDeviceSynchronize)();
}


