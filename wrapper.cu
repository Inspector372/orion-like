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
AtomMetaData atomMetaDataTable_cache[MAP_LENGTH];

/*
    When a real_cuLaunchKernel launch happens, OS creates an event and saves (key, event) pair.
    OS will try to 'reap' it by calling cudaEventQuery(), and if its cudaSuccess it will call table_delete(through hash_delete(key)).
    
    One problem is that OS may try to insert the key that is already used when real_cuLaunchKernel happens.
    When that happens, we need to do this:
    1. change (key, old_metadata) -> (key, new_metadata) of atomMetaDataTable and its cache.
    2. insert (key, new_event) of release_queue.
    3. mark (key, old_event) as REAPED.

    It's safe to just change key of old metadata, because GPU actually does not use that key address anymore.
    However, release_queue entry MUST be newly inserted because cudaEvent will be eventually signaled and OS will try to reap it.

*/
ReleaseQueueEntry release_queue[RELEASE_QUEUE_LENGTH];

/*
    insert and delete entry from atomMetaDataTable.
    hashing mechanism is just random now, if there is better way, it can be changed later.
    
    because of the internal synchronizing mechanism of cudaLaunchKernel(),
    those functions should not be used at the same time with cudaLaunchKernel(),
    otherwise it will cause a deadlock.

    Table should be ONLY modified using those APIs, otherwise it will break host cacheing mechanism.
    both insert() and delete() check atomMetaDataTable_host at host side to decide which block to remove,
    and atomMetaDataTable_cache MUST be updated too.

    when inserting, if there is already such key in table,

*/
void table_insert(uint64_t key, AtomMetaData value) {
    uint64_t idx = (key * 11400714819323198485ULL) % MAP_LENGTH;
    AtomMetaData metadata;

    cudaEvent_t copy;
    cudaEventCreateWithFlags(&copy, cudaEventBlockingSync | cudaEventDisableTiming);

    // fprintf(stderr, "trying to insert key: %lx, value: %lx, %lx, %d, %d inside table...\n", key, value.key, value.kernel, value.lidx, value.hidx);
    
    // check if the entry is already there!
    for(int i = 0; i < MAP_LENGTH; i++) {
        if(atomMetaDataTable_cache[i].key == key) {
            atomMetaDataTable_cache[i] = value;
            cudaMemcpyToSymbolAsync(atomMetaDataTable, &value, sizeof(AtomMetaData), sizeof(AtomMetaData) * i, cudaMemcpyHostToDevice, metadata_pass_stream);
            cudaEventRecord(copy, metadata_pass_stream);
            cudaEventSynchronize(copy);
            release_queue_mark_as_reaped(key);
            cudaEventDestroy(copy);
            return;
        }
    }

    for(int i = 0; i < MAP_LENGTH; i++) {
        if(atomMetaDataTable_cache[idx].key == 0) {
            atomMetaDataTable_cache[idx] = value;
            cudaMemcpyToSymbolAsync(atomMetaDataTable, &value, sizeof(AtomMetaData), sizeof(AtomMetaData) * idx, cudaMemcpyHostToDevice, metadata_pass_stream);
            cudaEventRecord(copy, metadata_pass_stream);
            cudaEventSynchronize(copy);
        }
        idx = (idx + 1) % MAP_LENGTH;
    }
    cudaEventDestroy(copy);
    return;

}

void table_delete(uint64_t key) {
    uint64_t idx = (key * 11400714819323198485ULL) % MAP_LENGTH;
    AtomMetaData metadata;
    metadata.key = 0;

    cudaEvent_t copy;
    cudaEventCreateWithFlags(&copy, cudaEventBlockingSync | cudaEventDisableTiming);
    
    for(int i = 0; i < MAP_LENGTH; i++) {
        if(atomMetaDataTable_cache[idx].key == key) {
            atomMetaDataTable_cache[idx].key = 0;
            cudaMemcpyToSymbolAsync(atomMetaDataTable, &metadata, sizeof(AtomMetaData), sizeof(AtomMetaData) * idx, cudaMemcpyHostToDevice, metadata_pass_stream);
            cudaEventRecord(copy, metadata_pass_stream);
            cudaEventSynchronize(copy);
            break;
        }
        idx = (idx + 1) % MAP_LENGTH;
    }
    cudaEventDestroy(copy);
    return;
}

/*
    Helper functions for release_queue.
*/
void release_queue_insert(uint64_t key, cudaEvent_t event) {
    for(int i = 0; i < RELEASE_QUEUE_LENGTH; i++) {
        if(release_queue[i].key == 0) {
            release_queue[i].key = key;
            release_queue[i].event = event;
            release_queue[i].reaped = false;
            return;
        }
    }
}

void release_queue_mark_as_reaped(uint64_t key) {
    for(int i = 0; i < RELEASE_QUEUE_LENGTH; i++) {
        if(release_queue[i].key == key) {
            release_queue[i].reaped = true;
            return;
        }
    }
}

void release_queue_delete(uint64_t key) {
    for(int i = 0; i < RELEASE_QUEUE_LENGTH; i++) {
        if(release_queue[i].key == key) {
            release_queue[i].key = 0;
            release_queue[i].event = 0;
            release_queue[i].reaped = false;
            return;
        }
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
            break;
        }
        idx = (idx + 1) % MAP_LENGTH;
    }

    void* kernel = (void*)atomMetaDataTable[idx].kernel;
    uint32_t lidx = atomMetaDataTable[idx].lidx;
    uint32_t hidx = atomMetaDataTable[idx].hidx;

    size_t block_idx = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    if (block_idx < lidx || block_idx >= hidx) return;

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


