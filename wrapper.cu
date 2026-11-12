/*
    wrapper.cu

    Defines wrapper.

*/ 
#define MAP_LENGTH 2048
#define MAGIC 0x2020064020200640ULL

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "wrapper.h"

__device__ AtomMetaData atomMetaDataTable[MAP_LENGTH];
void* atomMetaDataTable_symbol_addr;
AtomMetaData* atomMetaDataTable_cache;

CUfunction metadata_table_insert_handle;
CUfunction metadata_table_check_handle;


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

*/
void table_insert(uint64_t key, AtomMetaData value) {
    uint64_t idx = (key * 11400714819323198485ULL) % MAP_LENGTH;
    AtomMetaData metadata;

    fprintf(stderr, "trying to insert key: %lx, value: %lx, %lx, %d, %d inside table...\n", key, value.key, value.kernel, value.lidx, value.hidx);
    
    // check if the entry is already there!
    for(int i = 0; i < MAP_LENGTH; i++) {
        if(atomMetaDataTable_cache[i].key == key) {
            // If %gridid is unique, we should not fall here.
            fprintf(stderr, "Problem: gridid is not unique or tried to insert incorrect value: %d\n", key);
            std::exit(-1);
            return;
        }
    }

    for(int i = 0; i < MAP_LENGTH; i++) {
        if(atomMetaDataTable_cache[idx].key == 0) {
            atomMetaDataTable_cache[idx] = value;
            metadata_table_insert<<<1, 1, 0, metadata_pass_stream>>>(idx, value);
            metadata_table_check<<<1, 1, 0, metadata_pass_stream>>>(idx);
            /*
            while(cuMemcpyHtoDAsync((CUdeviceptr)atomMetaDataTable_symbol_addr + idx * sizeof(AtomMetaData), &atomMetaDataTable_cache[idx], sizeof(AtomMetaData), (CUstream)metadata_pass_stream) != CUDA_SUCCESS) {
                ;
            }
            atomMetaDataTable_cache[idx].concurrent_key = atomMetaDataTable_cache[idx].key;
            while(cuMemcpyHtoDAsync((CUdeviceptr)atomMetaDataTable_symbol_addr + idx * sizeof(AtomMetaData) + sizeof(uint64_t), &atomMetaDataTable_cache[idx], sizeof(uint64_t), (CUstream)metadata_pass_stream) != CUDA_SUCCESS) {
                ;
            }
            */
            break;
        }
        idx = (idx + 1) % MAP_LENGTH;
    }
    return;

}

void table_delete(uint64_t key) {
    uint64_t idx = (key * 11400714819323198485ULL) % MAP_LENGTH;
    AtomMetaData metadata;
    metadata.key = 0;

    CUevent copy;
    cuEventCreate(&copy, CU_EVENT_DISABLE_TIMING);
    // cudaEvent_t copy;
    // cudaEventCreateWithFlags(&copy, cudaEventBlockingSync | cudaEventDisableTiming);
    
    for(int i = 0; i < MAP_LENGTH; i++) {
        if(atomMetaDataTable_cache[idx].key == key) {
            atomMetaDataTable_cache[idx].key = 0;
            // TODO: this must be synchronous.
            metadata_table_insert<<<1, 1, 0, metadata_pass_stream>>>(idx, atomMetaDataTable_cache[idx]);
            /*
            while(cuMemcpyHtoDAsync((CUdeviceptr)atomMetaDataTable_symbol_addr + idx * sizeof(AtomMetaData), &atomMetaDataTable_cache[idx], sizeof(AtomMetaData), metadata_pass_stream) != CUDA_SUCCESS) {
                ;
            }
            */
            // cuEventSynchronize(copy);
            // cudaEventRecord(copy, metadata_pass_stream);
            // cudaEventSynchronize(copy);
            break;
        }
        idx = (idx + 1) % MAP_LENGTH;
    }
    cuEventDestroy(copy);
    // cudaEventDestroy(copy);
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
    fprintf(stderr, "release queue is full\n");
    std::exit(-1);
}

void release_queue_mark_as_reaped(uint64_t key) {
    for(int i = 0; i < RELEASE_QUEUE_LENGTH; i++) {
        if(release_queue[i].key == key && !release_queue[i].reaped) {
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

    // uint64_t ptr = (uint64_t)&argu;
    // uint64_t idx = (ptr * 11400714819323198485ULL) % MAP_LENGTH;
    uint64_t gid;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(gid));
    uint64_t idx = (gid * 11400714819323198485ULL) % MAP_LENGTH;
    

    // Spin until finding the concurrent key.
    for(;;) {
        if(atomMetaDataTable[idx].concurrent_key == gid) {
            break;
        }
        idx = (idx + 1) % MAP_LENGTH;
    }

    void* kernel = (void*)atomMetaDataTable[idx].kernel;
    uint32_t lidx = atomMetaDataTable[idx].lidx;
    uint32_t hidx = atomMetaDataTable[idx].hidx;

    size_t block_idx = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    // if(threadIdx.x + blockIdx.x * blockDim.x == 1) printf("key : %lx, lidx : %d, hidx : %d\n", gid, lidx, hidx);
    if (block_idx < lidx || block_idx >= hidx) return;

    ((func_ptr_t)kernel)();
} 

__global__ void metadata_table_insert(uint32_t index, AtomMetaData metadata) {
    if(index >= MAP_LENGTH) return;
    atomMetaDataTable[index] = metadata;
}

__global__ void metadata_table_check(uint32_t index) {
    if(index >= MAP_LENGTH) return;
    atomMetaDataTable[index].concurrent_key = atomMetaDataTable[index].key;
}
    
void setup_metadata() {
    AtomMetaData zero_ptrs[MAP_LENGTH];
    for(int i = 0; i < MAP_LENGTH; i++) {
        zero_ptrs[i].key = 0;
    }
    cudaMemcpyToSymbol(atomMetaDataTable, &zero_ptrs, sizeof(zero_ptrs));
    cudaMallocHost((void**)&atomMetaDataTable_cache, sizeof(AtomMetaData) * MAP_LENGTH);
    for(int i = 0; i < MAP_LENGTH; i++) {
        atomMetaDataTable_cache[i] = AtomMetaData{};
    }
    cudaGetSymbolAddress(&atomMetaDataTable_symbol_addr, atomMetaDataTable);
}


/*
    Runs only once when callback_mode = 0.
*/
void initial_wrapper_run() {
    uint64_t fakearg = MAGIC;
    wrapper<<<1, 1>>>(fakearg);

    (*actual_cudaDeviceSynchronize)();
}

/*
    Extracts PROGRAM_ADDRESS for Metadata table.
*/
void initial_kernels_run() {
    AtomMetaData blank;
    cudaFunction_t func;
    metadata_table_insert<<<1, 1>>>(MAP_LENGTH, blank);
    cudaGetFuncBySymbol(&func, (const void*)metadata_table_insert);
    metadata_table_insert_handle = (CUfunction)func;
    metadata_table_check<<<1, 1>>>(MAP_LENGTH);
    cudaGetFuncBySymbol(&func, (const void*)metadata_table_check);
    metadata_table_check_handle = (CUfunction)func;
    (*actual_cudaDeviceSynchronize)();
}


