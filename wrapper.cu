#include <stdio.h>
#include <stdlib.h>
#include "wrapper.h"
#include "libsmctrl.h"

struct AtomMetaData {
    uint64_t kernel;
    uint32_t lidx, hidx;
};

static __device__ AtomMetaData device_metadata[ATOM_SLOT_COUNT];
static CUdeviceptr metadata_address;
static AtomMetaData* staging;
static struct {
    CUevent ready, done;
    bool busy;
} slots[ATOM_SLOT_COUNT];

// The entry address selects the slot, leaving the client's parameter buffer intact.
template<int Slot>
__global__ void atom_wrapper() {
    const AtomMetaData meta = device_metadata[Slot];
    size_t block = ((size_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    if (block >= meta.lidx && block < meta.hidx)
        ((void (*)())meta.kernel)();
}

static __global__ void probe_stub() {}

void check_atom_driver(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        fprintf(stderr, "%s failed (%d)\n", operation, (int)status);
        exit(EXIT_FAILURE);
    }
}

static void check_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

void setup_metadata() {
    void* address;
    check_runtime(cudaGetSymbolAddress(&address, device_metadata), "get metadata address");
    metadata_address = (CUdeviceptr)address;
    check_runtime(cudaMallocHost((void**)&staging, sizeof(device_metadata)), "allocate metadata staging");
    for (int i = 0; i < ATOM_SLOT_COUNT; ++i) {
        check_atom_driver(cuEventCreate(&slots[i].ready, CU_EVENT_DISABLE_TIMING), "create ready event");
        check_atom_driver(cuEventCreate(&slots[i].done, CU_EVENT_DISABLE_TIMING), "create done event");
    }
}

template<int Count>
static void capture_wrappers() {
    capture_wrappers<Count - 1>();
    atom_capture_entry(Count - 1);
    atom_wrapper<Count - 1><<<1, 1>>>();
    check_runtime(cudaGetLastError(), "capture wrapper launch");
    atom_finish_entry();
}

template<> void capture_wrappers<0>() {}

void initial_wrapper_run() {
    // Device globals start zeroed, so each bootstrap wrapper has an empty range.
    capture_wrappers<ATOM_SLOT_COUNT>();
    atom_capture_entry(ATOM_PROBE_ENTRY);
    probe_stub<<<1, 1>>>();
    check_runtime(cudaGetLastError(), "capture probe launch");
    atom_finish_entry();
    check_runtime(actual_cudaDeviceSynchronize(), "initialize atom entries");
}

int acquire_atom_slot() {
    static unsigned next;
    for (unsigned i = 0; i < ATOM_SLOT_COUNT; ++i) {
        unsigned slot = (next + i) % ATOM_SLOT_COUNT;
        if (slots[slot].busy) {
            CUresult status = cuEventQuery(slots[slot].done);
            if (status == CUDA_ERROR_NOT_READY) continue;
            check_atom_driver(status, "query atom completion");
        }
        slots[slot].busy = true;
        next = (slot + 1) % ATOM_SLOT_COUNT;
        return slot;
    }
    return -1;
}

void prepare_atom_slot(int slot, uint64_t kernel, uint32_t lidx, uint32_t hidx, CUstream stream) {
    staging[slot] = AtomMetaData{kernel, lidx, hidx};
    check_atom_driver(cuMemcpyHtoDAsync(metadata_address + slot * sizeof(AtomMetaData),
        &staging[slot], sizeof(AtomMetaData), metadata_pass_stream), "upload atom metadata");
    check_atom_driver(cuEventRecord(slots[slot].ready, metadata_pass_stream), "record metadata ready");
    check_atom_driver(cuStreamWaitEvent(stream, slots[slot].ready, 0), "wait for metadata");
}

void finish_atom_slot(int slot, CUstream stream) {
    check_atom_driver(cuEventRecord(slots[slot].done, stream), "record atom completion");
}

void shutdown_metadata() {
    for (int i = 0; i < ATOM_SLOT_COUNT; ++i) {
        if (slots[i].busy)
            check_atom_driver(cuEventSynchronize(slots[i].done), "drain atom");
        check_atom_driver(cuEventDestroy(slots[i].ready), "destroy ready event");
        check_atom_driver(cuEventDestroy(slots[i].done), "destroy done event");
    }
    check_runtime(cudaFreeHost(staging), "free metadata staging");
    check_runtime(cudaStreamDestroy(metadata_pass_stream), "destroy metadata stream");
}
