#pragma once
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern cudaStream_t metadata_pass_stream;
extern cudaError_t (*actual_cudaDeviceSynchronize)(void);
void setup_metadata();
void initial_wrapper_run();
int acquire_atom_slot();
void prepare_atom_slot(int slot, uint64_t kernel, uint32_t lidx, uint32_t hidx, CUstream stream);
void finish_atom_slot(int slot, CUstream stream);
void shutdown_metadata();
void check_atom_driver(CUresult status, const char* operation);
