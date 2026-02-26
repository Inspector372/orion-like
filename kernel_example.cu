
// Example kernels.

#include <cuda_runtime.h>
#include <cuda.h>
#include "kernel_example.h"
#include <stdio.h>

__global__ void addKernel(int* a, int* b, int* out, int N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
        out[idx] = a[idx] + b[idx];
}

void addCheck(int* h_A, int* h_B, int* h_out, int N) {
    for(int i = 0; i < N; i++) {
        if(h_A[i] + h_B[i] != h_out[i]) {
            fprintf(stderr, "mismatch at %d\n", i);
            return;
        }
    }
}

extern "C" void* addKernel_wrap(void* arg) {
    int N;
    int* h_A, *h_B, *h_out;
    int* d_A, *d_B, *d_out;
    N = ((addKernel_arg*)arg)->N;
    h_A = ((addKernel_arg*)arg)->h_A;
    h_B = ((addKernel_arg*)arg)->h_B;
    h_out = ((addKernel_arg*)arg)->h_out;
    pthread_mutex_t* smutex = ((addKernel_arg*)arg)->smutex;
    
    // block before setting everything.
    pthread_mutex_lock(smutex);
    pthread_mutex_unlock(smutex);

    
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    addKernel<<<blocks, threads>>>(d_A, d_B, d_out, N);

    // can this be a problem? this is not a wrapped method.
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    addCheck(h_A, h_B, h_out, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    return nullptr;
}