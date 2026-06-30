/*
    env CUDA_VISIBLE_DEVICES=5 LD_PRELOAD=./hooklaunch.so ./kernel

    attempt to hook the whole library
    https://github.com/open-neutrino/neutrino/blob/main/neutrino/src/preload.c
    https://github.com/Project-HAMi/HAMi-core/blob/main/src/libvgpu.c

*/

#include <iostream>
#include <cuda_runtime.h>

#define N 10000  // number of elements
#define BLOCK_SIZE 256

// CUDA kernel: each thread computes one element of C
__global__ void addArrays(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    size_t size = N * sizeof(float);

    // Host arrays
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device arrays
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    addArrays<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (std::abs(h_C[i] - expected) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": got " << h_C[i]
                      << ", expected " << expected << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Success! All " << N << " elements match." << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}