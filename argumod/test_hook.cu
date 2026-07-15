
/*
    test_hook.cu:
    extra parameter을 이용한 cuLaunchKernel call / 
    wrapper로 addiational argument 접근 시도


*/

#include <iostream>
#include <stdio.h>
#include <cstring>
#include "libsmctrl.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>

typedef struct{
    unsigned char data[128];  
} bigbox;


__global__ void mul(double* res, double* op1, double* op2, uint64_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        res[workIndex] = op1[workIndex] * op2[workIndex];
    }
    return;
}

__global__ void wrapper(bigbox args) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex == 0) {
        printf("args magic: %lx\n", *((uint64_t*)&(args.data[24])));
    }
    return;
}

typedef void (*func_ptr_t)();

int main() {
    libsmctrl_false_launch_callback();
    callback_mode = 0;
    bigbox box;
    wrapper<<<1, 1>>>(box);
    cudaDeviceSynchronize();

    callback_mode = 3;

    // 1. Define the size of the arrays
    const size_t length = 10000;
    const size_t sizeInBytes = length * sizeof(double);

    // 2. Allocate Host Memory
    double* h_op1 = (double*)malloc(sizeInBytes);
    double* h_op2 = (double*)malloc(sizeInBytes);
    double* h_res = (double*)malloc(sizeInBytes);

    // 3. Initialize Host Data
    for (size_t i = 0; i < length; ++i) {
        h_op1[i] = static_cast<double>(i);
        h_op2[i] = 2.5; // Expected result for each element 'i' is i * 2.5
    }

    // 4. Allocate Device Memory
    double *d_op1 = nullptr, *d_op2 = nullptr, *d_res = nullptr;
    cudaMalloc(&d_op1, sizeInBytes);
    cudaMalloc(&d_op2, sizeInBytes);
    cudaMalloc(&d_res, sizeInBytes);

    // 5. Copy Data from Host to Device
    cudaMemcpy(d_op1, h_op1, sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_op2, h_op2, sizeInBytes, cudaMemcpyHostToDevice);

    // 6. Define Execution Configuration
    int threadsPerBlock = 256;
    // Calculate grid size, rounding up to ensure all elements are covered
    int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

    // 7. Launch Kernel
    mul<<<blocksPerGrid, threadsPerBlock>>>(d_res, d_op1, d_op2, length);
    cudaDeviceSynchronize();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 8. Copy Result Back to Host
    cudaMemcpy(h_res, d_res, sizeInBytes, cudaMemcpyDeviceToHost);

    // 9. Verify the Results
    bool success = true;
    const double epsilon = 1e-9;
    for (size_t i = 0; i < length; ++i) {
        double expected = h_op1[i] * h_op2[i];
        if (std::abs(h_res[i] - expected) > epsilon) {
            std::cerr << "Verification failed at index " << i 
                      << ": Expected " << expected << ", got " << h_res[i] << "\n";
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Success! All " << length << " elements multiplied correctly.\n";
    }

    // 10. Clean up memory
    cudaFree(d_op1);
    cudaFree(d_op2);
    cudaFree(d_res);
    free(h_op1);
    free(h_op2);
    free(h_res);

    return 0;
}
