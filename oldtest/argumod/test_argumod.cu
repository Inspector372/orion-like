/*
    How to test:
    gcc libsmctrl.c -c -o libsmctrl.o -fPIC
    ar rcs libsmctrl.a libsmctrl.o
    nvcc -g -G test_argumod.cu -o test_argumod libsmctrl.a -lcuda
*/


/*
    test_argumod.cu:
        LithOS 원래 구현은 다음과 같음.
        0. (fake launch) - 특정한 configuration을 이용해 QMD에서 detect, nothing-redirect
        1. __global__ f() atom을 실행
        2. atom에서, 원래의 argument 뒤에 (PROGRAM_ADDR(f), lidx, hidx)를 fetch
        3. wrapper()의 PROGRAM_ADDRESS로 QMD를 modify
        4. wrapper()에서 index filer
        5. wrapper()에서 f() 로 redirection

    이 파일은 (2)를 테스트함. 즉, QMD 레벨에서 argument의 크기를 임의로 '늘리는' 것이 가능한가?

    NVC3C0_QMDV02_02_CONSTANT_BUFFER_SIZE_SHIFTED4(i)          MW((1087+(i)*64):(1075+(i)*64))
    -> 먼저 여기를 inspect
    
    결과: CONSTANT_BUFFER_ADDR_LOWER/UPPER에서 얻은 address는 argument를 담는 주소 근처(test 에서는 8*44 byte 이후) 의 포인터를 가리킴
    argument는 여기서부터 선형으로 존재
    argument 뒤는 비어있었음

    -> 그러면 argument를 그냥 끼워넣어도 되지 않을까...?

    
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
} box1234;


__global__ void mul(double* res, double* op1, double* op2, uint64_t length, box1234 bigdummy) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        res[workIndex] = op1[workIndex] * op2[workIndex];
    }
    if(workIndex == 0) {
        printf("res : %p\n", res);
        printf("op1 : %p\n", op1);
    }
    return;
}

__global__ void printkernel(uint64_t ptr, uint64_t dummy, box1234 dummy2) {

    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex == 0) {
        uint64_t ptr_prime = ptr + 0x10000;
        printf("ptr : %p\n", ptr_prime);
        for(int i=0; i<100; i++) {
            printf("%dth : %p\n", i, *((uint64_t*)ptr_prime + i));
        }
    }
    return;
}

typedef void (*func_ptr_t)();

typedef struct{
    unsigned char data[256];  
} box256;

__global__ void wrapper(box256 box) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < 100 || workIndex > 800) return;
}


int main() {
    libsmctrl_false_launch_callback();
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
    box1234 box;
    for(int i=0; i<128; i++) {
        box.data[i] = 0xfc;
    }
    mul<<<blocksPerGrid, threadsPerBlock>>>(d_res, d_op1, d_op2, length, box);
    cudaDeviceSynchronize();
    printkernel<<<1, 1>>>(buffer_ptrs[0], 0x1234, box);
    cudaDeviceSynchronize();
    printkernel<<<1, 1>>>(buffer_ptrs[0], 0x2345, box);

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
