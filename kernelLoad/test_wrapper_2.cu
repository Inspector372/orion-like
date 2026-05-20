
/*
    test_wrapper_2.cu:
        wrapper->kernel으로 가는 과정에서,
            1. kernel의 인자 뒤에 '뭔가 더 붙어 있어도' 상관은 없음.
            2. kernel의 인자의 type은 중요하지 않음(test_simple_1.cu).
            3. kernel의 인자를 ubox로 설정하는 것은 불가능함(test_simple_4.cu).
            4. kernel 내에 다른 __device__ function call이 있어도 문제는 없음.
            5. kernel 내에 printf()가 있으면 printf() call 중 invalid instruction error 발생.
*/

#include <iostream>
#include <stdio.h>
#include <cstring>
#include "libsmctrl.h"
#include <cuda_runtime.h>
#include <cuda.h>

__device__ __noinline__ void addmore(double* A) {
    if(A != nullptr) A[50] += 50.0;
}

/*
    Tests basic kernel function.
*/
__global__ void add100(double* A, size_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        A[workIndex] += 100.0;
    }
    if(workIndex == 300) addmore(A);
    return;
}

typedef void (*func_ptr_t)();



__global__ void wrapper_for_add100(uint64_t arg1, uint64_t arg2, void* func, size_t lidx, size_t hidx) {
    // Index filtering.
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < lidx || workIndex >= hidx) return;

    // theory: "move" the context to actual kernel.
    (func_ptr_t(func))();
}


int main() {
    // Store kernel PROGRAM_ADDRESS to kerenl_ptrs[].
    libsmctrl_test();
    test_run = 1;
    cudaError_t err;

    // Fake launch.
    add100<<<4, 256>>>(nullptr, 0);
    mul<<<4, 256>>>(nullptr, nullptr, nullptr, 0);
    complicated<<<4, 256>>>(0.0, nullptr, nullptr, 0.0, 0.0, 0);
    cudaDeviceSynchronize();

    // Configure args.
    double* A = nullptr;
    cudaMallocManaged(&A, 1024 * sizeof(double));
    err = cudaGetLastError();
    printf("wrapper(add100) error: %s\n", cudaGetErrorString(err));
    
    wrapper_for_add100<<<4, 256>>>((uint64_t)A, 1000, (void *)kernel_ptrs[0], 0, 500);
    wrapper_for_add100<<<4, 256>>>((uint64_t)A, 1000, (void *)kernel_ptrs[0], 500, 1000);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    printf("wrapper(add100) error: %s\n", cudaGetErrorString(err));

    printf("are we alive?\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("A[%d] = %lf\n", i, A[i]);
    }
    return 0;
}
