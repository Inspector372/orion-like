
/*
    test_simple_1.cu:
        void (*func)(type1, type2, .. typeN)에서,
        type_k와 type'의 '크기'만 같다면,
        type_k를 type'으로 대체해도 문제가 없음.
*/

#include <iostream>
#include <stdio.h>
#include "libsmctrl.h"

// Let's say we don't have this functionality.
typedef void (*func_ptr_t)(uint64_t, uint64_t, uint32_t);

__global__ void fun2(double* A, double B, float C) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    A[workIndex] += 100.0;
    A[workIndex] += B;
    A[workIndex] += C;
    if(workIndex == 1) printf("Meow from fun2\n");
    return;
}

__device__ func_ptr_t fun2_ptr;

__global__ void wrapper(double* A, double B, float C) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < 100 || workIndex > 800) return;
    fun2_ptr((uint64_t)A, (uint64_t)B, (uint32_t)C);
}


int main() {
    libsmctrl_test();
    test_run = 1;

    double* A = nullptr;
    cudaMallocManaged(&A, 1024 * sizeof(double));

    fun2<<<4, 256>>>(A, 0.0, 0.0);
    cudaDeviceSynchronize();

    func_ptr_t h_fun2_dev_addr = (func_ptr_t)kernel_ptrs[0];

    cudaMemcpyToSymbol(fun2_ptr, &h_fun2_dev_addr, sizeof(func_ptr_t));
    cudaMemcpyFromSymbol(&h_fun2_dev_addr, fun2_ptr, sizeof(func_ptr_t));

    printf("2. fun2_ptr = %p\n", h_fun2_dev_addr);

    wrapper<<<4, 256>>>(A, 50.0, 50.0);

    cudaDeviceSynchronize();
    printf("are we alive?\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("A[%d] = %lf\n", i, A[i]);
    }
    return 0;
}
