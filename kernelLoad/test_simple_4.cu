/*
    test_simple_4.cu:
        test_simple_2와 비슷한 방식으로 test_simple_3의 box 선언을 wrapper 밖으로 옮김.
*/

#include <iostream>
#include <stdio.h>
#include <cstring>
#include "libsmctrl.h"
#include <cuda_runtime.h>
#include <cuda.h>

// Let's say we don't have this functionality.

__global__ void fun2(float Z, double* A, float B) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    A[workIndex] += 100.0;
    A[workIndex] += B;
    A[workIndex] += Z;
    if(workIndex == 200) printf("Meow from fun2\n");
    return;
}

typedef union{
    uint32_t data32;
    uint64_t data64;
    // ubox의 크기가 16 byte면 정상적으로 작동하지 않고,
    // 8 byte면 정상적으로 작동한다???
    // alignment issue?
    // float4 data128;  
} ubox;

typedef void (*func_ptr_t)(ubox, ubox, ubox);

__device__ func_ptr_t fun2_ptr;

__global__ void wrapper(ubox Z, ubox A, ubox B) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < 100 || workIndex > 800) return;
    fun2_ptr(Z, A, B);
}

/*
    fun2_ptr에 대한 type을 지정하는 함수.

*/
void type_assign() {
    size_t count = 0;
    size_t offset;
    size_t size;
    while(cudaFuncGetParamInfo((const void*)fun2, count, &offset, &size) == cudaSuccess) {
        printf("count: %ld, offset: %ld, size: %ld\n", count, offset, size);
        count++;
    }
}


int main() {
    libsmctrl_test();
    test_run = 1;

    type_assign();

    double* A = nullptr;
    cudaMallocManaged(&A, 1024 * sizeof(double));

    fun2<<<4, 256>>>(0.0, A, 0.0);
    cudaDeviceSynchronize();

    func_ptr_t h_fun2_dev_addr = (func_ptr_t)kernel_ptrs[0];

    cudaMemcpyToSymbol(fun2_ptr, &h_fun2_dev_addr, sizeof(func_ptr_t));
    cudaMemcpyFromSymbol(&h_fun2_dev_addr, fun2_ptr, sizeof(func_ptr_t));

    printf("2. fun2_ptr = %p\n", h_fun2_dev_addr);

    float B = 5.0;
    float Z = 1.0;
    uint32_t B_;
    uint32_t Z_;
    std::memcpy(&B_, &B, sizeof(float));
    std::memcpy(&Z_, &Z, sizeof(float));

    ubox Abox;
    ubox Bbox;
    ubox Zbox;
    Abox.data64 = (uint64_t)A;
    Bbox.data32 = B_;
    Zbox.data32 = Z_;

    wrapper<<<4, 256>>>(Zbox, Abox, Bbox);
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    printf("are we alive?\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("A[%d] = %lf\n", i, A[i]);
    }
    return 0;
}
