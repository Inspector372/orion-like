
/*
    test_simple_2.cu:
        wrapper에 전달되는 인자들이 실제 type이 아니어도,
        내부 function call에는 문제가 없을까?
        (= cuda 컴파일러가 실제로는 '잘못된' 타입 캐스팅을 바로잡는 것이 아닐까?)
*/

#include <iostream>
#include <stdio.h>
#include <cstring>
#include "libsmctrl.h"
#include <cuda_runtime.h>
#include <cuda.h>

// Let's say we don't have this functionality.
typedef void (*func_ptr_t)(uint32_t, uint64_t, uint32_t, uint64_t);

__global__ void fun2(float Z, double* A, float B, double C) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    A[workIndex] += 100.0;
    A[workIndex] += B;
    A[workIndex] += C;
    if(workIndex == 200) printf("Meow from fun2\n");
    return;
}

__device__ func_ptr_t fun2_ptr;

__global__ void wrapper(uint32_t Z, uint64_t A, uint32_t B, uint64_t C) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < 100 || workIndex > 800) return;
    fun2_ptr(Z, A, B, C);
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

    fun2<<<4, 256>>>(0.0, A, 0.0, 0.0);
    cudaDeviceSynchronize();

    func_ptr_t h_fun2_dev_addr = (func_ptr_t)kernel_ptrs[0];

    cudaMemcpyToSymbol(fun2_ptr, &h_fun2_dev_addr, sizeof(func_ptr_t));
    cudaMemcpyFromSymbol(&h_fun2_dev_addr, fun2_ptr, sizeof(func_ptr_t));

    printf("2. fun2_ptr = %p\n", h_fun2_dev_addr);

    float Z = 0.0;
    float B = 5.0;
    double C = 10.0;
    uint32_t Z_;
    uint32_t B_;
    uint64_t C_;
    std::memcpy(&B_, &B, sizeof(float));
    std::memcpy(&Z_, &Z, sizeof(float));
    std::memcpy(&C_, &C, sizeof(double));

    wrapper<<<4, 256>>>(Z_, (uint64_t)A, B_, C_);
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    printf("are we alive?\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("A[%d] = %lf\n", i, A[i]);
    }
    return 0;
}
