/*
  Test for kernel-inside-kernel launch.
  fun2() is __global__ kernel, but it's device entrypoint is extracted by libsmctrl_test().
  it's stored in kernel_ptrs[0], and stored in device-side global variable fun2_ptr.
  wrapper() calls fun2_ptr like __device__ kernel, and try to filter index < 100 or index > 800.
  not filtered A[index] will have value of 250,
  while filtered A[index] will have value of 100.
  working in google colab enviroment, but need to test it further.

  TODO: infer function structure from **any** function call.

  How to test:
  gcc libsmctrl.c -c -o libsmctrl.o -fPIC
  ar rcs libsmctrl.a libsmctrl.o
  nvcc -g -G -arch=sm_75 test_simple.cu -o test_simple libsmctrl.a -lcuda
*/

/*
    TODO
        given: kernel_address, arg**
        want: kernel_address(arg1, arg2, arg3, ... argN)

    문제를 쪼개보자.
    문제 1: argk의 type은 무엇?
        -> cuda runtime api가 '작동한다면', type의 크기와 offset은 알 수 있음
        그러면 uint_32t, uint_64t와 같은 '똑같은 크기의 type'으로 바꿀 수 있나?
        -> 예제 상에서는 가능했음, 실제 orion-like 구현체 내에서는?
    문제 2: kernel_address의 type은 무엇?
        void (*func)(type1, type2, .. typeN)을 dynamic하게 구현할 수 있을까?


*/

#include <iostream>
#include <stdio.h>
#include "libsmctrl.h"
#include <cuda_runtime.h>
#include <cuda.h>

// Let's say we don't have this functionality.
typedef void (*func_ptr_t)(uint32_t, uint64_t, uint64_t, uint32_t, uint32_t);

__global__ void fun2(float Z, double* A, double B, float C, float D) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    A[workIndex] += 100.0;
    A[workIndex] += B;
    A[workIndex] += C;
    if(workIndex == 1) printf("Meow from fun2\n");
    return;
}

__device__ func_ptr_t fun2_ptr;

__global__ void wrapper(float Z, double* A, double B, float C, float D) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < 100 || workIndex > 800) return;
    fun2_ptr((uint32_t)Z, (uint64_t)A, (uint64_t)B, (uint32_t)C, (uint32_t)D);
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

    fun2<<<4, 256>>>(0.0, A, 0.0, 0.0, 0.0);
    cudaDeviceSynchronize();

    func_ptr_t h_fun2_dev_addr = (func_ptr_t)kernel_ptrs[0];

    cudaMemcpyToSymbol(fun2_ptr, &h_fun2_dev_addr, sizeof(func_ptr_t));
    cudaMemcpyFromSymbol(&h_fun2_dev_addr, fun2_ptr, sizeof(func_ptr_t));

    printf("2. fun2_ptr = %p\n", h_fun2_dev_addr);

    wrapper<<<4, 256>>>(0.0, A, 50.0, 50.0, 0.0);

    cudaDeviceSynchronize();
    printf("are we alive?\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("A[%d] = %lf\n", i, A[i]);
    }
    return 0;
}
