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
  !gcc libsmctrl.c -c -o libsmctrl.o -fPIC -I/usr/local/cuda-12.8/targets/x86_64-linux/include
  !ar rcs libsmctrl.a libsmctrl.o
  !nvcc -g -G -arch=sm_75 test_simple.cu -o test_simple libsmctrl.a -lcuda -I/usr/local/cuda-12.8/targets/x86_64-linux/include
*/

#include <iostream>
#include <stdio.h>
#include "libsmctrl.h"

typedef void (*func_ptr_t)(int*);

__device__ void fun1(int* A) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    A[workIndex] += 50;
    if(workIndex == 1) printf("Meow from fun1\n");
    return;
}

__global__ void fun2(int* A) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    A[workIndex] += 100;
    if(workIndex == 1) printf("Meow from fun2\n");
    return;
}

__device__ func_ptr_t fun1_ptr = fun1;

__device__ func_ptr_t fun2_ptr;

__global__ void wrapper(int* A) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < 100 || workIndex > 800) return;
    fun1_ptr(A);
    fun2_ptr(A);
}


int main() {
    libsmctrl_test();
    test_run = 1;

    int* A = nullptr;
    cudaMallocManaged(&A, 1024 * sizeof(int));

    func_ptr_t h_fun1_dev_addr;
    cudaMemcpyFromSymbol(&h_fun1_dev_addr, fun1_ptr, sizeof(func_ptr_t));
    printf("fun1_ptr = %p\n", h_fun1_dev_addr);

    fun2<<<4, 256>>>(A);
    cudaDeviceSynchronize();

    func_ptr_t h_fun2_dev_addr = (func_ptr_t)kernel_ptrs[0];

    cudaMemcpyToSymbol(fun2_ptr, &h_fun2_dev_addr, sizeof(func_ptr_t));
    cudaMemcpyFromSymbol(&h_fun2_dev_addr, fun2_ptr, sizeof(func_ptr_t));

    printf("2. fun2_ptr = %p\n", h_fun2_dev_addr);

    wrapper<<<4, 256>>>(A);

    cudaDeviceSynchronize();
    printf("are we alive?\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("A[%d] = %d\n", i, A[i]);
    }
    return 0;
}
