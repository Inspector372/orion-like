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

#include <iostream>
#include <stdio.h>
#include <cstring>
#include "libsmctrl.h"
#include <cuda_runtime.h>
#include <cuda.h>


__global__ void add100(double* A, size_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        A[workIndex] += 100.0;
    }
    if(workIndex == 200) printf("Meow from add100, A: %ld, length : %ld\n", (size_t)A, length);
    return;
}

__global__ void mul(double* res, double* op1, double* op2, size_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        res[workIndex] = op1[workIndex] * op2[workIndex];
    }
    if(workIndex == 200) printf("Meow from mul\n");
    return;
}

__global__ void complicated(float A, double* B, double* C, float D, double E, size_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        B[workIndex] = C[workIndex];
        C[workIndex] = 2 * B[workIndex];
        B[workIndex] += A;
        C[workIndex] *= D;
        C[workIndex] -= E;
    }
    if(workIndex == 200) printf("Meow from complicated\n");
    return;
}

typedef union{
    uint32_t data32;
    uint64_t data64;
    __int128 data128;  
} ubox;

typedef void (*func_ptr_t_0arg)();
typedef void (*func_ptr_t_1arg)(ubox);
typedef void (*func_ptr_t_2arg)(ubox, ubox);
typedef void (*func_ptr_t_3arg)(ubox, ubox, ubox);
typedef void (*func_ptr_t_4arg)(ubox, ubox, ubox, ubox);
typedef void (*func_ptr_t_5arg)(ubox, ubox, ubox, ubox, ubox);
typedef void (*func_ptr_t_6arg)(ubox, ubox, ubox, ubox, ubox, ubox);
typedef void (*func_ptr_t_7arg)(ubox, ubox, ubox, ubox, ubox, ubox, ubox);
typedef void (*func_ptr_t_8arg)(ubox, ubox, ubox, ubox, ubox, ubox, ubox, ubox);

__device__ func_ptr_t_0arg fun_ptr_0arg;
__device__ func_ptr_t_1arg fun_ptr_1arg;
__device__ func_ptr_t_2arg fun_ptr_2arg;
__device__ func_ptr_t_3arg fun_ptr_3arg;
__device__ func_ptr_t_4arg fun_ptr_4arg;
__device__ func_ptr_t_5arg fun_ptr_5arg;
__device__ func_ptr_t_6arg fun_ptr_6arg;
__device__ func_ptr_t_7arg fun_ptr_7arg;
__device__ func_ptr_t_8arg fun_ptr_8arg;


/* 
    This is wrapper we will *actually* use in the project.
    func - device address(PROGRAM_ADDRESS) of kernel we are going to launch.
    args - array of pointers of arguments, with length of argnum. implicitly obtained by cudaFuncGetParamInfo().
    arg_size - array of size of each arguments in bytes. explicitly obtained by cudaFuncGetParamInfo().
    argnum - number of arguments.
    lidx, hidx - filters index not in ( lidx <= idx < hidx ).
*/
__global__ void wrapper(void* func, void** args, size_t* arg_size, size_t argnum, size_t lidx, size_t hidx) {
    // Index filtering.
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < lidx || workIndex >= hidx) return;

    // Create box for each args and fill it.
    // This is created **per thread**, so it's of high overhead now.
    // Need to test for __shared__, and test more about defining ubox outside wrapper(which caused a bug).
    ubox arg_box[8];
    for(int i = 0; i < argnum; i++) {
        if(arg_size[i] == 4) { // 32bit
            arg_box[i].data32 = *((uint32_t*)args[i]);
            if(workIndex == 200) {
                printf("arg_box[%d].data32 = %d\n", i, arg_box[i].data32);
            }
        } 
        else if(arg_size[i] == 8) { // 64bit
            arg_box[i].data64 = *((uint64_t*)args[i]);
            if(workIndex == 200) {
                printf("arg_box[%d].data64 = %ld\n", i, arg_box[i].data64);
            }
        }
    }

    // Launch the kernel as device kernel.
    switch(argnum) {
        case 0: ((func_ptr_t_0arg)func)();
        break;
        case 1: ((func_ptr_t_1arg)func)(arg_box[0]);
        break;
        case 2: ((func_ptr_t_2arg)func)(arg_box[0], arg_box[1]);
        break;
        case 3: ((func_ptr_t_3arg)func)(arg_box[0], arg_box[1], arg_box[2]);
        break;
        case 4: ((func_ptr_t_4arg)func)(arg_box[0], arg_box[1], arg_box[2], arg_box[3]);
        break;
        case 5: ((func_ptr_t_5arg)func)(arg_box[0], arg_box[1], arg_box[2], arg_box[3], arg_box[4]);
        break;
        case 6: ((func_ptr_t_6arg)func)(arg_box[0], arg_box[1], arg_box[2], arg_box[3], arg_box[4], arg_box[5]);
        break;
        case 7: ((func_ptr_t_7arg)func)(arg_box[0], arg_box[1], arg_box[2], arg_box[3], arg_box[4], arg_box[5], arg_box[6]);
        break;
        case 8: ((func_ptr_t_8arg)func)(arg_box[0], arg_box[1], arg_box[2], arg_box[3], arg_box[4], arg_box[5], arg_box[6], arg_box[7]);
        break;
        default:
        break;
    }
}


int main() {
    // Store kernel PROGRAM_ADDRESS to kerenl_ptrs[].
    libsmctrl_test();
    test_run = 1;

    // Fake launch.
    add100<<<4, 256>>>(nullptr, 0);
    mul<<<4, 256>>>(nullptr, nullptr, nullptr, 0);
    complicated<<<4, 256>>>(0.0, nullptr, nullptr, 0.0, 0.0, 0);
    cudaDeviceSynchronize();

    // Configure args.
    void*** args = nullptr;
    cudaMallocManaged(&args, 3 * sizeof(void **));

    cudaMallocManaged(&args[0], 2 * sizeof(void *));
    cudaMallocManaged(&args[1], 4 * sizeof(void *));
    cudaMallocManaged(&args[2], 6 * sizeof(void *));

    double** A_ptr = nullptr;
    cudaMallocManaged(&A_ptr, sizeof(double *));
    double* A = nullptr;
    cudaMallocManaged(&A, 1024 * sizeof(double));
    *A_ptr = A;
    args[0][0] = (void *)A_ptr;
    size_t* length_ptr = nullptr;
    cudaMallocManaged(&length_ptr, sizeof(size_t));
    *length_ptr = 1000;
    args[0][1] = (void *)length_ptr;
    

    // Configure arg_sizes.
    size_t **arg_sizes = nullptr;
    cudaMallocManaged(&arg_sizes, 3 * sizeof(size_t *));

    cudaMallocManaged(&arg_sizes[0], 2 * sizeof(size_t));
    cudaMallocManaged(&arg_sizes[1], 4 * sizeof(size_t));
    cudaMallocManaged(&arg_sizes[2], 6 * sizeof(size_t));
    arg_sizes[0][0] = sizeof(double*);
    arg_sizes[0][1] = sizeof(size_t);


    wrapper<<<4, 256>>>((void *)kernel_ptrs[0], args[0], arg_sizes[0], 2, 0, 500);

    wrapper<<<4, 256>>>((void *)kernel_ptrs[0], args[0], arg_sizes[0], 2, 500, 1000);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    printf("wrapper(add100) error: %s\n", cudaGetErrorString(err));

    printf("are we alive?\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("A[%d] = %lf\n", i, A[i]);
    }
    return 0;
}
