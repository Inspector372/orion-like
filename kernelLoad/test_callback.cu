/*
  How to test:
  gcc libsmctrl.c -c -o libsmctrl.o -fPIC
  ar rcs libsmctrl.a libsmctrl.o
  nvcc -g -G test_callback.cu -o test_callback libsmctrl.a -lcuda
*/

/*
    False Launch Solution.
    (a. QMD global callback를 적용해 첫 번째 launch(wrapper) 제외 모든 launch의 PROGRAM_ADDRESS를 do_nothing으로 변경)
    (b. launch of wrapper() -> PROGRAM_ADDRESS 저장)
    (c. launch of do_nothing() -> PROGRAM_ADDRESS 저장)
    1. cudaLaunchKernel()를 capture
    2. False Launch: real_cudaLaunchKernel()에 func를 넣고 ThreadDim=t로 바꾸고 실행
    (d. callback이 QMD를 건드려 do_nothing kernel로 silent exit )
    3. QMD callback에서 PROGRAM_ADDRESS를 F[THREAD_DIMENSION0]에 fetch
    4. Silent exit에서는 return된 에러가 없음
    5. fetch된 PROGRAM_ADDRESS와 kernel address를 이용해 real_cudaLaunchKernel()에 wrapper address와 적당한 argument를 넣고 실행

    문제 1: global callback에서, false launch를 어떻게 구분...?
     -> wrapper()를 제외한 모든 launch를 false launch로 간주?
     -> 이러면 모든 call을 wrapper()를 거쳐서 실행해야 함.
    문제 2: 이걸 어디로 fetch?
     -> ThreadDim을 key로 사용, F[ThreadDim] = PROGRAM_ADDRESS

    의문점 1. wrapper()의 PROGRAM_ADDRESS는 바뀌지 않는가?
      -> test에서는 바뀌지 않음, 다만 실제 launch에서는 configuration과 context에 따라 바뀔지도...?
      -> 그러면 wrapper()를 따로 구분할 방법이 있을까?

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

__global__ void do_nothing() {
    return;
}

/*
    Tests basic kernel function.
*/
__global__ void add100(double* A, size_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        A[workIndex] += 100.0;
    }
    return;
}

/*
    Tests additional launch.
*/
__global__ void add100_and_addmore(double* A, size_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        A[workIndex] += 100.0;
    }
    if(workIndex == 300) addmore(A);
    return;
}

/*
    Tests actual kernel operation.
*/
__global__ void mul(double* res, double* op1, double* op2, size_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        res[workIndex] = op1[workIndex] * op2[workIndex];
    }
    return;
}

/*
    Tests some complicated logic inside kernel, with various data structures.
*/
typedef struct{
    double Da[2];
    float Db;
} some_complex_struct;

__global__ void complicated(float A, double* B, double* C, some_complex_struct D, float4 E, size_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        C[workIndex] = B[workIndex];
        B[workIndex] += A;
        C[workIndex] *= D.Da[0];
        C[workIndex] *= D.Da[1];
        C[workIndex] += D.Db;
        C[workIndex] -= E.x;
    }
    return;
}

typedef struct{
    unsigned char data[128];  
} box128;

typedef void (*func_ptr_t)();



__global__ void wrapper128(box128 arg, void* func, size_t lidx, size_t hidx) {
    // Index filtering.
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < lidx || workIndex >= hidx) return;

    // theory: "move" the context to actual kernel.
    (func_ptr_t(func))();
}


int main() {
    // Store kernel PROGRAM_ADDRESS to kerenl_ptrs[].
    cudaError_t err;
    some_complex_struct empty_struct;
    float4 empty_float4;
    box128 args;

    // Wrapper fakelaunch.
    callback_mode = 0;
    libsmctrl_test(1);
    wrapper128<<<4, 256>>>(args, nullptr, 0, 0);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("Kernel launch error wrapper: %s\n", cudaGetErrorString(err));

    // do_nothing launch.
    callback_mode = 1;
    do_nothing<<<4, 256>>>();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("Kernel launch error nothing: %s\n", cudaGetErrorString(err));

    callback_mode = 2;

    // Fake launch.
    double* A = nullptr;
    cudaMallocManaged(&A, 1024 * sizeof(double));
    add100<<<4, 256>>>(A, 0);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("Kernel launch error 1: %s\n", cudaGetErrorString(err));

    add100_and_addmore<<<2, 512>>>(A, 0);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("Kernel launch error 2: %s\n", cudaGetErrorString(err));

    wrapper128<<<4, 4>>>(args, nullptr, 0, 0);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("Kernel launch error 3: %s\n", cudaGetErrorString(err));

    for(int i = 0; i < 1024; i += 10) {
        printf("A[%d] = %lf\n", i, A[i]);
    }

    return 0;
}
