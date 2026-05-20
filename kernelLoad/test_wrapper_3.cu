
/*
    test_wrapper_3.cu:
        인자 type 전달에 대한 궁극적인 해결법: struct를 만들어서, 그 안에 다 끼워넣는다.
        wrapper128()은 첫 인자로 실제 kernel의 argument 구조를 모방한 구조체, func, low_idx, high_idx를 받음,
        low_idx, high_idx로 filtering 이후, func를 '그대로' call(no arguments)
        그러면 func로 'control flow'만 옮겨진 것처럼 작동함.

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
    libsmctrl_test();
    test_run = 1;
    cudaError_t err;
    some_complex_struct empty_struct;
    float4 empty_float4;

    // Fake launch.
    add100<<<4, 256>>>(nullptr, 0);
    add100_and_addmore<<<4, 256>>>(nullptr, 0);
    mul<<<4, 256>>>(nullptr, nullptr, nullptr, 0);
    complicated<<<4, 256>>>(0.0, nullptr, nullptr, empty_struct, empty_float4, 0);
    cudaDeviceSynchronize();


    // add100 test.
    box128 arg_add100;

    double* A = nullptr;
    cudaMallocManaged(&A, 1024 * sizeof(double));

    size_t length = 1000;
    
    memcpy(&(arg_add100.data[0]), &A, sizeof(double*));
    memcpy(&(arg_add100.data[8]), &length, sizeof(size_t));
    
    wrapper128<<<4, 256>>>(arg_add100, (void *)kernel_ptrs[0], 0, 500);
    wrapper128<<<4, 256>>>(arg_add100, (void *)kernel_ptrs[0], 500, 1000);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    printf("wrapper(add100) error: %s\n", cudaGetErrorString(err));

    printf("wrapper(add100) test:\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("A[%d] = %lf\n", i, A[i]);
    }


    // add100_and_addmore test.
    box128 arg_add100_and_addmore;
    memset(&arg_add100_and_addmore, 0, sizeof(box128));

    double* A2 = nullptr;
    cudaMallocManaged(&A2, 1024 * sizeof(double));
    size_t length2 = 1000;

    // double* A at offset 0, size_t length at offset 8
    memcpy(&(arg_add100_and_addmore.data[0]), &A2, sizeof(double*));
    memcpy(&(arg_add100_and_addmore.data[8]), &length2, sizeof(size_t));

    wrapper128<<<4, 256>>>(arg_add100_and_addmore, (void*)kernel_ptrs[1], 0, 500);
    wrapper128<<<4, 256>>>(arg_add100_and_addmore, (void*)kernel_ptrs[1], 500, 1000);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    printf("wrapper(add100_and_addmore) error: %s\n", cudaGetErrorString(err));

    printf("wrapper(add100_and_addmore) test:\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("A2[%d] = %lf\n", i, A2[i]);
    }
    // A2[50] should be 150.0 (100 from kernel + 50 from addmore), rest 100.0
    printf("A2[50] = %lf (expected 150.0)\n", A2[50]);


    // mul test.
    box128 arg_mul;
    memset(&arg_mul, 0, sizeof(box128));

    double* res = nullptr;
    double* op1 = nullptr;
    double* op2 = nullptr;
    cudaMallocManaged(&res, 1024 * sizeof(double));
    cudaMallocManaged(&op1, 1024 * sizeof(double));
    cudaMallocManaged(&op2, 1024 * sizeof(double));
    size_t length_mul = 1000;

    for(int i = 0; i < 1024; i++) {
        op1[i] = (double)i;
        op2[i] = 2.0;
    }

    // double* res at offset 0, double* op1 at offset 8,
    // double* op2 at offset 16, size_t length at offset 24
    memcpy(&(arg_mul.data[0]),  &res,        sizeof(double*));
    memcpy(&(arg_mul.data[8]),  &op1,        sizeof(double*));
    memcpy(&(arg_mul.data[16]), &op2,        sizeof(double*));
    memcpy(&(arg_mul.data[24]), &length_mul, sizeof(size_t));

    wrapper128<<<4, 256>>>(arg_mul, (void*)kernel_ptrs[2], 0, 500);
    wrapper128<<<4, 256>>>(arg_mul, (void*)kernel_ptrs[2], 500, 1000);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    printf("wrapper(mul) error: %s\n", cudaGetErrorString(err));

    printf("wrapper(mul) test:\n");
    for(int i = 0; i < 1024; i += 10) {
        printf("res[%d] = %lf (expected %lf)\n", i, res[i], op1[i] * 2.0);
    }


    // complicated test.
    box128 arg_complicated;
    memset(&arg_complicated, 0, sizeof(box128));

    double* B = nullptr;
    double* C = nullptr;
    cudaMallocManaged(&B, 1024 * sizeof(double));
    cudaMallocManaged(&C, 1024 * sizeof(double));
    size_t length_comp = 1000;

    for(int i = 0; i < 1024; i++) {
        B[i] = (double)i;
        C[i] = 0.0;
    }

    float      comp_A = 5.0f;
    some_complex_struct comp_D;
    comp_D.Da[0] = 2.0;
    comp_D.Da[1] = 3.0;
    comp_D.Db    = 1.0f;
    float4     comp_E = {10.0f, 0.0f, 0.0f, 0.0f};

    // Layout (matches CUDA ABI struct packing):
    // offset  0: float A          (4 bytes)
    // offset  4: <padding>        (4 bytes, to align double* to 8)
    // offset  8: double* B        (8 bytes)
    // offset 16: double* C        (8 bytes)
    // offset 24: some_complex_struct D
    //              Da[0] double   (8 bytes) @ 24
    //              Da[1] double   (8 bytes) @ 32
    //              Db float       (4 bytes) @ 40
    //              <padding>      (4 bytes) @ 44, struct size rounds to 8
    // offset 48: float4 E         (16 bytes)
    // offset 64: size_t length    (8 bytes)
    memcpy(&(arg_complicated.data[0]),  &comp_A,      sizeof(float));
    memcpy(&(arg_complicated.data[8]),  &B,           sizeof(double*));
    memcpy(&(arg_complicated.data[16]), &C,           sizeof(double*));
    memcpy(&(arg_complicated.data[24]), &comp_D.Da[0],sizeof(double));
    memcpy(&(arg_complicated.data[32]), &comp_D.Da[1],sizeof(double));
    memcpy(&(arg_complicated.data[40]), &comp_D.Db,   sizeof(float));
    memcpy(&(arg_complicated.data[48]), &comp_E,      sizeof(float4));
    memcpy(&(arg_complicated.data[64]), &length_comp, sizeof(size_t));

    wrapper128<<<4, 256>>>(arg_complicated, (void*)kernel_ptrs[3], 0, 500);
    wrapper128<<<4, 256>>>(arg_complicated, (void*)kernel_ptrs[3], 500, 1000);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    printf("wrapper(complicated) error: %s\n", cudaGetErrorString(err));

    printf("wrapper(complicated) test:\n");
    // For workIndex i: C[i] = B_orig[i] * Da[0] * Da[1] + Db - E.x
    //                       = i          * 2.0   * 3.0   + 1.0 - 10.0
    //                       = 6i - 9
    // B[i] = B_orig[i] + A = i + 5
    for(int i = 0; i < 1024; i += 10) {
        double expected_C = (double)i * comp_D.Da[0] * comp_D.Da[1] + comp_D.Db - comp_E.x;
        double expected_B = (double)i + comp_A;
        printf("B[%d] = %lf (expected %lf), C[%d] = %lf (expected %lf)\n",
               i, B[i], expected_B, i, C[i], expected_C);
    }

    return 0;
}
