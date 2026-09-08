/*
    How to test:
    gcc libsmctrl.c -c -o libsmctrl.o -fPIC
    ar rcs libsmctrl.a libsmctrl.o
    nvcc -g -G -arch=sm_70 argu-addr.cu -o argu-addr libsmctrl.a -lcuda
    env CUDA_VISIBLE_DEVICES=3 ./argu-addr
*/

/*
    argu-addr.cu

    * QMD에서 추출한 CONSTANT_BUFFER 값과,
    * kernel 내에서 읽은 parameter의 주소 값이 어떤 관계를 가지는지 확인.

    결과:
    example_1argu CONSTANT_BUFFER[0]: 0x7f9ff0220000
    example_1argu argu_addr: 0x7f9ff0220160

    kernel function parameter의 'param memory' 주소를 읽어올 수 있음
    (The address of a kernel parameter may be moved into a register using the mov instruction.
    The resulting address is in the .param state space and is accessed using ld.param instructions.)

    mov.b64 	%rd1, _Z13example_1arguii_param_0;
    cvta.param.u64 	%rd3, %rd1;

    수정: sm_70 이상에서는 __grid_constant__의 argument의 pointer가 param address를 바로 가리킴.
    -> PTX를 사용하지 않아도 됨!

    첫 번째 parameter의 주소는 CONSTANT_BUFFER_ADDR_LOWER/UPPER(i)와 0x160만큼의 offset 차이가 남

    Kernel(i)를 Kernel launch의 i번째 atom이라 하자.
    1. QMD에서 CONSTANT_BUFFER_ADDR_LOWER/UPPER(0), PROGRAM_ADDRESS를 미리 fetch -> BufferAddr(i), ProgramAddr
    2. Global Memory에 Map[(Addr(i) +(or -) 0x168) -> (i, ProgramAddr)]을 저장 (Hash Table or something?)
    3. 실제 Kernel 실행에서, wrapper(int a); 형식에서 &a를 inline ptx로 읽음
    4. Map(&a) = (i, ProgramAddr)
    5. i는 index filtering에 사용, ProgramAddr은 jump에 사용


*/


#include <iostream>
#include <stdio.h>
#include <cstring>
#include "libsmctrl.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>


__global__ void mul(double* res, double* op1, double* op2, uint64_t length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex < length) {
        res[workIndex] = op1[workIndex] * op2[workIndex];
    }
    return;
}

__global__ void example_1argu(const __grid_constant__ int argu1, int argu2) {
    int* argu_addr;
    argu_addr = argu_addr + 1;
    printf("example_1argu argu_addr: %p\n", &argu1);
}


int main() {
    libsmctrl_false_launch_callback();
    callback_mode = 3;

    example_1argu<<<1, 1>>>(1234, 2345);
    printf("example_1argu CONSTANT_BUFFER[0]: %p\n", (void*)buffer_ptrs[0]);
    cudaDeviceSynchronize();

    return 0;
}
