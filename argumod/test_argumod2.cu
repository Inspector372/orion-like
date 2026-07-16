/*
    How to test:
    gcc libsmctrl.c -c -o libsmctrl.o -fPIC
    ar rcs libsmctrl.a libsmctrl.o
    nvcc -g -G test_argumod2.cu -o test_argumod2 libsmctrl.a -lcuda
*/


/*
    test_argumod2.cu:
    trying to print the whole constant memory.

    -> CUDA의 argument는 constant memory에 꼭 linear 하게 배열되지는 않음....?
    실제로 kernel 내에서 &a 식으로 접근하는 address는, stack(또는 비슷한 구조)으로 보임

    
*/

#include <iostream>
#include <stdio.h>
#include <cstring>
#include "libsmctrl.h"
#include <cuda_runtime.h>
#include <cuda.h>


__global__ void printkernel(uint64_t a, uint32_t b, uint64_t c, uint64_t d) {

    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex == 0) {
        printf("a : %lx\n", a);
        uint64_t* ptr_a = &a;
        uint32_t* ptr_b = &b;
        // uint64_t* ptr_c = &c;
        // uint64_t* ptr_d = &d;
        printf("ptr_a : %lx\n", ptr_a);
        printf("ptr_b : %lx\n", ptr_b);
        // printf("ptr_c : %lx\n", ptr_c);
        // printf("ptr_d : %lx\n", ptr_d);
        for(int i=0; i<1234; i++) {
            printf("%dth : %lx\n", i, *(ptr_a + i));
        }
    }
    return;
}


int main() {
    libsmctrl_false_launch_callback();
    callback_mode = 4;
    uint64_t a = 0xafafafafafafafaf;
    uint32_t b = 0xbfbfbfbf;
    uint64_t c = 0xcfcfcfcfcfcfcfcf;
    uint64_t d = 0xdfdfdfdfdfdfdfdf;
    printkernel<<<2, 256>>>(a, b, c, d);
    cudaDeviceSynchronize();

    return 0;
}
