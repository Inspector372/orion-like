
#include <iostream>
#include <stdio.h>

__global__ void a(int b) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if(workIndex == 0 || workIndex == 1 || workIndex == 2) {
        printf("b before mod: %d\n", b);
        if(workIndex == 0) b = 3;
        if(workIndex == 2) b = 4;
        printf("b after mod: %d\n", b);

    } 
}



int main() {
    a<<<1, 128>>>(2);
    cudaDeviceSynchronize();
    return 0;
}
