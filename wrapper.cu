// Wrapper for *any* kernels(Depending on total parameter size though).

#include "wrapper.h"

__global__ void wrapper128(box128 arg, void* func, size_t lidx, size_t hidx) {
    // Index filtering.
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < lidx || workIndex >= hidx) return;

    // theory: "move" the context to actual kernel.
    (func_ptr_t(func))();
}

void run_wrapper() {
    wrapper128<<<>>>();

}