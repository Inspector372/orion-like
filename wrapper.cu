// Wrapper for *any* kernels(Depending on total parameter size though).

#include "wrapper.h"

__global__ void wrapper256(box256 arg, void* func, size_t lidx, size_t hidx) {
    // Index filtering.
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < lidx || workIndex >= hidx) return;

    // theory: "move" the context to actual kernel.
    (func_ptr_t(func))();
}

__global__ void do_nothing() {
    return;
}

/* Runs when callback_mode=0. assigns wrapper256 to wrapper_ptr. */
void initial_wrapper_run() {
    box256 fakearg;
    wrapper256<<<1, 1>>>(fakearg, nullptr, 0, 0);
}

/* Runs when callback_mode=1. assigns do_nothing to nothing_ptr_upper/lower. */
void initial_nothing_run() {
    do_nothing<<<1, 1>>>();
}

void run_wrapper() {
    wrapper256<<<>>>();

}