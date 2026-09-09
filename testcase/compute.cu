#include "common.cuh"
namespace {
__host__ __device__ unsigned recurrence(unsigned x, int work) {
    for (int k = 0; k < work; ++k) x = (x * 1664525u + 1013904223u) ^ (x >> 7);
    return x;
}
__global__ void calculate(unsigned* out, std::size_t n, int work, unsigned seed) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = recurrence(static_cast<unsigned>(i) + seed, work);
}
}
testcase::Result testcase::compute(const Config& c) {
    Buffer<unsigned> d(c.size);
    std::vector<unsigned> out(c.size, 0u);
    check(cudaMemcpy(d.ptr, out.data(), c.size * sizeof(unsigned), cudaMemcpyHostToDevice));
    for (int r = 0; r < c.iterations; ++r) {
        calculate<<<blocks(c.size), 256>>>(d.ptr, c.size, c.work, c.seed);
        check(cudaGetLastError());
    }
    finish();
    check(cudaMemcpy(out.data(), d.ptr, c.size * sizeof(unsigned), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (std::size_t i = 0; i < c.size; ++i) ok &= out[i] == recurrence(static_cast<unsigned>(i) + c.seed, c.work);
    d.release();
    return {ok, ok ? "Compute recurrence verified" : "Compute mismatch"};
}
