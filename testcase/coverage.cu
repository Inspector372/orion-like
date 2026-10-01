#include "common.cuh"
#include <string>
namespace {
__global__ void visit(unsigned* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out + i, 1u);
}
}
testcase::Result testcase::coverage(const Config& c) {
    Buffer<unsigned> d(c.size);
    std::vector<unsigned> h(c.size, 0u);
    check(cudaMemcpy(d.ptr, h.data(), c.size * sizeof(unsigned), cudaMemcpyHostToDevice));
    for (int r = 0; r < c.iterations; ++r) {
        visit<<<blocks(c.size), 256>>>(d.ptr, c.size);
        check(cudaGetLastError());
    }
    finish();
    check(cudaMemcpy(h.data(), d.ptr, c.size * sizeof(unsigned), cudaMemcpyDeviceToHost));
    d.release();
    const unsigned expected = static_cast<unsigned>(c.iterations);
    for (std::size_t i = 0; i < c.size; ++i) {
        if (h[i] != expected) {
            return {false, "Coverage first mismatch at index " + std::to_string(i) +
                " (block " + std::to_string(i / 256) +
                ", thread " + std::to_string(i % 256) +
                "): expected=" + std::to_string(expected) +
                ", actual=" + std::to_string(h[i])};
        }
    }
    return {true, "Each logical thread executed exactly once per repetition"};
}
