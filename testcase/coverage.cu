#include "common.cuh"
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
    bool ok = true;
    for (auto value : h) ok &= value == static_cast<unsigned>(c.iterations);
    d.release();
    return {ok, ok ? "Each logical thread executed exactly once per repetition" : "Missing or duplicated execution"};
}
