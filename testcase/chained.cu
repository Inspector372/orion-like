#include "common.cuh"
namespace {
__global__ void initialize(unsigned* out, std::size_t n, unsigned seed) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<unsigned>(i) + seed;
}
__global__ void transform(unsigned* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = out[i] * 3u + 1u;
}
}
testcase::Result testcase::chained(const Config& c) {
    Buffer<unsigned> d(c.size);
    check(cudaMemset(d.ptr, 0, c.size * sizeof(unsigned)));
    initialize<<<blocks(c.size), 256>>>(d.ptr, c.size, c.seed);
    check(cudaGetLastError());
    for (int r = 0; r < c.iterations; ++r) {
        transform<<<blocks(c.size), 256>>>(d.ptr, c.size);
        check(cudaGetLastError());
    }
    finish();
    std::vector<unsigned> out(c.size);
    check(cudaMemcpy(out.data(), d.ptr, c.size * sizeof(unsigned), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (std::size_t i = 0; i < c.size; ++i) {
        unsigned expected = static_cast<unsigned>(i) + c.seed;
        for (int r = 0; r < c.iterations; ++r) expected = expected * 3u + 1u;
        ok &= out[i] == expected;
    }
    d.release();
    return {ok, ok ? "Dependent chain verified" : "Kernel ordering/output mismatch"};
}
