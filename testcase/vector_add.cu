#include "common.cuh"
namespace {
__global__ void add(const float* a, const float* b, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
}
testcase::Result testcase::vector_add(const Config& c) {
    std::vector<float> a(c.size), b(c.size), out(c.size);
    for (std::size_t i = 0; i < c.size; ++i) { a[i] = input(i, c.seed); b[i] = input(i, c.seed + 7); }
    Buffer<float> da(c.size), db(c.size), dc(c.size);
    const auto bytes = c.size * sizeof(float);
    check(cudaMemcpy(da.ptr, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.ptr, b.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemset(dc.ptr, 0xff, bytes));
    for (int r = 0; r < c.iterations; ++r) {
        add<<<blocks(c.size), 256>>>(da.ptr, db.ptr, dc.ptr, c.size);
        check(cudaGetLastError());
    }
    finish();
    check(cudaMemcpy(out.data(), dc.ptr, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (std::size_t i = 0; i < c.size; ++i) ok &= out[i] == a[i] + b[i];
    da.release(); db.release(); dc.release();
    return {ok, ok ? "Vector addition verified" : "Vector addition mismatch"};
}
