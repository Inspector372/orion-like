#include "common.cuh"
namespace {
__global__ void multiply(const float* a, const float* b, float* out, std::size_t m) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * m) return;
    std::size_t row = i / m, col = i % m;
    float sum = 0;
    for (std::size_t k = 0; k < m; ++k) sum += a[row*m+k] * b[k*m+col];
    out[i] = sum;
}
}
testcase::Result testcase::matmul(const Config& c) {
    const auto n = c.size * c.size, bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), out(n);
    for (std::size_t i = 0; i < n; ++i) { a[i] = input(i, c.seed); b[i] = input(i, c.seed + 7); }
    Buffer<float> da(n), db(n), dc(n);
    check(cudaMemcpy(da.ptr, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.ptr, b.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemset(dc.ptr, 0xff, bytes));
    for (int r = 0; r < c.iterations; ++r) {
        multiply<<<blocks(n), 256>>>(da.ptr, db.ptr, dc.ptr, c.size);
        check(cudaGetLastError());
    }
    finish();
    check(cudaMemcpy(out.data(), dc.ptr, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    // Binary-exact inputs and bounded dimensions permit exact comparison.
    for (std::size_t row = 0; row < c.size; ++row)
        for (std::size_t col = 0; col < c.size; ++col) {
            float expected = 0;
            for (std::size_t k = 0; k < c.size; ++k) expected += a[row*c.size+k] * b[k*c.size+col];
            ok &= out[row*c.size+col] == expected;
        }
    da.release(); db.release(); dc.release();
    return {ok, ok ? "1D matmul verified" : "Matmul mismatch"};
}
