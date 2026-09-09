#pragma once
#include "common.cuh"
#include <cublasLt.h>
#include <cmath>
#include <algorithm>
#include <limits>

namespace testcase { namespace lt_detail {
inline void lt_check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLASLt status " + std::to_string(static_cast<int>(s)));
}
struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t layout = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t algorithm{};
    Plan() = default;
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    ~Plan() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (layout) cublasLtMatrixLayoutDestroy(layout);
        if (op) cublasLtMatmulDescDestroy(op);
        if (handle) cublasLtDestroy(handle);
    }
    void setup(std::size_t m, std::size_t workspace_bytes) {
        lt_check(cublasLtCreate(&handle));
        lt_check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F));
        // All matrices are square, column-major, with no transposition.
        lt_check(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, m, m, m));
        lt_check(cublasLtMatmulPreferenceCreate(&pref));
        lt_check(cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
        int count = 0;
        lt_check(cublasLtMatmulAlgoGetHeuristic(handle, op, layout, layout,
            layout, layout, pref, 1, &algorithm, &count));
        if (count == 0) throw std::runtime_error("No supported cuBLASLt algorithm for this shape/device");
        lt_check(algorithm.state);
    }
    void multiply(const float* a, const float* b, float* out, void* workspace,
                  std::size_t workspace_bytes) {
        const float alpha = 1.0f, beta = 0.0f;
        lt_check(cublasLtMatmul(handle, op, &alpha, a, layout, b, layout,
            &beta, out, layout, out, layout, &algorithm.algo,
            workspace, workspace_bytes, 0));
        // The scheduler remaps streams; establish completion before consumption.
        finish();
    }
};
inline std::vector<double> reference(const std::vector<double>& a,
                                     const std::vector<double>& b, std::size_t m) {
    std::vector<double> out(m*m, 0.0);
    for (std::size_t col=0; col<m; ++col)
        for (std::size_t row=0; row<m; ++row)
            for (std::size_t k=0; k<m; ++k)
                out[row+col*m] += a[row+k*m] * b[k+col*m];
    return out;
}
inline bool matches(const std::vector<float>& actual, const std::vector<double>& expected) {
    for (std::size_t i=0; i<actual.size(); ++i)
        if (!std::isfinite(actual[i]) ||
            std::fabs(double(actual[i])-expected[i]) > 1e-4 + 2e-4*std::fabs(expected[i]))
            return false;
    return true;
}
inline Result run(const Config& c, bool chain) {
    if (c.size == 0 || c.size > 1024 || c.iterations < 1)
        throw std::invalid_argument("cuBLASLt requires size 1..1024 and positive iterations");
    const std::size_t n=c.size*c.size, bytes=n*sizeof(float);
    const std::size_t workspace_bytes=4*1024*1024;
    std::vector<float> a(n), b(n), out(n, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t i=0; i<n; ++i) {
        a[i]=input(i, c.seed)/16.0f;
        b[i]=input(i, c.seed+7)/16.0f;
    }
    std::vector<double> ad(a.begin(), a.end()), bd(b.begin(), b.end());
    const auto first=reference(ad, bd, c.size);
    const auto expected=chain ? reference(first, ad, c.size) : first;
    Buffer<float> da(n), db(n), intermediate(n), result(n);
    Buffer<unsigned char> workspace(workspace_bytes);
    Plan plan;
    plan.setup(c.size, workspace_bytes);
    check(cudaMemcpy(da.ptr,a.data(),bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.ptr,b.data(),bytes,cudaMemcpyHostToDevice));
    for (int iteration=0; iteration<c.iterations; ++iteration) {
        std::fill(out.begin(),out.end(),std::numeric_limits<float>::quiet_NaN());
        check(cudaMemcpy(intermediate.ptr,out.data(),bytes,cudaMemcpyHostToDevice));
        plan.multiply(da.ptr,db.ptr,intermediate.ptr,workspace.ptr,workspace_bytes);
        check(cudaMemcpy(out.data(),intermediate.ptr,bytes,cudaMemcpyDeviceToHost));
        if (!matches(out,first)) return {false,"cuBLASLt first GEMM mismatch"};
        if (chain) {
            std::fill(out.begin(),out.end(),std::numeric_limits<float>::quiet_NaN());
            check(cudaMemcpy(result.ptr,out.data(),bytes,cudaMemcpyHostToDevice));
            plan.multiply(intermediate.ptr,da.ptr,result.ptr,workspace.ptr,workspace_bytes);
            check(cudaMemcpy(out.data(),result.ptr,bytes,cudaMemcpyDeviceToHost));
            if (!matches(out,expected)) return {false,"cuBLASLt dependent GEMM mismatch"};
        }
    }
    da.release(); db.release(); intermediate.release(); result.release(); workspace.release();
    return {true,chain ? "cuBLASLt (A*B)*A verified" : "cuBLASLt A*B verified"};
}
} }
