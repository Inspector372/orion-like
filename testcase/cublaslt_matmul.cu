#include "testcase.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cublasLt.h>
#include <exception>
#include <algorithm>
#include <limits>
#include <cmath>

namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

float input(std::size_t i, std::uint32_t seed) {
    return static_cast<float>((i + seed) % 17) * 0.25f;
}

void lt_check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLASLt status " + std::to_string(static_cast<int>(s)));
    }
}

std::vector<double> reference(const std::vector<double>& a, const std::vector<double>& b,
                              std::size_t m) {
    std::vector<double> out(m * m, 0.0);
    for (std::size_t col = 0; col < m; ++col) {
        for (std::size_t row = 0; row < m; ++row) {
            for (std::size_t k = 0; k < m; ++k) {
                out[row + col * m] += a[row + k * m] * b[k + col * m];
            }
        }
    }
    return out;
}

bool matches(const std::vector<float>& actual, const std::vector<double>& expected) {
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            std::fabs(double(actual[i]) - expected[i]) > 1e-4 + 2e-4 * std::fabs(expected[i])) {
            return false;
        }
    }
    return true;
}
} // namespace

testcase::Result testcase::cublaslt_matmul(const Config& config) {
    if (config.size == 0 || config.size > 1024 || config.iterations < 1) {
        throw std::invalid_argument("cuBLASLt requires size 1..1024 and positive iterations");
    }
    const std::size_t n = config.size * config.size, bytes = n * sizeof(float);
    const std::size_t workspace_bytes = 4 * 1024 * 1024;
    std::vector<float> a(n), b(n), out(n, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = input(i, config.seed) / 16.0f;
        b[i] = input(i, config.seed + 7) / 16.0f;
    }
    std::vector<double> reference_a(a.begin(), a.end()), reference_b(b.begin(), b.end());
    const auto first = reference(reference_a, reference_b, config.size);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* intermediate = nullptr;
    float* result = nullptr;
    unsigned char* workspace = nullptr;
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t algorithm{};
    Result verdict{true, "cuBLASLt A*B verified"};
    std::exception_ptr failure;

    try {
        check_cuda(cudaMalloc(&device_a, bytes));
        check_cuda(cudaMalloc(&device_b, bytes));
        check_cuda(cudaMalloc(&intermediate, bytes));
        // Keep the second output allocation in the single-GEMM workload as well.
        check_cuda(cudaMalloc(&result, bytes));
        check_cuda(cudaMalloc(&workspace, workspace_bytes));
        lt_check(cublasLtCreate(&handle));
        lt_check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F));
        // All matrices are square, column-major, with no transposition.
        lt_check(
            cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, config.size, config.size, config.size));
        lt_check(cublasLtMatmulPreferenceCreate(&preference));
        lt_check(cublasLtMatmulPreferenceSetAttribute(preference,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_bytes, sizeof(workspace_bytes)));
        int count = 0;
        lt_check(cublasLtMatmulAlgoGetHeuristic(handle, operation, layout, layout, layout, layout,
                                                preference, 1, &algorithm, &count));
        if (count == 0) {
            throw std::runtime_error("No supported cuBLASLt algorithm for this shape/device");
        }
        lt_check(algorithm.state);

        const float alpha = 1.0f, beta = 0.0f;
        check_cuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
            std::fill(out.begin(), out.end(), std::numeric_limits<float>::quiet_NaN());
            check_cuda(cudaMemcpy(intermediate, out.data(), bytes, cudaMemcpyHostToDevice));
            lt_check(cublasLtMatmul(handle, operation, &alpha, device_a, layout, device_b, layout,
                                    &beta, intermediate, layout, intermediate, layout,
                                    &algorithm.algo, workspace, workspace_bytes, 0));
            // The scheduler remaps streams; establish completion before consumption.
            check_cuda(cudaGetLastError());
            check_cuda(cudaDeviceSynchronize());

            check_cuda(cudaMemcpy(out.data(), intermediate, bytes, cudaMemcpyDeviceToHost));
            if (!matches(out, first)) {
                verdict = {false, "cuBLASLt first GEMM mismatch"};
                break;
            }
        }
        // On success, report free errors. On failure, preserve the original diagnostic.
        if (verdict.passed) {
            check_cuda(cudaFree(std::exchange(device_a, nullptr)));
            check_cuda(cudaFree(std::exchange(device_b, nullptr)));
            check_cuda(cudaFree(std::exchange(intermediate, nullptr)));
            check_cuda(cudaFree(std::exchange(result, nullptr)));
            check_cuda(cudaFree(std::exchange(workspace, nullptr)));
        }
    } catch (...) {
        failure = std::current_exception();
    }

    if (preference) {
        cublasLtMatmulPreferenceDestroy(preference);
    }
    if (layout) {
        cublasLtMatrixLayoutDestroy(layout);
    }
    if (operation) {
        cublasLtMatmulDescDestroy(operation);
    }
    if (handle) {
        cublasLtDestroy(handle);
    }
    if (workspace) {
        cudaFree(workspace);
    }
    if (result) {
        cudaFree(result);
    }
    if (intermediate) {
        cudaFree(intermediate);
    }
    if (device_b) {
        cudaFree(device_b);
    }
    if (device_a) {
        cudaFree(device_a);
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    return verdict;
}
