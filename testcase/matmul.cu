#include "testcase.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include <limits>

namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

float input(std::size_t i, std::uint32_t seed) {
    return static_cast<float>((i + seed) % 17) * 0.25f;
}

__global__ void multiply(const float* a, const float* b, float* out, std::size_t m) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * m) {
        return;
    }
    std::size_t row = i / m, col = i % m;
    float sum = 0;
    for (std::size_t k = 0; k < m; ++k) {
        sum += a[row * m + k] * b[k * m + col];
    }
    out[i] = sum;
}
} // namespace

testcase::Result testcase::matmul(const Config& config) {
    const auto n = config.size * config.size, bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), out(n, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = input(i, config.seed);
        b[i] = input(i, config.seed + 7);
    }
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_output = nullptr;
    const unsigned grid_size = static_cast<unsigned>((n + 255) / 256);

    try {
        check_cuda(cudaMalloc(&device_a, n * sizeof(float)));
        check_cuda(cudaMalloc(&device_b, n * sizeof(float)));
        check_cuda(cudaMalloc(&device_output, n * sizeof(float)));
        check_cuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));
        // Seed output with NaNs so missing writes fail verification, without a memset.
        check_cuda(cudaMemcpy(device_output, out.data(), bytes, cudaMemcpyHostToDevice));
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
            multiply<<<grid_size, 256>>>(device_a, device_b, device_output, config.size);
            check_cuda(cudaGetLastError());
        }
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(out.data(), device_output, bytes, cudaMemcpyDeviceToHost));
        bool ok = true;
        // Binary-exact inputs and bounded dimensions permit exact comparison.
        for (std::size_t row = 0; row < config.size; ++row) {
            for (std::size_t col = 0; col < config.size; ++col) {
                float expected = 0;
                for (std::size_t k = 0; k < config.size; ++k) {
                    expected += a[row * config.size + k] * b[k * config.size + col];
                }
                ok &= out[row * config.size + col] == expected;
            }
        }
        check_cuda(cudaFree(std::exchange(device_a, nullptr)));
        check_cuda(cudaFree(std::exchange(device_b, nullptr)));
        check_cuda(cudaFree(std::exchange(device_output, nullptr)));
        return {ok, ok ? "1D matmul verified" : "Matmul mismatch"};
    } catch (...) {
        if (device_output) {
            cudaFree(device_output);
        }
        if (device_b) {
            cudaFree(device_b);
        }
        if (device_a) {
            cudaFree(device_a);
        }
        throw;
    }
}
