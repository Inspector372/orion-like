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

__global__ void add(const float* a, const float* b, float* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
} // namespace

testcase::Result testcase::vector_add(const Config& config) {
    std::vector<float> a(config.size), b(config.size),
        out(config.size, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t i = 0; i < config.size; ++i) {
        a[i] = input(i, config.seed);
        b[i] = input(i, config.seed + 7);
    }
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_output = nullptr;
    const unsigned grid_size = static_cast<unsigned>((config.size + 255) / 256);

    try {
        check_cuda(cudaMalloc(&device_a, config.size * sizeof(float)));
        check_cuda(cudaMalloc(&device_b, config.size * sizeof(float)));
        check_cuda(cudaMalloc(&device_output, config.size * sizeof(float)));
        const auto bytes = config.size * sizeof(float);
        check_cuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device_output, out.data(), bytes, cudaMemcpyHostToDevice));
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
            add<<<grid_size, 256>>>(device_a, device_b, device_output, config.size);
            check_cuda(cudaGetLastError());
        }
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(out.data(), device_output, bytes, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (std::size_t i = 0; i < config.size; ++i) {
            ok &= out[i] == a[i] + b[i];
        }
        check_cuda(cudaFree(std::exchange(device_a, nullptr)));
        check_cuda(cudaFree(std::exchange(device_b, nullptr)));
        check_cuda(cudaFree(std::exchange(device_output, nullptr)));
        return {ok, ok ? "Vector addition verified" : "Vector addition mismatch"};
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
