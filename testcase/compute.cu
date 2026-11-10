#include "testcase.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include <string>

namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__host__ __device__ unsigned recurrence(unsigned x, int work) {
    for (int k = 0; k < work; ++k) {
        x = (x * 1664525u + 1013904223u) ^ (x >> 7);
    }
    return x;
}

__global__ void calculate(unsigned* out, std::size_t n, int work, unsigned seed) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = recurrence(static_cast<unsigned>(i) + seed, work);
    }
}
} // namespace

testcase::Result testcase::compute(const Config& config) {
    unsigned* device_output = nullptr;
    const unsigned grid_size = static_cast<unsigned>((config.size + 255) / 256);

    try {
        check_cuda(cudaMalloc(&device_output, config.size * sizeof(unsigned)));
        std::vector<unsigned> out(config.size, 0u);
        check_cuda(cudaMemcpy(device_output, out.data(), config.size * sizeof(unsigned),
                              cudaMemcpyHostToDevice));
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
            calculate<<<grid_size, 256>>>(device_output, config.size, config.work, config.seed);
            check_cuda(cudaGetLastError());
        }
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(out.data(), device_output, config.size * sizeof(unsigned),
                              cudaMemcpyDeviceToHost));
        check_cuda(cudaFree(std::exchange(device_output, nullptr)));
        for (std::size_t i = 0; i < config.size; ++i) {
            const unsigned expected =
                recurrence(static_cast<unsigned>(i) + config.seed, config.work);
            if (out[i] != expected) {
                return {false, "Compute first mismatch at index " + std::to_string(i) + " (block " +
                                   std::to_string(i / 256) + ", thread " + std::to_string(i % 256) +
                                   "): expected=" + std::to_string(expected) +
                                   ", actual=" + std::to_string(out[i])};
            }
        }
        return {true, "Compute recurrence verified"};
    } catch (...) {
        if (device_output) {
            cudaFree(device_output);
        }
        throw;
    }
}
