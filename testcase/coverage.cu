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

__global__ void visit(unsigned* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(out + i, 1u);
    }
}
} // namespace

testcase::Result testcase::coverage(const Config& config) {
    unsigned* device_output = nullptr;
    const unsigned grid_size = static_cast<unsigned>((config.size + 255) / 256);

    try {
        check_cuda(cudaMalloc(&device_output, config.size * sizeof(unsigned)));
        std::vector<unsigned> visits(config.size, 0u);
        check_cuda(cudaMemcpy(device_output, visits.data(), config.size * sizeof(unsigned),
                              cudaMemcpyHostToDevice));
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
            visit<<<grid_size, 256>>>(device_output, config.size);
            check_cuda(cudaGetLastError());
        }
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(visits.data(), device_output, config.size * sizeof(unsigned),
                              cudaMemcpyDeviceToHost));
        check_cuda(cudaFree(std::exchange(device_output, nullptr)));
        const unsigned expected = static_cast<unsigned>(config.iterations);
        for (std::size_t i = 0; i < config.size; ++i) {
            if (visits[i] != expected) {
                return {false, "Coverage first mismatch at index " + std::to_string(i) +
                                   " (block " + std::to_string(i / 256) + ", thread " +
                                   std::to_string(i % 256) +
                                   "): expected=" + std::to_string(expected) +
                                   ", actual=" + std::to_string(visits[i])};
            }
        }
        return {true, "Each logical thread executed exactly once per repetition"};
    } catch (...) {
        if (device_output) {
            cudaFree(device_output);
        }
        throw;
    }
}
