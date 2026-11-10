#include "testcase.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__global__ void initialize(unsigned* out, std::size_t n, unsigned seed) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<unsigned>(i) + seed;
    }
}

__global__ void transform(unsigned* out, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = out[i] * 3u + 1u;
    }
}
} // namespace

testcase::Result testcase::chained(const Config& config) {
    unsigned* device_output = nullptr;
    const unsigned grid_size = static_cast<unsigned>((config.size + 255) / 256);

    try {
        check_cuda(cudaMalloc(&device_output, config.size * sizeof(unsigned)));
        std::vector<unsigned> out(config.size, 0u);
        check_cuda(cudaMemcpy(device_output, out.data(), config.size * sizeof(unsigned),
                              cudaMemcpyHostToDevice));
        initialize<<<grid_size, 256>>>(device_output, config.size, config.seed);
        check_cuda(cudaGetLastError());
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
            transform<<<grid_size, 256>>>(device_output, config.size);
            check_cuda(cudaGetLastError());
        }
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
        check_cuda(cudaMemcpy(out.data(), device_output, config.size * sizeof(unsigned),
                              cudaMemcpyDeviceToHost));
        bool ok = true;
        for (std::size_t i = 0; i < config.size; ++i) {
            unsigned expected = static_cast<unsigned>(i) + config.seed;
            for (int iteration = 0; iteration < config.iterations; ++iteration) {
                expected = expected * 3u + 1u;
            }
            ok &= out[i] == expected;
        }
        check_cuda(cudaFree(std::exchange(device_output, nullptr)));
        return {ok, ok ? "Dependent chain verified" : "Kernel ordering/output mismatch"};
    } catch (...) {
        if (device_output) {
            cudaFree(device_output);
        }
        throw;
    }
}
