#include "testcase.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include <algorithm>

// This test case tests handling implicit kernel call done by cudaMemset() and cudaMemsetAsync().
// Each round fills data, then runs two dependent kernels. Explicit device
// synchronization bridges the original stream and scheduler-remapped streams.
// This tests internal fill launches and correctness, not asynchronous overlap.
// size = unsigned elements; iterations * work = rounds (default 256 rounds).
namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__global__ void inspect_fill(unsigned* data, unsigned* visits, unsigned* errors, std::size_t n,
                             unsigned expected) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (data[i] != expected) {
            atomicAdd(errors + i, 1u);
        }
        data[i] = expected ^ 0xa5a5a5a5u;
        atomicAdd(visits + i, 1u);
    }
}

__global__ void inspect_transform(const unsigned* data, unsigned* visits, unsigned* errors,
                                  std::size_t n, unsigned expected) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (data[i] != (expected ^ 0xa5a5a5a5u)) {
            atomicAdd(errors + i, 1u);
        }
        atomicAdd(visits + i, 1u);
    }
}
} // namespace

testcase::Result testcase::memset(const Config& config) {
    const auto bytes = config.size * sizeof(unsigned);
    std::vector<unsigned> visits(config.size, 0u), errors(config.size, 0u);
    unsigned* device_data = nullptr;
    unsigned* device_visits = nullptr;
    unsigned* device_errors = nullptr;
    const unsigned grid_size = static_cast<unsigned>((config.size + 255) / 256);

    try {
        check_cuda(cudaMalloc(&device_data, config.size * sizeof(unsigned)));
        check_cuda(cudaMalloc(&device_visits, config.size * sizeof(unsigned)));
        check_cuda(cudaMalloc(&device_errors, config.size * sizeof(unsigned)));
        check_cuda(cudaMemcpy(device_visits, visits.data(), bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(device_errors, errors.data(), bytes, cudaMemcpyHostToDevice));
        for (int iteration = 0; iteration < config.iterations; ++iteration) {
            for (int round = 0; round < config.work; ++round) {
                unsigned byte = (config.seed + static_cast<unsigned>(iteration) +
                                 static_cast<unsigned>(round)) &
                                255u;
                unsigned expected = byte * 0x01010101u;
                // Alternate synchronous and asynchronous memset entry points.
                if ((round & 1) == 0) {
                    check_cuda(cudaMemset(device_data, static_cast<int>(byte), bytes));
                } else {
                    check_cuda(cudaMemsetAsync(device_data, static_cast<int>(byte), bytes, 0));
                }
                check_cuda(cudaGetLastError());
                check_cuda(cudaDeviceSynchronize());
                inspect_fill<<<grid_size, 256>>>(device_data, device_visits, device_errors,
                                                 config.size, expected);
                check_cuda(cudaGetLastError());
                check_cuda(cudaDeviceSynchronize());
                inspect_transform<<<grid_size, 256>>>(device_data, device_visits, device_errors,
                                                      config.size, expected);
                check_cuda(cudaGetLastError());
                check_cuda(cudaDeviceSynchronize());
                // Verify every round so a later memset cannot hide an earlier failure.
                check_cuda(cudaMemcpy(visits.data(), device_visits, bytes, cudaMemcpyDeviceToHost));
                check_cuda(cudaMemcpy(errors.data(), device_errors, bytes, cudaMemcpyDeviceToHost));
                const unsigned expected_visits = 2u * (static_cast<unsigned>(round) + 1u);
                for (std::size_t i = 0; i < config.size; ++i) {
                    if (errors[i] != 0 || visits[i] != expected_visits) {
                        Result mismatch{false, "Memset/chain mismatch at element " +
                                                   std::to_string(i) + ", iteration " +
                                                   std::to_string(iteration) + ", round " +
                                                   std::to_string(round)};
                        cudaFree(std::exchange(device_errors, nullptr));
                        cudaFree(std::exchange(device_visits, nullptr));
                        cudaFree(std::exchange(device_data, nullptr));
                        return mismatch;
                    }
                }
            }
            // Reset only the visit counter between iterations, using a host copy.
            std::fill(visits.begin(), visits.end(), 0u);
            check_cuda(cudaMemcpy(device_visits, visits.data(), bytes, cudaMemcpyHostToDevice));
        }
        check_cuda(cudaFree(std::exchange(device_data, nullptr)));
        check_cuda(cudaFree(std::exchange(device_visits, nullptr)));
        check_cuda(cudaFree(std::exchange(device_errors, nullptr)));
        return {true, "Repeated memset and dependent kernels verified"};
    } catch (...) {
        if (device_errors) {
            cudaFree(device_errors);
        }
        if (device_visits) {
            cudaFree(device_visits);
        }
        if (device_data) {
            cudaFree(device_data);
        }
        throw;
    }
}
