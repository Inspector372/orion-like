#include "common.cuh"
#include <algorithm>


// This test case tests handling implicit kernel call done by cudaMemset() and cudaMemsetAsync().
// Each round fills data, then runs two dependent kernels. Explicit device
// synchronization bridges the original stream and scheduler-remapped streams.
// This tests internal fill launches and correctness, not asynchronous overlap.
// size = unsigned elements; iterations * work = rounds (default 256 rounds).
namespace {
__global__ void inspect_fill(unsigned* data, unsigned* visits, unsigned* errors,
                             std::size_t n, unsigned expected) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (data[i] != expected) atomicAdd(errors + i, 1u);
        data[i] = expected ^ 0xa5a5a5a5u;
        atomicAdd(visits + i, 1u);
    }
}
__global__ void inspect_transform(const unsigned* data, unsigned* visits,
                                  unsigned* errors, std::size_t n,
                                  unsigned expected) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (data[i] != (expected ^ 0xa5a5a5a5u)) atomicAdd(errors + i, 1u);
        atomicAdd(visits + i, 1u);
    }
}
}
testcase::Result testcase::memset(const Config& c) {
    const auto bytes = c.size * sizeof(unsigned);
    std::vector<unsigned> visits(c.size, 0u), errors(c.size, 0u);
    Buffer<unsigned> data(c.size), counts(c.size), failures(c.size);
    check(cudaMemcpy(counts.ptr, visits.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(failures.ptr, errors.data(), bytes, cudaMemcpyHostToDevice));
    for (int iteration = 0; iteration < c.iterations; ++iteration) {
        for (int round = 0; round < c.work; ++round) {
            unsigned byte = (c.seed + static_cast<unsigned>(iteration) +
                             static_cast<unsigned>(round)) & 255u;
            unsigned expected = byte * 0x01010101u;
            // Alternate synchronous and asynchronous memset entry points.
            if ((round & 1) == 0)
                check(cudaMemset(data.ptr, static_cast<int>(byte), bytes));
            else
                check(cudaMemsetAsync(data.ptr, static_cast<int>(byte), bytes, 0));
            finish();
            inspect_fill<<<blocks(c.size), 256>>>(data.ptr, counts.ptr,
                                                 failures.ptr, c.size, expected);
            finish();
            inspect_transform<<<blocks(c.size), 256>>>(data.ptr, counts.ptr,
                                                      failures.ptr, c.size, expected);
            finish();
            // Verify every round so a later memset cannot hide an earlier failure.
            check(cudaMemcpy(visits.data(), counts.ptr, bytes, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(errors.data(), failures.ptr, bytes, cudaMemcpyDeviceToHost));
            const unsigned expected_visits = 2u * (static_cast<unsigned>(round) + 1u);
            for (std::size_t i = 0; i < c.size; ++i) {
                if (errors[i] != 0 || visits[i] != expected_visits)
                    return {false, "Memset/chain mismatch at element " + std::to_string(i) +
                                   ", iteration " + std::to_string(iteration) +
                                   ", round " + std::to_string(round)};
            }
        }
        // Reset only the visit counter between iterations, using a host copy.
        std::fill(visits.begin(), visits.end(), 0u);
        check(cudaMemcpy(counts.ptr, visits.data(), bytes, cudaMemcpyHostToDevice));
    }
    data.release(); counts.release(); failures.release();
    return {true, "Repeated memset and dependent kernels verified"};
}
