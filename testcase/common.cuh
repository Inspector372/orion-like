#pragma once
#include "testcase.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace testcase {
inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> class Buffer {
public:
    T* ptr = nullptr;
    explicit Buffer(std::size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T))); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void release() { T* p = ptr; ptr = nullptr; check(cudaFree(p)); }
};
inline unsigned blocks(std::size_t n) { return static_cast<unsigned>((n + 255) / 256); }
inline void finish() { check(cudaGetLastError()); check(cudaDeviceSynchronize()); }
inline float input(std::size_t i, std::uint32_t seed) {
    return static_cast<float>((i + seed) % 17) * 0.25f;
}
}
