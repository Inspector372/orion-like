#pragma once
#include "testcase.h"

#ifndef ORION_ENABLE_LIBTORCH
#define ORION_ENABLE_LIBTORCH 0
#endif

#if ORION_ENABLE_LIBTORCH
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <torch/cuda.h>
#include <stdexcept>

namespace testcase { namespace lt {
inline void require_cuda() {
    if (!torch::cuda::is_available())
        throw std::runtime_error("LibTorch CUDA is unavailable; use a CUDA-enabled LibTorch distribution");
}

// Host initialization avoids shared RNG state between client threads and does
// not explicitly invoke any GPU memset. Library internals may still do so.
inline at::Tensor input(at::IntArrayRef shape, std::uint32_t seed) {
    auto tensor = at::empty(shape, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    auto* values = tensor.data_ptr<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        values[i] = (static_cast<int>((seed >> 24) & 31u) - 16) / 128.0f;
    }
    return tensor;
}

inline Result check(const at::Tensor& output, const at::Tensor& expected,
                    int iteration, const char* name) {
    // Device-wide synchronization also surfaces asynchronous CUDA errors.
    torch::cuda::synchronize();
    auto actual = output.to(at::kCPU);
    if (!at::isfinite(actual).all().item<bool>() ||
        !at::allclose(actual, expected, 2e-3, 2e-4)) {
        const float error = (actual - expected).abs().max().item<float>();
        return {false, std::string(name) + " mismatch at iteration " +
                       std::to_string(iteration) + "; max absolute error=" + std::to_string(error)};
    }
    return {true, std::string(name) + " CUDA output matches CPU reference"};
}
}}
#endif
