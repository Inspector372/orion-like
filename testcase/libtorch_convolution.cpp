#include "testcase.h"

#if ORION_ENABLE_LIBTORCH
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <torch/cuda.h>
#include <stdexcept>

#include <ATen/ops/cudnn_convolution.h>
#include <vector>

namespace {
// Host initialization avoids shared RNG state between client threads and does
// not explicitly invoke any GPU memset. Library internals may still do so.
at::Tensor host_input(at::IntArrayRef shape, std::uint32_t seed) {
    auto tensor = at::empty(shape, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    auto* values = tensor.data_ptr<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        values[i] = (static_cast<int>((seed >> 24) & 31u) - 16) / 128.0f;
    }
    return tensor;
}

} // namespace
#endif

namespace testcase {
Result libtorch_convolution(const Config& config) {
#if ORION_ENABLE_LIBTORCH
    if (!config.size || config.size > 128 || config.iterations < 1) {
        return {false, "libtorch_convolution requires size in [1,128] and positive iterations"};
    }
    if (!torch::cuda::is_available()) {
        throw std::runtime_error(
            "LibTorch CUDA is unavailable; use a CUDA-enabled LibTorch distribution");
    }
    if (!torch::cuda::cudnn_is_available()) {
        return {false, "LibTorch cuDNN support is unavailable"};
    }
    c10::InferenceMode inference;
    const int64_t n = static_cast<int64_t>(config.size);
    const auto x = host_input({2, 8, n, n}, config.seed);
    const auto weight = host_input({16, 8, 3, 3}, config.seed + 1);
    const std::vector<int64_t> padding{1, 1}, stride{1, 1}, dilation{1, 1};
    const auto expected = at::relu(at::conv2d(x, weight, {}, stride, padding, dilation, 1));
    const auto gx = x.to(at::kCUDA), gw = weight.to(at::kCUDA);
    for (int i = 0; i < config.iterations; ++i) {
        // Explicit LibTorch cuDNN operator prevents a silent convolution
        // backend fallback. No benchmark search, TF32, or backward pass.
        const auto convolution =
            at::cudnn_convolution(gx, gw, padding, stride, dilation, 1, false, true, false);
        const auto output = at::relu(convolution);
        // Device-wide synchronization also surfaces asynchronous CUDA errors.
        torch::cuda::synchronize();
        auto actual = output.to(at::kCPU);
        if (!at::isfinite(actual).all().item<bool>() ||
            !at::allclose(actual, expected, 2e-3, 2e-4)) {
            const float error = (actual - expected).abs().max().item<float>();
            return {false, std::string("LibTorch cuDNN Conv2d-ReLU") + " mismatch at iteration " +
                               std::to_string(i) + "; max absolute error=" + std::to_string(error)};
        }
    }
    return {true, "LibTorch cuDNN Conv2d-ReLU CUDA output matches CPU reference"};
#else
    (void)config;
    return {false, "LibTorch testcases disabled; rebuild with ENABLE_LIBTORCH=1"};
#endif
}
} // namespace testcase
