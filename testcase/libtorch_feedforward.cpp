#include "testcase.h"

#if ORION_ENABLE_LIBTORCH
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <torch/cuda.h>
#include <stdexcept>

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
Result libtorch_feedforward(const Config& config) {
#if ORION_ENABLE_LIBTORCH
    if (!config.size || config.size > 512 || config.iterations < 1) {
        return {false, "libtorch_feedforward requires size in [1,512] and positive iterations"};
    }
    if (!torch::cuda::is_available()) {
        throw std::runtime_error(
            "LibTorch CUDA is unavailable; use a CUDA-enabled LibTorch distribution");
    }
    c10::InferenceMode inference;
    const int64_t n = static_cast<int64_t>(config.size);
    const auto x = host_input({8, n}, config.seed);
    const auto w1 = host_input({2 * n, n}, config.seed + 1);
    const auto b1 = host_input({2 * n}, config.seed + 2);
    const auto w2 = host_input({n, 2 * n}, config.seed + 3);
    const auto b2 = host_input({n}, config.seed + 4);
    const auto expected = at::linear(at::relu(at::linear(x, w1, b1)), w2, b2);
    const auto gx = x.to(at::kCUDA), gw1 = w1.to(at::kCUDA), gb1 = b1.to(at::kCUDA);
    const auto gw2 = w2.to(at::kCUDA), gb2 = b2.to(at::kCUDA);
    for (int i = 0; i < config.iterations; ++i) {
        const auto output = at::linear(at::relu(at::linear(gx, gw1, gb1)), gw2, gb2);
        // Device-wide synchronization also surfaces asynchronous CUDA errors.
        torch::cuda::synchronize();
        auto actual = output.to(at::kCPU);
        if (!at::isfinite(actual).all().item<bool>() ||
            !at::allclose(actual, expected, 2e-3, 2e-4)) {
            const float error = (actual - expected).abs().max().item<float>();
            return {false, std::string("LibTorch Linear-ReLU-Linear") + " mismatch at iteration " +
                               std::to_string(i) + "; max absolute error=" + std::to_string(error)};
        }
    }
    return {true, "LibTorch Linear-ReLU-Linear CUDA output matches CPU reference"};
#else
    (void)config;
    return {false, "LibTorch testcases disabled; rebuild with ENABLE_LIBTORCH=1"};
#endif
}
} // namespace testcase
