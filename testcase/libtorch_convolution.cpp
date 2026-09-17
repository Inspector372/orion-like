#include "libtorch_common.h"
#if ORION_ENABLE_LIBTORCH
#include <ATen/ops/cudnn_convolution.h>
#include <vector>
#endif

namespace testcase {
Result libtorch_convolution(const Config& config) {
#if ORION_ENABLE_LIBTORCH
    if (!config.size || config.size > 128 || config.iterations < 1)
        return {false, "libtorch_convolution requires size in [1,128] and positive iterations"};
    lt::require_cuda();
    if (!torch::cuda::cudnn_is_available())
        return {false, "LibTorch cuDNN support is unavailable"};
    c10::InferenceMode inference;
    const int64_t n = static_cast<int64_t>(config.size);
    const auto x = lt::input({2, 8, n, n}, config.seed);
    const auto weight = lt::input({16, 8, 3, 3}, config.seed + 1);
    const std::vector<int64_t> padding{1,1}, stride{1,1}, dilation{1,1};
    const auto expected = at::relu(at::conv2d(x, weight, {}, stride, padding, dilation, 1));
    const auto gx = x.to(at::kCUDA), gw = weight.to(at::kCUDA);
    Result result{false, "No iterations executed"};
    for (int i = 0; i < config.iterations; ++i) {
        // Explicit LibTorch cuDNN operator prevents a silent convolution
        // backend fallback. No benchmark search, TF32, or backward pass.
        const auto convolution = at::cudnn_convolution(
            gx, gw, padding, stride, dilation, 1, false, true, false);
        result = lt::check(at::relu(convolution), expected, i, "LibTorch cuDNN Conv2d-ReLU");
        if (!result.passed) return result;
    }
    return result;
#else
    (void)config;
    return {false, "LibTorch testcases disabled; rebuild with ENABLE_LIBTORCH=1"};
#endif
}
}
