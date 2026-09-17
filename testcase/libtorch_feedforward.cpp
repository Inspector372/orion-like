#include "libtorch_common.h"

namespace testcase {
Result libtorch_feedforward(const Config& config) {
#if ORION_ENABLE_LIBTORCH
    if (!config.size || config.size > 512 || config.iterations < 1)
        return {false, "libtorch_feedforward requires size in [1,512] and positive iterations"};
    lt::require_cuda();
    c10::InferenceMode inference;
    const int64_t n = static_cast<int64_t>(config.size);
    const auto x = lt::input({8, n}, config.seed);
    const auto w1 = lt::input({2*n, n}, config.seed + 1);
    const auto b1 = lt::input({2*n}, config.seed + 2);
    const auto w2 = lt::input({n, 2*n}, config.seed + 3);
    const auto b2 = lt::input({n}, config.seed + 4);
    const auto expected = at::linear(at::relu(at::linear(x, w1, b1)), w2, b2);
    const auto gx = x.to(at::kCUDA), gw1 = w1.to(at::kCUDA), gb1 = b1.to(at::kCUDA);
    const auto gw2 = w2.to(at::kCUDA), gb2 = b2.to(at::kCUDA);
    Result result{false, "No iterations executed"};
    for (int i = 0; i < config.iterations; ++i) {
        const auto output = at::linear(at::relu(at::linear(gx, gw1, gb1)), gw2, gb2);
        result = lt::check(output, expected, i, "LibTorch Linear-ReLU-Linear");
        if (!result.passed) return result;
    }
    return result;
#else
    (void)config;
    return {false, "LibTorch testcases disabled; rebuild with ENABLE_LIBTORCH=1"};
#endif
}
}
