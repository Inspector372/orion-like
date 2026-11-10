#include "testcase.h"

#if ORION_ENABLE_CUDNN
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>

namespace {
namespace fe = cudnn_frontend;
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void check_graph(fe::error_t status, const char* phase) {
    if (!status.is_good()) {
        throw std::runtime_error(std::string("cuDNN ") + phase + ": " + status.get_message());
    }
}

std::vector<__half> make_input(std::size_t n, std::uint32_t seed) {
    std::vector<__half> x(n);
    // Binary-exact values: CPU references use precisely the stored FP16 inputs.
    std::uint32_t state = seed;
    for (auto& v : x) {
        state = 1664525u * state + 1013904223u;
        v = __float2half_rn((static_cast<int>((state >> 16) % 33) - 16) / 32.0f);
    }
    return x;
}

std::vector<double> as_doubles(const std::vector<__half>& x) {
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = __half2float(x[i]);
    }
    return y;
}

std::vector<double> cpu_convolution(const std::vector<double>& x, const std::vector<double>& w,
                                    int side) {
    constexpr int channels = 8, outputs = 8;
    std::vector<double> y(side * side * outputs, 0.0);
    for (int h = 0; h < side; ++h) {
        for (int col = 0; col < side; ++col) {
            for (int k = 0; k < outputs; ++k) {
                for (int r = 0; r < 3; ++r) {
                    for (int s = 0; s < 3; ++s) {
                        int ih = h + r - 1, iw = col + s - 1;
                        if (ih < 0 || iw < 0 || ih >= side || iw >= side) {
                            continue;
                        }
                        for (int c = 0; c < channels; ++c) {
                            y[(h * side + col) * outputs + k] +=
                                x[(ih * side + iw) * channels + c] *
                                w[((k * 3 + r) * 3 + s) * channels + c];
                        }
                    }
                }
            }
        }
    }
    return y;
}
} // namespace
#endif

testcase::Result testcase::cudnn_convolution(const Config& config) {
#if ORION_ENABLE_CUDNN
    if (!config.size || config.size > 128 || config.iterations < 1) {
        throw std::invalid_argument("Invalid cuDNN workload size/iterations");
    }
    int64_t s = config.size, channels = 8;
    auto x = make_input(s * s * channels, config.seed),
         w = make_input(8 * 3 * 3 * 8, config.seed + 1);
    auto expected = cpu_convolution(as_doubles(x), as_doubles(w), s);
    __half* device_input = nullptr;
    __half* device_filter = nullptr;
    __half* device_output = nullptr;
    unsigned char* workspace = nullptr;
    cudnnHandle_t handle = nullptr;

    try {
        check_cuda(cudaMalloc(&device_input, x.size() * sizeof(__half)));
        check_cuda(cudaMalloc(&device_filter, w.size() * sizeof(__half)));
        check_cuda(cudaMalloc(&device_output, expected.size() * sizeof(__half)));
        check_cuda(
            cudaMemcpy(device_input, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice));
        check_cuda(
            cudaMemcpy(device_filter, w.data(), w.size() * sizeof(__half), cudaMemcpyHostToDevice));

        fe::graph::Graph graph;
        if (cudnnGetVersion() < 90000) {
            throw std::runtime_error("These cuDNN graph workloads require cuDNN 9.x or newer");
        }
        int device = 0;
        check_cuda(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        check_cuda(cudaGetDeviceProperties(&prop, device));
        if (prop.major < 8) {
            throw std::runtime_error(
                "These FP16 cuDNN graph workloads target Ampere (SM80) or newer GPUs");
        }
        graph.set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        if (auto status = cudnnCreate(&handle); status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error(cudnnGetErrorString(status));
        }

        auto tensor = [&](const char* name, std::vector<int64_t> dim, std::vector<int64_t> stride,
                          fe::DataType_t dtype = fe::DataType_t::HALF) {
            return graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name(name)
                                    .set_dim(dim)
                                    .set_stride(stride)
                                    .set_data_type(dtype));
        };
        auto X = tensor("X", {1, 8, s, s}, {s * s * 8, 1, s * 8, 8});
        auto W = tensor("W", {8, 8, 3, 3}, {72, 1, 24, 8});
        auto options =
            fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation(
                {1, 1});
        auto Y = graph.conv_fprop(X, W, options);
        Y->set_output(true).set_dim({1, 8, s, s}).set_stride({s * s * 8, 1, s * 8, 8});
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tensors = {
            {X, device_input}, {W, device_filter}, {Y, device_output}};

        check_graph(graph.validate(), "validate");
        check_graph(graph.build_operation_graph(handle), "operation graph");
        check_graph(graph.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
                    "heuristics");
        check_graph(graph.check_support(handle), "device/shape support");
        check_graph(graph.build_plans(handle), "build plans");
        int64_t bytes = 0;
        check_graph(graph.get_workspace_size(bytes), "workspace size");
        if (bytes < 0) {
            throw std::runtime_error("Invalid cuDNN workspace size");
        }
        const std::size_t workspace_bytes =
            std::max<std::size_t>(1, static_cast<std::size_t>(bytes));
        check_cuda(cudaMalloc(&workspace, workspace_bytes));

        Result verdict{true, "cuDNN convolution verified against CPU reference"};
        std::vector<__half> host_output(expected.size());
        for (int iteration = 0; iteration < config.iterations && verdict.passed; ++iteration) {
            std::fill(host_output.begin(), host_output.end(),
                      __float2half_rn(std::numeric_limits<float>::quiet_NaN()));
            check_cuda(
                cudaMemcpy(device_output, host_output.data(), host_output.size() * sizeof(__half),
                           cudaMemcpyHostToDevice)); // No explicit memset in cuDNN workloads.
            check_graph(graph.execute(handle, tensors, workspace), "execute");
            check_cuda(cudaGetLastError());
            check_cuda(cudaDeviceSynchronize());
            check_cuda(cudaMemcpy(host_output.data(), device_output,
                                  host_output.size() * sizeof(__half), cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < host_output.size(); ++i) {
                double got = __half2float(host_output[i]);
                // Allow FP16 output rounding and fused FP32 accumulation differences.
                if (!std::isfinite(got) ||
                    std::fabs(got - expected[i]) > 0.005 + 0.005 * std::fabs(expected[i])) {
                    verdict = {false, std::string("cuDNN convolution") + " mismatch at " +
                                          std::to_string(i) + " iteration " +
                                          std::to_string(iteration) + ": expected " +
                                          std::to_string(expected[i]) + ", got " +
                                          std::to_string(got)};
                    break;
                }
            }
        }

        cudaError_t workspace_status = cudaFree(std::exchange(workspace, nullptr));
        if (verdict.passed) {
            check_cuda(workspace_status);
        }
        check_cuda(cudaFree(std::exchange(device_input, nullptr)));
        check_cuda(cudaFree(std::exchange(device_filter, nullptr)));
        check_cuda(cudaFree(std::exchange(device_output, nullptr)));
        cudnnDestroy(handle);
        return verdict;
    } catch (...) {
        if (workspace) {
            cudaFree(workspace);
        }
        if (handle) {
            cudnnDestroy(handle);
        }
        if (device_output) {
            cudaFree(device_output);
        }
        if (device_filter) {
            cudaFree(device_filter);
        }
        if (device_input) {
            cudaFree(device_input);
        }
        throw;
    }
#else
    (void)config;
    return {false, "cuDNN workloads disabled: rebuild with ENABLE_CUDNN=1 "
                   "CUDNN_FRONTEND_DIR=/path/to/cudnn-frontend"};
#endif
}
