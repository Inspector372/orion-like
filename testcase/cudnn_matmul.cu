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

std::vector<double> cpu_matmul(const std::vector<double>& a, const std::vector<double>& b, int m) {
    std::vector<double> y(m * m, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                y[i * m + j] += a[i * m + k] * b[k * m + j];
            }
        }
    }
    return y;
}
} // namespace
#endif

testcase::Result testcase::cudnn_matmul(const Config& config) {
#if ORION_ENABLE_CUDNN
    if (!config.size || config.size > 512 || config.size % 8 != 0 || config.iterations < 1) {
        throw std::invalid_argument("Invalid cuDNN workload size/iterations");
    }
    int64_t m = config.size;
    auto a = make_input(m * m, config.seed), b = make_input(m * m, config.seed + 1);
    auto expected = cpu_matmul(as_doubles(a), as_doubles(b), m);
    __half* device_a = nullptr;
    __half* device_b = nullptr;
    __half* device_output = nullptr;
    unsigned char* workspace = nullptr;
    cudnnHandle_t handle = nullptr;

    try {
        check_cuda(cudaMalloc(&device_a, a.size() * sizeof(__half)));
        check_cuda(cudaMalloc(&device_b, b.size() * sizeof(__half)));
        check_cuda(cudaMalloc(&device_output, expected.size() * sizeof(__half)));
        check_cuda(
            cudaMemcpy(device_a, a.data(), a.size() * sizeof(__half), cudaMemcpyHostToDevice));
        check_cuda(
            cudaMemcpy(device_b, b.data(), b.size() * sizeof(__half), cudaMemcpyHostToDevice));

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
        auto A = tensor("A", {1, m, m}, {m * m, m, 1});
        auto B = tensor("B", {1, m, m}, {m * m, m, 1});
        auto Y = graph.matmul(A, B, fe::graph::Matmul_attributes());
        Y->set_output(true).set_dim({1, m, m}).set_stride({m * m, m, 1});
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tensors = {
            {A, device_a}, {B, device_b}, {Y, device_output}};

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

        Result verdict{true, "cuDNN matmul verified against CPU reference"};
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
                    verdict = {false,
                               std::string("cuDNN matmul") + " mismatch at " + std::to_string(i) +
                                   " iteration " + std::to_string(iteration) + ": expected " +
                                   std::to_string(expected[i]) + ", got " + std::to_string(got)};
                    break;
                }
            }
        }

        cudaError_t workspace_status = cudaFree(std::exchange(workspace, nullptr));
        if (verdict.passed) {
            check_cuda(workspace_status);
        }
        check_cuda(cudaFree(std::exchange(device_a, nullptr)));
        check_cuda(cudaFree(std::exchange(device_b, nullptr)));
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
        if (device_b) {
            cudaFree(device_b);
        }
        if (device_a) {
            cudaFree(device_a);
        }
        throw;
    }
#else
    (void)config;
    return {false, "cuDNN workloads disabled: rebuild with ENABLE_CUDNN=1 "
                   "CUDNN_FRONTEND_DIR=/path/to/cudnn-frontend"};
#endif
}
