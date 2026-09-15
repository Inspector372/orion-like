#pragma once
#include "testcase.h"
#ifndef ORION_ENABLE_CUDNN
#define ORION_ENABLE_CUDNN 0
#endif
#if ORION_ENABLE_CUDNN
#include "common.cuh"
#include "cudnn_reference.h"
#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>

namespace testcase { namespace dn {
namespace fe = cudnn_frontend;
using Tensor = std::shared_ptr<fe::graph::Tensor_attributes>;
using Pack = std::unordered_map<Tensor, void*>;
template<class Status> inline void status(Status e, const char* phase) {
    if (!e.is_good()) throw std::runtime_error(std::string("cuDNN ")+phase+": "+e.get_message());
}
inline void api(cudnnStatus_t s) {
    if (s != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(s));
}
struct Context {
    cudnnHandle_t handle=nullptr;
    fe::graph::Graph graph;
    Context() {
        if (cudnnGetVersion()<90000) throw std::runtime_error("These cuDNN graph workloads require cuDNN 9.x or newer");
        int device=0; check(cudaGetDevice(&device));
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,device));
        if (prop.major<8) throw std::runtime_error("These FP16 cuDNN graph workloads target Ampere (SM80) or newer GPUs");
        graph.set_io_data_type(fe::DataType_t::HALF)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);
        api(cudnnCreate(&handle));
    }
    ~Context() { if(handle) cudnnDestroy(handle); }
    Context(const Context&)=delete;
    Context& operator=(const Context&)=delete;
    Tensor tensor(const char* name, std::vector<int64_t> dim, std::vector<int64_t> stride,
                  fe::DataType_t dtype=fe::DataType_t::HALF) {
        return graph.tensor(fe::graph::Tensor_attributes().set_name(name).set_dim(dim)
                            .set_stride(stride).set_data_type(dtype));
    }
    std::size_t build() {
        status(graph.validate(),"validate");
        status(graph.build_operation_graph(handle),"operation graph");
        status(graph.create_execution_plans({fe::HeurMode_t::A,fe::HeurMode_t::FALLBACK}),"heuristics");
        status(graph.check_support(handle),"device/shape support");
        status(graph.build_plans(handle),"build plans");
        int64_t bytes=0; status(graph.get_workspace_size(bytes),"workspace size");
        if(bytes<0) throw std::runtime_error("Invalid cuDNN workspace size");
        return std::max<std::size_t>(1,static_cast<std::size_t>(bytes));
    }
};
inline std::vector<__half> values(std::size_t n, std::uint32_t seed) {
    std::vector<__half> x(n);
    // Binary-exact values: CPU references use precisely the stored FP16 inputs.
    std::uint32_t state=seed;
    for(auto& v:x) {
        state=1664525u*state+1013904223u;
        v=__float2half_rn((static_cast<int>((state>>16)%33)-16)/32.0f);
    }
    return x;
}
inline std::vector<double> doubles(const std::vector<__half>& x) {
    std::vector<double> y(x.size());
    for(std::size_t i=0;i<x.size();++i) y[i]=__half2float(x[i]);
    return y;
}
template<class T> inline void upload(Buffer<T>& d,const std::vector<T>& h) {
    check(cudaMemcpy(d.ptr,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));
}
inline Result execute(Context& ctx, Pack pack, Buffer<__half>& out,
                      const std::vector<double>& expected, int iterations,
                      const char* name) {
    Buffer<unsigned char> workspace(ctx.build());
    std::vector<__half> h(expected.size());
    for(int r=0;r<iterations;++r) {
        std::fill(h.begin(),h.end(),__float2half_rn(std::numeric_limits<float>::quiet_NaN()));
        upload(out,h); // No explicit memset in cuDNN workloads.
        status(ctx.graph.execute(ctx.handle,pack,workspace.ptr),"execute");
        finish(); // Original stream and scheduler-remapped streams must complete.
        check(cudaMemcpy(h.data(),out.ptr,h.size()*sizeof(__half),cudaMemcpyDeviceToHost));
        for(std::size_t i=0;i<h.size();++i) {
            double got=__half2float(h[i]);
            // Allow FP16 output rounding and fused FP32 accumulation differences.
            if(!std::isfinite(got) || std::fabs(got-expected[i])>0.005+0.005*std::fabs(expected[i]))
                return {false,std::string(name)+" mismatch at "+std::to_string(i)+
                    " iteration "+std::to_string(r)+": expected "+std::to_string(expected[i])+
                    ", got "+std::to_string(got)};
        }
    }
    workspace.release();
    return {true,std::string(name)+" verified against CPU reference"};
}
inline void validate(const Config& c, std::size_t max, std::size_t multiple=1) {
    if(!c.size || c.size>max || c.size%multiple || c.iterations<1)
        throw std::invalid_argument("Invalid cuDNN workload size/iterations");
}
} }
#else
namespace testcase { namespace dn {
inline Result disabled() {
    return {false,"cuDNN workloads disabled: rebuild with ENABLE_CUDNN=1 CUDNN_FRONTEND_DIR=/path/to/cudnn-frontend"};
}
} }
#endif
