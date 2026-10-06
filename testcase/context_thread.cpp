// Build (CUDA 12.x):
// g++ -std=c++11 -pthread context_thread.cpp -I/usr/local/cuda/include \
//     -L/usr/local/cuda/lib64/stubs -lcuda -o context_thread
// Run: ./context_thread [device ordinal]
// The stub is for linking only; execution requires the real NVIDIA driver.
// Standalone diagnostic: LD_PRELOAD requires your hook's scheduler/registration
// to be initialized separately. This program does not initialize Orion.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(CUresult result, const char* call) {
    if (result == CUDA_SUCCESS) return;
    const char *name = nullptr, *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(call) + ": " +
        (name ? name : "unknown") + " / " + (message ? message : "unknown"));
}
#define CU(call) check((call), #call)

// init: out[i] = i; add: out[i] += delta. Both guard the final partial block.
static const char ptx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64
.visible .entry init(.param .u64 out, .param .u32 n) {
    .reg .pred p;
    .reg .b32 r<5>;
    .reg .b64 a<3>;
    ld.param.u64 a0, [out];
    ld.param.u32 r0, [n];
    mov.u32 r1, %ctaid.x;
    mov.u32 r2, %ntid.x;
    mov.u32 r3, %tid.x;
    mad.lo.u32 r4, r1, r2, r3;
    setp.ge.u32 p, r4, r0;
    @p bra DONE;
    mul.wide.u32 a1, r4, 4;
    add.u64 a2, a0, a1;
    st.global.u32 [a2], r4;
DONE:
    ret;
}
.visible .entry add(.param .u64 out, .param .u32 n, .param .u32 delta) {
    .reg .pred p;
    .reg .b32 r<7>;
    .reg .b64 a<3>;
    ld.param.u64 a0, [out];
    ld.param.u32 r0, [n];
    ld.param.u32 r5, [delta];
    mov.u32 r1, %ctaid.x;
    mov.u32 r2, %ntid.x;
    mov.u32 r3, %tid.x;
    mad.lo.u32 r4, r1, r2, r3;
    setp.ge.u32 p, r4, r0;
    @p bra DONE;
    mul.wide.u32 a1, r4, 4;
    add.u64 a2, a0, a1;
    ld.global.u32 r6, [a2];
    add.u32 r6, r6, r5;
    st.global.u32 [a2], r6;
DONE:
    ret;
}
)ptx";

static void worker(CUdevice device) {
    CUcontext previous = nullptr, context = nullptr;
    CUmodule module = nullptr;
    CUstream stream = nullptr;
    CUdeviceptr output = 0;
    CU(cuCtxGetCurrent(&previous));
    std::printf("worker initial context=%p\n", (void*)previous);
    try {
        CU(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
        // cuCtxCreate already makes it current. Explicitly unbind and switch
        // back to demonstrate that current-context binding belongs to a thread.
        CU(cuCtxSetCurrent(nullptr));
        CU(cuCtxSetCurrent(context));
        CUcontext current = nullptr;
        CU(cuCtxGetCurrent(&current));
        if (current != context) throw std::runtime_error("context switch failed");
        std::printf("worker new/current context=%p\n", (void*)current);
        CU(cuModuleLoadData(&module, ptx));
        CUfunction initialize, increment;
        CU(cuModuleGetFunction(&initialize, module, "init"));
        CU(cuModuleGetFunction(&increment, module, "add"));
        CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        unsigned n = 1025, delta = 7;
        CU(cuMemAlloc(&output, n * sizeof(unsigned)));
        void* init_args[] = {&output, &n};
        void* add_args[] = {&output, &n, &delta};
        CU(cuLaunchKernel(initialize, (n + 255) / 256, 1, 1,
                          256, 1, 1, 0, stream, init_args, nullptr));
        CU(cuLaunchKernel(increment, (n + 255) / 256, 1, 1,
                          256, 1, 1, 0, stream, add_args, nullptr));
        CU(cuStreamSynchronize(stream));
        std::vector<unsigned> host(n);
        CU(cuMemcpyDtoH(host.data(), output, n * sizeof(unsigned)));
        for (unsigned i = 0; i < n; ++i) {
            if (host[i] != i + delta) {
                throw std::runtime_error("first mismatch at index " +
                    std::to_string(i) + ": expected=" + std::to_string(i + delta) +
                    ", actual=" + std::to_string(host[i]));
            }
        }
        CU(cuMemFree(output)); output = 0;
        CU(cuStreamDestroy(stream)); stream = nullptr;
        CU(cuModuleUnload(module)); module = nullptr;
        CU(cuCtxDestroy(context)); context = nullptr;
        CU(cuCtxSetCurrent(previous));
        std::puts("PASS: all 1025 elements equal index + 7; worker context destroyed");
    } catch (...) {
        // Best effort cleanup; preserve the original diagnostic.
        if (context) {
            cuCtxSetCurrent(context);
            cuCtxSynchronize();
            if (output) cuMemFree(output);
            if (stream) cuStreamDestroy(stream);
            if (module) cuModuleUnload(module);
            cuCtxDestroy(context);
        }
        cuCtxSetCurrent(previous);
        throw;
    }
}

int main(int argc, char** argv) {
    CUcontext primary = nullptr;
    CUdevice device = 0;
    bool retained = false;
    int status = 0;
    try {
        if (argc > 2) throw std::runtime_error("usage: context_thread [device ordinal]");
        int ordinal = 0;
        if (argc == 2) {
            std::size_t consumed = 0;
            ordinal = std::stoi(argv[1], &consumed);
            if (argv[1][consumed] != '\0' || ordinal < 0)
                throw std::runtime_error("invalid device ordinal");
        }
        CU(cuInit(0));
        CU(cuDeviceGet(&device, ordinal));
        CU(cuDevicePrimaryCtxRetain(&primary, device)); retained = true;
        CU(cuCtxSetCurrent(primary));
        std::printf("main primary context=%p\n", (void*)primary);
        std::exception_ptr failure;
        std::thread thread([&] {
            try { worker(device); } catch (...) { failure = std::current_exception(); }
        });
        thread.join();
        if (failure) std::rethrow_exception(failure);
        CUcontext current = nullptr;
        CU(cuCtxGetCurrent(&current));
        if (current != primary) throw std::runtime_error("main context changed unexpectedly");
        std::puts("PASS: main thread still has its original primary context");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        status = 1;
    }
    if (retained) {
        CUresult unbind = cuCtxSetCurrent(nullptr);
        CUresult release = cuDevicePrimaryCtxRelease(device);
        if (unbind != CUDA_SUCCESS || release != CUDA_SUCCESS) {
            std::fprintf(stderr, "FAIL: primary context cleanup (%d, %d)\n",
                         (int)unbind, (int)release);
            status = 1;
        }
    }
    return status;
}
