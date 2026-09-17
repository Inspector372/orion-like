# LibTorch CUDA smoke tests

These forward-only workloads exercise CUDA libraries through LibTorch's C++
tensor API. They use the existing registry and client threads; no Python,
model downloads, training loop, or CMake build is required.

| Name | Forward operation | `size` (default; maximum) |
| --- | --- | --- |
| `libtorch_feedforward` | Linear(size, 2*size), ReLU, Linear(2*size, size); batch 8 | feature width (64; 512) |
| `libtorch_convolution` | cuDNN 3x3 convolution, padding 1, stride 1, ReLU; N=2, Cin=8, Cout=16 | square image side (16; 128) |

The feed-forward case uses ordinary `at::linear` on CUDA FP32 tensors, exercising
LibTorch's GEMM path (cuBLAS/cuBLASLt selection is version-dependent). The
convolution explicitly uses `at::cudnn_convolution`, then ReLU. It fails if CUDA
or cuDNN is unavailable instead of silently using a different convolution
backend. This is an ATen operator shipped with LibTorch, not a direct call from
the testcase to the cuDNN C API. Its benchmark search and TF32 are disabled.

Each invocation owns its tensors. Inputs and weights are initialized on the CPU
with a local deterministic generator, then copied to CUDA. No global RNG seed or
global backend setting is modified by client threads. Inference mode disables
autograd. Every iteration synchronizes the device and compares the GPU output
to the equivalent CPU operations, checking finite values with rtol=0.002 and
atol=0.0002. Errors propagate to the existing harness as FAIL results.

## Build with the existing Makefile

Expected layout:

```text
parent/
  libtorch-cu126/libtorch/include/
  libtorch-cu126/libtorch/lib/
  orion-like/Makefile
```

Use the **CUDA-enabled shared LibTorch distribution**, preferably with bundled
dependencies and a CUDA version compatible with your installed toolkit/driver.
The LibTorch files on your machine are not downloaded by this Makefile.

```bash
make ENABLE_LIBTORCH=1
```

The default is `LIBTORCH_DIR=../libtorch-cu126/libtorch`. To override it:

```bash
make ENABLE_LIBTORCH=1 LIBTORCH_DIR=/absolute/path/to/libtorch
```

The enabled LibTorch `.cpp` testcases compile with `LIBTORCH_CXX` (default
`g++-12`) and C++20. Install that compiler or select a newer compatible one:

```bash
make ENABLE_LIBTORCH=1 LIBTORCH_CXX=g++-13
```

GCC 10 reports `__cplusplus=201709L` even in its experimental C++20 mode;
it fails ATen's `202002L` check. Changing only `-std=c++20` is not enough.
The `check-libtorch-compiler` target checks this before compiling the tests.
Do not redefine `__cplusplus` or remove ATen's check.

Other host files use `CXX` (currently `g++-10`), and nvcc keeps its existing
host-compiler selection. The runner does not include ATen and only needs
C++17 when LibTorch is enabled. A disabled LibTorch build uses `CXX` and
C++17 for its stubs; it does not require `g++-12`.

The LibTorch testcase files contain no custom CUDA kernel definitions and do not require
nvcc compilation; their tensor operations dispatch into LibTorch's CUDA
libraries. The existing nvcc link step remains in place. Both LibTorch include
directories and the CUDA/CPU Torch libraries are supplied by the Makefile.

`ENABLE_CUDNN=1` is only needed for the separate `cudnn_*` frontend testcases;
it is not required for LibTorch convolution. LibTorch supplies its own cuDNN
integration. If enabling both, their CUDA/cuDNN dependencies must be compatible.

## C++ ABI and rebuilds

The Makefile reads `_GLIBCXX_USE_CXX11_ABI` from the distribution's
`share/cmake/Torch/TorchConfig.cmake` when that flag is present, otherwise it
defaults to 1 (the cxx11 ABI). For an older pre-cxx11 ABI package, override:

```bash
make ENABLE_LIBTORCH=1 LIBTORCH_CXX11_ABI=0
```

The ABI flag is applied to all project C++ translation units, including the
registry and harness, because they exchange `std::string` in testcase results.
A configuration stamp rebuilds these objects when LibTorch settings change.
Replacing the installed LibTorch package in place requires `make clean` first.
Use the compiler/system requirements specified by your downloaded release.

LibTorch's absolute library directory is added to the executable's rpath. If
the loader cannot find bundled transitive dependencies, run with:

```bash
export LD_LIBRARY_PATH="$(pwd)/../libtorch-cu126/libtorch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Run

```bash
LD_PRELOAD=./hooking.so ./threading libtorch_feedforward:64:3
LD_PRELOAD=./hooking.so ./threading libtorch_convolution:16:3
```

The standard syntax is `name[:size[:iterations[:work[:seed]]]]`. `work` is unused
by these two cases. A single specification runs on every client. With four
clients, mixed workloads can be selected as usual:

```bash
LD_PRELOAD=./hooking.so ./threading \
  libtorch_feedforward:64:3 libtorch_convolution:16:3 \
  libtorch_feedforward:128:3 libtorch_convolution:32:3
```

Without `ENABLE_LIBTORCH=1`, the names remain listed but return an explicit
disabled-build failure. The default build needs no LibTorch headers/libraries.

## Interception expectations

These tests intentionally use normal library behavior: handle creation, memory
allocation, internal memset, GEMM/convolution kernels, and pointwise ReLU.
They contain no explicit GPU memset, but LibTorch and its dependencies may
issue them internally. There is no explicit CUDA Graph capture or replay.

Keep internal initialization/fill work intact. Preserving stream dependencies
and tensor lifetimes remains the scheduler's responsibility: synchronizing at
the end cannot repair reordered dependent operations earlier in a forward
pass. A mismatch or stall with interception enabled can therefore indicate a
scheduler issue rather than a missing LibTorch installation. The existing
`threading` harness enables its callback itself; merely omitting LD_PRELOAD
does not provide an independent, unhooked baseline.

A numerical PASS verifies results. To confirm the exact cuBLAS entry point or
engine used by a particular LibTorch release, inspect the launch backtrace or
profile it; the test does not assert a specific cuBLAS API name.

Reference: [official LibTorch installation documentation](https://docs.pytorch.org/cppdocs/installing.html).
