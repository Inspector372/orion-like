# cuDNN C++ Frontend workloads

These workloads use the actual cuDNN Frontend Graph C++ API, not cuBLAS or custom
CUDA replacements. They exercise forward/inference operations and return normal
registry Result values. No new runner or main() is required.

| Registry name | Operation | Meaning of size | Default | Supported test range |
|---|---|---|---:|---|
| cudnn_matmul | C = A B, row-major square matrices | Matrix dimension | 64 | 8..512, multiple of 8 |
| cudnn_convolution | 3x3 cross-correlation, stride 1, padding 1 | Image height and width | 16 | 1..128 |
| cudnn_layernorm | Per-row layer normalization with learned scale/bias | Hidden width | 128 | 8..4096, multiple of 8 |
| cudnn_attention | softmax(Q K^T / sqrt(64)) V | Sequence length | 64 | 64..512, multiple of 64 |

Convolution uses batch 1, eight input and eight output channels, NHWC storage
and KRSC filters. Layer normalization uses eight rows, population variance, and
epsilon 1e-5. Attention uses batch 1, two heads, head dimension 64, no causal mask,
no bias, and no dropout. Iterations repeats the graph; work is unused; seed
controls deterministic input generation.

## Dependencies and build

The source targets the API in NVIDIA cudnn-frontend v1.9.0, cuDNN 9.x, CUDA 12.x,
and an Ampere (SM80) or newer GPU. Exact operation support is checked while
building the execution plan. This is a selected test configuration, not a claim
that every cuDNN operation requires Ampere. cuDNN is installed separately from
the CUDA toolkit. Frontend headers are another separate dependency.

Install cuDNN development headers/libraries compatible with your CUDA toolkit,
and obtain NVIDIA's header-only Frontend checkout:

```sh
git clone --branch v1.9.0 --depth 1 https://github.com/NVIDIA/cudnn-frontend.git /path/to/cudnn-frontend
make ENABLE_CUDNN=1 \
    CUDNN_FRONTEND_DIR=/path/to/cudnn-frontend \
    CUDNN_INCLUDE_DIR=/path/to/cudnn/include \
    CUDNN_LIB_DIR=/path/to/cudnn/lib
```

Use paths from your installation. Ensure the cuDNN shared libraries can be found
by the dynamic loader (system configuration or LD_LIBRARY_PATH).
Only the four cuDNN objects use C++17 and the extra include flags; the existing
CUDA compilation settings remain intact. The files contain host calls into
cuDNN, so the existing sm_70 flag does not select cuDNN's internal GPU kernels.

By default ENABLE_CUDNN=0, allowing the original project to build without this
optional dependency. The four names remain listed but return a clear failure
message if selected in a disabled build. The cuDNN objects rebuild when invoking
make so switching ENABLE_CUDNN cannot silently reuse disabled/enabled objects.
Pass ENABLE_CUDNN=1 on subsequent build commands too.

## Run

```sh
LD_PRELOAD=./hooking.so ./threading cudnn_matmul:64:3
LD_PRELOAD=./hooking.so ./threading cudnn_convolution:16:3
LD_PRELOAD=./hooking.so ./threading cudnn_layernorm:128:3
LD_PRELOAD=./hooking.so ./threading cudnn_attention:64:3

# One different operation per client (THREAD_NUM=4).
LD_PRELOAD=./hooking.so ./threading \
    cudnn_matmul:64:3 cudnn_convolution:16:3 \
    cudnn_layernorm:128:3 cudnn_attention:64:3
```

Every graph has its own handle, plan, and workspace. Inputs and outputs are FP16,
with FP32 intermediate/compute types. CPU references use double accumulation on
the exact FP16 input values. Every output is checked on every iteration, including
non-finite values; tolerance is 0.005 + 0.005 * abs(reference). Outputs start as
NaNs copied from host memory so missing writes are detected. There are no
explicit cudaMemset calls; cuDNN may perform internal fills or other helper work.
The shared helper owns plan execution/error reporting, while cudnn_reference.h
contains CUDA-independent reference calculations.

## Scheduler compatibility and validation

These are library compatibility tests. cuDNN chooses launch geometry and may
submit multiple kernels or use entry points outside the current launch hooks.
Its graph API describes a cuDNN operation graph; these tests do not explicitly
capture or launch a CUDA Graph. Do not interpret a successful host API return
as proof of correct atomization. Validate the output and inspect launch traces.
The original scheduler's 1D thread filtering/injected kernel entry may not support
library kernels. Internal helper launches must remain outside patching unless
positively identified as intended scheduled work.

Execution uses a final device synchronization before verification, which can wait
for other clients. This is not an isolated latency benchmark. Unsupported graph
plans produce a failure with the Frontend error message, not a silent fallback
or a passing result. Header/library absence is a build configuration issue.

Host-only checks exercised analytical reference results, registry bounds, and
disabled-build behavior. CUDA-enabled compilation and GPU execution were not
available in the authoring environment and remain to be validated locally.

API examples used to check the implementation:
- https://github.com/NVIDIA/cudnn-frontend/tree/v1.9.0/samples/cpp/matmul
- https://github.com/NVIDIA/cudnn-frontend/tree/v1.9.0/samples/cpp/convolution
- https://github.com/NVIDIA/cudnn-frontend/blob/v1.9.0/samples/cpp/norm/layernorm.cpp
- https://github.com/NVIDIA/cudnn-frontend/blob/v1.9.0/samples/cpp/sdpa/fp16_fwd.cpp
