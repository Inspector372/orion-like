# CUDA workload registry

Each `.cu` defines kernels and one complete host workload declared in
`testcase.h`. Workloads own allocation, initialization, launches, synchronization,
CPU verification, and cleanup. They contain no scheduler mutexes. `registry.cpp`
maps names to functions. `thread_wrapper()` owns client startup and results.

## Build and run

```sh
make -j CUDA_HOME=/usr/local/cuda-12.8 ARCH=sm_70
./testcase_runner --list
./testcase_runner coverage:4097:3
./testcase_runner matmul:64:10
LD_PRELOAD=./hooking.so ./threading coverage:4097:3
```

Use the architecture/toolkit appropriate to your GPU. The default retains the
existing project's `sm_70` and `-G` injection requirements. This suite is not a
validated optimized benchmark; do not compare debug builds with optimized ones.
Run the standalone executable without the hooking library preloaded.

Specifications are `name[:size[:iterations[:work[:seed]]]]`:

- `size`: element count, or square matrix dimension for matmul.
- `iterations`: kernel repetitions; chained launches one initializer followed by
  this many dependent transforms.
- `work`: integer recurrence steps per thread, used only by compute.
- `seed`: deterministic input seed, including zero.

Sizes are 1..67108864 elements or 1..1024 for matmul; iterations/work are
1..1000000. Large settings can take substantial time, including CPU verification.
All supplied kernels use 1D grids and 256-thread blocks, aligned with the
scheduler's current 1024-thread atoms.

`threading` accepts no specs (coverage for all clients), one spec repeated for all
clients, or exactly `THREAD_NUM` specs. The current repository uses four clients.
Arguments are validated before scheduler initialization. Both runners return
nonzero for mismatches or errors. The scheduler runs until all clients finish
and queues drain; use an external watchdog for deadlocks, e.g.:

```sh
timeout 60s env LD_PRELOAD=./hooking.so ./threading coverage:4097:3
```

Run `make test-host` for host-only registry and invalid-argument checks; this
does not validate CUDA execution.

## Scheduling scenarios

These commands define scenarios without requiring a JSON parser:

```sh
# Compute versus memory, repeated across four clients.
LD_PRELOAD=./hooking.so ./threading compute:4096:10:4096 vector_add:1048576:10 compute:4096:10:4096 vector_add:1048576:10
# Short versus long kernels.
LD_PRELOAD=./hooking.so ./threading coverage:256:100 compute:16384:10:4096 coverage:256:100 compute:16384:10:4096
# Ordering and mixed workloads.
LD_PRELOAD=./hooking.so ./threading chained:4097:10 matmul:64:10 vector_add:65536:10 coverage:4097:10
```

Start by checking every workload standalone. Sweep coverage sizes 256, 768,
1024, 1025, 1280, 1792, 2048, 2304, and 4097, then run the same specs scheduled.
Coverage detects omitted and duplicate execution; chained checks dependencies;
vector_add exercises memory traffic; compute varies arithmetic duration; matmul
provides a complete 1D matrix workload with full CPU verification.

The hook redirects kernels to per-client scheduler streams. These initial tests
use a final device synchronization before reading outputs. Device-wide waits can
include other clients, so whole-function runtime is not isolated GPU latency.
Accurate per-client GPU timing needs events on the scheduler's actual streams.
No performance thresholds or fairness claims are imposed by this correctness suite.

## Add a workload

1. Create `testcase/name.cu` with `Result name(const Config&)` and local kernels.
2. Declare the function in `testcase.h`.
3. Add its name, function, and default size to `registry.cpp`; update parameter
   validation if it needs different limits or semantics.
4. Run `make`; CUDA source files in this directory are discovered automatically.
5. Verify standalone, then schedule it alongside existing workloads.

Use checked CUDA calls, deterministic per-call inputs, and meaningful output
verification. Resources must outlive all launches. Do not add a `main()` to the
workload files; `standalone.cpp` supplies it. Exceptions are caught by the runners,
and the common device buffer releases allocations during unwinding.

The existing low-level interception and metadata injection still need GPU
validation. Unsupported multidimensional, cooperative, or library launches are
not covered by these initial workloads.
