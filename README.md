# Orion-like

## Testcases

Each `.cu` file defines kernels and one complete host workload declared in
`testcase.h`. Workloads own allocation, initialization, launches, synchronization,
CPU verification, and cleanup. They contain no scheduler mutexes.

`registry.cpp` maps names to functions and parses workload parameters.
`thread_wrapper()` handles client startup, invokes the selected workload, and
records its result.

## Build and run

Build the scheduler and hooking library:

```sh
make hooking.so threading
```

List available workloads:

```sh
./threading --list
```

Run the same workload in every client:

```sh
LD_PRELOAD=./hooking.so ./threading coverage:4097:3
LD_PRELOAD=./hooking.so ./threading matmul:64:10
```

Use the CUDA toolkit and architecture appropriate to your GPU. Keep build
settings consistent when comparing results.

## Workload parameters

Specifications use this format:

```text
name[:size[:iterations[:work[:seed]]]]
```

* `size`: element count, or square matrix dimension for matmul.
* `iterations`: kernel repetitions. The chained workload launches one initializer
  followed by this many dependent transforms.
* `work`: integer recurrence steps per thread; used only by compute.
* `seed`: deterministic input seed, including zero.

Default workloads and sizes:

| Name         | Default size | Purpose                                       |
| ------------ | -----------: | --------------------------------------------- |
| `coverage`   |         4097 | Detect missing or duplicated execution        |
| `vector_add` |        65536 | Exercise memory traffic and verify addition   |
| `matmul`     |           64 | Verify matrix multiplication with 1D indexing |
| `compute`    |         4096 | Vary arithmetic work per thread               |
| `chained`    |         4097 | Verify dependent kernel execution             |

Defaults for `iterations`, `work`, and `seed` are 1, 256, and 1.

Supported sizes are 1–67108864 elements, or 1–1024 for matmul.
Iterations and work must be within 1–1000000. Large settings can take substantial
time, including CPU verification.

All supplied kernels use 1D grids and 256-thread blocks, aligned with the
scheduler's current 1024-thread atoms.

## Client selection

`threading` accepts:

* No specifications: run coverage with default parameters for every client.
* One specification: run that configuration for every client.
* Exactly `THREAD_NUM` specifications: assign one configuration to each client.

The current configuration uses four clients. Arguments are validated before
scheduler initialization. Workload mismatches or errors produce a nonzero
exit status.

The scheduler runs until all clients finish and queues drain. Use an external
timeout to detect hangs:

```sh
timeout 60s env LD_PRELOAD=./hooking.so ./threading coverage:4097:3
```

## Scheduling scenarios

Compute versus memory:

```sh
LD_PRELOAD=./hooking.so ./threading \
    compute:4096:10:4096 \
    vector_add:1048576:10 \
    compute:4096:10:4096 \
    vector_add:1048576:10
```

Short versus long kernels:

```sh
LD_PRELOAD=./hooking.so ./threading \
    coverage:256:100 \
    compute:16384:10:4096 \
    coverage:256:100 \
    compute:16384:10:4096
```

Ordering and mixed workloads:

```sh
LD_PRELOAD=./hooking.so ./threading \
    chained:4097:10 \
    matmul:64:10 \
    vector_add:65536:10 \
    coverage:4097:10
```

Start with coverage sizes 256, 768, 1024, 1025, 1280, 1792, 2048, 2304,
and 4097 to exercise launches around atom boundaries. Then run the other
workloads individually across clients before trying mixed scenarios.

## Timing and compatibility

The hook redirects kernels to per-client scheduler streams. These tests use a
final device synchronization before reading outputs. Device-wide waits can
include other clients, so whole-function runtime is not isolated GPU latency.

Accurate per-client GPU timing requires events on the scheduler's actual streams.
This correctness suite imposes no performance thresholds or fairness guarantees.

The low-level interception and metadata injection require GPU validation.
Multidimensional, cooperative, and library launches are not covered by these
initial workloads.

## Add a workload

1. Create `testcase/name.cu` with `Result name(const Config&)` and its kernels.
2. Declare the function in `testcase.h`.
3. Add its name, function, and default size to `registry.cpp`.
4. Update parameter validation if it needs different limits or semantics.
5. Run `make threading` to compile and link it.
6. Run the workload across clients, then alongside existing workloads.

The Makefile discovers workload `.cu` files through `TEST_SOURCES`.
`TEST_OBJECTS` must also include `testcase/registry.o`.

Use checked CUDA calls, deterministic per-call inputs, and meaningful output
verification. Resources must outlive all launches. Workload files should expose
their registered function rather than define `main()`.

Exceptions are caught by `thread_wrapper()`, and the common device buffer
releases allocations during exception unwinding.
