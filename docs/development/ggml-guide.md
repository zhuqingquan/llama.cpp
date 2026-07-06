# GGML Library: Build, Test & Debug Guide

GGML is the low-level tensor computation library that powers llama.cpp. This
document covers how ggml is built, what tests exist for it, and how to build
and run those tests.

---

## 1. Library Structure

```
ggml/
├── CMakeLists.txt          # ggml's own CMake build system
├── cmake/                  # CMake modules (FindSIMD, etc.)
├── include/
│   ├── ggml.h              # Core API: tensor ops, computation graph
│   ├── ggml-alloc.h        # Memory allocator
│   ├── ggml-backend.h      # Backend abstraction (CPU/CUDA/Vulkan/...)
│   ├── ggml-cpu.h          # CPU-specific API
│   ├── ggml-cuda.h         # CUDA backend header
│   ├── ggml-metal.h        # Metal backend header
│   └── ...
└── src/
    ├── ggml.c              # Core: tensor creation, graph compute, quant tables
    ├── ggml-alloc.c        # Memory allocation for backend buffers
    ├── ggml-backend.cpp    # Backend registry, CPU backend buffer types
    ├── ggml-backend-reg.cpp# Dynamic loading of backend .so libraries
    ├── ggml-quants.c       # Quantization/dequantization implementations
    ├── ggml-opt.cpp        # Numerical optimization (SGD, AdamW)
    ├── ggml-threading.cpp  # Thread pool for parallel graph execution
    ├── ggml-cpu/           # CPU backend: SIMD kernels (x86, ARM, RISC-V, ...)
    ├── ggml-cuda/          # CUDA backend: all GPU kernels
    ├── ggml-metal/         # Metal backend (Apple Silicon)
    ├── ggml-vulkan/        # Vulkan backend
    └── ...
```

### Key build targets (inside `build/bin/`)

| Binary | Source | Description |
|--------|--------|-------------|
| `libggml.so` | `ggml/src/ggml.c` + `ggml-cpu/` | Core ggml CPU implementation |
| `libggml-cpu.so` | `ggml/src/ggml-cpu/` | CPU backend (SIMD kernels) |
| `libggml-cuda.so` | `ggml/src/ggml-cuda/` | CUDA backend (GPU kernels) |
| `libggml-base.so` | `ggml/src/ggml.c` + `ggml-alloc.c` | Base layer (no backend) |
| `libggml-metal.so` | `ggml/src/ggml-metal/` | Metal backend (macOS) |
| `libggml-vulkan.so` | `ggml/src/ggml-vulkan/` | Vulkan backend |

---

## 2. Building ggml

### 2.1 As part of llama.cpp (normal usage)

```bash
cd llama.cpp
cmake -B build
cmake --build build -j$(nproc)
```

ggml is added as a subdirectory in llama.cpp's CMakeLists.txt and built
together. All ggml options are available via CMake cache variables:

```bash
# Enable CUDA backend with FlashAttention and NCCL
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_NCCL=ON

# Enable Vulkan backend
cmake -B build -DGGML_VULKAN=ON

# Debug build (slower, with symbols, assertions)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# RelWithDebInfo (optimized with debug symbols — best for debugging)
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### 2.2 Standalone ggml build

ggml can be built independently without llama.cpp:

```bash
cd ggml
cmake -B build
cmake --build build -j$(nproc)
```

When built standalone, all `GGML_*` options are available. Note: test
targets live in llama.cpp's `tests/CMakeLists.txt`, not in ggml's own
build.

### 2.3 Backend loading

Backends can be loaded in two ways:

| Mode | CMake option | Behavior |
|------|-------------|----------|
| Static | (default, `GGML_BACKEND_DL=OFF`) | Backends compiled into the main binary |
| Dynamic | `GGML_BACKEND_DL=ON` | Backends built as separate `.so` files, loaded at runtime |

In dynamic mode, `ggml_backend_load_all()` searches for `.so` files in the
directory specified by `GGML_BACKEND_DIR`.

### 2.4 Key CMake options

See `ggml/CMakeLists.txt` for the full list. Important ones:

| Option | Default | Purpose |
|--------|---------|---------|
| `GGML_NATIVE` | ON | `-march=native` for best perf on current CPU |
| `GGML_CUDA` | OFF | Enable CUDA backend |
| `GGML_CUDA_FA` | ON | FlashAttention CUDA kernels |
| `GGML_VULKAN` | OFF | Enable Vulkan backend |
| `GGML_METAL` | ON (macOS) | Metal backend |
| `GGML_BLAS` | OFF (Linux) | BLAS acceleration |
| `GGML_OPENMP` | ON | OpenMP threading |
| `GGML_LTO` | OFF | Link-time optimization |
| `GGML_BACKEND_DL` | OFF | Dynamic backend loading |
| `GGML_SCHED_NO_REALLOC` | OFF | Disable allocator reallocs (debugging) |

---

## 3. GGML-only Tests

These tests only `#include "ggml.h"` (or `ggml-cpu.h`, `ggml-backend.h`),
and do not depend on `llama.h` — they don't load models or do tokenization.

### 3.1 Test list

| # | Source file | Binary | Lines | Dependencies | What it tests |
|---|------------|--------|-------|-------------|---------------|
| 1 | `tests/test-backend-ops.cpp` | `test-backend-ops` | 9402 | `ggml.h`, `ggml-alloc.h`, `ggml-backend.h` | All ggml ops on all registered backends — forward pass consistency + backward gradient check |
| 2 | `tests/test-alloc.cpp` | `test-alloc` | 608 | `ggml.h`, `ggml-alloc.h`, `ggml-backend-impl.h` | Backend buffer allocator logic, buffer fragmentation, multi-buffer allocation |
| 3 | `tests/test-barrier.cpp` | `test-barrier` | 236 | `ggml.h`, `ggml-cpu.h` | Multi-threaded graph compute barrier correctness |
| 4 | `tests/test-quantize-fns.cpp` | `test-quantize-fns` | 196 | `ggml.h`, `ggml-cpu.h` | Quantize/dequantize accuracy + dot product error tolerance |
| 5 | `tests/test-quantize-perf.cpp` | `test-quantize-perf` | 356 | `ggml.h`, `ggml-cpu.h` | Quantization throughput benchmark on synthetic data (L1/L2/L3 cache tiers) |
| 6 | `tests/test-rope.cpp` | `test-rope` | 263 | `ggml.h`, `ggml-cpu.h` | RoPE positional encoding correctness against reference implementation |
| 7 | `tests/test-opt.cpp` | `test-opt` | 1003 | `ggml.h`, `ggml-opt.h` | Numerical optimizer (SGD/AdamW) — forward, backward, parameter update |
| 8 | `tests/test-gguf.cpp` | `test-gguf` | — | `ggml.h` | GGUF file format read/write |
| 9 | `tests/export-graph-ops.cpp` | `export-graph-ops` | 226 | `ggml.h`, `ggml-backend.h` | Dump all ops from a model's compute graph (no actual compute) |

### 3.2 POC / demo programs

| Directory | Binary | Lines | Description |
|-----------|--------|-------|-------------|
| `pocs/vdot/vdot.cpp` | `llama-vdot` | 311 | FP32/F16/Q4_0 vector dot product benchmark, compares CPU impls |
| `pocs/vdot/q8dot.cpp` | `llama-q8dot` | — | Q8_0 quantized dot product benchmark |

### 3.3 Key test: test-backend-ops

This is the most comprehensive test (9400+ lines). It verifies that every
ggml operation produces consistent results across all backends.

Key operations tested:
- Unary ops: `RELU`, `GELU`, `SILU`, `TANH`, `SQRT`, `NEG`, `STEP`, `SOFTMAX`, `RMS_NORM`, `LAYER_NORM`, `GROUP_NORM`
- Binary ops: `ADD`, `SUB`, `MUL`, `DIV`, `SQR`, `MAX`, `MIN`
- Matrix ops: `MUL_MAT`, `OUT_PROD`
- Attention: `FLASH_ATTN_EXT`
- Quantization: `QUANTIZE`, `DEQUANTIZE`
- Others: `ROPE`, `CPY`, `CONT`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`, `GET_ROWS`, `SET_ROWS`, `REPEAT`, `CONCAT`, `SCALE`, `CLAMP`, `DIAG_MASK`, `SOFT_CAP`, `SUM_ROWS`, `ARGSORT`, `COUNT_EQUAL`, `IM2COL`, `CONV2D`, `POOL2D`, `UPSCALE`, `CROSS_ENTROPY_LOSS`, `SSM_CONV`, `SSM_SCAN`, `GATED_DELTA_NET`, `RWKV_WKV6`, `GLA`, `OPT_STEP`

Usage:
```bash
# Run all tests on CPU backend
./build/bin/test-backend-ops

# Run on CUDA backend (device 0)
./build/bin/test-backend-ops -b CUDA0

# Run with gradient checking
./build/bin/test-backend-ops -o GRAD

# Run a single test
./build/bin/test-backend-ops -o ADD -b CPU

# Performance mode
./build/bin/test-backend-ops -o MUL_MAT -b CUDA0 perf
```

---

## 4. How to Build and Run Tests

### 4.1 Build all test targets

```bash
cd llama.cpp

# Build everything (tests included)
cmake --build build -j$(nproc)

# Or build specific test targets
cmake --build build -j$(nproc) --target \
    test-backend-ops test-rope test-barrier test-alloc \
    test-quantize-fns test-quantize-perf test-opt test-gguf \
    export-graph-ops llama-vdot llama-q8dot
```

All test binaries end up in `build/bin/`.

### 4.2 Run tests individually

```bash
# Correctness tests (pass/fail)
./build/bin/test-backend-ops
./build/bin/test-rope
./build/bin/test-barrier
./build/bin/test-alloc
./build/bin/test-quantize-fns
./build/bin/test-gguf
./build/bin/test-opt

# Benchmarks (print throughput numbers)
./build/bin/test-quantize-perf

# Tool (prints ops, requires model download)
./build/bin/export-graph-ops -m models/stories15M-q4_0.gguf

# POC demos
./build/bin/llama-vdot
./build/bin/llama-q8dot
```

### 4.3 Run tests via CTest

```bash
cd build
ctest -R "backend-ops|rope|barrier|alloc|quantize-fns|quantize-perf|opt|gguf$" -V
```

### 4.4 Test code pattern

All ggml tests follow a similar pattern. Here is a minimal example
(based on `test-rope.cpp`):

```cpp
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cassert>

int main() {
    // 1. Init ggml context (metadata pool)
    struct ggml_init_params params = {
        /*.mem_size   =*/ 128 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 2. Create tensors
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);

    // 3. Fill with test data (no_alloc=false → data is in ggml pool)
    for (int i = 0; i < ggml_nelements(a); i++) {
        ((float *)a->data)[i] = (float)i;
        ((float *)b->data)[i] = (float)(i * 2);
    }

    // 4. Build compute graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_tensor * result = ggml_add(ctx, a, b);  // result = a + b
    ggml_build_forward_expand(gf, result);

    // 5. Compute (CPU backend)
    int n_threads = 4;
    ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_backend_graph_compute(backend, gf);

    // 6. Check results
    for (int i = 0; i < ggml_nelements(result); i++) {
        float expected = i + (i * 2);
        assert(fabs(((float *)result->data)[i] - expected) < 1e-5);
    }

    printf("test passed\n");
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
```

> **Note on `no_alloc = false` in tests**: Most simple ggml tests set
> `no_alloc = false` in `ggml_init_params`. This means `ggml_new_tensor()`
> allocates tensor data from the context's memory pool, and you can
> directly write to `tensor->data`. In contrast, llama.cpp itself uses
> `no_alloc = true` and assigns backend buffers to `tensor->data` later.

### 4.5 Pattern for GPU backend tests

`test-backend-ops` is the model for testing GPU backends:

```cpp
// 1. Init backends
ggml_backend_t backend_cpu = ggml_backend_init_by_name("CPU", NULL);
ggml_backend_t backend_gpu = ggml_backend_init_by_name("CUDA0", NULL);

// 2. Use ggml_backend_alloc_ctx_tensors_from_buft for backend-owned mem
ggml_backend_buffer_type_t buft_cpu = ggml_backend_cpu_buffer_type();
ggml_backend_buffer_type_t buft_cuda = ggml_backend_cuda_buffer_type(0);

// 3. Build graph in CPU context, allocate tensors in GPU buffer
ggml_backend_alloc_ctx_tensors(ctx_gpu, buft_cuda);

// 4. Compute and compare
ggml_backend_graph_compute(backend_cpu, gf_cpu);
ggml_backend_graph_compute(backend_gpu, gf_gpu);
compare_tensors(result_cpu, result_gpu);
```

---

## 5. Programming with ggml: General-Purpose Computation Guide

This section provides a step-by-step guide to using ggml as a standalone
tensor computation library, without any dependency on llama.cpp's model
layer.

### 5.1 Quick Reference: The Six-Step Pattern

Every ggml program follows the same structure:

```
1. ggml_init()           → allocate metadata context
2. ggml_new_tensor_*()   → declare tensors (shape, type)
3. fill tensor->data     → populate input tensors with values
4. ggml_*() ops          → build a computation graph (lazily)
5. ggml_graph_compute()  → execute the graph on a backend
6. read result->data     → consume output values
```

---

### 5.2 Memory Management: Two Modes

ggml offers two distinct memory strategies. Choose one:

#### Mode A: Simple (`no_alloc = false`)

Tensor data lives inside ggml's internal memory pool. Best for quick
prototypes, toy programs, and simple unit tests.

```cpp
struct ggml_init_params params = {
    /*.mem_size   =*/ 128 * 1024 * 1024,  // 128 MB pool
    /*.mem_buffer =*/ NULL,               // let ggml malloc it
    /*.no_alloc   =*/ false,              // ❗ key: data goes in the pool
};
struct ggml_context * ctx = ggml_init(params);

// tensor->data points into ctx's mem_buffer
struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);

// write directly — this is YOUR memory
for (int i = 0; i < ggml_nelements(a); i++) {
    ((float *)a->data)[i] = data[i];
}

// ... build graph and compute ...

ggml_free(ctx);  // frees tensors AND data
```

- ✅ Zero boilerplate — no backend buffer setup
- ✅ Data and metadata share one free call
- ❌ Pure CPU only (data lives in a `malloc`'d buffer)
- ❌ Fixed pool size — must know max usage upfront

#### Mode B: Backend-managed (`no_alloc = true`)

ggml only stores tensor metadata (shape, type, op). Data buffers are
allocated separately by a backend (CPU, CUDA, Vulkan…). This is what
llama.cpp uses.

```cpp
// 1) Metadata context (no data, small pool)
struct ggml_init_params params = { 1024*1024, NULL, /*.no_alloc=*/ true };
struct ggml_context * ctx = ggml_init(params);

// 2) Declare tensors — data is NULL at this point
struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);
struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);

// 3) Build graph
struct ggml_cgraph * gf = ggml_new_graph(ctx);
struct ggml_tensor * result = ggml_add(ctx, a, b);
ggml_build_forward_expand(gf, result);

// 4) Allocate backend buffers (CPU in this example)
ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, ggml_graph_size(gf));
ggml_backend_tensor_alloc(buf, a, ggml_nbytes(a));
ggml_backend_tensor_alloc(buf, b, ggml_nbytes(b));
ggml_backend_tensor_alloc(buf, result, ggml_nbytes(result));

// 5) Copy input data to backend tensors
float host_a[128*32] = { ... };
ggml_backend_tensor_set(a, host_a, 0, ggml_nbytes(a));

// 6) Compute
ggml_backend_graph_compute(backend, gf);

// 7) Read results back
float host_result[128*32];
ggml_backend_tensor_get(result, host_result, 0, ggml_nbytes(result));

ggml_backend_buffer_free(buf);
ggml_backend_free(backend);
ggml_free(ctx);
```

- ✅ True GPU offloading — tensors live in device memory
- ✅ Multi-backend — same graph, different hardware
- ❌ More verbose setup

**One-shot shortcut:** `ggml_backend_alloc_ctx_tensors_from_buft` allocates
buffers for ALL tensors in a context at once:

```cpp
ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
// now every tensor in ctx has data allocated
```

---

### 5.3 Context and Tensor Lifecycle

```
ggml_init()          ctx ───────────────────────────────────── ggml_free(ctx)
                           │                    │
                    ggml_new_tensor_*()  ggml_new_graph()
                           │                    │
                      ggml_*_op()          ggml_build_forward_expand()
                           │                    │
                            └──────┬───────────┘
                                   │
                          ggml_graph_compute()
```

- **ggml_context** owns tensor *metadata* (shape, type, op graph). All
  `ggml_new_tensor_*` and `ggml_*` functions append objects into the context's
  linear pool.
- `ggml_free(ctx)` frees ALL tensors and graphs created from that context.
- Tensors CANNOT outlive their context.
- Graphs CANNOT outlive their context.
- Views (`ggml_view_*`, `ggml_reshape_*`) do not own data — they point into
  their parent tensor.

---

### 5.4 Building a Computation Graph

The graph is built lazily: op functions like `ggml_add(ctx, a, b)` only
record the relationship — no computation happens until you call
`ggml_graph_compute()`.

```cpp
// Declare inputs (mark them so ggml doesn't overwrite them)
struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 512);
ggml_set_input(input);
ggml_set_input(weight);

// Build the graph
struct ggml_cgraph * gf = ggml_new_graph(ctx);

// Layer 1: Linear + GELU
struct ggml_tensor * hidden = ggml_mul_mat(ctx, weight, ggml_reshape(ctx, input, 1, 256, 1, 1));
hidden = ggml_gelu(ctx, hidden);

// Layer 2: RMS Norm + add residual
struct ggml_tensor * normed = ggml_rms_norm(ctx, hidden, 1e-5f);
struct ggml_tensor * output = ggml_add(ctx, normed, hidden);
ggml_set_output(output);

// Expand the graph to include ALL reachable ops
ggml_build_forward_expand(gf, output);
```

Key points:
- **`ggml_set_input()`** marks a tensor whose data must not be overwritten
  during graph optimization or allocation.
- **`ggml_set_output()`** marks the final tensor(s) you need.
- **`ggml_build_forward_expand(gf, tensor)`** walks backward from `tensor` and
  adds every op it depends on to the graph.
- Graph pointer stability: if you call `ggml_build_forward_expand()` on the
  same `gf` multiple times, previous pointers into `gf->nodes[]` may be
  invalidated — collect the node count *after* the final expansion.

---

### 5.5 Backend Selection

```cpp
// CPU (always available)
ggml_backend_t cpu = ggml_backend_init_by_name("CPU", NULL);
ggml_backend_cpu_set_n_threads(cpu, 4);

// CUDA GPU 0 (requires GGML_CUDA=ON at build time)
ggml_backend_t cuda = ggml_backend_init_by_name("CUDA0", NULL);

// Vulkan GPU 0 (requires GGML_VULKAN=ON)
ggml_backend_t vk = ggml_backend_init_by_name("Vulkan0", NULL);

// Auto-pick: first GPU, fallback to CPU
ggml_backend_t best = ggml_backend_init_best();
```

Compute:
```cpp
ggml_backend_graph_compute(backend, gf);
```

For async backends (CUDA, Vulkan), synchronize before reading results:
```cpp
ggml_backend_synchronize(backend);
```

---

### 5.6 Operation Catalog

All ops take a `ggml_context *` as first argument and return a new tensor
node in the graph. Shapes must be broadcast-compatible.

#### Unary ops (one input → one output)

| Function | Description |
|----------|-------------|
| `ggml_dup(ctx, a)` | Copy tensor |
| `ggml_abs(ctx, a)` | Element-wise absolute value |
| `ggml_sgn(ctx, a)` | Element-wise sign |
| `ggml_neg(ctx, a)` | Element-wise negation |
| `ggml_step(ctx, a)` | Step function: 1 if a > 0, else 0 |
| `ggml_tanh(ctx, a)` | Hyperbolic tangent |
| `ggml_sigmoid(ctx, a)` | Sigmoid function |
| `ggml_relu(ctx, a)` | ReLU (rectified linear unit) |
| `ggml_gelu(ctx, a)` | GELU (tanh approximation, from `ggml.h:1152`) |
| `ggml_gelu_erf(ctx, a)` | GELU (erf-based exact) |
| `ggml_gelu_quick(ctx, a)` | GELU (fast sigmoid approximation) |
| `ggml_silu(ctx, a)` | SiLU / Swish |
| `ggml_hardsigmoid(ctx, a)` | Hard sigmoid |
| `ggml_hardswish(ctx, a)` | Hard Swish |
| `ggml_sqrt(ctx, a)` | Element-wise square root |
| `ggml_log(ctx, a)` | Natural logarithm |
| `ggml_exp(ctx, a)` | Element-wise exp |
| `ggml_sin(ctx, a)` | Element-wise sine |
| `ggml_cos(ctx, a)` | Element-wise cosine |
| `ggml_soft_max(ctx, a)` | Standard softmax over last dimension |
| `ggml_soft_max_ext(ctx, a, mask, scale, max_bias)` | Softmax with optional mask, scale, bias |
| `ggml_rms_norm(ctx, a, eps)` | RMS normalization over last dim |
| `ggml_norm(ctx, a, eps)` | Layer normalization over last dim |
| `ggml_group_norm(ctx, a, n_groups)` | Group normalization |

#### Binary ops (two inputs → one output)

| Function | Operation |
|----------|-----------|
| `ggml_add(ctx, a, b)` | a + b |
| `ggml_add_inplace(ctx, a, b)` | a + b, returning a view of a |
| `ggml_sub(ctx, a, b)` | a - b |
| `ggml_mul(ctx, a, b)` | a * b (element-wise) |
| `ggml_div(ctx, a, b)` | a / b |
| `ggml_max(ctx, a, b)` | max(a, b) element-wise |
| `ggml_min(ctx, a, b)` | min(a, b) element-wise |
| `ggml_sqr(ctx, a)` | a * a |

#### Matrix multiplication

| Function | Description |
|----------|-------------|
| `ggml_mul_mat(ctx, a, b)` | Matrix multiply: **b** @ **a** (note order) |
| `ggml_out_prod(ctx, a, b)` | Outer product: a ⊗ b |
| `ggml_mul_mat_id(ctx, as, ids, n_as, n_ids, b)` | Batched matmul with expert selection (MoE) |

> ⚠️ `ggml_mul_mat(ctx, A, B)` computes **B · A** (B times A), not A · B.
> This means A is the weight matrix and B is the input vector, matching
> `output = weight · input`.

#### Reshape and view

Views share the underlying data — they are zero-cost:

| Function | Description |
|----------|-------------|
| `ggml_reshape_1d/2d/3d/4d(ctx, a, ...)` | Reshape tensor dimensions |
| `ggml_view_1d/2d/3d(ctx, a, ...)` | View into a contiguous slice |
| `ggml_permute(ctx, a, axis0, axis1, axis2, axis3)` | Dimension permutation |
| `ggml_transpose(ctx, a)` | Transpose last two dimensions |
| `ggml_cont(ctx, a)` | Force contiguous layout |
| `ggml_cpy(ctx, a, b)` | Copy with type conversion |
| `ggml_repeat(ctx, a, b)` | Repeat a to match b's shape |
| `ggml_repeat_4d(ctx, a, rep0, rep1, rep2, rep3)` | Repeat along each dimension |
| `ggml_concat(ctx, a, b, dim)` | Concatenate along dimension |
| `ggml_pad(ctx, a, p0, p1, p2, p3)` | Zero-pad each dimension |
| `ggml_upscale(ctx, a, scale)` | Upscale with nearest-neighbor |

#### Reduction ops

| Function | Description |
|----------|-------------|
| `ggml_sum(ctx, a)` | Sum all elements |
| `ggml_sum_rows(ctx, a)` | Sum over rows |
| `ggml_mean(ctx, a)` | Mean of all elements |
| `ggml_argmax(ctx, a)` | Index of maximum value |
| `ggml_argsort(ctx, a, order)` | Indices that would sort the tensor |
| `ggml_count_equal(ctx, a, b)` | Count element-wise equal positions |

#### Special-purpose ops

| Function | Description |
|----------|-------------|
| `ggml_rope(ctx, a, b, n_dims, mode, n_ctx)` | Rotary Position Embedding |
| `ggml_rope_ext(ctx, a, b, c, ...)` | RoPE with extended options |
| `ggml_flash_attn_ext(ctx, q, k, v, mask, ...)` | Flash Attention |
| `ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1)` | 2D convolution |
| `ggml_im2col(ctx, a, b, ...)` | Image-to-column transform |
| `ggml_pool_2d(ctx, a, op, k0, k1, ...)` | 2D pooling (avg/max) |
| `ggml_scale(ctx, a, s)` | Scale tensor by float s |
| `ggml_clamp(ctx, a, min, max)` | Clamp values to range |
| `ggml_diag_mask_inf(ctx, a, n_past)` | Causal attention mask |
| `ggml_get_rows(ctx, a, b)` | Indexed row selection (embedding lookup) |
| `ggml_set_rows(ctx, a, b, c)` | Indexed row update |
| `ggml_timestep_embedding(ctx, a, dim, max_period)` | Sinusoidal timestep embedding |
| `ggml_cross_entropy_loss(ctx, a, b)` | Cross-entropy loss |
| `ggml_win_part(ctx, a, w)` | Sliding-window partition |
| `ggml_unwin(ctx, a, w, h, w)` | Sliding-window merge |

#### In-place variants

Most unary and binary ops have an `_inplace` variant that returns a view of
the first argument. These are building blocks for training/backprop.

---

### 5.7 Threading

ggml uses OpenMP for CPU parallelism. Control via:

```cpp
ggml_backend_cpu_set_n_threads(backend, 4);            // per-backend
// or
struct ggml_cplan cplan = ggml_graph_plan(gf, 4);   // per-graph
ggml_graph_compute_with_ctx(backend, gf, &cplan);
```

Critical: when computing from multiple threads, use `ggml_graph_plan()` +
`ggml_graph_compute_with_ctx()` to avoid a global mutex.

---

### 5.8 Quantized Data Types

ggml supports dozens of quantization formats for weight compression:

| Type Group | Examples | Bits/Element | Typical Use |
|------------|----------|-------------|-------------|
| Float | `F32`, `F16`, `BF16` | 32 / 16 / 16 | Activations, inputs, outputs |
| Q4 family | `Q4_0`, `Q4_1` | 4.5 | Weights (good accuracy/speed) |
| Q5 family | `Q5_0`, `Q5_1` | 5.5 | Weights (better accuracy) |
| Q8 | `Q8_0` | 8.5 | Weights (near-lossless) |
| K-quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` | 2.6–6.6 | Weights (best quality/size tradeoff) |
| I-quants | `IQ1_S`…`IQ4_NL` | 1.5–4.5 | Weights (importance-aware) |

To quantize on the fly:
```cpp
struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 256);
struct ggml_tensor * quant = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 256);
quant = ggml_cpy(ctx, src, quant);  // type conversion via cpy
```

Quantized tensors can be used directly in `ggml_mul_mat()` — the backend
automatically picks the optimized quantized matmul kernel.

---

### 5.9 Numerical Optimization (Training)

ggml includes a lightweight optimization module (`ggml-opt.h`):

```cpp
#include "ggml-opt.h"

// 1) Create optimization context
struct ggml_opt_context * opt = ggml_opt_init(
    ggml_opt_default_params(GGML_OPT_TYPE_ADAMW));

// 2) Build forward graph with ggml_set_output for the loss
struct ggml_cgraph * gf = ggml_new_graph(ctx);
struct ggml_tensor * loss = ...;
ggml_build_forward_expand(gf, loss);

// 3) Run one optimization step (forward + backward + param update)
ggml_opt_step(opt, gf);

// 4) Get updated parameters
struct ggml_tensor * updated = ggml_opt_get_params(opt, gf, 0);

ggml_opt_free(opt);
```

Supported optimizers: `GGML_OPT_TYPE_ADAMW`, `GGML_OPT_TYPE_SGD`.

See `tests/test-opt.cpp` for a complete linear regression example.

---

### 5.10 Performance Tips

1. **Use views, not copies.** `ggml_reshape`, `ggml_view`, and `ggml_permute`
   are zero-copy — they only update stride metadata.
2. **Mark inputs and outputs.** `ggml_set_input()` prevents the allocator
   from overwriting your data; `ggml_set_output()` tells the scheduler
   which results to keep.
3. **Let the backend pick the kernel.** `ggml_mul_mat` with quantized types
   automatically dispatches to the fastest available matmul kernel for that
   backend.
4. **Batch when possible.** Instead of N calls to `ggml_mul_mat` with batch
   1, use a single call with the batch dimension set.
5. **Fuse ops in the graph.** The graph scheduler reorders ops for maximum
   parallelism; declaring more ops in one graph is better than splitting
   across multiple graphs.

---

### 5.11 Linking Your Own Program Against ggml

```bash
# Compile
g++ -I./ggml/include -I./build/ggml/include \
    my_program.cpp \
    -L./build/bin -lggml -lggml-cpu -lggml-base \
    -fopenmp -pthread \
    -o my_program

# Or via CMake
# CMakeLists.txt
add_subdirectory(ggml)     # if building inside llama.cpp
target_link_libraries(my_program PRIVATE ggml ggml-cpu ggml-base)
```

---

## 6. Debugging ggml

### 6.1 Debug build with assertions

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

Debug builds enable:
- All `GGML_ASSERT(...)` checks (active only in Debug)
- No optimization (easy to step through in gdb)
- Address sanitizer: `-DGGML_SANITIZE_ADDRESS=ON`
- Thread sanitizer: `-DGGML_SANITIZE_THREAD=ON`

### 6.2 Disable realloc for debugging

```bash
cmake -B build -DGGML_SCHED_NO_REALLOC=ON
```

Prevents the graph scheduler from reallocating buffers mid-computation.
Makes memory usage deterministic for debugging.

### 6.3 gdb on a test

```bash
gdb --args ./build/bin/test-backend-ops -b CPU -o MUL_MAT
```

### 6.4 Useful environment variables

| Variable | Effect |
|----------|--------|
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | Use `cudaMallocManaged` instead of `cudaMalloc` |
| `GGML_CUDA_NO_PINNED` | Disable pinned (host) memory for CUDA |
| `GGML_BACKEND_PATH` | Path to out-of-tree backend `.so` file |

---

## 7. Architecture: How ggml is tested vs llama.cpp

```
┌──────────────────────────────────────────────────────────┐
│  ggml layer (pure computation)                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │ ggml_init → ggml_new_tensor → ggml_build_forward │    │
│  │             → ggml_graph_compute(backend)         │    │
│  └──────────────────────────────────────────────────┘    │
│  Tests: test-backend-ops, test-rope, test-barrier,        │
│         test-quantize-*, test-alloc, test-opt             │
└──────────────────────────────────────────────────────────┘
                           ↓  (depends on)
┌──────────────────────────────────────────────────────────┐
│  llama.cpp layer (model logic)                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │ llama_model_load → load_tensors → create_tensor   │    │
│  │                  → llama_decode → ggml_graph_*    │    │
│  └──────────────────────────────────────────────────┘    │
│  Tests: test-sampling, test-chat, test-tokenizer-*,      │
│         test-grammar-*, test-model-load-cancel, ...       │
└──────────────────────────────────────────────────────────┘
```

The ggml-only tests verify the tensor computation library in isolation.
They are fast, self-contained, and don't require model files (except
`export-graph-ops` and some `test-opt` modes). Most run in under a second.
