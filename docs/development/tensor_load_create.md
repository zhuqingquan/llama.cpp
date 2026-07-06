# Tensor Load and Create Pipeline

How `ggml_tensor` gets created, assigned to a backend memory buffer, and loaded
with weight data during model initialization.

## Overview

```
ggml_init(params)             → 只分配 tensor 元数据的内存池
                                      ↓
llama_model_load_from_file()  → 决定使用哪些设备（GPU/CPU）
                                      ↓
load_tensors()                → 为每个 tensor 选择合适的 buffer type
                                      ↓
create_tensor()               → 分配 backend buffer + 创建 tensor 元数据
                                      ↓
load_data_for()               → 从 .gguf 文件加载权重数据到 tensor 内存
```

## 1. ggml_init() — 元数据池，无关 backend

```c
struct ggml_init_params {
    size_t mem_size;   // metadata 内存池大小
    void * mem_buffer; // 如果 NULL 则内部 malloc
    bool   no_alloc;   // tensor data 不分配内存
};

struct ggml_context * ggml_init(struct ggml_init_params params);
```

`ggml_init()` (see `ggml/src/ggml.c`) 只做一件事：分配一个 **元数据内存池**
(`ggml_context->mem_buffer`)，用于后续 `ggml_new_tensor()` 创建的 tensor 对象
本身（shape、type、name、src 指针等）。**完全不涉及 backend 选择**。

- `no_alloc = true`: tensor 创建时 `data = NULL`，后续由 backend 分配
- `no_alloc = false`: ggml 从元数据池末尾分配 `data` 空间（传统用法，当前 llama.cpp
  不使用此模式）

> **结论**: `ggml_init()` 的参数只控制元数据内存池大小。Backend 的决定权在
> llama.cpp 上层的模型加载流程。

## 2. Backend 注册 — 程序启动时

```
common_params_init() / main()
  → common_arg_parse()
    → ggml_backend_load_all()
```

`ggml_backend_load_all()` (见 `ggml/src/ggml-backend-reg.cpp`) 按顺序尝试加载
动态库 `.so`。**只要 `.so` 存在就加载，不检查硬件兼容性**：

| 顺序 | Backend    | 动态库              |
|------|-----------|---------------------|
| 1    | blas      | `ggml-blas.so`      |
| 2    | zendnn    | `ggml-zendnn.so`    |
| 3    | cann      | `ggml-cann.so`      |
| 4    | cuda      | `ggml-cuda.so`      |
| 5    | hip       | `ggml-hip.so`       |
| 6    | metal     | `ggml-metal.so`     |
| 7    | rpc       | `ggml-rpc.so`       |
| 8    | sycl      | `ggml-sycl.so`      |
| 9    | vulkan    | `ggml-vulkan.so`    |
| 10   | opencl    | `ggml-opencl.so`    |
| 11   | openvino  | `ggml-openvino.so`  |
| 12   | cpu       | 内置（总是可用）     |

也可通过环境变量 `GGML_BACKEND_PATH` 加载外部 backend。

每个 backend 在 `.so` 加载后会注册自己的设备到全局设备列表。

## 3. Device 发现 — 模型加载时

```
llama_model_load_from_file(path_model, params)
  → llama_model_load_from_file_impl(..., params)
```

见 `src/llama.cpp:890`。

### 3.1 用户指定 device

```cpp
if (params.devices) {
    // 用户提供了设备列表 → 直接使用
    if (split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        // 包装为 Meta device
        model->devices = { ggml_backend_meta_device(devs, n, ...) };
    } else {
        for each dev in params.devices:
            model->devices.push_back(dev);
    }
}
```

### 3.2 自动发现 device（默认）

```cpp
if (params.devices == NULL) {
    // 枚举所有已注册设备，按类型分类
    for each ggml_backend_dev_get(i):
        switch ggml_backend_dev_type(dev):
            GGML_BACKEND_DEVICE_TYPE_CPU:   // 跳过（统一处理）
            GGML_BACKEND_DEVICE_TYPE_ACCEL: // 跳过
            GGML_BACKEND_DEVICE_TYPE_GPU:   // 加入 gpus[]
            GGML_BACKEND_DEVICE_TYPE_IGPU:  // 加入 igpus[]
    // 最终优先级：RPC servers > GPU (discrete) > iGPU (集成显卡)
    model->devices = rpc_servers + gpus + (gpus.empty() ? igpus : [])
}
```

如果是 `LLAMA_SPLIT_MODE_NONE`（单卡模式），只保留 `params.main_gpu` 指定的设备，
其余移除。

### 3.3 关键参数

`llama_model_params` (见 `include/llama.h:286`) 中控制 backend 的字段：

| 字段 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `devices` | `ggml_backend_dev_t *` | `NULL` | 设备列表，NULL=自动发现 |
| `n_gpu_layers` | `int32_t` | 0 | 放 GPU 的层数，负数=全部 |
| `split_mode` | `llama_split_mode` | `NONE` | 单卡/层并行/行并行 |
| `main_gpu` | `int32_t` | 0 | 单卡模式用哪张卡 |
| `tensor_split` | `const float *` | `NULL` | 多卡分配比例 |
| `tensor_buft_overrides` | `...` | `NULL` | per-tensor buffer 覆盖 |
| `no_host` | `bool` | false | 禁用 host (pinned) buffer |

## 4. Buffer Type 列表构建

在 `load_tensors()` (见 `src/llama-model.cpp:2954`) 中，为每个设备生成
buffer type 优先级链。

### 4.1 CPU 端 — `make_cpu_buft_list()`

```cpp
static buft_list_t make_cpu_buft_list(devices, use_extra_bufts, no_host) {
    buft_list_t list;

    // 1) ACCEL 设备（如 BLAS, OpenBLAS）
    for each ACCEL-type device:
        list.push_back(dev, ggml_backend_dev_buffer_type(dev));

    // 2) Host buffer (pinned memory, 如 cudaMallocHost)
    if (!no_host) {
        for each GPU device:
            host_buft = ggml_backend_dev_host_buffer_type(dev);
            if (host_buft) { list.push_back(dev, host_buft); break; }
    }

    // 3) Extra bufts (GPU weight repack 优化)
    if (use_extra_bufts):
        list.push_back(...extra bufts...);

    // 4) CPU 兜底 (posix_memalign / malloc)
    list.push_back(cpu_dev, ggml_backend_dev_buffer_type(cpu_dev));

    return list;
}
```

### 4.2 GPU 端 — `make_gpu_buft_list()`

```cpp
static buft_list_t make_gpu_buft_list(dev, split_mode, tensor_split) {
    buft_list_t list;

    // 1) Split buffer（LLAMA_SPLIT_MODE_ROW 时，矩阵按行切分到多 GPU）
    if (split_mode == LLAMA_SPLIT_MODE_ROW):
        list.push_back(split_buffer_type);

    // 2) Device default buffer (cudaMalloc / clCreateBuffer 等)
    list.push_back(dev, ggml_backend_dev_buffer_type(dev));

    // 3) Extra bufts（GPU 端额外优化）
    list.push_back(...extra bufts...);

    // 最后，CPU 列表作为 fallback 追加在后面
    // 见 load_tensors() 中:
    // buft_list.insert(list.end(), cpu_buft_list.begin(), cpu_buft_list.end());
    return list;
}
```

## 5. Layer 到设备的映射

```cpp
const int i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0);
const int act_gpu_layers = min(n_gpu_layers, n_layer + 1);

for (int il = 0; il < n_layer; ++il) {
    if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
        // 不在 offload 范围内 → CPU
        layer_dev = { cpu_dev, &cpu_buft_list };
    } else {
        // 根据 tensor_split 比例映射到某张 GPU
        int gpu_idx = upper_bound(splits, (il - i_gpu_start) / act_gpu_layers);
        layer_dev = { gpus[gpu_idx], &gpu_buft_list[gpus[gpu_idx]] };
    }
}
```

input 层默认始终在 CPU（几乎没有 offload 收益）。output 层按 `get_layer_buft_list(n_layer)` 分配。

## 6. Tensor 创建 — `create_tensor()`

当模型架构代码（如 `llama_model::load_tensors()`）为每一层的权重调用
`ml.create_tensor(tn, ne, flags)` 时：

```cpp
ggml_tensor * create_tensor(
    hparams, cpu_buft_list, dev_input, dev_output, buft_list_layer,
    tn, ne, flags)
{
    // 1. 确定使用哪个 buft_list
    //    优先级: tensor_buft_overrides > layer 的 buft_list > input/output/cpu

    // 2. 在 buft_list 中遍历，找到第一个能成功分配内存的 buft
    for (每个 buft in buft_list) {
        buffer = ggml_backend_buft_alloc_buffer(buft, size);
        if (buffer) break;  // 分配成功
    }

    // 3. 创建 ggml_tensor 对象（元数据在 ggml_context 的内存池中）
    tensor = ggml_new_tensor_impl(ctx, type, ne, ...);
    tensor->data   = buffer->base;      // 后端内存指针
    tensor->buffer = buffer;            // 后端 buffer 对象

    // 4. 注册到 backend 的 tensor 管理
    ggml_backend_buffer_init_tensor(buffer, tensor);

    return tensor;
}
```

### 6.1 Buffer 分配的实际后端实现

| Buffer Type | 分配函数 | 实际 API |
|-------------|---------|---------|
| CPU | `ggml_backend_cpu_buffer_type_alloc_buffer` | `posix_memalign(ptr, 64, size)` |
| CUDA | `ggml_backend_cuda_buffer_type_alloc_buffer` | `cudaMalloc(&ptr, size)` |
| CUDA Host | `ggml_backend_cuda_host_buffer_type_alloc_buffer` | `cudaMallocHost(&ptr, size)` |
| CUDA Split | `ggml_backend_cuda_split_buffer_type_alloc_buffer` | 多 GPU 各 `cudaMalloc` |
| Vulkan | `ggml_backend_vk_buffer_type_alloc_buffer` | `vkAllocateMemory` |
| HBM (CPU) | `ggml_backend_cpu_hbm_buffer_type_alloc_buffer` | `hbw_posix_memalign` |

## 7. 权重加载 — `load_data_for()`

Tensor 创建完成后，从 `.gguf` 文件读取权重数据：

```cpp
void load_data_for(struct ggml_tensor * cur) const {
    // 1. mmap 文件（如果 use_mmap=true）
    //    直接映射 .gguf 中的数据到进程地址空间

    // 2. 如果 tensor 在 CPU buffer 上且 mmap 可用
    //    → tensor->data 直接指向 mmap 区域，零拷贝

    // 3. 如果 tensor 在 GPU 或没有 mmap
    //    → 从文件读取 → cudaMemcpyAsync(..., HostToDevice)
    //    或 : read(fd, tensor->data, size)
    //         ↑ 这是 linux read()，可能触发块设备 DMA
}
```

当使用 `O_DIRECT` 打开文件时，`read()` 调用的 DMA 路径：

```cpp
// llama-mmap.cpp:190
fd = open(fname.c_str(), O_RDONLY | O_DIRECT);
// ...
ssize_t ret = ::read(fd, ptr, to_read);  // 块设备 DMA → ptr
// 如果 DMA 控制器不能访问 ptr（非对齐等）
if (ret == -1 && (errno == EFAULT || errno == EINVAL)) {
    // 回退到 buffered I/O（通过 kernel page cache 中转）
    close(fd);
    init_fp("rb");  // 改为 FILE* buffered I/O
}
```

## 8. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. ggml_init(params)                                                    │
│    只分配 tensor 元数据内存池 (ggml_context)                             │
│    tensor->data = NULL, tensor->buffer = NULL                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. llama_model_load_from_file(path_model, params)                       │
│    a) 设备发现 (根据 params.devices / params.split_mode)                │
│    b) load_tensors():                                                   │
│       - make_cpu_buft_list() → CPU 端 buffer 类型优先级链                │
│       - make_gpu_buft_list() → GPU 端 buffer 类型优先级链                │
│       - get_layer_buft_list() → 每层分配到哪个设备                        │
│       - create_tensor() → 关键路径:                                     │
│           ml.create_tensor(buft_list, tn, ne, flags)                    │
│           ↓                                                             │
│         for each buft in buft_list:                                     │
│           buf = ggml_backend_buft_alloc_buffer(buft, size)              │
│                → GPU: cudaMalloc                                       │
│                → CPU: posix_memalign                                   │
│                → Host: cudaMallocHost (pinned, GPU DMA 可直达)          │
│           if (buf) break                                                │
│           ↓                                                             │
│         tensor->data = buf->base                                        │
│         tensor->buffer = buf                                            │
│    c) load_data_for(tensor):                                            │
│       - mmap 直接映射 (零拷贝) 或 read() + cudaMemcpyAsync              │
└─────────────────────────────────────────────────────────────────────────┘
```
