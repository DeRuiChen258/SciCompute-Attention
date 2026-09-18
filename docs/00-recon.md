# Phase 0 侦察报告（上游审计 + 复用边界）

> 执行时间：2026-09-19
> 依据：`Prompt/SciCompute-Attention 提示词.md` §2、§3
> 原则：只读引用三个上游仓库；本项目不修改上游（需要修改时只提供 `upstream/patches/`）

## 1. 版本锚点（Phase 0 实测）

| 仓库 | HEAD | 备注 |
| --- | --- | --- |
| `SciComputeInfra` | `61492cccf692d0891e376df5e100f31373f7a010` | 基础设施层（Tensor/Device/Stream/Memory/Benchmark） |
| `vllm` | `f6d0ea9ee7c38e4575fab1bd73b3aaa374607adc` | C++ 服务层（`vllm::vllm` STATIC + `vllm_server`） |
| `RLHF` | `1f7ba72bff9abc023629053b48f31ddab182315b` | `mini-rlhf-stack`，Rollout/训练 |
| `SciCompute-Attention` | 本仓库（Phase 0 首次提交） | 中间层，本次交付对象 |

## 2. SciComputeInfra 资产清点（证据 = 文件:行号）

| 能力 | 证据 | 结论 |
| --- | --- | --- |
| 构建入口 | `CMakeLists.txt:1-3` `project(SciComputeInfra VERSION 0.1.0 LANGUAGES CXX CUDA)` | 宿主 C++20 / CUDA C++17；`SCI_BUILD_TESTS`/`SCI_BUILD_BENCHMARKS` 默认 OFF |
| 架构变量 | `CMakeLists.txt:20-22` `SCI_CUDA_ARCHITECTURES "75;89;90;120"` | 含 120，可直接 `add_subdirectory` |
| 目标拓扑 | `src/CMakeLists.txt:14-45` | `sci_compute` 是 INTERFACE 聚合；单点目标为 `sci_core/sci_device/sci_memory/sci_tensor/sci_math/sci_benchmark` |
| Tensor API | `include/tensor/tensor.hpp:21-100` | `Tensor(shape, dtype, device[, external_data, owns_data])`，view/reshape/slice 非拷贝，`to(DeviceType)` 返回 `Result<Tensor>` |
| 状态码 | `include/core/status.hpp:12-70` | `StatusCode` 含 `kCudaOutOfMemory/kInvalidArgument/kNotImplemented`；**无 `Result<void>`**（结论：空返回统一用 `Status`） |
| 流与事件 | `include/device/stream.hpp:18-60` | `Stream::Create/GetCurrent/SetCurrent/handle/synchronize`；`Event::elapsed_ms` |
| 设备抽象 | `include/device/device.hpp:26-60` | 纯虚 `Device`：allocate/copy/memset/synchronize/free_memory |
| 内存 | `include/memory/allocator.hpp:14-45` | `Allocator` 接口 + `HostAllocator`；`Stats{peak_bytes,...}` |
| 基准框架 | `include/benchmark/benchmark_runner.hpp:11-60` | `BenchmarkRunner` + `Timer` + `percentile/mean/stddev` |
| 数学（CPU 参考） | `include/math/softmax.hpp:12-35` | `softmax_stable` / `ref::softmax_f32` |
| 遗留 Attention | `cuda/kernels/attention.cuh:19,31` 只有声明 | `scaled_dot_product_attention_kernel` / `launch_flash_attention` **无实现、未进 CMake**（见 UP-002） |
| 遗留第二套 GPU 抽象 | `include/gpu/cuda_kernels.hpp:114` `scaled_dot_product_attention(...)` | 与 `sci::Tensor` 路径并存，边界不清（UP-004），本项目不复用 |

### 2.1 复用/重写判定（逐条）

| 能力 | 上游现状 | 本项目动作 | 判定依据 |
| --- | --- | --- | --- |
| Tensor/Shape/DType/Device/Result | 完整 | **复用（只读依赖）** | `include/tensor/tensor.hpp`、`include/core/status.hpp` |
| Stream/Event | 完整 | **复用**：所有公开 API 接受 `sci::Stream*` | `include/device/stream.hpp:18` |
| 内存分配 | 完整 | **复用** 于 workspace 与 KV 底层缓冲 | `include/memory/allocator.hpp` |
| Softmax（CPU 参考） | 可用 | **复用** 作单测 oracle，不作 GPU 主路径 | `include/math/softmax.hpp` |
| BenchmarkRunner | 可用 | **复用**；字段缺口在 `benchmarks/common/bench_export.hpp` 内补齐，不改上游 | §12.4 字段要求多于上游 `BenchmarkCase::Result` |
| `cuda/kernels/attention.cuh` | 只有声明 | **不复用**，本项目自研实现 | UP-002 |
| `include/gpu/*`（GpuTensor 路径） | 遗留 | **不复用、不扩展**，文档标注 legacy | UP-004 |
| Attention kernel / KV Cache / Paged KV / Dispatcher | 不存在 | **本项目实现** | — |
| vLLM(C++) attention 原型 | 存在但缺陷明确 | **作为 A/B 基线**，本项目提供替换实现 | §5 |

## 3. vLLM(C++) 资产清点

| 项 | 证据 | 内容 |
| --- | --- | --- |
| 库目标 | `vllm/CMakeLists.txt:18-19` | `add_library(vllm STATIC)` + `vllm::vllm` ALIAS |
| 入口结构体 | `include/vllm/attention/flash_attention.hpp:8-24` | `FlashAttentionParams`：FP32、BSHD、含 `num_kv_heads`/`scale`/`is_causal`/`stream` |
| 入口函数 | `include/vllm/attention/flash_attention.hpp:27` | `void flash_attention_forward(const FlashAttentionParams&)` |
| 现有实现 | `src/attention/flash_attention.cu`（81 行） | 单 warp（`:72` `dim3 block(32)`）；两遍遍历 K/V（`:33` 与 `:52`）；`:64` `out[...] += ...` 未初始化；`:19` GQA 用 `h % num_kv_heads`；仅 FP32 |
| KV Cache | `include/vllm/memory/kv_cache.hpp:11-46` | 每 layer 连续 `[num_blocks, block_size, num_kv_heads, head_dim]`，提供 `k_ptr/v_ptr/clear_block/total_bytes` |

**结论**：集成目标是这个 C++ vLLM（本机无 Python vLLM 安装）。适配层需把 FP32/BSHD 转成项目内部的 FP16/BF16 + BHSD，
并显式记录精度迁移；KV 侧因布局同构，优先走零拷贝指针映射。

## 4. RLHF 资产清点

| 项 | 证据 | 内容 |
| --- | --- | --- |
| 目标 | `RLHF/CMakeLists.txt:54,60,67,70` | `rlhf_core`、`rlhf_cuda`、`train_ppo`、`rollout_server` |
| Rollout 接口 | `rlhf/rollout_server.h:10-22` | `RolloutServer::serve/update_weights/load_checkpoint` |
| GPUActor | `rlhf/models/gpu_policy_model.h` + `rlhf/rollout/*` | `rollout(prompts)`，默认 `max_seq_len=2048` |
| KV Cache | `rlhf/rollout/kv_cache.h:9-30` | **CPU dense** `std::vector`，注释布局 `[num_layers, 2, batch, num_heads, seq_len, head_dim]` |
| 缺陷 1 | `RLHF/CMakeLists.txt:12` | `CMAKE_CUDA_ARCHITECTURES "80;86;89;90"` 不含 120，本机原生运行需覆盖 |
| 缺陷 2 | `RLHF/CMakeLists.txt:49-52` | `RLHF_CUDA_SOURCES` 只有 `fused_loss_kernel.cu` + `sampling_kernel.cu`，`cuda_ops/gpu_ops.cu` **未加入**（即使有 GPU 实现也不会被编译） |
| 缺陷 3 | `rlhf/models/gpu_policy_model.cpp` | GPU 路径无 attention kernel |

## 5. 三仓接口契约（落地文件）

```text
RLHF/Rollout ──① RolloutServer::serve / GPUActor::rollout──▶ requires Attention + KV Cache
   ▲                                                                  │
   │ ③ examples/rollout_engine_stub.cpp（直连路径）                     ▼
vLLM(C++) ──② vllm::flash_attention_forward(FlashAttentionParams)──▶ sca::flash_attention
   ▲                                                                  │
   │ vllm_backend/sca_*_adapter.{hpp,cpp}                             ▼
SciCompute-Attention ──④ sca::Tensor/Device/Stream/Memory/Benchmark──▶ SciComputeInfra
```

| 契约 | 本项目落地文件 |
| --- | --- |
| 与 SciComputeInfra（向下） | `cmake/SciComputeInfra.cmake`、`include/scicompute_attention/*`、`src/runtime/launcher.cpp`、`src/runtime/workspace.cpp` |
| 与 vLLM（向上） | `vllm_backend/*`、`benchmarks/benchmark_vllm.cpp`、`docs/vllm_integration.md` |
| 与 RLHF（向上） | `examples/rollout_engine_stub.cpp`、`docs/rollout_interface.md` |

依赖方向为单向：`RLHF → vLLM → SciCompute-Attention → SciComputeInfra`，禁止反向。

## 6. 风险清单

| ID | 风险 | 影响 | 缓解 |
| --- | --- | --- | --- |
| R-01 | 8 GB 显存（实测 8,177,909,760 B）且 Ollama 常驻约 4.9 GiB | 大 shape benchmark 直接 OOM | benchmark 前置显存预估 + 跳过策略写入 JSON；实验前 `nvidia-smi` |
| R-02 | `sci_bridges` 硬编码 `/usr/local/cuda-13.2/include`（UP-001） | 上游 `sci_compute` 聚合目标在非默认 CUDA 路径下构建失败 | 只链接精确目标，不链接 `sci_compute`（见 `.agent/decisions.md` D-004） |
| R-03 | sm_120 无 wgmma/tcgen05 | 不能照搬 Hopper/Blackwell 教程 | 主路径固定 `mma.sync m16n8k16` + `cp.async`/TMA，探针已复现（`docs/env_report.md`） |
| R-04 | Triton 动态 smem 上限 101,376 B | 大 tile 配置编译期 `OutOfResources` | autotune 前置 smem 模型校验（`python/tests/test_triton_config.py`） |
| R-05 | 真实模型为 Q8_0（4.28 GB），全量反量化 fp16 ≈ 8.04 GB > 显存 | 一次性反量化必然 OOM | 逐层流式处理，单次峰值 ≤ 1 GiB |
| R-06 | 上游 vLLM 原型数值不可信（`out` 未初始化、GQA 取模） | A/B 时不能把原型缺陷当作本项目误差 | A/B 以 torch FP32 为参考；上游缺陷单独登记（§15.7） |

## 7. 待确认问题

1. 是否允许向 `SciComputeInfra` 提交 `upstream/patches/0001-bridges-cuda-include.patch`？（默认不应用）
2. 真实权重 parity（Q/K/V 落盘）是否需要在本次交付内完成？（当前按 Phase 12.5 目标 2 执行，若受限则登记 Known Issue）

