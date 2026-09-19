# SciCompute-Attention

[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17%20%7C%2020-00599C?logo=cplusplus)](https://isocpp.org/)
[![Tests](https://img.shields.io/badge/ctest-17%2F17%20passing-brightgreen)](#测试)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> 面向大模型训练与推理的 GPU 原生高性能 Attention 基础设施，以 FlashAttention、KV Cache、
> Decode Attention 为核心，向 vLLM(C++) Serving 与 RLHF Rollout 提供统一的 Attention Runtime。

从零实现的 CUDA 算子栈：`mma.sync` + `ldmatrix` + `cp.async` 多级流水的 flash 内核、
自研 online softmax、KV Cache 全生命周期与 PagedAttention，并附带 GoogleTest 单测、
compute-sanitizer 运行时检查、可复现 benchmark 与 Nsight 采集脚本。

---

## 特性

- **五级算子覆盖**：Level 0 naive（正确性参考）→ Level 1 tiled（smem 驻留）→ Level 3
  flash（寄存器内 online softmax）→ Level 4 decode（split-K）→ Level 5 paged（页表寻址）。
- **零 N×N 物化**：flash / tiled 的 `workspace_bytes = 0`、`materialized_score_bytes = 0`，
  由单测断言；naive 路径保留物化实现作为 IO 基线。
- **完整 KV Cache 栈**：block-major 存储、LIFO 块分配器、page table 增量上卡、
  多序列生命周期与容量保护（超预算返回错误码而不是 OOM）。
- **可解释调度**：表驱动 Dispatcher + `Explain()`，`allow_fallback=false` 默认禁止静默降级。
- **本地模型可接入**：一条命令拉起带 CUDA runner 的 Ollama 服务并校验连通，
  用真实 Qwen3-4B-Thinking-2507-Q8 的 attention shape（H_q=32 / H_kv=8 / D=128）跑 benchmark。
- **证据可追溯**：每个性能数字都绑定命令、JSON 文件路径与 git sha；未采集项显式标注。

## 架构

```text
           RLHF / Rollout  ·  vLLM(C++)  ·  Python 脚本
                            │  只包含 include/scicompute_attention/*
                            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  SciCompute-Attention                                         │
    │  src/api       形状推导 / 校验 / 公共入口                      │
    │  src/runtime   capability · workspace · scratch · dispatcher  │
    │  src/backends  naive | tiled | flash | decode | paged         │
    │  src/kv_cache  KVCache · BlockManager · PagedKVCache          │
    │  src/cuda_common  mma.sync · ldmatrix · cp.async · numerics   │
    └───────────────────────────────────────────────────────────────┘
                            │  sci::Tensor / Device / Stream / Memory
                            ▼
                  SciComputeInfra（基础设施层，只读依赖）
```

## 快速开始

### 环境要求

| 项 | 要求 |
| --- | --- |
| GPU | NVIDIA，sm_80 及以上（实测 RTX 5070 Laptop / sm_120 / 36 SM / 8 GB） |
| CUDA / CMake | CUDA 13.2（nvcc V13.2.86）/ CMake 3.24+ |
| 编译器 | g++ 15.2（宿主 C++20，CUDA 翻译单元 C++17） |
| Python（可选） | 3.10+（模型接入脚本只用标准库） |
| 上游 | SciComputeInfra（默认 `../SciComputeInfra`，可用 `-DSCI_ATTENTION_SCICOMPUTE_INFRA_ROOT=` 指定） |

### 配置、构建、测试

```bash
bash scripts/configure.sh --build-type Release
bash scripts/build.sh -j"$(nproc)"
bash scripts/test.sh                       # 期望：100% tests passed, 0 tests failed out of 17
```

常用变体：

```bash
bash scripts/test.sh --sanitize            # compute-sanitizer 子集（L4）
bash scripts/test.sh --filter flash        # 只跑名称含 flash 的用例
bash scripts/configure.sh --stub --build-dir build-stub   # CPU-only 构建（GPU 结论无效）
cmake -S . -B build-werror -DSCI_ATTENTION_WERROR=ON      # 警告即错误
```

### 运行示例与基准

```bash
./build/examples/example_attention                       # 走公共 API + Explain 输出
./build/benchmarks/benchmark_naive --suite sanity --runs 50 --warmup 10
./build/benchmarks/benchmark_attention --suite main --runs 100 --warmup 20 --json out.json
python benchmarks/benchmark_sdpa.py --suite main --json sdpa.json       # torch SDPA 对照
./build/examples/rollout_engine_stub --n 4 --max-new 8 --json rollout.json
```

## 接入本地模型（Qwen3-4B-Thinking-2507-Q8）

权重不进 Git（4.28 GB 超过 GitHub 单文件 100 MiB 限制，详见 [model/README.md](model/README.md)）；
用模型卡里的脚本获取权重并校验 sha256，然后用项目脚本拉起服务、验证连通、采集基线。

```bash
# 1) 获取权重（也可自行下载 GGUF 后设置 SCA_GGUF_PATH）
SCA_MODEL_DIR=/path/to/Qwen3-4B-Thinking-2507-Q8 bash model/download_model.sh

# 2) 启动带 CUDA runner 的本地服务（默认 127.0.0.1:11435）
SCA_MODEL_DIR=/path/to/Qwen3-4B-Thinking-2507-Q8 \
SCA_OLLAMA_MODEL=qwen3:4b-thinking-2507-q8_0 bash scripts/serve_model.sh

# 3) 连通性检查 + 基线采集（TTFT / tokens/s / VRAM）
python tools/ollama_baseline.py --check
python tools/ollama_baseline.py --max-tokens 128 --runs 2 \
  --json docs/results/ollama_baseline.json

# 4) 读模型元数据、生成 attention shape 与 KV 预算
SCA_MODEL_DIR=/path/to/Qwen3-4B-Thinking-2507-Q8 python tools/model_probe.py
```

本机实测（RTX 5070 Laptop 8 GB，context 4096，Ollama 0.32.6）：

| 指标 | 值 | 来源 |
| --- | --- | --- |
| 服务连通 | `http://127.0.0.1:11435/api/version` → 0.32.6 | `docs/results/ollama_baseline.json` |
| 模型加载 | ≈ 2.9 s（首次） | 同上 |
| prompt eval（21 token） | ≈ 38 ms | 同上 |
| decode 吞吐 | **53.34 tok/s**（128 token × 2 次均值） | 同上 |
| 服务 VRAM | 4,952 MiB（模型 4,731 MiB） | `nvidia-smi` + `/api/ps` |
| 注意力 shape | 36 层 / H_q=32 / H_kv=8 / D=128 / causal | `docs/results/qwen3_4b_shapes.json` |
| KV 预算 | 144 KiB/token，4096 上下文 576 MiB | 同上 |

> 口径：Ollama 的数字是**完整模型服务**（分词 + 权重 + 采样 + KV）的指标，
> 与本项目 attention kernel 的 benchmark 不是同一件事，两者不混在同一张表比较（`docs/model_integration.md` §6）。

## 性能

RTX 5070 Laptop、CUDA 13.2、fp16、B=1、H_q=H_kv=8、D=128、warmup 20 / runs 100
（原始数据：`benchmarks/results/<date>-<sha>/{attention_main.json,sdpa.json}`，方法学见 [docs/benchmarking.md](docs/benchmarking.md)）：

| 实现 | S=1024 非 causal | S=4096 非 causal | S=8192 非 causal | S=4096 causal |
| --- | --- | --- | --- | --- |
| naive（物化 S） | 15.25 ms / 0.28 TF | 超预算跳过 | 超预算跳过 | 超预算跳过 |
| tiled | 4.38 ms / 1.0 TF | 68.4 ms / 1.0 TF | 264.7 ms / 1.0 TF | 35.4 ms / 7.8 TF |
| **flash（本项目）** | **0.365 ms / 11.8 TF** | **3.95 ms / 17.4 TF** | **14.61 ms / 18.8 TF** | **2.20 ms / 124.9 TF** |
| torch SDPA（flash 后端） | 0.173 ms / 24.8 TF | 2.24 ms / 30.7 TF | 9.54 ms / 28.8 TF | 1.19 ms / 29.0 TF |

flash 相对 naive 提速约 42×、相对 tiled 12–18×，达到 torch SDPA flash 后端的 58–65%；
decode（S_kv=4096）P50 0.244 ms（有效带宽 60–68 GB/s，HBM bound）。
完整表格与差距分析见 [docs/results/benchmark_report.md](docs/results/benchmark_report.md)。

正确性：7 种 head_dim × fp16/bf16 × causal/非 causal × GQA{1,2,4,8} × S∈{1,7,63…8192}
全部对齐 FP64 参考；`compute-sanitizer` 零错误（[docs/results/profiling_summary.md](docs/results/profiling_summary.md)）。

## 目录结构

```text
include/scicompute_attention/   唯一公共契约（上层只允许包含这里）
src/api runtime backends kv_cache cuda_common/   实现分层
tests/{unit,kernel,kv,integration}/              L1–L5 测试
benchmarks/                     naive / attention(flash,tiled,decode) / sdpa 对照
profiling/                      ncu / nsys 脚本、指标清单、CSV 摘要工具
tools/                          arch_probe、gguf_reader、model_probe、ollama_baseline、smem_calc
model/                          模型卡 + 下载脚本（权重不入库）
examples/                       API 示例、rollout stub
docs/                           技术文档与结果（见下）
scripts/                        configure / build / test / bench / profile / serve_model
TASK.md .agent/                 过程记录（state / decisions / memory / failures）
```

## 文档

| 主题 | 文档 |
| --- | --- |
| 架构与文件职责 | [docs/architecture.md](docs/architecture.md) |
| 数学定义 / online softmax | [docs/attention_math.md](docs/attention_math.md) · [docs/online_softmax.md](docs/online_softmax.md) |
| flash 内核与 tile 标定 | [docs/flash_attention.md](docs/flash_attention.md) · [docs/tile_config.md](docs/tile_config.md) · [docs/kernel_design.md](docs/kernel_design.md) |
| KV Cache / PagedAttention | [docs/kv_cache.md](docs/kv_cache.md) · [docs/paged_kv_cache.md](docs/paged_kv_cache.md) |
| Prefill vs Decode | [docs/prefill_decode.md](docs/prefill_decode.md) |
| 基准方法 / 结果 | [docs/benchmarking.md](docs/benchmarking.md) · [docs/results/benchmark_report.md](docs/results/benchmark_report.md) |
| Profiling / Roofline | [docs/profiling.md](docs/profiling.md) · [docs/roofline.md](docs/roofline.md) |
| 数值稳定性 | [docs/numerical_stability.md](docs/numerical_stability.md) |
| 模型接入 | [docs/model_integration.md](docs/model_integration.md) · [model/README.md](model/README.md) |
| 上/下游集成 | [docs/vllm_integration.md](docs/vllm_integration.md) · [docs/rollout_interface.md](docs/rollout_interface.md) · [docs/upstream_relationship.md](docs/upstream_relationship.md) |
| 环境与侦察 | [docs/env_report.md](docs/env_report.md) · [docs/00-recon.md](docs/00-recon.md) |
| 故障排查 | [docs/troubleshooting.md](docs/troubleshooting.md) |

## 已知限制与 Roadmap

本次交付未包含（各项均有文档说明未完成原因与复现命令）：

1. **Triton 双轨**（[docs/triton_optimization.md](docs/triton_optimization.md)）——仓库内不含任何未测量的 Triton 数字。
2. **Python `_core`（pybind11）绑定**——CMake 开关与目录已预留。
3. **vLLM(C++) 适配层与 A/B**（[docs/vllm_integration.md](docs/vllm_integration.md)）——接口契约已冻结。
4. **真实权重 Q/K/V parity**（[docs/model_integration.md](docs/model_integration.md)）——元数据/形状/预算已完成，权重级 parity 需要逐层反量化 + RoPE 前向。
5. **ncu/nsys 原始采集**——脚本就绪，需 GPU 独占运行。

Roadmap：prefix caching、block 共享/COW、量化 KV（fp8/int8）、融合 varlen kernel、
多卡/多流、GPU 侧 dequant-fused attention、Python vLLM 后端。

## 引用与致谢

- FlashAttention / FlashAttention-2（Dao et al.）：本项目为独立实现，未复制其代码，推导见 `docs/online_softmax.md`。
- vLLM 的 PagedAttention 与 KV block 概念：实现了同构的 block 布局，未复用其代码。
- CUTLASS / PTX ISA 文档：仅用于理解指令语义；指令级行为以本项目 `tools/arch_probe` 实测探针为准。
- 上游 SciComputeInfra 提供 Tensor / Device / Stream / Memory / Benchmark 基础设施。

## 许可证

本项目以 [MIT](LICENSE) 发布。模型权重遵循 Apache-2.0（详见 [model/README.md](model/README.md)）。

