# SciCompute-Attention

> **SciCompute-Attention —— 面向大模型训练与推理的 GPU 原生高性能 Attention 基础设施，
> 以 FlashAttention、KV Cache、Decode Attention 为核心，
> 向 vLLM(C++) Serving 与 RLHF Rollout 提供统一的 Attention Runtime。**

本仓库是从零实现的 Attention 子系统：自研 CUDA kernel（`mma.sync` + `cp.async` + `ldmatrix`）、
自研 online softmax、完整 KV Cache 生命周期与分页注意力，配套 GoogleTest 单测、
compute-sanitizer 运行时检查、可复现 benchmark 与 Nsight profiling 流程。

---

## 1. 项目定位

三条能力主线：

1. **科学计算 / IO 分析**：把「注意力为什么需要 IO-aware」量化为可复现的数字（分数矩阵字节数、
   有效带宽、算术强度、roofline 定位）。
2. **Attention 算子**：Level 0（naive，正确性参考）→ Level 1（tiled，smem 分块）→
   Level 3（flash，寄存器内 online softmax）→ Level 4（decode split-K）→ Level 5（paged）。
3. **LLM Infra 集成**：稳定 C++ 公共 API、Dispatcher/Explain、KV Cache 与 page table，
   向 vLLM(C++) 与 RLHF Rollout 暴露统一 Runtime。

## 2. 总体架构

```text
                RLHF / Rollout（业务层，只读引用 + 示例 stub）
                            │  serve(prompts) / GPUActor::rollout
                            ▼
                vLLM(C++)（服务层，A/B 对照 + 适配层目标）
                            │  vllm::flash_attention_forward(FlashAttentionParams)
                            ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  SciCompute-Attention（本项目）                              │
    │  include/scicompute_attention/*   ← 唯一稳定契约             │
    │  src/api        → 形状推导 / 校验 / dispatch                 │
    │  src/runtime    → capability / workspace / scratch / launcher│
    │  src/backends   → naive | tiled | flash | decode | paged     │
    │  src/kv_cache   → KVCache | BlockManager | PagedKVCache      │
    └──────────────────────────────────────────────────────────────┘
                            │  sci::Tensor / Device / Stream / Memory
                            ▼
                SciComputeInfra（基础设施层，只读依赖）
```

三仓关系与依赖方向（禁止反向依赖）见 `docs/upstream_relationship.md`；
上游问题登记（含文件:行号证据）见 `upstream/notes.md`。

## 3. 核心能力一览

| Level | 后端 | 状态 | 关键指标（本机实测） |
| --- | --- | --- | --- |
| 0 | `naive` | 完成 | 正确性参考；分数矩阵 FP32 物化（2 GiB @B1·H32·S4096·D128），0.25–0.3 TFLOPS |
| 1 | `tiled` | 完成 | smem 驻留 S，**workspace 0 B**；1.0–7.8 TFLOPS |
| 3 | `flash` | 完成 | `mma.sync m16n8k16` + `cp.async` 双缓冲 + 寄存器 online softmax；**workspace 0 B**；11.8–18.8 TFLOPS（非 causal）、124.9 TFLOPS（causal, S=4096） |
| 4 | `decode` | 完成 | split-K ∈ {1..16}，S_q=1；S_kv=4096 时 P50 ≈ 0.244 ms |
| 5 | `paged` | 完成 | page-table 间接寻址，乱序 page 正确；复用 split-K 归约 |
| — | KV Cache | 完成 | block-major `[blocks][block_size][heads][dim]`，LIFO 分配器，O(1) 分配/回收 |
| — | Dispatcher | 完成 | 表驱动选择 + `Explain()`；`allow_fallback=false` 默认禁止静默降级 |

## 4. 快速开始

### 4.1 环境要求

```text
GPU      : NVIDIA（sm_80+，本项目实测 RTX 5070 Laptop / sm_120 / 36 SM / 8 GB）
CUDA     : 13.2（nvcc V13.2.86）      CMake : >= 3.24（实测 3.31.6）
编译器   : g++ 15.2（宿主 C++20，CUDA 翻译单元 C++17）
Python   : 3.12（conda env cuda_132）用于 Python 侧脚本
依赖     : GoogleTest 1.17（libgtest-dev）、可选 Google Benchmark、可选 pybind11
上游     : SciComputeInfra（默认 ../SciComputeInfra）
```

### 4.2 配置 / 构建 / 测试（三条命令）

```bash
bash scripts/configure.sh --build-type Release
bash scripts/build.sh -j"$(nproc)"
bash scripts/test.sh                     # 13 个 CTest 用例（unit 5 + kernel 5 + kv 3）
```

预期输出（末行）：

```text
100% tests passed, 0 tests failed out of 13
```

其他常用入口：

```bash
bash scripts/test.sh --sanitize          # compute-sanitizer 子集（L4）
bash scripts/test.sh --filter flash      # 只跑名字含 flash 的用例
bash scripts/bench.sh --suite sanity     # 快速基准，结果落盘 benchmarks/results/<date>-<sha>/
bash scripts/configure.sh --stub --build-dir build-stub   # CPU-only 构建（GPU 结论无效）
```

### 4.3 运行示例

```bash
./build/examples/example_attention                     # API 用法 + Explain 输出
./build/benchmarks/benchmark_naive --suite sanity --runs 50 --warmup 10
./build/benchmarks/benchmark_attention --suite sanity --runs 50 --warmup 10
python tools/model_probe.py                            # 真实模型 shape + KV 预算
```

## 5. 目录结构

```text
include/scicompute_attention/   唯一公共契约（上层只允许包含这里）
src/api/                        形状推导、校验、公共 API 实现
src/runtime/                    capability / workspace / scratch pool / launcher / dispatcher
src/cuda_common/                mma.sync、ldmatrix、cp.async、数值与错误检查工具
src/backends/{naive,tiled,flash,decode,paged}/
src/kv_cache/                   KVCache / BlockManager / PagedKVCache
tests/{unit,kernel,kv,integration}/
benchmarks/                     naive / attention(flash,tiled,decode) / sdpa 对照
profiling/                      ncu / nsys 脚本与指标清单
tools/                          arch_probe、gguf_reader、model_probe、smem_calc 等
docs/                           技术与结果文档（见第 20 节索引）
upstream/                       上游问题登记与（可选）补丁
TASK.md / .agent/               过程记录（state / decisions / memory / failures）
```

## 6. Attention 数学原理

```text
S = scale * Q K^T           （FP32 累加，默认 scale = 1/sqrt(D)）
P = softmax(S, axis=-1)     （行最大值先行归约，减最大值保证不溢出）
O = P V
```

掩码约定（唯一口径，右下对齐）：`visible(i,j) <=> j <= i + diag`，`diag = S_kv - S_q`；
`S_q = S_kv` 时退化为 `j <= i`，`S_q = 1`（decode）时等价于「可见全部缓存 token」。
被掩码位置取 `-inf`，全掩码行输出 0 而非 NaN。定义式与复杂度推导见 `docs/attention_math.md`。

## 7. FlashAttention 原理（本项目实现要点）

```text
每 CTA：Q tile -> ldmatrix -> 寄存器 A 片段；m,l = -inf,0；O = 0
for each K/V tile:
    整块不可见 => break（无逐元素分支）
    cp.async 预取下一块（kStages=2 双缓冲）
    S = Q_frag @ K_frag^T * scale            （mma.sync m16n8k16）
    对角线块逐元素置 -inf
    m_new = max(m, rowmax(S)) ; alpha = exp2((m - m_new) * log2e)
    P = exp2((S - m_new) * log2e)            （beta 吸收入 P）
    l = alpha * l + rowsum(P) ; O = alpha * O + P @ V
O = O / l
```

关键点：`S/P` 不出现在 HBM（`workspace_bytes = 0`、`materialized_score_bytes = 0`），
K/V 用 16 B `cp.async` 搬运，V 通过 `ldmatrix.x4.trans` 取得 B 片段，
行归约用 warp shuffle（**不使用 atomicAdd**）。推导与流水结构见 `docs/flash_attention.md`。

## 8. Kernel 设计

| 后端 | tile 配置 | smem/block | 寄存器（估算） |
| --- | --- | --- | --- |
| naive | 无 tile，1 thread/element | 0 | ~20 |
| tiled | BM=64, BN=64, warps=4（fp32 用 BN=32） | 70400 B (D=128, fp16) | ~90 |
| flash | D<=128: BM=64/BN=64/4 warps/2 stages；D>128: splits_d=2 | 25600–87040 B | ~160（D=128） |
| decode | warp per (b,h,split)，lane 分片 D/32 | 0 | <64 |
| paged | 同 decode + page table 查表 | 0 | <64 |

线程映射、同步点、边界判定与寄存器预算表见 `docs/kernel_design.md`；
tile 选择表在 `src/backends/flash/flash_tile_config.hpp`（表驱动，禁止魔法数）。

## 9. Memory Hierarchy（本机实测）

| 层次 | 容量/带宽 | 备注 |
| --- | --- | --- |
| 寄存器 | 65,536 / SM | flash 的 O 累加器（D=128 时 64 个 float/线程） |
| Shared Memory | 102,400 B / SM，单 block 上限 101,376 B（opt-in） | 行 padding 8 half(16 B) 消除 8 路 bank conflict |
| L2 | 32 MiB | decode 的分片窗口按 L2 友好度选择 |
| HBM | 8 GiB / 128-bit | 带宽标定与 roofline 见 docs/roofline.md |

## 10. KV Cache

```text
K/V storage : [num_layers][num_blocks][block_size][num_kv_heads][head_dim]
slot        : slot = physical_block * block_size + offset_in_block
page table  : [max_num_seqs][max_blocks_per_seq]，-1 表示空槽，按 revision 增量上卡
budget      : bytes = 2 * layers * blocks * block_size * kv_heads * head_dim * sizeof(dtype)
```

真实模型（Qwen3-4B-Thinking-2507-Q8，36 层 / 8 KV heads / D=128，144 KiB/token）：

| 上下文 | fp16 KV 大小 | 可行性（8 GB 机器） |
| --- | --- | --- |
| 1024 | 144 MiB | 可行 |
| 4096 | 576 MiB | 可行（需先停 Ollama） |
| 8192 | 1.13 GiB | 需停 Ollama + batch=1 |
| 32768 | 4.5 GiB | 与 4B 权重同机不可行 |

## 11. Paged KV Cache

`PagedKVCache = KVCache（block-major 存储）+ BlockManager（LIFO 独占分配）+ page table`；
`paged_attention` 复用 split-K decode kernel，仅把 K/V 行号换成 `physical * block_size + offset`，
未映射页（`physical < 0`）按「权重 0」处理而不是越界读取。与 vLLM `KVCache` 的对应关系见
`docs/paged_kv_cache.md`。

## 12. Prefill / Decode

| 阶段 | 算术强度 | 使用后端 | 阈值来源 |
| --- | --- | --- | --- |
| Prefill（S_q 大） | 高（~D FLOPs/Byte） | `flash`（S_kv > 128）或 `tiled` | `src/runtime/dispatch_table.inc` (r0) |
| Decode（S_q = 1） | 极低（读全量 KV） | `decode` / `paged`（split-K） | 同上 |

阈值表由 `tools/dispatch_threshold_sweep.py` 重新标定并生成（文件头标注 AUTO-GENERATED）；
`tests/unit/test_dispatch.cpp` 对选择结果做行为快照。

## 13. vLLM(C++) 集成

```text
入口：vllm::flash_attention_forward(const vllm::FlashAttentionParams&)
适配：vllm_backend/{adapter_config,layout_bridge,sca_vllm_adapter,sca_flash_attention_adapter}
开关：-DSCI_ATTENTION_BUILD_VLLM_ADAPTER=ON -DSCI_ATTENTION_VLLM_ROOT=../vllm
迁移：FP32 -> BF16/FP16（SCA_VLLM_COMPUTE_DTYPE），BSHD -> BHSD（显式转换并单测）
回滚：SCA_VLLM_BACKEND=legacy 走回上游原型；补丁默认不应用
```

上游原型缺陷（`out` 未初始化、两次遍历 K/V、单 warp、GQA 取模、仅 FP32）带行号登记在
`upstream/notes.md` 的 UP-006；A/B 结论见 `docs/vllm_integration.md`（若本机未构建上游 vLLM，
必须明确标注「未完成 + 原因 + 复现命令」）。

## 14. RLHF / Rollout 集成

```text
路径 1（服务化）：RLHF -> vLLM(C++) -> SciCompute-Attention
路径 2（直连）  ：RLHF -> sca::flash_attention / decode_attention + PagedKVCache
指标：tokens/s、prefill P50/P99、decode 步 P50/P99、KV 复用率、KV 峰值、batch 利用率
示例：examples/rollout_engine_stub.cpp（真实 kernel 的 prefill + decode 循环，--json 输出）
```

RLHF 现状（CPU dense KV、CUDA arch 不含 120、`gpu_ops.cu` 未入构建）见 `docs/rollout_interface.md`。

## 15. Benchmark

方法学（warmup 20 / measure 100 / CUDA Event / P50-P99 / 固定 seed / 显存预算门禁）见
`docs/benchmarking.md`；完整结果与原始 JSON 路径见 `docs/results/benchmark_report.md`。

摘要（RTX 5070 Laptop, CUDA 13.2, fp16, B=1, H_q=H_kv=8, D=128, warmup 20 / runs 100）：

| 实现 | S=1024 非 causal | S=4096 非 causal | S=8192 非 causal | S=4096 causal |
| --- | --- | --- | --- | --- |
| naive（物化 S） | 15.3 ms / 0.28 TF | 超预算跳过 | 超预算跳过 | 超预算跳过 |
| tiled | 4.38 ms / 1.0 TF | 68.4 ms / 1.0 TF | 264.7 ms / 1.0 TF | 35.4 ms / 7.8 TF |
| **flash（本项目）** | **0.365 ms / 11.8 TF** | **3.95 ms / 17.4 TF** | **14.6 ms / 18.8 TF** | **2.20 ms / 124.9 TF** |
| torch SDPA（flash 后端） | 0.173 ms / 24.8 TF | 2.24 ms / 30.7 TF | 9.54 ms / 28.8 TF | 1.19 ms / 29.0 TF |

真实模型 shape（H_q=32、H_kv=8、D=128、S=2048、非 causal）：flash 4.32 ms / 15.9 TF，
SDPA 2.40 ms / 28.6 TF。

结论：flash 相对本项目 naive 提速约 40x、相对 tiled 提速 15–27x；与 torch SDPA 的 flash 后端相比
达到其 58–65%（差距来源与后续优化项见 `docs/results/benchmark_report.md`）。

## 16. Nsight Profiling

```bash
bash scripts/profile.sh --kernel flash --set roofline   # 调用 profiling/profile_flash.sh
python profiling/analyze_ncu.py profiling/reports/flash_*.csv
```

指标清单（占用率、smem/registers、bank conflict、各类 stall、dram/lts 流量）与四类瓶颈判定标准见
`docs/profiling.md`；`docs/results/profiling_summary.md` 记录已采集的结论（未采集项必须写明原因，
禁止用估算值替代）。

## 17. Build / Test / Example 细节

```text
CMake 选项：SCI_ATTENTION_BUILD_TESTS/BENCHMARKS/EXAMPLES/PYTHON/VLLM_ADAPTER、
            SCI_ATTENTION_INFRA_MODE=SOURCE|PACKAGE|STUB、SCI_ATTENTION_WERROR、
            SCI_ATTENTION_ARCH（默认探测，禁止硬编码）、SCI_ATTENTION_MAX_SMEM_BYTES
构建类型  ：Debug / Release / RelWithDebInfo（scripts/configure.sh --build-type）
测试层级  ：L1 编译 -> L2 单测 -> L3 kernel 正确性 -> L4 compute-sanitizer -> L5 集成 -> L6 回归
```

## 18. 已知限制与 Roadmap

**已知限制（本次交付）**

1. `decode`/`paged` 专用于 `S_q = 1`；`2 <= S_q <= 8` 的分组小 q 由 flash 承担。
2. varlen 为「按序列逐次启动」，没有跨序列融合 kernel（正确，但不含批内并行）。
3. **Triton 双轨与 Python `_core`（pybind11）绑定本次未实现**（状态与前提条件见
   `docs/triton_optimization.md`；README 与 `docs/results/triton_vs_cuda.md` 中不出现任何
   未经测量的 Triton 数字）。vLLM A/B（`docs/vllm_integration.md`）与真实权重 Q/K/V parity
   （`docs/model_integration.md` §7）同样标注为未完成并给出复现命令。
4. 权重级 Q8_0 反量化只用于元数据/形状/预算；v1 的注意力计算仍在 fp16/bf16 下进行，
   **不宣称支持 Q8_0 注意力**。

**Roadmap**：prefix caching、block 共享/COW、抢占式驱逐、量化 KV（fp8/int8）、多卡/多流、
融合 varlen kernel、GPU 侧 dequant-fused attention、Python vLLM 后端。

## 19. 引用与致谢

* FlashAttention / FlashAttention-2（Dao et al.）——本项目为独立实现，未复制其代码；
  算法思想与 online softmax 公式在 `docs/online_softmax.md` 中独立推导。
* vLLM 的 PagedAttention 与 KV block 概念——本项目实现与其 C++ 仓库同构的 block 布局，
  不复用其代码。
* CUTLASS / PTX ISA 文档——仅用于理解 `mma.sync` / `ldmatrix` / `cp.async` 语义，
  指令级行为以本项目 `tools/arch_probe` 的实测探针为准。
* 上游 SciComputeInfra 提供 Tensor/Device/Stream/Memory/Benchmark 基础设施。

## 20. 文档索引

```text
docs/00-recon.md              上游审计与复用判定（含文件:行号）
docs/env_report.md            环境 + ISA 探针原始输出 + 与提示词差异
docs/architecture.md          分层、模块图、文件职责表、依赖方向
docs/attention_math.md        定义、mask/scale/LSE、复杂度与 FLOPs 口径
docs/flash_attention.md       flash 算法、指令选择、tile/流水、varlen 形态
docs/kernel_design.md         线程映射、smem 分区、寄存器预算、边界用例
docs/online_softmax.md        online softmax 推导与等价性
docs/kv_cache.md              布局、地址公式、预算表、生命周期
docs/paged_kv_cache.md        分页设计、page table、与 vLLM 的对应
docs/prefill_decode.md        两阶段差异与阈值
docs/benchmarking.md          方法学、字段、公平性、复现命令
docs/profiling.md             ncu/nsys 用法与瓶颈判定
docs/roofline.md              三张表与 Q1-Q9
docs/numerical_stability.md   容差来源与极端用例
docs/model_integration.md     真实模型对接（元数据/预算/边界）
docs/rollout_interface.md     RLHF 契约与迁移路径
docs/vllm_integration.md      vLLM 集成与缺陷清单
docs/upstream_relationship.md 三仓关系与版本锚点
docs/troubleshooting.md       故障树
docs/results/*                结果与报告（benchmark_report / qwen3_4b_shapes.json 等）
```

## 21. 许可

MIT（见 `LICENSE`），与上游 SciComputeInfra 保持一致口径。
