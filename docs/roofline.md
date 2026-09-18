# Roofline / IO 分析

> 数据来源：`benchmarks/results/2026-09-19-51f877e/*.json`（flash/tiled/decode）+
> `2026-09-19-42fffd1/naive_main.json`；机器参数来自 `docs/env_report.md` 与实测。

## 表 A：机器参数

| 参数 | 值 | 来源 |
| --- | --- | --- |
| SM 数 | 36 | `arch_probe` device props |
| smem/SM、regs/SM、threads/SM | 102,400 B / 65,536 / 1,536 | 同上 |
| L2 | 32 MiB | 同上 |
| HBM 容量 / 位宽 / 时钟 | 8,177,909,760 B / 128-bit / 12,001,000 kHz | 同上（理论峰值约 384 GB/s） |
| 有效带宽（模型化口径） | 9–28 GB/s | `docs/results/benchmark_report.md` |
| 有效吞吐 | 11.8–18.8 TFLOPS（非 causal）、124.9 TFLOPS（causal 折算） | 同表 |

有效带宽按 `bytes_moved_modeled` 计算（`docs/benchmarking.md` §2），不含硬件预取与 L2 命中收益；
硬件级数字需 ncu 的 `dram__bytes.sum`（见 `docs/results/profiling_summary.md`）。

## 表 B：算法流量模型

| 实现 | 理想 HBM 读 | 理想 HBM 写 | 附加中间量 | 算术强度（FLOPs/Byte） |
| --- | --- | --- | --- | --- |
| naive | Q + K + V | O | 4 × B·H_q·S_q·S_kv·4 B（S/P 往返） | 约 0.06（S=1024, D=128） |
| tiled | Q + K + V（K/V 按 tile 复用） | O | 0 | 约 D/2 |
| flash | Q + K + V | O（+LSE） | 0 | 约 D/2 |
| decode | Q + K + V（全量 KV） | O | partial (m,l,O) | 1–2 |
| paged | 同上 + page table（4 B/token） | O | partial (m,l,O) | 约 2 |

## 表 C：实测对照（B=1, H_q=H_kv=8, D=128, fp16, 非 causal）

| 实现 | S | P50 (ms) | TFLOPS | 有效带宽 (GB/s) | 相对 naive |
| --- | --- | --- | --- | --- | --- |
| naive | 1024 | 15.25 | 0.28 | 9.4 | 1.0x |
| tiled | 1024 | 4.38 | 1.0 | 2.0 | 3.5x |
| flash | 1024 | 0.365 | 11.8 | 23.0 | 41.8x |
| flash | 4096 | 3.95 | 17.4 | 13.0 | naive 超预算跳过 |
| flash | 8192 | 14.61 | 18.8 | 16.0 | 同上 |
| SDPA（torch flash 后端） | 8192 | 9.54 | 28.8 | — | 对照 |

## Q1–Q10

**Q1 为什么标准 Attention 产生 O(N²) 中间矩阵？**
`S = QKᵀ` 形状为 `[B, H_q, S_q, S_kv]`；`B=1, H_q=32, S_q=S_kv=4096` 时是
`32 × 4096 × 4096 × 4 B = 2 GiB`，naive 后端实测物化（`materialized_score_bytes` 字段）。

**Q2 FlashAttention 为什么不需要存储完整 N×N？**
分块 + 在线 softmax：只保留 `(m, l, O)`，S/P 永不出现在 HBM。
证据：flash 的 `workspace_bytes = 0`、`materialized_score_bytes = 0`（`test_flash_correctness` 断言）。

**Q3 Online softmax 为什么数值稳定？**
先减行最大值，`exp` 自变量 ≤ 0；掩码用 `-inf` 得到精确 0；首块 `alpha = exp(-inf - m) = 0`。
推导见 `docs/online_softmax.md`；1e4 量级 logits 实测无 NaN/Inf（`test_numerics`）。

**Q4 收益来自计算减少还是 IO 减少？**
IO。非 causal 时 FLOPs 与 naive 相同但快 40x：naive 每次 forward 多搬 4×分数矩阵
（S=1024 时约 128 MiB 往返），flash 只读 Q/K/V 一次。

**Q5 Prefill 与 Decode 为何需要不同 kernel？**
算术强度差两个数量级：prefill（S=4096, D=128）约 64 FLOPs/Byte，decode 只有 1–2 FLOPs/Byte。
decode 因此靠 split-K 拉并发（S_kv=4096 时 0.244 ms）。

**Q6 Paged KV Cache 为什么对 serving 重要？**
连续 KV 需按最大长度预留：32K 上下文、36 层、fp16 需 4.5 GiB，与 4B 权重不可同机；paged 按
16-token block（2.25 MiB）分配，长度异构与多序列并发都不浪费。

**Q7 FlashAttention 与 PagedAttention 的职责边界？**
flash 决定「怎么算」（IO-aware 计算），paged 决定「怎么放与怎么找」（非连续页寻址）。
在 vLLM(C++) 中分别对应 attention 计算路径与 KV 存储/分配路径。

**Q8 为什么不用 wgmma/tcgen05？**
sm_120 上 ptxas 直接拒绝：`Instruction 'wgmma.fence' not supported on .target 'sm_120'`
（原始输出见 `docs/env_report.md` §2）；tcgen05 仅 sm_100/sm_103。主路径固定
`mma.sync m16n8k16` + `cp.async`。

**Q9 8 GB 显存下如何权衡 KV 与长序列？**
见 `docs/kv_cache.md` 预算表：4096 上下文 576 MiB（可行，需停 Ollama）、8192 需 1.13 GiB、
32768 需 4.5 GiB（不可行）。benchmark 在 case 级做预算检查并把跳过原因写入 JSON。

**Q10 CUDA 与 Triton 同 shape 的算术强度相同，为什么性能不同？**
Triton 轨道当前状态见 `docs/results/triton_vs_cuda.md`；已知约束：Triton 动态 smem 上限
101,376 B 迫使更小的 tile/stages，缺少 smem 分区与寄存器分配的细粒度控制，且 Python launch
开销在小 shape 下占比高（需 nsys 佐证）。

