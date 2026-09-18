# 性能报告

> 数据全部来自本仓库 `benchmarks/` 的可复现运行；每条结果都绑定 git sha、GPU 状态与原始 JSON 路径。
> 方法学见 `docs/benchmarking.md`（warmup 20 / measure 100 / P50-P99 / CUDA Event）。

## 0. 运行环境

```text
GPU            : NVIDIA GeForce RTX 5070 Laptop GPU (cc 12.0, 36 SM, 8,177,909,760 B)
Driver         : 615.71.09      CUDA runtime : 13.2
vLLM/Ollama    : 未运行（测量前 nvidia-smi 显示显存占用 65 MiB，避免服务抢占）
构建           : Release (-O3)，SCI_ATTENTION_ARCH=120
git sha        : 42fffd1（Phase 2 提交时的 HEAD）
原始数据       : benchmarks/results/2026-09-19-42fffd1/naive_main.json / .csv（本机路径，按 .gitignore 不入库）
复现命令       : ./build/benchmarks/benchmark_naive --suite main --runs 100 --warmup 20 \
                   --json benchmarks/results/<date>-<sha>/naive_main.json
```

## 1. Level 0（naive）：IO 基线

naive 后端**刻意**把 `S = QKᵀ` 物化到 HBM，用来量化 flash 路径省掉了什么。

| 实现 | dtype | S_q=S_kv | D | causal | P50 (ms) | P99 (ms) | TFLOPS | 有效带宽 (GB/s) | 分数矩阵 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | fp16 | 128 | 64 | 0 | 0.119 | 0.132 | 0.281 | 21.99 | 1 MiB |
| naive | fp16 | 128 | 64 | 1 | 0.119 | 0.121 | 1.136 | 22.02 | 1 MiB |
| naive | fp16 | 512 | 64 | 0 | 1.656 | 1.674 | 0.324 | 21.54 | 16 MiB |
| naive | fp16 | 512 | 64 | 1 | 1.702 | 1.828 | 1.264 | 20.94 | 16 MiB |
| naive | fp16 | 512 | 128 | 1 | 3.401 | 3.919 | 1.265 | 11.10 | 16 MiB |
| naive | fp16 | 2048 | 128 | 1 | 63.901 | 65.686 | 1.076 | 8.66 | 256 MiB |
| naive | fp16 | 4096 | 64 | 1 | 136.537 | 139.307 | 1.007 | 15.85 | 1024 MiB |
| naive | bf16 | 1024 | 128 | 1 | 62.033 | 64.205 | 4.435 | 8.99 | 256 MiB |
| naive | bf16 | 1024 | 128 | 0 | 62.232 | 65.971 | 0.276 | 8.96 | 256 MiB |

真实模型档（Qwen3-4B-Thinking-2507-Q8 的注意力 shape：H_q=32、H_kv=8、D=128、causal，GQA group=4）：

| 实现 | shape | P50 (ms) | P99 (ms) | TFLOPS | 有效带宽 (GB/s) | 分数矩阵 |
| --- | --- | --- | --- | --- | --- | --- |
| naive | B=1, H_q=32, H_kv=8, S=1024, D=128, causal | 62.72 | 65.51 | 4.394 | 8.89 | 256 MiB |
| naive | 同上，非 causal | 62.03 | 65.97 | 4.435 | 8.99 | 256 MiB |

观察（作为后续优化的对照基准）：

1. **算术强度极低**：非 causal 的 TFLOPS 约 0.27–0.32，causal 约 1.0–1.3（FLOPs 口径减半但耗时不变，
   因为 naive 不跳过被掩码的计算）——这是"物化分数矩阵 + 逐元素访问"的必然结果。
2. **有效带宽只有 9–22 GB/s**：远低于该卡的理论 HBM 带宽，说明瓶颈在访存模式（V 的列访问非合并、
   S 的多次往返）而不是带宽本身。
3. **API 封装开销可忽略**：`naive_api_*`（含输出分配 + 校验 + dispatch）与 `naive_*`（纯 kernel）
   在 S=2048/D=128 上分别为 63.89 ms 与 63.90 ms，差异 < 0.1%。

## 2. Level 1 / Level 3（tiled / flash）

数据在 Phase 6 的 benchmark suite 中生成，届时本节补齐：

```text
./build/benchmarks/benchmark_tiled  --suite main --json .../tiled.json
./build/benchmarks/benchmark_flash  --suite main --json .../flash.json
python benchmarks/benchmark_sdpa.py --suite main --json .../sdpa.json
```

当前的**正确性与资源证据**（Phase 3/5 已完成）：

| 项 | naive | tiled | flash |
| --- | --- | --- | --- |
| workspace 字节 | `B·H_q·S_q·S_kv·4` | **0** | **0** |
| 物化分数矩阵 | 是（`materialized_score_bytes > 0`） | 否 | 否 |
| head_dim 覆盖 | 32–256（全部） | 32/64/96/128 | 32/64/96/128/160/192/256 |
| dtype | fp32/fp16/bf16 | fp32/fp16/bf16 | fp16/bf16 |
| 数据来源 | `docs/results/benchmark_report.md` §1 | 同上（Phase 6） | 同上（Phase 6） |

## 3. 精度证据（与性能同源的正确性门禁）

| 测试 | 覆盖 | 结果 |
| --- | --- | --- |
| `tests/kernel/test_naive_correctness.cu` | fp32/fp16/bf16 × causal × GQA{1,2,4} × S∈{1..129} × D∈{64,128} | 通过 |
| `tests/kernel/test_tiled_correctness.cu` | 同上 × D∈{32,64,96,128} + smem 超限拒绝 | 通过 |
| `tests/kernel/test_flash_correctness.cu` | fp16/bf16 × causal × D∈{32..256} × S∈{1,7,63,64,65,127,128,129,1024} × GQA{1,2,4,8} + LSE + FP32 拒绝 | 通过 |

容差口径见 `docs/numerical_stability.md`（fp32 ≤1e-4/1e-6，fp16 ≤5e-3/5e-4，bf16 ≤2e-2/2e-3，
参考实现为同一份反量化输入上的 FP64 计算）。

## 4. 已知限制

```text
* 本节数字为 kernel-only（除显式标注 api 的行）；ncu/nsys 的寄存器/占用率/带宽利用率为 Phase 6 内容。
* SDPA / vLLM 原型对照数据尚未生成（Phase 6 / Phase 11）。
* 所有数据在 Ollama 未运行、显存占用 65 MiB 的状态下采集；与 Ollama 并发时不得复用这些数字。
```

