# Attention 数学定义与约定

> 本文件是**所有实现与测试的数学真相源**。任何 kernel、benchmark、测试代码中的 mask/scale/LSE
> 语义必须与本文件一致；不一致视为缺陷。

## 1. 定义

给定查询 `Q ∈ R^{B×H_q×S_q×D}`、键 `K ∈ R^{B×H_kv×S_kv×D}`、值 `V ∈ R^{B×H_kv×S_kv×D}`：

```
S_ij = scale * (Q_i · K_j)                                  (logits, FP32 累加)
P_ij = exp(S_ij - m_i) / Σ_k exp(S_ik - m_i)                 (softmax, m_i = max_k S_ik)
O_i  = Σ_j P_ij V_j                                          (输出, 与输入同 dtype)
```

* 默认 `scale = 1/sqrt(D)`；`AttentionConfig::scale > 0` 时覆盖默认值（见 `EffectiveScale()`）。
* 全部累加在 FP32 中进行；FP16/BF16 只用于存储与 MMA 输入。
* GQA/MQA：第 `h` 个 query head 使用第 `h_kv = h / group_size` 个 KV head，`group_size = H_q / H_kv`。

## 2. 掩码约定（唯一口径）

本项目的 causal 掩码采用**右下对齐**（bottom-right aligned）：

```
visible(i, j) ⟺ j ≤ i + diag,   diag = S_kv - S_q
```

* `S_q == S_kv`（标准 prefill）：退化为常见的 `j ≤ i`。
* `S_q == 1`（decode）：`diag = S_kv - 1`，等价于「可以看见全部已缓存 token」——这是 rollout decode
  步骤的语义。
* `S_q < S_kv`（chunked prefill）：查询块右对齐到 KV 末尾，与 vLLM/FlashAttention 的服务形态一致。

被掩码的 logits 取 `-inf`（`-CUDART_INF_F`），使 `exp(x - m) = 0` 精确成立；**禁止**用 `-1e30` 之类的
大负数近似——那会在极端 logits 下影响 `m` 的取值（见 §4）。

全掩码行（`m == -inf`）定义为输出全 0，而不是 NaN。

## 3. LSE（log-sum-exp）

```
LSE_i = m_i + log( Σ_j exp(S_ij - m_i) )    -- 只在 visible 的 j 上求和
```

* 单位为 logits 单位（未做任何归一化）。
* `AttentionConfig::return_lse = true` 时输出 `[B, H_q, S_q]` 的 FP32 张量。
* 校验：`tests/kernel/test_naive_correctness.cu::LseMatchesLogSumExp` 与宿主端 double 参考逐元素对比（1e-4）。

## 4. 数值稳定性

* 行最大值先行归约，保证 `exp` 的自变量 ≤ 0，不会上溢。
* `m` 初值为 `-inf` 时 `alpha = exp(-inf - m_new) = 0`，这是 online softmax 首块的正确行为
  （见 `docs/online_softmax.md`）。
* 极端 logits（±1e4 量级）用例见 `tests/kernel/test_numerics.cu`；实测结论写入
  `docs/numerical_stability.md`。

## 5. 复杂度与显存

| 项 | 标准实现（naive） | FlashAttention 风格（flash） |
| --- | --- | --- |
| 时间 | `O(B·H_q·S_q·S_kv·D)` | 同（FLOPs 相同） |
| 额外显存 | `B·H_q·S_q·S_kv·4 B`（FP32 分数矩阵） | `O(B·H_q·D)`（仅在线 softmax 状态） |
| HBM 流量 | Q/K/V + 4×分数矩阵往返 | Q/K/V 各读一次 + O 写一次 |

具体数字：`B=1, H_q=32, S_q=S_kv=4096, D=128` 时分数矩阵为 `1×32×4096×4096×4 = 2 GiB`，
而 flash 路径的 workspace 为 0 字节——这就是「不物化 N×N」的量化证据（`AttentionRuntimeStats::
materialized_score_bytes` 字段直接暴露该数值）。

## 6. FLOPs 口径（与 benchmark/roofline 保持同一定义）

```
非 causal: FLOPs = 4 · B · H_q · S_q · S_kv · D
causal   : FLOPs = 4 · B · H_q · D · Σ_i min(S_kv, i + diag + 1)
```

系数 2 来自乘加，另一个 2 来自 `QKᵀ` 与 `PV` 两次 GEMM。该公式在
`benchmarks/common/bench_utils.hpp::AttentionFlops()` 中实现一次，文档与报告只能引用它。

