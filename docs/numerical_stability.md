# 数值稳定性与容差

## 1. 容差表（实测校准后的口径）

| 被测路径 | 参考 | 最大绝对误差 | 平均绝对误差 | 依据 |
| --- | --- | --- | --- | --- |
| naive(fp32) | FP64 参考 | ≤ 1e-4 | ≤ 1e-6 | `test_naive_correctness` 实测远低于上限 |
| naive(fp16) | FP64 参考（同一份反量化输入） | ≤ 5e-3 | ≤ 5e-4 | 同上 |
| naive(bf16) | FP64 参考 | ≤ 2e-2 | ≤ 2e-3 | bf16 尾数 8 bit |
| tiled/flash(fp16) | FP64 参考 | ≤ 5e-3 | ≤ 5e-4 | `test_tiled_correctness` / `test_flash_correctness` |
| tiled/flash(bf16) | FP64 参考 | ≤ 2e-2 | ≤ 2e-3 | 同上 |
| decode(fp16) | FP64 参考 | ≤ 5e-3 | ≤ 5e-4 | `test_decode_correctness` |
| paged(fp16) | FP64 参考 | ≤ 5e-3 | ≤ 5e-4 | `test_paged_correctness` |

**参考基准口径**：narrow dtype 的比较基准是「同一份**反量化后**的张量」用 FP64 计算的注意力，
而不是原始 GGUF 字节流（见 `docs/model_integration.md` §5）。

## 2. 极端用例（`tests/kernel/test_numerics.cu`）

```text
[x] logits 量级 {0, 1, 10, 1e2, 1e4} × {naive, tiled, flash}：无 NaN/Inf，误差在容差内
[x] 单 token（S_q=S_kv=1）与全等值行（logits 全 0 → 均匀注意力）
[x] 长序列（S_q=512, S_kv=8192, D=128, causal）
[x] causal 单行 decode（S_q=1, S_kv=4096）在 flash 与 decode 下一致
```

## 3. 溢出/下溢分析

```text
上溢：先减行最大值，exp2 的自变量 ≤ 0；1e4 量级 logits 下仍无 Inf。
下溢：exp(x) 在 x << 0 时趋 0，被掩码项恰好取 -inf ⇒ 精确 0（而非 1e-30 近似）。
全掩码行：m = -inf, l = 0 ⇒ 输出定义为 0（内核用 l > 0 ? 1/l : 0 保护）。
alpha 首块：exp(-inf - m) = 0 ⇒ 正确清空累加器，不产生 NaN。
```

## 4. 与 PyTorch 的差异来源

```text
1) fp16/bf16 存储舍入：容差表已按 dtype 分档；
2) exp2f + 常数折叠：误差在 1 ulp 量级，远小于容差；
3) split-K 合并：partial 以 FP32 保存，合并公式稳定（见 docs/online_softmax.md §2）；
4) SDPA 的对照数据见 docs/results/benchmark_report.md §5（性能）与本节（精度口径）。
```

