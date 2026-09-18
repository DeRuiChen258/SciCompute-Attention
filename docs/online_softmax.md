# Online Softmax：推导、等价性与误差

## 1. 为什么需要它

一次性 softmax 需要先知道整行的最大值与分母，这意味着「先读完所有 logits 才能开始归一化」。
FlashAttention 把 K/V 分块流式处理，因此必须让 (m, l, O) 三元组支持**增量合并**：

```text
单块（前 t 块）已维护：m_t = max_{j<=t} s_j,  l_t = Σ_{j<=t} exp(s_j - m_t),
                        O_t = Σ_{j<=t} exp(s_j - m_t) v_j
加入新块（下标集合 B）：m_B = max_{j∈B} s_j, l_B = Σ_{j∈B} exp(s_j - m_B),
                        O_B = Σ_{j∈B} exp(s_j - m_B) v_j
```

## 2. 合并公式（本项目实现口径）

```text
m_new = max(m_t, m_B)
alpha = exp(m_t - m_new)                     // 旧块的重标定因子
beta  = exp(m_B - m_new)                     // 新块的重标定因子
l_new = alpha * l_t + beta * l_B
O_new = alpha * O_t + beta * O_B
最终  O = O_new / l_new
```

实现上把 `beta` 吸收进 P：`P = exp(s - m_new) = beta * exp(s - m_B)`，于是
`O_new = alpha * O_t + P @ V`、`l_new = alpha * l_t + rowsum(P)`，少一次逐元素乘法。

## 3. 等价性

对任意分块顺序，`(m,l,O)` 的递归都等于「一次性减最大值 softmax」的结果，因为
`exp(s - m_new) = exp(s - m_t) * exp(m_t - m_new)`，重标定因子对同一行的所有项一致，
而 `v_j` 项没有任何依赖于 `m` 的因子（归一化只在最后做一次）。因此
`O/l` 与 `Σ_j exp(s_j - m_max) v_j / Σ_j exp(s_j - m_max)` 恒等（浮点意义下误差仅来自舍入）。

## 4. 边界情形

| 情形 | 行为 | 依据 |
| --- | --- | --- |
| 首块 | `m_t = -inf ⇒ alpha = exp(-inf - m_new) = 0` | O 被正确清零，不产生 NaN |
| 全掩码块 | `m_B = -inf ⇒ beta = 0`（实现上 P 全 0） | 该块对 l/O 无贡献 |
| 全掩码行 | `m_new = -inf, l = 0` | 输出定义 0（`l > 0 ? 1/l : 0`） |
| 极端 logits（±1e4） | 减最大值后 exp 自变量 ≤ 0，无上溢 | `tests/kernel/test_numerics.cu` |

## 5. 实测证据

* 朴素路径（naive）的行 softmax 与 FLASH 的在线更新在相同输入上的最大差 < fp16 容差
  （`tests/kernel/test_flash_correctness.cu` 与 `test_naive_correctness.cu` 共享同一份 FP64 参考）。
* `tests/kernel/test_tiled_correctness.cu` 覆盖 D∈{32,64,96,128} × causal × S∈{1,17,64,65,129,256}。
* 极端用例（±1e4 logits、全等值、单 token）在 `tests/kernel/test_numerics.cu` 中校验无 NaN/Inf。

