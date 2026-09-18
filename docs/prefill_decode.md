# Prefill 与 Decode

## 1. 差异

| 维度 | Prefill | Decode |
| --- | --- | --- |
| S_q | 大（512–16384） | 1（或极小） |
| 算术强度 | ≈ D FLOPs/Byte（高） | ≈ 2 FLOPs/Byte（极低） |
| 瓶颈 | Tensor Core / smem 带宽 | HBM 读 K/V |
| 后端 | `flash`（S_kv > 128）、`tiled`（短序列） | `decode`（split-K）、`paged`（服务形态） |
| 并行策略 | 按 (b, h, q_block) 切分 | 按 split 切分 K/V 区间，提高并发 |

## 2. 阈值表（`src/runtime/dispatch_table.inc`，r0）

```text
decode_max_seq_q   = 1        # decode kernel 专用于 S_q=1
decode_min_seq_kv  = 256      # 太短的序列走 tiled/flash，启动开销更划算
tiled_max_seq_kv   = 128      # 短序列优先 tiled（寄存器压力更小）
tokens_per_split   = 512      # split-K 粒度
max_splits         = 16       # workspace 上限保护
```

选择顺序（`AttentionDispatcher::Select`）：

```text
1) Validate 失败 → 直接返回 invalid（不猜测）
2) 显式后端 → 检查 Supports()，不支持时按 allow_fallback 决定报错或降级
3) S_q == 1 且 S_kv >= 256 → decode
4) S_q * 8 <= S_kv → flash（分组小 q；decode kernel 只服务 S_q=1）
5) S_kv <= 128 → tiled
6) 其余 → flash；flash 不支持时（head_dim 超表）→ tiled → naive（需 allow_fallback=true）
```

## 3. 实测（RTX 5070, fp16, B=1, Hq=Hkv=8, D=128）

| 场景 | 实现 | P50 | TFLOPS |
| --- | --- | --- | --- |
| Prefill S=4096（非 causal） | flash | 3.95 ms | 17.4 |
| Prefill S=4096（causal） | flash | 2.20 ms | 124.9 |
| Decode S_kv=4096 | decode（split-K） | 0.244 ms | 0.6（算术强度极低） |
| Decode S_kv=4096（Hq=32） | decode | 0.246 ms | 8.7 |

结论：causal 路径通过整块跳过把 FLOPs 减半（口径见 `docs/attention_math.md` §6），
而 decode 的瓶颈在 HBM（有效带宽 60+ GB/s，算术强度 < 1 FLOPs/Byte）。

