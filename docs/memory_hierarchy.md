# Memory Hierarchy

## 1. 层次与实测数字

| 层次 | 容量 | 本项目用法 | 实测/依据 |
| --- | --- | --- | --- |
| 寄存器 | 65,536 / SM | flash 的 Q 片段、S/P 片段、O 累加器 | D=128 时 O 累加器 64 floats/线程（估算总量 ~160 regs） |
| Shared Memory | 102,400 B / SM；单 block opt-in 101,376 B | Q/K/V tile、tiled 的 S tile | `arch_probe` props |
| L2 | 32 MiB | decode 的 K/V 分片窗口、page table 命中 | 同上 |
| HBM | 8,177,909,760 B，128-bit，12,001,000 kHz | Q/K/V/O 与 KV Cache | 理论峰值 ≈ 384 GB/s |

## 2. 有效带宽实测（模型口径）

| 实现 | 场景 | 有效带宽 |
| --- | --- | --- |
| naive | S=1024, D=128, 非 causal | 9.4 GB/s |
| flash | S=1024, D=128, 非 causal | 23.0 GB/s |
| flash | S=8192, D=128, 非 causal | 16.0 GB/s |
| decode | S_kv=4096（Hq=32） | 68.5 GB/s |

模型口径见 `docs/benchmarking.md` §2；硬件口径（`dram__bytes.sum`）需 ncu 采集。

## 3. Bank conflict 处理

```text
问题：K/V tile 行距 = D×2 B；D=128 时为 256 B = 128 B 的整数倍 ⇒ ldmatrix 的 8 个行地址
      落在同一组 bank 上，产生 8 路冲突。
处理：每行 padding 8 个 half（16 B），使行距变成 (D+8)×2 B（D=128 → 272 B），
      既保持 16 B 对齐（ldmatrix 要求），又打破 128 B 的周期性。
校验：tools/smem_calc.py 与 kernel 内 static_assert 使用同一公式；
      实测冲突计数需 ncu（profiling 摘要中已标注未采集）。
```

## 4. L2 与 KV 复用

```text
* prefill：K/V tile 在同一 CTA 内被 kDTiles 次复用（smem 命中），跨 CTA 复用依赖 L2；
* decode：每个 (b,h) 读全量 K/V 一次，split-K 让不同 split 处理不同区间，窗口落在 L2 友好范围；
* paged：page table（4 B/token）额外读一次，但同一 block 内被 block_size 个 token 复用。
```

