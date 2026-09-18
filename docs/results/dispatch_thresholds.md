# Dispatch 阈值标定报告

## 1. 当前表（r0，编译内嵌）

```text
decode_max_seq_q   = 1
decode_min_seq_kv  = 256
tiled_max_seq_kv   = 128
tokens_per_split   = 512
max_splits         = 16
```

来源：提示词 §9.1 种子值 + 一项可验证收窄——decode kernel 只服务 `S_q = 1`
（`docs/prefill_decode.md` §2），`2..8` 的小 q 由 flash 承担。

## 2. 观测数据（后续 sweep 的起点）

| 场景 | 实测 | 支撑的判断 |
| --- | --- | --- |
| S_kv = 1024 / 4096 / 8192（S_q=1, Hq=8） | decode P50 = 0.055 / 0.244 / 0.49 ms | 短序列启动开销占比高，故设下限 256 |
| S=512（tiled vs flash） | tiled 1.10 ms vs flash 0.158 ms | `tiled_max_seq_kv=128` 偏保守，可考虑上调 |
| S=4096 | flash 3.95 ms（17.4 TFLOPS） | 长序列必须走 flash |

## 3. 复现

```bash
python tools/dispatch_threshold_sweep.py --collect \
  --out benchmarks/results/<date>-<sha>/dispatch_sweep.json
python tools/dispatch_threshold_sweep.py --emit-inc src/runtime/dispatch_table.inc --revision r1
bash scripts/test.sh --filter dispatch
```

`tests/unit/test_dispatch.cpp` 固定了多组 (shape, cfg) → backend/tile 行为快照，表变化必须伴随测试更新。

