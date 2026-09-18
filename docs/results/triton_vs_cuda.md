# Triton vs CUDA 对比（状态：未完成）

## 1. 结论（当前）

```text
Triton 轨道本次未实现（原因与前提条件见 docs/triton_optimization.md §3/§4），
因此本文件不包含任何 Triton 性能数字。任何「Triton 达到 CUDA 的 X%」的说法在数据生成前
均视为未验证声明（prompt §21.1 第 18 条禁止用估算或替身数据填充）。
```

## 2. 已有的对照数据（CUDA 侧 + SDPA）

| 实现 | S=1024 非 causal | S=4096 非 causal | S=8192 非 causal |
| --- | --- | --- | --- |
| 本项目 flash（CUDA） | 0.365 ms / 11.8 TFLOPS | 3.95 ms / 17.4 TFLOPS | 14.61 ms / 18.8 TFLOPS |
| torch SDPA（flash 后端） | 0.173 ms / 24.8 TFLOPS | 2.24 ms / 30.7 TFLOPS | 9.54 ms / 28.8 TFLOPS |

原始 JSON：`benchmarks/results/2026-09-19-51f877e/{attention_main.json,sdpa.json}`。

## 3. 生成 Triton 数据后的目标格式

```json
{
  "triton_version": "3.7.1",
  "cache_key": "<TRITON_CACHE_DIR 下的 hash>",
  "best_configs": [{"shape": "S=1024,D=128,causal", "BLOCK_M": 64, "BLOCK_N": 64,
                    "num_warps": 4, "num_stages": 2}],
  "cases": [{"impl": "triton", "kernel_only_ms": 0.0, "end_to_end_ms": 0.0,
             "autotune_search_s": 0.0, "configs_searched": 0}]
}
```

