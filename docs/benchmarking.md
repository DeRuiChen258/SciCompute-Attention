# Benchmark 方法论与数据格式

> 本文件定义**唯一口径**。任何报告、README、docs/results 中的数字都必须能追溯到这里定义的
> 方法、命令与字段；不允许为某一份报告自定义统计口径。

## 1. 固定方法（提示词 §12.1）

| 项 | 规定值 | 实现位置 |
| --- | --- | --- |
| warmup | 20 次（不计时） | 各 `benchmarks/benchmark_*.cu` 的 `--warmup`（默认 20） |
| measure | 100 次独立 launch，同一 stream | `--runs`（默认 100） |
| 计时 | CUDA Event 包住纯 kernel 调用 | `sca::bench::EventTimer` |
| 统计 | mean / median / std / P50 / P90 / P95 / P99（线性插值分位数） | `ComputeStats()` |
| 随机性 | 固定 seed（默认 1234），输入由 xorshift32 生成 | `sca::bench::MakeHeader` + `tests/common/test_utils.hpp` |
| 时钟 | 记录 SM clock（`cudaDevAttrClockRate`）与是否降频 | JSON `sm_clock_mhz` |
| 显存 | 记录 workspace 与峰值估算，并在运行前做预算检查 | JSON `workspace_mb` / `peak_mem_mb` |
| 公平性 | 同一 GPU 状态、同一 dtype/布局；不挑最好的一次；所有实现使用相同 warmup/runs | 见 §4 |

### 1.1 两种口径（必须同时报告）

```text
kernel : Event 只包住 kernel launch（缓冲与输出预先分配）——用于与其他 kernel 对比；
api    : Event 包住一次完整的公共 API 调用（含输出分配、校验、dispatch）——用于衡量封装开销。
```

报告中必须写清使用的是哪种口径。`benchmark_naive` 同时给出 `naive_*`（kernel）与
`naive_api_*`（api）两组 case。

## 2. 计算公式（与 `docs/roofline.md`、`docs/attention_math.md` 完全一致）

```text
FLOPs(非 causal) = 4 · B · H_q · S_q · S_kv · D
FLOPs(causal)    = 4 · B · H_q · D · Σ_i min(S_kv, i + diag + 1),  diag = S_kv - S_q
TFLOPS           = FLOPs / (P50_latency_ms × 1e-3) / 1e12
bytes_moved_modeled:
  naive  = q + 2·kv + out + 4·(B·H_q·S_q·S_kv·4)      # S/P 的 4 次往返
  tiled  = q + 2·kv + out                              # S 只在 smem
  flash  = q + 2·kv + out
  decode = q + 2·kv + out
effective_bw = bytes_moved_modeled / latency
```

> 「modeled」表示理论最小流量模型，不是 ncu 实测的 `dram__bytes`；两者对照见
> `docs/results/profiling_summary.md`。

## 3. 结果格式（JSON，字段已实现）

```json
{
  "benchmark_name": "benchmark_naive",
  "git_sha": "42fffd1", "build_type": "Release",
  "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU", "compute_cap": "12.0",
  "driver": "615.71.09", "cuda": "13.2", "sm_clock_mhz": 1425,
  "timestamp": "2026-09-19T00:22:11+0800",
  "warmup": 20, "runs": 100, "seed": 1234,
  "cases": [
    {
      "case_id": "naive_B1_Hq8_Hkv8_Sq512_Skv512_D128_fp16_causal1",
      "impl": "naive",
      "params": {"B": 1, "H": 8, "Hkv": 8, "Sq": 512, "Skv": 512, "D": 128,
                 "dtype": "fp16", "dtype_bytes": 2, "causal": true},
      "latency_ms": {"mean": 0, "median": 0, "std": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0},
      "flops_effective": 0, "tflops": 0,
      "bytes_moved_modeled": 0, "effective_bw_gbps": 0,
      "peak_mem_mb": 0, "workspace_mb": 0,
      "registers_per_thread": -1, "smem_bytes": 0, "achieved_occupancy_pct": -1,
      "reference_max_abs_err": -1, "note": "..."
    }
  ]
}
```

CSV 列：

```csv
name,case_id,warmup,runs,mean_ms,median_ms,std_ms,p50_ms,p90_ms,p95_ms,p99_ms,tflops,effective_bw_gbps,peak_mem_mb,unit
```

`registers_per_thread` / `achieved_occupancy_pct` 为 `-1` 表示「未采集」——这两个字段只有 Nsight 才能
填真值（Phase 6 的 ncu 流程），**禁止**用估算值填充。

## 4. 公平性与显存保护（8 GB 硬约束）

```text
1) 每个 case 运行前估算字节数 = Q + K + V + out + workspace + 中间缓冲；
   超过 3 GiB 直接跳过，并把 "skipped" 字段写入 JSON（禁止静默跳过、禁止 OOM 崩溃）。
2) 与 Ollama 服务互斥：SCA 侧实验前必须停 Ollama（约 4.9 GiB），或确认其未运行。
3) 一次只跑一个 benchmark 进程；ncu/nsys 采集与计时测量不得并行。
4) 所有对照实现（SDPA / vLLM 原型）使用同样的 warmup/runs/输入分布。
```

## 5. 复现命令模板

```bash
SHA=$(git -C $SCA_ROOT rev-parse --short HEAD)
OUT=$SCA_ROOT/benchmarks/results/$(date +%F)-${SHA}
mkdir -p "${OUT}"

# 0) 记录 GPU 状态
nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.sm --format=csv,noheader

# 1) 本项目 kernel
./build/benchmarks/benchmark_naive --suite main --seed 1234 --runs 100 --warmup 20 \
  --json "${OUT}/naive_main.json" --csv "${OUT}/naive_main.csv"

# 2) 统一入口
bash scripts/bench.sh --suite sanity --runs 50 --warmup 10
```

## 6. 结果解读规则

```text
* 报 P50 为主、P99 为辅；只报均值视为不完整。
* TFLOPS 必须写明是「有效 FLOPs / 延迟」的口径，不是硬件峰值占比。
* 任何「快 N 倍」的结论必须来自同一次运行（同一 JSON 文件）内的两条 case。
* 跳过（skipped）的 case 必须出现在报告的表格里，并写明原因。
```

