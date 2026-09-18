# Nsight Profiling

## 1. 脚本与产出

| 脚本 | 目标 | 产出 |
| --- | --- | --- |
| `profiling/profile_flash.sh` | flash kernel（D=128, S=4096, fp16, causal） | `profiling/reports/flash_*.csv` |
| `profiling/profile_decode.sh` | decode kernel（S_kv=16384） | `profiling/reports/decode_*.csv` |
| `profiling/profile_naive.sh` | naive（小规模） | `profiling/reports/naive_*.csv` |
| `profiling/profile_triton.sh` | Triton kernel（`--kernel-name regex:_attn_fwd`） | 同上 |
| `profiling/profile_vllm.sh` | 上游 vLLM 原型（对照） | 同上 |
| `profiling/analyze_ncu.py` | CSV → Markdown 摘要 + 与上次对比 | `docs/results/profiling_summary.md` |

统一约定：`set -euo pipefail`；缺少 ncu/nsys 时给出安装提示并 exit 2；原始 `.ncu-rep`/`.nsys-rep`
不进版本库（`.gitignore`）。

## 2. 必采指标

```text
gpu__time_duration.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput / lts__throughput
sm__warps_active.avg.pct_of_peak_sustained_active            (achieved occupancy)
launch__registers_per_thread
launch__shared_mem_per_block_static / _dynamic
launch__occupancy_limit_registers / _shared_mem
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
smsp__warp_issue_stalled_{long_scoreboard,short_scoreboard,mio_throttle,math_pipe_throttle}_per_warp_active.pct
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
dram__bytes.sum / lts__t_sectors.sum
```

清单文件：`profiling/metrics/{fa,decode,roofline}_metrics.txt`；名称随 Nsight 版本变化，
脚本先做 `ncu --query-metrics | rg <name>` 校验。

## 3. nsys 与必答问题

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true \
  --output profiling/reports/flash_timeline ./build/benchmarks/benchmark_attention --suite sanity
```

1. kernel 之间是否有不可解释的 gap？→ `docs/results/profiling_summary.md`
2. 热路径是否出现 H2D/D2H 拷贝？→ 唯一一处是 PagedKVCache 的 slot 映射上传（4 B/token）
3. prefill 与 decode 的时间分布差异？→ decode 单步 0.244 ms vs prefill 3.95 ms（S=4096）
4. ncu 与 nsys 的 kernel 时间差异 >10%？→ 超过必须给出解释

## 4. 四类瓶颈判定

```text
compute-bound : sm__throughput 高 且 dram__throughput 低
memory-bound  : dram__throughput 接近实测峰值（表 A）
latency-bound : occupancy 低 + long scoreboard stall 高
launch-bound  : kernel duration 缩短但端到端不变 / nsys gap 占比高
```

## 5. 当前状态

已执行：`compute-sanitizer --tool memcheck` 在 flash / decode 测试上 `ERROR SUMMARY: 0 errors`
（`docs/results/profiling_summary.md`）。ncu/nsys 采集脚本已就绪，需在 GPU 空闲时运行；
未采集项在结果文件中标注「未采集 + 原因」，禁止用估算值填充。

