# Profiling 使用说明

```bash
bash profiling/profile_flash.sh --set roofline     # ncu 采集 flash kernel
bash profiling/profile_decode.sh                   # ncu 采集 decode kernel
bash profiling/profile_naive.sh                    # 小规模 naive
python profiling/analyze_ncu.py profiling/reports/flash_*.csv
```

约定：

* 指标清单放在 `profiling/metrics/*.txt`，脚本用 `--metrics-file` 读取；
* 缺少 `ncu`/`nsys` 时脚本打印安装提示并以 exit 2 退出（不静默跳过）；
* 原始 `.ncu-rep`/`.nsys-rep` 落在 `profiling/reports/`，该目录被 `.gitignore` 排除；
* 采集前确认没有其他 GPU 负载（Ollama、其它 benchmark），否则数字无效。

通过标准：

```text
* compute-sanitizer：ERROR SUMMARY: 0 errors
* ncu：每个 kernel 至少给出寄存器数、smem、achieved occupancy、dram/sm throughput 与主要 stall
* 结论必须落成一句话的瓶颈判定（compute / memory / latency / launch bound）
```

