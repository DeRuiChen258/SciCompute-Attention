# Profiling 摘要

> 环境：RTX 5070 Laptop（sm_120, 36 SM）、CUDA 13.2、Release、Ollama 未运行。
> 原始报告目录：`profiling/reports/`（.ncu-rep / .nsys-rep 按 .gitignore 不入库）。

## 1. 已完成的运行时检查（compute-sanitizer）

```bash
$ compute-sanitizer --tool memcheck --launch-timeout 120 \
    ./build/tests/kernel/test_flash_correctness \
    --gtest_filter='*MatchesReferenceAcrossDtypesAndHeadDims'
[  PASSED  ] 1 test.
========= ERROR SUMMARY: 0 errors

$ compute-sanitizer --tool memcheck --launch-timeout 120 \
    ./build/tests/kernel/test_decode_correctness --gtest_filter='*EverySplitCount*'
[  PASSED  ] 1 test.
========= ERROR SUMMARY: 0 errors
```

`tests/CMakeLists.txt` 已把 sanitizer 子集注册为 `ctest -L sanitize`（默认运行被排除）。

## 2. 静态资源数据（编译期已知）

| kernel | smem/block (B) | 上限 (B) | 线程/block | warps | 备注 |
| --- | --- | --- | --- | --- | --- |
| `flash_fwd` D=128 | 87040 | 101376 | 128 | 4 | smem 受限，1 CTA/SM |
| `flash_fwd` D=256 | 84480 | 101376 | 128 | 4 | splits_d=2 |
| `tiled_attn` D=128 | 70400 | 101376 | 128 | 4 | S tile 在 smem |
| `decode_partial` | 0 | — | 128 | 4 | 纯寄存器 |

## 3. 未采集项与原因

| 项 | 状态 | 原因 / 下一步 |
| --- | --- | --- |
| sm/dram throughput | 未采集 | 需 ncu 独占 GPU；`bash scripts/profile.sh --kernel flash` 可直接运行 |
| achieved occupancy | 未采集 | 同上；预计 flash D=128 为 1 CTA/SM |
| bank conflict 计数 | 未采集 | 同上；行 padding 8 half 的目标是消除 8 路冲突，需实测确认 |
| nsys 时间线 | 未采集 | 同上 |
| registers/thread | 未采集 | 由 `launch__registers_per_thread` 提供 |

**纪律**：未采集项在 README 与报告中被显式标注，未用估算值冒充实测值。

