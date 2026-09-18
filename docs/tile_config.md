# Tile 配置标定

## 1. 方法

```text
① 计算：tools/smem_calc.py 按 smem/寄存器模型筛掉非法配置（上限 101,376 B，行 padding 8 half）
② 编译：flash_tile_config.hpp 的每个条目都有 static_assert（放不下就编译失败）
③ 运行：benchmark_attention --suite main 采集 P50/TFLOPS
④ 回归：tests/unit/test_tile_smem_calc.cpp 固定关键配置的 smem 字节数
```

## 2. 当前表（D=32..256，一维一组）

| head_dim | BLOCK_M | BLOCK_N | warps | stages | splits_d | smem (fp16) | 估算寄存器/线程 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | 64 | 4 | 2 | 1 | 25,600 B | 80 |
| 64 | 64 | 64 | 4 | 2 | 1 | 46,080 B | 104 |
| 96 | 64 | 64 | 4 | 2 | 1 | 66,560 B | 128 |
| 128 | 64 | 64 | 4 | 2 | 1 | 87,040 B | 152 |
| 160 | 64 | 32 | 8 | 2 | 2 | 64,512 B | 120 |
| 192 | 64 | 32 | 8 | 2 | 2 | 76,800 B | 136 |
| 256 | 32 | 32 | 4 | 2 | 2 | 84,480 B | 168 |

（寄存器列为模型估算：O 累加器 + Q 片段 + S/P 片段 + ~24 临时；实测值需 ncu 采集。）

## 3. 被否决的配置与理由

| 配置 | 否决理由 |
| --- | --- |
| D=128, BM=128, BN=64, stages=2 | smem = 34,816 + 2×34,816 = 104,448 B > 101,376 B |
| D=128, BM=128, BN=64, stages=3 | 同上，且 3 级流水收益被 smem 上限吃掉 |
| D=256, BM=64, BN=64, stages=2 | smem = 33,792 + 2×33,792 = 101,376 B（恰好在极限，无余量给 LSE/调试） |
| D=160/192/256 且 splits_d=1 | O 累加器 80/96/128 floats/线程，超寄存器预算并溢出 |
| 任意 D 的 BM=128 且 warps=8 以上 | 占用率不升反降（1 CTA/SM 时线程越多同步开销越大），本机 36 SM 下无收益 |

## 4. 复现

```bash
python tools/smem_calc.py --dtype fp16 --head-dims 32,64,96,128,160,192,256
python tools/smem_calc.py --dtype fp16 --head-dim 128 --tile 128 64 4 2   # 期望 exit 1（超限）
./build/benchmarks/benchmark_attention --suite main --only D128
ctest --test-dir build -R test_tile_smem_calc --output-on-failure
```

