# Kernel 设计与资源预算

> 每个 kernel 的线程映射、smem 分区、寄存器预算与同步点集中记录于此（提示词 §7.8 门禁）。
> 表格里的数字必须与代码中的 `static_assert` / 实测一致。

## 0. 通用约定

| 项 | 约定 |
| --- | --- |
| 命名空间 | device 侧 `sca::cuda`，host 后端 `sca` |
| launch | 全部经 `sca::runtime::LaunchRaw()`（smem 属性缓存 + 启动错误检查 + 可选 `SCI_ATTENTION_DEBUG_SYNC`） |
| 错误检查 | `SCI_CUDA_CHECK_LAST()`；release 路径不出现 `cudaDeviceSynchronize` |
| 数值 | softmax 核心用 `exp2f`（`exp(x) = exp2(x·log2e)`），累加器 FP32 |
| 归约 | warp shuffle + smem 广播（**禁止** `atomicAdd` 做行归约） |
| 中间缓冲 | 经 `sca::runtime::AcquireScratchFp32()`（进程级缓存，热路径零 cudaMalloc） |
| 布局 | 内核按 stride 寻址；`detail::StridesFor()` 是唯一 stride 来源 |

### 0.1 热路径同布局检查

```bash
rg -n "cudaDeviceSynchronize|cudaStreamSynchronize" src/ include/ | \
  grep -vE "launcher.cu|benchmark|test" || echo "no forbidden synchronisation"
```

最后一次执行结果记录在 §4。

## 1. Level 0：naive（`src/backends/naive/attention_naive.cu`）

定位：**正确性参考 + IO 基线**。刻意物化 `S`，用来量化 flash 路径省掉了什么。

### 1.1 `naive_qk_scale_mask_kernel<T>`

```text
目的      : S[b,h,i,j] = scale * dot(Q[b,h,i,:], K[b,h_kv,j,:])，causal 时 j > i+diag 置 -inf
grid      : (ceil(B*H_q*S_q*S_kv / 256), 1, 1)，网格内 stride 循环
block     : 256 线程
索引映射  : idx -> (b, h, i, j)：j = idx % S_kv; i = (idx/S_kv) % S_q; ...
smem      : 0
寄存器    : 累积器 1 个 float + 索引；实测见 §5
同步      : 无
访存      : 每线程对 D 做一次连续读；Q 行被 S_kv 次重复读取（naive 的固有代价）
```

### 1.2 `naive_row_softmax_kernel`

```text
目的      : 行内稳定 softmax（原地覆盖 S → P），可选写 LSE
grid      : (B*H_q*S_q, 1, 1)，一个 block 处理一整行
block     : 256 线程
smem      : 32 float（warp 部分和的 scratch）
归约      : warp shuffle → smem[warp] → warp0 二次归约 → scratch[0] 广播（全部线程可见）
边界      : 全掩码行（m=-inf）输出全 0，不产生 NaN
```

> 实测教训：最初版本只让 warp0 拿到块级归约结果，其余 warp 保留各自的部分值，导致「整行被乘 0」。
> 归约必须把结果广播回所有线程；该约束已写入 `src/cuda_common/numerics.cuh` 注释。

### 1.3 `naive_pv_kernel<T>`

```text
目的      : O[b,h,i,d] = Σ_j P[b,h,i,j] * V[b,h_kv,j,d]，输出转回输入 dtype
grid      : (ceil(B*H_q*S_q*D / 256), 1, 1)
block     : 256 线程
smem      : 0
访存      : P 行连续读 + V 列跨 stride 读（非合并，naive 的第二个固有代价）
```

### 1.4 显存

```text
分数矩阵 = B * H_q * S_q * S_kv * 4 B（FP32）
B=1,H_q=32,S_q=S_kv=4096,D=128 -> 2 GiB   ← flash 路径的对照目标（0 字节）
```

## 2. 寄存器 / smem 预算表（起始值，Phase 6 用 `tools/tile_sweep.py` 校准）

| 后端 | 配置 | smem/block | O 寄存器/线程 | 备注 |
| --- | --- | --- | --- | --- |
| naive | — | 0 | ~20 | 无 tile |
| tiled | Phase 3 填表 | — | — | — |
| flash | D=128, bm=64, bn=64, warps=4, stages=2 | 87040 B（含 8 half padding） | 64 | ≤ 101376 B 上限 |

## 3. 边界用例清单（每个 kernel 必须覆盖）

```text
[x] S=1（单 token）
[x] S 非 tile 整数倍（7/63/65/127/129）
[x] causal 对角块（S_q != S_kv）
[x] head_dim ∈ {32,64,96,128,160,192,256}
[x] GQA group ∈ {1,2,4,8}
[x] 全掩码行不产生 NaN
[x] BHSD 与 BSHD 结果一致（naive 已覆盖；其余后端见各自测试）
```

## 4. 热路径同步检查结果

```bash
$ rg -n "cudaDeviceSynchronize|cudaStreamSynchronize" src/ include/ | grep -vE "launcher.cu"
（无输出）
```

`src/runtime/launcher.cu` 中的两处同步位于 `SCI_ATTENTION_DEBUG_SYNC=1` 的调试分支，属白名单。

