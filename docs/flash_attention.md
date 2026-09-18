# FlashAttention 设计与实现

> 实现文件：`src/backends/flash/*`（kernel）、`src/cuda_common/*`（指令原语）。
> 数学口径见 `docs/attention_math.md`，资源预算见 `docs/kernel_design.md`。

## 1. 为什么是"IO-aware"

标准实现把 `S = QKᵀ` 写成 `[B, H_q, S_q, S_kv]` 的 FP32 张量，再把 `P = softmax(S)` 写回 HBM，
`PV` 时再读回来。以 `B=1, H_q=32, S_q=S_kv=4096, D=128` 为例：

```text
S 缓冲 = 1 × 32 × 4096 × 4096 × 4 B = 2 GiB
每次 forward 的 S/P 往返 = 写 S + 读 S + 写 P + 读 P = 4 × 2 GiB = 8 GiB 的额外 HBM 流量
```

FlashAttention 用「分块 + 在线 softmax」把这块显存彻底消除：`S/P` 只存在于 mma fragment（寄存器）与
smem 中，HBM 流量回到「读 Q/K/V 一次 + 写 O 一次」。

本项目对该结论给出**可验证的字段**：`AttentionRuntimeStats::materialized_score_bytes`
（naive 为 `B·H_q·S_q·S_kv·4`，flash 为 0）与 `workspace_bytes`（flash 为 0）。

## 2. 算法（逐步对应 `flash_fwd_impl.cuh`）

```text
Grid  = (ceil(S_q / BLOCK_M), H_q, B)        Block = warps × 32
每个 warp 负责 16 行（一个 m16 tile），kSplitsD 个 warp 共享同一组行
每 CTA：load Q tile → Q fragments（ldmatrix.x4）
        m,l = -inf,0 ; O = 0（寄存器）
        for j0 in 0..ceil(S_kv / BLOCK_N):
            if causal and j0 整块不可见: break            // 整块判定，无逐元素分支
            cp.async 载入 K/V tile（多级流水）
            S = Q_frag @ K_fragᵀ * scale                   // mma.sync m16n8k16, FP32 累加
            if 需要掩码: 对角线块逐元素置 -inf
            m_new = max(m, rowmax(S))
            alpha = exp2((m - m_new) * log2e)
            P     = exp2((S - m_new) * log2e)              // beta 吸收进 P
            l     = alpha * l + rowsum(P)
            O     = alpha * O + P @ V                      // P 打包成 A-fragment，V 用 ldmatrix.trans
        O = O / l ;  (可选) LSE = m + log(l)
```

与 `docs/attention_math.md` 的公式逐条对应；`alpha/beta` 的等价性推导见 `docs/online_softmax.md`。

## 3. 指令级选择（依据 `docs/env_report.md`）

| 环节 | 使用 | 原因（实测） |
| --- | --- | --- |
| QKᵀ / PV | `mma.sync.aligned.m16n8k16.row.col.f32.f16/bf16` | sm_120 支持；`wgmma` 被 ptxas 拒绝 |
| fragment 装载 | `ldmatrix.sync.aligned.m8n8.x4`（K/Q）与 `.x4.trans`（V） | 运行时探针已验证 fragment 布局 |
| tile 搬运 | `cp.async.cg` 16 B + `commit_group`/`wait_group` | 双缓冲流水，避免寄存器搬运 |
| 禁用 | `wgmma`、`tcgen05`、Hopper cluster | 见 `.agent/decisions.md` D-001 |

> `.trans` 的语义（lane 收到 `M[2c][g], M[2c+1][g]`）来自 walk-through 推导 + 探针；
> 若与实现不符，`test_flash_correctness` 的 PV 结果会立即暴露（误差量级 ≥ 0.1 而非 1e-3）。

## 4. Tile 与流水

每个 head_dim 一组标定配置（`src/backends/flash/flash_tile_config.hpp`）：

| head_dim | BLOCK_M | BLOCK_N | warps | stages | splits_d | smem（fp16, 含 padding） |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | 64 | 4 | 2 | 1 | 25,600 B |
| 64 | 64 | 64 | 4 | 2 | 1 | 46,080 B |
| 96 | 64 | 64 | 4 | 2 | 1 | 66,560 B |
| 128 | 64 | 64 | 4 | 2 | 1 | 87,040 B |
| 160 | 64 | 32 | 8 | 2 | 2 | 64,512 B |
| 192 | 64 | 32 | 8 | 2 | 2 | 76,800 B |
| 256 | 32 | 32 | 4 | 2 | 2 | 84,480 B |

要点：

* 行 padding 8 个 half（16 B）：既保证 ldmatrix 的 16 B 对齐，又把行距从 128 B 的整数倍上挪开，
  消除 8 路 bank conflict。
* `head_dim > 128` 时 `splits_d = 2`：两个 warp 共享同一组 16 行，各自累加一半的 D 维输出，
  使 O 累加器保持在寄存器预算内（代价是 QKᵀ 被重复计算一次，已在文档中声明）。
* smem 上限由 `static_assert`（编译期）与 `DeviceCapability::SmemLimitPerBlock()`（运行期）双重把关。

流水结构（`kStages` 级，默认 2）：

```text
prologue : issue tile 0..kStages-2 → 每次 commit 一个 group
loop j   : wait_group(kStages-2)      // 第 j 块已就绪
           __syncthreads()
           issue tile j+kStages-1 到 (j-1) mod kStages 缓冲   // 与下面的计算重叠
           commit
           compute(tile j)
           __syncthreads()
```

## 5. varlen 支持（当前形态）

* 接口：`flash_attention_varlen(q, k, v, num_seqs, cu_seqlens_q, cu_seqlens_kv, max_seq_q,
  max_seq_kv, cfg)`；`q/k/v` 为打包的 `[T, H, D]`，`cu_seqlens_*` 为**宿主**数组（长度 num_seqs+1）。
* 当前实现：对每个序列发起一次 `(1, S_q, S_kv)` 的 kernel —— 正确，但没有跨序列并行。
* Roadmap：融合 varlen kernel（把 `cu_seqlens` 放上设备、用 block 级偏移直接索引），
  以及在 decode 阶段与 PagedKVCache 结合。
* LSE 输出在 varlen 路径暂不提供（返回 `kUnsupportedFeature`），避免在半成品接口上给出错误保证。

## 6. FP32 边界

flash 后端只实例化 FP16/BF16：FP32 会把寄存器需求翻倍（O 累加器 64 → 128 floats/线程），
且 FP32 在本项目中的定位是**数值参考**（naive 后端）与教学/tile 对照（tiled 后端）。
调用 `flash_attention` 传 FP32 会得到 `kUnsupportedDtype`，错误消息直接指向 naive/tiled。

## 7. 复现

```bash
bash scripts/configure.sh --build-type Release && bash scripts/build.sh -j"$(nproc)"
./build/tests/kernel/test_flash_correctness            # 精度矩阵
compute-sanitizer --tool racecheck ./build/tests/kernel/test_flash_correctness
```

