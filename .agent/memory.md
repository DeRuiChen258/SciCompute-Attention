# 可复用经验

### M-001 sm_120 的可用指令集边界
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 与 `...bf16.bf16.f32` 可用；
  `cp.async.cg.shared.global`、`ldmatrix.x4.b16`、`cp.async.bulk`(1D/2D) 可用；`wgmma` 不可用。
- 复现：`tools/arch_probe/run_probe.sh`。

### M-002 Python 通道必须显式指定
- 默认 shell 的 `python3` 解析到 `unitree_rt`（torch 2.14.0+cu130 / triton 3.8.0），与本项目环境不一致。
- 本项目统一用 `/home/violet/Workspace/miniconda/envs/cuda_132/bin/python`。

### M-003 Triton 侧两条硬约束（实测）
- 动态 smem 上限 101,376 B：`BM=64,D=128,stages=2` 需要 106,496 B → `OutOfResources`。
- `@triton.jit` 必须定义在真实 `.py` 文件中（stdin/exec 会 `OSError: could not get source code`）。

### M-004 上游 SciComputeInfra 目标拓扑
- `sci_compute` 是 INTERFACE 聚合目标；单点链接应直接用 `sci_tensor/sci_memory/sci_core/sci_device/sci_math/sci_benchmark`。
- `sci_bridges`（OBJECT）内硬编码 `/usr/local/cuda-13.2/include`，非默认 CUDA 安装位置会构建失败。

### M-005 ldmatrix.x4 / .trans 的寄存器语义（实测）
- 非 trans：`reg_m` = 第 m 个 8×8 块；lane 拿到 `M[g][2c], M[g][2c+1]`（g = lane>>2, c = lane&3）。
- trans：lane 拿到 `M[2c][g], M[2c+1][g]`；因此对 16×16 的 V tile，
  `(reg0,reg1)` = 列块 0 的 (b0,b1)，`(reg2,reg3)` = 列块 +8 的 (b0,b1)。
- 复现：`tools/arch_probe` 的探针 + walk-through 程序（见 `failures.md` F-003）。

### M-006 块级归约必须广播（否则 softmax 静默归零）
- 症状是"输出全 0"而不是 NaN，很难从数值看出来的根因是 `all_masked` 判定被逐 warp 分裂。
- 通用规则：任何 `BlockReduce*` 都必须把结果写回共享单元并让所有线程读取（见 `numerics.cuh`）。

### M-007 flash 资源预算
- D=128 / bm=64 / bn=64 / stages=2 / fp16：smem 87,040 B（1 CTA/SM），寄存器约 160/线程。
- head_dim > 128 采用 `splits_d=2`（两个 warp 共享 16 行、各累加一半 D），否则 O 累加器超预算。
