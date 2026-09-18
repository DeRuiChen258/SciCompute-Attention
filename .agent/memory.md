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

