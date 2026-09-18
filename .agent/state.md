# Agent State

> 更新时间：2026-09-19（Phase 1 结束）

## 当前阶段

Phase 1（工程骨架与公共 API）已完成，Phase 2（Naive Attention，Level 0）进行中。

## 已完成

- 环境基线实测：RTX 5070 Laptop / sm_120 / 36 SM / smem 102400 B / regs 65536 / threads 1536 / L2 32 MiB /
  显存 8177909760 B；CUDA 13.2（V13.2.86）；g++ 15.2.0；CMake 3.31.6。
- Python 通道复核：`cuda_132`（3.12.13 / torch 2.13.0+cu132 / triton 3.7.1 / pybind11 3.1.0 / pytest 9.1.1 / ninja 可用）。
- ISA 探针复跑：mma.sync m16n8k16 / cp.async.cg 16B+commit+wait / ldmatrix.x4.b16 / TMA bulk 1D / TMA tensor 2D 全部 PASS；
  wgmma 在 sm_120 被 ptxas 拒绝（预期 FAIL）。
- 三仓资产清点与版本锚点：SciComputeInfra `61492ccc`、vllm `f6d0ea9e`、RLHF `1f7ba72b`。
- 上游缺陷登记 UP-001..UP-006（见 `upstream/notes.md`）。

## 下一步

Phase 2：`src/backends/naive/*`（QKᵀ / mask+scale / 行 softmax / PV 四个 kernel）+ 
`tests/kernel/test_naive_correctness.cu` + `benchmark_naive.cu`。

## 阻塞

无。

## Phase 1 完成明细

- CMake：`CMakeLists.txt` + `cmake/{SciAttentionOptions,SciAttentionArch,SciComputeInfra,SciAttentionDeps}.cmake`；
  警告策略、WERROR 开关、安装规则、配置摘要已就位。
- 上游集成：因上游 CMake 全部使用 `${CMAKE_SOURCE_DIR}`（实验证明嵌套 `add_subdirectory` 会错解析），
  改为「只读引用上游源文件 + 精确目标」的 SOURCE 模式（UP-008，`.agent/decisions.md` D-006）。
- 公共 API：18 个头文件 + `detail/*` 辅助头全部落地并可单独包含。
- 运行时：`capability`（探测+缓存）、`workspace`（256B 切片+越界检测）、`launcher`（smem 属性缓存+错误检查）、
  `dispatcher`（表驱动选择 + Explain + 回退链）、`backend_lookup`。
- 验证证据：
  - `bash scripts/configure.sh --build-type Release && bash scripts/build.sh -j32` → exit 0
  - `bash scripts/test.sh` → 5/5 CTest 通过（unit 标签）
  - `bash scripts/configure.sh --stub --build-dir build-stub && bash scripts/build.sh --build-dir build-stub` → exit 0，CPU-only 模式下单测同样 5/5
  - `-DSCI_ATTENTION_WERROR=ON` 构建 → exit 0（零警告）
  - `./build/examples/example_attention` → 正确打印 Explain 与各后端未实现原因（exit 3，符合 Phase 1 预期）

