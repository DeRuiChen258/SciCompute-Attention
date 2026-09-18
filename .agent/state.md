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


## Phase 7/8/9 完成明细（KV Cache / Decode / Paged）

- KV Cache：`src/kv_cache/{kv_cache.cpp,kv_cache_kernels.cu,block_manager.cpp,paged_kv_cache.cpp}`；
  block-major 布局、LIFO free list、page table 按 revision 增量上卡、容量耗尽状态不变。
- Decode：`src/backends/decode/{decode_splitk.cu,decode_config.hpp,decode_backend.cpp}`；
  split-K ∈ {1,2,3,4,8,16}，warp 内 lane 分片 + 稳定合并公式。
- Paged：`src/backends/paged/{page_table.cuh,paged_attention.cu,paged_backend.cpp}`；
  复用同一套 split-K kernel，仅把 K/V 寻址换成 page table 查表（零拷贝复用）。
- 验证证据：
  - `bash scripts/test.sh` → 13/13 CTest 通过（unit 5 + kernel 5 + kv 3）
  - `compute-sanitizer --tool memcheck ./build/tests/kernel/test_decode_correctness` → ERROR SUMMARY: 0 errors
  - 乱序 page：`PagedAttentionCorrectness.PhysicalBlocksAreOutOfOrder` 断言物理块非连续且结果与 FP64 参考一致
- 契约补充：
  - `KVCache::Append/Reset/Gather*` 的 `slot_mapping`/`block_ids` 必须是**设备指针**（避免隐藏同步）；
  - `PagedAttentionParams::layer` 新增（多 layer KV 下层号无法从参数推断，见 D-010）。
- 已知限制：
  - decode 只处理 S_q=1；2..8 的 small-q 走 flash（dispatch 表 r0 已如此路由）；
  - `PagedKVCache::AppendTokens` 每次 append 会上传 4 B/token 的 slot mapping（H2D）。
