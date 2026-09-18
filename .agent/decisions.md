# 关键设计决策记录

> 格式：D-NNN | 日期 | 决策 | 备选方案 | 理由 | 影响

### D-001 | 2026-09-19 | 主 MMA 路径选 `mma.sync.aligned.m16n8k16`，禁用 wgmma/tcgen05
- 备选：Hopper/Blackwell 教程式 wgmma、tcgen05。
- 理由：本机 sm_120 上 `ptxas` 实测拒绝 `wgmma.fence`；tcgen05 仅 sm_100/sm_103。
- 影响：tile 设计围绕 m16n8k16 的 fragment 布局；`arch_features.cuh` 中 `kHasWgmma = false` 并注释证据。

### D-002 | 2026-09-19 | 内部计算布局选 BHSD，对外允许 BSHD 输入
- 备选：BSHD 内部布局（与上游 vLLM 原型一致）。
- 理由：GQA 映射 `h_kv = h_q / group_size` 后，Q/K/V 的 head 维相邻可减少 kernel 内 stride 计算；
  公开 API 接受两种布局并在入口做 stride 归一化，避免 kernel 内分支。
- 影响：`detail/layout_traits.hpp` 是唯一布局换算实现（DRY）。

### D-003 | 2026-09-19 | KV 布局选 block-major `[layers][blocks][block_size][n_kv_heads][head_dim]`
- 备选：token-major `[layers][n_kv_heads][seq][head_dim]`。
- 理由：与上游 `vllm::KVCache` 同构，可零拷贝映射；分页复用粒度为 block，避免整体搬迁。
- 影响：`ComputeSlotMapping` 成为唯一 slot 计算入口（C++/Python/vLLM 适配层共用）。

### D-004 | 2026-09-19 | 链接上游按「精确目标」而非 `sci_compute` INTERFACE
- 备选：`target_link_libraries(sci_attention PUBLIC sci_compute)`。
- 理由：`sci_compute` 传递依赖 `sci_gpu`/`sci_bridges`/cuBLAS/cuSPARSE，其中 `sci_bridges` 硬编码
  `/usr/local/cuda-13.2/include`（UP-001）。本项目只需 Tensor/Device/Memory/Math/Benchmark 目标。
- 影响：只链接 `sci_core sci_device sci_memory sci_tensor sci_math sci_benchmark`，规避硬编码路径风险。

### D-005 | 2026-09-19 | 未通过正确性测试的 kernel 不进入 benchmark
- 依据：提示词 §0.3 纪律 2 与 §结语优先级。
- 影响：每个 Phase 固定闭环「编译 → 测试 → 正确性 → benchmark → 记录 → commit」。

