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

### D-006 | 2026-09-19 | 上游集成改为「只读引用源文件 + 精确目标」，不再使用 add_subdirectory
- 备选：`add_subdirectory(${SCI_ATTENTION_SCICOMPUTE_INFRA_ROOT})`（提示词 §5.2 模式 1 默认）。
- 实测：上游所有 CMakeLists 使用 `${CMAKE_SOURCE_DIR}/...`。嵌套 `add_subdirectory` 时该变量仍指向**本仓**
  根目录（子目录里的 `project()` 会把它重新指向顶层），导致 include 目录与源文件路径错解析；
  实验记录见 `/tmp/sca_wrap_test` 的失败输出（`No SOURCES given to target: sci_bridges`）。
  另尝试「wrapper 子目录覆盖 CMAKE_SOURCE_DIR」，同样被嵌套 `project()` 重置 —— 两条路都被实测否决。
- 决策：`cmake/SciComputeInfra.cmake` 在 SOURCE 模式下把上游的 13 个 .cpp 编译成 `sci_device/sci_memory/
  sci_tensor/sci_math/sci_benchmark` 精确目标（不修改、不复制上游源码），并保留 PACKAGE 与 STUB 模式。
- 影响：新增 UP-008 登记；`docs/architecture.md` 与 README 需说明该集成方式；STUB 模式可用 CPU-only 编译 API 层。

### D-007 | 2026-09-19 | CUDA 编译单元不使用 `-Wpedantic`
- 理由：nvcc 生成的 `cudafe1.cpp` 会输出 GCC 视为扩展的 `#line` 指令，产生大量与代码无关的告警，
  使 `SCI_ATTENTION_WERROR=ON` 无法用于 CUDA 目标。改为 CUDA 侧 `-Wall -Wextra`，宿主侧保留 `-Wpedantic`。
- 影响：WERROR 构建实测 exit 0。

### D-008 | 2026-09-19 | varlen 入口显式增加 num_seqs 参数
- 背景：提示词 §6.5 的 `flash_attention_varlen(q,k,v,cu_seqlens_q,cu_seqlens_kv,max_seq_q,max_seq_kv,cfg)`
  无法确定序列个数（打包张量的 `dim(0)` 是总 token 数，不是序列数）。
- 决策：在 varlen 入口增加 `int64_t num_seqs`；同时把 `cu_seqlens_*` 明确为**宿主**数组
  （长度 num_seqs+1），使 `ValidateVarlenHost()` 可以直接校验，避免 D2H 同步。
- 影响：`IAttentionBackend::ForwardVarlen` 与 5 个后端的签名同步更新；文档记录于 `docs/flash_attention.md` §5。

### D-009 | 2026-09-19 | commit 粒度按"可构建的交付单元"而非严格逐 Phase
- 背景：提示词要求"每个 Phase 独立 commit，禁止巨型 commit"，同时要求"历史可 bisect"。
- 观察：Phase 逐条提交会让中间提交引用尚不存在的文件（例如 tests/kernel/CMakeLists.txt 提前登记
  后续 Phase 的测试目标），从而破坏 bisect 可构建性。
- 决策：按"每个提交都能配置/编译/测试通过"的交付单元提交（Phase 0、Phase 1、Level 0/1/3 内核、
  KV/Decode/Paged、Python+Triton、集成与文档），并在 `.agent/state.md` 中逐 Phase 记录完成项与验证证据。
