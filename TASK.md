# TASK: 从零构建 SciCompute-Attention（GPU 原生 Attention 基础设施）

> 创建时间：2026-09-19
> Workflow：rl-infra（contract: `~/.codex/workflows/contracts/rl-infra.skills.md`）
> 唯一执行提示词：`Prompt/SciCompute-Attention 提示词.md`（v1.0, 3256 行）

## Objective

在 `/home/violet/Workspace/Code/Project/RL_infra/SciCompute-Attention` 从零构建 SciCompute-Attention：
面向大模型训练与推理的 GPU 原生高性能 Attention 基础设施，以 FlashAttention、KV Cache、Decode Attention 为核心，
向 vLLM(C++) Serving 与 RLHF Rollout 提供统一的 Attention Runtime。

## Requirements

- R1 自研 CUDA Attention kernel（naive / tiled / flash / decode / paged），支持 FP16/BF16（FP32 为参考路径）、
  causal 与非 causal、head_dim ∈ {32,64,96,128,160,192,256}、GQA/MQA。
- R2 自研 online softmax（含推导文档与极端 logits 用例），flash 路径不得物化 N×N（workspace 字节数证据）。
- R3 KV Cache 完整生命周期（append/gather/reset/stats）+ BlockManager + PagedKVCache（乱序 page 正确）。
- R4 公共 C++ API（`include/scicompute_attention/`，无异常，`sci::Status`/`Result<T>`）+ Dispatcher + Explain()。
- R5 复用 SciComputeInfra 的 Tensor/Device/Stream/Memory/Benchmark，禁止重复实现（见 §2.3 复用矩阵）。
- R6 测试体系 L1–L6：GoogleTest + compute-sanitizer + pytest。
- R7 Benchmark 体系（warmup 20 / measure 100 / P50-P99 / JSON+CSV+Markdown / 绑定 git sha）。
- R8 文档：README（19 章）+ `docs/`（22 篇）+ `docs/results/`。
- R9 RLHF/Rollout 形态 demo（prefill + decode 循环，输出 tokens/s 与 KV 峰值）。
- R10 每 Phase 独立 commit，过程记录落盘。

## Constraints

- 硬件：RTX 5070 Laptop（sm_120，36 SM，smem/SM 102400 B，regs/SM 65536，threads/SM 1536，L2 32 MiB，显存 8177909760 B）。
- CUDA 13.2（V13.2.86）；宿主 C++20、CUDA C++17；CMake 3.31.6（提示词记录为 4.4.0-rc1，差异见 `docs/env_report.md`）。
- Python 通道固定为 `cuda_132` 环境：`/home/violet/Workspace/miniconda/envs/cuda_132/bin/python`（3.12.13 / torch 2.13.0+cu132 / triton 3.7.1）。
- 禁止 wgmma / tcgen05（sm_120 ptxas 实测拒绝）；禁止硬编码 `sm_120` 与 CUDA 安装路径。
- 禁止修改 `SciComputeInfra` / `vllm` / `RLHF` 三个上游仓库（只读 + `upstream/notes.md` 登记）。
- 显存 8 GB 硬约束：任何 benchmark 前必须做显存预算检查，禁止与 Ollama（约 4882 MiB）并发压满。
- 正确性优先级：正确性 > 数值稳定 > 显存效率 > 占用率 > Tensor Core 利用率 > 指令效率 > 端到端吞吐。

## Verification Criteria（验收标准）

- [ ] V1 `bash scripts/configure.sh --build-type Release && bash scripts/build.sh && bash scripts/test.sh` 全绿（exit 0）。
- [ ] V2 naive/tiled/flash/decode/paged 与 torch fp32/fp64 参考的误差在 §11.4 容差表内（可追溯 JSON）。
- [ ] V3 compute-sanitizer memcheck/racecheck/initcheck 对 kernel 测试零错误。
- [ ] V4 flash 路径 workspace 字节数 < N×N 物化所需字节数（给出倍数）。
- [ ] V5 KV Cache 全部 §8.7 用例通过（block_size 1/8/16/32/64、容量耗尽、多序列隔离）。
- [ ] V6 Benchmark JSON/CSV 落盘且字段与 §12.4 一致，含 git sha 与 GPU 状态。
- [ ] V7 `docs/results/*.md` 与 README 中的每个数字都能追溯到命令 + 数据文件。
- [ ] V8 README 19 章齐全，5 分钟可跑通（干净环境 clone + configure + build + test + example）。

## Plan（阶段分解）

### Phase 0: 审计上游与环境侦察
- [x] Task 0.1: 环境基线采集（GPU/CUDA/编译器/Python 通道）
- [x] Task 0.2: ISA 探针（mma.sync / cp.async / ldmatrix / TMA / wgmma）
- [x] Task 0.3: 三仓资产清点与版本锚点
- [x] Task 0.4: `docs/00-recon.md`、`docs/env_report.md`、`upstream/notes.md`、`.agent/*`

### Phase 1: 工程骨架与公共 API
- [ ] Task 1.1: CMake（options/arch/infra/deps）+ 公共头文件 + capability/workspace + dispatcher 骨架
- [ ] Task 1.2: `tests/unit/*`（config/status/capability/dispatch）全绿

### Phase 2: Naive Attention（Level 0，正确性参考）
- [ ] Task 2.1: 4 kernel（QKᵀ / mask+scale / row softmax / PV）+ workspace 显存检查
- [ ] Task 2.2: `tests/kernel/test_naive_correctness.cu` + `benchmark_naive.cu`

### Phase 3: Tiled Attention（Level 1）
- [ ] Task 3.1: smem 分块 + 寄存器累加（不物化 N×N）
- [ ] Task 3.2: 正确性 + sanitizer + 显存对比

### Phase 4: Online Softmax（独立可测）
- [ ] Task 4.1: 单块/前缀块更新路径，与两遍精确参考一致
- [ ] Task 4.2: 极端 logits 用例 + `docs/online_softmax.md`

### Phase 5: FlashAttention Kernel（Level 3）
- [ ] Task 5.1: mma.sync m16n8k16 + ldmatrix + cp.async 多级流水内核
- [ ] Task 5.2: causal/非 causal、GQA、D=64/128/256、边界用例
- [ ] Task 5.3: tile 配置表 + `docs/flash_attention.md` / `docs/kernel_design.md`

### Phase 6: Benchmark + Profiling
- [ ] Task 6.1: benchmark suite（flash/decode/paged/naive/tiled + sdpa 对照）
- [ ] Task 6.2: profiling 脚本 + ncu/nsys 结论 + roofline

### Phase 6.5: Triton 轨道
- [ ] Task 6.5.1: triton_kernels（flash/decode/paged/softmax）+ autotune 配置校验
- [ ] Task 6.5.2: 三方一致性 + `docs/results/triton_vs_cuda.md`

### Phase 7: KV Cache（Level 4 前置）
- [ ] Task 7.1: KVCache/BlockManager/PagedKVCache + 显存预算
- [ ] Task 7.2: `tests/kv/*`

### Phase 8: Decode Attention（Level 4）
- [ ] Task 8.1: split-K decode + reduce + 阈值标定

### Phase 9: Paged KV / Paged Attention（Level 5）
- [ ] Task 9.1: page table 间接寻址 + 乱序 page 正确性

### Phase 10: Python Binding
- [ ] Task 10.1: pybind11 `_core` + `ops.py`/`kv_cache.py` + pytest

### Phase 11: vLLM(C++) Adapter
- [ ] Task 11.1: 适配层（dtype/layout 迁移）+ A/B harness + 回滚开关

### Phase 12: RLHF / Rollout 接口
- [ ] Task 12.1: `examples/rollout_engine_stub.cpp` + `docs/rollout_interface.md`

### Phase 12.5: 真实模型对接（Qwen3-4B-Thinking-2507-Q8）
- [ ] Task 12.5.1: GGUF 元数据/形状/KV 预算 + 真实 shape benchmark
- [ ] Task 12.5.2: 真实权重 Q/K/V parity（可行则做，否则登记 Known Issue）

### Phase 13: 完整文档与收尾
- [ ] Task 13.1: README 19 章 + `docs/results/*` + Known Issues/Roadmap

## Checklist

- [x] 任务理解（Stage 0）
- [x] 创建 TASK.md 与 .agent/ 记录
- [x] 环境确认 / 数据检查
- [x] 方案设计（见 `.agent/decisions.md`）
- [ ] 实现
- [ ] 测试
- [ ] 验证（对照 Verification Criteria）
- [ ] 产出 Artifacts
- [ ] 更新 Memory

## Current Stage

Phase 0 完成 → 进入 Phase 1（工程骨架与公共 API）。

## Known Issues

- 详见各 Phase 结束时的记录与 README「已知限制与 Roadmap」章节。

## 备注

- 关键决策索引：`.agent/decisions.md`
- 失败记录索引：`.agent/failures.md`
- 复现命令与版本锚点：`docs/env_report.md`

