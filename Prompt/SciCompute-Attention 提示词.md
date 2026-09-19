# SciCompute-Attention 实施提示词（详细扩写版 v1.0）

> 本文件是 `Attention Infra 子系统.md`（大纲版）的逐节扩写。
> 用途：作为交给工程 Agent 的唯一执行提示词，粒度到「文件名称 + 文件内容 + 验收判据」。
> 定位：不是 FlashAttention 复现 demo，而是 **GPU 原生 Attention 基础设施**。
> 生成时间：2026-09-18。所有环境参数、上游 API、ISA 结论均为本机实测，来源见附录 B。

---

## 0. 使用说明（执行前必读）

### 0.1 你的角色与任务

你是资深 CUDA / LLM Infra 工程师 Agent。你的任务是在：

```text
$SCA_ROOT
```

从零构建 **SciCompute-Attention**：一个具有明确分层、可编译、可测试、可 benchmark、可 profiling、可被 vLLM(C++) 与 RLHF Rollout 集成的 Attention 子系统。

### 0.2 交付形态

| 维度   | 要求                                                                                   |
| ---- | ------------------------------------------------------------------------------------ |
| 代码   | C++17（CUDA 侧）/ C++20 宿主（与上游 SciComputeInfra 一致）、CUDA 13.2、CMake                      |
| 算子双轨 | CUDA kernel（C++/vLLM 路径，主实现）+ Triton kernel（Python 轨道，快速迭代 + autotune + 交叉验证，见 §7.9） |
| 接口   | C++ 公共 API + Python（pybind11）API                                                     |
| 验证   | GoogleTest 单测 + Python pytest + compute-sanitizer + benchmark + Nsight profiling     |
| 文档   | README + `docs/` 十余篇技术文档 + 可复现的性能报告                                                  |
| 语言   | 文档中文；代码标识符、注释、commit message 英文；公式注释可用数学符号                                           |
| 仓库   | 独立 git 仓库，按 Phase 分别 commit，禁止一个巨型 commit                                            |

### 0.3 执行纪律（硬性）

1. **先读后写**：任何文件在创建前，先确认上游是否已有等价实现（见 §2 复用矩阵）。
2. **先正确后优化**：未通过正确性测试的 kernel 不允许进入 benchmark 阶段。
3. **小步提交**：每个 Phase 结束必须「编译 → 测试 → 正确性 → benchmark → 记录 → commit」六步闭环。
4. **禁止一次性生成全部代码**：禁止在没有跑通 Level 0 的情况下写 Level 3。
5. **过程留痕**：项目内必须维护 `TASK.md` 与 `.agent/{state.md, decisions.md, memory.md, failures.md}`，每个 Phase 结束更新一次；决策与失败必须落盘，不允许只存在于对话里。
6. **回复语言**：过程汇报与最终总结使用简体中文；代码、命令、路径、指标名保留原文。
7. **数字必须可追溯**：任何写进 README/docs 的性能数字，必须同时给出「命令 + 数据文件路径 + GPU 状态」。

### 0.4 每阶段回复格式（固定）

```text
## Phase N：<名称>
完成项：<逐条>
验证证据：<命令 + 关键输出摘要（含 exit code）>
新增/修改文件：<绝对路径清单>
已知问题：<无 / 具体问题 + 影响 + 计划>
下一阶段入口条件：<是否满足，依据>
```

---

## 1. 环境事实基线（先复核，禁止假设）

### 1.1 本机实测环境（2026-09-18）

| 项                  | 实测值                                                                                                                               | 证据来源                                                                                               |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| GPU                | NVIDIA GeForce RTX 5070 Laptop GPU                                                                                                | `nvidia-smi`                                                                                       |
| Compute Capability | 12.0（`sm_120`，Blackwell consumer）                                                                                                 | `torch.cuda.get_device_properties(0)`                                                              |
| SM 数量              | 36                                                                                                                                | 同上                                                                                                 |
| 显存                 | 8,177,909,760 B（约 7.80 GiB），`nvidia-smi` 报 8151 MiB                                                                               | 同上                                                                                                 |
| L2 Cache           | 33,554,432 B（32 MiB）                                                                                                              | 同上                                                                                                 |
| 显存位宽 / 显存时钟        | 128-bit / 12,001,000 kHz                                                                                                          | 同上                                                                                                 |
| Shared Memory      | 每 SM 102,400 B；每 block 默认 49,152 B（需 `cudaFuncSetAttribute` 提升上限）                                                                 | 同上                                                                                                 |
| 寄存器                | 每 SM 65,536 个 32-bit 寄存器                                                                                                          | 同上                                                                                                 |
| 线程上限               | 每 SM 1536；每 block 1024；warp = 32                                                                                                  | 同上                                                                                                 |
| CUDA Toolkit       | 13.2（`nvcc V13.2.86`）                                                                                                             | `nvcc --version`                                                                                   |
| 宿主编译器              | g++ 15.2.0                                                                                                                        | `g++ --version`                                                                                    |
| CMake              | 4.4.0-rc1                                                                                                                         | `cmake --version`                                                                                  |
| Python（项目环境）       | 3.12.13（conda env `cuda_132`）                                                                                                     | `${SCA_PYTHON:-python3} -V`                                     |
| PyTorch（项目环境）      | 2.13.0+cu132（`torch.cuda.is_available()==True`，`torch.version.cuda==13.2`）                                                        | 同上 `-c "import torch..."`                                                                          |
| **Triton（项目环境）**   | **3.7.1**（本机 sm_120 上可正常 JIT 编译并运行）                                                                                               | `.../cuda_132/bin/python -c "import triton; print(triton.__version__)"`                            |
| pybind11           | 3.1.0                                                                                                                             | `.../cuda_132/bin/python -c "import pybind11; print(pybind11.__version__)"`                        |
| pytest             | 9.1.1                                                                                                                             | `.../cuda_132/bin/python -c "import pytest; print(pytest.__version__)"`                            |
| ninja              | 可用（Triton JIT 需要）                                                                                                                 | `.../cuda_132/bin/python -c "import importlib.util as u; print(u.find_spec('ninja') is not None)"` |
| GoogleTest         | 1.17.0（`libgtest-dev`）                                                                                                            | `dpkg -l \| grep gtest`                                                                            |
| Google Benchmark   | 系统未安装；vcpkg 已构建（`$SCA_BENCHMARK_PREFIX/share/benchmark/benchmarkConfig.cmake`） | `find / -name benchmarkConfig.cmake`                                                               |
| Python vLLM        | **未安装**（`import vllm` → ModuleNotFoundError）                                                                                      | `python3 -c "import vllm"`                                                                         |
| 本机 vLLM            | 是 **C++ 实现**：`$SCA_VLLM_ROOT`                                                                  | 目录结构                                                                                               |
| 真实模型               | `$SCA_MODEL_DIR`（GGUF Q8_0，4.28 GB，qwen3 架构，36 层 / H_q=32 / H_kv=8 / D=128，见 §23）    | 目录 + `/api/show`                                                                                   |
| 模型服务               | Ollama 用户级 server `http://127.0.0.1:11435`（tag `qwen3:4b-thinking-2507-q8_0`，GPU 占用 ≈ 4882 MiB）                                   | `serve.sh` / `/api/ps`                                                                             |

**结论**：单卡、无 NVLink、无 MIG、显存受限（8 GB）。所有 kernel 与 KV Cache 设计必须把「显存上限检查」当作一等公民，而不是事后补丁。

**Python 环境陷阱（必须遵守）**：默认 shell 的 `python3` 解析到 `unitree_rt` 环境（Python 3.12.9 / torch 2.14.0+cu130 / triton 3.8.0），与本项目环境**不一致**。本项目所有 Python 相关命令必须使用 `cuda_132`：

```bash
conda activate cuda_132          # 或显式使用 ${SCA_PYTHON:-python3}
python -c "import torch, triton; print(torch.__version__, triton.__version__)"   # 期望：2.13.0+cu132 3.7.1
```

所有 Python 测试、benchmark、pybind11 构建必须在该环境下执行；报告中必须记录实际使用的解释器绝对路径与版本。

### 1.2 ISA 能力探测结论（本机 `nvcc -arch=sm_120` 实测）

| 能力                                                                           | 结论             | 探针结果                                                                                                                          |
| ---------------------------------------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`                          | ✅ 支持           | 编译通过（Amp 级 MMA，FP16/BF16 → FP32 累加）                                                                                           |
| `cp.async.cg.shared.global` + `commit_group/wait_group`                      | ✅ 支持           | 编译通过（异步拷贝 + 多级流水）                                                                                                             |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16`                                   | ✅ 支持           | 编译通过（片段装载）                                                                                                                    |
| `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`（TMA 1D bulk） | ✅ 支持           | 编译通过（需 mbarrier）                                                                                                              |
| `cp.async.bulk.tensor.2d.shared::cta.global...`（TMA tensor，需 `CUtensorMap`）  | ✅ 支持           | 编译通过                                                                                                                          |
| `wgmma.*`（Hopper 异步 warpgroup MMA）                                           | ❌ 不支持          | `ptxas: Instruction 'wgmma.fence' not supported on .target 'sm_120'`                                                          |
| `tcgen05.*`（数据中心 Blackwell 张量核心）                                             | ⛔ 不适用          | sm_100/sm_103 专属，禁止照搬 CUTLASS Blackwell 教程                                                                                    |
| Triton JIT 运行（FP16 attention）                                                | ✅ 可用           | triton 3.7.1（cuda_132）在本机编译运行 FP16 FlashAttention 风格 kernel，与 `torch SDPA` 最大绝对误差 ≈ 9.8e-4（causal, D=128）/ ≈ 1.2e-4（非 causal） |
| Triton 动态 smem 上限                                                            | **101,376 B**  | `BM=64, D=128, num_stages=2` 触发 `OutOfResources: Required 106496, Hardware limit 101376`；Triton 配置必须按此上限校验                    |
| Triton kernel 定义位置                                                           | 必须是真实 `.py` 文件 | `@triton.jit` 需要 `inspect.getsourcelines`，在 stdin/exec 字符串中定义会报 `OSError: could not get source code`                          |

**由此确定的实现路线（硬约束）**：

```text
主 MMA 路径   : mma.sync m16n8k16（FP16 → FP32 / BF16 → FP32 累加）
主异步路径    : cp.async 16B + 多级流水（stages = 2..3）
可选优化路径  : TMA（cp.async.bulk / cp.async.bulk.tensor），必须由 Phase 0 探针复测后再启用
禁用路径      : wgmma、tcgen05、Hopper TMA cluster 特性
```

### 1.3 环境复核要求

- 新建 `tools/arch_probe/arch_probe.cu`，把 §1.2 的每一条探针写成**可独立编译的源码**，并在 Phase 0 复跑。
- 输出写入 `docs/env_report.md`（模板见 §5.7），必须包含：原始命令、编译退出码、失败信息原文、`deviceQuery` 式属性表。
- 后续所有 tile size / stages / smem 决策必须能引用这份报告，禁止「凭记忆」写死参数。

---

## 2. 复用边界：上游工程事实与复用矩阵

### 2.1 上游 SciComputeInfra（必须先读、优先复用）

路径：`$SCA_INFRA_ROOT`

已核实的工程事实：

| 事实                  | 内容                                                                                                                                                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层构建                | `project(SciComputeInfra VERSION 0.1.0 LANGUAGES CXX CUDA)`；宿主 `CMAKE_CXX_STANDARD 20`，CUDA `17`                                                                                                                     |
| 选项                  | `SCI_BUILD_TESTS`(OFF)、`SCI_BUILD_BENCHMARKS`(OFF)、`SCI_ENABLE_CUDA`(ON)、`SCI_ENABLE_OPENMP`(ON)                                                                                                                     |
| 架构变量                | `SCI_CUDA_ARCHITECTURES` 默认 `"75;89;90;120"`（可覆盖）                                                                                                                                                                    |
| 依赖                  | `find_package(GTest QUIET)`、`find_package(benchmark QUIET)`（缺失则跳过对应目标）                                                                                                                                               |
| 库目标                 | `sci_core`、`sci_memory`、`sci_tensor`、`sci_scheduler`、`sci_bridges`(OBJECT，含 `cuda/kernels/softmax.cu`+`layernorm.cu`)、`sci_compute`(INTERFACE)                                                                       |
| 关键可用 API            | `sci::Tensor`/`TensorShape`/`DType`/`Device`/`Layout`/`Result<T>`/`Status`（`include/core/status.hpp`、`include/tensor/tensor.hpp`）                                                                                    |
| 设备/流                | `sci::Stream`（`Create/GetCurrent/handle/synchronize`）、`sci::Event`、`DeviceManager`（`include/device/*.hpp`）                                                                                                           |
| 内存                  | `MemoryPool`/`Allocator`/`BufferHandle`（`include/memory/*.hpp`）                                                                                                                                                      |
| 数学                  | `math::softmax_stable`、`math::log_softmax`、`math::ref::softmax_f32`（`include/math/softmax.hpp`）                                                                                                                      |
| 基准框架                | `sci::benchmark::BenchmarkRunner`（`add_case/run/export_json/export_csv/export_markdown`）、`Timer`、`percentile`、`mean`、`stddev`、`estimate_bandwidth_gbps`、`calculate_throughput`                                       |
| 遗留 Attention 资产     | `cuda/kernels/attention.cuh` **只有声明**：`scaled_dot_product_attention_kernel`、`launch_scaled_dot_product_attention`、`launch_flash_attention`、`masked_attention_kernel`、RoPE kernel；仓库内**无对应 `.cu` 实现**，且未加入任何 CMake 目标 |
| 第二套遗留路径             | `include/gpu/gpu_tensor.hpp`（`GpuTensor`/`GpuDType`）与 `include/gpu/cuda_kernels.hpp:114` 的 `scaled_dot_product_attention(...)` 声明                                                                                    |
| 已知缺陷（需在 Phase 0 复核） | `src/bridges/CMakeLists.txt` 仍硬编码 `target_include_directories(... /usr/local/cuda-13.2/include)`；`cuda/helpers/`、`cuda/launch/`、`src/kernel/`、`src/graph/`、`python/*` 为空目录                                           |

### 2.2 上游 vLLM（C++）现状

路径：`$SCA_VLLM_ROOT`

| 文件                                             | 现状                                                                                                                                                                                   |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `include/vllm/attention/attention_backend.hpp` | `enum class AttentionBackend { kPagedAttention, kFlashAttention, kAuto };` + `to_string`                                                                                             |
| `include/vllm/attention/flash_attention.hpp`   | `struct FlashAttentionParams`（**FP32**、布局 `[batch, seq_len, num_heads, head_dim]`、含 `num_kv_heads`、`is_causal`、`stream`）；`void flash_attention_forward(const FlashAttentionParams&)` |
| `src/attention/flash_attention.cu`（81 行）       | 单 warp（`dim3 block(32)`）原型：`shared_qk` 仅存 Q、两次遍历 K/V（先求 max/l，再算输出）、`out` 未初始化、无 GQA 正确映射、FP32 only                                                                                  |
| `include/vllm/memory/kv_cache.hpp`             | 按 layer 连续分配的 block KV：[num_blocks × block_size × num_kv_heads × head_dim]，提供 `k_ptr/v_ptr/clear_block/total_bytes`                                                                  |
| `include/vllm/core/block_manager.hpp`          | 存在 block 管理接口（Phase 0 必读）                                                                                                                                                            |
| 其它                                             | `src/kernels/*`、`src/layers/attention.cpp`、`src/core/*` 齐全，但 attention 计算路径即上述原型                                                                                                     |

**结论**：SciCompute-Attention 的 vLLM 集成目标是这个 **C++ vLLM**，不是 Python vLLM。适配层需要把 `FlashAttentionParams`（FP32/BSHD）转成本项目 API（FP16/BF16 + BHSD 内部布局），并在文档中显式声明精度变化与迁移策略。

### 2.3 复用矩阵（逐能力判定，禁止越界重写）

| 能力                                                  | 上游现状         | 本项目动作                                                                          |
| --------------------------------------------------- | ------------ | ------------------------------------------------------------------------------ |
| Tensor / Shape / DType / Device / Result            | 完整可用         | **复用**（只读依赖）。禁止在本项目再定义一个 `Tensor` 类                                            |
| Stream / Event                                      | 完整可用         | **复用**；所有公开 API 接受 `sci::Stream*`                                              |
| 内存分配（pool/allocator）                                | 完整可用         | **复用** 于 workspace 与 KV Cache 底层缓冲                                             |
| `math::softmax_stable` / `ref::softmax_f32`         | 可用（CPU 参考为主） | **复用** 于 CPU 参考对照与单测 oracle，不作为 GPU 主路径                                        |
| BenchmarkRunner / Timer / 统计函数                      | 可用           | **复用**；若缺 P50/P90/P99 与 JSON schema 字段，则在 `benchmarks/common/` 内补齐导出层，**不改上游** |
| `cuda/kernels/attention.cuh` 声明                     | 无实现、未编译      | **保留但标记 deprecated**（只加注释与 `#include` 守卫说明），实现放在本项目                            |
| `include/gpu/*`（GpuTensor 路径）                       | 遗留第二套 GPU 抽象 | **不复用、不扩展**；文档中标注为 legacy，避免双轨                                                 |
| Attention kernel / KV Cache / Paged KV / Dispatcher | 不存在          | **本项目实现**                                                                      |
| vLLM attention 原型                                   | 存在但缺陷明确      | **作为基线与对照**，本项目提供替换实现与 A/B 数据                                                  |

### 2.4 与上游的链接方式（按优先级，必须在 CMake 中显式实现）

```text
模式 1（默认，推荐）：add_subdirectory(${SCI_ATTENTION_SCICOMPUTE_INFRA_ROOT})
    - 默认探测 ../SciComputeInfra
    - 强制把上游 SCI_BUILD_TESTS / SCI_BUILD_BENCHMARKS 置 OFF，避免目标名冲突
模式 2（未来）：find_package(SciComputeInfra) —— 仅当上游提供 install/export 时启用
模式 3（兜底，仅 CPU 接口编译）：SCI_ATTENTION_INFRA_MODE=STUB
    - 允许在没有上游时编译 API 层与单测骨架
    - 任何 GPU kernel 验收与性能结论禁止使用 STUB 模式
```

**硬性禁令**：禁止把 SciComputeInfra 源码复制进本项目；禁止修改上游文件（若确有必要，只在 `upstream/patches/NNNN-*.patch` 中提供补丁并在 `.agent/decisions.md` 记录理由，且必须由用户确认后再应用）。

---

## 3. Phase 0：侦察清单（不可跳过）

### 3.1 必跑命令（逐条执行并把原始输出摘要写入 `docs/00-recon.md`）

```bash
# 0) 先激活项目 Python 环境（默认 shell 的 python3 属于 unitree_rt，版本与本项目不一致）
conda activate cuda_132
python -c "import torch, triton; print(torch.__version__, triton.__version__)"   # 期望 2.13.0+cu132 / 3.7.1

# 0) 环境基线
uname -a
nvidia-smi
nvidia-smi -q -d CLOCK,PERFORMANCE,POWER | head -80
nvcc --version && cmake --version && g++ --version | head -1
python3 -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python3 -c "import torch;print(torch.cuda.get_device_properties(0))"
python3 -c "import pybind11;print(pybind11.__version__)"

# 1) 上游资产清点
cd $SCA_INFRA_ROOT
rg --files -g '!build*' | sort
rg -n "flash_attention|scaled_dot_product|attention" --glob '!build*' -S .
cat CMakeLists.txt src/CMakeLists.txt src/bridges/CMakeLists.txt

# 2) vLLM(C++) 资产清点
cd $SCA_VLLM_ROOT
git log -1 --format='%H %ad %s'
rg --files include/vllm src/attention src/layers | sort
cat include/vllm/attention/*.hpp include/vllm/memory/kv_cache.hpp

# 3) RLHF 侧接口清点
cd $SCA_RLHF_ROOT
rg --files rlhf | sort
rg -n "rollout|kv_cache|attention" rlhf/rollout CMakeLists.txt | head -50

# 4) PyTorch 参考能力清点
python3 - <<'PY'
import torch, torch.nn.functional as F
print("sdpa backends:", torch.backends.cuda.flash_sdp_enabled(),
      torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
print("sdpa avail:", F.scaled_dot_product_attention is not None)
PY

# 5) 真实模型资产清点（Qwen3-4B-Thinking-2507-Q8）
ls -la $SCA_MODEL_DIR
sha256sum $SCA_MODEL_DIR/Qwen3-4B-Thinking-2507-Q8_0.gguf | head -1
curl -s --max-time 5  http://127.0.0.1:11435/api/version
curl -s --max-time 10 http://127.0.0.1:11435/api/show \
  -d '{"model":"qwen3:4b-thinking-2507-q8_0"}' | python3 -c \
  "import json,sys; mi=json.load(sys.stdin)['model_info']; print({k:v for k,v in mi.items() if k.startswith('qwen3.') or k.startswith('general.')})"
curl -s --max-time 5  http://127.0.0.1:11435/api/ps        # 已加载模型与显存占用
```

### 3.2 必读文件清单（读不完不许开工）

```text
SciComputeInfra:
  CMakeLists.txt, src/CMakeLists.txt, src/*/CMakeLists.txt, benchmarks/CMakeLists.txt, tests/CMakeLists.txt
  include/core/{status.hpp,types.hpp,common.hpp,timer.hpp,macros.hpp}
  include/tensor/{tensor.hpp,tensor_shape.hpp}
  include/device/{device.hpp,device_info.hpp,cuda_device.hpp,stream.hpp}
  include/memory/{allocator.hpp,memory_pool.hpp,buffer_handle.hpp}
  include/math/softmax.hpp
  include/benchmark/{benchmark_case.hpp,benchmark_runner.hpp}
  cuda/CMakeLists.txt, cuda/kernels/attention.cuh, cuda/kernels/softmax.cuh
  README.md, TASK.md, .agent/state.md
vLLM(C++):
  CMakeLists.txt, include/vllm/attention/*.hpp, include/vllm/memory/kv_cache.hpp, include/vllm/core/*.hpp,
  src/attention/*.cu*, src/layers/attention.cpp
RLHF:
  CMakeLists.txt, README.md, rlhf/rollout/*（接口与类型）
```

### 3.3 产出物与门禁

| 产出                    | 内容要求                                                             |
| --------------------- | ---------------------------------------------------------------- |
| `docs/00-recon.md`    | 上游资产表、复用/重写判定表（与 §2.3 对齐并逐条给出证据文件行号）、风险清单、待确认问题                  |
| `docs/env_report.md`  | 环境表 + ISA 探针结果原始输出 + 与本提示词 §1 的差异说明                              |
| `TASK.md`             | 目标、阶段计划、验收标准（模板：`~/.codex/workflows/templates/TASK.md`）          |
| `.agent/state.md`     | 当前阶段、已完成、下一步、阻塞                                                  |
| `.agent/decisions.md` | 至少记录：为何采用 `mma.sync` 而非 wgmma；为何内部布局选 BHSD；为何 KV 布局选 block-major |

**门禁**：§1.2 的探针结论若与实测不一致（例如 TMA 不可用），必须先更新本提示词对应章节的假设与本项目设计，再进入 Phase 1；不允许带着错误假设写 kernel。

### 2.5 三仓协同关系（本项目 = 中间层）

三个已有项目与本项目构成一条**单向依赖链**，禁止反向依赖，禁止循环依赖：

```text
                        RLHF / Rollout（业务层）
                        $HOME/.../RL_infra/RLHF
                        project(mini-rlhf-stack)  targets: rlhf_core, rlhf_cuda,
                                                  rollout_server, train_ppo
                                    │
                                    │ ① RolloutServer::serve(prompts)
                                    │ ② GPUActor::rollout(prompts)  → 需要 Attention + KV Cache
                                    ▼
                        vLLM(C++)（服务层 / 本次集成目标）
                        $HOME/.../RL_infra/vllm
                        project(vllm-cpp)  targets: vllm::vllm, vllm_server
                                    │
                                    │ ③ vllm::flash_attention_forward(FlashAttentionParams)
                                    │ ④ vllm::KVCache（block 布局）
                                    ▼
        ┌───────────────────────────────────────────────────────────┐
        │  SciCompute-Attention（本项目，Attention Infra 子系统）    │
        │  targets: sci_attention, sci_attention_kernels,            │
        │           sci_attention_vllm_adapter, scicompute_attention │
        └───────────────────────────────────────────────────────────┘
                                    │
                                    │ ⑤ 复用 Tensor/Device/Stream/Memory/Benchmark
                                    ▼
                        SciComputeInfra（基础设施层）
                        $HOME/.../RL_infra/SciComputeInfra
                        project(SciComputeInfra)  targets: sci_core, sci_memory,
                        sci_tensor, sci_scheduler, sci_bridges, sci_compute
```

依赖方向硬规则：

> 真实负载来源：`$SCA_MODEL_DIR`（Qwen3-4B-Thinking-2507 Q8_0 GGUF，
> 36 层 / H_q=32 / H_kv=8 / D=128 / causal），本项目的真实 shape、真实权重与服务基线均取自该模型，详见 §23。

```text
RLHF            → 依赖 vLLM(C++) 与本项目（可选；通过 adapter 或直接调用稳定 C++ API）
vLLM(C++)       → 依赖本项目（通过 vllm_backend/ 适配层）
本项目          → 依赖 SciComputeInfra（通过 CMake add_subdirectory）
SciComputeInfra → 禁止依赖本项目/vLLM/RLHF
本项目          → 禁止被 SciComputeInfra 反向 include
```

### 2.6 逐仓接口契约（必须逐条落地并验证）

#### 2.6.1 与 SciComputeInfra 的契约（向下）

| 契约项    | 具体内容                                                                                               | 落地文件                                              |
| ------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 构建复用   | `add_subdirectory(${SCI_ATTENTION_SCICOMPUTE_INFRA_ROOT})`，默认 `../SciComputeInfra`                 | `cmake/SciComputeInfra.cmake`、顶层 `CMakeLists.txt` |
| 类型复用   | 对外 API 只使用 `sci::Tensor`/`sci::TensorShape`/`sci::DType`/`sci::Device`/`sci::Result`/`sci::Status` | `include/scicompute_attention/*.hpp`              |
| 流复用    | 所有 kernel 启动必须落到 `sci::Stream`（`stream == nullptr` 时用 `Stream::GetCurrent()`）                      | `src/runtime/launcher.cpp`                        |
| 内存复用   | workspace 与 KV Cache 缓冲经 `sci::MemoryPool`/`Allocator` 获取；本项目禁止自建 `cudaMalloc` 封装层                 | `src/runtime/workspace.cpp`、`src/kv_cache/*.cpp`  |
| 基准复用   | 优先用 `sci::benchmark::BenchmarkRunner` 产出 JSON/CSV/Markdown；缺口在 `benchmarks/common/` 补齐，不改上游        | `benchmarks/common/bench_export.hpp`              |
| 上游缺陷回馈 | 上游 `src/bridges/CMakeLists.txt` 硬编码 `/usr/local/cuda-13.2/include` 等问题：**只登记 + 提供 patch**，不直接改     | `upstream/notes.md`、`upstream/patches/`           |
| 版本锚点   | 记录上游 `git rev-parse HEAD` 到 `docs/env_report.md`；不匹配时给出显式 warning                                  | `cmake/SciComputeInfra.cmake`                     |

#### 2.6.2 与 vLLM(C++) 的契约（向上，替换现有 attention 原型）

| 契约项    | 具体内容                                                                                                                               | 落地文件                                                      |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 入口函数   | 提供 `void flash_attention_forward_via_sca(const vllm::FlashAttentionParams&)`，签名与上游 `flash_attention_forward` 对齐                    | `vllm_backend/sca_flash_attention_adapter.{hpp,cpp}`      |
| 精度迁移   | 上游 FP32 → 本项目 FP16/BF16，必须在适配层显式转换并记录精度损失；提供 `SCA_VLLM_COMPUTE_DTYPE=bf16\|fp16` 开关（默认 `bf16`）                                     | 同上                                                        |
| 布局迁移   | 上游 `[B, S, H, D]`（BSHD，`num_kv_heads` 存在于参数中）→ 本项目内部布局；转换必须显式且可单测                                                                  | `vllm_backend/layout_bridge.hpp`                          |
| KV 迁移  | `vllm::KVCache`（`[num_blocks, block_size, num_kv_heads, head_dim]` 按 layer 连续）→ `sca::PagedKVCache`（同构 block 布局）；提供指针映射而非拷贝（零拷贝优先） | `vllm_backend/sca_paged_attention_adapter.{hpp,cpp}`      |
| 后端开关   | `vllm::AttentionBackend::kAuto / kFlashAttention / kPagedAttention` 三种取值都要有确定映射；不支持组合返回明确错误码，禁止静默 fallback                         | `vllm_backend/adapter_config.hpp`                         |
| 不污染上游  | 不修改上游 `src/attention/*`；如需 hook，只产出 `vllm_backend/patches/0001-attention-dispatch.patch` 并写明 apply 命令与回滚方式                         | `vllm_backend/patches/`、`docs/vllm_integration.md`        |
| A/B 验证 | 同一机器、同一输入，跑「上游原型 vs SCA」，输出延迟/精度对比表；精度对比以 PyTorch FP32 为参考                                                                         | `benchmarks/benchmark_vllm.cpp`、`docs/results/vllm_ab.md` |
| 缺陷登记   | 明示上游原型缺陷（单 warp、`out` 未初始化、两遍 K/V、GQA 映射错误、FP32 only），并在报告中给出行号证据                                                                  | `docs/vllm_integration.md`                                |

#### 2.6.3 与 RLHF / Rollout 的契约（向上，业务层）

| 契约项         | 具体内容                                                                                                                                                                                                                                                         | 落地文件                                                                                           |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| 现状事实        | `rlhf::RolloutServer`（`serve/update_weights/load_checkpoint`）与 `rlhf::GPUActor`（`rollout(prompts)`，默认 `max_seq_len=2048`）目前**没有 GPU attention 实现**；`rlhf::KVCache` 是 CPU `std::vector` dense 存储，布局注释为 `[num_layers, 2, batch, num_heads, seq_len, head_dim]` | 只读引用，写入 `docs/rollout_interface.md`                                                            |
| 迁移路径        | `rlhf::KVCache`（CPU dense）→ `vllm::KVCache`（block）→ `sca::PagedKVCache`（device block + block table）；三段路径与转换函数必须画进文档                                                                                                                                          | `docs/rollout_interface.md`、`examples/rollout_engine_stub.cpp`                                 |
| 接口要求        | 本项目提供 batch 推理接口（定长 + varlen）、KV 复用接口（append / reuse / reset）、连续推理（decode loop）所必需的最小 API；**不实现 PPO/GRPO 训练逻辑**                                                                                                                                              | `include/scicompute_attention/attention.hpp`、`include/scicompute_attention/paged_kv_cache.hpp` |
| 指标要求        | 面向 rollout 给出并记录：tokens/s、prefill 与 decode 的 P50/P99、KV 复用命中率、显存峰值                                                                                                                                                                                           | `docs/rollout_interface.md`、`benchmarks/benchmark_e2e_prefill_decode.cu`                       |
| RLHF 构建缺陷登记 | RLHF `CMakeLists.txt` 的 `CMAKE_CUDA_ARCHITECTURES "80;86;89;90"` 不含 `120`，本机实跑需覆盖；`gpu_ops.cu` 未加入 `rlhf_cuda` 源列表                                                                                                                                           | `docs/rollout_interface.md`（Known Issues）                                                      |

### 2.7 跨仓工作方式（本次执行的硬性要求）

1. **只写本项目 + 只读引用另两仓**：除非用户单独确认，本任务不允许在 `SciComputeInfra`、`vllm`、`RLHF` 三个仓库中产生任何提交。
2. 需要上游改动时，走 `upstream/patches/` + `upstream/notes.md` 路线，并在 `.agent/decisions.md` 记录「为什么必须改上游」。
3. 每个 Phase 的验证命令必须能独立执行，且路径写绝对路径，例如：

```bash
cmake -S $SCA_ROOT -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DSCI_ATTENTION_SCICOMPUTE_INFRA_ROOT=$SCA_INFRA_ROOT \
      -DSCI_ATTENTION_BUILD_TESTS=ON
```

4. 三仓关联的验收以「真实可跑的跨仓 demo」为准：

```text
Level A：本项目自测通过（单仓）
Level B：vllm_backend/ 适配层编译通过 + A/B benchmark 有数据        → 与 vllm 仓关联成立
Level C：examples/rollout_engine_stub.cpp 跑通 prefill+decode 循环 → 与 RLHF 仓关联成立
```

---

## 4. 目录与文件总清单（唯一真相源）

下列结构是**强制基线**。允许在实现中新增文件，但每个新增文件必须在 `docs/architecture.md` 的「文件职责表」里登记；不允许出现「无主文件」（没有任何文档说明其职责的源文件）。

```text
SciCompute-Attention/
├── CMakeLists.txt                       # 顶层构建：选项、上游引入、子目录、安装、CTest
├── README.md                            # 项目定位/架构/快速开始/性能表/限制（§18 规定章节）
├── LICENSE                              # MIT（与上游保持一致）
├── TASK.md                              # 阶段计划与验收标准（执行期维护）
├── .gitignore                           # build*/、*.o、*.so、__pycache__、profiling/reports/ 等
├── .clang-format                        # 与 vllm 仓保持一致的 C++ 风格
│
├── .agent/
│   ├── state.md                         # 当前 Phase、已完成、下一步、阻塞
│   ├── decisions.md                     # 关键设计决策（含被否决方案与理由）
│   ├── memory.md                        # 可复用经验（tile 参数、踩坑结论）
│   └── failures.md                      # 失败尝试与解决路径（禁止删除历史）
│
├── cmake/
│   ├── SciComputeInfra.cmake            # 定位/引入上游；校验目标存在；打印上游版本锚点
│   ├── SciAttentionOptions.cmake        # 全部 option 定义 + 互斥校验 + 默认值表
│   ├── SciAttentionArch.cmake           # 架构探测与校验（禁止硬编码 sm_120）
│   └── SciAttentionDeps.cmake           # GTest / Google Benchmark / pybind11 定位与降级策略
│
├── include/scicompute_attention/        # 【稳定公共 API：上层只允许包含这里的头文件】
│   ├── attention.hpp                    # umbrella 头：聚合导出全部公共 API + 命名空间说明
│   ├── attention_types.hpp              # AttnLayout/AttentionShape/KVLayout/枚举与常量
│   ├── attention_config.hpp             # AttentionConfig + Validate() + 默认 scale 规则
│   ├── attention_result.hpp             # AttentionResult（out/lse/stats）+ 生命周期说明
│   ├── backends.hpp                     # IAttentionBackend 抽象接口 + CapabilityReport
│   ├── naive_attention.hpp              # naive_attention(q,k,v,cfg)
│   ├── tiled_attention.hpp              # tiled_attention(q,k,v,cfg)
│   ├── flash_attention.hpp              # flash_attention(q,k,v,cfg) + 变长/varlen 入口
│   ├── decode_attention.hpp             # decode_attention(q,k,v,cfg) + split-K 参数
│   ├── paged_attention.hpp              # paged_attention(q, paged_kv, block_table, seq_lens, cfg)
│   ├── kv_cache.hpp                     # KVCache + KVCacheConfig + SlotMapping 工具
│   ├── block_manager.hpp                # BlockManager + BlockTable + 分配统计
│   ├── paged_kv_cache.hpp               # PagedKVCache（多序列、block table、device 视图）
│   ├── dispatcher.hpp                   # AttentionDispatcher + explain()（可解释决策）
│   ├── capability.hpp                   # DeviceCapability 探测结果结构体
│   ├── workspace.hpp                    # Workspace（预分配、复用、字节对齐、上限检查）
│   ├── status.hpp                       # 错误码扩展 + ToString + Result 使用约定
│   ├── version.hpp                      # SCI_ATTENTION_VERSION_MAJOR/MINOR/PATCH/STRING
│   ├── export.hpp                       # SCI_ATTENTION_API / visibility / noexcept 约定
│   └── detail/
│       ├── dtype_traits.hpp             # DType ↔ C++ 类型↔ CUDA MMA 类型 映射（模板）
│       ├── tile_config.hpp              # TileConfig 结构体（BLOCK_M/N/WARPS/STAGES 等）
│       ├── layout_traits.hpp            # BHSD/BSHD 的 stride/offset 计算（编译期 + 运行期）
│       └── host_utils.hpp               # 头文件内部小工具（对齐、整除、ceil_div）
│
├── src/
│   ├── CMakeLists.txt                   # 目标定义：sci_attention_kernels / sci_attention
│   ├── api/
│   │   ├── attention_api.cpp            # 公共 API 实现（形状检查→dispatcher→backend→Result）
│   │   └── api_validate.cpp             # 参数校验集中实现（唯一校验点，避免重复）
│   ├── cuda_common/                     # 【所有 kernel 共享的 CUDA 工具（纯头文件）】
│   │   ├── cuda_check.cuh               # SCI_CUDA_CHECK / SCI_CUDA_CHECK_LAST / debug 同步宏
│   │   ├── arch_features.cuh            # __CUDA_ARCH__ 分支 + 运行期 TMA/smem 能力查询
│   │   ├── vec_memory.cuh               # 128-bit 向量化 load/store + 对齐/边界保护
│   │   ├── numerics.cuh                 # exp2f 缩放、稳定 exp、NaN/Inf 哨兵、饱和处理
│   │   ├── mma_policy.cuh               # mma.sync 片段布局与 MMA 封装（FP16/BF16→FP32）
│   │   ├── ldmatrix.cuh                 # ldmatrix.x4 装载封装 + bank conflict 说明
│   │   └── cp_async.cuh                 # cp.async 多级流水（prologue/steady/epilogue）
│   ├── backends/
│   │   ├── naive/
│   │   │   ├── attention_naive.cuh      # 4 个 kernel 声明：QKᵀ / mask+scale / row softmax / PV
│   │   │   ├── attention_naive.cu       # naive 实现 + 模板实例化（fp32/fp16/bf16）
│   │   │   └── naive_backend.cpp        # IAttentionBackend 适配（含 workspace 需求）
│   │   ├── tiled/
│   │   │   ├── attention_tiled.cuh      # tiled kernel（Q/K/V block、smem 布局、warp 分工）
│   │   │   ├── attention_tiled.cu       # 启动与实例化（含 smem 容量检查）
│   │   │   └── tiled_backend.cpp        # 后端适配（限制：D>128 走 flash）
│   │   ├── flash/
│   │   │   ├── flash_fwd_kernel.cuh     # 核心 FlashAttention forward kernel（模板）
│   │   │   ├── flash_online_softmax.cuh # online softmax 状态更新（m/l/α/β + 输出重标定）
│   │   │   ├── flash_smem_layout.cuh    # smem 分区结构体 + 每配置字节数静态断言
│   │   │   ├── flash_mma_tile.cuh       # MMA 级 tile：Q×Kᵀ 与 P×V 的 fragment 映射
│   │   │   ├── flash_causal_mask.cuh    # 因果掩码（块级判定 + 对角线块特例，无逐元素分支）
│   │   │   ├── flash_tile_config.hpp    # host 侧 tile 选择表（表驱动，禁止 if-else 魔法数）
│   │   │   ├── flash_fwd_launch.cu      # 模板实例化 + cudaFuncSetAttribute + grid 计算 + launch
│   │   │   ├── flash_fwd_dispatch.cpp   # 运行期选择实例；不支持时返回明确错误码
│   │   │   └── flash_backend.cpp        # IAttentionBackend 适配
│   │   ├── decode/
│   │   │   ├── decode_splitk.cuh        # split-K decode kernel（partial m/l/O 写 workspace）
│   │   │   ├── decode_reduce.cuh        # 合并 partial（稳定合并 (m,l,O) 三元组）
│   │   │   ├── decode_config.hpp        # split 数规则（按 SM 数 36 与 seq_kv 推导）
│   │   │   ├── decode_attention.cu      # 启动与实例化（含 workspace 大小计算）
│   │   │   └── decode_backend.cpp       # 后端适配
│   │   └── paged/
│   │       ├── page_table.cuh           # device 侧 block table 查表（越界保护）
│   │       ├── paged_attention.cu       # paged kernel（不连续 page 的 K/V 访问）
│   │       └── paged_backend.cpp        # 后端适配（无 page table 时显式报错，不静默退化）
│   ├── kv_cache/
│   │   ├── kv_cache.cpp                 # KVCache 生命周期 + append/gather + 显存预算检查
│   │   ├── kv_cache_layout.cpp          # 地址计算（block/head/token/dim → offset）单点实现
│   │   ├── kv_cache_stats.cpp           # 使用统计：used/free blocks、碎片、峰值
│   │   ├── block_manager.cpp            # BlockManager：free list 分配/回收/block table 打包
│   │   └── paged_kv_cache.cpp           # PagedKVCache：多序列 append / page table 上卡 / reset
│   ├── runtime/
│   │   ├── dispatcher.cpp               # 后端选择 + 阈值读取 + explain() 文本
│   │   ├── capability.cpp               # DeviceCapability 探测（含首次调用缓存）
│   │   ├── launcher.cpp                 # 统一 launch 辅助（grid/smem/stream/错误检查）
│   │   ├── workspace.cpp                # 预分配与切片复用（256B 对齐，禁止 per-token malloc）
│   │   └── dispatch_table.inc           # 【脚本生成】阈值表（禁止手改，见 §9.4）
│   └── bindings/
│       └── python_module.cpp            # pybind11 模块 `scicompute_attention._core`
│
├── python/
│   ├── pyproject.toml                   # scikit-build-core 构建配置、依赖、entry points
│   ├── scicompute_attention/
│   │   ├── __init__.py                  # 导出 ops/KVCache/version/backend 列表
│   │   ├── ops.py                       # 参数校验 + 调用 _core + 错误映射 + 布局转换
│   │   ├── dispatch.py                  # Python 侧后端路由：cuda(C++ _core) / triton / auto
│   │   ├── reference.py                 # 纯 PyTorch 参考（naive / 逐步 online softmax，教学用）
│   │   ├── kv_cache.py                  # KVCache / PagedKVCache 封装（含 context manager）
│   │   ├── benchmark.py                 # 读 benchmark JSON → 打印 Markdown 对比表
│   │   ├── utils.py                     # dtype/layout 工具、断言、显存估算
│   │   ├── triton_kernels/              # 【Triton 轨道】所有 @triton.jit kernel 必须放这里
│   │   │   ├── __init__.py              # 导出 kernel 与 autotune 开关
│   │   │   ├── flash_attn_fwd.py        # Triton prefill（online softmax + causal + GQA）
│   │   │   ├── decode_attn.py           # Triton decode（split-K + 归约 kernel）
│   │   │   ├── paged_attn.py            # Triton paged attention（page table 间接寻址）
│   │   │   ├── online_softmax.py        # 独立 online softmax 算子（教学 + 逐块验证）
│   │   │   ├── softmax.py               # 行 softmax 算子（与 math::softmax / torch 对照）
│   │   │   └── autotune.py              # autotune 配置空间、缓存、关闭开关、结果导出
│   │   ├── version.py                   # __version__
│   │   └── py.typed
│   └── tests/
│       ├── conftest.py                  # device/dtype fixture、skip 条件（无 GPU 时 skip）
│       ├── test_api.py                  # 四个 backend 的调用与形状/错误路径
│       ├── test_parity.py               # 与 torch SDPA / reference.py 的数值一致性
│       ├── test_triton_parity.py        # Triton vs SDPA vs CUDA kernel 三方一致性
│       ├── test_triton_config.py        # autotune 配置合法性（smem ≤ 101376 B 等）
│       ├── test_qwen3_shapes.py         # 真实模型元数据与 KV 预算断言（§23）
│       ├── test_qwen3_parity.py         # 真实 Q/K/V 上的三方 parity
│       ├── test_kv_cache.py             # KV append/gather/reset、显存上限
│       └── test_errors.py               # 非法 dtype/head_dim/layout 的错误码与消息
│
├── tests/
│   ├── CMakeLists.txt                   # gtest_discover_tests + sanitize 目标
│   ├── data/qwen3_4b/                   # 真实 Q/K/V 落盘目录（大文件 gitignore，仅提交 manifest/README）
│   ├── common/
│   │   ├── test_utils.hpp               # 随机填充、误差统计（max/mean）、NaN/Inf 检测
│   │   └── torch_reference.py           # 生成 fp32/fp64 参考输出（供 C++ 测试读取）
│   ├── unit/
│   │   ├── test_config.cpp              # 配置校验、scale 默认规则、非法参数
│   │   ├── test_dispatch.cpp            # 后端选择正确性 + explain() 文本
│   │   ├── test_capability.cpp          # 能力探测字段与缓存行为
│   │   └── test_status.cpp              # 错误码 → 消息映射
│   ├── kernel/
│   │   ├── test_naive_correctness.cu    # Level 0 正确性（fp32/fp16/bf16）
│   │   ├── test_tiled_correctness.cu    # Level 1 正确性
│   │   ├── test_flash_correctness.cu    # Level 3 正确性（非因果）
│   │   ├── test_flash_causal.cu         # 因果路径（含对角线块边界）
│   │   ├── test_head_dims.cu            # D=64/128/256 × dtype 全组合
│   │   ├── test_numerics.cu             # 大 logits、全等值、单 token、极长序列
│   │   ├── test_layout.cu               # BHSD/BSHD 一致性与非法 layout 拒绝
│   │   ├── test_decode_correctness.cu   # decode（split-K 合并）
│   │   └── test_paged_correctness.cu    # paged（不连续 page、GQA）
│   ├── kv/
│   │   ├── test_kv_cache.cpp            # 生命周期/边界/统计
│   │   ├── test_block_manager.cpp       # 分配回收、耗尽、跨请求隔离
│   │   └── test_paged_kv.cpp            # 多序列 page table 正确性
│   └── integration/
│       ├── test_attention_api.cpp       # 走公共 API 的端到端用例
│       ├── test_dispatch_explain.cpp    # 阈值表驱动的行为快照
│       └── test_vllm_adapter.cpp        # 适配层编译+调用（启用时构建）
│
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── common/
│   │   ├── bench_utils.hpp              # warmup/measure、CUDA event 计时、clock 记录、seed
│   │   ├── bench_export.hpp             # JSON/CSV/Markdown 导出（补上游缺口）
│   │   └── bench_matrix.hpp             # 参数矩阵定义（sanity/main/long 三档）
│   ├── benchmark_memory_bw.cu           # 峰值带宽标定（roofline 的机器参数来源）
│   ├── benchmark_naive.cu               # Level 0
│   ├── benchmark_tiled.cu               # Level 1
│   ├── benchmark_flash.cu               # Level 3（核心）
│   ├── benchmark_decode.cu              # Level 4
│   ├── benchmark_paged.cu               # Level 5
│   ├── benchmark_sdpa.py                # PyTorch SDPA 对照（显式指定 backend）
│   ├── benchmark_triton.py              # Triton kernel 基准（含 autotune 扫描与 CUDA 对照）
│   ├── benchmark_qwen3_shapes.py        # 真实模型 shape（H_q=32/H_kv=8/D=128）的 prefill/decode 基准
│   ├── benchmark_vllm.cpp               # 上游 C++ vLLM 原型 vs SCA 的 A/B
│   ├── benchmark_e2e_prefill_decode.cu  # prefill+decode 端到端（rollout 形态）
│   └── results/.gitkeep                 # 结果落盘目录（大文件进 .gitignore）
│
├── profiling/
│   ├── README.md                        # 用法、指标解释、通过标准
│   ├── profile_naive.sh
│   ├── profile_flash.sh
│   ├── profile_decode.sh
│   ├── profile_triton.sh                # ncu/nsys 采集 Triton kernel（过滤 JIT 生成的 kernel 名）
│   ├── profile_vllm.sh
│   ├── analyze_ncu.py                   # ncu CSV → Markdown 摘要 + 与上次对比
│   └── metrics/
│       ├── fa_metrics.txt               # FlashAttention 必采指标列表
│       ├── decode_metrics.txt           # Decode 必采指标列表
│       └── roofline_metrics.txt         # roofline 所需指标
│
├── vllm_backend/
│   ├── CMakeLists.txt                   # 独立目标 sci_attention_vllm_adapter
│   ├── README.md                        # 集成步骤、开关、回滚
│   ├── adapter_config.hpp               # dtype 选择、开关、fail-fast 策略
│   ├── layout_bridge.hpp                # BSHD ↔ BHSD 转换（显式、可单测）
│   ├── sca_vllm_adapter.{hpp,cpp}       # 总适配器（参数转换 + 调用 + 错误映射）
│   ├── sca_flash_attention_adapter.{hpp,cpp}  # 对齐 vllm::flash_attention_forward
│   ├── sca_paged_attention_adapter.{hpp,cpp}  # 对齐 vllm::KVCache block 布局
│   └── patches/0001-attention-dispatch.patch  # 可选上游 hook（默认不应用）
│
├── examples/
│   ├── CMakeLists.txt
│   ├── example_attention.cpp            # C++ 最小调用示例（4 个 backend）
│   ├── example_kv_cache.cpp             # KV Cache 生命周期示例
│   ├── example_decode_loop.cu           # 自回归 decode 循环（含 KV 复用）
│   ├── example_qwen3_attention.py       # 真实 Qwen3-4B 权重（Q8_0 反量化）上的三方对比示例
│   ├── rollout_engine_stub.cpp          # mini rollout：批请求→prefill→decode→指标
│   └── example_python.py                # Python 调用与参考对比
│
├── tools/
│   ├── arch_probe/arch_probe.cu         # ISA/smem/TMA/wgmma 探针（§1.2 复现）
│   ├── smem_calc.py                     # 按 tile 配置算 smem/寄存器/occupancy 上限
│   ├── tile_sweep.py                    # 扫描 BLOCK_M/N/stages，产出最优配置表
│   ├── dispatch_threshold_sweep.py      # 生成 dispatch_table.inc + 阈值报告
│   ├── compare_with_sdpa.py             # 与 SDPA 的误差 + 性能对比一站式脚本
│   ├── triton_autotune_report.py        # 运行 autotune → 导出配置表与最优组合报告
│   ├── gguf_reader.py                   # 纯 Python GGUF v3 读取器（含 Q8_0 反量化）
│   ├── model_probe.py                   # GGUF/Ollama 元数据 → shapes.json + KV 预算表
│   ├── dump_qwen3_qkv.py                # 逐层反量化取真实 Q/K/V → tests/data/qwen3_4b/
│   ├── ollama_baseline.py               # Ollama HTTP API 基线采集（TTFT / tokens/s / VRAM）
│   ├── gen_qkv.py                       # 固定 seed 生成测试/基准输入
│   └── parse_ncu_csv.py                 # ncu/nvprof CSV 解析
│
├── docs/
│   ├── 00-recon.md                      # Phase 0 侦察报告
│   ├── env_report.md                    # 环境 + ISA 探针结论
│   ├── architecture.md                  # 分层、模块图、文件职责表、依赖方向
│   ├── attention_math.md                # 数学定义、复杂度、mask/scale 约定
│   ├── online_softmax.md                # 推导 + 数值稳定性分析 + 与 torch 对比
│   ├── flash_attention.md               # 算法、tile、流水、与 SDPA 的差异
│   ├── kernel_design.md                 # 每个 kernel 的线程映射/smem/寄存器表
│   ├── tile_config.md                   # tile 参数表来源与标定过程
│   ├── memory_hierarchy.md              # HBM/L2/SMEM/寄存器 层次与实测数字
│   ├── kv_cache.md                      # KV 布局、生命周期、显存预算
│   ├── paged_kv_cache.md                # 分页设计、block table、与 vLLM 的对应
│   ├── prefill_decode.md                # 两阶段差异、阈值标定结论
│   ├── benchmarking.md                  # 方法论、矩阵、复现命令、结果解释
│   ├── profiling.md                     # ncu/nsys 用法、指标含义、瓶颈判定
│   ├── roofline.md                      # AI/带宽/FLOPs 分析（必答 9 问）
│   ├── numerical_stability.md           # 稳定性分析与极端用例
│   ├── triton_optimization.md           # Triton 轨道：kernel 设计、autotune、与 CUDA 的取舍
│   ├── model_integration.md             # Qwen3-4B-Thinking-2507-Q8 对接：元数据/反量化/显存/基线
│   ├── vllm_integration.md              # 集成方式、A/B 数据、上游缺陷清单
│   ├── rollout_interface.md             # RLHF/Rollout 契约与迁移路径
│   ├── upstream_relationship.md         # 三仓关系、版本锚点、patch 清单
│   ├── troubleshooting.md               # 故障树（§附录 D 的正文版）
│   └── results/
│       ├── benchmark_report.md          # 主性能报告（含原始 JSON 路径）
│       ├── profiling_summary.md         # ncu/nsys 摘要与瓶颈判定
│       ├── triton_vs_cuda.md            # Triton / CUDA / SDPA 三方对比（精度 + 性能 + 配置）
│       ├── qwen3_4b_shapes.json         # 真实模型 shape + KV 预算（脚本生成）
│       ├── ollama_baseline.json         # Ollama 服务基线（TTFT / tokens/s / VRAM）
│       ├── vllm_ab.md                   # 与 vLLM 原型的 A/B
│       └── dispatch_thresholds.md       # 阈值标定报告
│
└── upstream/
    ├── notes.md                         # 上游缺陷/差异登记（含文件行号证据）
    └── patches/                         # 需要上游改动时的补丁（默认不应用）
```

文件数量控制原则（KISS/YAGNI）：

1. 允许合并职责相近的小文件（例如 `api_validate.cpp` 可并入 `attention_api.cpp`），但**不允许**把多个 kernel 塞进一个 `.cu`。
2. 不允许为了「以后可能用到」而创建空文件；每个文件在合入时必须已有实现或明确 TODO + Issue 记录。
3. `docs/` 必须与代码同步演进，禁止最后一次性补文档。

---

## 5. 构建与工程文件规格（逐文件内容）

### 5.1 `CMakeLists.txt`（顶层）

必须包含（顺序固定）：

```cmake
cmake_minimum_required(VERSION 3.24)
project(SciComputeAttention VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)          # 与 SciComputeInfra 一致（宿主侧）
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)         # CUDA 翻译单元保持 17
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # 供 pybind11/vLLM 静态链接

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(SciAttentionOptions)        # 选项与默认值（§5.3）
include(SciAttentionArch)           # 架构校验（§5.4）
include(SciComputeInfra)            # 上游定位与引入（§5.2）
include(SciAttentionDeps)           # 测试/基准/绑定依赖（§5.5）

add_subdirectory(src)
add_subdirectory(examples)
if(SCI_ATTENTION_BUILD_TESTS)     enable_testing(); add_subdirectory(tests) endif()
if(SCI_ATTENTION_BUILD_BENCHMARKS) add_subdirectory(benchmarks) endif()
if(SCI_ATTENTION_BUILD_VLLM_ADAPTER) add_subdirectory(vllm_backend) endif()
```

要求：

- 编译警告策略：`-Wall -Wextra -Wpedantic`（GCC/Clang），CUDA 侧额外 `--expt-relaxed-constexpr`。
- 必须打印一张配置摘要（CUDA、架构、上游路径与 HEAD、测试/基准/绑定开关、GC 版本）。
- 安装规则：`install(TARGETS ...)` + `install(DIRECTORY include/ DESTINATION include)`，供未来 `find_package(SciComputeAttention)`。
- `-DSCI_ATTENTION_WERROR=ON` 时把警告升级为错误（CI 用），默认 OFF。

### 5.2 `cmake/SciComputeInfra.cmake`

```cmake
set(SCI_ATTENTION_SCICOMPUTE_INFRA_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../SciComputeInfra"
    CACHE PATH "Path to SciComputeInfra (upstream infra layer)")
set(SCI_ATTENTION_INFRA_MODE "SOURCE" CACHE STRING "SOURCE|PACKAGE|STUB")
```

必须实现：

1. `SOURCE` 模式：校验路径存在（含 `CMakeLists.txt` 与 `include/tensor/tensor.hpp`），把上游 `SCI_BUILD_TESTS`/`SCI_BUILD_BENCHMARKS` 关闭后再 `add_subdirectory(... EXCLUDE_FROM_ALL)`，并校验目标 `sci_tensor`/`sci_core`/`sci_memory` 存在。
2. 读取上游版本锚点：`execute_process(COMMAND git -C ${ROOT} rev-parse --short HEAD ...)`，写入 `SCI_ATTENTION_INFRA_SHA` 并 `message(STATUS ...)`；失败时告警不中断。
3. `PACKAGE` 模式：`find_package(SciComputeInfra QUIET)`，缺失时 `FATAL_ERROR` 并给出 `SOURCE` 模式的示例命令。
4. `STUB` 模式：设置编译定义 `SCI_ATTENTION_INFRA_STUB=1`，仅允许 CPU 接口层与单元测试编译；**必须**在配置阶段打印醒目警告「GPU 验收数据在 STUB 模式下无效」。
5. 任何模式下都不得向 `include/` 复制上游头文件。

### 5.3 `cmake/SciAttentionOptions.cmake`

| Option                             | 默认        | 说明                                             |
| ---------------------------------- | --------- | ---------------------------------------------- |
| `SCI_ATTENTION_BUILD_TESTS`        | ON        | GoogleTest 目标与 CTest 注册                        |
| `SCI_ATTENTION_BUILD_BENCHMARKS`   | ON        | 基准目标（缺 Google Benchmark 时自动降级并打印提示）            |
| `SCI_ATTENTION_BUILD_PYTHON`       | ON        | pybind11 模块                                    |
| `SCI_ATTENTION_BUILD_VLLM_ADAPTER` | OFF       | vLLM(C++) 适配层（需要 `SCI_ATTENTION_VLLM_ROOT`）    |
| `SCI_ATTENTION_VLLM_ROOT`          | `../vllm` | 上游 vLLM(C++) 仓库路径                              |
| `SCI_ATTENTION_ENABLE_TMA`         | AUTO      | `ON/OFF/AUTO`；AUTO 表示由 `arch_features` 运行期探测决定 |
| `SCI_ATTENTION_ENABLE_FP8`         | OFF       | 可选实验路径（仅 probe 通过后允许）                          |
| `SCI_ATTENTION_MAX_SMEM_BYTES`     | 102400    | 覆盖 smem 上限（默认取 device props）                   |
| `SCI_ATTENTION_WERROR`             | OFF       | 警告即错误                                          |

互斥校验：`BUILD_VLLM_ADAPTER=ON` 时 `SCI_ATTENTION_VLLM_ROOT` 必须存在且含 `include/vllm/attention/flash_attention.hpp`，否则 `FATAL_ERROR`。

### 5.4 `cmake/SciAttentionArch.cmake`

1. 未显式指定时，用 `nvcc --list-gpu-arch` + 运行期 `cudaDeviceProp::major/minor` 生成 `SCI_ATTENTION_ARCH`（本机应为 `120`）。
2. 支持 `-DSCI_ATTENTION_ARCH="80;86;89;120"` 覆盖；若列表包含 `<80` 则 `FATAL_ERROR`（本项目使用 `mma.sync` 与 `cp.async`）。
3. 生成 `SCI_ATTENTION_ARCH_LIST` 与编译定义 `SCI_ATTENTION_ARCH_<N>=1`，供 `arch_features.cuh` 使用。
4. 配置阶段打印：GPU 名称、CC、SM 数、smem/SM、L2 大小（来自 Phase 0 的 `env_report.md` 或运行期探测程序）。
5. **禁止**在源码中写死 `sm_120` 或 `-arch=sm_120`。

### 5.5 `cmake/SciAttentionDeps.cmake`

| 依赖               | 查找方式                                                                                                                              | 缺失行为                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| GTest            | `find_package(GTest QUIET)`                                                                                                       | 关闭测试目标并打印提示（不静默）                       |
| Google Benchmark | `find_package(benchmark QUIET)`；支持 `-DCMAKE_PREFIX_PATH=$SCA_BENCHMARK_PREFIX` | 关闭对应目标；本项目自带 `bench_export.hpp` 仍可单独编译 |
| pybind11         | `find_package(pybind11 QUIET)`，失败则 `python3 -m pybind11 --cmakedir`                                                               | 关闭 Python 模块并打印可用安装命令                  |
| Python           | `find_package(Python3 COMPONENTS Interpreter Development.Module)`                                                                 | 同上                                     |
| Nsight           | 不参与构建，仅脚本检测                                                                                                                       | 脚本内 `command -v ncu/nsys` 检查并给出提示      |

### 5.6 工程辅助文件

| 文件              | 必须内容                                                                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.clang-format` | `BasedOnStyle: Google`、`IndentWidth: 4`、`ColumnLimit: 100`、`PointerAlignment: Left`、`SortIncludes: true`；与 vllm 仓风格对齐（若冲突以本文件为准并记录）                      |
| `.gitignore`    | `build*/`、`*.o`、`*.so`、`*.a`、`__pycache__/`、`.venv/`、`*.ncu-rep`、`*.nsys-rep`、`profiling/reports/`、`benchmarks/results/*.json/csv`（保留 `.gitkeep` 与汇总 md） |
| `LICENSE`       | MIT，版权与上游一致口径                                                                                                                                            |
| `TASK.md`       | 目标 / Requirements(R1..Rn) / Constraints / 验收标准(V1..Vn) / 阶段计划（勾选式）                                                                                       |
| `.agent/*.md`   | 四文件模板化维护，禁止删除历史记录                                                                                                                                        |

### 5.7 `scripts/` 与 `tools/` 规格

每个脚本必须：`set -euo pipefail`、支持 `--help`、幂等、失败非零退出、路径使用绝对路径或 `SCRIPT_DIR` 推导。

| 脚本                     | 作用                                        | 关键参数                                                         |
| ---------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| `scripts/configure.sh` | 生成 build 目录（Debug/Release/RelWithDebInfo） | `--build-type`、`--arch`、`--no-tests`、`--with-vllm`           |
| `scripts/build.sh`     | `cmake --build` 并行构建                      | `--target`、`-j`                                              |
| `scripts/test.sh`      | 构建 + `ctest --output-on-failure` + pytest | `--filter`、`--with-python`                                   |
| `scripts/bench.sh`     | 跑基准并落盘 `benchmarks/results/<date>-<sha>/` | `--suite sanity\|main\|long`、`--only <case>`                 |
| `scripts/profile.sh`   | 调 `profiling/profile_*.sh`                | `--kernel flash\|decode\|naive\|vllm`、`--set full\|roofline` |
| `scripts/clean.sh`     | 清理 build 与报告（保留源码与结果）                     | `--all`（含结果）                                                 |

`tools/arch_probe/arch_probe.cu` 必须逐条探测并打印表格：

```text
[probe] mma.sync.m16n8k16.f16.f32 : PASS/FAIL  (compile-time; failure => 本文件不得编译进主库)
[probe] cp.async.cg 16B           : PASS/FAIL
[probe] ldmatrix.x4.b16           : PASS/FAIL
[probe] bulk copy (TMA 1D)        : PASS/FAIL
[probe] bulk tensor 2D (CUtensorMap): PASS/FAIL
[probe] wgmma.fence               : PASS/FAIL（预期 FAIL on sm_120）
[props] sm=36 smem/SM=102400 regs/SM=65536 L2=33554432 threads/SM=1536
```

实现要求：每种能力用独立函数 + `#if __CUDA_ARCH__ >= ...` 守卫；主函数先打印 `device props`，再逐项打印结论；结果同时输出 `--json` 供 `env_report.md` 引用。

---

## 6. 公共 API 头文件规格（签名级，逐文件）

总原则：

1. `include/scicompute_attention/` 是本项目**唯一稳定契约**；上层（vLLM / RLHF / Python）只允许包含这里。
2. 空返回值用 `sci::Status`，有返回值用 `sci::Result<T>`。**禁止假定 `Result<void>` 存在**（上游未声明）；需要空返回时统一用 `Status`。
3. 所有公开 API 不得抛出异常；CUDA 错误、参数错误、容量错误一律转成 `Status`。
4. 头文件必须自包含（单独 include 可编译），不依赖 `.cu` 内部头。

### 6.1 `attention_types.hpp`

```cpp
namespace sca {

// 张量布局：内部计算布局与对外可接受布局
enum class AttnLayout { kBHSD = 0, kBSHD = 1 };   // [B,H,S,D] / [B,S,H,D]
enum class KvLayout  { kBlockMajor = 0 };          // v1 只实现 [num_blocks, block_size, n_kv_heads, head_dim]

struct AttentionShape {
    int64_t batch{0};
    int64_t seq_q{0};
    int64_t seq_kv{0};
    int64_t num_heads{0};      // query 头数 H_q
    int64_t num_kv_heads{0};   // H_kv；GQA/MQA 时 H_q % H_kv == 0
    int64_t head_dim{0};

    int64_t QueryElements() const noexcept;      // batch*seq_q*num_heads*head_dim
    int64_t KvElements()    const noexcept;
    int64_t OutputElements() const noexcept;
    bool    IsGqa()         const noexcept { return num_kv_heads > 0 && num_kv_heads != num_heads; }
    int64_t GroupSize()     const noexcept;      // num_heads / num_kv_heads
};

inline constexpr int64_t kMinSupportedHeadDim = 32;
inline constexpr int64_t kMaxSupportedHeadDim = 256;
// 受支持 head_dim 集合：{32, 64, 96, 128, 160, 192, 256}；其余返回 kUnsupportedHeadDim

}  // namespace sca
```

不变式（必须有静态/运行期检查）：

```text
head_dim  ∈ {32,64,96,128,160,192,256}
num_heads % num_kv_heads == 0
causal == true  ⇒ seq_kv >= seq_q
batch >= 1, seq_q >= 1, seq_kv >= 1
varlen 时 cu_seqlens_* 必须非空且单调递增、长度 = batch + 1
```

### 6.2 `attention_config.hpp`

```cpp
namespace sca {

struct AttentionConfig {
    BackendKind   backend{BackendKind::kAuto};
    AttnLayout    layout{AttnLayout::kBHSD};
    bool          causal{false};
    bool          return_lse{false};
    float         scale{0.0f};        // 0 ⇒ 1/sqrt(head_dim)，必须在文档与日志中明确
    float         softcap{0.0f};      // 0 ⇒ 关闭；>0 时 logits = softcap*tanh(logits/softcap)
    int64_t       sliding_window{0};  // 0 ⇒ 关闭（v1 可仅实现 causal 路径，其余返回 kNotImplemented）
    int64_t       num_splits{0};      // decode split-K；0 ⇒ 由 decode_config.hpp 推导
    size_t        workspace_limit_bytes{0};  // 0 ⇒ 使用 DeviceCapability 推导上限
    bool          allow_fallback{false};     // 默认 false：禁止静默降级
    const int32_t* cu_seqlens_q{nullptr};    // varlen 入口
    const int32_t* cu_seqlens_kv{nullptr};
    sci::Stream*  stream{nullptr};           // nullptr ⇒ sci::Stream::GetCurrent()
};

// 唯一校验入口：所有公共 API 必须先调用
sci::Status Validate(const AttentionConfig& cfg, const AttentionShape& shape, std::string* detail = nullptr);

// 推荐 tile（供 explain() 与 benchmark 对照显示）
TileConfig RecommendTile(const AttentionConfig& cfg, const AttentionShape& shape);

}  // namespace sca
```

`Validate` 必须逐字段检查并返回**带字段名的错误消息**，例如：

```text
AttentionConfig invalid: head_dim=48 (supported: 32,64,96,128,160,192,256)
AttentionConfig invalid: causal=true but seq_kv(128) < seq_q(256)
AttentionConfig invalid: allow_fallback=false 但 backend=kAuto 且无可用后端（shape/B/H/S/D 组合）
```

### 6.3 `attention_result.hpp`

```cpp
namespace sca {

struct AttentionRuntimeStats {
    BackendKind  used_backend{BackendKind::kAuto};
    TileConfig   tile{};                  // 实际使用的 tile
    size_t       workspace_bytes{0};      // 实际占用
    size_t       kv_bytes{0};             // 若使用 KV Cache
    std::string  note;                    // 例如 "fallback: fp8 unsupported"
};

struct AttentionResult {
    sci::Tensor out;        // 与输入同 dtype/布局；生命周期由调用者持有
    sci::Tensor lse;        // 可选：FP32，形状 [B, H_q, S_q]；return_lse=false 时为空张量
    AttentionRuntimeStats stats;
};

}  // namespace sca
```

约定：

- `out` 必须是**新分配**的连续张量（内部计算布局），不得复用输入缓冲。
- `lse` 定义为 `log(sum_j exp(s_ij - m_i)) + m_i`，取值单位与 logits 一致；写入 `docs/attention_math.md` 并给出与 PyTorch 对照的用例。
- `stats.note` 为空字符串表示「按请求执行、无降级」，任何降级都必须非空。

### 6.4 `backends.hpp`

```cpp
namespace sca {

struct CapabilityReport {
    bool         supported{false};
    std::string  reason;               // 不支持时的可读原因（必须是具体数值/条件）
    BackendKind  recommended_backend{BackendKind::kAuto};
    TileConfig   tile{};
    size_t       workspace_bytes{0};
};

class IAttentionBackend {
public:
    virtual ~IAttentionBackend() = default;
    virtual const char* Name() const noexcept = 0;
    virtual CapabilityReport Supports(const AttentionConfig&, const AttentionShape&) const = 0;
    virtual size_t WorkspaceBytes(const AttentionConfig&, const AttentionShape&) const = 0;
    virtual sci::Result<AttentionResult> Forward(const sci::Tensor& q, const sci::Tensor& k,
                                                 const sci::Tensor& v,
                                                 const AttentionConfig& cfg) = 0;
    virtual sci::Result<AttentionResult> ForwardVarlen(
        const sci::Tensor& q, const sci::Tensor& k, const sci::Tensor& v,
        const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_kv,
        int64_t max_seq_q, int64_t max_seq_kv, const AttentionConfig& cfg) = 0;
};

// 后端枚举与查表（实现在 src/runtime/dispatcher.cpp）
enum class BackendKind { kAuto = 0, kNaive, kTiled, kFlash, kDecode, kPaged };
const IAttentionBackend* GetBackend(BackendKind kind) noexcept;
const char* BackendName(BackendKind kind) noexcept;

}  // namespace sca
```

要求：

- `Supports()` 必须回答「为什么」，禁止只返回 `false`。示例：`reason = "head_dim=256 with BLOCK_N=64 requires 104857 B smem > 102400 B limit"`。
- `ForwardVarlen` 允许在早期 Phase 返回 `kNotImplemented`，但必须显式返回错误，不允许忽略 varlen 参数。
- 每个后端一个翻译单元，禁止在头文件里 `#include` CUDA 实现细节（保持 API 层可被 CPU 编译器解析，STUB 模式下也能编译）。

### 6.5 四个后端入口头文件

每个文件提供「自由函数（主入口）」+「backend 类（供 dispatcher 使用）」两套，函数签名必须一致：

```cpp
// naive_attention.hpp / tiled_attention.hpp / flash_attention.hpp / decode_attention.hpp
sci::Result<AttentionResult> naive_attention(const sci::Tensor& q, const sci::Tensor& k,
                                             const sci::Tensor& v, const AttentionConfig& cfg);
sci::Result<AttentionResult> tiled_attention(...);   // 同上
sci::Result<AttentionResult> flash_attention(...);   // 同上
sci::Result<AttentionResult> decode_attention(...);  // 同上（要求 seq_q 小，见 §9.3）

// 变长入口（varlen）
sci::Result<AttentionResult> flash_attention_varlen(
    const sci::Tensor& q, const sci::Tensor& k, const sci::Tensor& v,
    const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_kv,
    int64_t max_seq_q, int64_t max_seq_kv, const AttentionConfig& cfg);
```

`decode_attention.hpp` 额外暴露：

```cpp
struct DecodeConfig {
    int64_t num_splits{0};        // 0 ⇒ 自动（见 decode_config.hpp）
    int64_t max_splits{16};       // 硬上限，防止 workspace 爆炸
    bool    use_splitk{true};     // false ⇒ 单块归约（对照用）
};
DecodeConfig RecommendDecodeConfig(const AttentionShape& shape, const DeviceCapability& cap);
```

### 6.6 `kv_cache.hpp`

```cpp
namespace sca {

struct KVCacheConfig {
    int64_t   num_layers{0};
    int64_t   num_kv_heads{0};
    int64_t   head_dim{0};
    int64_t   block_size{16};      // 每 page token 数（vLLM 默认 16）
    int64_t   num_blocks{0};       // 物理 block 总数（由显存预算推导，见 §8.4）
    int64_t   max_num_seqs{1};
    sci::DType dtype{sci::DType::kFloat16};
    KvLayout  layout{KvLayout::kBlockMajor};
    bool      enable_stats{true};
};

struct KVCacheStats {
    int64_t num_blocks{0}, used_blocks{0}, free_blocks{0};
    int64_t num_appends{0}, num_resets{0};
    size_t  peak_bytes{0}, bytes{0};
};

class KVCache {
public:
    static sci::Result<KVCache> Create(const KVCacheConfig& cfg, sci::Device& device);

    // 写入：slot_mapping[t] = 该 token 在物理块内的槽位索引（block_id*block_size + offset）
    sci::Status Append(int64_t layer, const sci::Tensor& k_new, const sci::Tensor& v_new,
                       const int32_t* slot_mapping, int64_t num_tokens, sci::Stream* stream);

    // 调试/测试用 gather：把逻辑块序列拼成连续张量（不进入推理热路径）
    sci::Result<sci::Tensor> GatherK(int64_t layer, const int32_t* block_ids, int64_t num_blocks,
                                     sci::Stream* stream) const;
    sci::Result<sci::Tensor> GatherV(...) const;

    sci::Status Reset(const int32_t* block_ids, int64_t num_blocks, sci::Stream* stream);
    const sci::Tensor& K(int64_t layer) const;   // [num_blocks, block_size, n_kv_heads, head_dim]
    const sci::Tensor& V(int64_t layer) const;
    KVCacheStats Stats() const;
    size_t MemoryBytes() const noexcept;
};

// slot_mapping 计算（唯一实现，C++/Python/vLLM 适配层共用）
sci::Result<std::vector<int32_t>> ComputeSlotMapping(
    const int32_t* block_table_row, int64_t seq_start, int64_t num_tokens,
    int64_t block_size, int64_t max_blocks);

}  // namespace sca
```

### 6.7 `block_manager.hpp`

```cpp
namespace sca {

struct BlockTable {
    std::vector<int32_t> host;     // [max_num_seqs, max_blocks_per_seq]，-1 表示空槽
    sci::Tensor device;            // 设备镜像，dtype = kInt32
    uint64_t revision{0};          // 每次修改 +1；上卡时对比，避免重复拷贝
};

class BlockManager {
public:
    static sci::Result<BlockManager> Create(int64_t num_blocks, int64_t max_num_seqs,
                                           int64_t max_blocks_per_seq);
    sci::Result<std::vector<int32_t>> Allocate(int64_t num_blocks);   // 不足 ⇒ kKVCapacityExceeded
    void Free(const int32_t* block_ids, int64_t count);
    int64_t NumFreeBlocks() const noexcept;
    int64_t NumTotalBlocks() const noexcept;
    sci::Status AppendToTable(int64_t seq_id, const int32_t* block_ids, int64_t count);
    sci::Status ResetSeq(int64_t seq_id);
    const BlockTable& Table() const noexcept;
    sci::Status SyncTableToDevice(sci::Stream* stream);   // 按 revision 增量同步
};

}  // namespace sca
```

设计约束：

- v1 采用**独占式分配**（block 不共享、不做 COW、不做 prefix caching）；`fork/ref_cnt` 仅在文档 Roadmap 中列出，禁止提前实现（YAGNI）。
- `Allocate/Free` 必须是 O(1) 均摊（free list 用栈式 `std::vector<int32_t>`）。
- 必须包含「容量耗尽」用例：分配超过 `num_blocks` 时返回 `kKVCapacityExceeded` 且状态不被破坏。

### 6.8 `paged_kv_cache.hpp`

```cpp
namespace sca {

class PagedKVCache {
public:
    static sci::Result<PagedKVCache> Create(const KVCacheConfig& cfg, sci::Device& device);

    // 一个请求追加 tokens：内部完成 block 分配 + slot_mapping 生成 + 写入
    sci::Status AppendTokens(int64_t seq_id, int64_t layer, const sci::Tensor& k_new,
                             const sci::Tensor& v_new, sci::Stream* stream);

    // 释放序列占用（LRU/回收策略由调用方决定）
    sci::Status ResetSeq(int64_t seq_id, sci::Stream* stream);

    // 供 paged attention kernel 使用
    const int32_t* PageTableDevicePtr() const noexcept;  // [max_num_seqs, max_blocks_per_seq]
    int64_t LogicalBlockCount(int64_t seq_id) const noexcept;
    int64_t BlockSize() const noexcept;
    const KVCache& Storage() const noexcept;             // 底层 K/V 存储
    sci::Status SyncPageTable(sci::Stream* stream);
    KVCacheStats Stats() const;
};

}  // namespace sca
```

### 6.9 `paged_attention.hpp`

```cpp
namespace sca {

struct PagedAttentionParams {
    const int32_t* page_table{nullptr};   // [num_seqs, max_blocks_per_seq]
    const int32_t* seq_lens{nullptr};     // [num_seqs]（已含当前 token）
    int64_t num_seqs{0};
    int64_t block_size{16};
    int64_t max_blocks_per_seq{0};
};

sci::Result<AttentionResult> paged_attention(
    const sci::Tensor& q,                 // [num_seqs, H_q, seq_q, D]（decode 时 seq_q=1）
    const PagedKVCache& kv, const PagedAttentionParams& params, const AttentionConfig& cfg);

}  // namespace sca
```

### 6.10 `dispatcher.hpp`

```cpp
namespace sca {

struct DispatchDecision {
    BackendKind  backend{BackendKind::kAuto};
    TileConfig   tile{};
    size_t       workspace_bytes{0};
    std::string  reason;      // 机器可读前缀 + 人类可读解释，例如 "auto: seq_q=1 ⇒ decode"
};

class AttentionDispatcher {
public:
    explicit AttentionDispatcher(const AttentionConfig& cfg);   // cfg 作为默认配置
    DispatchDecision Select(const AttentionConfig&, const AttentionShape&) const;
    sci::Result<AttentionResult> Forward(const sci::Tensor& q, const sci::Tensor& k,
                                         const sci::Tensor& v, const AttentionConfig& cfg) const;
    std::string Explain(const AttentionConfig&, const AttentionShape&) const;
    static sci::Status ReloadDispatchTable(const char* path = nullptr);  // 默认读 dispatch_table.inc 编译期内嵌值
};

}  // namespace sca
```

### 6.11 `capability.hpp` / `workspace.hpp` / `status.hpp` / `version.hpp` / `export.hpp`

```cpp
// capability.hpp
struct DeviceCapability {
    std::string device_name;
    int  major{0}, minor{0};
    int  sm_count{0};
    size_t smem_per_sm{0}, smem_per_block_default{0}, smem_per_block_optin{0};
    size_t regs_per_sm{0}, max_threads_per_sm{0}, l2_bytes{0}, total_mem_bytes{0};
    bool has_mma_m16n8k16{false}, has_cp_async{false}, has_tma_bulk{false},
         has_wgmma{false}, has_fp8_mma{false};
    static const DeviceCapability& ForDevice(int device_id = 0);   // 首次探测后缓存
    static sci::Status Refresh(int device_id = 0);
};

// workspace.hpp
class Workspace {
public:
    static sci::Result<Workspace> Create(size_t bytes, sci::Device& device);
    void*  Data() noexcept; size_t Bytes() const noexcept;
    sci::Result<void*> Slice(size_t offset, size_t bytes, size_t align = 256);
    sci::Status Reset();                     // 逻辑重置（不释放显存）
    size_t PeakBytes() const noexcept;
};

// status.hpp：在 sci::StatusCode 语义上扩展本域错误码（禁止改上游枚举）
enum class AttnStatusCode : int32_t {
    kOk = 0,
    kUnsupportedHeadDim   = 2001,
    kUnsupportedDtype     = 2002,
    kUnsupportedLayout    = 2003,
    kLayoutMismatch       = 2004,
    kShapeMismatch        = 2005,
    kSeqTooLong           = 2006,
    kKVCapacityExceeded   = 2007,
    kWorkspaceExceeded    = 2008,
    kUnsupportedFeature   = 2009,   // 例如 sliding_window / softcap 未实现
    kDeviceCapability     = 2010,   // 例如 smem 超限
};
const char* ToString(AttnStatusCode code) noexcept;
sci::Status MakeStatus(AttnStatusCode code, std::string detail);

// version.hpp
#define SCI_ATTENTION_VERSION_MAJOR 0
#define SCI_ATTENTION_VERSION_MINOR 1
#define SCI_ATTENTION_VERSION_PATCH 0
#define SCI_ATTENTION_VERSION_STRING "0.1.0"

// export.hpp
#define SCI_ATTENTION_API __attribute__((visibility("default")))
#define SCI_ATTENTION_INTERNAL
```

### 6.12 `attention.hpp`（umbrella）

- 只做聚合：`#include` 上述公共头 + 一段注释说明「上层只允许 include 本文件或具体子头」。
- 提供 4 个最高层便捷入口，内部走 dispatcher：

```cpp
namespace sca {
sci::Result<AttentionResult> attention(const sci::Tensor& q, const sci::Tensor& k,
                                       const sci::Tensor& v, const AttentionConfig& cfg = {});
inline sci::Result<AttentionResult> flash(const sci::Tensor& q, const sci::Tensor& k,
                                          const sci::Tensor& v, const AttentionConfig& cfg = {}) {
    AttentionConfig c = cfg; c.backend = BackendKind::kFlash; return attention(q, k, v, c);
}
sci::Result<AttentionResult> decode(const sci::Tensor& q, const sci::Tensor& k,
                                    const sci::Tensor& v, const AttentionConfig& cfg = {});
sci::Result<AttentionResult> paged(const sci::Tensor& q, const PagedKVCache& kv,
                                   const PagedAttentionParams& p, const AttentionConfig& cfg = {});
}  // namespace sca
```

### 6.13 `detail/` 内部头

| 文件                  | 必须内容                                                                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dtype_traits.hpp`  | `DTypeTraits<DType>`：C++ 类型、CUDA 类型（`__half`/`__nv_bfloat16`）、MMA 类型、字节数、是否支持 MMA 的 `constexpr` 映射                                                            |
| `tile_config.hpp`   | `struct TileConfig { int block_m, block_n, warps, stages, use_tma; int smem_bytes() const; int regs_budget() const; std::string ToString() const; };` + 合理性校验 |
| `layout_traits.hpp` | `OffsetBHSD/OffsetBSHD`、`StridesFor(layout, shape)`、`IsSupportedLayout()`；**所有布局换算只允许在这里实现一次**（DRY）                                                           |
| `host_utils.hpp`    | `CeilDiv/Lcm/AlignUp/IsAligned/CheckedMul`（溢出检测，返回 `Result`）                                                                                                  |

---

## 7. CUDA Kernel 规格（逐文件）

### 7.0 统一约定（所有 `.cu/.cuh` 必须遵守）

| 约定         | 内容                                                                                                                                 |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 命名空间       | `sca::cuda`（kernel 与 launch），`sca`（host 侧后端实现）                                                                                     |
| 文件命名       | `*_kernel.cuh`（设备函数）、`*_launch.cu`（实例化+启动）、`*_backend.cpp`（接口适配）                                                                   |
| 模板形态       | `template <typename T, int kHeadDim, int kBlockM, int kBlockN, int kStages> __global__ void ...`；head_dim 必须编译期常量                  |
| launch ABI | `template <...> sci::Status LaunchFlashFwd(const FlashFwdParams& p, sci::Stream* s);`，所有参数装在 `struct FlashFwdParams` 里（禁止 20 个裸参数） |
| 错误检查       | kernel 启动后必须 `SCI_CUDA_CHECK_LAST()`；release 路径禁止 `cudaDeviceSynchronize()`                                                        |
| 线程数        | 每 block = `warps * 32`，禁止使用非 warp 倍数                                                                                               |
| 动态 smem    | 只能通过 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` 提升；必须检查返回值                                    |
| 越界保护       | 所有全局读写必须做边界判定（`row < seq` / `col < seq`），禁止依赖「测试用例恰好整除」                                                                            |
| 数值         | softmax 内部统一用 `exp2f`（`scale` 折算 `log2(e)`）；累加器用 FP32                                                                              |
| 注释         | 英文注释；数学处用公式注释；每个 kernel 头部必须注释：线程映射、tile 映射、smem 分区、寄存器预算                                                                          |
| 禁止         | 禁止 `printf` 出现在 release 路径、禁止 `assert` 代替错误码、禁止 device 侧动态分配、禁止把多个 kernel 写进一个文件                                                   |
| 禁用指令       | 禁止 `wgmma.*`、禁止 `tcgen05.*`（sm_120 不支持，实测见 §1.2）                                                                                   |

### 7.1 `src/cuda_common/*`（共享工具，逐文件）

| 文件                  | 必须内容                                                                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cuda_check.cuh`    | `SCI_CUDA_CHECK(expr)` / `SCI_CUDA_CHECK_LAST()`（带文件行号）；`SCI_CUDA_SYNC_DEBUG(stream)` 仅在 `NDEBUG` 未定义时同步；返回 `sci::Status`（不抛异常）                                             |
| `arch_features.cuh` | 编译期：`SCI_ATTENTION_ARCH_SM == 120` 之类宏；运行期：`bool HasTmaBulk()`（查询 `DeviceCapability`）；`constexpr bool kHasWgmma = false;` 并注释「sm_120 ptxas 拒绝 wgmma，实测见 docs/env_report.md」 |
| `vec_memory.cuh`    | `float4/half8` 级别 load/store 封装；`LoadVec<T,N>`/`StoreVec<T,N>`；对齐断言（`reinterpret_cast` 前校验 16B 对齐）；边界安全的重载                                                                  |
| `numerics.cuh`      | `inline __device__ float Log2e();`、`FastExp2(float)`（默认用 `exp2f`，可选 `__expf` 由宏 `SCI_ATTENTION_FAST_MATH` 控制）、`RowMaxStable`、`RowSumStable`、NaN/Inf 哨兵常量与检测辅助               |
| `mma_policy.cuh`    | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 与 `...bf16.bf16.f32` 封装（`MmaF16<TS, TD>`）；fragment 布局注释（A: 每线程 4×b32，B: 2×b32，C: 4×f32）；**必须注释说明为何不用 wgmma/tcgen05**    |
| `ldmatrix.cuh`      | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 封装；smem 地址计算（`__cvta_generic_to_shared`）；bank conflict 说明与 padding 约定                                                            |
| `cp_async.cuh`      | 多级流水：`CpAsyncStage<kStages>` + `Commit/WaitGroup<k>`；`Prologue()/Steady()/Epilogue()` 三段式；`cp.async.cg` 16B 优先，尾块用 `cp.async.ca` 或逐元素兜底                                     |

### 7.2 `src/backends/naive/*`（Level 0，正确性参考）

`attention_naive.cuh` 声明 4 个 kernel（可分开 launch，也可融合 mask+scale）：

```text
qk_gemm_kernel     : S[B,H,S_q,S_kv] = Q @ Kᵀ                  （允许用 cublas 或自写 tiled GEMM，但必须显式说明选择）
mask_scale_kernel  : S = S * scale (+ causal mask: j>i ⇒ -inf)  （与非 causal 路径共用）
row_softmax_kernel : P = softmax(S, axis=-1)（数值稳定版，减行最大）
pv_gemm_kernel     : O[B,H,S_q,D] = P @ V
```

`attention_naive.cu` 要求：

1. 显式物化 `S`（`[B,H,S_q,S_kv]`），并在注释中写出该缓冲的字节数公式 `B*H*S_q*S_kv*sizeof(T)`——这正是 FlashAttention 要消除的对象。
2. 支持 `FP32/FP16/BF16`，其中 `FP32` 是数值参考路径；`S` 与 `P` 在 FP16 路径下仍需 FP32 存储（说明理由：中间精度会破坏对照意义）。
3. 提供 `bool use_cublas` 开关（默认 false，自写实现为主；cublas 仅作 GEMM 对照，需在文档中区分）。
4. 显存检查：`S` 缓冲超出 `workspace_limit_bytes` 时返回 `kWorkspaceExceeded`，并给出建议（改用 tiled/flash）。

验收：与 PyTorch FP32 参考的最大绝对误差（见 §11.4 容差表）；同时给出「S 缓冲峰值显存」数字，作为后续 IO 分析的基线。

### 7.3 `src/backends/tiled/*`（Level 1，真正的 CUDA tiling）

目标：把 Q/K/V 分块搬入 smem，用寄存器累加，**不物化 N×N**，但保留「在 smem 中做整行 softmax」的中间形态（`S_tile` 存 smem）。

`attention_tiled.cuh` 必须包含：

```text
thread block decomposition : 每 block 负责一个 (b, h, q_block)
warp mapping               : warp w 负责 Q tile 中连续 16 行（m16n8k16 的 m 维度），
                             warp 内 32 线程按 n 维切分
shared-memory layout       : Q_smem[BM][D] / K_smem[BN][D] / V_smem[BN][D] / S_smem[BM][BN]
                             每块均按 8 元素（16B）对齐；S_smem 行步长加 padding 避免 bank conflict
register accumulation      : O_acc[BM/线程行数][D] 存寄存器；S_tile 逐块写回 smem 后做行 softmax
```

限制与判定：当 `D > 128` 或 `smem 需求 > 100 KB` 时，`tiled` 必须返回 `kDeviceCapability` 并建议 flash 后端（禁止硬撑）。

验收：`docs/kernel_design.md` 中给出 tiled 的 smem 占用表与实测 `sm__throughput`/`dram__throughput` 对比（证明 tiling 相比 naive 减少了哪些流量）。

### 7.4 `src/backends/flash/*`（Level 3，核心）

#### 7.4.1 `flash_fwd_kernel.cuh` 主循环（伪代码，必须逐行体现在实现里）

```text
// Grid: (ceil(S_q / BLOCK_M), H_q, B)      Block: (WARPS * 32)
// Per block: one (b, h_q, q_block)
1  load Q tile [BLOCK_M, D] → smem → (ldmatrix) → 寄存器 A_frag
2  m_i = -inf (BLOCK_M), l_i = 0 (BLOCK_M), O_acc = 0 [BLOCK_M, D]
3  h_kv = h_q / group_size                        // GQA/MQA 映射
4  for j0 in 0 .. ceil(S_kv / BLOCK_N):
5      if causal and (j0*BLOCK_N > i0*BLOCK_M + BLOCK_M - 1) break      // 整块被掩
6      cp.async K tile [BLOCK_N, D] → smem (stage s)
7      S = (A_frag @ B_frag) * scale              // mma.sync m16n8k16, FP32 累加
8      if causal: 对角线块内逐元素置 -inf (j > i)；非对角块整块保留
9      m_new = max(m_i, rowmax(S))
10     alpha = exp2((m_i - m_new) * log2e)
11     P     = exp2((S - m_new) * log2e)          // 行方向广播
12     l_i   = alpha * l_i + rowsum(P)
13     O_acc = alpha * O_acc                      // 重标定历史输出
14     cp.async V tile [BLOCK_N, D] → smem
15     O_acc += P_frag @ V_frag                   // mma.sync，FP32 累加
16     m_i = m_new
17 end for
18 O = O_acc / l_i (用 rcp 近似或精确除法，需在文档声明)
19 store O tile [BLOCK_M, D]（可选：同时存 LSE = m_i + log(l_i)）
```

必须写进注释的数学（公式注释用 `// math:` 前缀）：

```text
// math: m_i = max_j S_ij
// math: l_i = sum_j exp(S_ij - m_i)
// math: O_i = sum_j exp(S_ij - m_i) V_j / l_i
// math: 分块更新（online softmax）
// math:   m_new = max(m_old, m_block)
// math:   alpha = exp(m_old - m_new),  beta = exp(m_block - m_new)
// math:   l_new = alpha * l_old + beta * l_block
// math:   O_new = (alpha * l_old * O_old + beta * O_block) / l_new
// math: 实现中把 beta 吸收进 P（P = exp(S - m_new)），等价且少一次乘法
// math: exp2f 版本：exp(x) = exp2(x * log2e)，均匀缩放可提出常数以减少指令
```

#### 7.4.2 `flash_online_softmax.cuh`

```cpp
// 结构：一次 K/V block 迭代内的状态更新，全部在寄存器完成
template <int kBlockM, int kBlockN>
struct OnlineSoftmaxState {
    float m_i[kBlockM / WARPS_PER_TILE];   // row max（每线程持有的行数 = 1）
    float l_i[...];
    __device__ void Update(const float* s_tile_row, float* o_acc_row, int d, ...);
};
```

要求：

1. 行内归约必须用 `__shfl_xor_sync` 实现 warp 内 4 线程（m16n8 的 N 维）归约 + `__reduce_add_sync` 或 butterfly；禁止用 `atomicAdd`。
2. `alpha` 与 `m_new` 的更新必须与 §7.4.1 的公式完全一致，并在单测 `test_numerics.cu` 中用「手工两段式参考」交叉验证。
3. 必须支持 `m_i` 初值 `-inf`（第一块 `alpha = exp(-inf - m_new) = 0`），并在注释中说明该边界为何安全。

#### 7.4.3 `flash_smem_layout.cuh`

```cpp
template <typename T, int kHeadDim, int kBlockM, int kBlockN, int kStages>
struct FlashSmemLayout {
    // 分区（每段按 16B 对齐，且用 padding 避免 bank conflict）
    T q[kBlockM * kHeadDim];                 // Q tile（可从 smem 直接由 ldmatrix 读取）
    T k[kStages][kBlockN * (kHeadDim + kPadK)];  // 多级流水 K
    T v[kStages][kBlockN * (kHeadDim + kPadV)];  // 多级流水 V
    float lse[kBlockM];                      // 可选 LSE 暂存
    static constexpr int kBytes = ...;       // 必须提供编译期字节数
    static_assert(kBytes <= 102400, "smem exceeds device limit");   // 上限来自 env_report
};
```

要求：

- 每个 tile 配置都必须能打印 `kBytes`，并由 `tools/smem_calc.py` 交叉验证（脚本与 `static_assert` 数值必须一致，测试用例 `test_tile_config.cpp` 断言两处一致）。
- S/P 矩阵默认**不进 smem**（FA2 风格，寄存器内完成 P 的打包与 MMA）；若某配置寄存器溢出，必须在文档中记录该配置被禁用的原因与实测寄存器数。

#### 7.4.4 `flash_causal_mask.cuh`

```text
if (j0 * BLOCK_N + BLOCK_N - 1 <= i0 * BLOCK_M)          → 整块无需掩码
else if (j0 * BLOCK_N > i0 * BLOCK_M + BLOCK_M - 1)      → 整块跳过（break）
else                                                     → 对角线块：逐元素 j > i 置 -inf
```

要求：对角线块之外零分支；文档给出「causal 路径理论 FLOPs ≈ 非 causal 的一半」的推导与实测对比。

#### 7.4.5 `flash_tile_config.hpp`（表驱动，禁止魔法数）

```cpp
struct TileConfig { int block_m, block_n, warps, stages; bool use_tma; };
// 启动时的初始推荐表（必须由 tile_sweep.py 实测校准后写回本表，校准记录进 docs/tile_config.md）
inline constexpr TileConfig kFlashTileTable[][...] = {
  /* head_dim=64  */ { {128, 64, 4, 2, false}, {128,128, 4, 2, false}, { 64, 64, 4, 3, false} },
  /* head_dim=128 */ { { 64, 64, 4, 2, false}, {128, 64, 8, 2, false}, { 64,128, 4, 2, false} },
  /* head_dim=256 */ { { 64, 32, 8, 2, false}, { 64, 64, 8, 1, false}, { 32, 64, 4, 2, false} },
};
```

选择逻辑：`SelectTile(dtype, head_dim, dprops, smem_limit, register_limit, seq_kv, causal)`，返回 `Result<TileConfig>`；无可用配置时必须返回错误与原因，不允许「回退到任意配置」。

### 7.5 `src/backends/decode/*`（Level 4）

与 prefill 的本质差异（必须写进 `docs/prefill_decode.md`）：

```text
prefill : S_q 大、S_kv 大 → 计算强度高，适合 tile + MMA，瓶颈常在 Tensor Core / smem 带宽
decode  : S_q = 1（或很小）、S_kv 大 → 算术强度极低，瓶颈在 HBM 读 K/V；需要 split-K + 高并发
```

`decode_splitk.cuh` 要求：

```text
grid.z = num_splits；每个 split 处理一段 K/V（约 ceil(S_kv / num_splits) token）
输出：partial_m[num_splits], partial_l[num_splits], partial_o[num_splits][H_q][D]（写 workspace，FP32）
每个 CTA 内：warp 内规约 + smem 规约；K/V 读取用 float4/half8 向量化 + __ldg
```

`decode_reduce.cuh` 要求：

```text
对每个 (b, h, d)：m_all = max_s m_s；l_all = Σ_s l_s * exp(m_s - m_all)
O = Σ_s (O_s * l_s * exp(m_s - m_all)) / l_all
必须给出该合并公式的推导注释；单测覆盖 num_splits ∈ {1,2,3,4,8,16} 的正确性
```

`decode_config.hpp`：`num_splits = clamp(ceil(S_kv / kTokensPerSplit), 1, max_splits)`，其中 `kTokensPerSplit` 由实测标定（初值 512 或 `S_kv / (sm_count * 2)`），并在 `docs/prefill_decode.md` 给出标定曲线。

### 7.6 `src/backends/paged/*`（Level 5）

`page_table.cuh`：

```text
logical_block = token_index / block_size
physical      = __ldg(&page_table[seq_id * max_blocks_per_seq + logical_block])
if (physical < 0) → 该 token 视为不存在（返回 0 权重，而不是读越界）
offset        = physical * block_size * n_kv_heads * head_dim + (token_index % block_size) * ...
```

`paged_attention.cu`：在 decode kernel 基础上把 K/V 寻址换成 page table 间接寻址；要求：

1. 单测覆盖：不连续 page（人为打乱 block 顺序）、block_size ∈ {1, 8, 16, 32}、seq 长度非 block_size 整数倍。
2. 必须论证 page table 访存开销（每 token 一次额外 4B 读），在 `docs/paged_kv_cache.md` 给出实测影响（<2% 为目标，超出需给结论）。

### 7.7 Tile / smem / 寄存器预算表（起始推荐，必须实测校准）

计算口径：

```text
smem_bytes ≈ BLOCK_M*D*2 (Q) + 2*STAGES*BLOCK_N*D*2 (K,V 各 stages 份) + padding
O 累加器寄存器/线程 = BLOCK_M * D / (WARPS * 32)          // FP32 累加
每 SM 上限：smem 102400 B、寄存器 65536、线程 1536（实测，见 §1.1）
```

| head_dim | dtype                  | BLOCK_M | BLOCK_N | warps | stages | 估算 smem | O 寄存器/线程 | 备注                                  |
| -------- | ---------------------- | ------- | ------- | ----- | ------ | ------- | -------- | ----------------------------------- |
| 64       | fp16/bf16              | 128     | 64      | 4     | 2      | ≈ 49 KB | 64       | 默认配置，1–2 CTA/SM                     |
| 64       | fp16/bf16              | 128     | 128     | 4     | 2      | ≈ 82 KB | 64       | 大 BLOCK_N 提升 K 复用                   |
| 64       | fp16/bf16              | 64      | 64      | 4     | 3      | ≈ 41 KB | 32       | 长序列 / 显存紧张                          |
| 128      | fp16/bf16              | 128     | 64      | 8     | 2      | ≈ 98 KB | 64       | 临界，需 `cudaFuncSetAttribute` 提升 smem |
| 128      | fp16/bf16              | 64      | 64      | 4     | 2      | ≈ 49 KB | 64       | 默认配置                                |
| 128      | fp16/bf16              | 64      | 128     | 4     | 2      | ≈ 82 KB | 64       | 高 K 复用                              |
| 256      | fp16/bf16              | 64      | 32      | 8     | 2      | ≈ 65 KB | 64       | 默认配置（长序列）                           |
| 256      | fp16/bf16              | 64      | 16      | 8     | 2      | ≈ 49 KB | 64       | smem 紧张时                            |
| 256      | fp32（仅 naive/tiled 对照） | 64      | 32      | 8     | 1      | ≈ 98 KB | 128      | 仅用于正确性对照                            |

**强制要求**：

1. 上表是起点，不是结论。必须用 `tools/tile_sweep.py` 扫描并写入 `docs/tile_config.md`（表格含：配置、smem 实测、registers/thread、achieved occupancy、P50 延迟、TFLOPS）。
2. 每个配置必须通过 `tools/smem_calc.py` 与 kernel 内 `static_assert` 双重校验。
3. 任何「为统一而牺牲性能」的做法（例如所有 head_dim 用同一 BLOCK_M/N）禁止出现。

### 7.8 每个 kernel 的代码质量门禁

```text
[ ] 文件头注释包含：目的、线程映射、tile 映射、smem 分区、寄存器预算、参考公式
[ ] 无 -Wall -Wextra -Wpedantic 警告（WERROR=ON 时零警告）
[ ] compute-sanitizer memcheck/racecheck 零错误（§11.5）
[ ] 边界用例：S=1 / S=非 tile 整数倍 / causal 对角线 / D=非 16 倍数（96/160/192）全部通过
[ ] 无未使用参数、无复制粘贴残留、无魔法数（tile 参数只能来自 TileConfig）
```

### 7.9 Triton 算子实现规格（Python 侧第二实现轨道）

#### 7.9.1 定位与边界（必须写进 `docs/triton_optimization.md`）

```text
定位：Triton 是本项目的「算子优化第二轨道」，用于
  1) 快速迭代 kernel 设计（在线 softmax / mask / GQA 的算法验证）
  2) 自动调优：BLOCK_M/BLOCK_N/num_warps/num_stages 的组合搜索
  3) 交叉验证：与 CUDA kernel、PyTorch 参考构成三方一致性检验（算法错误 vs 实现错误可分离）
  4) 面向 Python 生态的研究/教学/rollout 实验入口

边界（硬约束）：
  - Triton kernel 只存在于 Python 侧（`python/scicompute_attention/triton_kernels/`）
  - C++ 公共 API、vLLM(C++) 适配层、RLHF C++ 侧仍走 CUDA kernel；不得声称 Triton 覆盖 C++ 路径
  - 不允许以 Triton 替代 CUDA 主实现（CUDA kernel 是必须交付项，见 §21）
  - 两套实现必须共享同一套数学约定（scale/causal/mask/LSE 定义），差异只允许出现在实现层
```

#### 7.9.2 环境与约束（本机实测，必须复核）

| 项           | 实测值/约束                                                                                                    |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| Triton 版本   | 3.7.1（conda env `cuda_132`，Python 3.12.13，torch 2.13.0+cu132）                                             |
| ninja       | 可用（Triton JIT 编译需要；缺失时 Triton 会退化为慢速编译路径）                                                                 |
| kernel 定义位置 | 必须是真实 `.py` 文件；stdin/`exec` 字符串会报 `OSError: could not get source code`                                    |
| 动态 smem 上限  | **101,376 B**（实测 `BM=64, D=128, num_stages=2` → `OutOfResources: Required 106496, Hardware limit 101376`） |
| 降低 smem 的手段 | 减小 `BLOCK_M/BLOCK_N`、减小 `num_stages`、拆分 D 维度、改用 `tl.range(..., num_stages=k)` 精细控制                        |
| 张量核心路径      | `tl.dot` 在 sm_120 上落到 mma.sync（与 §1.2 一致）；`wgmma` 不可用，无需也无法开启                                             |
| 首次编译开销      | 首次 launch 包含 JIT 编译（秒级），benchmark 必须预热（≥20 次）并将编译时间单独记录，不计入 kernel 时间                                     |
| 编译缓存        | 默认 `~/.triton/cache`；基准/测试必须固定 `TRITON_CACHE_DIR` 到仓库内 `.triton_cache/`（gitignore），并在结果中记录 cache key      |
| 基准数据（参考）    | fp16 causal，S=512、D=128、B=4、BM=64/warps=4/stages=1 → 与 SDPA 最大绝对误差 ≈ 9.8e-4；非 causal（S=1024,B=2）≈ 1.2e-4  |

#### 7.9.3 逐文件规格

| 文件                                 | 必须内容                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `triton_kernels/flash_attn_fwd.py` | `@triton.jit def flash_attn_fwd_kernel(Q, K, V, Out, LSE, sm_scale, stride_*, S: tl.constexpr, D: tl.constexpr, BLOCK_M/N: tl.constexpr, CAUSAL: tl.constexpr, GROUP: tl.constexpr, STORE_LSE: tl.constexpr)`；grid = `(ceil(S_q/BLOCK_M), H_q, B)` 或展平；实现 §7.4.1 的 9 步主循环；GQA 用 `GROUP` 做 head 映射；`tl.exp2` + `* 1.44269504`；掩码用 `tl.where`；结尾 `acc / l` 后 `tl.store` |
| `triton_kernels/decode_attn.py`    | 两个 kernel：`decode_partial_kernel`（split-K 计算 partial m/l/O，写 workspace）+ `decode_reduce_kernel`（按 §7.5 的合并公式归约）；`num_splits` 由 host 侧 `decode_config.hpp` 同源规则决定（Python 侧读同一张表，避免两套逻辑）                                                                                                                                                                                |
| `triton_kernels/paged_attn.py`     | `paged_attn_decode_kernel`：`page_table` 间接寻址 + `block_size: tl.constexpr` + 边界掩码（`physical < 0` 视为不存在）；单测必须覆盖乱序 page 与 `block_size ∈ {1,8,16,32}`                                                                                                                                                                                                                     |
| `triton_kernels/online_softmax.py` | 独立算子：输入 `[N]` 序列按块累积 `(m, l, O)`，返回 online 结果与「一次性精确 softmax」对照；用于教学与 `docs/online_softmax.md` 的实证                                                                                                                                                                                                                                                                    |
| `triton_kernels/softmax.py`        | 行 softmax 算子（可选融合 mask/scale）；与 `math::softmax_stable`、`torch.softmax` 三方对照                                                                                                                                                                                                                                                                                           |
| `triton_kernels/autotune.py`       | 配置空间与缓存：`BLOCK_M ∈ {32,64,128}`、`BLOCK_N ∈ {32,64,128}`、`num_warps ∈ {4,8}`、`num_stages ∈ {1,2,3,4}`；**配置合法性校验：按 smem 模型估算 ≤ 101,376 B，超限配置在编译前剔除**；`SCA_TRITON_AUTOTUNE=0` 时使用固化配置表（保证测试确定性）；结果导出到 `python/scicompute_attention/triton_kernels/best_configs.json`（脚本生成，禁止手改）                                                                                         |
| `dispatch.py`（Python 侧）            | `backend ∈ {"auto","cuda","triton"}`：`auto` 规则 = 有 CUDA kernel 可用且 shape 在其能力内 → cuda；否则若 triton 可用 → triton；两者都不可用 → 明确报错（禁止静默回退 torch SDPA）                                                                                                                                                                                                                         |

#### 7.9.4 正确性要求（三方一致性）

```text
三方：triton_attention  vs  sca CUDA flash_attention  vs  torch SDPA(fp32/fp64 参考)
容差：与 §11.4 的 CUDA 路径同一张表（容器内不得为 Triton 单独放宽）
必须覆盖：causal / 非 causal；D ∈ {64,128,256}；S ∈ {1,7,63,64,65,127,128,129,1024,4096}；
          GQA group ∈ {1,2,4,8}；非 block 整数倍序列长度；全 -inf 行（全掩码）不得产生 NaN
额外要求：同一输入下 triton 与 cuda 的输出差异必须单独报告（这是「两套实现互检」的核心数据）
```

#### 7.9.5 性能要求

```text
必须报告三组数字（缺一不可）：
  1) kernel-only 时间：用 triton.testing.do_bench（或 CUDA event 包住 launch）
  2) 端到端时间：包含 Python 侧 launch 开销与 dtype/layout 转换（与 §12 同一口径）
  3) autotune 搜索开销：总搜索时间、配置数、最优配置（首次运行与缓存命中的差异）
对照对象：CUDA flash kernel（同 shape/dtype/布局）、torch SDPA、上游 C++ vLLM 原型（可选）
必须解释差距来源（示例）：
  - Triton 的 smem 上限 101,376 B 迫使 BLOCK_M/N 或 stages 更小
  - Triton 无法像手写 CUDA 那样精细控制 smem 分区与寄存器分配
  - Python launch 开销在小 shape 下占比高（ncu 看不到，必须用 nsys/CPU 计时佐证）
profiling：profiling/profile_triton.sh 必须能采集 Triton kernel（kernel 名通常形如 `flash_attn_fwd_kernel`，
  用 --kernel-name regex:_attn_fwd / regex:flash_attn 过滤），并把结果并入 analyze_ncu.py 的对比
```

#### 7.9.6 文档与交付要求

```text
docs/triton_optimization.md 必须包含：
  - 为什么引入 Triton（开发效率、autotune、生态）与不替代 CUDA 的理由（控制力、极限性能、C++ 侧可用性）
  - kernel 逐个说明：签名、tile 配置、smem 估算、实测最佳配置
  - autotune 方法论：配置空间、搜索策略、缓存、确定性开关
  - 三方对比表（精度 + 延迟 + 峰值显存）与结论：什么场景用 Triton、什么场景必须用 CUDA
  - 复现命令（含 TRITON_CACHE_DIR 与 autotune 开关）
docs/results/triton_vs_cuda.md 必须落盘：
  - 与 benchmark_triton.py / benchmark_flash.py 同一次运行的 JSON 路径
  - autotune 最优配置表（按 shape/dtype 分组）
  - 「Triton 是否达到 CUDA 的 X%」的明确结论（X 为实测数字）
```

#### 7.9.7 Triton 侧代码质量与禁止事项

```text
[ ] 所有 @triton.jit kernel 都放在 triton_kernels/ 下的 .py 文件（不允许字符串 exec / 动态生成）
[ ] tile 参数一律用 tl.constexpr 传入；host 侧显式校验（形状、head_dim、block 整除关系）
[ ] 热路径禁止 Python 循环逐 token 调用（decode 循环必须在 GPU 端或批量展开）
[ ] 测试默认关闭 autotune（SCA_TRITON_AUTOTUNE=0 + 固定配置），性能报告才开启
[ ] 禁止用 torch SDPA 顶替 Triton kernel 产出「Triton 数据」
[ ] 禁止把 Triton kernel 用于 vLLM(C++) 适配层或 C++ 公共 API 的实现路径
[ ] 必须记录 Triton 版本、CUDA 版本、cache key；版本变化后必须重跑 autotune 与一致性测试
```

---

## 8. KV Cache 规格（Level 4 前置，逐文件）

### 8.1 内存布局（唯一权威定义）

```text
K storage: [num_layers][num_blocks][block_size][num_kv_heads][head_dim]    (KvLayout::kBlockMajor)
V storage: 同 K，独立缓冲
block table: [max_num_seqs][max_blocks_per_seq]（int32，-1 = 空槽）
slot_mapping[t] = physical_block * block_size + offset_in_block
```

与上游 vLLM(C++) 的对应关系（必须逐字写进 `docs/paged_kv_cache.md`）：

| 上游                                                                                                          | 本项目                                                                           | 关系                                                |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------- |
| `vllm::KVCache`：每 layer 连续 `[num_blocks * block_size * num_kv_heads * head_dim]`                            | `sca::KVCache`：`[num_layers][num_blocks][block_size][num_kv_heads][head_dim]` | **同构**，可零拷贝映射（层指针直接传出）                            |
| `vllm` 无 block table 抽象（Phase 0 复核 `block_manager.hpp`）                                                     | `sca::BlockManager` + `BlockTable`                                            | 若上游已有等价抽象，则适配层复用其分配结果，不重复分配                       |
| RLHF `rlhf::KVCache`：CPU dense `[num_layers, 2, batch, num_heads, seq_len, head_dim]`，`max_seq_len` 默认 2048 | `sca::PagedKVCache`                                                           | 迁移路径：dense CPU → block GPU；文档给出转换伪代码与收益（显存、并发序列数） |

### 8.2 `src/kv_cache/kv_cache.cpp`

必须实现：

```text
Create(cfg, device)
  1) 校验 cfg（num_layers/num_kv_heads/head_dim/block_size 均为正；block_size ∈ {1,8,16,32,64}）
  2) 计算每层字节数：num_blocks * block_size * num_kv_heads * head_dim * sizeof(dtype)
  3) 计算总字节数：2 * num_layers * 上述，并用 cudaMemGetInfo 检查剩余显存
     - 若 > 剩余显存 * kKvBudgetRatio（默认 0.8）→ 返回 kKVCapacityExceeded，并打印「建议 num_blocks ≤ N」
  4) 经 sci::MemoryPool/Allocator 申请；失败转 kCudaOutOfMemory
  5) 初始化 stats（bytes / peak_bytes）
Append(layer, k_new, v_new, slot_mapping, num_tokens, stream)
  1) 校验 shape：[num_tokens, num_kv_heads, head_dim]，dtype 一致，slot_mapping 非空
  2) 启动 gather/scatter kernel（src/kv_cache/kv_cache_layout.cpp 提供 address 计算）
  3) 不做 device 同步；错误延迟到 SCI_CUDA_CHECK_LAST
Reset(block_ids, num_blocks, stream)  → 清零对应 block（memset 风格 kernel，非 cudaMemset 逐块）
Stats() / MemoryBytes()              → 只读，无锁（单线程调用约定须写入文档）
```

### 8.3 `src/kv_cache/kv_cache_layout.cpp`

唯一地址计算实现（DRY），提供 `__host__ __device__` 双版本：

```cpp
__host__ __device__ inline int64_t KvOffset(int64_t layer, int64_t block, int64_t offset_in_block,
                                            int64_t kv_head, int64_t d, const KVCacheShape& s);
```

要求：单元测试 `tests/kv/test_kv_cache.cpp` 用「逐字节手工地址」与公式交叉验证（至少 3 组 shape），并在 `docs/kv_cache.md` 贴出地址公式与示例计算。

### 8.4 显存预算（8 GB 机器必须给数字）

公式：

```text
KV_bytes = 2 * num_layers * num_blocks * block_size * num_kv_heads * head_dim * sizeof(dtype)
max_tokens_total = num_blocks * block_size
```

参考示例（必须写进 `docs/kv_cache.md`，并用本机 `cudaMemGetInfo` 复算）：

| 配置（示例）                                         | 每 block 字节         | 3 GiB 预算可容纳 blocks | 可容纳总 token |
| ---------------------------------------------- | ------------------ | ------------------ | ---------- |
| layers=32, n_kv_heads=8, D=128, block=16, fp16 | 2,097,152 B（2 MiB） | 1536               | 24,576     |
| layers=32, n_kv_heads=8, D=64, block=16, fp16  | 1,048,576 B（1 MiB） | 3072               | 49,152     |
| layers=28, n_kv_heads=4, D=128, block=16, bf16 | 917,504 B          | 3510               | 56,160     |

### 8.5 `src/kv_cache/block_manager.cpp`

```text
free_list: std::vector<int32_t>（栈式 LIFO，O(1) 分配/回收）
Allocate(n) : n > free_count ⇒ kKVCapacityExceeded（状态不变，必须可回滚）
Free(ids,n) : 校验 id 范围与「未重复释放」（debug 断言 + release 统计计数）
AppendToTable(seq_id, ids, n) : 追加到 host 表；revision++
SyncTableToDevice(stream) : 仅在 revision 变化时 cudaMemcpyAsync（默认 stream）
```

要求：`Stats()` 暴露 `num_free_blocks / num_total_blocks / num_fragment_events`（碎片事件定义为「单次请求需要 > 剩余连续空闲... 不适用，因 block 无连续性要求」→ 因此本项改为 `allocation_failures` 计数）。

### 8.6 `src/kv_cache/paged_kv_cache.cpp`

```text
Create(cfg, device)          : 组合 KVCache + BlockManager + BlockTable(device)
AppendTokens(seq_id, layer, k_new, v_new, stream)
  1) 计算 needed_blocks = ceil((current_len + num_new) / block_size) - owned_blocks
  2) needed > 0 ⇒ BlockManager::Allocate；失败 ⇒ kKVCapacityExceeded（不半途修改状态）
  3) 生成 slot_mapping（ComputeSlotMapping，唯一实现）
  4) KVCache::Append
  5) 更新 seq 元数据（长度、owned_blocks）；revision++
ResetSeq(seq_id, stream)     : 释放 blocks + 清表项 + 长度归零（顺序：先清 device 表再回收，避免悬空引用）
SyncPageTable(stream)        : 按 revision 增量上卡
```

### 8.7 测试点（逐条对应到 `tests/kv/*`）

```text
[ ] block_size ∈ {1,8,16,32,64} 全通过
[ ] 长度非 block_size 整数倍（1,17,33 token）
[ ] 容量耗尽：请求 blocks > num_blocks → kKVCapacityExceeded 且旧数据完好
[ ] ResetSeq 后再次 Append 复用 block，无数据串扰（用已知 pattern 校验）
[ ] 多序列并发（≥8 序列）交错 append，页表互不影响
[ ] Stats 的 used/free/peak 与实际一致（与手工计算对比）
```

### 8.8 YAGNI 边界（明确不做，写进 Roadmap）

```text
不做：prefix caching、block 共享/COW、抢占式驱逐、多卡 KV 分片、量化 KV（fp8/int8）
理由：当前目标是「可用的 paged KV + decode 路径」，上述能力需要调度器配合，属于后续版本
```

---

## 9. Runtime / Dispatcher / 阈值标定

### 9.1 `src/runtime/dispatcher.cpp`

选择算法（必须与代码逐行对应，并可由 `Explain()` 复现）：

```text
Select(cfg, shape):
  1) Validate(cfg, shape)；失败立刻返回（不做任何猜测）
  2) 若 cfg.backend != kAuto ⇒ 检查该后端 Supports()；不支持则返回错误（allow_fallback 决定是否降级，默认 false）
  3) 若提供 page_table ⇒ kPaged
  4) 若 shape.seq_q == 1 或 seq_q * 8 <= seq_kv ⇒ 阈值表判定 prefill/decode（表来自 dispatch_table.inc）
  5) 否则 ⇒ kFlash（若 head_dim/seq_kv 超出 flash 能力 ⇒ 尝试 kTiled；仍不支持 ⇒ kNaive + 显式 note）
  6) 记录 reason 字符串（含关键数值），供 Explain() 与测试快照使用
```

规则表（初值，须由阈值 sweep 校准）：

| seq_q | seq_kv | head_dim | 选择                    | 理由（必须由实测支撑）             |
| ----- | ------ | -------- | --------------------- | ----------------------- |
| 1     | 任意     | 任意       | Decode                | 算术强度极低，split-K 并行度更关键   |
| ≤ 8   | ≥ 2048 | ≤ 128    | Decode(seq_q 分组)      | 小 Q 复用 K/V，避免 tile 启动开销 |
| ≥ 16  | ≥ 16   | 任意       | Flash                 | tile + MMA 效率最高         |
| 任意    | ≤ 128  | ≤ 128    | Tiled 或 Flash（按实测）    | 短序列下 tiled 更省寄存器        |
| 任意    | 任意     | 任意       | Paged（有 page_table 时） | 与 vLLM 服务形态一致           |

### 9.2 `src/runtime/capability.cpp`

```text
ForDevice(device_id) : 首次调用执行 cudaGetDeviceProperties + 探针结果，缓存到静态结构体
Refresh(device_id)   : 强制重新探测（测试用）
```

必须探测：`sm_count / smem_per_sm / smem_per_block_optin（用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 或 cudaFuncSetAttribute 试探）/ regs_per_sm / l2_bytes / total_mem_bytes / cc`；TMA 支持由 `arch_features` 的编译期 + 运行期共同判定。

### 9.3 `src/runtime/launcher.cpp`

```text
LaunchKernel(func, grid, block, smem_bytes, stream, args...)
  1) 若 smem_bytes > 48 KB ⇒ 首次调用 cudaFuncSetAttribute（按函数指针缓存已设置值，避免重复调用）
  2) SCI_CUDA_CHECK_LAST() 检查启动错误
  3) 若 SCI_ATTENTION_DEBUG_SYNC=1 ⇒ cudaStreamSynchronize + 二次错误检查
  4) 返回 sci::Status
```

禁止：在该函数外任何地方直接写 `<<<>>>` 启动（全部收敛到这里或各 backend 的 launch 文件）。

### 9.4 `src/runtime/workspace.cpp` 与 `dispatch_table.inc`

`Workspace` 规则：

```text
一次性分配 = max(各候选后端 WorkspaceBytes)
对齐 256 B；Slice 记录 (offset,bytes) 用于越界检测
推理路径禁止 cudaMalloc / cudaFree（仅 Create 阶段允许）
超出 workspace_limit_bytes ⇒ kWorkspaceExceeded（含建议值）
```

`dispatch_table.inc` 生成流程（必须固化为脚本，可复现）：

```bash
# 1) 采集：扫描 seq_q × seq_kv × head_dim，比较 flash / decode / tiled 的 P50 kernel 时间
python3 tools/dispatch_threshold_sweep.py --dtype bf16 --out benchmarks/results/$(date +%F)-$(git rev-parse --short HEAD)/dispatch_sweep.json
# 2) 生成：写入 C++ 常量表（禁止手改，文件头自动加 "AUTO-GENERATED, DO NOT EDIT"）
python3 tools/dispatch_threshold_sweep.py --emit-inc src/runtime/dispatch_table.inc
# 3) 回归：tests/integration/test_dispatch_explain.cpp 对若干 (shape,cfg) 组合做行为快照
```

### 9.5 异步与错误处理（硬约束）

```text
[ ] 推理热路径零 cudaDeviceSynchronize / 零 cudaStreamSynchronize
[ ] 允许同步的位置：benchmark 计时、显式 Synchronize() API、SCI_ATTENTION_DEBUG_SYNC=1、错误注入测试
[ ] 所有 CUDA API 返回值必须检查（cudaMalloc/cudaMemcpy/cudaGetLastError/cudaStreamSynchronize 等）
[ ] kernel launch 后必须有 launch error 检查
[ ] 所有公开 API 的错误必须是 sci::Status / sci::Result，消息含具体数值与建议动作
```

验证手段：`rg -n "cudaDeviceSynchronize|cudaStreamSynchronize" src/ include/` 结果必须在白名单内（白名单文件：`src/runtime/launcher.cpp` 的 debug 分支、benchmark 计时路径），并在 `docs/kernel_design.md` 记录该检查命令与结果。

---

## 10. Python Binding 规格

### 10.1 模块与文件

```text
python/scicompute_attention/__init__.py     : 导出 API、__version__、list_backends()
python/scicompute_attention/ops.py          : 顶层函数（参数校验 + 调用 _core + 错误映射）
python/scicompute_attention/reference.py    : 纯 PyTorch 参考实现（含逐步 online softmax 版本）
python/scicompute_attention/kv_cache.py     : KVCache / PagedKVCache 封装
python/scicompute_attention/benchmark.py    : 读取 benchmark JSON 并格式化输出
python/scicompute_attention/utils.py        : dtype/layout 工具、显存估算、梯度/连续性检查
python/scicompute_attention/version.py
python/scicompute_attention/py.typed
```

### 10.2 对外 API（必须与 C++ API 一一对应）

```python
import scicompute_attention as sca

# 基础入口（张量：torch.Tensor，零拷贝传入；返回 torch.Tensor）
out = sca.attention(q, k, v, causal=True, layout="bhsd", scale=None, backend="auto")
# backend ∈ {"auto","cuda","triton"}：auto = 有 CUDA kernel 走 CUDA；否则走 Triton；两者都不可用则显式报错
out = sca.naive_attention(q, k, v, causal=False)
out = sca.tiled_attention(q, k, v, causal=False)
out = sca.flash_attention(q, k, v, causal=True, return_lse=False)
out = sca.decode_attention(q, k, v, num_splits=0)

# Triton 轨道（Python 侧实现，见 §7.9；不参与 C++/vLLM 路径）
out = sca.triton_attention(q, k, v, causal=True, autotune=False)   # autotune=False ⇒ 用固化配置，测试用
out = sca.triton_paged_attention(q, paged, seq_lens=..., causal=True)
cfg = sca.triton_best_config(q, k, v, causal=True)                 # 返回当前 shape 的最优配置（只读）
sca.triton_set_autotune(False)                                     # 等价于 SCA_TRITON_AUTOTUNE=0

# KV Cache
cache = sca.KVCache(num_layers=32, num_kv_heads=8, head_dim=128,
                    block_size=16, num_blocks=1536, dtype="fp16", device=0)
cache.append(layer=0, k=k_new, v=v_new)            # 内部生成 slot_mapping
view = cache.k(layer=0)                            # 返回 torch.Tensor 视图（零拷贝）
cache.reset()

paged = sca.PagedKVCache(...)                      # 多序列
out = sca.paged_attention(q, paged, seq_lens=..., causal=True)

# 调试与可解释性
info = sca.dispatch_info(q, k, v, causal=True)     # {"backend": "...", "reason": "...", "tile": {...}}
ref  = sca.reference_attention(q, k, v, causal=True)   # 纯 PyTorch FP32/FP64 参考
sca.synchronize()
```

### 10.3 `src/bindings/python_module.cpp`

```text
模块名：scicompute_attention._core
绑定：
  - attention / naive_attention / tiled_attention / flash_attention / decode_attention / paged_attention
  - KVCache / PagedKVCache（构造、append、reset、k/v 视图、stats）
  - dispatch_info / list_backends / synchronize / device_capability
  - version()
张量互操作：接受 torch 张量对象 → 取 data_ptr/shape/stride/dtype/device_index →
  用 sci::Tensor(shape, dtype, device, external_data, owns_data=false) 包装（零拷贝）
  返回：以 DLPack 或 __cuda_array_interface__ 包装为 torch 张量（由 Python 侧完成更易维护）
错误：C++ Status → 抛 RuntimeError(f"[{code}] {message}")；不吞异常
```

补充要求：

```text
- `_core` 只暴露 CUDA 侧能力；Triton 相关函数由 Python 层（dispatch.py + triton_kernels/）实现
- `sca.list_backends()` 必须返回可用性诊断，例如：
    {"cuda": {"available": true, "reason": ""},
     "triton": {"available": true, "version": "3.7.1", "reason": ""},
     "torch_sdpa_fallback": {"available": false, "reason": "禁止作为实现路径（仅参考）"}}
- 任何「Triton 不可用」的情况必须给出可操作原因（未安装 / 版本过低 / GPU 不支持 / cache 不可写）
```

### 10.4 构建与安装

`python/pyproject.toml` 必须包含：

```toml
[build-system]
requires = ["scikit-build-core>=0.9", "pybind11>=2.12"]
build-backend = "scikit_build_core.build"

[project]
name = "scicompute-attention"
requires-python = ">=3.10"
dependencies = ["torch>=2.2"]

[tool.scikit-build]
cmake.args = ["-DSCI_ATTENTION_BUILD_TESTS=OFF", "-DSCI_ATTENTION_BUILD_BENCHMARKS=OFF",
              "-DSCI_ATTENTION_BUILD_PYTHON=ON"]
wheel.packages = ["scicompute_attention"]
```

安装与验证命令（必须写入 README）：

```bash
cd python && pip install -e . && python -c "import scicompute_attention as sca; print(sca.__version__)"
python -m pytest python/tests -v
```

### 10.5 Python 测试清单

```text
test_api.py     : 4 个 backend 的 CPU/GPU 调用、非法参数错误、dispatch_info 字段完整性
test_parity.py  : 与 reference.py / torch SDPA 的最大/平均误差（容差见 §11.4），覆盖 causal 与非 causal
test_kv_cache.py: append/reset/统计、容量耗尽、显存上限估算函数
test_errors.py  : 错误码字符串、异常类型、消息包含关键数值
test_triton_parity.py : Triton vs SDPA vs CUDA kernel 三方误差（按 §7.9.4 的 shape/dtype 矩阵）
test_triton_config.py : autotune 配置合法性（smem 估算 ≤ 101,376 B）、SCA_TRITON_AUTOTUNE=0 时的确定性
```

---

## 11. 测试规格（逐文件）

### 11.1 测试层级（与 rl-infra workflow 对齐）

| Level | 内容         | 判据                                         |
| ----- | ---------- | ------------------------------------------ |
| L1    | 编译         | 零错误；`SCI_ATTENTION_WERROR=ON` 时零警告         |
| L2    | 单元测试       | `tests/unit`、`tests/kv` 全绿                 |
| L3    | Kernel 正确性 | `tests/kernel` 全绿（含边界与多 dtype）             |
| L4    | 运行时检查      | `compute-sanitizer memcheck/racecheck` 零错误 |
| L5    | 集成         | `tests/integration` + `python/tests` 全绿    |
| L6    | 回归         | 每次修改后重跑 L1–L5；benchmark 数据与 git sha 绑定     |

### 11.2 逐文件测试规格

| 文件                                            | 必测内容                                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------------ |
| `tests/unit/test_config.cpp`                  | `Validate()` 的每条规则（head_dim 集合、causal 与 seq 关系、varlen 单调性、layout 合法性）；错误消息含字段名 |
| `tests/unit/test_dispatch.cpp`                | 6 组 (cfg, shape) 的选择结果与 `reason` 字段                                            |
| `tests/unit/test_capability.cpp`              | 探测字段非零、`Refresh()` 生效、缓存语义                                                     |
| `tests/unit/test_status.cpp`                  | 每个错误码的 `ToString` 与 `MakeStatus` 拼接                                            |
| `tests/kernel/test_naive_correctness.cu`      | fp32/fp16/bf16 × causal/非 causal；与 torch fp32 参考对比                             |
| `tests/kernel/test_tiled_correctness.cu`      | 同上，且断言「未物化 N×N」（用显存峰值或 workspace 字节数验证）                                        |
| `tests/kernel/test_flash_correctness.cu`      | D=64/128/256；S ∈ {1,7,63,64,65,127,128,129,1024,4096}；非 causal                 |
| `tests/kernel/test_flash_causal.cu`           | causal 路径；对角线块（S_q≠S_kv）；S_q=1；S_q=65 等非 tile 整数倍                              |
| `tests/kernel/test_head_dims.cu`              | head_dim ∈ {32,64,96,128,160,192,256} 全覆盖；不支持值报错                               |
| `tests/kernel/test_numerics.cu`               | logits ∈ {±1e4, ±1e2, 全 0, 全相同}、极大序列（S=8192）无 NaN/Inf；与 fp64 参考对比              |
| `tests/kernel/test_layout.cu`                 | BHSD/BSHD 结果一致（同一数据两种布局）；非法 layout 拒绝；非连续输入拒绝并给出原因                             |
| `tests/kernel/test_decode_correctness.cu`     | num_splits ∈ {1,2,3,4,8,16}；S_kv ∈ {1, 127, 1024, 8192}                        |
| `tests/kernel/test_paged_correctness.cu`      | 乱序 page、block_size ∈ {1,8,16,32}、GQA（H_q/H_kv = 2,4,8）                         |
| `tests/kv/test_kv_cache.cpp`                  | 见 §8.7                                                                         |
| `tests/kv/test_block_manager.cpp`             | 分配/回收/耗尽/重复释放/多序列隔离                                                            |
| `tests/kv/test_paged_kv.cpp`                  | 多序列交错、页表上卡、统计一致性                                                               |
| `tests/integration/test_attention_api.cpp`    | 公共 API 端到端（含默认 cfg、错误路径）                                                       |
| `tests/integration/test_dispatch_explain.cpp` | 阈值表行为快照（表变化必须导致测试更新，避免静默漂移）                                                    |
| `tests/integration/test_vllm_adapter.cpp`     | 仅 `SCI_ATTENTION_BUILD_VLLM_ADAPTER=ON`；参数转换 + 数值对照                            |
| `python/tests/test_triton_parity.py`          | 三方一致性：Triton / CUDA kernel / SDPA（§7.9.4）；输出三方误差表                              |
| `python/tests/test_triton_config.py`          | 配置校验：smem 超限配置被剔除、autotune 关闭时结果可复现（同输入两次调用逐位一致）                               |

### 11.3 测试数据与随机性

```text
所有随机输入：固定 seed（默认 1234），生成脚本 tools/gen_qkv.py，输出 .npy/.bin 到 tests/data/
每种 (dtype, layout, causal) 组合保留最小可复现样例；测试失败必须能直接复现：
  pytest / ctest 输出中必须包含「seed + shape + dtype + causal + 生成命令」
```

### 11.4 数值容差表（初值，必须实测校准）

参考实现优先级：`torch.float64` 参考 > `torch.float32` 参考 > `math::ref::softmax_f32`。

| 被测路径              | 参考                  | 最大绝对误差 | 平均绝对误差 | 说明                 |
| ----------------- | ------------------- | ------ | ------ | ------------------ |
| naive(fp32)       | torch fp64          | ≤ 1e-4 | ≤ 1e-6 | 数值参考               |
| naive(fp16)       | torch fp32          | ≤ 5e-3 | ≤ 5e-4 | 输入用 fp16，参考用 fp32  |
| naive(bf16)       | torch fp32          | ≤ 2e-2 | ≤ 2e-3 | bf16 尾数 8 bit，放宽合理 |
| tiled/flash(fp16) | torch fp32          | ≤ 5e-3 | ≤ 5e-4 | 与 naive(fp16) 同量级  |
| tiled/flash(bf16) | torch fp32          | ≤ 2e-2 | ≤ 2e-3 | 同上                 |
| decode(fp16)      | flash(fp32) 同输入     | ≤ 1e-3 | ≤ 1e-4 | split-K 合并精度       |
| paged(fp16)       | 连续 KV 的 flash(fp16) | ≤ 1e-3 | ≤ 1e-4 | 分页不应引入额外误差         |

校准规则：Phase 2/5 结束后按实测值收紧（不得高于实测值的 2 倍）；每次调整必须在 `docs/numerical_stability.md` 记录「旧值 → 新值 → 依据实测数据」。

### 11.5 运行时检查命令（必须进 CI 脚本）

```bash
compute-sanitizer --tool memcheck --launch-timeout 120 \
  ./build/tests/kernel/test_flash_correctness
compute-sanitizer --tool racecheck --launch-timeout 120 \
  ./build/tests/kernel/test_tiled_correctness
compute-sanitizer --tool initcheck \
  ./build/tests/kernel/test_decode_correctness
```

通过标准：`========= ERROR SUMMARY: 0 errors`；任何 non-zero 视为阻塞缺陷。

### 11.6 一键验证入口

```bash
bash scripts/configure.sh --build-type Release
bash scripts/build.sh -j"$(nproc)"
bash scripts/test.sh --with-python          # L1-L5
bash scripts/test.sh --filter flash --sanitize   # L4 子集
```

---

## 12. Benchmark 规格（逐文件）

### 12.1 方法论（固定，不允许每份报告自定义）

```text
warmup     : 20 次（不含计时）
measure    : 100 次（每次独立 launch，同一 stream）
计时       : CUDA Event（start/stop 包住纯 kernel 调用）；端到端另计（含 H2D/D2H 与调度）
统计       : mean / median / std / P50 / P90 / P95 / P99
固定       : seed、输入张量、dtype、layout、clock 状态（记录 SM clock 与是否降频）
公平性     : 同一 GPU 状态、同一 dtype、同一布局；SDPA 必须显式指定 backend；
             禁止只报告最优 run；禁止对不同实现使用不同 warmup/measure 次数
显存       : 记录 cudaMemGetInfo 前后差 + workspace 峰值
```

### 12.2 逐文件

| 文件                                           | 内容                                                                                                           |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `benchmarks/common/bench_utils.hpp`          | CUDA Event 计时器、warmup/measure 循环、clock 查询、结果结构体 `CaseResult`                                                 |
| `benchmarks/common/bench_export.hpp`         | JSON/CSV/Markdown 导出（字段见 §12.4）；与 `sci::benchmark::BenchmarkRunner` 的适配                                      |
| `benchmarks/common/bench_matrix.hpp`         | 三档矩阵：`sanity`（快速回归）、`main`（论文/报告用）、`long`（长序列）                                                               |
| `benchmarks/benchmark_memory_bw.cu`          | 峰值带宽标定（copy/read/write 三类 kernel），输出 GB/s；roofline 的机器参数来源                                                   |
| `benchmarks/benchmark_naive.cu`              | Level 0（仅小序列，含显存跳过逻辑）                                                                                        |
| `benchmarks/benchmark_tiled.cu`              | Level 1                                                                                                      |
| `benchmarks/benchmark_flash.cu`              | Level 3（核心，含 causal/非 causal，D=64/128/256）                                                                   |
| `benchmarks/benchmark_decode.cu`             | Level 4（含 num_splits 敏感性）                                                                                    |
| `benchmarks/benchmark_paged.cu`              | Level 5（含 block_size 敏感性）                                                                                    |
| `benchmarks/benchmark_sdpa.py`               | PyTorch SDPA 对照（`flash`/`mem_efficient`/`math` 三种 backend 分别记录）                                              |
| `benchmarks/benchmark_triton.py`             | Triton kernel 基准：kernel-only（do_bench）+ 端到端 + autotune 搜索开销；与 CUDA flash 同 shape 对照，输出 JSON 含最优配置与 cache key |
| `benchmarks/benchmark_vllm.cpp`              | 上游 C++ vLLM 原型 vs SCA（A/B，见 §15.5）                                                                           |
| `benchmarks/benchmark_e2e_prefill_decode.cu` | prefill + decode 连续循环（rollout 形态），输出 tokens/s 与 KV 复用开销                                                      |

### 12.3 参数矩阵（`main` 档）

```text
B      : 1, 2, 4, 8
H_q    : 8, 16, 32      H_kv : 8（GQA 覆盖 H_q=16/32）
S_q    : 128, 256, 512, 1024, 2048, 4096, 8192, 16384
S_kv   : = S_q（prefill）；decode 时 S_kv ∈ {1024, 4096, 16384}
D      : 64, 128, 256
dtype  : fp16, bf16（fp32 仅 naive 对照）
causal : true, false
```

真实模型档（`main` 必跑，来自 §23 的 Qwen3-4B-Thinking-2507-Q8）：

```text
档位名   : real-qwen3-4b
B=1, H_q=32, H_kv=8, D=128, causal=True  （GQA group = 4）
prefill  : S ∈ {512, 1024, 2048, 4096}
decode   : S_kv ∈ {1024, 4096, 8192}（>8192 需 --allow-large-kv 且显存检查通过）
数据来源 : tests/data/qwen3_4b/ 的真实 Q/K/V（tools/dump_qwen3_qkv.py 生成）
执行脚本 : benchmarks/benchmark_qwen3_shapes.py；结果单独一节写入 benchmark_report.md
```

显存保护（8 GB 机器硬约束）：

```text
预估字节 = q + k + v + out + workspace + (naive 时的 B*H*S_q*S_kv*sizeof(float))
若预估 > 3 GiB ⇒ 跳过该 case，并在 JSON 中写 "skipped": "estimated 34.4 GiB > budget 3 GiB"
禁止因 OOM 导致进程崩溃；禁止静默跳过（必须出现在结果文件里）
```

Triton 轨道（`benchmark_triton.py`）额外规则：

```text
矩阵：默认只跑 sanity + main（不含 long）；long 仅对 autotune 后的最优配置跑 1 组
autotune：先跑一次完整搜索（记录搜索耗时与配置数），再固定 best_configs.json 计时；
          报告必须区分「autotune 搜索耗时」与「kernel 计时」，禁止把编译/搜索时间计入延迟
对照组：必须与同 shape 的 CUDA flash kernel 在同一进程/同一 GPU 状态下测量
输出：benchmarks/results/<date>-<sha>/triton.json（含 triton 版本、cache key、最优配置表）
```

### 12.4 结果格式（JSON 必须逐字段实现）

```json
{
  "benchmark_name": "benchmark_flash",
  "git_sha": "abc1234", "build_type": "Release",
  "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU", "compute_cap": "12.0",
  "driver": "5xx.xx", "cuda": "13.2", "sm_clock_mhz": 2400,
  "timestamp": "2026-09-18T22:10:00+08:00",
  "warmup": 20, "runs": 100, "seed": 1234,
  "cases": [
    {
      "case_id": "flash_B1_H16_Sq1024_Skv1024_D128_bf16_causal0",
      "params": {"B": 1, "H": 16, "Sq": 1024, "Skv": 1024, "D": 128,
                 "dtype": "bf16", "causal": false, "tile": {"m": 64, "n": 64, "warps": 4, "stages": 2}},
      "latency_ms": {"mean": 0.0, "median": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
      "flops_effective": 0, "tflops": 0.0,
      "bytes_moved_modeled": 0, "effective_bw_gbps": 0.0,
      "peak_mem_mb": 0.0, "workspace_mb": 0.0,
      "registers_per_thread": 0, "smem_bytes": 0, "achieved_occupancy_pct": 0.0,
      "reference_max_abs_err": 0.0
    }
  ]
}
```

CSV 列（与 benchmark skill 口径一致）：

```csv
name,case_id,warmup,runs,mean_ms,median_ms,std_ms,p50_ms,p90_ms,p95_ms,p99_ms,tflops,effective_bw_gbps,peak_mem_mb,unit
```

FLOPs 口径（必须写在 `docs/benchmarking.md`）：

```text
非 causal: FLOPs = 2 * B * H_q * S_q * S_kv * D (QKᵀ) + 2 * B * H_q * S_q * S_kv * D (PV)
causal   : 上述按 Σ_i (i+1) / (S_q*S_kv) 折算（对角线块按实际有效 token 数精确计算）
TFLOPS   = FLOPs / latency
```

### 12.5 复现命令模板（必须写入 README 与报告）

```bash
SHA=$(git -C $SCA_ROOT rev-parse --short HEAD)
OUT=$SCA_ROOT/benchmarks/results/$(date +%F)-${SHA}
mkdir -p "${OUT}"
./build/benchmarks/benchmark_flash --suite main --seed 1234 --runs 100 --warmup 20 \
  --json "${OUT}/flash.json" --csv "${OUT}/flash.csv"
python3 $SCA_ROOT/benchmarks/benchmark_sdpa.py \
  --suite main --seed 1234 --json "${OUT}/sdpa.json"
```

---

## 13. Profiling 规格（Nsight）

### 13.1 脚本与产出

| 文件                            | 内容                                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `profiling/profile_naive.sh`  | `ncu` 采集 naive kernel（小规模）                                                                              |
| `profiling/profile_flash.sh`  | `ncu` 采集 flash（默认 D=128, S=4096, bf16, causal）                                                          |
| `profiling/profile_decode.sh` | `ncu` 采集 decode（S_kv=16384）                                                                             |
| `profiling/profile_vllm.sh`   | `ncu` 采集上游 C++ vLLM 原型 kernel（对照）                                                                       |
| `profiling/profile_triton.sh` | `ncu`/`nsys` 采集 Triton kernel（`--kernel-name regex:_attn_fwd\|flash_attn`）；同时用 nsys 量化 Python launch 开销 |
| `profiling/metrics/*.txt`     | 指标清单（见 §13.2），由脚本 `--metrics-file` 读取                                                                   |
| `profiling/analyze_ncu.py`    | ncu CSV → Markdown 摘要 + 与上一次运行对比（变化 >5% 高亮）                                                             |

统一要求：

```bash
set -euo pipefail
command -v ncu >/dev/null || { echo "ncu not found; 安装 Nsight Compute 或改用 --set full 的 nvprof 替代"; exit 2; }
ncu --target-processes all --kernel-name regex:"flash|decode|naive" \
    --metrics-file profiling/metrics/fa_metrics.txt \
    --csv --page raw --log-file profiling/reports/flash_$(date +%F-%H%M).csv \
    ./build/benchmarks/benchmark_flash --suite sanity
python3 profiling/analyze_ncu.py profiling/reports/flash_*.csv
```

### 13.2 必采指标（先用 `ncu --query-metrics | rg <name>` 校验名称随版本的差异）

```text
gpu__time_duration.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_elapsed
lts__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active          // achieved occupancy
launch__registers_per_thread
launch__shared_mem_per_block_static
launch__shared_mem_per_block_dynamic
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active   // 名称随架构变化，需查询确认
dram__bytes.sum
lts__t_sectors.sum
```

### 13.3 nsys（时序与端到端）

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true \
  --output profiling/reports/flash_timeline_$(date +%F-%H%M) \
  ./build/benchmarks/benchmark_flash --suite sanity
```

必须回答的问题（写入 `docs/profiling.md` 与 `docs/results/profiling_summary.md`）：

```text
1) kernel 之间是否有不可解释的 gap（启动开销 / 同步 / 分配）？
2) 是否存在 H2D/D2H 拷贝出现在热路径？
3) prefill 与 decode 的时间分布差异？
4) 与 ncu 的 kernel 时间是否一致（差异 >10% 需给出解释）？
```

### 13.4 瓶颈判定（必须给结论标签 + 证据）

```text
compute-bound : sm__throughput 高 且 dram__throughput 低
memory-bound  : dram__throughput 接近实测峰值（§14.2）
latency-bound : occupancy 低 + long scoreboard stall 高
launch-bound  : kernel duration 缩短但端到端不变 / nsys 显示 gap 占比高
```

---

## 14. Roofline / IO 分析规格（`docs/roofline.md` 必答）

### 14.1 必须建立的三张表

```text
表 A：机器参数表（实测）
  - 峰值 HBM 带宽（来自 benchmarks/benchmark_memory_bw.cu，取 copy/read/write 最大值）
  - 峰值 FP16/BF16 Tensor Core TFLOPS（理论值 + 实测 GEMM 值，注明来源与差距）
  - L2 带宽（可选用 lts__throughput 推导）
  - smem 带宽（可按每 SM 128 B/clk 估算，需注明假设）
表 B：算法流量表（每个实现）
  列：实现 | 理想 HBM 读字节 | 理想 HBM 写字节 | 附加中间量字节 | 计算量 FLOPs | 算术强度 FLOPs/Byte
  行：naive / tiled / flash / decode / paged
表 C：实测对照表
  列：实现 | S | P50 latency | TFLOPS | 实测有效带宽 GB/s | 理论有效带宽 GB/s | 利用率
```

### 14.2 流量模型（必须给出公式）

```text
naive : 读 Q,K,V + 写/读 S（B*H*S_q*S_kv*4B 多次往返）+ 读 P + 写 O
tiled : 读 Q,K,V（K 被每个 Q block 重复读，重复次数 = ceil(S_q/BLOCK_M)）+ 写 O
flash : 读 Q,K,V 一次（无 S 往返）+ 写 O（+ 可选 LSE）
decode: 每步读 K,V 全量（S_kv 有关），几乎不读 Q；瓶颈 = 2 * B * H_kv * S_kv * D * sizeof(dtype)
paged : 同 decode，但 K/V 非连续（额外 page table 读，见 §7.6）
```

### 14.3 必答问题（每条必须带数字与出处）

```text
Q1 为什么标准 Attention 会产生 O(N²) 中间矩阵？给出具体张量形状与字节数（含 B/H 数值）
Q2 FlashAttention 为什么不需要存储完整 N×N？用「分块 + 在线 softmax」论证，并给出 workspace 字节数
Q3 Online softmax 为什么数值稳定？给出推导与极端 logits 实测（引用 test_numerics 数据）
Q4 收益来自计算减少还是 IO 减少？用表 C 的 TFLOPS 与有效带宽同时回答，明确结论
Q5 Prefill 与 Decode 为何需要不同 kernel？用算术强度与实测 P50 数据回答
Q6 Paged KV Cache 为什么对 serving 重要？用显存碎片/并发序列数（§8.4 表）与实测吞吐回答
Q7 FlashAttention 与 PagedAttention 的职责边界是什么？画图并说明两者在 vLLM(C++) 中分别替换了什么
Q8 为什么本机（sm_120）不采用 wgmma/tcgen05 路径？引用 §1.2 的 ptxas 原始报错
Q9 8 GB 显存下如何权衡 KV Cache 与长序列 benchmark？给出预算表与跳过策略实测
Q10 CUDA kernel 与 Triton kernel 在同一 shape 下的算术强度相同、性能却不同的原因是什么？
    （提示方向：smem 上限 101,376 B 导致的 tile/stages 差异、寄存器与 smem 分配控制力、
      Python launch 开销、Triton 编译器的调度选择；必须用 ncu 的 smem/occupancy/stall 数据佐证）
```

补充要求：`docs/roofline.md` 的表 C 必须**同时包含 CUDA 与 Triton 两列**（同 shape、同 dtype、同布局），否则视为未完成。

---

## 15. vLLM(C++) 集成规格（向上集成，逐文件）

### 15.1 目标与边界

```text
目标仓库：$SCA_VLLM_ROOT     （project(vllm-cpp VERSION 0.1.0)）
目标库目标：vllm::vllm（STATIC） + 可执行 vllm_server
集成方式：本项目提供独立适配层目标 sci_attention_vllm_adapter，链接 vllm::vllm 与本项目
硬边界：不修改上游 src/attention/*；如需 hook，只提供补丁文件并由用户决定是否应用
```

### 15.2 上游接口快照（必须按当前真实代码核对，不能凭本提示词臆测）

```cpp
// include/vllm/attention/attention_backend.hpp
enum class AttentionBackend { kPagedAttention, kFlashAttention, kAuto };

// include/vllm/attention/flash_attention.hpp
struct FlashAttentionParams {
    float* output;             // [batch, seq_len, num_heads, head_dim]    ← FP32, BSHD
    const float* query;        // 同布局
    const float* key;          // [batch, seq_len, num_kv_heads, head_dim]
    const float* value;        // 同上
    int32_t batch, seq_len_q, seq_len_kv, num_heads, num_kv_heads, head_dim;
    float scale;
    bool is_causal;
    cudaStream_t stream;
};
void flash_attention_forward(const FlashAttentionParams& params);   // 现有实现：81 行单 warp 原型

// include/vllm/memory/kv_cache.hpp
class KVCache {  // 每 layer 连续分配，[num_blocks, block_size, num_kv_heads, head_dim]
  public: void* k_ptr(size_t layer, size_t physical_block);
          void* v_ptr(size_t layer, size_t physical_block);
          void clear_block(size_t layer, size_t physical_block);
          size_t total_bytes() const;
};
```

Phase 0 必须复核：上述签名、`src/attention/flash_attention.cu` 当前实现细节、`include/vllm/core/block_manager.hpp` 是否已提供可复用的 block 分配（若已有，适配层复用其分配结果）。

### 15.3 适配层逐文件规格

| 文件                                                   | 必须内容                                                                                                                                                                                                        |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vllm_backend/CMakeLists.txt`                        | `add_library(sci_attention_vllm_adapter STATIC ...)`；`target_link_libraries(... PUBLIC vllm::vllm sci_attention)`；要求 `SCI_ATTENTION_VLLM_ROOT` 存在且包含 `include/vllm/vllm.hpp`；不修改上游 CMake                    |
| `vllm_backend/adapter_config.hpp`                    | `struct AdapterConfig { ComputeDtype dtype{kBFloat16}; bool allow_fp32_reference{true}; bool fail_fast{true}; int64_t min_seq_len_for_flash{16}; };` + 环境变量解析 `SCA_VLLM_COMPUTE_DTYPE`、`SCA_VLLM_FAIL_FAST` |
| `vllm_backend/layout_bridge.hpp`                     | `Result<sci::Tensor> WrapBshdAsBhsd(const float* ptr, ...)`（零拷贝视图 + stride 计算）；`Result<std::vector<half/bfloat16>> CastFp32To(...)`（显式转换，附性能与误差说明）；**所有转换必须可单测**                                            |
| `vllm_backend/sca_vllm_adapter.{hpp,cpp}`            | 总入口：`bool sca_flash_attention_forward(const vllm::FlashAttentionParams&, std::string* err)`；内部流程 = 校验 → 转换/包装 → 调 `sca::flash_attention` → 转回 FP32 写回 `params.output`                                       |
| `vllm_backend/sca_flash_attention_adapter.{hpp,cpp}` | 与上游 `flash_attention_forward` 对齐的 drop-in 版本：`void flash_attention_forward_via_sca(const vllm::FlashAttentionParams&)`；失败时按 `fail_fast` 决定 abort/回退并 `LOG(ERROR)`                                           |
| `vllm_backend/sca_paged_attention_adapter.{hpp,cpp}` | 把 `vllm::KVCache` 的 `(layer, physical_block)` 指针映射为 `sca::PagedKVCache` 的 block 视图；提供 `MapVllmKvToSca()`；如上游 block 分配缺失，则用 `sca::BlockManager` 生成 page table                                                |
| `vllm_backend/patches/0001-attention-dispatch.patch` | 可选：在 `vllm::AttentionBackend` 分派点插入 `SCA_VLLM_BACKEND=sca                                                                                                                                                   |
| `vllm_backend/README.md`                             | 三步集成：配置 → 编译适配层 → A/B 运行；含开关、回滚、已知精度影响                                                                                                                                                                      |

### 15.4 精度与布局迁移规则

| 维度     | 上游                   | 本项目                  | 规则                                                             |
| ------ | -------------------- | -------------------- | -------------------------------------------------------------- |
| dtype  | 全 FP32               | FP16/BF16 计算，FP32 累加 | 输入 FP32 → 转换成计算 dtype（一次显式 kernel/循环），输出转回 FP32；转换吞吐与误差必须测量并记录 |
| layout | BSHD（`[B,S,H,D]`）    | 内部 BHSD 优先           | 视图包装优先（stride 传递），无法包装时执行一次显式 transpose kernel；禁止在 kernel 内散访存 |
| GQA    | `num_kv_heads` 已在参数中 | 支持 H_q % H_kv == 0   | 映射 `h_kv = h_q / group_size`，单测覆盖 group_size ∈ {1,2,4,8}       |
| scale  | 上游传入                 | `cfg.scale` 直接使用     | 若上游传 `scale <= 0` 则按 `1/sqrt(D)` 兜底并打日志                        |
| causal | `is_causal`          | `cfg.causal`         | 直传；不允许忽略                                                       |

**必须写进 `docs/vllm_integration.md` 的声明**：SCA 路径默认使用 BF16 计算，与上游 FP32 原型相比存在量化误差（量级见 §11.4），若业务需要 FP32 精度必须显式选择 `allow_fp32_reference=true` 的回退路径。

### 15.5 A/B 验证设计（`benchmarks/benchmark_vllm.cpp`）

```text
固定输入：B=1,H=16,H_kv=8,S=1024,D=128,causal=true（+ 2 组其它 shape）
路线 A：vllm::flash_attention_forward（上游原型）
路线 B：sca::flash_attention（经适配层，含 dtype 转换开销与不含转换开销两种口径）
输出表：
  | 路线 | P50 (ms) | P99 (ms) | TFLOPS | 与 FP32 参考最大误差 | 备注 |
同时报告：转换开销占比、显存占用差异、GQA 正确性（上游原型在此 shape 下的误差会暴露其缺陷）
```

要求：如果不具备直接链接上游库的条件（上游构建失败/缺依赖），必须在报告中写「未完成 + 原因 + 复现命令」，禁止伪造数据。

### 15.6 集成开关与回滚

```text
CMake: -DSCI_ATTENTION_BUILD_VLLM_ADAPTER=ON -DSCI_ATTENTION_VLLM_ROOT=../vllm
运行期: SCA_VLLM_BACKEND=sca|legacy（默认 sca）、SCA_VLLM_COMPUTE_DTYPE=bf16|fp16、SCA_VLLM_FAIL_FAST=0|1
回滚: 设置 SCA_VLLM_BACKEND=legacy 即可回到上游实现；补丁可 git apply -R 撤销
```

### 15.7 上游缺陷清单（必须带文件:行号证据）

```text
[ ] src/attention/flash_attention.cu：out 未初始化即累加（当前实现直接 `out[...] += ...`）
[ ] 同上：两次遍历 K/V（先统计 max/l，再算输出），未实现真正 online softmax
[ ] 同上：单 warp（block=32）处理整行，无 tiling / 无 MMA
[ ] 同上：GQA 用 `h % num_kv_heads` 映射，H_q 与 H_kv 不成组时语义错误
[ ] 同上：仅支持 FP32，无 FP16/BF16 路径
[ ] 性能：S=1024 时延迟与 SDPA 的差距（必须实测给数字）
```

### 15.8 关于 Python vLLM（当前环境不可用）

```text
事实：本机未安装 Python vLLM（import vllm 失败）；因此本阶段不实现 Python 版 vLLM 后端。
要求：在 docs/vllm_integration.md 中留「未来接入设计」小节，写清需要检查的三件事：
  1) 目标 vLLM 版本的 attention backend 接口（不同版本差异大）
  2) KV cache 布局（block_size、num_kv_heads、dtype、是否 fp8）
  3) 注册方式（backend 注册表 / 环境变量选择），以及版本升级时的兼容策略
禁止：声称「已支持 Python vLLM」而没有实际安装与验证。
```

---

## 16. RLHF / Rollout 集成规格

### 16.1 目标

为 rollout 形态提供**最小但真实可用**的推理接口：批量 prefill + 连续 decode + KV 复用，并能量化吞吐。**不实现 PPO/GRPO 训练逻辑**。

### 16.2 上游现状（只读，必须写入 `docs/rollout_interface.md`）

```text
仓库：$SCA_RLHF_ROOT   project(mini-rlhf-stack VERSION 0.1.0)
目标：rlhf_core（STATIC）、rlhf_cuda（STATIC）、rollout_server（EXE）、train_ppo（EXE）
接口：
  rlhf::RolloutServer { RolloutResult serve(const std::vector<std::string>& prompts);
                        void update_weights(const std::unordered_map<std::string, std::vector<float>>&);
                        bool load_checkpoint(const std::string& tag, const std::string& dir); }
  rlhf::GPUActor     { RolloutResult rollout(const std::vector<std::string>& prompts);
                        GPUActor(shared_ptr<GPUPolicyModel>, SamplingConfig, int max_seq_len = 2048); }
  rlhf::KVCache      { KVCache(num_layers, num_heads, head_dim, max_seq_len);
                        append(layer, k, v); get(layer, seq_len, k_out, v_out); clear(); }
                       // 存储：CPU std::vector<std::vector<std::vector<float>>>，注释布局
                       // [num_layers, 2, batch, num_heads, seq_len, head_dim]
缺陷（登记用）：
  - CMake 的 CMAKE_CUDA_ARCHITECTURES = "80;86;89;90"，不含 120（本机需覆盖才能原生运行）
  - rlhf_cuda 源列表未包含 rlhf/cuda_ops/gpu_ops.cu（即使存在 GPU 实现也不会被编译）
  - GPU 路径缺少 attention kernel（gpu_policy_model.cpp 无 attention 计算）
```

### 16.3 本项目必须提供的接口（供 rollout 使用）

```text
[ ] 批量 prefill：flash_attention（支持 varlen cu_seqlens）
[ ] 连续 decode：decode_attention + 每步 KV append
[ ] KV 复用：PagedKVCache（append / reset / stats），支持多序列并发
[ ] 显存预算：KV 上限检查与错误返回（不得 OOM 崩溃）
[ ] C++ 与 Python 双入口（C++ 供 rollout_server 直接链接；Python 供快速实验）
```

### 16.4 `examples/rollout_engine_stub.cpp`（必须可运行）

```text
流程：
  1) 构造 N 个请求（不同 prompt 长度，模拟 batch 内长度差异）
  2) prefill：varlen flash_attention 一次前向
  3) decode 循环：每步 1 token，KV append 复用；达到 max_new_tokens 或遇到 EOS 停止
  4) 统计并打印：prefill P50、decode 每步 P50/P99、tokens/s、KV 占用峰值、平均 batch 利用率
要求：
  - 使用真实 kernel（不允许用假循环），无 GPU 时给出明确提示并退出码 2
  - 结果通过 --json 输出，字段与 §12.4 保持同一套命名
  - 配置必须采用真实模型档（§23）：H_q=32、H_kv=8、D=128、causal=True、层数 36，
    KV 长度上限按 §23.2 的显存预算裁定（默认 4096，允许参数覆盖但必须显存检查）
  - 可选：与 Ollama 服务基线（tools/ollama_baseline.py）对比，但必须标注「完整模型 vs 注意力层」口径差异
```

### 16.5 与 vLLM 的关系（两条可选路径，必须都写进文档）

```text
路径 1（服务化）：RLHF → vLLM(C++) serve → SciCompute-Attention 后端（§15 适配层）
路径 2（直连）：RLHF → 直接调用 sca::* API（跳过 vLLM，适合实验/教学）
文档必须给出两条路径的取舍：功能（调度/批处理/服务化 vs 简单直接）、性能开销、维护成本
```

### 16.6 指标定义（必须与 benchmark 一致）

```text
tokens/s            = 生成 token 总数 / 端到端时间（含 prefill 与 decode）
prefill_latency     = 单次 prefill 的 P50/P99
decode_step_latency = 单步 decode 的 P50/P99
kv_reuse_ratio      = 复用 KV 的 token 数 / 总 token 数
peak_kv_bytes       = KV Cache 峰值占用
batch_utilization   = 实际参与计算的序列数 / max_num_seqs
```

---

## 17. 与 SciComputeInfra 的协同与回馈

### 17.1 版本锚点（Phase 0 与每次交付前都要更新）

在 `docs/env_report.md` 与 `upstream/notes.md` 记录：

```bash
for repo in SciComputeInfra SciCompute-Attention vllm RLHF; do
  printf '%-22s %s\n' "$repo" "$(git -C $HOME/Workspace/Code/Project/RL_infra/$repo rev-parse HEAD 2>/dev/null || echo NA)"
done
```

### 17.2 上游问题登记（`upstream/notes.md` 格式）

```markdown
### UP-001: src/bridges/CMakeLists.txt 硬编码 CUDA include 路径
- 证据：`src/bridges/CMakeLists.txt:19` → `target_include_directories(sci_bridges PUBLIC /usr/local/cuda-13.2/include)`
- 影响：CUDA 版本升级或非默认安装路径时构建失败
- 建议：改为 `CUDA::cudart` 传递的 include 目录（`find_package(CUDAToolkit)` 已在上层执行）
- 本项目回避方式：不修改上游；本项目 CMake 不依赖该硬编码（使用 `CUDA::cudart`）
- 状态：待用户确认后可提 patch
```

至少登记以下条目（其余按 Phase 0 实况补充）：

```text
UP-001  src/bridges/CMakeLists.txt 硬编码 CUDA include 路径
UP-002  cuda/kernels/attention.cuh 只有声明无实现，且未加入任何 CMake 目标（易误认为已有能力）
UP-003  cuda/helpers/、cuda/launch/、src/kernel/、src/graph/、python/* 为空目录（目录存在但无内容）
UP-004  存在两套 GPU 抽象（sci::Tensor/Device 与 gpu::GpuTensor/GpuDType），边界不清
UP-005  上游无 Python 绑定（python/bindings 为空），本项目 Python 模块需自带封装
```

### 17.3 回馈上游的流程（默认不执行）

```text
1) 在 upstream/patches/ 生成补丁：git -C ../SciComputeInfra diff > upstream/patches/NNNN-<slug>.patch
2) 校验：git -C ../SciComputeInfra apply --check upstream/patches/NNNN-*.patch
3) 记录：upstream/notes.md 写明动机、影响面、回滚命令
4) 执行：必须由用户明确确认后再应用；本任务默认不触碰上游仓库
```

### 17.4 上游能力复用自查表（每个 Phase 结束自查）

```text
[ ] 新写的代码里是否重复实现了上游已有的 Tensor/Memory/Stream/Benchmark 能力？
[ ] 新引入的依赖是否必要（YAGNI）？是否有上游等价物？
[ ] 与上游的类型转换是否通过公共 API（而非 reinterpret_cast 硬转）？
[ ] 上游变更是否会破坏本项目（版本锚点是否变化）？
```

---

## 18. 文档规格（逐文件大纲）

### 18.1 文档总规则

```text
语言：正文中文；代码、命令、路径、指标名、错误码保留原文
数字：任何性能/精度数字必须标注「命令 + 数据文件 + 时间 + git sha」
图：优先用 ASCII 图（保证可 diff）；确有必要再用外部工具生成并提交源文件
一致性：同一指标在不同文档中必须同名同单位（单位统一 ms / GB/s / TFLOPS / MiB）
更新：代码变更导致数值变化的，必须同一次提交内更新对应文档
```

### 18.2 `README.md` 章节（必须齐全）

```text
1.  项目定位（含中文定位句与三条能力主线：科学计算 / Attention / LLM Infra）
2.  总体架构（LLM Infra Stack ASCII 图 + 三仓关系图）
3.  核心能力一览表（Level 0-5 × 状态 × 关键指标）
4.  快速开始（环境要求 / 配置 / 构建 / 测试 / 运行示例，含真实命令与预期输出）
5.  目录结构（与 §4 保持一致）
6.  Attention 数学原理（定义、复杂度、mask/scale 约定）
7.  FlashAttention 原理（tiling + online softmax + 为什么省 IO）
8.  Kernel 设计（tile 配置表、线程映射图、smem 布局图）
9.  Memory Hierarchy（HBM/L2/SMEM/寄存器 的实测数字与含义）
10. KV Cache（布局、生命周期、显存预算表）
11. Paged KV Cache（block table、与 vLLM 的对应关系）
12. Prefill / Decode（差异、阈值标定结论）
13. vLLM(C++) 集成（三步接入 + 开关 + A/B 数据 + 回滚）
14. RLHF / Rollout 集成（接口契约 + stub 运行方式 + 指标）
15. Benchmark（方法、矩阵、复现命令、结果表）
16. Nsight Profiling（脚本用法、指标含义、瓶颈判定结论）
17. Build / Test / Example（与快速开始呼应，含 Debug/Release/RelWithDebInfo）
18. 已知限制与 Roadmap（明确列出未实现项：prefix caching、fp8、多卡等）
19. 引用与致谢（FlashAttention 论文、vLLM 论文、CUTLASS 等，若使用需标注边界）
```

### 18.3 `docs/` 逐文件必须包含的小节

| 文档                         | 必须包含                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| `00-recon.md`              | 上游资产表（文件:行号）、复用/重写判定、风险、待确认项                                                                           |
| `env_report.md`            | 环境表、ISA 探针原始输出、三仓 git sha、与 §1 的差异说明                                                                   |
| `architecture.md`          | 分层图、模块图、数据流、文件职责表、依赖方向、扩展点                                                                             |
| `attention_math.md`        | 定义式、复杂度推导、scale/softcap/滑窗约定、LSE 定义                                                                    |
| `online_softmax.md`        | 推导（含 alpha/beta）、等价性证明、误差分析、极端用例实测                                                                     |
| `flash_attention.md`       | 算法伪代码、tile 选择理由、流水线示意、与 SDPA 的差异与数据                                                                    |
| `kernel_design.md`         | 每个 kernel 的线程映射图、smem 分区表、寄存器表、同步点、同步检查命令结果                                                            |
| `tile_config.md`           | 标定方法、扫描结果表、最终配置与理由（含被否决配置）                                                                             |
| `memory_hierarchy.md`      | 层次图、实测带宽/延迟数字、bank conflict 数据                                                                         |
| `kv_cache.md`              | 布局图、地址公式与示例计算、显存预算表、生命周期时序                                                                             |
| `paged_kv_cache.md`        | 分页设计、block table 结构、与 vLLM KV 的对应、实测影响                                                                 |
| `prefill_decode.md`        | 差异分析、阈值标定曲线、调度建议、实测数据                                                                                  |
| `benchmarking.md`          | 方法论、矩阵、JSON schema、公平性规则、结果解读、复现命令                                                                     |
| `profiling.md`             | ncu/nsys 用法、指标清单与含义、四种瓶颈判定标准、实测结论                                                                      |
| `roofline.md`              | §14 的三张表 + Q1–Q9 的逐条回答                                                                                 |
| `numerical_stability.md`   | 容差表来源、极端用例数据、溢出/下溢分析、与 torch 的对比                                                                       |
| `triton_optimization.md`   | Triton 轨道定位与边界、逐 kernel 设计、autotune 方法与配置表、三方对比、Triton/CUDA 取舍结论、复现命令                                  |
| `vllm_integration.md`      | 上游接口快照、适配层结构、dtype/layout 迁移、A/B 数据、缺陷清单、Python vLLM 未来设计                                              |
| `rollout_interface.md`     | RLHF 现状（含缺陷）、两条集成路径、指标定义、stub 运行结果                                                                     |
| `model_integration.md`     | Qwen3-4B-Thinking-2507-Q8 事实表、GGUF 读取与 Q8_0 反量化、KV 预算表（§23.2）、真实 shape 数据、Ollama 基线对照、显存共存规则、量化边界、复现命令 |
| `upstream_relationship.md` | 三仓关系图、版本锚点表、patch 清单、复用清单与边界                                                                           |
| `troubleshooting.md`       | 附录 D 故障树的完整版（含真实报错原文与修复命令）                                                                             |

### 18.4 文档与代码的一致性检查（必须脚本化）

```bash
# README 中出现的每个构建/测试命令都必须真实存在于 scripts/ 或 CMake 目标中
rg -n "scripts/|build/benchmarks/|build/tests/" README.md docs/ | wc -l
# 文档中引用的结果文件必须存在（不存在即视为未完成）
ls docs/results/*.md
```

---

## 19. 分阶段执行计划（Phase 0–13，含 Phase 6.5 Triton 轨道与 Phase 12.5 真实模型对接）

> 每个 Phase 的固定闭环：`配置 → 编译 → 单测 → 正确性 → benchmark → 记录(.agent/state.md) → commit`。
> 未通过验证的 Phase 不允许开始下一个；被阻塞时必须在 `.agent/failures.md` 写清「现象 / 已尝试 / 结论 / 下一步」。

### Phase 0：审计上游（SciComputeInfra + vLLM + RLHF）

```text
任务：§3 的全部命令与必读文件；产出 docs/00-recon.md、docs/env_report.md、TASK.md、.agent/*
DoD：
  [ ] 三仓 git sha 已记录
  [ ] ISA 探针复跑结果与 §1.2 一致（或明确列出差异）
  [ ] 复用矩阵逐条给出证据（文件:行号）
  [ ] 上游缺陷登记 ≥ 5 条（§17.2）
验证：cat docs/00-recon.md / docs/env_report.md
commit: "docs: phase 0 recon and environment report"
```

### Phase 1：建立工程骨架与 Attention API

```text
任务：§5 构建文件 + §6 全部公共头文件（实现可为 not-implemented）+ §9 的 capability/workspace 骨架
DoD：
  [ ] 空实现也能编译通过（含 STUB 模式）
  [ ] tests/unit/test_config.cpp、test_status.cpp、test_capability.cpp 全绿
  [ ] README/build 命令可跑通
验证：bash scripts/configure.sh --build-type Release && bash scripts/build.sh -j"$(nproc)" && ctest --test-dir build --output-on-failure
commit: "feat: add attention api surface and build skeleton"
```

### Phase 2：Naive Attention（Level 0）

```text
任务：§7.2；tests/kernel/test_naive_correctness.cu；benchmark_naive.cu
DoD：
  [ ] fp32/fp16/bf16 × causal/非 causal 全部通过容差
  [ ] S 缓冲峰值显存已记录（作为 IO 分析基线）
  [ ] 生成 docs/results/benchmark_report.md 的第一版（naive 一列）
验证：ctest -R naive --output-on-failure
commit: "feat: add naive attention reference kernels"
```

### Phase 3：Tiled Attention（Level 1）

```text
任务：§7.3；test_tiled_correctness.cu；benchmark_tiled.cu；docs/kernel_design.md（tiled 部分）
DoD：
  [ ] 与 naive 同输入误差在容差内
  [ ] 显存峰值显著低于 naive（给出倍数与命令）
  [ ] compute-sanitizer memcheck 零错误
commit: "feat: add tiled attention kernel with shared-memory staging"
```

### Phase 4：Online Softmax（独立可测）

```text
任务：§7.4.2；tests/kernel/test_numerics.cu；docs/online_softmax.md
DoD：
  [ ] 单块与前缀块两种更新路径与「两遍精确参考」一致
  [ ] 极端 logits（±1e4）无 NaN/Inf，误差在容差内
  [ ] 文档给出推导与误差分析
commit: "feat: implement and validate online softmax"
```

### Phase 5：FlashAttention Kernel（Level 3）

```text
任务：§7.4 全部文件；test_flash_correctness.cu / test_flash_causal.cu / test_head_dims.cu / test_layout.cu
DoD：
  [ ] D=64/128/256 × fp16/bf16 × causal/非 causal 全绿
  [ ] 未物化 N×N（workspace 字节数证明）
  [ ] tile 配置表来自 tile_sweep 实测（docs/tile_config.md）
  [ ] compute-sanitizer racecheck 零错误
commit: "feat: add flash attention forward kernel"
```

### Phase 6：Benchmark + Nsight（核心数据）

```text
任务：§12 全部基准文件 + §13 全部脚本；docs/benchmarking.md、docs/profiling.md
DoD：
  [ ] main 矩阵跑通，JSON/CSV 落盘，SDPA 对照数据齐全
  [ ] 至少 3 个 kernel 有 ncu 报告与瓶颈判定结论
  [ ] docs/results/benchmark_report.md 含延迟/TFLOPS/带宽/显存四张表
commit: "feat: add benchmark suite and profiling pipeline"
```

### Phase 6.5：Triton 算子优化轨道（与 CUDA 双轨对照）

```text
任务：§7.9 全部文件 + benchmarks/benchmark_triton.py + profiling/profile_triton.sh
      + python/tests/test_triton_parity.py + python/tests/test_triton_config.py
      + docs/triton_optimization.md + docs/results/triton_vs_cuda.md
说明：本 Phase 只依赖 torch + triton（不需要 pybind11）；因此先创建
      python/scicompute_attention/{__init__.py, dispatch.py, triton_kernels/*}，
      ops.py 的 `_core` 导入必须做延迟/可选处理（`_core` 缺失时 cuda 后端报明确错误，triton 后端仍可用）。
DoD：
  [ ] Triton prefill / decode / paged kernel 全部与 CUDA kernel、SDPA 三方一致（容差同 §11.4）
  [ ] autotune 扫描完成，best_configs.json 生成且可复现（含 cache key 与配置表）
  [ ] 配置校验拦截 smem > 101,376 B 的组合（有测试用例证明拦截生效）
  [ ] 报告 kernel-only 与端到端两组延迟，并解释与 CUDA 的差距来源（有 ncu/nsys 证据）
  [ ] 明确写出「哪些 shape 用 Triton 更合适、哪些必须用 CUDA」的结论
验证：conda run -n cuda_132 python -m pytest python/tests/test_triton_parity.py -v
      python benchmarks/benchmark_triton.py --suite main --json benchmarks/results/<date>-<sha>/triton.json
commit: "feat: add triton attention kernels with autotuning"
```

### Phase 7：KV Cache（Level 4 前置）

```text
任务：§8 全部文件；tests/kv/*
DoD：
  [ ] 全部 §8.7 用例通过
  [ ] 显存预算表用本机 cudaMemGetInfo 复算
  [ ] docs/kv_cache.md 完成
commit: "feat: add paged-ready kv cache with block manager"
```

### Phase 8：Decode Attention（Level 4）

```text
任务：§7.5；test_decode_correctness.cu；benchmark_decode.cu；docs/prefill_decode.md
DoD：
  [ ] split-K ∈ {1,2,3,4,8,16} 全部正确
  [ ] 与 flash（seq_q=1）对比给出阈值结论，并写入 dispatch_table.inc
  [ ] decode 的瓶颈判定（memory-bound）有 ncu 证据
commit: "feat: add split-k decode attention and threshold calibration"
```

### Phase 9：Paged KV Cache / Paged Attention（Level 5）

```text
任务：§6.8/§6.9/§7.6/§8.5-8.6；test_paged_correctness.cu
DoD：
  [ ] 乱序 page、block_size ∈ {1,8,16,32}、GQA 全通过
  [ ] page table 访存开销已量化（<2% 目标）
  [ ] docs/paged_kv_cache.md 完成（含与 vLLM KV 的对应）
commit: "feat: add paged kv cache and paged attention"
```

### Phase 10：Python Binding

```text
任务：§10 全部文件
说明：Triton 相关的 Python 文件已在 Phase 6.5 建立；本 Phase 补齐 `_core`（pybind11）绑定、
      ops.py 的 CUDA 路由、kv_cache.py、benchmark.py 与全部 pytest。
DoD：
  [ ] pip install -e python 成功；import + 4 个 backend 调用成功
  [ ] python/tests 全绿；与 torch SDPA 的误差在容差内
  [ ] zero-copy 已验证（data_ptr 与输出一致性）
commit: "feat: add python bindings and parity tests"
```

### Phase 11：vLLM(C++) Adapter

```text
任务：§15 全部文件
DoD：
  [ ] 适配层编译通过（-DSCI_ATTENTION_BUILD_VLLM_ADAPTER=ON）
  [ ] A/B 数据落盘 docs/results/vllm_ab.md（含上游缺陷实证）
  [ ] 回滚路径验证：SCA_VLLM_BACKEND=legacy 可恢复上游行为
commit: "feat: add vllm c++ adapter with a/b harness"
```

### Phase 12：RLHF / Rollout 接口

```text
任务：§16；examples/rollout_engine_stub.cpp；docs/rollout_interface.md
DoD：
  [ ] stub 跑通 prefill+decode，输出 tokens/s 与 KV 峰值
  [ ] 两条集成路径（经 vLLM / 直连）均有说明与命令
  [ ] RLHF 侧缺陷与迁移建议写入文档
commit: "feat: add rollout-oriented interface and example"
```

### Phase 12.5：真实模型对接（Qwen3-4B-Thinking-2507-Q8）

```text
任务：§23 全部文件（tools/gguf_reader.py、model_probe.py、dump_qwen3_qkv.py、ollama_baseline.py、
      benchmarks/benchmark_qwen3_shapes.py、examples/example_qwen3_attention.py、
      tests/python/test_qwen3_{shapes,parity}.py、docs/model_integration.md、docs/results/qwen3_4b_shapes.json）
DoD：
  [ ] GGUF 元数据解析结果与 §23.1 完全一致（36/32/8/128/2560/5e6/262144）
  [ ] 逐层反量化峰值显存 ≤ 1 GiB，且打印前后 cudaMemGetInfo
  [ ] 真实 Q/K/V 上三方 parity 通过（causal + GQA group=4，容差同 §11.4）
  [ ] 真实 shape benchmark（prefill ≤4096 / decode ≤8192）落盘，并含显存与 ollama_running 标记
  [ ] Ollama 基线采集完成，文档明确「注意力层 vs 完整模型服务」边界
验证：
  conda run -n cuda_132 python tools/model_probe.py --gguf $SCA_MODEL_DIR/Qwen3-4B-Thinking-2507-Q8_0.gguf
  conda run -n cuda_132 python tools/dump_qwen3_qkv.py --layers 0-3 --out tests/data/qwen3_4b/
  conda run -n cuda_132 python -m pytest python/tests/test_qwen3_parity.py -v
commit: "feat: integrate qwen3-4b gguf model shapes and weights"
```

### Phase 13：完整文档与收尾

```text
任务：§18 全部文档；docs/results/*；README 定稿；Known Issues；Roadmap
DoD：
  [ ] README 5 分钟可跑通（新环境验证：干净目录 clone + 构建 + 测试）
  [ ] 所有数字可追溯到结果文件
  [ ] §20 验收清单全绿
  [ ] .agent/memory.md 与 failures.md 已沉淀高价值结论
commit: "docs: complete documentation and performance report"
```

---

## 20. 验收清单（最终 Gate，必须逐条勾选）

### 20.1 功能与正确性

```text
[ ] 自研 CUDA Attention kernel（非任何第三方 wrapper）
[ ] 自研 online softmax（含推导文档与极端用例数据）
[ ] FlashAttention-style IO-aware 执行，未物化 N×N（有 workspace 字节数证据）
[ ] 支持 FP16 / BF16（FP32 作为参考路径）
[ ] 支持 causal 与非 causal 两条路径
[ ] 支持 head_dim ∈ {32,64,96,128,160,192,256}
[ ] 支持 Prefill（含 varlen）与 Decode
[ ] KV Cache 完整生命周期（append/read/reset/stats）
[ ] Paged KV Cache + Paged Attention（乱序 page 正确）
[ ] GQA/MQA 支持（group_size 1/2/4/8 均有用例）
[ ] 数值稳定性：极端 logits 无 NaN/Inf，误差在容差表内
[ ] compute-sanitizer memcheck/racecheck/initcheck 零错误
[ ] Triton kernel 与 CUDA kernel、SDPA 三方一致（同一容差表，覆盖 causal/非 causal 与 D=64/128/256）
[ ] Triton autotune 配置合法性校验生效（smem > 101,376 B 的组合被剔除）
[ ] 真实模型对接：GGUF 元数据解析一致、Q8_0 反量化正确、真实 Q/K/V 三方 parity 通过
[ ] KV 预算表与本机显存复算一致（144 KiB/token × 36 层，3072 token 量级可行、32K 不可行）
```

### 20.2 接口与集成

```text
[ ] C++ 公共 API（include/scicompute_attention/*）稳定、自包含、无异常
[ ] Python API（pip 安装可用，与 torch 零拷贝互操作）
[ ] Dispatcher + Explain()：选择可解释、阈值来自实测
[ ] vLLM(C++) 适配层编译通过 + A/B 数据落盘 + 回滚开关可用
[ ] RLHF/Rollout 接口可用（stub 跑通并输出 tokens/s / KV 峰值）
[ ] 与 SciComputeInfra 的复用关系文档化（含版本锚点与上游问题登记）
```

### 20.3 性能与可观测

```text
[ ] Benchmark 体系完整（naive/tiled/flash/decode/paged/sdpa/vllm）
[ ] JSON/CSV/Markdown 三格式结果落盘并绑定 git sha
[ ] warmup 20 + measure 100 + P50/P90/P95/P99 全部记录
[ ] Nsight Compute 报告 ≥ 3 个 kernel，且给出一句话瓶颈判定
[ ] Nsight Systems 时序报告：证明热路径无非必要同步与拷贝
[ ] Roofline/IO 分析（docs/roofline.md）回答 Q1–Q9，且结论有数字
[ ] Triton vs CUDA 对比数据落盘（kernel-only + 端到端两组延迟 + 最优配置），并解释差距来源
[ ] 真实模型 shape 的 prefill/decode 数据落盘（benchmark_qwen3_shapes.py），并与 Ollama 服务基线对照
[ ] 显存预算表与本机 cudaMemGetInfo 复算一致
```

### 20.4 工程质量

```text
[ ] C++17(CUDA)/C++20(host)、RAII、const 正确、最小全局状态
[ ] 关键 kernel 英文注释 + 公式注释 + 线程/tile/smem/寄存器说明
[ ] 无 cudaDeviceSynchronize 出现在推理热路径（有检查命令与结果）
[ ] 无 per-token cudaMalloc / cudaFree
[ ] 所有 CUDA 调用与 kernel launch 均有错误检查
[ ] 构建零警告（WERROR=ON 时通过）
[ ] 每 Phase 独立 commit，历史可 bisect
[ ] README 5 分钟可跑通；docs 与代码一致
[ ] Known Issues 与 Roadmap 已列出（含未实现项与原因）
```

---

## 21. 禁止事项与反模式

### 21.1 绝对禁止（违反即视为任务失败）

```text
1. 直接把现成 FlashAttention / PagedAttention 项目复制改名
2. 仅调用 torch.nn.functional.scaled_dot_product_attention 冒充实现（它只允许作为 baseline）
3. 只写 Python wrapper，没有自己的 CUDA kernel
4. 只有 benchmark 没有正确性测试，或只有 kernel 没有测试
5. 只有性能数字没有 profiling 证据
6. 过早修改 vLLM(C++) 大量源码（只能通过 vllm_backend/ 适配层 + 可选 patch）
7. 重复实现 SciComputeInfra 已有的 Tensor/Memory/Stream/Benchmark 能力
8. 到处使用 cudaDeviceSynchronize / cudaStreamSynchronize（热路径零同步）
9. 把所有逻辑塞进一个 .cu 文件
10. 使用 wgmma / tcgen05（本机 sm_120 不支持，ptxas 实测拒绝）
11. 硬编码 GPU 架构（如写死 sm_120 或 /usr/local/cuda-13.2）
12. 把 CUTLASS 作为核心实现（若使用，必须限定在明确边界并写进文档）
13. 伪造、外推或 cherry-pick benchmark 数据；跳过失败的 case 而不记录
14. 未经用户确认就修改 SciComputeInfra / vllm / RLHF 三个上游仓库
15. 在未通过 Stage/Phase 验证的情况下宣告完成
16. 用 Triton 替代 CUDA 主实现（Triton 是第二轨道，CUDA kernel 仍是必须交付项）
17. 用 Triton 实现 C++/vLLM(C++) 路径（Triton 仅在 Python 侧；跨语言路径必须走 CUDA kernel）
18. 用 torch SDPA 或 triton.testing.do_bench 的数据冒充「本项目 Triton kernel」的性能数据
19. 把 autotune 的首次编译时间混入 kernel 计时（必须预热并单独记录编译/搜索开销）
20. 一次性反量化整个 Q8_0 模型（fp16 约 8.04 GB，必然 OOM；必须逐层流式处理）
21. 把 Q8_0 原始字节直接送入 fp16/bf16 kernel，或在未反量化的张量上做 parity
22. 把 Ollama 的端到端服务数字当作本项目 kernel 的性能结论（两者口径不同）
23. 在 Ollama 已占 4882 MiB 显存的情况下运行 SCA 显存敏感 benchmark 并输出「有效」数据
```

### 21.2 反模式（出现即需在评审中整改）

```text
- tile 参数散落在多个文件、用魔法数硬编码
- 用 atomicAdd 做行归约
- 在 kernel 内做 device 侧动态分配
- 用「回退到最慢后端」掩盖不支持路径（默认 allow_fallback=false 的意义所在）
- 用 C++ 异常跨 API 边界传递 CUDA 错误
- 用双缓冲/流水线之前就宣称优化完成（先测量再优化）
- 文档中写「大约快 2 倍」而不给数值与命令
```

---

## 22. 交付与回复规范

### 22.1 最终交付物

```text
Source Code      : 本仓库全部源码（C++/CUDA/Python）
Triton Kernels   : python/scicompute_attention/triton_kernels/*（含 best_configs.json 与 autotune 证据）
Model Assets     : tools/gguf_reader.py、tools/model_probe.py、tools/dump_qwen3_qkv.py、
                   tests/data/qwen3_4b/（manifest）、docs/results/qwen3_4b_shapes.json、docs/model_integration.md
Tests            : tests/ + python/tests（L1–L6）
Benchmarks       : benchmarks/ + benchmarks/results/<date>-<sha>/*.json|csv
Profiling        : profiling/reports/*（摘要 md 提交；原始 rep 文件 gitignore）
Docs             : README.md + docs/*.md + docs/results/*.md
Upstream Notes   : upstream/notes.md（含三仓版本锚点与问题清单）
Known Issues     : README「已知限制」+ docs/troubleshooting.md
Roadmap          : prefix caching、fp8、多卡、Python vLLM 适配等
Process Records  : TASK.md + .agent/{state,decisions,memory,failures}.md
```

### 22.2 最终回复格式（必须包含）

```text
1) 交付摘要：一句话定位 + 三条能力主线的完成情况
2) 验证证据：构建/测试/benchmark 的关键命令与结果（含 exit code、P50 数字）
3) 性能表：flash vs SDPA vs vLLM 原型的对比（至少 3 个 shape）
4) 三仓关系实测：SciComputeInfra 复用点、vLLM A/B 结论、RLHF stub 指标
5) 未完成项与原因（禁止掩饰）
6) 复现入口：README 路径 + 三条命令
```

### 22.3 汇报纪律

```text
- 每个 Phase 结束汇报一次，格式见 §0.4；不允许「攒一大批再汇报」
- 失败必须立即上报（现象 + 已尝试 + 当前假设 + 需要的决策），不允许静默重试超过 3 次
- 需要用户决策的事项（例如是否应用上游 patch、是否放宽容差）必须显式提出，不自作决定
```

---

## 23. 真实模型对接：Qwen3-4B-Thinking-2507-Q8（GGUF / Ollama）

> 目的：把「合成张量 benchmark」升级为「真实模型 shape + 真实权重 + 真实服务基线」的验证，
> 让 Attention 子系统的结论对上层（vLLM(C++) / RLHF Rollout）具备直接参考价值。

### 23.1 模型资产事实（Phase 0 必须复核并写入 `docs/model_integration.md`）

| 项              | 值                                                                                              | 来源                                   |
| -------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------ |
| 目录             | `$SCA_MODEL_DIR`                                  | 目录清单                                 |
| 权重文件           | `Qwen3-4B-Thinking-2507-Q8_0.gguf`（4,280,404,960 B ≈ 4.28 GB）                                  | 目录清单                                 |
| sha256         | `012aa2736c32b7b74c3ca7b2da181b9e1d24a3973abf5510dee5590b27445440`                             | 模型 README                            |
| GGUF 版本 / 张量数  | version 3 / 398 tensors                                                                        | 模型 README                            |
| 架构             | `qwen3`，file_type 7（**Q8_0**），参数量 4,022,468,096                                                | GGUF 元数据 / Ollama API                |
| 层数             | `qwen3.block_count = 36`                                                                       | `/api/show` → `model_info`           |
| 注意力头           | `head_count = 32`，`head_count_kv = 8`（**GQA group = 4**）                                       | 同上                                   |
| head_dim       | `key_length = value_length = 128`                                                              | 同上                                   |
| hidden size    | `embedding_length = 2560`                                                                      | 同上                                   |
| 上下文            | `context_length = 262144`；本机 Ollama 运行时上下文 4096                                                | 同上 / 模型 README                       |
| RoPE / RMS eps | `rope.freq_base = 5e6`，`layer_norm_rms_epsilon = 1e-6`                                         | 同上                                   |
| 服务             | Ollama 用户级 server：`http://127.0.0.1:11435`，tag `qwen3:4b-thinking-2507-q8_0`，启动脚本 `./serve.sh` | 模型 README / `serve.sh`               |
| 服务显存占用         | 100% GPU，上下文 4096，VRAM 4882 MiB / 8151 MiB                                                     | 模型 README（Phase 0 用 `nvidia-smi` 复测） |
| Python 依赖可用性   | `transformers` / `safetensors` / `requests` 可用；**`gguf`、`llama_cpp` 未安装**                      | cuda_132 实测                          |

**由模型推导出的注意力配置**：

```text
B=1, H_q=32, H_kv=8, D=128, group_size=4, causal=True
结论：head_dim=128 落在本项目 3 个主力 head_dim 之一，group_size=4 覆盖 GQA 主路径
```

### 23.2 KV Cache 显存预算（本模型 × 本项目布局，fp16 KV）

```text
每 token 字节 = 2(K,V) × 36(layers) × 8(kv_heads) × 128(head_dim) × 2(fp16) = 147,456 B ≈ 144 KiB/token
```

| 上下文长度  | KV Cache 大小 | 8 GB 显存可行性                |
| ------ | ----------- | ------------------------- |
| 512    | ≈ 72 MiB    | 可行                        |
| 1024   | ≈ 144 MiB   | 可行                        |
| 2048   | ≈ 288 MiB   | 可行                        |
| 4096   | ≈ 576 MiB   | 可行（需先停 Ollama 服务）         |
| 8192   | ≈ 1.13 GiB  | 需停服务并限制 batch=1           |
| 16384  | ≈ 2.25 GiB  | 需停服务 + 严格显存预算             |
| 32768  | ≈ 4.5 GiB   | 与 4B 权重同机不可行（给出结论，不做无用尝试） |
| 262144 | ≈ 36 GiB    | 仅记录为设计上限                  |

**权重侧约束（必须写进文档）**：Q8_0 全量反量化为 fp16 约 8.04 GB，已超过本机 7.8 GiB 可用显存。
因此**禁止**一次性 dequantize 全模型，必须逐层流式处理，单次实验峰值显存 ≤ 1 GiB：

```text
每层 attention 相关权重（fp16 估算）：
  q_proj 2560×4096 + k_proj 2560×1024 + v_proj 2560×1024 + o_proj 4096×2560 ≈ 26.2M 参数 ≈ 52 MB/layer
36 层全量 ≈ 1.9 GB（仅在显式预算允许时才可整体驻留）
```

### 23.3 对接目标（四条，缺一不可）

```text
目标 1（Shape 对齐）：从 GGUF / Ollama 元数据推导注意力维度，生成真实 shape 档并写入 benchmark 矩阵
目标 2（权重级正确性）：Q8_0 → fp16/bf16 反量化后取真实 Q/K/V，做 CUDA / Triton / SDPA 三方 parity
目标 3（真实 shape 性能）：用 (H_q=32, H_kv=8, D=128, causal) 跑 prefill 与 decode benchmark
目标 4（服务基线对照）：采集本机 Ollama 的 TTFT / prefill / decode 指标，与本项目 rollout stub 对照，
                       并明确边界——SCA 交付的是注意力层，不是完整模型服务
```

### 23.4 逐文件规格

| 文件                                     | 必须内容                                                                                                                                                                                                                                                                 |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tools/gguf_reader.py`                 | 纯 Python GGUF v3 读取器（当前环境无 `gguf` 包）：解析 header、metadata KV（覆盖全部标量类型与数组类型）、tensor info 表；CLI 支持 `--meta`、`--list-tensors`、`--dump-tensor <name>`；支持 `Q8_0`（block = 2 B fp16 scale + 32×int8，34 B / 32 权重）与 `F16/F32` 反量化；遇到不支持的类型必须显式报错并打印类型名（禁止静默跳过）                 |
| `tools/model_probe.py`                 | 读取 GGUF 元数据 → 输出 `docs/results/qwen3_4b_shapes.json`：注意力维度、张量清单与字节数、KV 预算表（§23.2）、Ollama 状态（`/api/version`、`/api/ps` 显存）；并打印可用于 benchmark 的 shape 档建议                                                                                                                |
| `tools/dump_qwen3_qkv.py`              | 逐层流式反量化 `q_proj/k_proj/v_proj/o_proj`（含 RMSNorm 权重）→ 用 torch 前向得到真实 Q/K/V → 保存到 `tests/data/qwen3_4b/`（`.npy`/`.pt` + `manifest.json` 记录 sha256、shape、dtype、seed、峰值显存）；必须打印 `cudaMemGetInfo` 前后值，峰值 ≤ 1 GiB                                                          |
| `tools/ollama_baseline.py`             | 通过 HTTP API（`/api/generate`、`/api/ps`、`/api/version`）采集基线：prompt eval 时间、eval 时间、tokens/s、TTFT、VRAM；输出 `benchmarks/results/<date>-<sha>/ollama_baseline.json`；支持 `--host/--model/--max-tokens/--prompt-file`；服务未启动时给出明确提示与 `./serve.sh` 命令                           |
| `benchmarks/benchmark_qwen3_shapes.py` | 真实配置（B=1、H_q=32、H_kv=8、D=128、causal）下的 prefill（S ∈ {512,1024,2048,4096}）与 decode（S_kv ∈ {1024,4096,8192}；超过 8192 必须显式 `--allow-large-kv` 且先检查可用显存）；对照 CUDA / Triton / SDPA；JSON 字段与 §12.4 一致，额外含 `"model": "qwen3-4b-thinking-2507-q8_0"` 与 `"source": "real-shape"` |
| `examples/example_qwen3_attention.py`  | 加载 `tests/data/qwen3_4b/` 的真实 Q/K/V → 分别用 `sca.flash_attention`、`sca.triton_attention`、`torch SDPA` 计算 → 打印三方误差表与延迟表；可选调用 `tools/ollama_baseline.py` 打印服务基线                                                                                                          |
| `tests/python/test_qwen3_shapes.py`    | 断言 GGUF 元数据解析结果与 §23.1 一致（36 / 32 / 8 / 128 / 2560 / 5e6）；断言 KV 预算公式与 §23.2 一致；数据缺失时 `skip` 并打印生成命令                                                                                                                                                                  |
| `tests/python/test_qwen3_parity.py`    | 真实 Q/K/V 上的三方 parity（容差同 §11.4），覆盖 causal 与 GQA group=4，输出最大/平均误差                                                                                                                                                                                                    |
| `docs/model_integration.md`            | 模型事实表、GGUF 读取与反量化说明、KV 预算表、真实 shape 结果、Ollama 基线对照、显存共存规则、复现命令、局限（无 HF tokenizer 文件 → 分词走 GGUF vocab 或 Ollama 计数）                                                                                                                                                    |
| `tests/data/qwen3_4b/README.md`        | 数据来源与生成命令、sha256 清单、体积说明（大文件进 `.gitignore`，仅提交 manifest 与 README）                                                                                                                                                                                                    |

### 23.5 显存共存与实验顺序（8 GB 硬约束）

```text
实验顺序（每次运行前必须打印 nvidia-smi）：
  1) 先做 SCA 侧实验：停止 Ollama 服务，释放约 4882 MiB
  2) 再做 Ollama 基线采集：cd 模型目录 && ./serve.sh
  3) 禁止「两者同时压满显存」后运行 benchmark 并声明数据有效
命令：
  curl -s http://127.0.0.1:11435/api/ps
  $SCA_OLLAMA_BIN stop qwen3:4b-thinking-2507-q8_0
  cd $SCA_MODEL_DIR && ./serve.sh
记录要求：每条结果必须包含 "gpu_mem_free_before_mb" 与 "ollama_running": true/false
```

### 23.6 量化边界（必须写进 `docs/model_integration.md`）

```text
1) Q8_0 是权重量化，不是激活/KV 量化；本项目 v1 的注意力计算仍在 fp16/bf16 下进行
2) Q8_0 反量化本身引入误差（block=32 + fp16 scale）；parity 的参考基准必须是
   「同一份反量化后的 fp16/bf16 张量」用 torch fp32 计算的结果，而不是原始 GGUF 字节流
3) 禁止把 Q8_0 原始字节直接送入 fp16 kernel；禁止宣称「已支持 Q8_0 注意力」
4) GPU 侧 dequant-fused attention（Q8 权重直接参与计算）属于 Roadmap，不属于 v1 交付
5) Ollama 的数字只能作为「完整模型服务基线」，不得与本项目 kernel 数据混为一谈
```

### 23.7 验收要点

```text
[ ] docs/results/qwen3_4b_shapes.json 生成，且与 §23.1 / §23.2 数值一致
[ ] tests/data/qwen3_4b/ 内含真实 Q/K/V + manifest（sha256、shape、峰值显存）
[ ] 真实 Q/K/V 上完成 CUDA / Triton / SDPA 三方 parity（causal + GQA group=4）
[ ] benchmark_qwen3_shapes.py 产出 prefill（≤4096）与 decode（≤8192）数据，并写入
    docs/results/benchmark_report.md 的「真实模型」小节
[ ] ollama_baseline.json 落盘，文档明确「注意力层 vs 完整模型服务」的边界
[ ] 所有结果记录 ollama_running 与显存状态，不存在无效的并发占用数据
```

---

## 附录 A：SciComputeInfra API 快照（Phase 0 必须逐条复核）

```cpp
// include/tensor/tensor.hpp
class Tensor {
  Tensor(); Tensor(const TensorShape&, DType, Device&);
  Tensor(const TensorShape&, DType, Device&, void* external_data, bool owns_data = false);
  static Tensor Empty/Zeros/Ones/Full(...); static Tensor Rand/Randn(...);
  const TensorShape& shape() const; index_t ndims()/dim(i)/num_elements() const;
  DType dtype() const; Device& device() const; DeviceType device_type() const;
  void* data(); const void* data() const; size_t num_bytes() const;
  bool is_contiguous() const; Layout layout() const; std::vector<index_t> strides() const;
  TensorView view()/reshape()/slice()/transpose();   // 非拷贝
  Tensor clone() const; Tensor to(Device&) const; Result<Tensor> to(DeviceType, int = 0) const;
  void fill(const void*); fill_zero(); fill_ones();
  void copy_from(const Tensor&); void copy_from(const void*, size_t);
};

// include/device/stream.hpp
class Stream {
  static Stream Create(DeviceType, Priority = Priority::kNormal);
  static Stream& GetCurrent(); static void SetCurrent(const Stream&);
  void* handle() const; DeviceType device_type() const;
  void wait(Event&); void synchronize(); bool is_query_complete() const;
};   // 非拷贝；可移动

// include/math/softmax.hpp
Result<Tensor> softmax(const Tensor&, int axis = -1, Stream* = nullptr);
Result<Tensor> softmax_stable(const Tensor&, int axis = -1, Stream* = nullptr);
namespace ref { void softmax_f32(const float*, float*, size_t, size_t); }

// include/benchmark/benchmark_runner.hpp
class BenchmarkRunner {
  BenchmarkRunner& set_filter(...); set_output_format("json|csv|markdown"); set_output_file(...); set_min_time_ms(...);
  void add_case(std::shared_ptr<BenchmarkCase>); void run();
  void export_json/export_csv/export_markdown(const std::string&) const; void print_results() const;
};
double mean(const std::vector<double>&); double stddev(...); double percentile(const std::vector<double>&, double p);
double estimate_bandwidth_gbps(size_t bytes, double time_ms);

// include/core/status.hpp
enum class StatusCode : int32_t { kOk, kInvalidArgument, kNotImplemented, kOutOfMemory,
                                  kCudaError, kCudaNotAvailable, kCudaOutOfMemory, ... };
```

构建目标与选项：

```text
targets : sci_core / sci_memory / sci_tensor / sci_scheduler / sci_bridges(OBJECT) / sci_compute(INTERFACE)
options : SCI_BUILD_TESTS(OFF) SCI_BUILD_BENCHMARKS(OFF) SCI_ENABLE_CUDA(ON) SCI_ENABLE_OPENMP(ON)
arch    : SCI_CUDA_ARCHITECTURES 默认 "75;89;90;120"
```

**注意**：`cuda/kernels/attention.cuh` 的 `launch_flash_attention` / `launch_scaled_dot_product_attention` 只有声明、无实现、未参与构建。若本项目或上层代码直接引用，将出现链接错误——Phase 0 必须用 `nm`/`rg` 复核并在 `upstream/notes.md` 登记（UP-002）。

---

## 附录 B：本机 ISA / 设备探测记录（来源与复现方式）

### B.1 设备属性（`python3 -c "import torch;print(torch.cuda.get_device_properties(0))"`）

```text
name=NVIDIA GeForce RTX 5070 Laptop GPU  major=12 minor=0
total_memory=8177909760  multi_processor_count=36  L2_cache_size=33554432
shared_memory_per_block=49152  shared_memory_per_multiprocessor=102400
regs_per_multiprocessor=65536  max_threads_per_multi_processor=1536
memory_clock_rate=12001000 kHz  memory_bus_width=128  warp_size=32  max_threads_per_block=1024
```

### B.2 ISA 探针（`nvcc -std=c++17 -arch=sm_120 -c <probe>.cu`）

```text
[PASS] mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
[PASS] cp.async.cg.shared.global [..], [..], 16;  + commit_group + wait_group
[PASS] ldmatrix.sync.aligned.m8n8.x4.shared.b16
[PASS] cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes        (TMA 1D bulk)
[PASS] cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes (需 CUtensorMap)
[FAIL] wgmma.fence  →  ptxas: Instruction 'wgmma.fence' not supported on .target 'sm_120'
```

### B.2.1 Triton 探针（conda env `cuda_132`，triton 3.7.1 / torch 2.13.0+cu132 / Python 3.12.13）

```text
最小 FP16 FlashAttention 风格 kernel（BM=64, num_warps=4, 在线 softmax + causal）实测：
  S=512  D=128 B=4 causal=True  BM=64 warps=4 stages=1  → OK  max_abs_err vs SDPA = 0.00098
  S=512  D=128 B=4 causal=True  BM=32 warps=4 stages=2  → OK  max_abs_err = 0.00098
  S=512  D=128 B=4 causal=True  BM=64 warps=8 stages=2  → OK  max_abs_err = 0.00098
  S=1024 D=128 B=2 causal=False BM=64 warps=4 stages=1  → OK  max_abs_err = 0.00012
  S=2048 D=64  B=2 causal=True  BM=64 warps=4 stages=2  → OK  max_abs_err = 0.00098
  S=512  D=256 B=2 causal=True  BM=32 warps=4 stages=1  → OK  max_abs_err = 0.00098
  S=4096 D=64  B=1 causal=True  BM=128 warps=4 stages=2 → OK  max_abs_err = 0.00098
  S=512  D=128 B=4 causal=True  BM=64 warps=4 stages=2  → FAIL(OutOfResources):
        Required shared memory 106496 B > Hardware limit 101376 B

约束结论：
  1) sm_120 上 Triton 可用，`tl.dot` 走 mma.sync（与 §1.2 一致）
  2) Triton 配置必须按 101,376 B 上限校验 smem，否则编译期/启动期抛 OutOfResources
  3) `@triton.jit` kernel 必须定义在真实 .py 文件（否则 inspect 失败）
  4) 上述误差为「算法规模参考」，正式数据必须由本项目的 benchmark_triton.py 产出
```

### B.3 结论对实现的影响

```text
1) 主路径：mma.sync m16n8k16 + cp.async 多级流水 + ldmatrix
2) TMA 作为可选优化（SCI_ATTENTION_ENABLE_TMA=AUTO 时按运行期探测启用）
3) 禁止 wgmma / tcgen05 相关教程式照搬
4) tile 选择必须基于 smem 102400 B / regs 65536 / threads 1536 三条上限
5) Triton 轨道第二实现：配置上限按 101,376 B，与 CUDA 主实现三方互检（§7.9）
```

---

## 附录 C：配置与容差速查表

### C.1 环境速查

| 项                                                     | 值                                                  |
| ----------------------------------------------------- | -------------------------------------------------- |
| GPU / CC                                              | RTX 5070 Laptop / 12.0（sm_120）                     |
| SM / smem/SM / regs/SM / threads/SM                   | 36 / 102400 B / 65536 / 1536                       |
| smem/block 默认 / 可提升                                   | 49152 B / 100 KB 量级（需 `cudaFuncSetAttribute`，实测确认） |
| CUDA / nvcc                                           | 13.2 / V13.2.86                                    |
| Python / PyTorch / pybind11 / pytest（项目环境 `cuda_132`） | 3.12.13 / 2.13.0+cu132 / 3.1.0 / 9.1.1             |
| Triton（项目环境 `cuda_132`）                               | 3.7.1（动态 smem 上限 101,376 B）                        |
| CMake / g++                                           | 4.4.0-rc1 / 15.2.0                                 |

### C.2 数值容差（初值，见 §11.4）

| 路径                          | 最大绝对误差 | 平均绝对误差 |
| --------------------------- | ------ | ------ |
| fp32 vs fp64                | 1e-4   | 1e-6   |
| fp16 vs fp32                | 5e-3   | 5e-4   |
| bf16 vs fp32                | 2e-2   | 2e-3   |
| decode/paged vs 连续 KV flash | 1e-3   | 1e-4   |

### C.3 三仓关键路径速查

```text
SciComputeInfra : $SCA_INFRA_ROOT
vLLM(C++)       : $SCA_VLLM_ROOT
RLHF            : $SCA_RLHF_ROOT
本项目          : $SCA_ROOT
真实模型        : $SCA_MODEL_DIR（GGUF Q8_0, qwen3, 36L/32H/8KV/D128）
模型服务        : http://127.0.0.1:11435（Ollama，tag qwen3:4b-thinking-2507-q8_0，启动：./serve.sh）
vLLM 关键头文件 : include/vllm/attention/{attention_backend,flash_attention,paged_attention}.hpp, include/vllm/memory/kv_cache.hpp
RLHF 关键文件   : rlhf/rollout_server.h, rlhf/rollout/{actor,gpu_actor,kv_cache}.h, rlhf/cuda_ops/*
```

---

## 附录 D：故障排查树

```text
[构建失败]
  ├─ 找不到 SciComputeInfra 头 → 检查 SCI_ATTENTION_SCICOMPUTE_INFRA_ROOT 与 include 路径
  ├─ undefined reference to launch_flash_attention → 上游只有声明（UP-002），改用本项目实现
  ├─ CUDA arch 错误 → 检查 SCI_ATTENTION_ARCH 是否包含 120；禁止沿用上游 "80;86;89;90"
  └─ pybind11 找不到 → python3 -m pybind11 --cmakedir 并加入 CMAKE_PREFIX_PATH

[运行期崩溃 / 校验错误]
  ├─ 非法内存访问 → compute-sanitizer memcheck 定位；优先查 slot_mapping 与 page table 边界
  ├─ 竞态 → racecheck；检查 __syncthreads 位置与 cp.async 的 wait_group 计数
  ├─ 未初始化读 → initcheck；检查输出缓冲是否清零/是否未写满边界块
  └─ 显存 OOM → 查显存预算表（§8.4）与 benchmark 跳过策略；禁止靠重试掩盖

[数值不正确]
  ├─ 误差随 S 增大而增大 → 检查是否真的用了 online softmax（是否两遍遍历）
  ├─ 出现 NaN/Inf → 检查 mask 是否用 -inf、alpha/beta 计算、l 是否为 0
  ├─ causal 路径错误 → 检查对角线块谓词 (j > i) 与块级 break 条件
  ├─ GQA 结果错 → 检查 h_kv = h_q / group_size 与 H_q % H_kv
  └─ decode 与 flash 不一致 → 检查 split-K 合并公式（m/l 权重）

[性能不达标]
  ├─ 先用 ncu 判定四类瓶颈（§13.4），禁止盲调
  ├─ Tensor Core 利用率低 → 检查 mma.sync 是否真正使用、ldmatrix 是否命中、bank conflict 计数
  ├─ 带宽利用率低 → 检查向量化（float4/half8）、cp.async 深度、L2 命中
  ├─ occupancy 过低 → 检查 registers/thread、smem/block（ncu 的 occupancy limit 指标）
  └─ 端到端不变 → 检查 launch 开销与同步（nsys）

[vLLM 集成失败]
  ├─ 链接 vllm::vllm 失败 → 上游依赖 spdlog/nlohmann_json（通过 vcpkg 提供）缺失
  ├─ 结果与上游不一致 → 检查 dtype 转换与 layout 包装（BSHD↔BHSD）
  └─ 上游原型本身结果错 → 记录为上游缺陷（§15.7），不要试图「对齐错误值」

[RLHF 集成失败]
  ├─ CUDA arch 不含 120 → 本地构建时覆盖 CMAKE_CUDA_ARCHITECTURES
  ├─ GPU attention 缺失 → 确认 rlhf 侧尚未接入 SCA（本提示词只提供接口与示例）
  └─ CPU KV Cache 布局不一致 → 按 §8.1 的迁移表转换，禁止直接 memcpy 结构体

[Profiling 权限]
  └─ ncu 报权限不足 → 需要管理员权限或调整 NVreg_RestrictProfilingToAdminUsers（需用户确认）
```

---

## 附录 E：术语与口径

| 术语                 | 口径（本项目统一）                                                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Prefill            | 一次性处理 S_q 个 token（S_q 大），可用 MMA + tile                                                                                  |
| Decode             | 每步 1 个（或极少）新 token，读全量 KV，瓶颈在 HBM                                                                                       |
| GQA / MQA          | 多个 query head 共享一组 KV head；`group_size = H_q / H_kv`                                                                    |
| Online softmax     | 分块累积 `(m, l, O)` 的稳定 softmax，无需物化 N×N                                                                                   |
| LSE                | `log-sum-exp`：`m_i + log(l_i)`，可选输出                                                                                     |
| 有效带宽               | `bytes_moved_modeled / latency`，模型公式见 §14.2                                                                             |
| 算术强度               | `FLOPs / bytes_moved_modeled`（FLOPs/Byte）                                                                               |
| P50/P90/P95/P99    | 100 次测量的分位数（§12.1）                                                                                                      |
| Achieved Occupancy | ncu 的 `sm__warps_active.avg.pct_of_peak_sustained_active`                                                               |
| 物化 N×N             | 把 `[B,H,S_q,S_kv]` 分数矩阵写入显存——本项目核心目标就是消除它                                                                               |
| GGUF / Q8_0        | GGUF v3 模型容器格式；Q8_0 = 每 32 个权重一组（fp16 scale + 32×int8），本项目仅对权重做 CPU/GPU 反量化，不改变注意力计算精度                                  |
| Ollama 基线          | `http://127.0.0.1:11435` 上运行 Qwen3-4B-Thinking-2507-Q8（100% GPU，上下文 4096，约 4882 MiB VRAM）的端到端服务指标，仅作参照，不代表本项目 kernel 性能 |

---

## 结语（写给执行者）

本提示词的判定标准只有一条：**产物能否用真实数据证明「Attention 计算在 GPU 上被正确、稳定、可解释地执行」**。

```text
正确性 > 数值稳定 > 显存效率 > 占用率 > Tensor Core 利用率 > 指令效率 > 端到端吞吐
```

顺序不可调换。任何为了数字更好看而破坏正确性、可维护性或 API 一致性的做法，都算任务失败。

最终定位（README 与总结必须使用同一句话）：

```text
SciCompute-Attention —— 面向大模型训练与推理的 GPU 原生高性能 Attention 基础设施，
以 FlashAttention、KV Cache、Decode Attention 为核心，
向 vLLM(C++) Serving 与 RLHF Rollout 提供统一的 Attention Runtime。
```
