# 环境与 ISA 能力报告

> 采集时间：2026-09-19（Phase 0）
> 采集方式：本机实跑，命令与原始输出见下文；禁止后续 kernel 设计脱离本报告

## 1. 环境基线

| 项 | 实测值 | 证据 |
| --- | --- | --- |
| OS / Kernel | `Linux violet-ThinkBook-16p-G6-AFR 7.0.0-31-generic #31-Ubuntu SMP PREEMPT_DYNAMIC Sat Aug 1 04:26:38 UTC 2026 x86_64` | `uname -a` |
| GPU | `NVIDIA GeForce RTX 5070 Laptop GPU`，driver `615.71.09` | `nvidia-smi --query-gpu=...` |
| 显存 | 8151 MiB 报告值 / 8,177,909,760 B 精确值（采集时 used 65 MiB） | `nvidia-smi` + `torch.cuda.get_device_properties(0)` |
| Compute Capability | 12.0（`sm_120`） | 同上 |
| SM 数量 | 36 | 同上 |
| L2 Cache | 33,554,432 B（32 MiB） | 同上 |
| smem/SM、smem/block 默认 / optin | 102,400 B / 49,152 B / **101,376 B** | `arch_probe` props |
| 寄存器/SM、线程/SM、线程/block、warp | 65,536 / 1,536 / 1,024 / 32 | 同上 |
| 显存时钟 / 位宽 | 12,001,000 kHz / 128-bit | 同上 |
| CUDA Toolkit | 13.2（`V13.2.86`） | `nvcc --version` |
| 宿主编译器 | g++ 15.2.0 | `g++ --version` |
| CMake | **3.31.6** | `cmake --version` |
| Python（项目通道） | 3.12.13 | `/home/violet/Workspace/miniconda/envs/cuda_132/bin/python -V` |
| PyTorch | 2.13.0+cu132，`cuda_available=True` | 同上 |
| Triton | 3.7.1 | 同上 |
| pybind11 / pytest / ninja | 3.1.0 / 9.1.1 / 可用 | 同上 |
| GoogleTest | 已安装（`libgtest-dev`，dpkg 命中 1 项） | `dpkg -l \| grep -c gtest` |
| Google Benchmark | vcpkg 预构建（`/home/violet/Workspace/IDE/vcpkg-2026.06.01/packages/benchmark_x64-linux`） | 目录存在 |
| Python vLLM | 未安装 | `import vllm` 失败（结论：集成对象是本机 C++ vLLM） |

**Python 通道陷阱（必须遵守）**：默认 shell 的 `python3` 指向 `unitree_rt`（torch 2.14.0+cu130 / triton 3.8.0），
与本项目环境不一致。本项目全部 Python 命令使用：

```bash
/home/violet/Workspace/miniconda/envs/cuda_132/bin/python
```

## 2. ISA 探针（复跑，运行时校验）

复现命令：

```bash
bash /home/violet/Workspace/Code/Project/RL_infra/SciCompute-Attention/tools/arch_probe/run_probe.sh
# 原始输出：profiling/reports/arch_probe.log
# JSON     ：profiling/reports/arch_probe.json
# wgmma    ：profiling/reports/wgmma_probe.log
```

设备属性原始输出：

```text
[props] name=NVIDIA GeForce RTX 5070 Laptop GPU cc=12.0 sm=36
[props] smem/SM=102400 smem/block(default)=49152 smem/block(optin)=101376
[props] regs/SM=65536 threads/SM=1536 threads/block=1024 warp=32 L2=33554432
[props] mem.total=8177909760 mem.clock=12001000 kHz bus.width=128
```

探针结论（`profiling/reports/arch_probe.log`，exit code 0）：

```text
[probe] mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32        : PASS  (identity-GEMM C==B verified over 32 lanes)
[probe] cp.async.cg.shared.global + commit_group/wait_group      : PASS  (16B async copy of 4 KiB verified)
[probe] ldmatrix.sync.aligned.m8n8.x4.shared.b16                 : PASS  (16x16 tile fragment layout verified)
[probe] cp.async.bulk.shared::cta.global (TMA 1D)                : PASS  (2 KiB bulk copy with mbarrier verified)
[probe] cp.async.bulk.tensor.2d (TMA tensor 2D)                  : PASS  (32x32 fp32 tile via CUtensorMap verified)
[probe] wgmma : FAIL (expected) -- ptxas diagnostic:
  ptxas ..., line 28; error : Instruction 'wgmma.fence' not supported on .target 'sm_120'
  ptxas ..., line 28; error : Instruction 'wgmma.fence' cannot be compiled for architecture 'sm_120'
  ptxas fatal : Ptx assembly aborted due to errors
```

与提示词 §1.2 的一致性：**逐条一致**（TMA 1D/2D 在本机实测均可用，且被提升为可选优化路径）。

### 2.1 额外收益：fragment 布局假设已被实测确认

`arch_probe` 不只是「编译通过」，而是把 kernel 依赖的两条布局假设做成了运行时断言：

1. `mma.sync.m16n8k16` 的 A/B/C fragment 索引（用于 flash 内核的 Q/K/P/V/V 片段映射）——
   以 A=I 的恒等 GEMM 验证 `C == B`，32 lane 全部一致。
2. `ldmatrix.x4` 的地址供给与回归（用于 K/V 片段装载）——16×16 tile 的每个 lane 元素逐一比对源张量。

该结论直接决定了 flash kernel 中「谁持有 S 的哪两个元素」以及 warp 内 4 线程行归约（`__shfl_xor_sync` 掩码 1/2）的正确性。

## 3. 三仓版本锚点

```text
SciComputeInfra        61492cccf692d0891e376df5e100f31373f7a010
SciCompute-Attention   （本仓库，Phase 0 首次提交）
vllm                   f6d0ea9ee7c38e4575fab1bd73b3aaa374607adc
RLHF                   1f7ba72bff9abc023629053b48f31ddab182315b
```

上游构建可用性实测：`cmake -S ../SciComputeInfra -B /tmp/sciinfra_probe -DCMAKE_BUILD_TYPE=Release` → 配置成功（exit 0），
CUDAToolkit 13.2.86 被正确发现。

## 4. 与提示词 §1 的差异说明

| 项 | 提示词记录 | 本次实测 | 处理 |
| --- | --- | --- | --- |
| CMake | 4.4.0-rc1 | 3.31.6 | 非阻塞；本项目 `cmake_minimum_required(VERSION 3.24)`，功能不受影响 |
| nvidia-smi 报告显存 | 8151 MiB | 8151 MiB | 一致 |
| smem/block optin | 「需 `cudaFuncSetAttribute` 提升上限」 | 实测 **101,376 B** | 已作为 `SCI_ATTENTION_MAX_SMEM_BYTES` 的默认口径 |
| TMA | 编译通过（§1.2） | 编译 + **运行期值校验通过**（1D 与 2D tensor） | 提升为可选路径（`SCI_ATTENTION_ENABLE_TMA=AUTO`） |
| Triton 动态 smem 上限 | 101,376 B | 同（与 device props 一致） | autotune 前置校验口径锁定 |

## 5. 环境对设计的硬约束（进入 Phase 1 的输入）

```text
1) 主 MMA 路径 : mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}
2) 主搬运路径  : cp.async.cg 16B + commit_group/wait_group 多级流水
3) 可选路径    : TMA（cp.async.bulk / cp.async.bulk.tensor.2d + mbarrier）
4) 禁用路径    : wgmma、tcgen05、Hopper cluster 特性
5) 预算约束    : smem ≤ 101,376 B/block、regs ≤ 65,536/SM、threads ≤ 1,536/SM
6) 显存约束    : 8,177,909,760 B 总量，Ollama 常驻时可用显存约 3.2 GiB
```

