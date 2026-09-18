# 三仓关系与版本锚点

## 1. 依赖链（单向）

```text
RLHF（业务层）  → vLLM(C++)（服务层） → SciCompute-Attention（本项目） → SciComputeInfra（基础设施）
```

禁止反向依赖，禁止循环依赖：`SciComputeInfra` 不得 include 本项目；本项目不得被上游 include。

## 2. 版本锚点（2026-09-19 实测）

| 仓库 | HEAD | 备注 |
| --- | --- | --- |
| SciComputeInfra | `61492cccf692d0891e376df5e100f31373f7a010` | Tensor/Device/Stream/Memory/Benchmark |
| vLLM(C++) | `f6d0ea9ee7c38e4575fab1bd73b3aaa374607adc` | `vllm::vllm` + `vllm_server` |
| RLHF | `1f7ba72bff9abc023629053b48f31ddab182315b` | `mini-rlhf-stack` |

重新采集：

```bash
for repo in SciComputeInfra SciCompute-Attention vllm RLHF; do
  printf '%-22s %s\n' "$repo" "$(git -C ../$repo rev-parse HEAD 2>/dev/null || echo NA)"
done
```

本项目 CMake 配置阶段会把上游 HEAD 写入 `SCI_ATTENTION_INFRA_SHA` 并在摘要中打印
（见 `cmake/SciComputeInfra.cmake`），版本漂移可被立刻发现。

## 3. 复用清单与边界

| 上游能力 | 本项目使用方式 |
| --- | --- |
| `sci::Tensor/TensorShape` | 公共 API 的张量类型（外部数据可零拷贝包装） |
| `sci::Device/CudaDevice` | 显存分配与 H2D/D2H 拷贝 |
| `sci::Stream/Event` | kernel 启动与计时（`stream == nullptr` ⇒ `GetCurrent()`） |
| `MemoryPool/Allocator` | workspace/scratch 的底层缓冲 |
| `benchmark::BenchmarkRunner` | 可选；本项目自带 CUDA-event harness 以对齐 §12.4 字段 |
| `cuda/kernels/attention.cuh` | **不复用**（只有声明无实现，UP-002） |
| `include/gpu/*`（GpuTensor 路径） | **不复用**（legacy 双轨，UP-004） |

## 4. Patch 清单

```text
upstream/patches/            目前为空：所有上游问题只登记 + 提供建议，未生成补丁
应用流程（需用户确认）：git -C ../SciComputeInfra apply upstream/patches/NNNN-*.patch
回滚：git -C ../SciComputeInfra apply -R upstream/patches/NNNN-*.patch
```

## 5. 上游问题索引

见 `upstream/notes.md`：UP-001 硬编码 CUDA include、UP-002 attention.cuh 只有声明、
UP-003 空目录、UP-004 双套 GPU 抽象、UP-005 无 Python 绑定、UP-006 vLLM 原型缺陷、
UP-007 RLHF 构建缺陷、UP-008 上游 CMake 使用 `${CMAKE_SOURCE_DIR}` 无法安全嵌套。

