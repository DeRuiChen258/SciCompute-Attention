# 上游问题登记

> 规则：本项目**不修改**三个上游仓库。发现问题只在此登记（带 `文件:行号` 证据），
> 需要改动时只提供 `upstream/patches/NNNN-*.patch`，并由用户确认后才应用。

## 版本锚点

| 仓库 | HEAD |
| --- | --- |
| `SciComputeInfra` | `61492cccf692d0891e376df5e100f31373f7a010` |
| `vllm` | `f6d0ea9ee7c38e4575fab1bd73b3aaa374607adc` |
| `RLHF` | `1f7ba72bff9abc023629053b48f31ddab182315b` |

---

### UP-001: `src/bridges/CMakeLists.txt` 硬编码 CUDA include 路径
- 证据：`SciComputeInfra/src/bridges/CMakeLists.txt:26` →
  `target_include_directories(sci_bridges PUBLIC /usr/local/cuda-13.2/include)`
- 影响：CUDA 版本升级或非默认安装路径下构建失败；当前机器该目录存在，故未暴露。
- 建议：改为依赖 `find_package(CUDAToolkit)` 提供的 `CUDA::cudart` 传递 include 目录。
- 本项目回避方式：只链接精确目标（`sci_core/sci_device/sci_memory/sci_tensor/sci_math/sci_benchmark`），
  不链接聚合目标 `sci_compute`（它传递依赖 `sci_bridges`）。见 `.agent/decisions.md` D-004。
- 状态：待用户确认后可提 patch（默认不应用）。

### UP-002: `cuda/kernels/attention.cuh` 只有声明、无实现、未参与构建
- 证据：`SciComputeInfra/cuda/kernels/attention.cuh:19,31` 声明
  `scaled_dot_product_attention_kernel` / `launch_flash_attention`；
  `rg` 全仓无对应 `.cu` 定义，`cuda/CMakeLists.txt` 未包含该文件。
- 影响：上层若直接引用会出现链接错误（`undefined reference to launch_flash_attention`），
  且容易被误认为上游已具备 Attention 能力。
- 建议：在头文件加 `[[deprecated]]` 注释说明，或补最小实现。
- 本项目回避方式：不复用该声明，Attention 由本项目自研（`src/backends/*`）。
- 状态：已登记。

### UP-003: 空目录存在但无内容
- 证据：`cuda/helpers/`、`cuda/launch/`、`src/kernel/`、`src/graph/`、`python/bindings/` 均为空目录。
- 影响：目录结构给人「能力已存在」的错误预期。
- 建议：补充 README 占位说明或移除空目录。
- 状态：已登记。

### UP-004: 两套 GPU 抽象并存且边界不清
- 证据：`SciComputeInfra/include/gpu/gpu_tensor.hpp`（`GpuTensor`/`GpuDType`）与
  `include/gpu/cuda_kernels.hpp:114` 的 `scaled_dot_product_attention(...)` 声明，
  与 `include/tensor/tensor.hpp` 的 `sci::Tensor` 路径并存。
- 影响：使用哪套类型不明确，容易出现隐式转换与重复实现。
- 建议：文档明确 legacy 边界，长期合并为单一路径。
- 本项目回避方式：只使用 `sci::Tensor`/`sci::Device`，`include/gpu/*` 标记为 legacy，不复用不扩展。
- 状态：已登记。

### UP-005: 上游无 Python 绑定
- 证据：`python/bindings/` 为空目录，`python/` 下只有 `benchmarks/`、`examples/`。
- 影响：本项目 Python 侧无法复用上游绑定，需要自带 pybind11 模块。
- 建议：上游提供统一绑定后本项目可复用。
- 本项目做法：自建 `src/bindings/python_module.cpp` + `python/scicompute_attention/*`。
- 状态：已登记。

### UP-006: vLLM(C++) 的 `flash_attention_forward` 原型存在多处实现缺陷
- 证据：
  - `vllm/src/attention/flash_attention.cu:72` → `dim3 block(32)`：单个 warp 承担整行，无 tiling、无 MMA。
  - 同文件 `:33` 与 `:52`：两遍遍历 K/V（第一遍求 max/分母，第二遍算输出），不是 online softmax。
  - 同文件 `:64` → `out[out_base + d] += weight * value[...]`：输出缓冲未初始化即累加。
  - 同文件 `:19` → `int kv_h = h % num_kv_heads;`：GQA 映射用取模而非 `h / group_size`，
    `H_q % H_kv != 0` 时语义错误。
  - 同文件：仅 FP32 路径（`FlashAttentionParams` 全 `float*`），无 FP16/BF16。
- 影响：作为 A/B 基线时必须把「原型缺陷」与「本项目误差」分开表述；
  以 PyTorch FP32 为数值参考，而不是以上游原型为参考。
- 建议：上游切到本项目适配层（`vllm_backend/sca_flash_attention_adapter.*`），或按 `vllm_backend/patches/` 打补丁。
- 状态：已登记；A/B 数据见 `docs/results/vllm_ab.md`。

### UP-007: RLHF 构建配置与本机架构不匹配
- 证据：
  - `RLHF/CMakeLists.txt:12` → `set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")`，不含 `120`。
  - `RLHF/CMakeLists.txt:49-52` → `RLHF_CUDA_SOURCES` 只含 `fused_loss_kernel.cu`、`sampling_kernel.cu`，
    `rlhf/cuda_ops/gpu_ops.cu` 未加入（存在但不会被编译）。
- 影响：本机需覆盖架构才能原生运行；GPU 算子即使实现也不会进入构建。
- 建议：改为架构探测；把 `gpu_ops.cu` 加入源列表。
- 本项目做法：不改上游；`examples/rollout_engine_stub.cpp` 直接链接本项目，演示 rollout 形态。
- 状态：已登记。


### UP-008: 上游 CMakeLists 使用 `${CMAKE_SOURCE_DIR}`，无法安全嵌套 `add_subdirectory`
- 证据：`SciComputeInfra/src/tensor/CMakeLists.txt:5`、`src/core/CMakeLists.txt:3`、`src/math/CMakeLists.txt:8`、
  `src/device/CMakeLists.txt:7`、`cuda/CMakeLists.txt:25`、`examples/CMakeLists.txt:6-9` 均使用
  `${CMAKE_SOURCE_DIR}/...`。
- 实测：在被本项目 `add_subdirectory` 后，子目录里的 `project()` 会把 `CMAKE_SOURCE_DIR` 重新指向包含工程的顶层
  目录，于是上游的 include 目录与源文件路径解析到本仓，配置失败（原始报错见下）。用中间 wrapper 目录
  `set(CMAKE_SOURCE_DIR ...)` 覆盖同样无效（被嵌套 `project()` 重置）。
  ```text
  CMake Error at SciComputeInfra/src/bridges/CMakeLists.txt:17 (add_library):
    No SOURCES given to target: sci_bridges
  CMake Error at SciComputeInfra/examples/CMakeLists.txt:5 (add_executable):
    No SOURCES given to target: bridge_demo_gpu
  ```
- 建议：上游把 `${CMAKE_SOURCE_DIR}` 换成 `${PROJECT_SOURCE_DIR}` 或 `${CMAKE_CURRENT_SOURCE_DIR}`；
  并给 `examples/`、`cuda/` 增加可关闭的 option。
- 本项目做法：不修改上游；`cmake/SciComputeInfra.cmake` 的 SOURCE 模式只读引用上游源文件，
  编译成 `sci_device/sci_memory/sci_tensor/sci_math/sci_benchmark` 精确目标（见 `.agent/decisions.md` D-006）。
- 状态：已登记；如需上游修复可提 patch，但默认不应用。
