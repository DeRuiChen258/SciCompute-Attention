# 架构与文件职责

## 1. 分层

```text
上层（vLLM(C++) / RLHF / Python）
        │  只包含 include/scicompute_attention/*.hpp
        ▼
API 层      src/api/*            形状推导、参数校验（唯一校验点）、公共入口
调度层      src/runtime/*        capability / workspace / scratch / launcher / dispatcher
后端层      src/backends/*       naive | tiled | flash | decode | paged
存储层      src/kv_cache/*       KVCache / BlockManager / PagedKVCache
设备原语    src/cuda_common/*    cuda_check / numerics / mma_policy / ldmatrix / cp_async / arch_features
        ▼
基础设施    SciComputeInfra（Tensor / Device / Stream / Memory / Benchmark，只读依赖）
```

依赖方向单向：`上层 → API → 调度 → 后端 → 原语 → SciComputeInfra`。
禁止反向包含；`src/api` 与 `src/runtime` 不得包含 CUDA 头（device 调用统一经由
`runtime/device_probe.h` 的 C ABI 与 `.cu` 实现）。

## 2. 数据流（一次 `sca::attention()` 调用）

```text
q/k/v (sci::Tensor)
  → InferShape()          维度/布局/连续性/dtype/device 校验
  → Validate()            head_dim 集合、GQA、causal 关系、varlen、workspace 等
  → AttentionDispatcher::Select()   表驱动选择 + Supports() 询问 + allow_fallback 策略
  → backend->Forward()   分配输出 → 组装参数 → LaunchRaw() → stats 回填
  → AttentionResult{out, lse, stats}
```

`stats.note` 为空表示「严格按请求执行、无降级」；任何降级/近似都必须写入该字段。

## 3. 文件职责表（新增文件必须在此登记）

| 文件 | 职责 | 备注 |
| --- | --- | --- |
| `src/api/status.cpp` | 本域错误码 ↔ 上游 StatusCode 映射、版本、名称 | 不修改上游枚举 |
| `src/api/api_validate.cpp` | `Validate` / `EffectiveScale` / `RecommendTile` / `ValidateVarlenHost` | 唯一校验点 |
| `src/api/attention_api.cpp` | 高层入口 `attention/decode/paged/ExplainAttention` | 无异常 |
| `src/runtime/capability.cpp` | 设备能力探测与缓存 | 禁止硬编码架构 |
| `src/runtime/device_probe.{h,cu}` | CUDA 调用的 C ABI 边界 | host TU 不包含 cuda 头 |
| `src/runtime/launcher.{hpp,cu}` | kernel 启动、smem 属性缓存、错误检查 | 唯一 launch 出口 |
| `src/runtime/workspace.cpp` | 预分配切片与越界检测 | 256 B 对齐 |
| `src/runtime/scratch_pool.{hpp,cpp}` | 中间缓冲池（热路径零 cudaMalloc） | 进程级、故意泄漏 |
| `src/runtime/dispatcher.cpp` | 选择算法 + `Explain` + 回退链 | 与 `dispatch_table.inc` 对应 |
| `src/runtime/dispatch_table.inc` | AUTO-GENERATED 阈值表 | 由 sweep 脚本生成 |
| `src/cuda_common/*` | 指令封装与数值工具 | 无状态、纯头文件 |
| `src/backends/*` | 各 Level 的 kernel 与后端适配 | 一个 kernel 一个文件 |
| `src/kv_cache/*` | KV 存储、块管理、分页缓存 | 地址计算单点实现 |
| `benchmarks/benchmark_attention.cu` | tiled/flash/decode 基准 | 合并说明见文件头（共享 90% 代码） |

## 4. 扩展点

* 新增后端：实现 `IAttentionBackend` → 在 `runtime/backend_lookup.cpp` 注册 → 在
  `dispatch_table.inc` 增加阈值规则 → 补测试。
* 新增 dtype：扩展 `detail/dtype_traits.hpp` 与 `cuda_common/numerics.cuh` 的 `ToFloat/FromFloatT`，
  并在 kernel 实例化表中增加分支。
* 新增 tile：只改 `flash_tile_config.hpp`（编译期 `static_assert` 会拦住放不下的配置）。

## 5. 并发与线程模型

```text
* 公开 API 无全局可变状态（除只读能力缓存与 scratch pool）。
* scratch pool 与 PagedKVCache 的 slot-mapping 缓冲遵循「单线程、非重叠调用」约定；
  需要并发的调用方应使用独立进程或未来的 Workspace 显式接口（Roadmap）。
* kernel 内部同步点：tiled/flash 的 `__syncthreads()` 只用于 smem 生产-消费边界；
  decode/paged 的归约仅使用 warp shuffle。
```

