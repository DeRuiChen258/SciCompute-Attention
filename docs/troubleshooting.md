# 故障排查

## 1. 构建失败

```text
[找不到 SciComputeInfra 头]
  → 检查 -DSCI_ATTENTION_SCICOMPUTE_INFRA_ROOT=<abs path>；
     或 -DSCI_ATTENTION_INFRA_MODE=STUB 做 CPU-only 构建（GPU 结论无效）
[undefined reference to launch_flash_attention]
  → 上游 attention.cuh 只有声明（UP-002）；本项目的实现位于 src/backends/flash/
[CUDA arch 错误 / ptxas 拒绝指令]
  → 检查 -DSCI_ATTENTION_ARCH 是否包含目标架构（≥80）；
     禁止使用 wgmma/tcgen05（sm_120 实测被 ptxas 拒绝）
[pybind11 找不到]
  → python3 -m pybind11 --cmakedir 并加入 CMAKE_PREFIX_PATH
[WERROR=ON 时出现第三方告警]
  → 上游头以 SYSTEM include 引入；CUDA 目标不使用 -Wpedantic（见 .agent/decisions.md D-007）
```

## 2. 运行期崩溃 / 校验错误

```text
[非法内存访问] → compute-sanitizer --tool memcheck；优先查 slot_mapping/page table 边界
[竞态]        → racecheck；检查 __syncthreads 位置与 cp.async 的 wait_group 计数
[未初始化读]   → initcheck；检查输出/部分缓冲是否写满
[显存 OOM]     → 查 docs/kv_cache.md 预算表；KVCache::Create 会提前返回 kKVCapacityExceeded
[进程退出时 SIGSEGV]
  → 持有设备内存的静态对象析构早于/晚于 CUDA 上下文；scratch pool 采用进程级故意泄漏
```

## 3. 数值不正确

```text
[误差随 S 增大]         → 检查是否真的用了 online softmax（而非两遍遍历）
[出现 NaN/Inf]          → 检查掩码是否用 -inf、alpha/l 是否为 0、全掩码行是否被保护
[causal 路径错误]       → 检查右下对齐口径 diag = S_kv - S_q 与块级 break 条件
[GQA 结果错]            → 检查 h_kv = h_q / group_size 与 H_q % H_kv
[decode 与 flash 不一致] → 检查 split-K 合并公式与 seq_lens 的取值
[paged 与连续 KV 不一致] → 检查页面布局 [block][offset][head][dim]（token-major）与
                          参考实现的 head-major 差异（曾因此误判，见 tests/kernel/test_paged_correctness.cu 注释）
```

## 4. 性能不达标

```text
[先判定四类瓶颈] → ncu：compute / memory / latency / launch bound（docs/profiling.md §4）
[Tensor Core 利用率低] → 检查 mma.sync 是否真的执行、ldmatrix 是否命中、bank conflict 计数
[带宽利用率低]  → 检查向量化（16B）、cp.async 深度、L2 命中率
[占用率过低]    → 检查 registers/thread、smem/block（ncu 的 occupancy limit 指标）
[端到端不变]    → 检查启动开销与同步（nsys）
```

## 5. 工具链权限

```text
[ncu 报权限不足] → 需要管理员权限或调整 NVreg_RestrictProfilingToAdminUsers（需用户确认）
[compute-sanitizer 很慢] → 用 --launch-timeout 120 并只跑单个 gtest filter
```

## 6. 环境陷阱（本机）

```text
[python3 解析到 unitree_rt] → 本项目所有 Python 命令必须用
  ${SCA_PYTHON:-python3}（torch 2.13.0+cu132 / triton 3.7.1）
[Ollama 占用 4.9 GiB] → 显存敏感实验前确认 /api/ps 不可达或先 stop
```

