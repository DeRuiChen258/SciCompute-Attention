# Triton 轨道（状态：未实现，属 Roadmap）

## 1. 定位与边界（设计已固定）

```text
定位：Python 侧第二实现轨道，用于快速迭代、autotune 与三方交叉验证（Triton / CUDA / SDPA）。
边界（硬约束）：
  - 只存在于 Python 侧（python/scicompute_attention/triton_kernels/）；
  - C++/vLLM(C++) 路径永远走 CUDA kernel，不得用 Triton 替代；
  - 两套实现共享同一套数学约定（scale / causal / mask / LSE 定义）。
```

## 2. 环境约束（实测，Phase 0 复核）

| 项 | 值 | 影响 |
| --- | --- | --- |
| Triton 版本 | 3.7.1（conda env `cuda_132`） | 已验证可在 sm_120 上 JIT 编译运行 |
| 动态 smem 上限 | 101,376 B | `BM=64, D=128, stages=2` 需 106,496 B → `OutOfResources`，配置必须前置校验 |
| kernel 定义位置 | 必须是真实 `.py` 文件 | `exec`/stdin 定义会 `OSError: could not get source code` |
| 首次编译开销 | 秒级 | benchmark 必须预热 ≥20 次并单独记录编译时间 |

## 3. 当前状态（本次交付）

| 项 | 状态 | 原因 |
| --- | --- | --- |
| `python/scicompute_attention/triton_kernels/*` | 未实现 | 本次交付优先闭环「自研 CUDA kernel + 单测 + benchmark + 文档」主线 |
| `benchmarks/benchmark_triton.py` | 未实现 | 同上 |
| `python/tests/test_triton_parity.py` | 未实现 | 同上 |
| 配置合法性校验（smem ≤ 101,376 B） | 设计已定 | 落地时按 §2 的上限做前置筛除 |
| 三方一致性矩阵（D∈{64,128,256} × S 边界 × GQA） | 设计已定 | 容差与 CUDA 路径同一张表（`docs/numerical_stability.md`） |

## 4. 已具备的前提条件

```text
[x] Triton 可用性已验证（本机 sm_120 + triton 3.7.1 编译运行成功，见附录 B.2.1）
[x] CUDA kernel 侧已具备三方对照的基准（flash / decode / paged 全部有 FP64 参考断言）
[x] SDPA 对照数据已落盘（benchmarks/benchmark_sdpa.py），可作为 Triton 的第三方参照
[x] 数值容差表与配置上限（101,376 B）已固化，可直接复用
```

## 5. 复现路径（后续实现时）

```bash
conda run -n cuda_132 python -m pytest python/tests/test_triton_parity.py -v
TRITON_CACHE_DIR=.triton_cache python benchmarks/benchmark_triton.py --suite main \
  --json benchmarks/results/<date>-<sha>/triton.json
```

**声明纪律**：本节明确说明 Triton 轨道未完成；README 第 18 节与
`docs/results/triton_vs_cuda.md` 均不得出现未经测量的 Triton 性能数字。

