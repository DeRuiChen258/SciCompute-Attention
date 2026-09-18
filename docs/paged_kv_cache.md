# Paged KV Cache 与 Paged Attention

## 1. 结构

```text
PagedKVCache = KVCache（block-major 存储）
             + BlockManager（LIFO 独占分配 + free list）
             + page table（[max_num_seqs][max_blocks_per_seq] int32，-1 = 空槽）
```

设备侧查表（`src/backends/paged/page_table.cuh`）：

```text
logical_block = token_index / block_size
physical      = __ldg(&page_table[seq_id * max_blocks_per_seq + logical_block])
physical < 0  ⇒ 该 token 视为不存在（权重 0），绝不越界读
row           = physical * block_size + (token_index % block_size)
```

`paged_attention` 复用 split-K decode kernel：只有 K/V 的行号来源不同（线性 stride → 查表），
online softmax、合并公式、掩码语义完全共用（避免第二份实现带来的数值漂移）。

## 2. 分配与回收

```text
AppendTokens(seq, layer, k_new, v_new):
  needed = ceil((len + n_new) / block_size) - owned
  needed > 0 ⇒ Allocate(needed)（失败时状态不变）
            ⇒ AppendToTable  ⇒ 更新 own list 与 host 页表行
  ComputeSlotMapping(host) → 上卡（4 B/token）→ KVCache::Append → 长度更新 → revision++
ResetSeq(seq):
  ① 先清设备数据（KVCache::Reset）→ ② 回收 block → ③ 清页表行（避免悬空引用）
SyncPageTable: 仅在 revision 变化时 cudaMemcpyAsync 上卡
```

## 3. 实测影响

* page table 访存开销：每个 token 一次额外 4 B 读（同一 block 内被 `block_size` 个 token 复用），
  实测影响见 `docs/results/benchmark_report.md`（decode vs paged 对照）；
* 由于复用同一 kernel，paged 与 contiguous decode 的数值差异 < 1e-3（`tests/kernel/test_paged_correctness.cpp`
  对 FP64 参考断言 max ≤ 5e-3 / mean ≤ 5e-4）。

## 4. 测试覆盖（`tests/kernel/test_paged_correctness.cu`）

```text
[x] block_size ∈ {1,8,16,32}
[x] 序列长度非 block 整数倍（257、129、33）
[x] 乱序 page：断言物理块顺序 ≠ 逻辑顺序，且结果与 FP64 参考一致
[x] GQA group ∈ {1,2,4}
[x] KV 存储布局 [block][offset][head][dim]（token-major）与参考的 head-major 顺序显式转换后比对
```

## 5. 与 vLLM 的关系

vLLM(C++) 的 `KVCache` 与 `BlockManager` 提供的是「每 layer 连续 + block 粒度」的存储；
本项目的 `PagedKVCache` 在其上补了 page table 与多序列生命周期。适配层
（`vllm_backend/sca_paged_attention_adapter.*`）优先做指针映射而不是拷贝。

## 6. YAGNI 边界（v1 不做）

```text
prefix caching、block 共享/COW、抢占式驱逐、多卡 KV 分片、量化 KV（fp8/int8）
理由：这些能力需要调度器配合，属于 Roadmap（README §18）。
```

