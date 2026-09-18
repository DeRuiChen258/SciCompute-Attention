# KV Cache 设计

## 1. 权威布局（唯一口径）

```text
K/V : [num_layers][num_blocks][block_size][num_kv_heads][head_dim]     (KvLayout::kBlockMajor)
slot: slot = physical_block * block_size + offset_in_block
页表: [max_num_seqs][max_blocks_per_seq]（int32，-1 = 空槽）
```

地址公式（`src/kv_cache/kv_cache_layout.cuh`，host 与 device 共用同一实现）：

```text
offset(layer, block, off, head, d) = ((block * block_size + off) * num_kv_heads + head) * head_dim + d
layer_size = num_blocks * block_size * num_kv_heads * head_dim
```

示例（block_size=8, Hkv=2, D=32）：token 在 slot 9 → block 1 offset 1 → 该 token 的 head 0 从元素
`((1*8+1)*2+0)*32 = 576` 开始，head 1 从 608 开始。

## 2. 显存预算

```text
bytes = 2 * num_layers * num_blocks * block_size * num_kv_heads * head_dim * sizeof(dtype)
```

| 配置 | 每 block | 3 GiB 预算可容纳 | 总 token |
| --- | --- | --- | --- |
| layers=32, n_kv=8, D=128, block=16, fp16 | 2 MiB | 1536 blocks | 24576 |
| layers=32, n_kv=8, D=64, block=16, fp16 | 1 MiB | 3072 blocks | 49152 |
| layers=28, n_kv=4, D=128, block=16, bf16 | 896 KiB | 3510 blocks | 56160 |
| **Qwen3-4B (36 层, n_kv=8, D=128), fp16** | **2.25 MiB** | **1330 blocks** | **21280** |

`KVCache::Create` 会用 `cudaMemGetInfo` 复算：超过剩余显存的 80% 时返回 `kKVCapacityExceeded`
并给出建议 `num_blocks <= N`（不允许靠 OOM 崩溃暴露问题）。

## 3. 生命周期

```text
Create   → 校验配置 → 预算检查 → 逐层分配 K/V → 初始化 stats
Append   → 校验 shape/dtype → scatter kernel（slot → block/offset）→ 统计 num_appends
Gather*  → 调试/测试用：把逻辑块拼成连续张量（不进入推理热路径）
Reset    → 逐层清零给定 block（memset 风格 kernel）
Stats    → 只读快照
```

接口契约（重要）：

* `Append` 的 `slot_mapping` 与 `Reset/Gather*` 的 `block_ids` 必须是**设备指针**
  （int32），避免任何隐藏的 H2D 同步；`ComputeSlotMapping()` 生成宿主侧映射，
  由调用方上卡（`PagedKVCache::AppendTokens` 内部就是这么做的）。
* `slot_mapping[t] < 0` 表示「该 token 不写入」，内核显式跳过而不是越界写。

## 4. 与 vLLM(C++) 的对应

| 上游 | 本项目 | 关系 |
| --- | --- | --- |
| `vllm::KVCache`：每 layer 连续 `[num_blocks * block_size * num_kv_heads * head_dim]` | `sca::KVCache`：`[num_layers][num_blocks][block_size][num_kv_heads][head_dim]` | **同构**，可零拷贝映射（层指针直接传出） |
| 上游无 block table 抽象 | `sca::BlockManager` + `BlockTable` | 若上游已有等价分配器，适配层复用其分配结果 |
| RLHF `rlhf::KVCache`（CPU dense） | `sca::PagedKVCache` | 迁移路径见 `docs/rollout_interface.md` |

## 5. 测试覆盖（`tests/kv/test_kv_cache.cpp`）

```text
[x] block_size ∈ {1,8,16,32,64}
[x] 长度非 block_size 整数倍（3 token / block_size=8）
[x] 容量耗尽 → kKVCapacityExceeded 且状态不变
[x] Reset 后重新 append 无串扰（已验证 block 之间互不影响）
[x] 预算公式与手工计算一致（1536 blocks × 2 MiB = 3 GiB）
[x] 非法配置（block_size=7、head_dim=48、num_blocks=0）被拒绝
```

