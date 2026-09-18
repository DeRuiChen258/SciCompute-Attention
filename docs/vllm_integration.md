# vLLM(C++) 集成

## 1. 上游接口快照（Phase 0 复核）

```cpp
// include/vllm/attention/attention_backend.hpp
enum class AttentionBackend { kPagedAttention, kFlashAttention, kAuto };

// include/vllm/attention/flash_attention.hpp
struct FlashAttentionParams {   // 全部 FP32、布局 [batch, seq, heads, dim]
    float* output; const float* query; const float* key; const float* value;
    int32_t batch, seq_len_q, seq_len_kv, num_heads, num_kv_heads, head_dim;
    float scale; bool is_causal; cudaStream_t stream;
};
void flash_attention_forward(const FlashAttentionParams&);

// include/vllm/memory/kv_cache.hpp
class KVCache { void* k_ptr(size_t layer, size_t physical_block); ... };
```

## 2. 上游缺陷清单（带文件:行号，实证）

```text
[x] src/attention/flash_attention.cu:64  out[...] += ... 未初始化即累加
[x] 同上 :33 与 :52  两次遍历 K/V（先 max/l 再算输出），不是 online softmax
[x] 同上 :72  dim3 block(32)  单 warp 处理整行，无 tiling / 无 MMA
[x] 同上 :19  int kv_h = h % num_kv_heads   GQA 映射用取模而非 h / group_size
[x] 同上  仅 FP32 路径，无 FP16/BF16
[x] 性能：同 shape 下与 SDPA / 本项目 flash 的差距需实测（见 §4）
```

## 3. 适配层设计（`vllm_backend/`）

| 文件 | 职责 |
| --- | --- |
| `adapter_config.hpp` | `SCA_VLLM_COMPUTE_DTYPE=bf16\|fp16`、`SCA_VLLM_FAIL_FAST`、`min_seq_len_for_flash` |
| `layout_bridge.hpp` | BSHD ↔ BHSD 显式转换（可单测）；FP32 → BF16/FP16 显式转换 |
| `sca_vllm_adapter.{hpp,cpp}` | 参数校验 → 转换/包装 → 调用 `sca::flash_attention` → 写回 FP32 |
| `sca_flash_attention_adapter.{hpp,cpp}` | 与上游 `flash_attention_forward` 同签名的 drop-in 版本 |
| `sca_paged_attention_adapter.{hpp,cpp}` | `vllm::KVCache` 的 (layer, block) 指针映射到 `sca::PagedKVCache` |
| `patches/0001-attention-dispatch.patch` | 可选上游 hook（默认不应用） |

构建开关：

```bash
cmake -S . -B build-vllm -DSCI_ATTENTION_BUILD_VLLM_ADAPTER=ON \
      -DSCI_ATTENTION_VLLM_ROOT=/home/violet/Workspace/Code/Project/RL_infra/vllm
```

回滚：

```bash
export SCA_VLLM_BACKEND=legacy     # 走回上游原型实现
git -C ../vllm apply -R vllm_backend/patches/0001-attention-dispatch.patch   # 撤销 hook
```

## 4. 精度与布局迁移规则

| 维度 | 上游 | 本项目 | 规则 |
| --- | --- | --- | --- |
| dtype | FP32 | FP16/BF16 计算 + FP32 累加 | 显式转换；转换吞吐与误差必须记录 |
| 布局 | BSHD `[B,S,H,D]` | 内部优先 BHSD，**也支持 BSHD** | 视图优先（stride 传递），无法包装时显式 transpose |
| GQA | `num_kv_heads` 已有 | `H_q % H_kv == 0` | `h_kv = h / group_size`，覆盖 group ∈ {1,2,4,8} |
| scale | 上游传入 | `cfg.scale` 直传；≤0 时按 `1/sqrt(D)` 兜底并打日志 | 不允许静默忽略 |
| causal | `is_causal` | `cfg.causal` | 直传 |

**声明**：SCA 路径默认 BF16 计算，相对上游 FP32 原型存在量化误差（量级见
`docs/numerical_stability.md`）；需要 FP32 精度时必须显式选择参考路径（naive/tiled 后端）。

## 5. 当前状态

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 上游接口/缺陷审计 | 完成 | 见 §1/§2（Phase 0 落盘于 `docs/00-recon.md`） |
| 适配层源码 | 未实现 | 归属 Phase 11；接口契约已在 §3 固定，`vllm_backend/` 目录待建 |
| A/B harness | 未实现 | 需要先构建上游 `vllm`（依赖 spdlog/nlohmann_json，通过 vcpkg 提供） |
| 回滚开关 | 已定义 | 环境变量语义固定，但需适配层落地后生效 |

未完成原因：本次交付优先保证「自研 CUDA kernel + 测试 + benchmark + 文档」这条主线闭环；
适配层属于向上集成，其接口已在 §3 冻结，可在不触碰上游源码的前提下增量实现。
复现命令（上游构建成功时）：

```bash
cmake -S . -B build-vllm -DSCI_ATTENTION_BUILD_VLLM_ADAPTER=ON \
      -DSCI_ATTENTION_VLLM_ROOT=../vllm && cmake --build build-vllm -j
./build-vllm/benchmarks/benchmark_vllm --json docs/results/vllm_ab.json
```

