#pragma once

#include <cstdint>
#include <vector>

#include "core/status.hpp"
#include "tensor/tensor.hpp"

#include "scicompute_attention/attention_types.hpp"
#include "scicompute_attention/export.hpp"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {

struct KVCacheConfig {
    int64_t num_layers{0};
    int64_t num_kv_heads{0};
    int64_t head_dim{0};
    int64_t block_size{16};  // tokens per page (vLLM default)
    int64_t num_blocks{0};   // physical blocks (derive from the budget formula, §8.4)
    int64_t max_num_seqs{1};
    sci::DType dtype{sci::DType::kFloat16};
    KvLayout layout{KvLayout::kBlockMajor};
    bool enable_stats{true};
};

struct KVCacheStats {
    int64_t num_blocks{0};
    int64_t used_blocks{0};
    int64_t free_blocks{0};
    int64_t num_appends{0};
    int64_t num_resets{0};
    size_t peak_bytes{0};
    size_t bytes{0};
};

// Block-major KV storage (authoritative layout, prompt §8.1):
//   K/V: [num_layers][num_blocks][block_size][num_kv_heads][head_dim]
// This is isomorphic to vllm::KVCache's per-layer contiguous buffers, which allows a zero-copy
// mapping in the vLLM adapter.
class SCI_ATTENTION_API KVCache {
public:
    KVCache() = default;

    static sci::Result<KVCache> Create(const KVCacheConfig& cfg, sci::Device& device);

    // k_new/v_new: [num_tokens, num_kv_heads, head_dim]; slot_mapping: [num_tokens], where
    // slot = physical_block * block_size + offset_in_block.
    sci::Status Append(int64_t layer, const sci::Tensor& k_new, const sci::Tensor& v_new,
                       const int32_t* slot_mapping, int64_t num_tokens, sci::Stream* stream);

    // Debug/test only: gathers logical blocks into a contiguous tensor. Not on the inference path.
    sci::Result<sci::Tensor> GatherK(int64_t layer, const int32_t* block_ids, int64_t num_blocks,
                                     sci::Stream* stream) const;
    sci::Result<sci::Tensor> GatherV(int64_t layer, const int32_t* block_ids, int64_t num_blocks,
                                     sci::Stream* stream) const;

    sci::Status Reset(const int32_t* block_ids, int64_t num_blocks, sci::Stream* stream);

    const sci::Tensor& K(int64_t layer) const;
    const sci::Tensor& V(int64_t layer) const;

    KVCacheStats Stats() const;
    size_t MemoryBytes() const noexcept;
    int64_t LayerElements() const noexcept { return layer_elements_; }
    const KVCacheConfig& Config() const noexcept { return cfg_; }

private:
    KVCacheConfig cfg_{};
    int64_t layer_elements_{0};
    std::vector<sci::Tensor> k_layers_;
    std::vector<sci::Tensor> v_layers_;
    KVCacheStats stats_{};
};

// Slot mapping helper: the single implementation shared by C++/Python/vLLM adapter.
// Computes, for tokens [seq_start, seq_start + num_tokens), the slot index inside the physical
// blocks listed in block_table_row.
SCI_ATTENTION_API sci::Result<std::vector<int32_t>> ComputeSlotMapping(
    const int32_t* block_table_row, int64_t seq_start, int64_t num_tokens, int64_t block_size,
    int64_t max_blocks);

// KV byte budget helper (prompt §8.4):
//   bytes = 2 * num_layers * num_blocks * block_size * num_kv_heads * head_dim * sizeof(dtype)
SCI_ATTENTION_API sci::Result<size_t> KvCacheBytes(const KVCacheConfig& cfg);

// Largest num_blocks that fits `budget_bytes` (at least 1). Returns 0 when even one block does
// not fit.
SCI_ATTENTION_API int64_t MaxBlocksForBudget(const KVCacheConfig& cfg, size_t budget_bytes);

}  // namespace sca

