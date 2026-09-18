#pragma once

#include <cstdint>
#include <vector>

#include "core/status.hpp"
#include "tensor/tensor.hpp"

#include "scicompute_attention/block_manager.hpp"
#include "scicompute_attention/export.hpp"
#include "scicompute_attention/kv_cache.hpp"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {

// Multi-sequence paged KV cache: KVCache storage + BlockManager + page table.
class SCI_ATTENTION_API PagedKVCache {
public:
    PagedKVCache() = default;

    static sci::Result<PagedKVCache> Create(const KVCacheConfig& cfg, sci::Device& device);

    // Appends tokens to one request. Internally: allocate blocks -> ComputeSlotMapping ->
    // KVCache::Append -> update metadata -> bump revision.
    sci::Status AppendTokens(int64_t seq_id, int64_t layer, const sci::Tensor& k_new,
                             const sci::Tensor& v_new, sci::Stream* stream);

    // Releases the request's blocks and clears its table row.
    sci::Status ResetSeq(int64_t seq_id, sci::Stream* stream);

    const int32_t* PageTableDevicePtr() const noexcept;
    int64_t LogicalBlockCount(int64_t seq_id) const noexcept;
    int64_t SeqLength(int64_t seq_id) const noexcept;
    int64_t BlockSize() const noexcept { return cfg_.block_size; }
    int64_t MaxBlocksPerSeq() const noexcept { return max_blocks_per_seq_; }
    int64_t MaxNumSeqs() const noexcept { return cfg_.max_num_seqs; }

    const KVCache& Storage() const noexcept { return storage_; }
    KVCache& MutableStorage() noexcept { return storage_; }
    const BlockManager& Blocks() const noexcept { return blocks_; }

    sci::Status SyncPageTable(sci::Stream* stream);
    KVCacheStats Stats() const;

    // Test/debug hook: regenerates the device mirror from the host table regardless of revision.
    sci::Status ForceSyncPageTable(sci::Stream* stream);

private:
    KVCacheConfig cfg_{};
    int64_t max_blocks_per_seq_{0};
    KVCache storage_{};
    BlockManager blocks_{};
    std::vector<int64_t> seq_lengths_;
    std::vector<std::vector<int32_t>> seq_blocks_;
    std::vector<int32_t> page_table_host_;
    sci::Tensor page_table_device_;
    // Reusable device buffer for the freshly computed slot mapping (see AppendTokens): the kernel
    // needs device-resident slots, so the host-side mapping is uploaded once per append.
    sci::Tensor slot_mapping_device_;
    int64_t slot_mapping_capacity_{0};
    uint64_t revision_{0};
    uint64_t synced_revision_{0};
    int64_t num_appends_{0};
    int64_t num_resets_{0};
};

}  // namespace sca
