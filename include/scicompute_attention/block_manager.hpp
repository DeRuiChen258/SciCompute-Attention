#pragma once

#include <cstdint>
#include <vector>

#include "core/status.hpp"
#include "tensor/tensor.hpp"

#include "scicompute_attention/export.hpp"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {

// Host block table plus its device mirror.
//   host   : [max_num_seqs, max_blocks_per_seq], -1 marks an empty slot
//   device : int32 mirror, synced only when `revision` advances
struct BlockTable {
    std::vector<int32_t> host;
    sci::Tensor device;
    uint64_t revision{0};
    uint64_t synced_revision{0};  // last revision copied to `device`
    int64_t max_num_seqs{0};
    int64_t max_blocks_per_seq{0};
};

// Exclusive (non-shared) block allocator. v1 deliberately has no reference counting, no
// copy-on-write and no prefix caching (YAGNI, prompt §6.7/§8.8).
class SCI_ATTENTION_API BlockManager {
public:
    BlockManager() = default;

    static sci::Result<BlockManager> Create(int64_t num_blocks, int64_t max_num_seqs,
                                            int64_t max_blocks_per_seq, sci::Device& device);

    // Allocates `count` blocks. On failure the manager state is unchanged.
    sci::Result<std::vector<int32_t>> Allocate(int64_t count);
    void Free(const int32_t* block_ids, int64_t count);

    int64_t NumFreeBlocks() const noexcept { return static_cast<int64_t>(free_list_.size()); }
    int64_t NumTotalBlocks() const noexcept { return num_blocks_; }
    int64_t AllocationFailures() const noexcept { return allocation_failures_; }
    int64_t DoubleFreeEvents() const noexcept { return double_free_events_; }

    sci::Status AppendToTable(int64_t seq_id, const int32_t* block_ids, int64_t count);
    sci::Status ResetSeq(int64_t seq_id);
    sci::Status SyncTableToDevice(sci::Stream* stream);

    const BlockTable& Table() const noexcept { return table_; }

private:
    int64_t num_blocks_{0};
    // LIFO free list: O(1) amortized allocate/free.
    std::vector<int32_t> free_list_;
    std::vector<uint8_t> in_use_;
    BlockTable table_;
    int64_t allocation_failures_{0};
    int64_t double_free_events_{0};
};

}  // namespace sca
