// Device-side page-table lookup (prompt §7.6).
//
// The KV cache stores one physical block per logical block; the page table maps
//   logical_block = token_index / block_size
// to a physical block id, and -1 marks "not allocated".
//
// Cost model: one extra 4 B read per K/V row touched (the page table entry is reused across the
// whole block, so the amortised cost is 4 B per `block_size` rows). The measured impact is recorded
// in docs/paged_kv_cache.md.
#pragma once

#include <cstdint>

namespace sca {
namespace cuda {

// Returns the physical block id for a token position, or -1 when the logical block is not mapped.
__device__ __forceinline__ int32_t PageTableLookup(const int32_t* page_table, int64_t seq_id,
                                                   int64_t token_index, int64_t block_size,
                                                   int64_t max_blocks_per_seq) {
    const int64_t logical_block = token_index / block_size;
    if (logical_block >= max_blocks_per_seq) return -1;
    return __ldg(&page_table[seq_id * max_blocks_per_seq + logical_block]);
}

// Physical row index (along the token axis of the paged KV storage) for a token position.
// Returns -1 when the token is not mapped or the block id is out of range.
__device__ __forceinline__ int64_t PageTableRow(const int32_t* page_table, int64_t seq_id,
                                                int64_t token_index, int64_t block_size,
                                                int64_t max_blocks_per_seq,
                                                int64_t num_kv_blocks) {
    const int32_t physical = PageTableLookup(page_table, seq_id, token_index, block_size,
                                             max_blocks_per_seq);
    if (physical < 0 || physical >= num_kv_blocks) return -1;
    return static_cast<int64_t>(physical) * block_size + (token_index % block_size);
}

}  // namespace cuda
}  // namespace sca

