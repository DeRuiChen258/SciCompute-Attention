// Device kernels for the KV cache: scatter-append, block reset and gather.
//
// All three are pure data-movement kernels over the block-major layout; the store is never
// transposed (that is what makes the zero-copy vLLM mapping possible).
#pragma once

#include <cstdint>

#include "core/status.hpp"
#include "kv_cache/kv_cache_layout.cuh"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {
namespace cuda {

// Copies [num_tokens, num_kv_heads, head_dim] rows to their slots in one layer.
sci::Status LaunchKvAppend(void* layer_k, void* layer_v, const void* k_new, const void* v_new,
                           const int32_t* slot_mapping, int64_t num_tokens,
                           const KVCacheShape& shape, int64_t elem_bytes, sci::Stream* stream);

// Zeroes the given blocks of one layer (both K and V).
sci::Status LaunchKvResetBlock(void* layer_k, void* layer_v, const int32_t* block_ids,
                               int64_t num_blocks, const KVCacheShape& shape, int64_t elem_bytes,
                               sci::Stream* stream);

// Copies `num_blocks` logical blocks from one layer into a contiguous buffer
// ([num_blocks * block_size, num_kv_heads, head_dim]); test/debug helper, not on the hot path.
sci::Status LaunchKvGather(const void* layer, const int32_t* block_ids, int64_t num_blocks,
                           void* out, const KVCacheShape& shape, int64_t elem_bytes,
                           sci::Stream* stream);

}  // namespace cuda
}  // namespace sca

