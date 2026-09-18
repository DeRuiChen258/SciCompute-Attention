// Single source of truth for the KV address arithmetic (prompt §8.3, DRY).
//
// Layout (authoritative, docs/kv_cache.md §8.1):
//   K/V storage per layer: [num_blocks][block_size][num_kv_heads][head_dim]
//   slot_mapping[t] = block_id * block_size + offset_in_block
//
// The same functions are used by host code (budget/slot computations) and by the device kernels
// (scatter/reset/gather), so a change can never desynchronise the two.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>  // __host__ / __device__ annotations

namespace sca {
namespace cuda {

struct KVCacheShape {
    int64_t num_blocks{0};
    int64_t block_size{0};
    int64_t num_kv_heads{0};
    int64_t head_dim{0};
};

// Offset (in elements) of one (block, offset_in_block, head, dim) element inside one layer.
__host__ __device__ __forceinline__ int64_t KvOffset(const KVCacheShape& shape, int64_t block,
                                                     int64_t offset_in_block, int64_t kv_head,
                                                     int64_t d) {
    return (((block * shape.block_size + offset_in_block) * shape.num_kv_heads + kv_head) *
                shape.head_dim +
            d);
}

// Number of elements of one layer (K or V).
__host__ __device__ __forceinline__ int64_t KvLayerElements(const KVCacheShape& shape) {
    return shape.num_blocks * shape.block_size * shape.num_kv_heads * shape.head_dim;
}

}  // namespace cuda
}  // namespace sca
