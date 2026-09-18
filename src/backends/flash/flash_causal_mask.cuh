// Causal mask predicates (prompt §7.4.4).
//
// Convention (docs/attention_math.md §2): bottom-right aligned, i.e. query i sees key j when
//   j <= i + diagonal,   diagonal = S_kv - S_q.
// Outside the diagonal block no per-element branch is needed: whole tiles are either fully visible
// or fully masked.
#pragma once

#include <cstdint>

namespace sca {
namespace cuda {

__device__ __forceinline__ bool MaskElementVisible(int64_t row, int64_t col, int64_t diagonal) {
    return col <= row + diagonal;
}

// A whole K/V tile is invisible from every row of the CTA: stop the loop entirely.
__device__ __forceinline__ bool MaskTileFullyMasked(int64_t i0, int block_m, int64_t j_base,
                                                    int64_t diagonal) {
    return j_base > i0 + block_m - 1 + diagonal;
}

// Every column of the tile is visible from every row handled by this warp: skip element masking.
__device__ __forceinline__ bool MaskWarpFullyVisible(int64_t warp_row0, int64_t j_base, int block_n,
                                                     int64_t diagonal) {
    return j_base + block_n - 1 <= warp_row0 + diagonal;
}

}  // namespace cuda
}  // namespace sca

