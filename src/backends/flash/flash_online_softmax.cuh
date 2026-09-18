// Online softmax state for one 16-row mma tile (prompt §7.4.2).
//
// math:
//   m_new = max(m_old, rowmax(S_tile))
//   alpha = exp2((m_old - m_new) * log2e)          // 0 when m_old == -inf (first tile)
//   P     = exp2((S_tile - m_new) * log2e)         // beta absorbed into P
//   l_new = alpha * l_old + rowsum(P)
//   O_new = alpha * O_old + P V
//
// Each thread owns two rows of the 16-row tile: row g and row g + 8 (g = lane >> 2). All
// reductions are warp shuffles over the four lanes that share a row (the lane&3 group); no atomics
// and no shared memory are involved.
#pragma once

#include <cuda_runtime.h>

#include "cuda_common/numerics.cuh"

namespace sca {
namespace cuda {

__device__ __forceinline__ float RowReduceMax4(float value, unsigned mask) {
    value = fmaxf(value, ::__shfl_xor_sync(mask, value, 1));
    value = fmaxf(value, ::__shfl_xor_sync(mask, value, 2));
    return value;
}

__device__ __forceinline__ float RowReduceSum4(float value, unsigned mask) {
    value += ::__shfl_xor_sync(mask, value, 1);
    value += ::__shfl_xor_sync(mask, value, 2);
    return value;
}

template <int kBlockN>
struct OnlineSoftmax {
    static constexpr int kNTiles = kBlockN / 8;

    float m[2];      // row max for rows (warp_row + g) and (warp_row + g + 8)
    float l[2];      // row sum of exp(S - m)
    float alpha[2];  // per-tile rescale factor applied to the O accumulator

    __device__ __forceinline__ void Init() {
        m[0] = NegInf();
        m[1] = NegInf();
        l[0] = 0.0f;
        l[1] = 0.0f;
        alpha[0] = 0.0f;
        alpha[1] = 0.0f;
    }

    // Step 1: fold the tile row-max into the running max and derive alpha.
    __device__ __forceinline__ void UpdateMax(const float s_frag[kNTiles][4], unsigned mask) {
        float tile_max0 = NegInf();
        float tile_max1 = NegInf();
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            tile_max0 = fmaxf(tile_max0, fmaxf(s_frag[nt][0], s_frag[nt][1]));
            tile_max1 = fmaxf(tile_max1, fmaxf(s_frag[nt][2], s_frag[nt][3]));
        }
        tile_max0 = RowReduceMax4(tile_max0, mask);
        tile_max1 = RowReduceMax4(tile_max1, mask);

        const float m_new0 = fmaxf(m[0], tile_max0);
        const float m_new1 = fmaxf(m[1], tile_max1);
        // m_old == -inf (or an entirely masked tile) yields alpha = 0, which correctly drops the
        // previous accumulator instead of producing a NaN.
        alpha[0] = (m[0] == NegInf() || m_new0 == NegInf()) ? 0.0f
                                                            : FastExp2((m[0] - m_new0) * kLog2e);
        alpha[1] = (m[1] == NegInf() || m_new1 == NegInf()) ? 0.0f
                                                            : FastExp2((m[1] - m_new1) * kLog2e);
        m[0] = m_new0;
        m[1] = m_new1;
    }

    // Step 2: convert S fragments to P in place and update the running sum.
    __device__ __forceinline__ void UpdateProbabilities(float s_frag[kNTiles][4], unsigned mask) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            const float row0 = (m[0] == NegInf()) ? 0.0f : FastExp2((s_frag[nt][0] - m[0]) * kLog2e);
            const float row0b = (m[0] == NegInf()) ? 0.0f : FastExp2((s_frag[nt][1] - m[0]) * kLog2e);
            const float row1 = (m[1] == NegInf()) ? 0.0f : FastExp2((s_frag[nt][2] - m[1]) * kLog2e);
            const float row1b = (m[1] == NegInf()) ? 0.0f : FastExp2((s_frag[nt][3] - m[1]) * kLog2e);
            s_frag[nt][0] = row0;
            s_frag[nt][1] = row0b;
            s_frag[nt][2] = row1;
            s_frag[nt][3] = row1b;
            sum0 += row0 + row0b;
            sum1 += row1 + row1b;
        }
        l[0] = alpha[0] * l[0] + RowReduceSum4(sum0, mask);
        l[1] = alpha[1] * l[1] + RowReduceSum4(sum1, mask);
    }

    __device__ __forceinline__ float Lse(int row_index) const {
        return (l[row_index] > 0.0f) ? (m[row_index] + logf(l[row_index])) : NegInf();
    }
};

}  // namespace cuda
}  // namespace sca
