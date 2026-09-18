// Device-side implementation details of the flash kernel: K/V tile loader and the kernel itself.
// Split from flash_fwd_kernel.cuh so that host-only translation units can include the parameters
// and the smem sizing without pulling in device code.
#pragma once

#include <cuda_runtime.h>

#include "backends/flash/flash_causal_mask.cuh"
#include "backends/flash/flash_fwd_kernel.cuh"
#include "backends/flash/flash_online_softmax.cuh"
#include "cuda_common/arch_features.cuh"
#include "cuda_common/cp_async.cuh"
#include "cuda_common/ldmatrix.cuh"
#include "cuda_common/mma_policy.cuh"
#include "cuda_common/numerics.cuh"

namespace sca {
namespace cuda {

// Loads one [kBlockN x kHeadDim] tile with 16 B cp.async chunks; rows beyond seq_kv are zero-filled
// so that P * V can never produce NaN from stale data.
template <typename T, int kHeadDim, int kBlockN>
__device__ __forceinline__ void FlashLoadTile(const T* gmem_row0, int64_t gmem_row_stride,
                                              T* smem_tile, int64_t j_base, int64_t seq_kv,
                                              int tid, int n_threads) {
    constexpr int kRowStride = kHeadDim + kRowPadElems;
    constexpr int kChunksPerRow = (kHeadDim * static_cast<int>(sizeof(T))) / kVectorAlignBytes;
    constexpr int kElemsPerChunk = kVectorAlignBytes / static_cast<int>(sizeof(T));
    constexpr int kTotalChunks = kBlockN * kChunksPerRow;

    for (int idx = tid; idx < kTotalChunks; idx += n_threads) {
        const int row = idx / kChunksPerRow;
        const int chunk = idx % kChunksPerRow;
        T* dst = smem_tile + row * kRowStride + chunk * kElemsPerChunk;
        const int64_t global_row = j_base + row;
        if (global_row < seq_kv) {
            const T* src = gmem_row0 + global_row * gmem_row_stride + chunk * kElemsPerChunk;
            CpAsync16(dst, src);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

template <typename T, int kHeadDim, int kBlockM, int kBlockN, int kStages, int kSplitsD, bool kCausal>
__global__ void __launch_bounds__((kBlockM / 16) * kSplitsD * 32) flash_fwd_kernel(
    FlashFwdParams p) {
    constexpr int kWarpsM = kBlockM / 16;
    constexpr int kWarps = kWarpsM * kSplitsD;
    constexpr int kThreads = kWarps * 32;
    constexpr int kRowStride = kHeadDim + kRowPadElems;
    constexpr int kKvTileElems = kBlockN * kRowStride;
    constexpr int kNTiles = kBlockN / 8;   // S/P fragments per warp
    constexpr int kQKTiles = kHeadDim / 16;
    constexpr int kDTiles = kHeadDim / 8;  // O fragments for the whole head dimension
    constexpr int kDTilesPerWarp = kDTiles / kSplitsD;
    static_assert(kDTiles % kSplitsD == 0, "head_dim must split evenly across D splits");
    static_assert(kBlockN % 16 == 0, "kBlockN must be a multiple of 16 (PV k-step)");

    extern __shared__ __align__(128) unsigned char smem_raw[];
    T* q_smem = reinterpret_cast<T*>(smem_raw);
    T* k_smem = q_smem + kBlockM * kRowStride;
    T* v_smem = k_smem + kStages * kKvTileElems;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp / kSplitsD;
    const int d_split = warp % kSplitsD;
    const int g = LaneGroup(lane);
    const int c = LaneCol(lane);

    const int i0 = blockIdx.x * kBlockM;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int64_t group = p.num_heads / p.num_kv_heads;
    const int64_t h_kv = head / group;

    const T* q_row0 = reinterpret_cast<const T*>(p.q) + batch * p.q_sb + head * p.q_sh +
                      static_cast<int64_t>(i0) * p.q_ss;
    const T* k_row0 = reinterpret_cast<const T*>(p.k) + batch * p.k_sb + h_kv * p.k_sh;
    const T* v_row0 = reinterpret_cast<const T*>(p.v) + batch * p.v_sb + h_kv * p.v_sh;

    // ---- Q tile + Q fragments ----------------------------------------------------------------
    for (int idx = tid; idx < kBlockM * kHeadDim; idx += kThreads) {
        const int row = idx / kHeadDim;
        const int col = idx % kHeadDim;
        const bool valid = static_cast<int64_t>(i0) + row < p.seq_q;
        q_smem[row * kRowStride + col] =
            valid ? q_row0[row * p.q_ss + col] : FromFloatT<T>(0.0f);
    }
    __syncthreads();

    uint32_t q_frag[kQKTiles][4];
#pragma unroll
    for (int kt = 0; kt < kQKTiles; ++kt) {
        const int matrix = lane >> 3;
        const int r = lane & 7;
        const int row = warp_m * 16 + (matrix & 1) * 8 + r;
        const int col = 16 * kt + (matrix >> 1) * 8;
        LdMatrixX4(q_frag[kt], SmemAddress(&q_smem[row * kRowStride + col]));
    }

    OnlineSoftmax<kBlockN> softmax;
    softmax.Init();
    float o_acc[kDTilesPerWarp][4];
#pragma unroll
    for (int dt = 0; dt < kDTilesPerWarp; ++dt) {
        o_acc[dt][0] = 0.0f;
        o_acc[dt][1] = 0.0f;
        o_acc[dt][2] = 0.0f;
        o_acc[dt][3] = 0.0f;
    }

    const unsigned kMask = 0xffffffffu;
    const int64_t j_tiles = (p.seq_kv + kBlockN - 1) / kBlockN;

    // ---- pipeline prologue: kStages - 1 tiles in flight ---------------------------------------
    const int kPrefetch = kStages - 1;
    for (int s = 0; s < kPrefetch; ++s) {
        if (s >= j_tiles) break;
        const int64_t j0 = static_cast<int64_t>(s) * kBlockN;
        FlashLoadTile<T, kHeadDim, kBlockN>(k_row0, p.k_ss, k_smem + s * kKvTileElems, j0,
                                            p.seq_kv, tid, kThreads);
        FlashLoadTile<T, kHeadDim, kBlockN>(v_row0, p.v_ss, v_smem + s * kKvTileElems, j0,
                                            p.seq_kv, tid, kThreads);
        CpAsyncCommit();
    }

    for (int64_t j_idx = 0; j_idx < j_tiles; ++j_idx) {
        const int64_t j_base = j_idx * kBlockN;
        if (kCausal && MaskTileFullyMasked(i0, kBlockM, j_base, p.diagonal)) break;

        const int stage = static_cast<int>(j_idx % kStages);
        // Tile j_idx is complete once at most kStages-2 groups are still pending.
        CpAsyncWaitGroup<kStages - 2>();
        __syncthreads();

        const T* k_tile = k_smem + stage * kKvTileElems;
        const T* v_tile = v_smem + stage * kKvTileElems;

        // Refill the buffer consumed by the *previous* iteration; the copy overlaps with the compute
        // below. The barrier above guarantees no thread is still reading that buffer.
        {
            const int64_t prefetch_idx = j_idx + kStages - 1;
            if (prefetch_idx < j_tiles) {
                const int pf_stage = static_cast<int>(prefetch_idx % kStages);
                const int64_t pf_base = prefetch_idx * kBlockN;
                FlashLoadTile<T, kHeadDim, kBlockN>(k_row0, p.k_ss,
                                                    k_smem + pf_stage * kKvTileElems, pf_base,
                                                    p.seq_kv, tid, kThreads);
                FlashLoadTile<T, kHeadDim, kBlockN>(v_row0, p.v_ss,
                                                    v_smem + pf_stage * kKvTileElems, pf_base,
                                                    p.seq_kv, tid, kThreads);
            }
            CpAsyncCommit();  // one group per iteration keeps the pending count predictable
        }

        // ---- S = Q K^T (mma.sync m16n8k16) ----------------------------------------------------
        float s_frag[kNTiles][4];
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            s_frag[nt][0] = 0.0f;
            s_frag[nt][1] = 0.0f;
            s_frag[nt][2] = 0.0f;
            s_frag[nt][3] = 0.0f;
        }
#pragma unroll
        for (int kt = 0; kt < kQKTiles; ++kt) {
#pragma unroll
            for (int np = 0; np < kNTiles / 2; ++np) {
                uint32_t kb[4];
                const int matrix = lane >> 3;
                const int r = lane & 7;
                const int row = 16 * np + (matrix & 1) * 8 + r;
                const int col = 16 * kt + (matrix >> 1) * 8;
                LdMatrixX4(kb, SmemAddress(&k_tile[row * kRowStride + col]));
                MmaM16N8K16<T>(s_frag[2 * np], q_frag[kt], kb[0], kb[2]);
                MmaM16N8K16<T>(s_frag[2 * np + 1], q_frag[kt], kb[1], kb[3]);
            }
        }

        // ---- scale + causal masking (no per-element branch off the diagonal block) ------------
        {
            const int64_t warp_row0 = static_cast<int64_t>(i0) + warp_m * 16;
            const bool tile_in_range = (j_base + kBlockN) <= p.seq_kv;
            const bool fully_visible =
                !kCausal || MaskWarpFullyVisible(warp_row0, j_base, kBlockN, p.diagonal);
            const bool need_mask = !tile_in_range || !fully_visible;
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                const int64_t col0 = j_base + 8 * nt + 2 * c;
                const int64_t row_lo = warp_row0 + g;
                const int64_t row_hi = row_lo + 8;
                bool vis0 = true, vis1 = true, vis2 = true, vis3 = true;
                if (need_mask) {
                    vis0 = col0 < p.seq_kv && (!kCausal || MaskElementVisible(row_lo, col0, p.diagonal));
                    vis1 = (col0 + 1) < p.seq_kv &&
                           (!kCausal || MaskElementVisible(row_lo, col0 + 1, p.diagonal));
                    vis2 = col0 < p.seq_kv && (!kCausal || MaskElementVisible(row_hi, col0, p.diagonal));
                    vis3 = (col0 + 1) < p.seq_kv &&
                           (!kCausal || MaskElementVisible(row_hi, col0 + 1, p.diagonal));
                }
                s_frag[nt][0] = vis0 ? s_frag[nt][0] * p.scale : NegInf();
                s_frag[nt][1] = vis1 ? s_frag[nt][1] * p.scale : NegInf();
                s_frag[nt][2] = vis2 ? s_frag[nt][2] * p.scale : NegInf();
                s_frag[nt][3] = vis3 ? s_frag[nt][3] * p.scale : NegInf();
            }
        }

        // ---- online softmax update ------------------------------------------------------------
        softmax.UpdateMax(s_frag, kMask);
        softmax.UpdateProbabilities(s_frag, kMask);

#pragma unroll
        for (int dt = 0; dt < kDTilesPerWarp; ++dt) {
            o_acc[dt][0] *= softmax.alpha[0];
            o_acc[dt][1] *= softmax.alpha[0];
            o_acc[dt][2] *= softmax.alpha[1];
            o_acc[dt][3] *= softmax.alpha[1];
        }

        // ---- O += P V -------------------------------------------------------------------------
#pragma unroll
        for (int kt = 0; kt < kBlockN / 16; ++kt) {
            uint32_t pa[4];
            pa[0] = PackPair<T>(s_frag[2 * kt][0], s_frag[2 * kt][1]);
            pa[1] = PackPair<T>(s_frag[2 * kt][2], s_frag[2 * kt][3]);
            pa[2] = PackPair<T>(s_frag[2 * kt + 1][0], s_frag[2 * kt + 1][1]);
            pa[3] = PackPair<T>(s_frag[2 * kt + 1][2], s_frag[2 * kt + 1][3]);
#pragma unroll
            for (int dp = 0; dp < kDTilesPerWarp / 2; ++dp) {
                uint32_t vb[4];
                const int matrix = lane >> 3;
                const int r = lane & 7;
                const int vrow = 16 * kt + (matrix & 1) * 8 + r;
                const int vcol = (d_split * kDTilesPerWarp + 2 * dp) * 8 + (matrix >> 1) * 8;
                LdMatrixX4Trans(vb, SmemAddress(&v_tile[vrow * kRowStride + vcol]));
                // ldmatrix.x4.trans register meaning (measured with tools/arch_probe + a dedicated
                // walk-through probe): reg0 = (M[2c][g], M[2c+1][g]) for rows 0-7 / cols 0-7,
                // reg1 = the same columns for rows 8-15, reg2/reg3 = the +8 column block.
                // The mma B fragment needs b0 = B[2c][g] and b1 = B[2c+8][g], i.e. (reg0, reg1)
                // for the first n-tile and (reg2, reg3) for the second.
                MmaM16N8K16<T>(o_acc[2 * dp], pa, vb[0], vb[1]);
                MmaM16N8K16<T>(o_acc[2 * dp + 1], pa, vb[2], vb[3]);
            }
        }

        // Everyone must be done reading the tile before the next iteration refills its buffer.
        __syncthreads();
    }

    // ---- epilogue: O / l, optional LSE -------------------------------------------------------
    T* out_row0 = reinterpret_cast<T*>(p.out) + batch * p.o_sb + head * p.o_sh +
                  static_cast<int64_t>(i0) * p.o_ss;
    const float inv0 = softmax.l[0] > 0.0f ? (1.0f / softmax.l[0]) : 0.0f;
    const float inv1 = softmax.l[1] > 0.0f ? (1.0f / softmax.l[1]) : 0.0f;
    const int64_t row_lo = static_cast<int64_t>(i0) + warp_m * 16 + g;
    const int64_t row_hi = row_lo + 8;
#pragma unroll
    for (int dt = 0; dt < kDTilesPerWarp; ++dt) {
        const int dcol = (d_split * kDTilesPerWarp + dt) * 8 + 2 * c;
        if (row_lo < p.seq_q) {
            T* dst = out_row0 + (row_lo - i0) * p.o_ss + dcol;
            dst[0] = FromFloatT<T>(o_acc[dt][0] * inv0);
            dst[1] = FromFloatT<T>(o_acc[dt][1] * inv0);
        }
        if (row_hi < p.seq_q) {
            T* dst = out_row0 + (row_hi - i0) * p.o_ss + dcol;
            dst[0] = FromFloatT<T>(o_acc[dt][2] * inv1);
            dst[1] = FromFloatT<T>(o_acc[dt][3] * inv1);
        }
    }
    if (p.return_lse && p.lse != nullptr && c == 0) {
        const int64_t base = (batch * p.num_heads + head) * p.seq_q;
        if (row_lo < p.seq_q) p.lse[base + row_lo] = softmax.Lse(0);
        if (row_hi < p.seq_q) p.lse[base + row_hi] = softmax.Lse(1);
    }
}

}  // namespace cuda
}  // namespace sca
