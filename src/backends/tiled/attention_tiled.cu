// Level 1 (tiled) attention implementation.
//
// Kernel map (single fused kernel):
//   tiled_attn_kernel<T, kHeadDim, kBlockM, kBlockN, kWarps>
//     grid  : (ceil(S_q / kBlockM), H_q, B)
//     block : kWarps * 32 threads
//     smem  : Q tile + K tile + V tile (each row padded by 8 halves = 16 B)
//             + S tile (FP32, padded by 4 floats/row) + row vectors (m, l, alpha)
//     regs  : O accumulator held as float4 slots: kBlockM * (kHeadDim/4) slots shared out over the
//             threads, i.e. kSlotsPerThread float4 = 4*kSlotsPerThread registers
//
// math (online softmax, identical formulation to the flash kernel):
//   m_new = max(m_old, rowmax(S_tile))
//   alpha = exp2((m_old - m_new) * log2e)
//   P     = exp2((S_tile - m_new) * log2e)      // beta absorbed into P
//   l_new = alpha * l_old + rowsum(P)
//   O_new = alpha * O_old + P @ V
//   O     = O_new / l_new                       // applied once, in the epilogue

#include "backends/tiled/attention_tiled.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>

#include "cuda_common/cuda_check.cuh"
#include "cuda_common/numerics.cuh"
#include "device/stream.hpp"
#include "runtime/launcher.hpp"

namespace sca {
namespace cuda {
namespace {

constexpr int kQSplit = 8;  // halves of padding per Q/K/V row (16 B)
constexpr int kSPad = 4;    // floats of padding per S row
constexpr int kNumRowVectors = 3;  // m, l, alpha

__device__ __forceinline__ float WarpMax(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, ::__shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float WarpSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += ::__shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename T, int kHeadDim, int kBlockM, int kBlockN, int kWarps>
__global__ void __launch_bounds__(kWarps * 32) tiled_attn_kernel(TiledFwdParams p) {
    constexpr int kRowStride = kHeadDim + kQSplit;
    constexpr int kSRowStride = kBlockN + kSPad;
    constexpr int kSlotsPerRow = kHeadDim / 4;
    constexpr int kTotalSlots = kBlockM * kSlotsPerRow;
    constexpr int kThreads = kWarps * 32;
    constexpr int kSlotsPerThread = (kTotalSlots + kThreads - 1) / kThreads;
    static_assert(kHeadDim % 4 == 0, "head_dim must be a multiple of 4 for float4 slots");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* q_smem = reinterpret_cast<T*>(smem_raw);
    T* k_smem = q_smem + kBlockM * kRowStride;
    T* v_smem = k_smem + kBlockN * kRowStride;
    float* s_tile = reinterpret_cast<float*>(v_smem + kBlockN * kRowStride);
    float* m_row = s_tile + kBlockM * kSRowStride;
    float* l_row = m_row + kBlockM;
    float* alpha_row = l_row + kBlockM;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr int kNumWarps = kWarps;

    const int i0 = blockIdx.x * kBlockM;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int64_t group = p.num_heads / p.num_kv_heads;
    const int64_t h_kv = head / group;

    const T* q_base = reinterpret_cast<const T*>(p.q) + batch * p.q_stride.s_b +
                      head * p.q_stride.s_h + static_cast<int64_t>(i0) * p.q_stride.s_s;
    const T* k_base =
        reinterpret_cast<const T*>(p.k) + batch * p.k_stride.s_b + h_kv * p.k_stride.s_h;
    const T* v_base =
        reinterpret_cast<const T*>(p.v) + batch * p.v_stride.s_b + h_kv * p.v_stride.s_h;

    // ---- Q tile ------------------------------------------------------------------------------
    for (int idx = tid; idx < kBlockM * kHeadDim; idx += kThreads) {
        const int row = idx / kHeadDim;
        const int col = idx % kHeadDim;
        const bool valid = static_cast<int64_t>(i0) + row < p.seq_q;
        q_smem[row * kRowStride + col] =
            valid ? q_base[row * p.q_stride.s_s + col] : FromFloatT<T>(0.0f);
    }
    for (int row = tid; row < kBlockM; row += kThreads) {
        m_row[row] = NegInf();
        l_row[row] = 0.0f;
    }

    float4 acc[kSlotsPerThread];
#pragma unroll
    for (int s = 0; s < kSlotsPerThread; ++s) acc[s] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();

    const int64_t j_tiles = (p.seq_kv + kBlockN - 1) / kBlockN;
    for (int64_t j0 = 0; j0 < j_tiles; ++j0) {
        const int64_t j_base = j0 * kBlockN;
        // Whole tile masked => nothing beyond it can be visible either (causal is monotone).
        if (p.causal &&
            j_base > static_cast<int64_t>(i0) + kBlockM - 1 + p.diagonal) {
            break;
        }

        for (int idx = tid; idx < kBlockN * kHeadDim; idx += kThreads) {
            const int row = idx / kHeadDim;
            const int col = idx % kHeadDim;
            const int64_t global_col = j_base + row;
            if (global_col < p.seq_kv) {
                k_smem[row * kRowStride + col] = k_base[global_col * p.k_stride.s_s + col];
                v_smem[row * kRowStride + col] = v_base[global_col * p.v_stride.s_s + col];
            } else {
                k_smem[row * kRowStride + col] = FromFloatT<T>(0.0f);
                v_smem[row * kRowStride + col] = FromFloatT<T>(0.0f);
            }
        }
        __syncthreads();

        // ---- S = Q K^T with scale and causal mask -------------------------------------------
        for (int idx = tid; idx < kBlockM * kBlockN; idx += kThreads) {
            const int row = idx / kBlockN;
            const int col = idx % kBlockN;
            float dot = 0.0f;
#pragma unroll 8
            for (int d = 0; d < kHeadDim; ++d) {
                dot += ToFloat(q_smem[row * kRowStride + d]) *
                       ToFloat(k_smem[col * kRowStride + d]);
            }
            const int64_t global_col = j_base + col;
            const bool masked =
                (global_col >= p.seq_kv) ||
                (p.causal && global_col > static_cast<int64_t>(i0) + row + p.diagonal);
            s_tile[row * kSRowStride + col] = masked ? NegInf() : dot * p.scale;
        }
        __syncthreads();

        // ---- per-row online softmax update (one warp per row) --------------------------------
        for (int row = warp; row < kBlockM; row += kNumWarps) {
            float local_max = NegInf();
            for (int col = lane; col < kBlockN; col += 32) {
                local_max = fmaxf(local_max, s_tile[row * kSRowStride + col]);
            }
            const float tile_max = WarpMax(local_max);

            const float m_old = m_row[row];
            const float m_new = fmaxf(m_old, tile_max);
            const float alpha =
                (m_old == NegInf() || m_new == NegInf()) ? 0.0f
                                                         : FastExp2((m_old - m_new) * kLog2e);

            float local_sum = 0.0f;
            for (int col = lane; col < kBlockN; col += 32) {
                const float value = s_tile[row * kSRowStride + col];
                const float prob =
                    (m_new == NegInf()) ? 0.0f : FastExp2((value - m_new) * kLog2e);
                s_tile[row * kSRowStride + col] = prob;
                local_sum += prob;
            }
            const float tile_sum = WarpSum(local_sum);
            if (lane == 0) {
                l_row[row] = alpha * l_row[row] + tile_sum;
                m_row[row] = m_new;
                alpha_row[row] = alpha;
            }
        }
        __syncthreads();

        // ---- O = alpha * O + P V (register accumulation) -------------------------------------
        for (int slot = tid; slot < kTotalSlots; slot += kThreads) {
            const int row = slot / kSlotsPerRow;
            const int d0 = (slot % kSlotsPerRow) * 4;
            float4 partial = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int col = 0; col < kBlockN; ++col) {
                const float prob = s_tile[row * kSRowStride + col];
                if (prob == 0.0f) continue;  // masked columns contribute exactly nothing
                partial.x += prob * ToFloat(v_smem[col * kRowStride + d0 + 0]);
                partial.y += prob * ToFloat(v_smem[col * kRowStride + d0 + 1]);
                partial.z += prob * ToFloat(v_smem[col * kRowStride + d0 + 2]);
                partial.w += prob * ToFloat(v_smem[col * kRowStride + d0 + 3]);
            }
            float4& previous = acc[slot / kThreads];
            const float alpha = alpha_row[row];
            previous.x = alpha * previous.x + partial.x;
            previous.y = alpha * previous.y + partial.y;
            previous.z = alpha * previous.z + partial.z;
            previous.w = alpha * previous.w + partial.w;
        }
        __syncthreads();
    }

    // ---- epilogue ----------------------------------------------------------------------------
    T* out_base = reinterpret_cast<T*>(p.out) + batch * p.o_stride.s_b + head * p.o_stride.s_h +
                  static_cast<int64_t>(i0) * p.o_stride.s_s;
    for (int slot = tid; slot < kTotalSlots; slot += kThreads) {
        const int row = slot / kSlotsPerRow;
        const int d0 = (slot % kSlotsPerRow) * 4;
        if (static_cast<int64_t>(i0) + row >= p.seq_q) continue;
        const float sum = l_row[row];
        const float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
        const float4 value = acc[slot / kThreads];
        T* dst = out_base + row * p.o_stride.s_s + d0;
        dst[0] = FromFloatT<T>(value.x * inv);
        dst[1] = FromFloatT<T>(value.y * inv);
        dst[2] = FromFloatT<T>(value.z * inv);
        dst[3] = FromFloatT<T>(value.w * inv);
    }
    if (p.return_lse && p.lse != nullptr) {
        for (int row = tid; row < kBlockM; row += kThreads) {
            if (static_cast<int64_t>(i0) + row >= p.seq_q) continue;
            const float m = m_row[row];
            const float l = l_row[row];
            p.lse[(batch * p.num_heads + head) * p.seq_q + i0 + row] =
                (l > 0.0f) ? (m + logf(l)) : NegInf();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// host side launch helpers
// ---------------------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kBlockM, int kBlockN, int kWarps>
sci::Status LaunchTile(const TiledFwdParams& p, sci::Stream* stream) {
    const size_t smem = static_cast<size_t>(
        TiledSmemBytes(kHeadDim, static_cast<int64_t>(sizeof(T)), kBlockM, kBlockN));
    void* args[] = {const_cast<TiledFwdParams*>(&p)};
    const int grid_x = static_cast<int>((p.seq_q + kBlockM - 1) / kBlockM);
    return runtime::LaunchRaw(
        reinterpret_cast<const void*>(&tiled_attn_kernel<T, kHeadDim, kBlockM, kBlockN, kWarps>),
        grid_x, static_cast<int>(p.num_heads), static_cast<int>(p.batch), kWarps * 32, 1, 1, smem,
        stream, args);
}

template <typename T, int kHeadDim>
sci::Status LaunchWithDim(const TiledFwdParams& p, sci::Stream* stream, int block_m, int block_n,
                          int warps) {
    if (block_m == 64 && block_n == 64 && warps == 4) {
        return LaunchTile<T, kHeadDim, 64, 64, 4>(p, stream);
    }
    if (block_m == 64 && block_n == 32 && warps == 4) {
        return LaunchTile<T, kHeadDim, 64, 32, 4>(p, stream);
    }
    if (block_m == 128 && block_n == 32 && warps == 8) {
        return LaunchTile<T, kHeadDim, 128, 32, 8>(p, stream);
    }
    return sci::Status::InvalidArgument("tiled: unsupported tile configuration bm=" +
                                        std::to_string(block_m) + " bn=" +
                                        std::to_string(block_n) + " warps=" +
                                        std::to_string(warps));
}

template <typename T>
sci::Status LaunchTyped(const TiledFwdParams& p, sci::Stream* stream, int block_m, int block_n,
                        int warps) {
    switch (p.head_dim) {
        case 32: return LaunchWithDim<T, 32>(p, stream, block_m, block_n, warps);
        case 64: return LaunchWithDim<T, 64>(p, stream, block_m, block_n, warps);
        case 96: return LaunchWithDim<T, 96>(p, stream, block_m, block_n, warps);
        case 128: return LaunchWithDim<T, 128>(p, stream, block_m, block_n, warps);
        default: break;
    }
    return sci::Status::InvalidArgument(
        "tiled attention: head_dim=" + std::to_string(p.head_dim) +
        " is not instantiated (tiled covers 32/64/96/128; use flash for 160/192/256)");
}

}  // namespace

int64_t TiledSmemBytes(int32_t head_dim, int64_t elem_bytes, int32_t block_m, int32_t block_n) {
    if (head_dim <= 0 || block_m <= 0 || block_n <= 0) return -1;
    const int64_t q = static_cast<int64_t>(block_m) * (head_dim + kQSplit) * elem_bytes;
    const int64_t kv = static_cast<int64_t>(block_n) * (head_dim + kQSplit) * elem_bytes;
    const int64_t s = static_cast<int64_t>(block_m) * (block_n + kSPad) * 4;
    const int64_t rows = static_cast<int64_t>(block_m) * kNumRowVectors * 4;
    return q + 2 * kv + s + rows;
}

sci::Status LaunchTiledForward(const TiledFwdParams& params, sci::Stream* stream, int32_t block_m,
                               int32_t block_n, int32_t warps) {
    if (params.q == nullptr || params.k == nullptr || params.v == nullptr ||
        params.out == nullptr) {
        return sci::Status::InvalidArgument("tiled attention: null tensor pointer");
    }
    switch (params.dtype) {
        case TiledDtype::kFp32: return LaunchTyped<float>(params, stream, block_m, block_n, warps);
        case TiledDtype::kFp16: return LaunchTyped<__half>(params, stream, block_m, block_n, warps);
        case TiledDtype::kBf16:
            return LaunchTyped<__nv_bfloat16>(params, stream, block_m, block_n, warps);
    }
    return sci::Status::InvalidArgument("tiled attention: unsupported dtype id");
}

}  // namespace cuda
}  // namespace sca
