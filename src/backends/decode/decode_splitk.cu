// Split-K decode attention.
//
// Kernel map
//   decode_partial_kernel<T> : one warp per (batch, head, split); each lane owns head_dim/32
//                              output elements and the matching K/V slice; online softmax over the
//                              split's token range, partial (m, l, O) written to workspace.
//   decode_reduce_kernel     : merges the splits with the stable rescaling formula.
//
// math (merge, prompt §7.5):
//   m_all = max_s m_s
//   l_all = sum_s l_s * exp(m_s - m_all)
//   O     = sum_s (O_s * l_s * exp(m_s - m_all)) / l_all
// The derivation is the standard one: each split holds the unnormalised sum O_s = sum_j exp(s_j -
// m_s) V_j, so rescaling to the global max and dividing by the global sum yields the exact result.
//
// Memory access: lane L loads the 8-byte (4 halves) chunk of K/V at column L*(head_dim/32), so a
// warp reads one full row of the head dimension as a contiguous 128-256 B block.

#include "backends/decode/decode_splitk.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "cuda_common/cuda_check.cuh"
#include "cuda_common/numerics.cuh"
#include "device/stream.hpp"
#include "runtime/launcher.hpp"

#include "backends/paged/page_table.cuh"

namespace sca {
namespace cuda {
namespace {

// Resolves the K/V row pointer for token `j`. With a page table the physical row is looked up;
// unmapped tokens (physical < 0) yield nullptr and are skipped, which is exactly "weight 0" per the
// prompt's §7.6 requirement and never reads out of bounds.
template <typename T>
__device__ __forceinline__ const T* KvRowPtr(const T* base, int64_t j, int64_t token_stride,
                                             const DecodeFwdParams& p, int64_t seq_id) {
    if (p.page_table == nullptr) return base + j * token_stride;
    const int64_t row = PageTableRow(p.page_table, seq_id, j, p.block_size,
                                     p.max_blocks_per_seq, p.num_kv_blocks);
    if (row < 0) return nullptr;
    return base + row * token_stride;
}

__device__ __forceinline__ void WarpReduceDot(float* dot) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        *dot += ::__shfl_xor_sync(0xffffffffu, *dot, offset);
    }
}

// Each lane owns a contiguous slice of the head dimension.
template <typename T, int kLaneElems>
struct LaneSlice {
    float q[kLaneElems];
    float acc[kLaneElems];
    float m;
    float l;
};

template <typename T, int kHeadDim>
__global__ void decode_partial_kernel(DecodeFwdParams p) {
    constexpr int kLaneElems = kHeadDim / 32;
    static_assert(kHeadDim % 32 == 0, "head_dim must be a multiple of 32");

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t warps_per_request = static_cast<int64_t>(blockDim.x) >> 5;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * warps_per_request + warp;
    const int64_t group = p.num_heads / p.num_kv_heads;
    const int64_t requests = p.batch * p.num_heads * p.num_splits;
    if (flat >= requests) return;

    const int64_t split = flat % p.num_splits;
    const int64_t head = (flat / p.num_splits) % p.num_heads;
    const int64_t batch = flat / (p.num_splits * p.num_heads);
    const int64_t h_kv = head / group;

    const T* q_row = reinterpret_cast<const T*>(p.q) + batch * p.q_stride.s_b +
                     head * p.q_stride.s_h;
    const T* k_base = reinterpret_cast<const T*>(p.k) + batch * p.k_stride.s_b +
                      h_kv * p.k_stride.s_h;
    const T* v_base = reinterpret_cast<const T*>(p.v) + batch * p.v_stride.s_b +
                      h_kv * p.v_stride.s_h;

    float q_slice[kLaneElems];
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
        q_slice[i] = ToFloat(q_row[lane * kLaneElems + i]);
    }
    float acc[kLaneElems];
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) acc[i] = 0.0f;
    float m = NegInf();
    float l = 0.0f;

    const int64_t seq_len = (p.seq_lens != nullptr) ? p.seq_lens[batch] : p.seq_kv;
    const int64_t chunk = (seq_len + p.num_splits - 1) / p.num_splits;
    const int64_t j_begin = split * chunk;
    const int64_t j_end = (j_begin + chunk < seq_len) ? (j_begin + chunk) : seq_len;

    for (int64_t j = j_begin; j < j_end; ++j) {
        const T* k_row = KvRowPtr(k_base, j, p.k_stride.s_s, p, batch);
        if (k_row == nullptr) continue;  // unmapped page: token contributes weight 0
        const T* v_row = KvRowPtr(v_base, j, p.v_stride.s_s, p, batch);
        if (v_row == nullptr) continue;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < kLaneElems; ++i) {
            dot += q_slice[i] * ToFloat(k_row[lane * kLaneElems + i]);
        }
        WarpReduceDot(&dot);
        dot *= p.scale;

        const float m_new = fmaxf(m, dot);
        const float alpha = (m == NegInf()) ? 0.0f : FastExp2((m - m_new) * kLog2e);
        const float p_j = FastExp2((dot - m_new) * kLog2e);
        l = alpha * l + p_j;
#pragma unroll
        for (int i = 0; i < kLaneElems; ++i) {
            acc[i] = alpha * acc[i] + p_j * ToFloat(v_row[lane * kLaneElems + i]);
        }
        m = m_new;
    }

    const int64_t partial_index = flat;
    if (lane == 0) {
        p.partial_m[partial_index] = m;
        p.partial_l[partial_index] = l;
    }
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
        p.partial_o[partial_index * kHeadDim + lane * kLaneElems + i] = acc[i];
    }
}

// One warp per (batch, head): merges the splits and writes the normalised output.
template <typename T, int kHeadDim>
__global__ void decode_reduce_kernel(DecodeFwdParams p) {
    constexpr int kLaneElems = kHeadDim / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t warps_per_request = static_cast<int64_t>(blockDim.x) >> 5;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * warps_per_request + warp;
    const int64_t requests = p.batch * p.num_heads;
    if (flat >= requests) return;

    const int64_t head = flat % p.num_heads;
    const int64_t batch = flat / p.num_heads;
    const int64_t base = flat * p.num_splits;

    float m_all = NegInf();
    for (int64_t s = 0; s < p.num_splits; ++s) {
        m_all = fmaxf(m_all, p.partial_m[base + s]);
    }
    float l_all = 0.0f;
    for (int64_t s = 0; s < p.num_splits; ++s) {
        const float m_s = p.partial_m[base + s];
        if (m_s == NegInf()) continue;
        l_all += p.partial_l[base + s] * FastExp2((m_s - m_all) * kLog2e);
    }

    T* out_row = reinterpret_cast<T*>(p.out) + batch * p.o_stride.s_b + head * p.o_stride.s_h;
    const float inv_l = l_all > 0.0f ? (1.0f / l_all) : 0.0f;
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
        const int d = lane * kLaneElems + i;
        float total = 0.0f;
        for (int64_t s = 0; s < p.num_splits; ++s) {
            const float m_s = p.partial_m[base + s];
            if (m_s == NegInf()) continue;
            const float weight = FastExp2((m_s - m_all) * kLog2e);
            total += p.partial_o[(base + s) * kHeadDim + d] * weight;
        }
        out_row[d] = FromFloatT<T>(total * inv_l);
    }
}

template <typename T, int kHeadDim>
sci::Status LaunchForDim(const DecodeFwdParams& p, sci::Stream* stream) {
    constexpr int kThreads = 128;  // 4 warps per CTA
    const int64_t partial_requests = p.batch * p.num_heads * p.num_splits;
    const int64_t reduce_requests = p.batch * p.num_heads;
    const int64_t partial_blocks = (partial_requests + 3) / 4;
    const int64_t reduce_blocks = (reduce_requests + 3) / 4;

    DecodeFwdParams params = p;
    {
        void* args[] = {&params};
        const sci::Status status = runtime::LaunchRaw(
            reinterpret_cast<const void*>(&decode_partial_kernel<T, kHeadDim>),
            static_cast<int>(partial_blocks), 1, 1, kThreads, 1, 1, 0, stream, args);
        if (!status.ok()) return status;
    }
    {
        void* args[] = {&params};
        const sci::Status status = runtime::LaunchRaw(
            reinterpret_cast<const void*>(&decode_reduce_kernel<T, kHeadDim>),
            static_cast<int>(reduce_blocks), 1, 1, kThreads, 1, 1, 0, stream, args);
        if (!status.ok()) return status;
    }
    SCI_CUDA_CHECK_LAST();
    return sci::Status::Ok();
}

template <typename T>
sci::Status LaunchTyped(const DecodeFwdParams& p, sci::Stream* stream) {
    switch (p.head_dim) {
        case 32: return LaunchForDim<T, 32>(p, stream);
        case 64: return LaunchForDim<T, 64>(p, stream);
        case 96: return LaunchForDim<T, 96>(p, stream);
        case 128: return LaunchForDim<T, 128>(p, stream);
        case 160: return LaunchForDim<T, 160>(p, stream);
        case 192: return LaunchForDim<T, 192>(p, stream);
        case 256: return LaunchForDim<T, 256>(p, stream);
        default: break;
    }
    return sci::Status::InvalidArgument("decode: head_dim=" + std::to_string(p.head_dim) +
                                        " is not instantiated");
}

}  // namespace

sci::Status LaunchDecodeSplitK(const DecodeFwdParams& params, sci::Stream* stream) {
    if (params.q == nullptr || params.k == nullptr || params.v == nullptr || params.out == nullptr ||
        params.partial_m == nullptr || params.partial_l == nullptr || params.partial_o == nullptr) {
        return sci::Status::InvalidArgument("decode: null pointer");
    }
    if (params.num_splits < 1) {
        return sci::Status::InvalidArgument("decode: num_splits must be >= 1");
    }
    switch (params.dtype) {
        case DecodeDtype::kFp32: return LaunchTyped<float>(params, stream);
        case DecodeDtype::kFp16: return LaunchTyped<__half>(params, stream);
        case DecodeDtype::kBf16: return LaunchTyped<__nv_bfloat16>(params, stream);
    }
    return sci::Status::InvalidArgument("decode: unsupported dtype id");
}

sci::Status LaunchDecodeSplitKPaged(const DecodeFwdParams& params, sci::Stream* stream) {
    if (params.page_table == nullptr) {
        return sci::Status::InvalidArgument(
            "paged attention: page_table is null (use the contiguous entry point instead)");
    }
    if (params.block_size <= 0 || params.max_blocks_per_seq <= 0 || params.num_kv_blocks <= 0) {
        return sci::Status::InvalidArgument(
            "paged attention: block_size/max_blocks_per_seq/num_kv_blocks must be positive");
    }
    return LaunchDecodeSplitK(params, stream);
}

}  // namespace cuda
}  // namespace sca
