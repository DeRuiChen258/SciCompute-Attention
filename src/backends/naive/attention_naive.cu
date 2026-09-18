// Level 0 (naive) attention implementation.
//
// Kernel map
//   naive_qk_scale_mask_kernel<T> : S[b,h,i,j] = scale * dot(Q[b,h,i,:], K[b,g,j,:]) with
//                                   causal masking (j > i + diag => -inf); FP32 accumulation.
//   naive_row_softmax_kernel      : in-place stable row softmax over the S_kv axis; optionally
//                                   writes LSE = m + log(l).
//   naive_pv_kernel<T>            : O[b,h,i,:] = sum_j P[b,h,i,j] * V[b,g,j,:], cast back to T.
//
// math: S = Q K^T / sqrt(D); P = softmax(S); O = P V
// math: causal convention (bottom-right aligned):
//         j <= i + diag,  diag = S_kv - S_q
//       For S_q == S_kv this is the familiar j <= i; for S_q == 1 (decode) it means "attend to
//       every cached token", which is what a rollout decode step needs.
//
// Register/thread mapping: one thread owns one (or a strided set of) output element(s); no shared
// memory tiling is used here on purpose - this backend is the numerical oracle and its IO pattern
// is the *baseline* that flash attention is compared against.

#include "backends/naive/attention_naive.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda_common/cuda_check.cuh"
#include "cuda_common/numerics.cuh"
#include "device/stream.hpp"
#include "runtime/launcher.hpp"

namespace sca {
namespace cuda {
namespace {

constexpr int kThreads = 256;

// ---------------------------------------------------------------------------------------------
// [1] QK^T with scale and causal mask.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void naive_qk_scale_mask_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                           float* __restrict__ scores, NaiveFwdParams p) {
    const int64_t total = p.batch * p.num_heads * p.seq_q * p.seq_kv;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t group = p.num_heads / p.num_kv_heads;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        const int64_t j = idx % p.seq_kv;
        const int64_t i = (idx / p.seq_kv) % p.seq_q;
        const int64_t h = (idx / (p.seq_kv * p.seq_q)) % p.num_heads;
        const int64_t b = idx / (p.seq_kv * p.seq_q * p.num_heads);
        const int64_t h_kv = h / group;

        const T* q_row = q + b * p.q_stride.s_b + h * p.q_stride.s_h + i * p.q_stride.s_s;
        const T* k_row = k + b * p.k_stride.s_b + h_kv * p.k_stride.s_h + j * p.k_stride.s_s;

        float acc = 0.0f;
        // TODO(perf): vectorize with float4/half8 once the reference path is verified; the naive
        // backend exists for correctness and as an IO baseline, not for speed.
        for (int64_t d = 0; d < p.head_dim; ++d) {
            acc += ToFloat(q_row[d]) * ToFloat(k_row[d]);
        }
        acc *= p.scale;
        if (p.causal && j > i + p.diagonal) acc = NegInf();
        scores[idx] = acc;
    }
}

// ---------------------------------------------------------------------------------------------
// [2] Row softmax over S_kv (block per row).
// ---------------------------------------------------------------------------------------------
__global__ void naive_row_softmax_kernel(float* __restrict__ scores, float* __restrict__ lse,
                                         int64_t rows, int64_t cols, bool write_lse) {
    __shared__ float scratch[32];
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    float* data = scores + row * cols;

    float local_max = NegInf();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, data[c]);
    }
    const float m = BlockReduceMax(local_max, scratch);

    // A fully masked row (m == -inf) is defined to produce zeros rather than NaNs.
    const bool all_masked = !IsFinite(m);
    float local_sum = 0.0f;
    if (!all_masked) {
        for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
            const float value = FastExp2((data[c] - m) * kLog2e);
            data[c] = value;
            local_sum += value;
        }
    }
    const float sum = all_masked ? 1.0f : BlockReduceSum(local_sum, scratch);
    const float inv_sum = all_masked ? 0.0f : (1.0f / sum);

    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        data[c] *= inv_sum;
    }
    if (write_lse && threadIdx.x == 0) {
        lse[row] = all_masked ? NegInf() : (m + logf(sum));
    }
}

// ---------------------------------------------------------------------------------------------
// [3] P V with output cast back to the input dtype.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void naive_pv_kernel(const float* __restrict__ probs, const T* __restrict__ v,
                                T* __restrict__ out, NaiveFwdParams p) {
    const int64_t total = p.batch * p.num_heads * p.seq_q * p.head_dim;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t group = p.num_heads / p.num_kv_heads;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        const int64_t d = idx % p.head_dim;
        const int64_t i = (idx / p.head_dim) % p.seq_q;
        const int64_t h = (idx / (p.head_dim * p.seq_q)) % p.num_heads;
        const int64_t b = idx / (p.head_dim * p.seq_q * p.num_heads);
        const int64_t h_kv = h / group;

        const float* p_row = probs + ((b * p.num_heads + h) * p.seq_q + i) * p.seq_kv;
        const T* v_col = v + b * p.v_stride.s_b + h_kv * p.v_stride.s_h + d;

        float acc = 0.0f;
        for (int64_t j = 0; j < p.seq_kv; ++j) {
            acc += p_row[j] * ToFloat(v_col[j * p.v_stride.s_s]);
        }
        out[b * p.o_stride.s_b + h * p.o_stride.s_h + i * p.o_stride.s_s + d] = FromFloatT<T>(acc);
    }
}

template <typename T>
sci::Status LaunchTyped(const NaiveFwdParams& p, sci::Stream* stream) {
    const int64_t qk_total = p.batch * p.num_heads * p.seq_q * p.seq_kv;
    const int64_t pv_total = p.batch * p.num_heads * p.seq_q * p.head_dim;
    const int qk_blocks = static_cast<int>(std::min<int64_t>(
        std::max<int64_t>(1, (qk_total + kThreads - 1) / kThreads), 1 << 20));
    const int pv_blocks = static_cast<int>(std::min<int64_t>(
        std::max<int64_t>(1, (pv_total + kThreads - 1) / kThreads), 1 << 20));
    const int64_t rows = p.batch * p.num_heads * p.seq_q;
    const int softmax_blocks = static_cast<int>(std::min<int64_t>(std::max<int64_t>(1, rows), 1 << 24));

    // cudaLaunchKernel takes an array of argument addresses, so every argument needs a stable,
    // non-const local copy.
    const T* q_ptr = static_cast<const T*>(p.q);
    const T* k_ptr = static_cast<const T*>(p.k);
    const T* v_ptr = static_cast<const T*>(p.v);
    T* out_ptr = static_cast<T*>(p.out);
    float* scores_ptr = p.scores;
    float* lse_ptr = p.lse;
    NaiveFwdParams params = p;
    params.scores = scores_ptr;
    params.lse = lse_ptr;
    const int64_t rows_local = rows;
    const int64_t cols_local = p.seq_kv;
    const bool write_lse = p.return_lse;

    {
        void* args[] = {&q_ptr, &k_ptr, &scores_ptr, &params};
        const sci::Status status = runtime::LaunchRaw(
            reinterpret_cast<const void*>(&naive_qk_scale_mask_kernel<T>), qk_blocks, 1, 1,
            kThreads, 1, 1, 0, stream, args);
        if (!status.ok()) return status;
    }
    {
        void* args[] = {&scores_ptr, &lse_ptr, const_cast<int64_t*>(&rows_local),
                        const_cast<int64_t*>(&cols_local), const_cast<bool*>(&write_lse)};
        const sci::Status status = runtime::LaunchRaw(
            reinterpret_cast<const void*>(&naive_row_softmax_kernel), softmax_blocks, 1, 1,
            kThreads, 1, 1, 0, stream, args);
        if (!status.ok()) return status;
    }
    {
        void* args[] = {&scores_ptr, &v_ptr, &out_ptr, &params};
        const sci::Status status = runtime::LaunchRaw(
            reinterpret_cast<const void*>(&naive_pv_kernel<T>), pv_blocks, 1, 1, kThreads, 1, 1, 0,
            stream, args);
        if (!status.ok()) return status;
    }
    SCI_CUDA_CHECK_LAST();
    return sci::Status::Ok();
}

}  // namespace

sci::Status LaunchNaiveForward(const NaiveFwdParams& params, sci::Stream* stream) {
    if (params.q == nullptr || params.k == nullptr || params.v == nullptr ||
        params.out == nullptr || params.scores == nullptr) {
        return sci::Status::InvalidArgument("naive attention: null tensor pointer");
    }
    switch (params.dtype) {
        case NaiveDtype::kFp32: return LaunchTyped<float>(params, stream);
        case NaiveDtype::kFp16: return LaunchTyped<__half>(params, stream);
        case NaiveDtype::kBf16: return LaunchTyped<__nv_bfloat16>(params, stream);
    }
    return sci::Status::InvalidArgument("naive attention: unsupported dtype id");
}

}  // namespace cuda
}  // namespace sca
