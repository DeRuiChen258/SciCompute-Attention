// Level 0 (naive) attention kernels: the correctness reference.
//
// Design notes (prompt §7.2):
//   * S = Q K^T is *materialized* in FP32 global memory - on purpose. The byte count of that
//     buffer (B*H*S_q*S_kv*4) is reported in AttentionRuntimeStats.materialized_score_bytes and is
//     the baseline that the tiled/flash backends are measured against.
//   * mask + scale are fused into the QK kernel (explicitly allowed by the prompt), the row
//     softmax runs separately so that it can be reused as an oracle, and PV closes the pipeline.
//   * FP16/BF16 inputs still accumulate in FP32; only the load/store path uses the narrow type.
//   * Strides are passed explicitly, so both BHSD and BSHD inputs are supported without a copy.
#pragma once

#include <cstdint>

#include "core/status.hpp"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {
namespace cuda {

struct StridesFwd {
    int64_t s_b{0};
    int64_t s_h{0};
    int64_t s_s{0};
};

// Dtype ids kept in sync with sca::detail::kDtypeId* (host side maps sci::DType).
enum class NaiveDtype : int32_t { kFp32 = 0, kFp16 = 1, kBf16 = 2 };

struct NaiveFwdParams {
    const void* q{nullptr};
    const void* k{nullptr};
    const void* v{nullptr};
    void* out{nullptr};
    float* scores{nullptr};  // FP32 workspace, B*H*S_q*S_kv elements
    int64_t batch{0};
    int64_t num_heads{0};
    int64_t num_kv_heads{0};
    int64_t seq_q{0};
    int64_t seq_kv{0};
    int64_t head_dim{0};
    StridesFwd q_stride{};
    StridesFwd k_stride{};
    StridesFwd v_stride{};
    StridesFwd o_stride{};  // output is always contiguous BHSD
    float scale{0.0f};
    bool causal{false};
    bool return_lse{false};
    // Causal diagonal offset (bottom-right aligned): a position j is visible from query i when
    // j <= i + diagonal. Host side sets it to seq_kv - seq_q for causal runs and 0 otherwise.
    int64_t diagonal{0};
    float* lse{nullptr};  // optional FP32 [B, H, S_q]
    NaiveDtype dtype{NaiveDtype::kFp32};
};

// Launches the three naive kernels in order. `stream == nullptr` means Stream::GetCurrent().
sci::Status LaunchNaiveForward(const NaiveFwdParams& params, sci::Stream* stream);

}  // namespace cuda
}  // namespace sca
