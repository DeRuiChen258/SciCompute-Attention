#pragma once

#include "scicompute_attention/attention_config.hpp"
#include "scicompute_attention/attention_result.hpp"
#include "scicompute_attention/export.hpp"

namespace sca {

// Level 3: IO-aware flash attention. The N x N score matrix never reaches HBM; the online
// softmax state (m, l, O) lives in registers and is rescaled per K/V tile.
SCI_ATTENTION_API sci::Result<AttentionResult> flash_attention(const sci::Tensor& q,
                                                               const sci::Tensor& k,
                                                               const sci::Tensor& v,
                                                               const AttentionConfig& cfg);

// Variable-length entry point for packed batches (see docs/flash_attention.md):
//   * q is [total_q_tokens, H_q, D]; k/v are [total_kv_tokens, H_kv, D]; all contiguous;
//   * cu_seqlens_q/kv are **host** int32 arrays of length num_seqs + 1, strictly increasing and
//     starting at 0 (validate them with ValidateVarlenHost());
//   * num_seqs is explicit: the packed shapes cannot reveal how many sequences are present.
//     (The prompt's §6.5 signature omits it, which makes the call under-determined - see
//     .agent/decisions.md D-008.)
//   * LSE output is not supported on this entry point in v1 (kUnsupportedFeature).
SCI_ATTENTION_API sci::Result<AttentionResult> flash_attention_varlen(
    const sci::Tensor& q, const sci::Tensor& k, const sci::Tensor& v,
    int64_t num_seqs, const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_kv,
    int64_t max_seq_q, int64_t max_seq_kv, const AttentionConfig& cfg);

}  // namespace sca
