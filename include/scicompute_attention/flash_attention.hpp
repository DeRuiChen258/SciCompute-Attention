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

// Variable-length entry point. cu_seqlens_* are device-resident int32 arrays of length batch+1
// describing flattened token offsets (see docs/flash_attention.md); host-side metadata must be
// validated with ValidateVarlenHost() before the call.
SCI_ATTENTION_API sci::Result<AttentionResult> flash_attention_varlen(
    const sci::Tensor& q, const sci::Tensor& k, const sci::Tensor& v,
    const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_kv, int64_t max_seq_q,
    int64_t max_seq_kv, const AttentionConfig& cfg);

}  // namespace sca

