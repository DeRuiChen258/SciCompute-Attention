#pragma once

#include "scicompute_attention/attention_config.hpp"
#include "scicompute_attention/attention_result.hpp"
#include "scicompute_attention/capability.hpp"
#include "scicompute_attention/export.hpp"

namespace sca {

struct DecodeConfig {
    int64_t num_splits{0};   // 0 => automatic (see src/backends/decode/decode_config.hpp)
    int64_t max_splits{16};  // hard cap: keeps the partial workspace bounded
    bool use_splitk{true};   // false => single-block reduction (A/B comparison path)
};

SCI_ATTENTION_API DecodeConfig RecommendDecodeConfig(const AttentionShape& shape,
                                                     const DeviceCapability& cap);

// Level 4: decode attention (S_q small, S_kv large, low arithmetic intensity). Split-K partials
// (m, l, O) are written to workspace and merged with the stable rescaling formula.
SCI_ATTENTION_API sci::Result<AttentionResult> decode_attention(const sci::Tensor& q,
                                                                const sci::Tensor& k,
                                                                const sci::Tensor& v,
                                                                const AttentionConfig& cfg);

SCI_ATTENTION_API sci::Result<AttentionResult> decode_attention(const sci::Tensor& q,
                                                                const sci::Tensor& k,
                                                                const sci::Tensor& v,
                                                                const AttentionConfig& cfg,
                                                                const DecodeConfig& decode_cfg);

}  // namespace sca

