#pragma once

#include "scicompute_attention/attention_config.hpp"
#include "scicompute_attention/attention_result.hpp"
#include "scicompute_attention/export.hpp"

namespace sca {

// Level 0: correctness reference. Materializes S = Q K^T in workspace and runs an explicit row
// softmax, so it is the slowest and the most memory hungry backend. FP32 keeps FP32 storage for
// S/P; FP16/BF16 inputs still accumulate in FP32 (see docs/kernel_design.md).
SCI_ATTENTION_API sci::Result<AttentionResult> naive_attention(const sci::Tensor& q,
                                                               const sci::Tensor& k,
                                                               const sci::Tensor& v,
                                                               const AttentionConfig& cfg);

}  // namespace sca

