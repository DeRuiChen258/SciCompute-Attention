#pragma once

#include "scicompute_attention/attention_config.hpp"
#include "scicompute_attention/attention_result.hpp"
#include "scicompute_attention/export.hpp"

namespace sca {

// Level 1: shared-memory tiling with register accumulation; S is tiled in shared memory instead
// of being materialized in HBM. Rejects configurations whose smem needs exceed the device limit
// (returns kDeviceCapability and recommends flash).
SCI_ATTENTION_API sci::Result<AttentionResult> tiled_attention(const sci::Tensor& q,
                                                               const sci::Tensor& k,
                                                               const sci::Tensor& v,
                                                               const AttentionConfig& cfg);

}  // namespace sca

