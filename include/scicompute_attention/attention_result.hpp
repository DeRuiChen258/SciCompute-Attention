#pragma once

#include <cstddef>
#include <string>

#include "tensor/tensor.hpp"

#include "scicompute_attention/attention_types.hpp"
#include "scicompute_attention/detail/tile_config.hpp"
#include "scicompute_attention/export.hpp"

namespace sca {

struct AttentionRuntimeStats {
    BackendKind used_backend{BackendKind::kAuto};
    TileConfig tile{};
    size_t workspace_bytes{0};
    size_t kv_bytes{0};
    // Empty string means "executed exactly as requested, no degradation". Any fallback,
    // approximation or clamped parameter must be described here.
    std::string note;
    // Populated by the paged/decode paths; 0 means "not applicable".
    int64_t num_splits{0};
    // Bytes of a materialized [B, H_q, S_q, S_kv] score matrix, i.e. what flash avoids.
    int64_t materialized_score_bytes{0};
};

struct AttentionResult {
    // Newly allocated, contiguous tensor in the internal compute layout (BHSD).
    sci::Tensor out;
    // Optional FP32 [B, H_q, S_q] tensor; empty unless cfg.return_lse was set.
    sci::Tensor lse;
    AttentionRuntimeStats stats;
};

}  // namespace sca

