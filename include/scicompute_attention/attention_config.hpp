#pragma once

#include <cstddef>
#include <string>

#include "core/status.hpp"
#include "core/types.hpp"
#include "scicompute_attention/attention_types.hpp"
#include "scicompute_attention/detail/tile_config.hpp"
#include "scicompute_attention/export.hpp"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {

struct AttentionConfig {
    BackendKind backend{BackendKind::kAuto};
    AttnLayout layout{AttnLayout::kBHSD};
    bool causal{false};
    bool return_lse{false};
    float scale{0.0f};        // 0 => 1/sqrt(head_dim); resolved by EffectiveScale()
    float softcap{0.0f};      // 0 => disabled; non-zero requires the Phase 5 softcap path
    int64_t sliding_window{0};  // 0 => disabled (v1 returns kUnsupportedFeature when > 0)
    int64_t num_splits{0};      // decode split-K; 0 => derive from decode_config.hpp
    size_t workspace_limit_bytes{0};  // 0 => derive from DeviceCapability
    bool allow_fallback{false};       // default false: silent degradation is forbidden
    // varlen entry: host-resident int32 offsets of length num_seqs+1 (see flash_attention_varlen).
    const int32_t* cu_seqlens_q{nullptr};
    const int32_t* cu_seqlens_kv{nullptr};
    sci::Stream* stream{nullptr};           // nullptr => sci::Stream::GetCurrent()
};

// Resolved scale (1/sqrt(head_dim) when cfg.scale == 0).
SCI_ATTENTION_API float EffectiveScale(const AttentionConfig& cfg,
                                       const AttentionShape& shape) noexcept;

// Single validation entry point; every public API calls this first.
//
// Checks head_dim support, geometric invariants, causal/seq relation, dtype-independent shape
// bounds and feature availability. Error messages always carry the field name and its numeric
// value (prompt §6.2).
SCI_ATTENTION_API sci::Status Validate(const AttentionConfig& cfg, const AttentionShape& shape,
                                       std::string* detail = nullptr);

// Tile recommendation used by explain() and the benchmark tables.
SCI_ATTENTION_API TileConfig RecommendTile(const AttentionConfig& cfg, const AttentionShape& shape);

// Validates varlen metadata that lives on the host (monotonic, non-negative, length batch+1).
// The kernel side additionally clamps indices defensively.
SCI_ATTENTION_API sci::Status ValidateVarlenHost(const int32_t* host_cu_seqlens, int64_t batch,
                                                 int64_t* max_seq_out = nullptr);

}  // namespace sca
