#pragma once

#include <cstddef>
#include <string>

#include "core/status.hpp"

#include "scicompute_attention/attention_config.hpp"
#include "scicompute_attention/attention_result.hpp"
#include "scicompute_attention/export.hpp"

namespace sca {

struct DispatchDecision {
    BackendKind backend{BackendKind::kAuto};
    TileConfig tile{};
    size_t workspace_bytes{0};
    // Machine-readable prefix + human-readable explanation, e.g.
    //   "auto: seq_q=1 => decode (splits=8, seq_kv=8192)"
    std::string reason;
};

// Backend selection + execution. Selection is table driven (src/runtime/dispatch_table.inc,
// produced by tools/dispatch_threshold_sweep.py) so that Explain() and the actual execution can
// never drift apart.
class SCI_ATTENTION_API AttentionDispatcher {
public:
    AttentionDispatcher() = default;
    explicit AttentionDispatcher(const AttentionConfig& default_cfg) : default_cfg_(default_cfg) {}

    DispatchDecision Select(const AttentionConfig& cfg, const AttentionShape& shape) const;

    sci::Result<AttentionResult> Forward(const sci::Tensor& q, const sci::Tensor& k,
                                         const sci::Tensor& v,
                                         const AttentionConfig& cfg) const;

    sci::Result<AttentionResult> Forward(const sci::Tensor& q, const sci::Tensor& k,
                                         const sci::Tensor& v) const;

    // Human readable explanation of the decision (same code path as Select()).
    std::string Explain(const AttentionConfig& cfg, const AttentionShape& shape) const;

    // Reloads thresholds from a JSON file (nullptr restores the compiled-in table).
    static sci::Status ReloadDispatchTable(const char* path = nullptr);

    const AttentionConfig& DefaultConfig() const noexcept { return default_cfg_; }

private:
    AttentionConfig default_cfg_{};
};

// Shape of Q/K/V as seen through the tensor objects (layout aware).
SCI_ATTENTION_API sci::Result<AttentionShape> InferShape(const sci::Tensor& q, const sci::Tensor& k,
                                                         const sci::Tensor& v,
                                                         AttnLayout layout);

}  // namespace sca

