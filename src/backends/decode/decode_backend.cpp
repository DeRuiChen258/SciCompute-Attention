// Decode (Level 4) backend adapter - kernels arrive in Phase 8.

#include "scicompute_attention/decode_attention.hpp"

#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/detail/host_utils.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"
#include "runtime/dispatch_table.inc"

namespace sca {
namespace {

constexpr const char* kNotImplementedReason =
    "decode backend kernels are implemented in Phase 8 (see TASK.md)";

class DecodeBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kDecode; }
    const char* Name() const noexcept override { return "decode"; }

    CapabilityReport Supports(const AttentionConfig&, const AttentionShape&) const override {
        CapabilityReport report;
        report.supported = false;
        report.reason = kNotImplementedReason;
        report.recommended_backend = BackendKind::kFlash;
        return report;
    }

    size_t WorkspaceBytes(const AttentionConfig&, const AttentionShape&) const override {
        return 0;
    }

    sci::Result<AttentionResult> Forward(const sci::Tensor&, const sci::Tensor&,
                                         const sci::Tensor&,
                                         const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(
            MakeStatus(AttnStatusCode::kUnsupportedFeature, kNotImplementedReason));
    }

    sci::Result<AttentionResult> ForwardVarlen(const sci::Tensor&, const sci::Tensor&,
                                               const sci::Tensor&, int64_t, const int32_t*,
                                               const int32_t*, int64_t, int64_t,
                                               const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedFeature, "decode does not implement varlen; use flash"));
    }
};

const DecodeBackend kDecodeBackend;

}  // namespace

namespace runtime {
const IAttentionBackend* DecodeBackendInstance() noexcept { return &kDecodeBackend; }
}  // namespace runtime

DecodeConfig RecommendDecodeConfig(const AttentionShape& shape, const DeviceCapability& cap) {
    DecodeConfig cfg;
    cfg.max_splits = detail::kDispatchThresholds.max_splits;
    const int64_t tokens_per_split = detail::kDispatchThresholds.tokens_per_split;
    if (tokens_per_split <= 0 || shape.seq_kv <= 0) {
        cfg.num_splits = 1;
        return cfg;
    }
    int64_t splits = (shape.seq_kv + tokens_per_split - 1) / tokens_per_split;
    // Do not create more splits than the device can keep busy with 2 CTAs per SM.
    const int64_t concurrency_cap = static_cast<int64_t>(cap.sm_count) * 2;
    if (concurrency_cap > 0) splits = detail::Min(splits, concurrency_cap);
    cfg.num_splits = detail::Max<int64_t>(1, detail::Min(splits, cfg.max_splits));
    return cfg;
}

sci::Result<AttentionResult> decode_attention(const sci::Tensor& q, const sci::Tensor& k,
                                              const sci::Tensor& v, const AttentionConfig& cfg) {
    return kDecodeBackend.Forward(q, k, v, cfg);
}

sci::Result<AttentionResult> decode_attention(const sci::Tensor& q, const sci::Tensor& k,
                                              const sci::Tensor& v, const AttentionConfig& cfg,
                                              const DecodeConfig& decode_cfg) {
    AttentionConfig resolved = cfg;
    resolved.num_splits = decode_cfg.num_splits;
    return kDecodeBackend.Forward(q, k, v, resolved);
}

}  // namespace sca
