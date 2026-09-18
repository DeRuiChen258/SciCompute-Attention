// Tiled (Level 1) backend adapter - kernel arrives in Phase 3.

#include "scicompute_attention/tiled_attention.hpp"

#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"

namespace sca {
namespace {

constexpr const char* kNotImplementedReason =
    "tiled backend kernels are implemented in Phase 3 (see TASK.md)";

class TiledBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kTiled; }
    const char* Name() const noexcept override { return "tiled"; }

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
                                               const sci::Tensor&, const int32_t*, const int32_t*,
                                               int64_t, int64_t,
                                               const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedFeature, "tiled does not implement varlen; use flash"));
    }
};

const TiledBackend kTiledBackend;

}  // namespace

namespace runtime {
const IAttentionBackend* TiledBackendInstance() noexcept { return &kTiledBackend; }
}  // namespace runtime

sci::Result<AttentionResult> tiled_attention(const sci::Tensor& q, const sci::Tensor& k,
                                             const sci::Tensor& v, const AttentionConfig& cfg) {
    return kTiledBackend.Forward(q, k, v, cfg);
}

}  // namespace sca

