// Flash (Level 3) backend adapter - kernels arrive in Phase 5.

#include "scicompute_attention/flash_attention.hpp"

#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"

namespace sca {
namespace {

constexpr const char* kNotImplementedReason =
    "flash backend kernels are implemented in Phase 5 (see TASK.md)";

class FlashBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kFlash; }
    const char* Name() const noexcept override { return "flash"; }

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
        return sci::MakeUnexpected<AttentionResult>(
            MakeStatus(AttnStatusCode::kUnsupportedFeature, kNotImplementedReason));
    }
};

const FlashBackend kFlashBackend;

}  // namespace

namespace runtime {
const IAttentionBackend* FlashBackendInstance() noexcept { return &kFlashBackend; }
}  // namespace runtime

sci::Result<AttentionResult> flash_attention(const sci::Tensor& q, const sci::Tensor& k,
                                             const sci::Tensor& v, const AttentionConfig& cfg) {
    return kFlashBackend.Forward(q, k, v, cfg);
}

sci::Result<AttentionResult> flash_attention_varlen(const sci::Tensor& q, const sci::Tensor& k,
                                                    const sci::Tensor& v, const int32_t* cu_q,
                                                    const int32_t* cu_kv, int64_t max_seq_q,
                                                    int64_t max_seq_kv,
                                                    const AttentionConfig& cfg) {
    return kFlashBackend.ForwardVarlen(q, k, v, cu_q, cu_kv, max_seq_q, max_seq_kv, cfg);
}

}  // namespace sca

