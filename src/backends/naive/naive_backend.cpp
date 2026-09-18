// Naive (Level 0) backend adapter.
//
// Phase 1 status: the API surface is in place, the kernels arrive in Phase 2 together with
// tests/kernel/test_naive_correctness.cu. Until then Supports() reports the explicit reason so
// that the dispatcher can never silently run an unimplemented path.

#include "scicompute_attention/naive_attention.hpp"

#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"

namespace sca {
namespace {

constexpr const char* kNotImplementedReason =
    "naive backend kernels are implemented in Phase 2 (see TASK.md); "
    "no fallback is attempted because allow_fallback defaults to false";

class NaiveBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kNaive; }
    const char* Name() const noexcept override { return "naive"; }

    CapabilityReport Supports(const AttentionConfig&, const AttentionShape&) const override {
        CapabilityReport report;
        report.supported = false;
        report.reason = kNotImplementedReason;
        report.recommended_backend = BackendKind::kNaive;
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
            AttnStatusCode::kUnsupportedFeature,
            std::string("naive does not implement the varlen entry point (flash does)")));
    }
};

const NaiveBackend kNaiveBackend;

}  // namespace

namespace runtime {
const IAttentionBackend* NaiveBackendInstance() noexcept { return &kNaiveBackend; }
}  // namespace runtime

sci::Result<AttentionResult> naive_attention(const sci::Tensor& q, const sci::Tensor& k,
                                             const sci::Tensor& v, const AttentionConfig& cfg) {
    return kNaiveBackend.Forward(q, k, v, cfg);
}

}  // namespace sca

