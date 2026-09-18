// Paged (Level 5) backend adapter - kernels arrive in Phase 9.

#include "scicompute_attention/paged_attention.hpp"

#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"

namespace sca {
namespace {

constexpr const char* kNotImplementedReason =
    "paged backend kernels are implemented in Phase 9 (see TASK.md)";

class PagedBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kPaged; }
    const char* Name() const noexcept override { return "paged"; }

    CapabilityReport Supports(const AttentionConfig&, const AttentionShape&) const override {
        CapabilityReport report;
        report.supported = false;
        report.reason = kNotImplementedReason;
        report.recommended_backend = BackendKind::kDecode;
        return report;
    }

    size_t WorkspaceBytes(const AttentionConfig&, const AttentionShape&) const override {
        return 0;
    }

    sci::Result<AttentionResult> Forward(const sci::Tensor&, const sci::Tensor&,
                                         const sci::Tensor&,
                                         const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedFeature,
            "paged attention must be called through sca::paged_attention(q, kv, params, cfg)"));
    }

    sci::Result<AttentionResult> ForwardVarlen(const sci::Tensor&, const sci::Tensor&,
                                               const sci::Tensor&, int64_t, const int32_t*,
                                               const int32_t*, int64_t, int64_t,
                                               const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(
            MakeStatus(AttnStatusCode::kUnsupportedFeature, kNotImplementedReason));
    }
};

const PagedBackend kPagedBackend;

sci::Result<AttentionResult> PagedForwardImpl(const sci::Tensor& q, const PagedKVCache& kv,
                                              const PagedAttentionParams& params,
                                              const AttentionConfig& cfg) {
    (void)q;
    (void)kv;
    (void)params;
    (void)cfg;
    return sci::MakeUnexpected<AttentionResult>(
        MakeStatus(AttnStatusCode::kUnsupportedFeature, kNotImplementedReason));
}

}  // namespace

namespace runtime {
const IAttentionBackend* PagedBackendInstance() noexcept { return &kPagedBackend; }
}  // namespace runtime

sci::Result<AttentionResult> paged_attention(const sci::Tensor& q, const PagedKVCache& kv,
                                             const PagedAttentionParams& params,
                                             const AttentionConfig& cfg) {
    return PagedForwardImpl(q, kv, params, cfg);
}

sci::Result<AttentionResult> paged_attention(const sci::Tensor& q, const PagedKVCache& kv,
                                             const AttentionConfig& cfg) {
    // Phase 9 fills in the page-table hand-off from the cache object; until PagedKVCache has a
    // device page table (Phase 7), the parameters stay default-initialised on purpose so that no
    // partially initialised cache can be consumed silently.
    PagedAttentionParams params;
    return PagedForwardImpl(q, kv, params, cfg);
}

}  // namespace sca
