#include "scicompute_attention/dispatcher.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/detail/host_utils.hpp"
#include "scicompute_attention/detail/layout_traits.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"
#include "runtime/dispatch_table.inc"

namespace sca {
namespace {

// Fallback candidates in decreasing order of preference. allow_fallback defaults to false, so this
// chain only runs when the caller explicitly opted in (prompt §6.2/§9.1 step 2).
std::vector<BackendKind> FallbackChain(BackendKind preferred) {
    switch (preferred) {
        case BackendKind::kFlash: return {BackendKind::kTiled, BackendKind::kNaive};
        case BackendKind::kDecode: return {BackendKind::kFlash, BackendKind::kTiled, BackendKind::kNaive};
        case BackendKind::kPaged: return {BackendKind::kDecode, BackendKind::kTiled, BackendKind::kNaive};
        case BackendKind::kTiled: return {BackendKind::kNaive};
        default: return {};
    }
}

std::string SummarizeShape(const AttentionShape& shape) {
    return "B=" + std::to_string(shape.batch) + ",Hq=" + std::to_string(shape.num_heads) +
           ",Hkv=" + std::to_string(shape.num_kv_heads) +
           ",Sq=" + std::to_string(shape.seq_q) + ",Skv=" + std::to_string(shape.seq_kv) +
           ",D=" + std::to_string(shape.head_dim);
}

}  // namespace

sci::Result<AttentionShape> InferShape(const sci::Tensor& q, const sci::Tensor& k,
                                       const sci::Tensor& v, AttnLayout layout) {
    auto fail = [](AttnStatusCode code, const std::string& msg) {
        return sci::MakeUnexpected<AttentionShape>(MakeStatus(code, msg));
    };
    if (!detail::LayoutSupported(layout)) {
        return fail(AttnStatusCode::kUnsupportedLayout,
                    std::string("layout=") + LayoutName(layout) + " (supported: bhsd, bshd)");
    }
    for (const auto* t : {&q, &k, &v}) {
        if (t->ndims() != 4) {
            return fail(AttnStatusCode::kShapeMismatch,
                        "tensor ndims=" + std::to_string(t->ndims()) + " (expected 4)");
        }
        if (!t->is_contiguous()) {
            return fail(AttnStatusCode::kLayoutMismatch,
                        "non-contiguous input rejected; call .clone() or pass a contiguous tensor");
        }
        if (!IsSupportedComputeDtype(t->dtype())) {
            return fail(AttnStatusCode::kUnsupportedDtype,
                        std::string("dtype=") + sci::kDTypeName(t->dtype()) +
                            " (supported: float32, float16, bfloat16)");
        }
        if (t->device_type() != sci::DeviceType::kCUDA) {
            return fail(AttnStatusCode::kUnsupportedFeature,
                        std::string("device=") + sci::kDeviceTypeName(t->device_type()) +
                            " (v1 executes attention kernels on CUDA devices only)");
        }
    }
    if (q.dtype() != k.dtype() || q.dtype() != v.dtype()) {
        return fail(AttnStatusCode::kUnsupportedDtype,
                    std::string("dtype mismatch: q=") + sci::kDTypeName(q.dtype()) +
                        ", k=" + sci::kDTypeName(k.dtype()) + ", v=" + sci::kDTypeName(v.dtype()));
    }

    AttentionShape shape;
    if (layout == AttnLayout::kBHSD) {
        shape.batch = q.dim(0);
        shape.num_heads = q.dim(1);
        shape.seq_q = q.dim(2);
        shape.head_dim = q.dim(3);
        shape.seq_kv = k.dim(2);
        shape.num_kv_heads = k.dim(1);
        if (v.dim(1) != shape.num_kv_heads || v.dim(2) != shape.seq_kv || v.dim(3) != shape.head_dim) {
            return fail(AttnStatusCode::kShapeMismatch,
                        "k/v shape mismatch: k=[B,Hkv,Skv,D]=" + SummarizeShape(shape) + ", v=[" +
                            std::to_string(v.dim(0)) + "," + std::to_string(v.dim(1)) + "," +
                            std::to_string(v.dim(2)) + "," + std::to_string(v.dim(3)) + "]");
        }
    } else {
        shape.batch = q.dim(0);
        shape.seq_q = q.dim(1);
        shape.num_heads = q.dim(2);
        shape.head_dim = q.dim(3);
        shape.seq_kv = k.dim(1);
        shape.num_kv_heads = k.dim(2);
        if (v.dim(1) != shape.seq_kv || v.dim(2) != shape.num_kv_heads || v.dim(3) != shape.head_dim) {
            return fail(AttnStatusCode::kShapeMismatch,
                        "k/v shape mismatch for bshd: v=[" + std::to_string(v.dim(0)) + "," +
                            std::to_string(v.dim(1)) + "," + std::to_string(v.dim(2)) + "," +
                            std::to_string(v.dim(3)) + "]");
        }
    }
    if (k.dim(0) != shape.batch || v.dim(0) != shape.batch) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "batch mismatch: q=" + std::to_string(shape.batch) +
                        ", k=" + std::to_string(k.dim(0)) + ", v=" + std::to_string(v.dim(0)));
    }
    return sci::Ok(shape);
}

DispatchDecision AttentionDispatcher::Select(const AttentionConfig& cfg,
                                             const AttentionShape& shape) const {
    DispatchDecision decision;
    std::string detail;
    const sci::Status valid = Validate(cfg, shape, &detail);
    if (!valid.ok()) {
        decision.backend = BackendKind::kAuto;
        decision.reason = "invalid: " + detail;
        return decision;
    }

    const auto& th = detail::kDispatchThresholds;
    BackendKind preferred = cfg.backend;
    std::string prefix;
    std::string why;

    if (preferred != BackendKind::kAuto) {
        prefix = "explicit";
        why = "backend=" + std::string(BackendName(preferred));
    } else if (shape.seq_q <= th.decode_max_seq_q && shape.seq_kv >= th.decode_min_seq_kv) {
        preferred = BackendKind::kDecode;
        prefix = "auto";
        why = "seq_q=" + std::to_string(shape.seq_q) +
              " <=" + std::to_string(th.decode_max_seq_q) + " and seq_kv=" +
              std::to_string(shape.seq_kv) + " >= " + std::to_string(th.decode_min_seq_kv) +
              " => decode (low arithmetic intensity)";
    } else if (shape.seq_q * 8 <= shape.seq_kv && shape.seq_kv >= th.decode_min_seq_kv) {
        // Grouped small-q (S_q in 2..8) has enough arithmetic intensity for the tiled MMA path, and
        // the decode kernel is specialised for S_q == 1, so it is routed to flash here. This is the
        // seed rule of dispatch_table.inc r0; the sweep may refine the boundary.
        preferred = BackendKind::kFlash;
        prefix = "auto";
        why = "seq_q*8=" + std::to_string(shape.seq_q * 8) + " <= seq_kv=" +
              std::to_string(shape.seq_kv) + " => grouped small-q prefill on flash "
              "(decode kernel is S_q=1 only)";
    } else if (shape.seq_kv <= th.tiled_max_seq_kv) {
        preferred = BackendKind::kTiled;
        prefix = "auto";
        why = "seq_kv=" + std::to_string(shape.seq_kv) + " <= " +
              std::to_string(th.tiled_max_seq_kv) + " => tiled (short sequence, smem resident)";
    } else {
        preferred = BackendKind::kFlash;
        prefix = "auto";
        why = "seq_q=" + std::to_string(shape.seq_q) + ", seq_kv=" + std::to_string(shape.seq_kv) +
              " => flash (tile + online softmax)";
    }

    decision.backend = preferred;
    decision.tile = RecommendTile(cfg, shape);

    const IAttentionBackend* backend = GetBackend(preferred);
    CapabilityReport report = backend->Supports(cfg, shape);
    decision.workspace_bytes = report.workspace_bytes;

    if (report.supported) {
        if (!report.tile.Empty()) decision.tile = report.tile;
        decision.reason = prefix + ": " + why + " [table=" + th.revision +
                          ", workspace=" + std::to_string(decision.workspace_bytes) + " B]";
        return decision;
    }

    if (!cfg.allow_fallback) {
        decision.reason = prefix + ": " + why + " => unsupported(" +
                          std::string(BackendName(preferred)) + "): " + report.reason +
                          " [table=" + th.revision + ", allow_fallback=false]";
        return decision;
    }

    for (BackendKind candidate : FallbackChain(preferred)) {
        const IAttentionBackend* alt = GetBackend(candidate);
        if (alt == nullptr) continue;
        const CapabilityReport alt_report = alt->Supports(cfg, shape);
        if (alt_report.supported) {
            decision.backend = candidate;
            if (!alt_report.tile.Empty()) decision.tile = alt_report.tile;
            decision.workspace_bytes = alt_report.workspace_bytes;
            decision.reason = "fallback: " + std::string(BackendName(preferred)) + " unsupported (" +
                              report.reason + ") => " + std::string(BackendName(candidate)) + "; " +
                              prefix + ": " + why + " [table=" + th.revision +
                              ", allow_fallback=true]";
            return decision;
        }
    }

    decision.backend = preferred;
    decision.reason = prefix + ": " + why + " => unsupported(" +
                      std::string(BackendName(preferred)) + "): " + report.reason +
                      "; no fallback backend supports shape " + SummarizeShape(shape);
    return decision;
}

std::string AttentionDispatcher::Explain(const AttentionConfig& cfg,
                                         const AttentionShape& shape) const {
    const DispatchDecision decision = Select(cfg, shape);
    return "backend=" + std::string(BackendName(decision.backend)) +
           "; tile={" + decision.tile.ToString() + "}; workspace=" +
           std::to_string(decision.workspace_bytes) + " B; reason=" + decision.reason;
}

sci::Result<AttentionResult> AttentionDispatcher::Forward(const sci::Tensor& q,
                                                          const sci::Tensor& k,
                                                          const sci::Tensor& v,
                                                          const AttentionConfig& cfg) const {
    const sci::Result<AttentionShape> shape = InferShape(q, k, v, cfg.layout);
    if (!shape.ok()) return sci::MakeUnexpected<AttentionResult>(shape.error());

    const DispatchDecision decision = Select(cfg, *shape);
    if (decision.backend == BackendKind::kAuto) {
        return sci::MakeUnexpected<AttentionResult>(
            MakeStatus(AttnStatusCode::kShapeMismatch, decision.reason));
    }
    const IAttentionBackend* backend = GetBackend(decision.backend);
    if (backend == nullptr) {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kDeviceCapability,
            std::string("no backend registered for ") + BackendName(decision.backend)));
    }
    const CapabilityReport report = backend->Supports(cfg, *shape);
    if (!report.supported) {
        return sci::MakeUnexpected<AttentionResult>(
            MakeStatus(AttnStatusCode::kDeviceCapability, report.reason));
    }
    return backend->Forward(q, k, v, cfg);
}

sci::Result<AttentionResult> AttentionDispatcher::Forward(const sci::Tensor& q,
                                                          const sci::Tensor& k,
                                                          const sci::Tensor& v) const {
    return Forward(q, k, v, default_cfg_);
}

sci::Status AttentionDispatcher::ReloadDispatchTable(const char* path) {
    if (path == nullptr) return sci::Status::Ok();
    return sci::Status::NotImplemented(
        "runtime dispatch-table reload requires the threshold sweep tool; compiled-in table "
        "\"" +
        std::string(detail::kDispatchThresholds.revision) + "\" is in use");
}

}  // namespace sca
