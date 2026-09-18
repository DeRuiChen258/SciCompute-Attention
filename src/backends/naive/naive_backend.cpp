// Naive (Level 0) backend adapter: validation -> workspace sizing -> kernel launch -> stats.
//
// This backend is the numerical oracle of the project and the IO baseline for docs/roofline.md:
// it deliberately materializes the [B, H_q, S_q, S_kv] score matrix in FP32 HBM, which is exactly
// what the flash path removes.

#include "scicompute_attention/naive_attention.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/capability.hpp"
#include "scicompute_attention/detail/dtype_traits.hpp"
#include "scicompute_attention/detail/host_utils.hpp"
#include "scicompute_attention/detail/layout_traits.hpp"
#include "scicompute_attention/dispatcher.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"
#include "runtime/scratch_pool.hpp"

#include "backends/naive/attention_naive.cuh"

namespace sca {
namespace {

using cuda::NaiveDtype;
using cuda::NaiveFwdParams;
using cuda::StridesFwd;

bool DtypeIdFor(sci::DType dtype, NaiveDtype* out) {
    switch (dtype) {
        case sci::DType::kFloat32: *out = NaiveDtype::kFp32; return true;
        case sci::DType::kFloat16: *out = NaiveDtype::kFp16; return true;
        case sci::DType::kBFloat16: *out = NaiveDtype::kBf16; return true;
        default: return false;
    }
}

StridesFwd ToStrides(const detail::Strides& s) {
    StridesFwd out;
    out.s_b = s.s_b;
    out.s_h = s.s_h;
    out.s_s = s.s_s;
    return out;
}

// Score matrix bytes: B * H_q * S_q * S_kv * sizeof(float) - the buffer flash attention avoids.
size_t ScoreBytes(const AttentionShape& shape) {
    return static_cast<size_t>(shape.batch) * static_cast<size_t>(shape.num_heads) *
           static_cast<size_t>(shape.seq_q) * static_cast<size_t>(shape.seq_kv) *
           sizeof(float);
}

size_t EffectiveWorkspaceLimit(const AttentionConfig& cfg) {
    if (cfg.workspace_limit_bytes != 0) return cfg.workspace_limit_bytes;
    const DeviceCapability& cap = DeviceCapability::ForDevice(0);
    // Half of the currently free device memory: generous but still refuses the cases that would
    // evict everything else on an 8 GB laptop GPU.
    return cap.free_mem_bytes / 2;
}

class NaiveBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kNaive; }
    const char* Name() const noexcept override { return "naive"; }

    CapabilityReport Supports(const AttentionConfig& cfg, const AttentionShape& shape) const override {
        CapabilityReport report;
        report.recommended_backend = BackendKind::kNaive;
        report.workspace_bytes = ScoreBytes(shape);

        std::string reason;
        if (!DeviceCapability::CudaAvailable()) {
            reason = "no CUDA device available (DeviceCapability reports cuda-unavailable)";
        } else if (SupportedHeadDimIndex(shape.head_dim) < 0) {
            reason = "head_dim=" + std::to_string(shape.head_dim) +
                     " is outside {32,64,96,128,160,192,256}";
        } else if (cfg.sliding_window > 0) {
            reason = "sliding_window=" + std::to_string(cfg.sliding_window) +
                     " is not implemented by the naive backend";
        } else {
            const size_t limit = EffectiveWorkspaceLimit(cfg);
            const size_t need = report.workspace_bytes;
            if (limit != 0 && need > limit) {
                reason = "score matrix needs " + std::to_string(need) +
                         " B > workspace limit " + std::to_string(limit) +
                         " B; use the tiled or flash backend";
            }
        }

        report.supported = reason.empty();
        report.reason = report.supported
                            ? "B*H_q*S_q*S_kv*4 = " + std::to_string(report.workspace_bytes) +
                                  " B of FP32 score storage fits the workspace limit"
                            : reason;
        return report;
    }

    size_t WorkspaceBytes(const AttentionConfig&, const AttentionShape& shape) const override {
        return ScoreBytes(shape);
    }

    sci::Result<AttentionResult> Forward(const sci::Tensor& q, const sci::Tensor& k,
                                        const sci::Tensor& v,
                                        const AttentionConfig& cfg) const override {
        const sci::Result<AttentionShape> shape_result = InferShape(q, k, v, cfg.layout);
        if (!shape_result.ok()) return sci::MakeUnexpected<AttentionResult>(shape_result.error());
        const AttentionShape shape = *shape_result;

        std::string detail;
        const sci::Status valid = Validate(cfg, shape, &detail);
        if (!valid.ok()) return sci::MakeUnexpected<AttentionResult>(valid);

        const size_t need = ScoreBytes(shape);
        const size_t limit = EffectiveWorkspaceLimit(cfg);
        if (limit != 0 && need > limit) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kWorkspaceExceeded,
                "naive score matrix requires " + std::to_string(need) +
                    " B but the workspace limit is " + std::to_string(limit) +
                    " B; switch to tiled/flash or raise workspace_limit_bytes"));
        }
        if (cfg.causal && shape.seq_kv < shape.seq_q) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kShapeMismatch, "causal requires seq_kv >= seq_q"));
        }

        NaiveDtype dtype_id = NaiveDtype::kFp32;
        if (!DtypeIdFor(q.dtype(), &dtype_id)) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kUnsupportedDtype,
                std::string("dtype=") + sci::kDTypeName(q.dtype()) + " is not a compute dtype"));
        }

        sci::Device& device = q.device();
        // Scratch is pooled: the inference path must not call cudaMalloc (prompt §9.5).
        const sci::Result<float*> scores_result = runtime::AcquireScratchFp32(
            runtime::ScratchKind::kScoresFp32, need / sizeof(float), device);
        if (!scores_result.ok()) {
            return sci::MakeUnexpected<AttentionResult>(scores_result.error());
        }
        float* scores = *scores_result;

        AttentionResult result;
        result.out = sci::Tensor(sci::TensorShape({static_cast<sci::index_t>(shape.batch),
                                                   static_cast<sci::index_t>(shape.num_heads),
                                                   static_cast<sci::index_t>(shape.seq_q),
                                                   static_cast<sci::index_t>(shape.head_dim)}),
                                 q.dtype(), device);
        if (result.out.data() == nullptr) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kWorkspaceExceeded, "allocation of the attention output failed"));
        }
        if (cfg.return_lse) {
            result.lse = sci::Tensor(
                sci::TensorShape({static_cast<sci::index_t>(shape.batch),
                                  static_cast<sci::index_t>(shape.num_heads),
                                  static_cast<sci::index_t>(shape.seq_q)}),
                sci::DType::kFloat32, device);
            if (result.lse.data() == nullptr) {
                return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                    AttnStatusCode::kWorkspaceExceeded, "allocation of the LSE buffer failed"));
            }
        }

        const detail::Strides q_strides =
            detail::StridesFor(cfg.layout, shape.seq_q, shape.num_heads, shape.head_dim);
        const detail::Strides k_strides =
            detail::StridesFor(cfg.layout, shape.seq_kv, shape.num_kv_heads, shape.head_dim);
        const detail::Strides v_strides = k_strides;
        const detail::Strides o_strides =
            detail::StridesFor(AttnLayout::kBHSD, shape.seq_q, shape.num_heads, shape.head_dim);

        NaiveFwdParams params;
        params.q = q.data();
        params.k = k.data();
        params.v = v.data();
        params.out = result.out.data();
        params.scores = scores;
        params.batch = shape.batch;
        params.num_heads = shape.num_heads;
        params.num_kv_heads = shape.num_kv_heads;
        params.seq_q = shape.seq_q;
        params.seq_kv = shape.seq_kv;
        params.head_dim = shape.head_dim;
        params.q_stride = ToStrides(q_strides);
        params.k_stride = ToStrides(k_strides);
        params.v_stride = ToStrides(v_strides);
        params.o_stride = ToStrides(o_strides);
        params.scale = EffectiveScale(cfg, shape);
        params.causal = cfg.causal;
        params.return_lse = cfg.return_lse;
        params.diagonal = cfg.causal ? (shape.seq_kv - shape.seq_q) : 0;
        params.lse = cfg.return_lse ? static_cast<float*>(result.lse.data()) : nullptr;
        params.dtype = dtype_id;

        const sci::Status status = cuda::LaunchNaiveForward(params, cfg.stream);
        if (!status.ok()) return sci::MakeUnexpected<AttentionResult>(status);

        result.stats.used_backend = BackendKind::kNaive;
        result.stats.workspace_bytes = need;
        result.stats.materialized_score_bytes = static_cast<int64_t>(need);
        result.stats.note =
            std::string("reference path: FP32 score matrix materialized in HBM (") +
            std::to_string(need) + " B); layout=" + LayoutName(cfg.layout);
        if (cfg.causal) result.stats.note += ", causal(alignment=bottom-right)";
        return sci::Ok(std::move(result));
    }

    sci::Result<AttentionResult> ForwardVarlen(const sci::Tensor&, const sci::Tensor&,
                                               const sci::Tensor&, int64_t, const int32_t*,
                                               const int32_t*, int64_t, int64_t,
                                               const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedFeature,
            "naive backend does not implement the varlen entry point; use flash_attention_varlen"));
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
