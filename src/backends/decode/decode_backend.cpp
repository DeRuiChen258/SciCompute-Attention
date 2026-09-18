// Level 4 (decode) backend adapter.
//
// Scope: S_q == 1 per (batch, head) - the real decode step. Larger S_q is rejected with a reason
// that points at the flash backend, so the dispatcher never silently routes a prefill-shaped
// request into the decode kernel (docs/prefill_decode.md explains why the two are separate).

#include "scicompute_attention/decode_attention.hpp"

#include <string>
#include <vector>

#include "backends/decode/decode_config.hpp"
#include "backends/decode/decode_splitk.cuh"
#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/capability.hpp"
#include "scicompute_attention/detail/dtype_traits.hpp"
#include "scicompute_attention/detail/layout_traits.hpp"
#include "scicompute_attention/dispatcher.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"
#include "runtime/scratch_pool.hpp"

namespace sca {
namespace {

using cuda::DecodeDtype;
using cuda::DecodeFwdParams;
using cuda::DecodeStrides;

bool DtypeIdFor(sci::DType dtype, DecodeDtype* out) {
    switch (dtype) {
        case sci::DType::kFloat32: *out = DecodeDtype::kFp32; return true;
        case sci::DType::kFloat16: *out = DecodeDtype::kFp16; return true;
        case sci::DType::kBFloat16: *out = DecodeDtype::kBf16; return true;
        default: return false;
    }
}

DecodeStrides ToStrides(const detail::Strides& s) {
    DecodeStrides out;
    out.s_b = s.s_b;
    out.s_h = s.s_h;
    out.s_s = s.s_s;
    return out;
}

size_t PartialBytes(const AttentionShape& shape, int64_t splits) {
    return static_cast<size_t>(decode::PartialElements(shape.batch, shape.num_heads, splits,
                                                       shape.head_dim)) *
           sizeof(float);
}

int64_t ResolveSplits(const AttentionConfig& cfg, const AttentionShape& shape,
                      const DeviceCapability& cap) {
    if (cfg.num_splits > 0) {
        return detail::Min<int64_t>(cfg.num_splits, decode::kMaxSplits);
    }
    return decode::SplitsFor(shape.seq_kv, cap.sm_count);
}

class DecodeBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kDecode; }
    const char* Name() const noexcept override { return "decode"; }

    CapabilityReport Supports(const AttentionConfig& cfg, const AttentionShape& shape) const override {
        CapabilityReport report;
        report.recommended_backend = BackendKind::kDecode;

        const DeviceCapability& cap = DeviceCapability::ForDevice(0);
        const int64_t splits = ResolveSplits(cfg, shape, cap);
        report.workspace_bytes = PartialBytes(shape, splits);

        std::string reason;
        if (!DeviceCapability::CudaAvailable()) {
            reason = "no CUDA device available";
        } else if (shape.seq_q != decode::kNativeSeqQ) {
            reason = "decode kernel handles seq_q=" + std::to_string(decode::kNativeSeqQ) +
                     "; seq_q=" + std::to_string(shape.seq_q) +
                     " is routed to the flash backend (grouped decode is a Roadmap item)";
        } else if (SupportedHeadDimIndex(shape.head_dim) < 0) {
            reason = "head_dim=" + std::to_string(shape.head_dim) + " is unsupported";
        } else if (shape.head_dim % 32 != 0) {
            reason = "decode requires head_dim % 32 == 0; head_dim=" +
                     std::to_string(shape.head_dim);
        } else if (cfg.sliding_window > 0) {
            reason = "sliding_window is not implemented by the decode backend";
        }
        report.supported = reason.empty();
        report.reason = report.supported
                            ? "split-K decode with " + std::to_string(splits) +
                                  " splits, workspace " + std::to_string(report.workspace_bytes) +
                                  " B (FP32 partials)"
                            : reason;
        return report;
    }

    size_t WorkspaceBytes(const AttentionConfig& cfg, const AttentionShape& shape) const override {
        const DeviceCapability& cap = DeviceCapability::ForDevice(0);
        return PartialBytes(shape, ResolveSplits(cfg, shape, cap));
    }

    sci::Result<AttentionResult> Forward(const sci::Tensor& q, const sci::Tensor& k,
                                        const sci::Tensor& v,
                                        const AttentionConfig& cfg) const override {
        const sci::Result<AttentionShape> shape_result = InferShape(q, k, v, cfg.layout);
        if (!shape_result.ok()) return sci::MakeUnexpected<AttentionResult>(shape_result.error());
        const AttentionShape shape = *shape_result;

        const sci::Status valid = Validate(cfg, shape);
        if (!valid.ok()) return sci::MakeUnexpected<AttentionResult>(valid);
        if (shape.seq_q != decode::kNativeSeqQ) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kUnsupportedFeature,
                "decode backend requires seq_q=" + std::to_string(decode::kNativeSeqQ) +
                    "; seq_q=" + std::to_string(shape.seq_q) + " belongs to the flash backend"));
        }
        if (shape.head_dim % 32 != 0) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kUnsupportedHeadDim,
                "decode requires head_dim % 32 == 0; head_dim=" + std::to_string(shape.head_dim)));
        }
        DecodeDtype dtype_id = DecodeDtype::kFp16;
        if (!DtypeIdFor(q.dtype(), &dtype_id)) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kUnsupportedDtype,
                std::string("dtype=") + sci::kDTypeName(q.dtype()) + " is not a compute dtype"));
        }

        const DeviceCapability& cap = DeviceCapability::ForDevice(0);
        const int64_t splits = ResolveSplits(cfg, shape, cap);
        const size_t partial_bytes = PartialBytes(shape, splits);

        const sci::Result<float*> partials = runtime::AcquireScratchFp32(
            runtime::ScratchKind::kDecodePartials, partial_bytes / sizeof(float), q.device());
        if (!partials.ok()) return sci::MakeUnexpected<AttentionResult>(partials.error());

        const int64_t base = shape.batch * shape.num_heads * splits;
        float* partial_m = *partials;
        float* partial_l = partial_m + base;
        float* partial_o = partial_l + base;

        sci::Device& device = q.device();
        AttentionResult result;
        result.out = sci::Tensor(sci::TensorShape({static_cast<sci::index_t>(shape.batch),
                                                   static_cast<sci::index_t>(shape.num_heads),
                                                   static_cast<sci::index_t>(shape.seq_q),
                                                   static_cast<sci::index_t>(shape.head_dim)}),
                                 q.dtype(), device);
        if (result.out.data() == nullptr) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kWorkspaceExceeded, "allocation of the decode output failed"));
        }

        DecodeFwdParams params;
        params.q = q.data();
        params.k = k.data();
        params.v = v.data();
        params.out = result.out.data();
        params.partial_m = partial_m;
        params.partial_l = partial_l;
        params.partial_o = partial_o;
        params.batch = shape.batch;
        params.num_heads = shape.num_heads;
        params.num_kv_heads = shape.num_kv_heads;
        params.seq_kv = shape.seq_kv;
        params.head_dim = shape.head_dim;
        params.num_splits = splits;
        params.q_stride =
            ToStrides(detail::StridesFor(cfg.layout, shape.seq_q, shape.num_heads, shape.head_dim));
        params.k_stride = ToStrides(
            detail::StridesFor(cfg.layout, shape.seq_kv, shape.num_kv_heads, shape.head_dim));
        params.v_stride = params.k_stride;
        params.o_stride = ToStrides(
            detail::StridesFor(AttnLayout::kBHSD, shape.seq_q, shape.num_heads, shape.head_dim));
        params.scale = EffectiveScale(cfg, shape);
        params.dtype = dtype_id;

        const sci::Status status = cuda::LaunchDecodeSplitK(params, cfg.stream);
        if (!status.ok()) return sci::MakeUnexpected<AttentionResult>(status);

        result.stats.used_backend = BackendKind::kDecode;
        result.stats.workspace_bytes = partial_bytes;
        result.stats.num_splits = splits;
        result.stats.note = "split-K decode: " + std::to_string(splits) +
                            " splits, FP32 partials in workspace; layout=" +
                            std::string(LayoutName(cfg.layout));
        return sci::Ok(std::move(result));
    }

    sci::Result<AttentionResult> ForwardVarlen(const sci::Tensor&, const sci::Tensor&,
                                               const sci::Tensor&, int64_t, const int32_t*,
                                               const int32_t*, int64_t, int64_t,
                                               const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedFeature,
            "decode backend does not implement varlen; use flash_attention_varlen"));
    }
};

const DecodeBackend kDecodeBackend;

}  // namespace

namespace runtime {
const IAttentionBackend* DecodeBackendInstance() noexcept { return &kDecodeBackend; }
}  // namespace runtime

DecodeConfig RecommendDecodeConfig(const AttentionShape& shape, const DeviceCapability& cap) {
    DecodeConfig cfg;
    cfg.max_splits = decode::kMaxSplits;
    cfg.num_splits = decode::SplitsFor(shape.seq_kv, cap.sm_count);
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

