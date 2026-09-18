// Flash (Level 3) backend adapter.
//
// Guarantees offered to callers:
//   * the N x N score matrix is never written to HBM (workspace_bytes == 0,
//     materialized_score_bytes == 0);
//   * one tile geometry per head_dim from flash_tile_config.hpp, validated against the measured
//     smem limit both at compile time (static_assert) and at runtime;
//   * FP32 inputs are rejected with a reason that points at the naive/tiled reference backends
//     instead of being silently down-cast.
//
// The varlen entry point is implemented as a per-sequence loop over the same kernel: correct, but
// without cross-sequence batching (documented as a Roadmap item in docs/flash_attention.md).

#include "scicompute_attention/flash_attention.hpp"

#include <string>
#include <vector>

#include "backends/flash/flash_fwd_kernel.cuh"
#include "backends/flash/flash_tile_config.hpp"
#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/capability.hpp"
#include "scicompute_attention/detail/dtype_traits.hpp"
#include "scicompute_attention/detail/layout_traits.hpp"
#include "scicompute_attention/dispatcher.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"

namespace sca {
namespace {

using cuda::FlashFwdParams;

int DtypeIdFor(sci::DType dtype) {
    switch (dtype) {
        case sci::DType::kFloat16: return 1;
        case sci::DType::kBFloat16: return 2;
        default: return 0;  // FP32 is a reference-only dtype
    }
}

// Common capability answer shared by Supports() and Forward().
sci::Status ValidateFlashRequest(const sci::Tensor& q, const AttentionConfig& cfg,
                                 const AttentionShape& shape, flash::FlashTile* tile_out) {
    if (!DeviceCapability::CudaAvailable()) {
        return MakeStatus(AttnStatusCode::kUnsupportedFeature,
                          "flash backend: no CUDA device available");
    }
    if (DtypeIdFor(q.dtype()) == 0) {
        return MakeStatus(AttnStatusCode::kUnsupportedDtype,
                          std::string("flash backend computes in fp16/bf16; dtype=") +
                              sci::kDTypeName(q.dtype()) +
                              " is served by the naive (reference) or tiled backend");
    }
    const flash::FlashTile* tile = flash::FindFlashTile(shape.head_dim);
    if (tile == nullptr) {
        return MakeStatus(AttnStatusCode::kUnsupportedHeadDim,
                          "flash tile table has no entry for head_dim=" +
                              std::to_string(shape.head_dim));
    }
    const DeviceCapability& cap = DeviceCapability::ForDevice(0);
    const size_t limit = DeviceCapability::SmemLimitPerBlock(cap);
    const int64_t smem = flash::FlashTileSmemBytes(*tile, shape.head_dim, DtypeSize(q.dtype()));
    if (smem < 0 || static_cast<size_t>(smem) > limit) {
        return MakeStatus(AttnStatusCode::kDeviceCapability,
                          "flash tile " + tile->ToString() + " needs " + std::to_string(smem) +
                              " B > smem limit " + std::to_string(limit) + " B for head_dim=" +
                              std::to_string(shape.head_dim));
    }
    if (cfg.sliding_window > 0) {
        return MakeStatus(AttnStatusCode::kUnsupportedFeature,
                          "flash backend: sliding_window=" + std::to_string(cfg.sliding_window) +
                              " is not implemented in v1");
    }
    if (tile_out != nullptr) *tile_out = *tile;
    return sci::Status::Ok();
}

FlashFwdParams MakeParams(const sci::Tensor& q, const sci::Tensor& k, const sci::Tensor& v,
                          sci::Tensor* out, sci::Tensor* lse, const AttentionConfig& cfg,
                          const AttentionShape& shape, int64_t q_offset, int64_t kv_offset,
                          int64_t seq_q, int64_t seq_kv, int dtype_id, float scale) {
    const detail::Strides q_strides =
        detail::StridesFor(cfg.layout, shape.seq_q, shape.num_heads, shape.head_dim);
    const detail::Strides k_strides =
        detail::StridesFor(cfg.layout, shape.seq_kv, shape.num_kv_heads, shape.head_dim);
    const detail::Strides o_strides =
        detail::StridesFor(AttnLayout::kBHSD, shape.seq_q, shape.num_heads, shape.head_dim);

    FlashFwdParams params;
    params.q = q.data();
    params.k = k.data();
    params.v = v.data();
    params.out = out->data();
    params.lse = lse != nullptr ? static_cast<float*>(lse->data()) : nullptr;
    params.batch = shape.batch;
    params.num_heads = shape.num_heads;
    params.num_kv_heads = shape.num_kv_heads;
    params.seq_q = seq_q;
    params.seq_kv = seq_kv;
    params.head_dim = shape.head_dim;
    params.q_sb = q_strides.s_b;
    params.q_sh = q_strides.s_h;
    params.q_ss = q_strides.s_s;
    params.k_sb = k_strides.s_b;
    params.k_sh = k_strides.s_h;
    params.k_ss = k_strides.s_s;
    params.v_sb = k_strides.s_b;
    params.v_sh = k_strides.s_h;
    params.v_ss = k_strides.s_s;
    params.o_sb = o_strides.s_b;
    params.o_sh = o_strides.s_h;
    params.o_ss = o_strides.s_s;
    params.scale = scale;
    params.causal = cfg.causal;
    params.return_lse = cfg.return_lse;
    params.diagonal = cfg.causal ? (seq_kv - seq_q) : 0;
    params.dtype_id = dtype_id;
    (void)q_offset;
    (void)kv_offset;
    return params;
}

class FlashBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kFlash; }
    const char* Name() const noexcept override { return "flash"; }

    CapabilityReport Supports(const AttentionConfig& cfg, const AttentionShape& shape) const override {
        CapabilityReport report;
        report.recommended_backend = BackendKind::kFlash;
        report.workspace_bytes = 0;

        std::string reason;
        const flash::FlashTile* tile = flash::FindFlashTile(shape.head_dim);
        if (!DeviceCapability::CudaAvailable()) {
            reason = "no CUDA device available";
        } else if (tile == nullptr) {
            reason = "head_dim=" + std::to_string(shape.head_dim) +
                     " has no flash tile (supported: 32,64,96,128,160,192,256)";
        } else if (cfg.sliding_window > 0) {
            reason = "sliding_window=" + std::to_string(cfg.sliding_window) + " not implemented";
        } else {
            const DeviceCapability& cap = DeviceCapability::ForDevice(0);
            const size_t limit = DeviceCapability::SmemLimitPerBlock(cap);
            const int64_t smem = flash::FlashTileSmemBytes(*tile, shape.head_dim, 2);
            if (static_cast<size_t>(smem) > limit) {
                reason = "tile " + tile->ToString() + " needs " + std::to_string(smem) +
                         " B > smem limit " + std::to_string(limit) + " B";
            } else {
                report.tile.block_m = tile->block_m;
                report.tile.block_n = tile->block_n;
                report.tile.warps = tile->warps;
                report.tile.stages = tile->stages;
                report.tile.use_tma = false;
            }
        }

        report.supported = reason.empty();
        report.reason = report.supported
                            ? "online softmax in registers (workspace 0 B), tile " + tile->ToString()
                            : reason;
        return report;
    }

    size_t WorkspaceBytes(const AttentionConfig&, const AttentionShape&) const override { return 0; }

    sci::Result<AttentionResult> Forward(const sci::Tensor& q, const sci::Tensor& k,
                                        const sci::Tensor& v,
                                        const AttentionConfig& cfg) const override {
        const sci::Result<AttentionShape> shape_result = InferShape(q, k, v, cfg.layout);
        if (!shape_result.ok()) return sci::MakeUnexpected<AttentionResult>(shape_result.error());
        const AttentionShape shape = *shape_result;

        const sci::Status valid = Validate(cfg, shape);
        if (!valid.ok()) return sci::MakeUnexpected<AttentionResult>(valid);

        flash::FlashTile tile{};
        const sci::Status request = ValidateFlashRequest(q, cfg, shape, &tile);
        if (!request.ok()) return sci::MakeUnexpected<AttentionResult>(request);

        sci::Device& device = q.device();
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

        const FlashFwdParams params =
            MakeParams(q, k, v, &result.out, cfg.return_lse ? &result.lse : nullptr, cfg, shape, 0,
                       0, shape.seq_q, shape.seq_kv, DtypeIdFor(q.dtype()), EffectiveScale(cfg, shape));
        const sci::Status status = cuda::LaunchFlashFwd(params, cfg.stream);
        if (!status.ok()) return sci::MakeUnexpected<AttentionResult>(status);

        result.stats.used_backend = BackendKind::kFlash;
        result.stats.tile.block_m = tile.block_m;
        result.stats.tile.block_n = tile.block_n;
        result.stats.tile.warps = tile.warps;
        result.stats.tile.stages = tile.stages;
        result.stats.workspace_bytes = 0;
        result.stats.materialized_score_bytes = 0;
        result.stats.note = "flash: mma.sync m16n8k16 + cp.async pipeline, tile " + tile.ToString() +
                            ", S/P in registers; layout=" + LayoutName(cfg.layout);
        return sci::Ok(std::move(result));
    }

    // varlen: q/k/v are packed [total_tokens, H, D] tensors with host-side cu_seqlens offsets.
    // v1 launches one (1, S_q, S_kv) kernel per sequence over the same packed buffers, which is
    // correct but loses cross-sequence batching (Roadmap: a fused varlen kernel).
    sci::Result<AttentionResult> ForwardVarlen(const sci::Tensor& q, const sci::Tensor& k,
                                               const sci::Tensor& v, int64_t num_seqs,
                                               const int32_t* cu_seqlens_q,
                                               const int32_t* cu_seqlens_kv, int64_t max_seq_q,
                                               int64_t max_seq_kv,
                                               const AttentionConfig& cfg) const override {
        if (cu_seqlens_q == nullptr || cu_seqlens_kv == nullptr) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kShapeMismatch, "varlen requires host-resident cu_seqlens_q/kv"));
        }
        if (num_seqs < 1) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kShapeMismatch,
                "varlen num_seqs=" + std::to_string(num_seqs) + " (must be >= 1)"));
        }
        if (cfg.return_lse) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kUnsupportedFeature,
                "varlen does not expose LSE yet; call the per-sequence entry point if it is needed"));
        }
        if (cfg.causal && max_seq_kv < max_seq_q) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kShapeMismatch, "varlen causal requires max_seq_kv >= max_seq_q"));
        }
        for (const auto* tensor : {&q, &k, &v}) {
            if (tensor->ndims() != 3) {
                return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                    AttnStatusCode::kShapeMismatch,
                    "varlen expects packed [T, H, D] tensors; got ndims=" +
                        std::to_string(tensor->ndims())));
            }
            if (tensor->device_type() != sci::DeviceType::kCUDA) {
                return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                    AttnStatusCode::kUnsupportedFeature, "varlen requires CUDA tensors"));
            }
        }
        if (DtypeIdFor(q.dtype()) == 0) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kUnsupportedDtype, "flash varlen supports fp16/bf16 only"));
        }
        if (q.dtype() != k.dtype() || q.dtype() != v.dtype()) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kUnsupportedDtype, "varlen requires q/k/v to share one dtype"));
        }

        const int64_t num_heads = q.dim(1);
        const int64_t head_dim = q.dim(2);
        const int64_t num_kv_heads = k.dim(1);
        if (k.dim(2) != head_dim || v.dim(2) != head_dim || v.dim(1) != num_kv_heads) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kShapeMismatch,
                "varlen expects k/v head dims to match q; got D=" + std::to_string(head_dim)));
        }
        if (num_heads % num_kv_heads != 0) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kShapeMismatch, "varlen requires H_q % H_kv == 0"));
        }

        AttentionShape probe_shape;
        probe_shape.batch = 1;
        probe_shape.seq_q = max_seq_q;
        probe_shape.seq_kv = max_seq_kv;
        probe_shape.num_heads = num_heads;
        probe_shape.num_kv_heads = num_kv_heads;
        probe_shape.head_dim = head_dim;
        flash::FlashTile tile{};
        const sci::Status request = ValidateFlashRequest(q, cfg, probe_shape, &tile);
        if (!request.ok()) return sci::MakeUnexpected<AttentionResult>(request);

        sci::Device& device = q.device();
        AttentionResult result;
        result.out = sci::Tensor(
            sci::TensorShape({q.dim(0), num_heads, head_dim}), q.dtype(), device);
        if (result.out.data() == nullptr) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kWorkspaceExceeded, "allocation of the varlen output failed"));
        }

        const int dtype_id = DtypeIdFor(q.dtype());
        const float scale = EffectiveScale(cfg, probe_shape);
        const auto* q_bytes = static_cast<const uint8_t*>(q.data());
        const auto* k_bytes = static_cast<const uint8_t*>(k.data());
        const auto* v_bytes = static_cast<const uint8_t*>(v.data());
        const int64_t q_token_bytes = num_heads * head_dim * DtypeSize(q.dtype());
        const int64_t kv_token_bytes = num_kv_heads * head_dim * DtypeSize(q.dtype());

        for (int64_t s = 0; s < num_seqs; ++s) {
            const int64_t sq = cu_seqlens_q[s + 1] - cu_seqlens_q[s];
            const int64_t skv = cu_seqlens_kv[s + 1] - cu_seqlens_kv[s];
            if (sq <= 0 || skv <= 0) {
                return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                    AttnStatusCode::kShapeMismatch,
                    "varlen sequence " + std::to_string(s) + " has S_q=" + std::to_string(sq) +
                        ", S_kv=" + std::to_string(skv) + " (both must be > 0)"));
            }
            AttentionShape shape;
            shape.batch = 1;
            shape.seq_q = sq;
            shape.seq_kv = skv;
            shape.num_heads = num_heads;
            shape.num_kv_heads = num_kv_heads;
            shape.head_dim = head_dim;

            FlashFwdParams params;
            // Packed inputs are [T, H, D], i.e. BSHD with B = 1; the strides below say exactly that.
            params.q = q_bytes + cu_seqlens_q[s] * q_token_bytes;
            params.k = k_bytes + cu_seqlens_kv[s] * kv_token_bytes;
            params.v = v_bytes + cu_seqlens_kv[s] * kv_token_bytes;
            params.out = static_cast<uint8_t*>(result.out.data()) +
                         static_cast<int64_t>(cu_seqlens_q[s]) * q_token_bytes;
            params.lse = nullptr;
            params.batch = 1;
            params.num_heads = num_heads;
            params.num_kv_heads = num_kv_heads;
            params.seq_q = sq;
            params.seq_kv = skv;
            params.head_dim = head_dim;
            params.q_sb = sq * num_heads * head_dim;
            params.q_sh = head_dim;
            params.q_ss = num_heads * head_dim;
            params.k_sb = skv * num_kv_heads * head_dim;
            params.k_sh = head_dim;
            params.k_ss = num_kv_heads * head_dim;
            params.v_sb = params.k_sb;
            params.v_sh = params.k_sh;
            params.v_ss = params.k_ss;
            params.o_sb = sq * num_heads * head_dim;
            params.o_sh = head_dim;
            params.o_ss = num_heads * head_dim;
            params.scale = scale;
            params.causal = cfg.causal;
            params.return_lse = false;
            params.diagonal = cfg.causal ? (skv - sq) : 0;
            params.dtype_id = dtype_id;

            const sci::Status status = cuda::LaunchFlashFwd(params, cfg.stream);
            if (!status.ok()) return sci::MakeUnexpected<AttentionResult>(status);
        }

        result.stats.used_backend = BackendKind::kFlash;
        result.stats.workspace_bytes = 0;
        result.stats.materialized_score_bytes = 0;
        result.stats.tile.block_m = tile.block_m;
        result.stats.tile.block_n = tile.block_n;
        result.stats.tile.warps = tile.warps;
        result.stats.tile.stages = tile.stages;
        result.stats.note = "varlen: " + std::to_string(num_seqs) +
                            " per-sequence launches over packed buffers (tile " + tile.ToString() +
                            ")";
        return sci::Ok(std::move(result));
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
                                                    const sci::Tensor& v, int64_t num_seqs,
                                                    const int32_t* cu_q, const int32_t* cu_kv,
                                                    int64_t max_seq_q, int64_t max_seq_kv,
                                                    const AttentionConfig& cfg) {
    return kFlashBackend.ForwardVarlen(q, k, v, num_seqs, cu_q, cu_kv, max_seq_q, max_seq_kv, cfg);
}

}  // namespace sca
