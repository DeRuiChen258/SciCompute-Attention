// Tiled (Level 1) backend adapter.
//
// Contract: tiled never materializes anything in HBM (its workspace is 0 bytes) and never degrades
// silently. Configurations it cannot serve (head_dim > 128, smem over the device limit) are
// rejected with a reason that names the numbers and points at the flash backend.

#include "scicompute_attention/tiled_attention.hpp"

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

#include "backends/tiled/attention_tiled.cuh"

namespace sca {
namespace {

using cuda::TiledDtype;
using cuda::TiledFwdParams;
using cuda::TiledStrides;

struct TileCandidate {
    int32_t block_m;
    int32_t block_n;
    int32_t warps;
};

// Preference order per dtype; the first entry whose smem footprint fits is used (prompt §7.3:
// "no magic numbers - a table plus an explicit rejection reason").
constexpr TileCandidate kFp16Candidates[] = {{64, 64, 4}, {64, 32, 4}, {128, 32, 8}};
constexpr TileCandidate kFp32Candidates[] = {{64, 32, 4}, {64, 64, 4}, {128, 32, 8}};

int64_t MaxTiledHeadDim() { return 128; }

bool DtypeIdFor(sci::DType dtype, TiledDtype* out) {
    switch (dtype) {
        case sci::DType::kFloat32: *out = TiledDtype::kFp32; return true;
        case sci::DType::kFloat16: *out = TiledDtype::kFp16; return true;
        case sci::DType::kBFloat16: *out = TiledDtype::kBf16; return true;
        default: return false;
    }
}

TiledStrides ToStrides(const detail::Strides& s) {
    TiledStrides out;
    out.s_b = s.s_b;
    out.s_h = s.s_h;
    out.s_s = s.s_s;
    return out;
}

// Chooses the first candidate that fits the smem ceiling; returns false with a reason otherwise.
bool SelectTile(sci::DType dtype, int64_t head_dim, size_t smem_limit, TileCandidate* out,
                std::string* reason) {
    const int64_t elem_bytes = DtypeSize(dtype);
    const TileCandidate* candidates =
        dtype == sci::DType::kFloat32 ? kFp32Candidates : kFp16Candidates;
    const size_t count = dtype == sci::DType::kFloat32
                             ? sizeof(kFp32Candidates) / sizeof(kFp32Candidates[0])
                             : sizeof(kFp16Candidates) / sizeof(kFp16Candidates[0]);
    std::string tried;
    for (size_t i = 0; i < count; ++i) {
        const int64_t bytes = cuda::TiledSmemBytes(static_cast<int32_t>(head_dim), elem_bytes,
                                                   candidates[i].block_m, candidates[i].block_n);
        if (bytes > 0 && static_cast<size_t>(bytes) <= smem_limit) {
            *out = candidates[i];
            return true;
        }
        if (!tried.empty()) tried += ", ";
        tried += "(" + std::to_string(candidates[i].block_m) + "x" +
                 std::to_string(candidates[i].block_n) + " -> " + std::to_string(bytes) + " B)";
    }
    *reason = "no tiled configuration fits smem limit " + std::to_string(smem_limit) +
              " B for head_dim=" + std::to_string(head_dim) + " " +
              sci::kDTypeName(dtype) + "; tried " + tried + "; use the flash backend";
    return false;
}

class TiledBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kTiled; }
    const char* Name() const noexcept override { return "tiled"; }

    CapabilityReport Supports(const AttentionConfig& cfg, const AttentionShape& shape) const override {
        CapabilityReport report;
        report.recommended_backend = BackendKind::kTiled;
        report.workspace_bytes = 0;  // everything stays in smem + registers

        std::string reason;
        if (!DeviceCapability::CudaAvailable()) {
            reason = "no CUDA device available";
        } else if (SupportedHeadDimIndex(shape.head_dim) < 0) {
            reason = "head_dim=" + std::to_string(shape.head_dim) +
                     " is outside {32,64,96,128,160,192,256}";
        } else if (shape.head_dim > MaxTiledHeadDim()) {
            reason = "head_dim=" + std::to_string(shape.head_dim) + " > " +
                     std::to_string(MaxTiledHeadDim()) +
                     " is served by the flash backend (smem budget for tiled)";
        } else if (cfg.sliding_window > 0) {
            reason = "sliding_window is not implemented by the tiled backend";
        } else {
            // Use the same smem ceiling the launch path enforces.
            const DeviceCapability& cap = DeviceCapability::ForDevice(0);
            const size_t limit = DeviceCapability::SmemLimitPerBlock(cap);
            const sci::DType probe_dtype = sci::DType::kFloat16;  // geometry-only decision
            TileCandidate tile{};
            std::string tile_reason;
            if (!SelectTile(probe_dtype, shape.head_dim, limit, &tile, &tile_reason)) {
                reason = tile_reason;
            } else {
                report.tile.block_m = tile.block_m;
                report.tile.block_n = tile.block_n;
                report.tile.warps = tile.warps;
                report.tile.stages = 1;  // no cp.async pipeline in Level 1
                reason.clear();
            }
        }

        report.supported = reason.empty();
        report.reason = report.supported
                            ? "tiled tile=" + report.tile.ToString() +
                                  " keeps S in smem (workspace 0 B)"
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
        if (shape.head_dim > MaxTiledHeadDim()) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kDeviceCapability,
                "tiled supports head_dim <= " + std::to_string(MaxTiledHeadDim()) +
                    "; head_dim=" + std::to_string(shape.head_dim) + " needs the flash backend"));
        }

        const DeviceCapability& cap = DeviceCapability::ForDevice(0);
        const size_t smem_limit = DeviceCapability::SmemLimitPerBlock(cap);
        TileCandidate tile{};
        std::string reason;
        if (!SelectTile(q.dtype(), shape.head_dim, smem_limit, &tile, &reason)) {
            return sci::MakeUnexpected<AttentionResult>(
                MakeStatus(AttnStatusCode::kDeviceCapability, reason));
        }

        TiledDtype dtype_id = TiledDtype::kFp16;
        if (!DtypeIdFor(q.dtype(), &dtype_id)) {
            return sci::MakeUnexpected<AttentionResult>(MakeStatus(
                AttnStatusCode::kUnsupportedDtype,
                std::string("dtype=") + sci::kDTypeName(q.dtype()) + " is not a compute dtype"));
        }

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

        TiledFwdParams params;
        params.q = q.data();
        params.k = k.data();
        params.v = v.data();
        params.out = result.out.data();
        params.lse = cfg.return_lse ? static_cast<float*>(result.lse.data()) : nullptr;
        params.batch = shape.batch;
        params.num_heads = shape.num_heads;
        params.num_kv_heads = shape.num_kv_heads;
        params.seq_q = shape.seq_q;
        params.seq_kv = shape.seq_kv;
        params.head_dim = shape.head_dim;
        params.q_stride = ToStrides(detail::StridesFor(cfg.layout, shape.seq_q, shape.num_heads,
                                                       shape.head_dim));
        params.k_stride = ToStrides(detail::StridesFor(cfg.layout, shape.seq_kv, shape.num_kv_heads,
                                                       shape.head_dim));
        params.v_stride = params.k_stride;
        params.o_stride =
            ToStrides(detail::StridesFor(AttnLayout::kBHSD, shape.seq_q, shape.num_heads,
                                         shape.head_dim));
        params.scale = EffectiveScale(cfg, shape);
        params.causal = cfg.causal;
        params.return_lse = cfg.return_lse;
        params.diagonal = cfg.causal ? (shape.seq_kv - shape.seq_q) : 0;
        params.dtype = dtype_id;

        const sci::Status status =
            cuda::LaunchTiledForward(params, cfg.stream, tile.block_m, tile.block_n, tile.warps);
        if (!status.ok()) return sci::MakeUnexpected<AttentionResult>(status);

        result.stats.used_backend = BackendKind::kTiled;
        result.stats.tile.block_m = tile.block_m;
        result.stats.tile.block_n = tile.block_n;
        result.stats.tile.warps = tile.warps;
        result.stats.tile.stages = 1;
        result.stats.workspace_bytes = 0;
        result.stats.materialized_score_bytes = 0;  // S stays in smem
        result.stats.note = "online softmax with S staged in smem; workspace 0 B; layout=" +
                            std::string(LayoutName(cfg.layout));
        return sci::Ok(std::move(result));
    }

    sci::Result<AttentionResult> ForwardVarlen(const sci::Tensor&, const sci::Tensor&,
                                               const sci::Tensor&, int64_t, const int32_t*,
                                               const int32_t*, int64_t, int64_t,
                                               const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedFeature,
            "tiled backend does not implement the varlen entry point; use flash_attention_varlen"));
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
