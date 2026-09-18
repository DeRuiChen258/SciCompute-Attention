#include "scicompute_attention/attention_config.hpp"

#include <cmath>
#include <string>

#include "scicompute_attention/detail/host_utils.hpp"
#include "scicompute_attention/detail/layout_traits.hpp"
#include "scicompute_attention/status.hpp"

#include "backends/flash/flash_tile_config.hpp"

namespace sca {
namespace {

std::string HeadDimList() {
    std::string out;
    for (int64_t i = 0; i < kNumSupportedHeadDims; ++i) {
        if (i != 0) out += ",";
        out += std::to_string(kSupportedHeadDims[i]);
    }
    return out;
}

bool IsReferenceOnlyBackend(BackendKind kind) noexcept {
    return kind == BackendKind::kNaive || kind == BackendKind::kTiled ||
           kind == BackendKind::kDecode || kind == BackendKind::kPaged;
}

}  // namespace

float EffectiveScale(const AttentionConfig& cfg, const AttentionShape& shape) noexcept {
    if (cfg.scale > 0.0f) return cfg.scale;
    if (shape.head_dim <= 0) return 0.0f;
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(shape.head_dim)));
}

sci::Status Validate(const AttentionConfig& cfg, const AttentionShape& shape,
                     std::string* detail) {
    auto fail = [&](AttnStatusCode code, const std::string& msg) {
        if (detail != nullptr) *detail = msg;
        return MakeStatus(code, msg);
    };

    if (!detail::LayoutSupported(cfg.layout)) {
        return fail(AttnStatusCode::kUnsupportedLayout,
                    "AttentionConfig invalid: layout=" + std::to_string(static_cast<int>(cfg.layout)) +
                        " (supported: bhsd, bshd)");
    }
    if (shape.batch < 1) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionShape invalid: batch=" + std::to_string(shape.batch) +
                        " (must be >= 1)");
    }
    if (shape.seq_q < 1) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionShape invalid: seq_q=" + std::to_string(shape.seq_q) +
                        " (must be >= 1)");
    }
    if (shape.seq_kv < 1) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionShape invalid: seq_kv=" + std::to_string(shape.seq_kv) +
                        " (must be >= 1)");
    }
    if (shape.num_heads < 1) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionShape invalid: num_heads=" + std::to_string(shape.num_heads) +
                        " (must be >= 1)");
    }
    if (shape.num_kv_heads < 1) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionShape invalid: num_kv_heads=" + std::to_string(shape.num_kv_heads) +
                        " (must be >= 1)");
    }
    if (SupportedHeadDimIndex(shape.head_dim) < 0) {
        return fail(AttnStatusCode::kUnsupportedHeadDim,
                    "AttentionConfig invalid: head_dim=" + std::to_string(shape.head_dim) +
                        " (supported: " + HeadDimList() + ")");
    }
    if (shape.num_heads % shape.num_kv_heads != 0) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionShape invalid: num_heads=" + std::to_string(shape.num_heads) +
                        " % num_kv_heads=" + std::to_string(shape.num_kv_heads) +
                        " != 0 (GQA requires an integer group size)");
    }
    if (cfg.causal && shape.seq_kv < shape.seq_q) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionConfig invalid: causal=true but seq_kv(" +
                        std::to_string(shape.seq_kv) + ") < seq_q(" +
                        std::to_string(shape.seq_q) + ")");
    }
    if (shape.num_kv_heads > shape.num_heads) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionShape invalid: num_kv_heads=" + std::to_string(shape.num_kv_heads) +
                        " > num_heads=" + std::to_string(shape.num_heads) +
                        " (MQA/GQA require H_kv <= H_q)");
    }
    if (cfg.sliding_window > 0) {
        return fail(AttnStatusCode::kUnsupportedFeature,
                    "AttentionConfig invalid: sliding_window=" +
                        std::to_string(cfg.sliding_window) +
                        " is not implemented in v1 (see README Roadmap)");
    }
    if (cfg.softcap < 0.0f) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionConfig invalid: softcap=" + std::to_string(cfg.softcap) +
                        " (must be 0 or > 0)");
    }
    if (cfg.softcap > 0.0f && IsReferenceOnlyBackend(cfg.backend)) {
        return fail(AttnStatusCode::kUnsupportedFeature,
                    "AttentionConfig invalid: softcap=" + std::to_string(cfg.softcap) +
                        " is implemented in the flash backend only, backend=" +
                        BackendName(cfg.backend));
    }
    if (cfg.scale < 0.0f) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionConfig invalid: scale=" + std::to_string(cfg.scale) +
                        " (must be 0 for the 1/sqrt(head_dim) default or > 0)");
    }
    if (cfg.num_splits < 0) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionConfig invalid: num_splits=" + std::to_string(cfg.num_splits) +
                        " (must be >= 0)");
    }
    if (cfg.cu_seqlens_q != nullptr && cfg.cu_seqlens_kv == nullptr) {
        return fail(AttnStatusCode::kShapeMismatch,
                    "AttentionConfig invalid: cu_seqlens_q is set but cu_seqlens_kv is null");
    }
    if (detail != nullptr) *detail = "ok";
    return sci::Status::Ok();
}

TileConfig RecommendTile(const AttentionConfig& cfg, const AttentionShape& shape) {
    (void)cfg;
    // Single source of truth: the flash tile table (src/backends/flash/flash_tile_config.hpp).
    // The dispatcher reports the same geometry, so Explain() and the kernels can never disagree.
    const flash::FlashTile* entry = flash::FindFlashTile(shape.head_dim);
    if (entry == nullptr) return TileConfig{};
    TileConfig tile;
    tile.block_m = entry->block_m;
    tile.block_n = entry->block_n;
    tile.warps = entry->warps;
    tile.stages = entry->stages;
    tile.use_tma = false;
    return tile;
}

sci::Status ValidateVarlenHost(const int32_t* host_cu_seqlens, int64_t batch,
                               int64_t* max_seq_out) {
    if (host_cu_seqlens == nullptr) {
        return MakeStatus(AttnStatusCode::kShapeMismatch, "cu_seqlens is null");
    }
    if (batch < 1) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "varlen batch=" + std::to_string(batch) + " (must be >= 1)");
    }
    if (host_cu_seqlens[0] != 0) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "cu_seqlens[0]=" + std::to_string(host_cu_seqlens[0]) +
                              " (must start at 0)");
    }
    int64_t max_seq = 0;
    for (int64_t i = 0; i < batch; ++i) {
        const int32_t lo = host_cu_seqlens[i];
        const int32_t hi = host_cu_seqlens[i + 1];
        if (hi <= lo) {
            return MakeStatus(AttnStatusCode::kShapeMismatch,
                              "cu_seqlens not strictly increasing at index " + std::to_string(i) +
                                  ": [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
        }
        max_seq = detail::Max<int64_t>(max_seq, static_cast<int64_t>(hi - lo));
    }
    if (max_seq_out != nullptr) *max_seq_out = max_seq;
    return sci::Status::Ok();
}

}  // namespace sca
