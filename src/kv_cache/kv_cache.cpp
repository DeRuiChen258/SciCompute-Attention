#include "scicompute_attention/kv_cache.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "device/device.hpp"
#include "device/stream.hpp"
#include "kv_cache/kv_cache_kernels.cuh"
#include "runtime/device_probe.h"
#include "scicompute_attention/status.hpp"

namespace sca {
namespace {

bool IsSupportedBlockSize(int64_t block_size) {
    switch (block_size) {
        case 1:
        case 8:
        case 16:
        case 32:
        case 64: return true;
        default: return false;
    }
}

// Fraction of the currently free device memory a KV cache may claim by default.
constexpr double kKvBudgetRatio = 0.8;

cuda::KVCacheShape ToDeviceShape(const KVCacheConfig& cfg) {
    cuda::KVCacheShape shape;
    shape.num_blocks = cfg.num_blocks;
    shape.block_size = cfg.block_size;
    shape.num_kv_heads = cfg.num_kv_heads;
    shape.head_dim = cfg.head_dim;
    return shape;
}

sci::Status ValidateConfig(const KVCacheConfig& cfg) {
    if (cfg.num_layers <= 0) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "KVCacheConfig invalid: num_layers=" + std::to_string(cfg.num_layers));
    }
    if (cfg.num_kv_heads <= 0) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "KVCacheConfig invalid: num_kv_heads=" + std::to_string(cfg.num_kv_heads));
    }
    if (SupportedHeadDimIndex(cfg.head_dim) < 0) {
        return MakeStatus(AttnStatusCode::kUnsupportedHeadDim,
                          "KVCacheConfig invalid: head_dim=" + std::to_string(cfg.head_dim));
    }
    if (!IsSupportedBlockSize(cfg.block_size)) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "KVCacheConfig invalid: block_size=" + std::to_string(cfg.block_size) +
                              " (supported: 1, 8, 16, 32, 64)");
    }
    if (cfg.num_blocks <= 0) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "KVCacheConfig invalid: num_blocks=" + std::to_string(cfg.num_blocks));
    }
    if (cfg.max_num_seqs <= 0) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "KVCacheConfig invalid: max_num_seqs=" + std::to_string(cfg.max_num_seqs));
    }
    if (!IsSupportedComputeDtype(cfg.dtype)) {
        return MakeStatus(AttnStatusCode::kUnsupportedDtype,
                          std::string("KVCacheConfig invalid: dtype=") + sci::kDTypeName(cfg.dtype));
    }
    if (cfg.layout != KvLayout::kBlockMajor) {
        return MakeStatus(AttnStatusCode::kUnsupportedLayout,
                          std::string("KVCacheConfig invalid: layout=") +
                              KvLayoutName(cfg.layout) + " (v1 implements block-major only)");
    }
    return sci::Status::Ok();
}

}  // namespace

sci::Result<size_t> KvCacheBytes(const KVCacheConfig& cfg) {
    const sci::Status valid = ValidateConfig(cfg);
    if (!valid.ok()) return sci::MakeUnexpected<size_t>(valid);
    const int64_t per_layer = cfg.num_blocks * cfg.block_size * cfg.num_kv_heads * cfg.head_dim;
    const int64_t total = 2 * cfg.num_layers * per_layer * DtypeSize(cfg.dtype);
    if (total <= 0) {
        return sci::MakeUnexpected<size_t>(
            MakeStatus(AttnStatusCode::kShapeMismatch, "KVCacheConfig overflow"));
    }
    return sci::Ok(static_cast<size_t>(total));
}

int64_t MaxBlocksForBudget(const KVCacheConfig& cfg, size_t budget_bytes) {
    const int64_t per_block_bytes =
        2 * cfg.num_layers * cfg.block_size * cfg.num_kv_heads * cfg.head_dim *
        DtypeSize(cfg.dtype);
    if (per_block_bytes <= 0) return 0;
    return static_cast<int64_t>(budget_bytes / static_cast<size_t>(per_block_bytes));
}

sci::Result<std::vector<int32_t>> ComputeSlotMapping(const int32_t* block_table_row,
                                                     int64_t seq_start, int64_t num_tokens,
                                                     int64_t block_size, int64_t max_blocks) {
    if (block_table_row == nullptr) {
        return sci::MakeUnexpected<std::vector<int32_t>>(
            MakeStatus(AttnStatusCode::kShapeMismatch, "ComputeSlotMapping: null block table row"));
    }
    if (block_size <= 0 || num_tokens <= 0) {
        return sci::MakeUnexpected<std::vector<int32_t>>(MakeStatus(
            AttnStatusCode::kShapeMismatch, "ComputeSlotMapping: block_size/num_tokens <= 0"));
    }
    std::vector<int32_t> slots(static_cast<size_t>(num_tokens), -1);
    for (int64_t t = 0; t < num_tokens; ++t) {
        const int64_t position = seq_start + t;
        const int64_t logical_block = position / block_size;
        const int64_t offset_in_block = position % block_size;
        if (logical_block >= max_blocks) {
            return sci::MakeUnexpected<std::vector<int32_t>>(MakeStatus(
                AttnStatusCode::kKVCapacityExceeded,
                "ComputeSlotMapping: logical block " + std::to_string(logical_block) +
                    " >= max_blocks=" + std::to_string(max_blocks)));
        }
        const int32_t physical = block_table_row[logical_block];
        if (physical < 0) {
            return sci::MakeUnexpected<std::vector<int32_t>>(MakeStatus(
                AttnStatusCode::kShapeMismatch,
                "ComputeSlotMapping: block table slot " + std::to_string(logical_block) +
                    " is empty (-1) for position " + std::to_string(position)));
        }
        slots[static_cast<size_t>(t)] =
            static_cast<int32_t>(physical * block_size + offset_in_block);
    }
    return sci::Ok(std::move(slots));
}

sci::Result<KVCache> KVCache::Create(const KVCacheConfig& cfg, sci::Device& device) {
    const sci::Status valid = ValidateConfig(cfg);
    if (!valid.ok()) return sci::MakeUnexpected<KVCache>(valid);

    const sci::Result<size_t> bytes = KvCacheBytes(cfg);
    if (!bytes.ok()) return sci::MakeUnexpected<KVCache>(bytes.error());

    unsigned long long free_bytes = 0, total_bytes = 0;
    if (sca_mem_get_info(&free_bytes, &total_bytes) == 0) {
        const double budget = static_cast<double>(free_bytes) * kKvBudgetRatio;
        if (static_cast<double>(*bytes) > budget) {
            const int64_t suggested = MaxBlocksForBudget(
                cfg, static_cast<size_t>(budget / (2 * cfg.num_layers)));
            return sci::MakeUnexpected<KVCache>(MakeStatus(
                AttnStatusCode::kKVCapacityExceeded,
                "KV cache needs " + std::to_string(*bytes / (1024 * 1024)) +
                    " MiB but only " + std::to_string(free_bytes / (1024 * 1024)) +
                    " MiB is free (budget " + std::to_string(budget / (1024 * 1024)) +
                    " MiB); suggest num_blocks <= " + std::to_string(std::max<int64_t>(1, suggested))));
        }
    }

    KVCache cache;
    cache.cfg_ = cfg;
    cache.layer_elements_ = cfg.num_blocks * cfg.block_size * cfg.num_kv_heads * cfg.head_dim;
    const sci::TensorShape layer_shape({cfg.num_blocks, cfg.block_size, cfg.num_kv_heads,
                                        cfg.head_dim});
    cache.k_layers_.reserve(static_cast<size_t>(cfg.num_layers));
    cache.v_layers_.reserve(static_cast<size_t>(cfg.num_layers));
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        cache.k_layers_.emplace_back(layer_shape, cfg.dtype, device);
        cache.v_layers_.emplace_back(layer_shape, cfg.dtype, device);
        if (cache.k_layers_.back().data() == nullptr || cache.v_layers_.back().data() == nullptr) {
            return sci::MakeUnexpected<KVCache>(MakeStatus(
                AttnStatusCode::kKVCapacityExceeded,
                "KV cache allocation failed at layer " + std::to_string(layer) + " (" +
                    std::to_string(*bytes / (1024 * 1024)) + " MiB requested)"));
        }
    }
    cache.stats_.num_blocks = cfg.num_blocks;
    cache.stats_.free_blocks = cfg.num_blocks;
    cache.stats_.bytes = *bytes;
    cache.stats_.peak_bytes = *bytes;
    return sci::Ok(std::move(cache));
}

sci::Status KVCache::Append(int64_t layer, const sci::Tensor& k_new, const sci::Tensor& v_new,
                            const int32_t* slot_mapping, int64_t num_tokens, sci::Stream* stream) {
    if (layer < 0 || layer >= cfg_.num_layers) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "KVCache::Append layer=" + std::to_string(layer) +
                              " out of range [0," + std::to_string(cfg_.num_layers) + ")");
    }
    if (k_new.dtype() != cfg_.dtype || v_new.dtype() != cfg_.dtype) {
        return MakeStatus(AttnStatusCode::kUnsupportedDtype,
                          std::string("KVCache::Append dtype mismatch: cache=") +
                              sci::kDTypeName(cfg_.dtype) + ", k=" + sci::kDTypeName(k_new.dtype()));
    }
    if (k_new.ndims() != 3 || k_new.dim(0) != num_tokens || k_new.dim(1) != cfg_.num_kv_heads ||
        k_new.dim(2) != cfg_.head_dim) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "KVCache::Append expects k_new [num_tokens, num_kv_heads, head_dim]=" +
                              std::to_string(num_tokens) + "x" + std::to_string(cfg_.num_kv_heads) +
                              "x" + std::to_string(cfg_.head_dim));
    }
    if (slot_mapping == nullptr) {
        return MakeStatus(AttnStatusCode::kShapeMismatch, "KVCache::Append null slot_mapping");
    }

    const cuda::KVCacheShape shape = ToDeviceShape(cfg_);
    const sci::Status status =
        cuda::LaunchKvAppend(k_layers_[static_cast<size_t>(layer)].data(),
                             v_layers_[static_cast<size_t>(layer)].data(), k_new.data(),
                             v_new.data(), slot_mapping, num_tokens, shape, DtypeSize(cfg_.dtype),
                             stream);
    if (!status.ok()) return status;
    ++stats_.num_appends;
    return sci::Status::Ok();
}

sci::Result<sci::Tensor> KVCache::GatherK(int64_t layer, const int32_t* block_ids,
                                           int64_t num_blocks, sci::Stream* stream) const {
    return GatherInternal(layer, block_ids, num_blocks, true, stream);
}

sci::Result<sci::Tensor> KVCache::GatherV(int64_t layer, const int32_t* block_ids,
                                           int64_t num_blocks, sci::Stream* stream) const {
    return GatherInternal(layer, block_ids, num_blocks, false, stream);
}

sci::Result<sci::Tensor> KVCache::GatherInternal(int64_t layer, const int32_t* block_ids,
                                                 int64_t num_blocks, bool is_key,
                                                 sci::Stream* stream) const {
    if (layer < 0 || layer >= cfg_.num_layers) {
        return sci::MakeUnexpected<sci::Tensor>(MakeStatus(
            AttnStatusCode::kShapeMismatch,
            "KVCache::Gather layer=" + std::to_string(layer) + " out of range"));
    }
    if (block_ids == nullptr || num_blocks <= 0) {
        return sci::MakeUnexpected<sci::Tensor>(MakeStatus(
            AttnStatusCode::kShapeMismatch, "KVCache::Gather requires block_ids and num_blocks>0"));
    }
    const sci::Tensor& src =
        is_key ? k_layers_[static_cast<size_t>(layer)] : v_layers_[static_cast<size_t>(layer)];
    sci::Tensor out(sci::TensorShape({num_blocks, cfg_.block_size, cfg_.num_kv_heads,
                                      cfg_.head_dim}),
                    cfg_.dtype, src.device());
    if (out.data() == nullptr) {
        return sci::MakeUnexpected<sci::Tensor>(MakeStatus(
            AttnStatusCode::kWorkspaceExceeded, "KVCache::Gather allocation failed"));
    }
    const cuda::KVCacheShape shape = ToDeviceShape(cfg_);
    const sci::Status status =
        cuda::LaunchKvGather(src.data(), block_ids, num_blocks, out.data(), shape,
                             DtypeSize(cfg_.dtype), stream);
    if (!status.ok()) return sci::MakeUnexpected<sci::Tensor>(status);
    return sci::Ok(std::move(out));
}

sci::Status KVCache::Reset(const int32_t* block_ids, int64_t num_blocks, sci::Stream* stream) {
    if (block_ids == nullptr) {
        return MakeStatus(AttnStatusCode::kShapeMismatch, "KVCache::Reset null block_ids");
    }
    if (num_blocks <= 0) return sci::Status::Ok();
    const cuda::KVCacheShape shape = ToDeviceShape(cfg_);
    for (int64_t layer = 0; layer < cfg_.num_layers; ++layer) {
        const sci::Status status =
            cuda::LaunchKvResetBlock(k_layers_[static_cast<size_t>(layer)].data(),
                                     v_layers_[static_cast<size_t>(layer)].data(), block_ids,
                                     num_blocks, shape, DtypeSize(cfg_.dtype), stream);
        if (!status.ok()) return status;
    }
    ++stats_.num_resets;
    return sci::Status::Ok();
}

const sci::Tensor& KVCache::K(int64_t layer) const {
    static const sci::Tensor kEmpty;
    if (layer < 0 || layer >= cfg_.num_layers) return kEmpty;
    return k_layers_[static_cast<size_t>(layer)];
}

const sci::Tensor& KVCache::V(int64_t layer) const {
    static const sci::Tensor kEmpty;
    if (layer < 0 || layer >= cfg_.num_layers) return kEmpty;
    return v_layers_[static_cast<size_t>(layer)];
}

KVCacheStats KVCache::Stats() const { return stats_; }

size_t KVCache::MemoryBytes() const noexcept { return stats_.bytes; }

}  // namespace sca

