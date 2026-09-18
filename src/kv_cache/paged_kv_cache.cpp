#include "scicompute_attention/paged_kv_cache.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "device/device.hpp"
#include "device/stream.hpp"
#include "scicompute_attention/status.hpp"

namespace sca {

sci::Result<PagedKVCache> PagedKVCache::Create(const KVCacheConfig& cfg, sci::Device& device) {
    if (cfg.max_num_seqs <= 0) {
        return sci::MakeUnexpected<PagedKVCache>(MakeStatus(
            AttnStatusCode::kShapeMismatch,
            "PagedKVCacheConfig invalid: max_num_seqs=" + std::to_string(cfg.max_num_seqs)));
    }
    sci::Result<KVCache> storage = KVCache::Create(cfg, device);
    if (!storage.ok()) return sci::MakeUnexpected<PagedKVCache>(storage.error());

    // Blocks are exclusive per sequence, so the table row length is simply all blocks of the pool
    // (the caller-sized pool already encodes the concurrency budget).
    const int64_t blocks_per_seq = cfg.num_blocks;
    sci::Result<BlockManager> blocks = BlockManager::Create(cfg.num_blocks, cfg.max_num_seqs,
                                                            blocks_per_seq, device);
    if (!blocks.ok()) return sci::MakeUnexpected<PagedKVCache>(blocks.error());

    PagedKVCache cache;
    cache.cfg_ = cfg;
    cache.max_blocks_per_seq_ = blocks_per_seq;
    cache.storage_ = std::move(*storage);
    cache.blocks_ = std::move(*blocks);
    cache.seq_lengths_.assign(static_cast<size_t>(cfg.max_num_seqs), 0);
    cache.seq_blocks_.assign(static_cast<size_t>(cfg.max_num_seqs), {});
    cache.page_table_host_.assign(static_cast<size_t>(cfg.max_num_seqs * blocks_per_seq), -1);
    cache.page_table_device_ =
        sci::Tensor(sci::TensorShape({cfg.max_num_seqs, blocks_per_seq}), sci::DType::kInt32,
                    device);
    if (cache.page_table_device_.data() == nullptr) {
        return sci::MakeUnexpected<PagedKVCache>(MakeStatus(
            AttnStatusCode::kKVCapacityExceeded, "PagedKVCache: page table allocation failed"));
    }
    cache.page_table_device_.copy_from(cache.page_table_host_.data(),
                                       cache.page_table_host_.size() * sizeof(int32_t));
    device.synchronize();
    cache.revision_ = 1;
    cache.synced_revision_ = 1;
    return sci::Ok(std::move(cache));
}

sci::Status PagedKVCache::AppendTokens(int64_t seq_id, int64_t layer,
                                       const sci::Tensor& k_new, const sci::Tensor& v_new,
                                       sci::Stream* stream) {
    if (seq_id < 0 || seq_id >= cfg_.max_num_seqs) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "PagedKVCache::AppendTokens seq_id=" + std::to_string(seq_id) +
                              " out of range [0," + std::to_string(cfg_.max_num_seqs) + ")");
    }
    if (layer < 0 || layer >= cfg_.num_layers) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "PagedKVCache::AppendTokens layer=" + std::to_string(layer) +
                              " out of range [0," + std::to_string(cfg_.num_layers) + ")");
    }
    if (k_new.ndims() != 3 || v_new.ndims() != 3 || k_new.dim(0) != v_new.dim(0)) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "PagedKVCache::AppendTokens expects [num_tokens, n_kv_heads, head_dim] "
                          "tensors");
    }
    const int64_t num_new = k_new.dim(0);
    if (num_new <= 0) {
        return MakeStatus(AttnStatusCode::kShapeMismatch, "PagedKVCache::AppendTokens: 0 tokens");
    }

    auto& owned = seq_blocks_[static_cast<size_t>(seq_id)];
    const int64_t current_len = seq_lengths_[static_cast<size_t>(seq_id)];
    const int64_t needed_blocks =
        (current_len + num_new + cfg_.block_size - 1) / cfg_.block_size;
    const int64_t to_allocate = needed_blocks - static_cast<int64_t>(owned.size());

    std::vector<int32_t> new_blocks;
    if (to_allocate > 0) {
        sci::Result<std::vector<int32_t>> allocated = blocks_.Allocate(to_allocate);
        if (!allocated.ok()) {
            // State is untouched: no partial update on failure (prompt §8.6).
            return MakeStatus(AttnStatusCode::kKVCapacityExceeded,
                              "PagedKVCache::AppendTokens seq=" + std::to_string(seq_id) +
                                  " needs " + std::to_string(to_allocate) +
                                  " more blocks but only " +
                                  std::to_string(blocks_.NumFreeBlocks()) + " are free");
        }
        new_blocks = std::move(*allocated);
    }

    sci::Status status = sci::Status::Ok();
    if (!new_blocks.empty()) {
        status = blocks_.AppendToTable(seq_id, new_blocks.data(),
                                       static_cast<int64_t>(new_blocks.size()));
        if (!status.ok()) {
            blocks_.Free(new_blocks.data(), static_cast<int64_t>(new_blocks.size()));
            return status;
        }
        for (const int32_t block : new_blocks) owned.push_back(block);
    }

    // Refresh the host page table row for this sequence.
    int32_t* row = page_table_host_.data() + seq_id * max_blocks_per_seq_;
    for (int64_t i = 0; i < max_blocks_per_seq_; ++i) row[i] = -1;
    for (size_t i = 0; i < owned.size(); ++i) row[i] = owned[i];

    const sci::Result<std::vector<int32_t>> slots =
        ComputeSlotMapping(row, current_len, num_new, cfg_.block_size, max_blocks_per_seq_);
    if (!slots.ok()) return slots.error();

    // The KV append kernel consumes device-resident slots; upload the freshly computed mapping.
    // This is a 4 B per token H2D copy on the append path (documented in docs/kv_cache.md); a future
    // scheduler-driven variant will let the caller supply device-side slots directly.
    if (slot_mapping_capacity_ < num_new) {
        const int64_t capacity = std::max<int64_t>(num_new, 1024);
        slot_mapping_device_ = sci::Tensor(sci::TensorShape({capacity}), sci::DType::kInt32,
                                           storage_.K(0).device());
        if (slot_mapping_device_.data() == nullptr) {
            return MakeStatus(AttnStatusCode::kKVCapacityExceeded,
                              "PagedKVCache: slot-mapping buffer allocation failed");
        }
        slot_mapping_capacity_ = capacity;
    }
    storage_.K(0).device().copy_to_device(slot_mapping_device_.data(), slots->data(),
                                          static_cast<size_t>(num_new) * sizeof(int32_t));

    status = storage_.Append(layer, k_new, v_new,
                             static_cast<const int32_t*>(slot_mapping_device_.data()), num_new,
                             stream);
    if (!status.ok()) {
        // The blocks are already attached to the sequence; the caller can ResetSeq to recover.
        return status;
    }
    seq_lengths_[static_cast<size_t>(seq_id)] = current_len + num_new;
    ++revision_;
    ++num_appends_;
    return sci::Status::Ok();
}

sci::Status PagedKVCache::ResetSeq(int64_t seq_id, sci::Stream* stream) {
    if (seq_id < 0 || seq_id >= cfg_.max_num_seqs) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "PagedKVCache::ResetSeq seq_id=" + std::to_string(seq_id) +
                              " out of range");
    }
    auto& owned = seq_blocks_[static_cast<size_t>(seq_id)];
    if (!owned.empty()) {
        // Order matters: clear device data first, then release the blocks, then drop the table row
        // (prompt §8.6) so no dangling reference can be observed.
        const sci::Status cleared =
            storage_.Reset(owned.data(), static_cast<int64_t>(owned.size()), stream);
        if (!cleared.ok()) return cleared;
    }
    blocks_.ResetSeq(seq_id);
    owned.clear();
    seq_lengths_[static_cast<size_t>(seq_id)] = 0;
    int32_t* row = page_table_host_.data() + seq_id * max_blocks_per_seq_;
    for (int64_t i = 0; i < max_blocks_per_seq_; ++i) row[i] = -1;
    ++revision_;
    ++num_resets_;
    return sci::Status::Ok();
}

const int32_t* PagedKVCache::PageTableDevicePtr() const noexcept {
    return static_cast<const int32_t*>(page_table_device_.data());
}

int64_t PagedKVCache::LogicalBlockCount(int64_t seq_id) const noexcept {
    if (seq_id < 0 || seq_id >= cfg_.max_num_seqs) return 0;
    return static_cast<int64_t>(seq_blocks_[static_cast<size_t>(seq_id)].size());
}

int64_t PagedKVCache::SeqLength(int64_t seq_id) const noexcept {
    if (seq_id < 0 || seq_id >= cfg_.max_num_seqs) return 0;
    return seq_lengths_[static_cast<size_t>(seq_id)];
}

sci::Status PagedKVCache::SyncPageTable(sci::Stream* stream) {
    if (revision_ == synced_revision_) return sci::Status::Ok();
    return ForceSyncPageTable(stream);
}

sci::Status PagedKVCache::ForceSyncPageTable(sci::Stream* stream) {
    if (page_table_device_.data() == nullptr) {
        return MakeStatus(AttnStatusCode::kShapeMismatch, "PagedKVCache: page table missing");
    }
    if (stream != nullptr) {
        sci::Device& device = page_table_device_.device();
        device.copy_async(page_table_device_.data(), page_table_host_.data(),
                          page_table_host_.size() * sizeof(int32_t), *stream);
    } else {
        page_table_device_.copy_from(page_table_host_.data(),
                                     page_table_host_.size() * sizeof(int32_t));
    }
    synced_revision_ = revision_;
    return sci::Status::Ok();
}

KVCacheStats PagedKVCache::Stats() const {
    KVCacheStats stats = storage_.Stats();
    stats.used_blocks = blocks_.NumTotalBlocks() - blocks_.NumFreeBlocks();
    stats.free_blocks = blocks_.NumFreeBlocks();
    stats.num_blocks = blocks_.NumTotalBlocks();
    stats.num_appends = num_appends_;
    stats.num_resets = num_resets_;
    return stats;
}

}  // namespace sca
