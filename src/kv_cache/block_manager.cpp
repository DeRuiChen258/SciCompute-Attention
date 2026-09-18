#include "scicompute_attention/block_manager.hpp"

#include <string>

#include "device/device.hpp"
#include "device/stream.hpp"
#include "scicompute_attention/status.hpp"

namespace sca {

sci::Result<BlockManager> BlockManager::Create(int64_t num_blocks, int64_t max_num_seqs,
                                               int64_t max_blocks_per_seq, sci::Device& device) {
    if (num_blocks <= 0 || max_num_seqs <= 0 || max_blocks_per_seq <= 0) {
        return sci::MakeUnexpected<BlockManager>(MakeStatus(
            AttnStatusCode::kShapeMismatch,
            "BlockManager::Create invalid arguments: num_blocks=" + std::to_string(num_blocks) +
                ", max_num_seqs=" + std::to_string(max_num_seqs) +
                ", max_blocks_per_seq=" + std::to_string(max_blocks_per_seq)));
    }
    BlockManager manager;
    manager.num_blocks_ = num_blocks;
    manager.in_use_.assign(static_cast<size_t>(num_blocks), 0);
    // LIFO free list: allocate pops from the back, free pushes back -> O(1) amortised.
    manager.free_list_.reserve(static_cast<size_t>(num_blocks));
    for (int64_t i = num_blocks - 1; i >= 0; --i) {
        manager.free_list_.push_back(static_cast<int32_t>(i));
    }
    manager.table_.max_num_seqs = max_num_seqs;
    manager.table_.max_blocks_per_seq = max_blocks_per_seq;
    manager.table_.host.assign(static_cast<size_t>(max_num_seqs * max_blocks_per_seq), -1);
    manager.table_.device = sci::Tensor(
        sci::TensorShape({max_num_seqs, max_blocks_per_seq}), sci::DType::kInt32, device);
    if (manager.table_.device.data() == nullptr) {
        return sci::MakeUnexpected<BlockManager>(MakeStatus(
            AttnStatusCode::kKVCapacityExceeded,
            "BlockManager: page table allocation of " +
                std::to_string(max_num_seqs * max_blocks_per_seq * 4) + " B failed"));
    }
    // Mirror the host table (all -1) so a device-side read before the first sync is well defined.
    manager.table_.device.copy_from(manager.table_.host.data(),
                                    manager.table_.host.size() * sizeof(int32_t));
    device.synchronize();
    return sci::Ok(std::move(manager));
}

sci::Result<std::vector<int32_t>> BlockManager::Allocate(int64_t count) {
    if (count <= 0) {
        return sci::MakeUnexpected<std::vector<int32_t>>(MakeStatus(
            AttnStatusCode::kShapeMismatch, "BlockManager::Allocate count=" + std::to_string(count)));
    }
    if (count > static_cast<int64_t>(free_list_.size())) {
        ++allocation_failures_;
        return sci::MakeUnexpected<std::vector<int32_t>>(MakeStatus(
            AttnStatusCode::kKVCapacityExceeded,
            "BlockManager::Allocate requested " + std::to_string(count) + " blocks but only " +
                std::to_string(free_list_.size()) + " of " + std::to_string(num_blocks_) +
                " are free"));
    }
    std::vector<int32_t> allocated(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        const int32_t block = free_list_.back();
        free_list_.pop_back();
        in_use_[static_cast<size_t>(block)] = 1;
        allocated[static_cast<size_t>(i)] = block;
    }
    return sci::Ok(std::move(allocated));
}

void BlockManager::Free(const int32_t* block_ids, int64_t count) {
    if (block_ids == nullptr || count <= 0) return;
    for (int64_t i = 0; i < count; ++i) {
        const int32_t block = block_ids[i];
        if (block < 0 || block >= num_blocks_) {
            ++double_free_events_;
            continue;
        }
        if (in_use_[static_cast<size_t>(block)] == 0) {
            ++double_free_events_;  // releasing a block that was not allocated
            continue;
        }
        in_use_[static_cast<size_t>(block)] = 0;
        free_list_.push_back(block);
    }
}

sci::Status BlockManager::AppendToTable(int64_t seq_id, const int32_t* block_ids, int64_t count) {
    if (seq_id < 0 || seq_id >= table_.max_num_seqs) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "BlockManager::AppendToTable seq_id=" + std::to_string(seq_id) +
                              " out of range [0," + std::to_string(table_.max_num_seqs) + ")");
    }
    if (block_ids == nullptr || count <= 0) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "BlockManager::AppendToTable requires block_ids and count > 0");
    }
    int32_t* row = table_.host.data() + seq_id * table_.max_blocks_per_seq;
    int64_t used = 0;
    while (used < table_.max_blocks_per_seq && row[used] >= 0) ++used;
    if (used + count > table_.max_blocks_per_seq) {
        return MakeStatus(AttnStatusCode::kKVCapacityExceeded,
                          "BlockManager::AppendToTable sequence " + std::to_string(seq_id) +
                              " would need " + std::to_string(used + count) +
                              " blocks > max_blocks_per_seq=" +
                              std::to_string(table_.max_blocks_per_seq));
    }
    for (int64_t i = 0; i < count; ++i) row[used + i] = block_ids[i];
    ++table_.revision;
    return sci::Status::Ok();
}

sci::Status BlockManager::ResetSeq(int64_t seq_id) {
    if (seq_id < 0 || seq_id >= table_.max_num_seqs) {
        return MakeStatus(AttnStatusCode::kShapeMismatch,
                          "BlockManager::ResetSeq seq_id=" + std::to_string(seq_id) +
                              " out of range");
    }
    int32_t* row = table_.host.data() + seq_id * table_.max_blocks_per_seq;
    int64_t count = 0;
    for (int64_t i = 0; i < table_.max_blocks_per_seq; ++i) {
        if (row[i] >= 0) ++count;
    }
    if (count > 0) Free(row, count);
    for (int64_t i = 0; i < table_.max_blocks_per_seq; ++i) row[i] = -1;
    ++table_.revision;
    return sci::Status::Ok();
}

sci::Status BlockManager::SyncTableToDevice(sci::Stream* stream) {
    if (table_.synced_revision == table_.revision && table_.device.data() != nullptr) {
        return sci::Status::Ok();  // nothing changed since the last upload
    }
    if (table_.device.data() == nullptr) {
        return MakeStatus(AttnStatusCode::kShapeMismatch, "BlockManager: page table not allocated");
    }
    if (stream != nullptr) {
        sci::Device& device = table_.device.device();
        device.copy_async(table_.device.data(), table_.host.data(),
                          table_.host.size() * sizeof(int32_t), *stream);
    } else {
        table_.device.copy_from(table_.host.data(), table_.host.size() * sizeof(int32_t));
    }
    table_.synced_revision = table_.revision;
    return sci::Status::Ok();
}

}  // namespace sca

