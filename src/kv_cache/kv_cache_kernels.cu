#include "kv_cache/kv_cache_kernels.cuh"

#include <cuda_runtime.h>

#include <cstdint>

#include "device/stream.hpp"
#include "runtime/launcher.hpp"

namespace sca {
namespace cuda {
namespace {

constexpr int kThreads = 256;

// append: dst[slot, head, :] = src[token, head, :]
__global__ void kv_append_kernel(void* layer_k, void* layer_v, const void* k_new,
                                 const void* v_new, const int32_t* slot_mapping,
                                 int64_t num_tokens, KVCacheShape shape, int64_t elem_bytes) {
    const int64_t total_rows = num_tokens * shape.num_kv_heads;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < total_rows; row += stride) {
        const int64_t token = row / shape.num_kv_heads;
        const int64_t head = row % shape.num_kv_heads;
        const int32_t slot = slot_mapping[token];
        if (slot < 0) continue;  // documented sentinel: negative slot means "do not store"
        const int64_t block = slot / shape.block_size;
        const int64_t offset_in_block = slot % shape.block_size;
        if (block >= shape.num_blocks) continue;  // defensive: never write out of range

        const int64_t row_bytes = shape.head_dim * elem_bytes;
        const unsigned char* src_k = static_cast<const unsigned char*>(k_new) +
                                     (token * shape.num_kv_heads + head) * row_bytes;
        const unsigned char* src_v = static_cast<const unsigned char*>(v_new) +
                                     (token * shape.num_kv_heads + head) * row_bytes;
        unsigned char* dst_k = static_cast<unsigned char*>(layer_k) +
                               KvOffset(shape, block, offset_in_block, head, 0) * elem_bytes;
        unsigned char* dst_v = static_cast<unsigned char*>(layer_v) +
                               KvOffset(shape, block, offset_in_block, head, 0) * elem_bytes;
        for (int64_t b = 0; b < row_bytes; ++b) {
            dst_k[b] = src_k[b];
            dst_v[b] = src_v[b];
        }
    }
}

__global__ void kv_reset_kernel(void* layer_k, void* layer_v, const int32_t* block_ids,
                                int64_t num_blocks, KVCacheShape shape, int64_t elem_bytes) {
    const int64_t per_block = shape.block_size * shape.num_kv_heads * shape.head_dim * elem_bytes;
    const int64_t total = num_blocks * per_block;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    auto* k_bytes = static_cast<unsigned char*>(layer_k);
    auto* v_bytes = static_cast<unsigned char*>(layer_v);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        const int64_t block_index = idx / per_block;
        const int64_t offset = idx % per_block;
        const int32_t physical = block_ids[block_index];
        if (physical < 0 || physical >= shape.num_blocks) continue;
        const int64_t base = static_cast<int64_t>(physical) * per_block + offset;
        k_bytes[base] = 0;
        v_bytes[base] = 0;
    }
}

__global__ void kv_gather_kernel(const void* layer, const int32_t* block_ids, int64_t num_blocks,
                                 void* out, KVCacheShape shape, int64_t elem_bytes) {
    const int64_t per_block = shape.block_size * shape.num_kv_heads * shape.head_dim * elem_bytes;
    const int64_t total = num_blocks * per_block;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const auto* src = static_cast<const unsigned char*>(layer);
    auto* dst = static_cast<unsigned char*>(out);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        const int64_t block_index = idx / per_block;
        const int64_t offset = idx % per_block;
        const int32_t physical = block_ids[block_index];
        if (physical < 0 || physical >= shape.num_blocks) {
            dst[idx] = 0;
            continue;
        }
        dst[idx] = src[static_cast<int64_t>(physical) * per_block + offset];
    }
}

int BlocksFor(int64_t work_items) {
    const int64_t blocks = (work_items + kThreads - 1) / kThreads;
    if (blocks < 1) return 1;
    return static_cast<int>(blocks > (1 << 20) ? (1 << 20) : blocks);
}

}  // namespace

sci::Status LaunchKvAppend(void* layer_k, void* layer_v, const void* k_new, const void* v_new,
                           const int32_t* slot_mapping, int64_t num_tokens,
                           const KVCacheShape& shape, int64_t elem_bytes, sci::Stream* stream) {
    if (layer_k == nullptr || layer_v == nullptr || k_new == nullptr || v_new == nullptr ||
        slot_mapping == nullptr) {
        return sci::Status::InvalidArgument("kv append: null pointer");
    }
    if (num_tokens <= 0) return sci::Status::InvalidArgument("kv append: num_tokens <= 0");
    const int blocks = BlocksFor(num_tokens * shape.num_kv_heads);
    KVCacheShape shape_copy = shape;
    void* k_dst = layer_k;
    void* v_dst = layer_v;
    const void* k_src = k_new;
    const void* v_src = v_new;
    const int32_t* slots = slot_mapping;
    int64_t tokens = num_tokens;
    int64_t bytes = elem_bytes;
    void* args[] = {&k_dst, &v_dst, &k_src, &v_src, &slots, &tokens, &shape_copy, &bytes};
    return runtime::LaunchRaw(reinterpret_cast<const void*>(&kv_append_kernel), blocks, 1, 1,
                              kThreads, 1, 1, 0, stream, args);
}

sci::Status LaunchKvResetBlock(void* layer_k, void* layer_v, const int32_t* block_ids,
                               int64_t num_blocks, const KVCacheShape& shape, int64_t elem_bytes,
                               sci::Stream* stream) {
    if (layer_k == nullptr || layer_v == nullptr || block_ids == nullptr) {
        return sci::Status::InvalidArgument("kv reset: null pointer");
    }
    if (num_blocks <= 0) return sci::Status::Ok();
    const int64_t per_block = shape.block_size * shape.num_kv_heads * shape.head_dim * elem_bytes;
    const int blocks = BlocksFor(num_blocks * per_block);
    KVCacheShape shape_copy = shape;
    void* k_dst = layer_k;
    void* v_dst = layer_v;
    const int32_t* ids = block_ids;
    int64_t count = num_blocks;
    int64_t bytes = elem_bytes;
    void* args[] = {&k_dst, &v_dst, &ids, &count, &shape_copy, &bytes};
    return runtime::LaunchRaw(reinterpret_cast<const void*>(&kv_reset_kernel), blocks, 1, 1,
                              kThreads, 1, 1, 0, stream, args);
}

sci::Status LaunchKvGather(const void* layer, const int32_t* block_ids, int64_t num_blocks,
                           void* out, const KVCacheShape& shape, int64_t elem_bytes,
                           sci::Stream* stream) {
    if (layer == nullptr || block_ids == nullptr || out == nullptr) {
        return sci::Status::InvalidArgument("kv gather: null pointer");
    }
    if (num_blocks <= 0) return sci::Status::Ok();
    const int64_t per_block = shape.block_size * shape.num_kv_heads * shape.head_dim * elem_bytes;
    const int blocks = BlocksFor(num_blocks * per_block);
    KVCacheShape shape_copy = shape;
    const int32_t* ids = block_ids;
    int64_t count = num_blocks;
    int64_t bytes = elem_bytes;
    void* dst = out;
    const void* src = layer;
    void* args[] = {&src, &ids, &count, &dst, &shape_copy, &bytes};
    return runtime::LaunchRaw(reinterpret_cast<const void*>(&kv_gather_kernel), blocks, 1, 1,
                              kThreads, 1, 1, 0, stream, args);
}

}  // namespace cuda
}  // namespace sca

