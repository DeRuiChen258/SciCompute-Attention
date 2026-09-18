// Split-K decode attention kernels (prompt §7.5).
#pragma once

#include <cstdint>

#include "core/status.hpp"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {
namespace cuda {

struct DecodeStrides {
    int64_t s_b{0};
    int64_t s_h{0};
    int64_t s_s{0};
};

enum class DecodeDtype : int32_t { kFp32 = 0, kFp16 = 1, kBf16 = 2 };

struct DecodeFwdParams {
    const void* q{nullptr};
    const void* k{nullptr};
    const void* v{nullptr};
    void* out{nullptr};
    float* partial_m{nullptr};
    float* partial_l{nullptr};
    float* partial_o{nullptr};
    int64_t batch{0};
    int64_t num_heads{0};
    int64_t num_kv_heads{0};
    int64_t seq_kv{0};
    int64_t head_dim{0};
    int64_t num_splits{1};
    DecodeStrides q_stride{};
    DecodeStrides k_stride{};
    DecodeStrides v_stride{};
    DecodeStrides o_stride{};
    float scale{0.0f};
    DecodeDtype dtype{DecodeDtype::kFp16};
    // Optional page-table indirection (paged attention). When page_table != nullptr the token axis
    // of K/V is resolved through the page table instead of a linear stride.
    const int32_t* page_table{nullptr};
    int64_t block_size{0};
    int64_t max_blocks_per_seq{0};
    int64_t num_kv_blocks{0};
    // Optional per-sequence lengths (paged/multi-request decode). nullptr => use seq_kv for all.
    const int32_t* seq_lens{nullptr};
};

sci::Status LaunchDecodeSplitK(const DecodeFwdParams& params, sci::Stream* stream);

// Same kernels with the page table enabled (declared here so that both entry points share one
// template instantiation).
sci::Status LaunchDecodeSplitKPaged(const DecodeFwdParams& params, sci::Stream* stream);

}  // namespace cuda
}  // namespace sca
