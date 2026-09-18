// Paged attention entry point.
//
// Implementation note: paged attention shares the split-K decode kernel (the S_q = 1 serving case)
// and only swaps the K/V address computation for a page-table lookup. Re-using the kernel instead
// of copying it keeps the merge formula, the masking convention and the numerics in one place
// (prompt §7.6 asks for the indirection, not for a second implementation of the softmax).
#pragma once

#include <cstdint>

#include "backends/decode/decode_splitk.cuh"
#include "core/status.hpp"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {
namespace cuda {

struct PagedDecodeParams {
    DecodeFwdParams base{};          // q/k/v/out/partials and geometry
    const int32_t* page_table{nullptr};   // [num_seqs, max_blocks_per_seq]
    int64_t block_size{16};
    int64_t max_blocks_per_seq{0};
    int64_t num_kv_blocks{0};
};

// K/V pointers in `params.base` must point at the *storage origin* (layer base);
// `base.k_stride.s_s` / `base.v_stride.s_s` are ignored for the token axis (replaced by the page
// table row lookup), while the per-row element layout stays [num_kv_heads][head_dim].
sci::Status LaunchPagedAttention(const PagedDecodeParams& params, sci::Stream* stream);

}  // namespace cuda
}  // namespace sca

