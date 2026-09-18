// FlashAttention-style forward kernel (prompt §7.4).
//
// Grid : (ceil(S_q / kBlockM), H_q, B)      Block : kWarps * 32 threads
// Each CTA owns one (batch, head, q-block); each warp pair/group owns 16 rows.
//
// Warp decomposition:
//   kWarpsM   = kBlockM / 16          row groups (one m16 tile per warp)
//   kSplitsD  = 1 for head_dim <= 128, 2 for larger head dims
//   warp_m    = warp / kSplitsD       row group index
//   d_split   = warp % kSplitsD       which half of the head dimension this warp accumulates
//   For kSplitsD == 2 the two warps of a pair recompute the same S tiles (duplicated QK^T) and
//   split the PV output; this keeps Q/O fragments inside the register budget for D=256.
//
// Registers per thread (D=128, kBlockN=64, kSplitsD=1):
//   Q fragments 8*4 = 32, P/S fragments 8*4 = 32, O accumulator 16*4 = 64, temps ~24  -> ~160
//
// math: S = Q K^T * scale, then online softmax (see flash_online_softmax.cuh), O = P V.

#pragma once

#include <cstdint>

#include "core/status.hpp"

namespace sci {
class Stream;
}

namespace sca {
namespace cuda {

struct FlashFwdParams {
    const void* q{nullptr};
    const void* k{nullptr};
    const void* v{nullptr};
    void* out{nullptr};
    float* lse{nullptr};
    int64_t batch{0};
    int64_t num_heads{0};
    int64_t num_kv_heads{0};
    int64_t seq_q{0};
    int64_t seq_kv{0};
    int64_t head_dim{0};
    int64_t q_sb{0}, q_sh{0}, q_ss{0};
    int64_t k_sb{0}, k_sh{0}, k_ss{0};
    int64_t v_sb{0}, v_sh{0}, v_ss{0};
    int64_t o_sb{0}, o_sh{0}, o_ss{0};
    float scale{0.0f};
    bool causal{false};
    bool return_lse{false};
    int64_t diagonal{0};
    int dtype_id{0};  // 0 = fp32 (reference), 1 = fp16, 2 = bf16
};

// Compile-time smem footprint of one flash configuration (used by the launch-time checks).
int64_t FlashSmemBytesFor(int32_t head_dim, int64_t elem_bytes, int32_t block_m, int32_t block_n,
                          int32_t stages);

// Launches the flash forward kernel for the geometry recorded in the tile table.
sci::Status LaunchFlashFwd(const FlashFwdParams& params, sci::Stream* stream);

}  // namespace cuda
}  // namespace sca
