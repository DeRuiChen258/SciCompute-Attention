// Level 1 (tiled) attention: the online-softmax algorithm with S staged in shared memory.
//
// Difference from the flash backend (Level 3):
//   * S/P tiles live in shared memory, not in MMA fragments - this is the explicit "keep the row
//     softmax in smem" intermediate form asked for in the prompt (§7.3), and it is the reference
//     that the register-resident flash path is validated against;
//   * computation uses SIMT FMA instead of mma.sync, so no tensor-core layout constraints apply;
//   * O is accumulated in registers (float4 per thread), never in HBM.
//
// Supported range: head_dim <= 128 (above that the smem budget is exceeded and the backend tells
// the caller to use flash - it never silently degrades).
#pragma once

#include <cstdint>

#include "core/status.hpp"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {
namespace cuda {

struct TiledStrides {
    int64_t s_b{0};
    int64_t s_h{0};
    int64_t s_s{0};
};

enum class TiledDtype : int32_t { kFp32 = 0, kFp16 = 1, kBf16 = 2 };

struct TiledFwdParams {
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
    TiledStrides q_stride{};
    TiledStrides k_stride{};
    TiledStrides v_stride{};
    TiledStrides o_stride{};
    float scale{0.0f};
    bool causal{false};
    bool return_lse{false};
    int64_t diagonal{0};
    TiledDtype dtype{TiledDtype::kFp16};
};

// smem bytes for one (head_dim, block_m, block_n) configuration; -1 when invalid.
int64_t TiledSmemBytes(int32_t head_dim, int64_t elem_bytes, int32_t block_m, int32_t block_n);

sci::Status LaunchTiledForward(const TiledFwdParams& params, sci::Stream* stream,
                               int32_t block_m, int32_t block_n, int32_t warps);

}  // namespace cuda
}  // namespace sca

