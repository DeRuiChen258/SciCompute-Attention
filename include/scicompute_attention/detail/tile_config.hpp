#pragma once

#include <cstdint>
#include <string>

#include "scicompute_attention/export.hpp"

namespace sca {

// Tile configuration shared by all tiled/MMA backends.
//
// Semantics (all backends use the same field names, prompt §7.0):
//   block_m : rows of Q handled by one CTA
//   block_n : columns of K/V consumed per pipeline iteration
//   warps   : warps per CTA (block threads = warps * 32, always a warp multiple)
//   stages  : cp.async / TMA pipeline depth
//   use_tma : request the TMA bulk path when the runtime probe allows it
struct TileConfig {
    int32_t block_m{0};
    int32_t block_n{0};
    int32_t warps{0};
    int32_t stages{0};
    bool use_tma{false};

    constexpr int Threads() const noexcept { return warps * 32; }
    constexpr bool Empty() const noexcept { return block_m == 0 || block_n == 0 || warps == 0; }

    // Sizes the shared-memory footprint of the flash-style layout for a given head_dim:
    //   Q tile + `stages` K tiles + `stages` V tiles, each row padded by kSmemPadElems halves so
    //   that the per-row byte stride stops being a multiple of 128 B (bank-conflict mitigation).
    // Returns -1 when the configuration is structurally invalid.
    static constexpr int64_t SmemBytesFlash(int32_t head_dim, int64_t elem_bytes, int32_t block_m,
                                            int32_t block_n, int32_t warps, int32_t stages) {
        if (head_dim <= 0 || block_m <= 0 || block_n <= 0 || warps <= 0 || stages <= 0) return -1;
        const int64_t pad = kSmemPadElems;
        const int64_t q = static_cast<int64_t>(block_m) * (head_dim + pad) * elem_bytes;
        const int64_t kv = static_cast<int64_t>(block_n) * (head_dim + pad) * elem_bytes;
        return q + 2 * stages * kv;
    }

    int64_t SmemBytesFlash(int32_t head_dim, int64_t elem_bytes) const {
        return SmemBytesFlash(head_dim, elem_bytes, block_m, block_n, warps, stages);
    }

    // 16 B (8 fp16 elements) keeps every padded row 16 B aligned, which ldmatrix requires.
    static constexpr int32_t kSmemPadElems = 8;

    std::string ToString() const;
};

inline std::string TileConfig::ToString() const {
    return "bm=" + std::to_string(block_m) + ",bn=" + std::to_string(block_n) +
           ",warps=" + std::to_string(warps) + ",stages=" + std::to_string(stages) +
           (use_tma ? ",tma=on" : ",tma=off");
}

}  // namespace sca

