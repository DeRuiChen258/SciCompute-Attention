// Host-side flash tile table (prompt §7.4.5).
//
// The numbers are seed values derived from the smem/register budget in docs/kernel_design.md; they
// are re-calibrated by tools/tile_sweep.py in Phase 6 and the outcome is recorded in
// docs/tile_config.md. Nothing in the kernel is allowed to hardcode a tile: it always comes from
// this table (or from a calibrated replacement).
#pragma once

#include <cstdint>
#include <string>

#include "cuda_common/arch_features.cuh"

namespace sca {
namespace flash {

struct FlashTile {
    int32_t block_m{0};
    int32_t block_n{0};
    int32_t warps{0};
    int32_t stages{0};
    int32_t splits_d{1};

    std::string ToString() const {
        return "bm=" + std::to_string(block_m) + ",bn=" + std::to_string(block_n) +
               ",warps=" + std::to_string(warps) + ",stages=" + std::to_string(stages) +
               ",splits_d=" + std::to_string(splits_d);
    }
};

struct FlashTileEntry {
    int64_t head_dim;
    FlashTile tile;
};

// One calibrated configuration per supported head dimension. head_dim > 128 splits the PV output
// across warp pairs so that the O accumulator stays inside the register budget.
inline constexpr FlashTileEntry kFlashTiles[] = {
    {32, {64, 64, 4, 2, 1}},
    {64, {64, 64, 4, 2, 1}},
    {96, {64, 64, 4, 2, 1}},
    {128, {64, 64, 4, 2, 1}},
    {160, {64, 32, 8, 2, 2}},
    {192, {64, 32, 8, 2, 2}},
    {256, {32, 32, 4, 2, 2}},
};

inline const FlashTile* FindFlashTile(int64_t head_dim) {
    for (const FlashTileEntry& entry : kFlashTiles) {
        if (entry.head_dim == head_dim) return &entry.tile;
    }
    return nullptr;
}

// Shared-memory footprint of a flash tile: Q + kStages K + kStages V, rows padded by kRowPadElems.
inline int64_t FlashTileSmemBytes(const FlashTile& tile, int64_t head_dim, int64_t elem_bytes) {
    const int64_t row_stride = head_dim + cuda::kRowPadElems;
    const int64_t q = static_cast<int64_t>(tile.block_m) * row_stride * elem_bytes;
    const int64_t kv =
        static_cast<int64_t>(tile.block_n) * row_stride * elem_bytes * tile.stages;
    return q + 2 * kv;
}

}  // namespace flash
}  // namespace sca
