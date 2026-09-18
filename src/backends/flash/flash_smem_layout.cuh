// Shared-memory partition of the flash kernel (prompt §7.4.3).
//
// Layout: Q tile | K tiles (kStages) | V tiles (kStages), every row padded by kRowPadElems halves.
// The S/P tiles are NOT in shared memory: they live in the mma fragments (registers), which is the
// FA2-style target of this project.
#pragma once

#include <cstdint>

#include "cuda_common/arch_features.cuh"

namespace sca {
namespace cuda {

template <typename T, int kHeadDim, int kBlockM, int kBlockN, int kStages>
struct FlashSmemLayout {
    static constexpr int kRowStride = kHeadDim + kRowPadElems;
    static constexpr int kQTileElems = kBlockM * kRowStride;
    static constexpr int kKvTileElems = kBlockN * kRowStride;

    T q[kQTileElems];
    T k[kStages][kKvTileElems];
    T v[kStages][kKvTileElems];

    static constexpr int64_t kBytes =
        static_cast<int64_t>(sizeof(T)) * (kQTileElems + 2 * kStages * kKvTileElems);

    // Upper bound from docs/env_report.md (cudaDevAttrMaxSharedMemoryPerBlockOptin = 101376 B).
    // tools/smem_calc.py mirrors this number so that the two can never drift apart.
    static_assert(kBytes <= kMaxDynamicSmemBytes,
                  "flash tile exceeds the measured shared-memory limit; "
                  "regenerate the tile table with tools/tile_sweep.py");
};

}  // namespace cuda
}  // namespace sca

