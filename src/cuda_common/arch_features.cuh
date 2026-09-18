// Compile-time ISA facts and shared constants (prompt §7.1).
//
// Every constant here is backed by a measurement recorded in docs/env_report.md; nothing is
// hardcoded per device model.
#pragma once

#include <cstdint>

namespace sca {
namespace cuda {

// ptxas rejects wgmma on sm_120 (raw diagnostic in docs/env_report.md §2). Keeping the constant
// makes the intent explicit: no code path may select wgmma/tcgen05.
inline constexpr bool kHasWgmma = false;
inline constexpr bool kHasTcgen05 = false;

// Measured cudaDevAttrMaxSharedMemoryPerBlockOptin on this machine (101376 B). Configurations above
// it are rejected at tile-selection time, and the kernels assert it at compile time.
inline constexpr int kMaxDynamicSmemBytes = 101376;

// cp.async and ldmatrix require 16 B alignment; TMA (unused in v1 kernels) requires 128 B.
inline constexpr int kVectorAlignBytes = 16;
inline constexpr int kTmaAlignBytes = 128;

// Rows are padded by 8 halves (16 B) so that (a) every row stays 16 B aligned for ldmatrix and
// (b) the row pitch is no longer a multiple of 128 B, which removes the 8-way bank conflict that a
// D*2 = 256 B pitch would otherwise create.
inline constexpr int kRowPadElems = 8;

}  // namespace cuda
}  // namespace sca

