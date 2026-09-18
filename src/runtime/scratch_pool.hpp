// Process-wide device scratch pool.
//
// Why this exists: the public API hands back a freshly allocated output tensor (by contract), but
// the *intermediate* buffers (score matrix, split-K partials, ...) must not be allocated per call -
// prompt §9.5 forbids cudaMalloc/cudaFree on the inference path.
//
// Convention: one buffer per (device, kind). Backends acquire their scratch at the start of a
// forward call and release it implicitly when the next call takes it. This assumes the documented
// single-threaded, non-overlapping call convention (see docs/kernel_design.md); a caller that wants
// concurrency must pass its own workspace (future Workspace-based API).
#pragma once

#include <cstddef>

#include "core/status.hpp"
#include "tensor/tensor.hpp"

namespace sci {
class Device;
}  // namespace sci

namespace sca {
namespace runtime {

enum class ScratchKind : int {
    kScoresFp32 = 0,     // naive: [B, H, S_q, S_kv] FP32
    kDecodePartials = 1, // decode split-K (m, l, O) partials
    kGeneric = 2,
    kCount = 3,
};

// Returns a device-resident FP32 buffer with at least `elements` elements. The returned pointer
// stays valid until the next AcquireScratch() call with the same (device, kind).
sci::Result<float*> AcquireScratchFp32(ScratchKind kind, size_t elements, sci::Device& device);

// Releases every cached buffer (tests and memory-pressure paths).
sci::Status ReleaseScratch();

// High-water mark in bytes across all cached buffers (reported in benchmark/JSON output).
size_t ScratchHighWaterMarkBytes();

}  // namespace runtime
}  // namespace sca

