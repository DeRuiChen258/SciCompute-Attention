// Internal launch helper (prompt §9.3). Every kernel launch in this project goes through here:
//   * dynamic shared memory above 48 KiB is enabled once per function, then cached;
//   * launch errors are checked immediately;
//   * SCI_ATTENTION_DEBUG_SYNC=1 adds a stream synchronise + second error check (debug only).
#pragma once

#include <cstddef>
#include <string>

#include "core/status.hpp"

namespace sci {
class Stream;
}  // namespace sci

namespace sca {
namespace runtime {

// Launches `kernel` with an explicit argument array (see cudaLaunchKernel).
// Returns a sci::Status describing the first failing CUDA call.
sci::Status LaunchRaw(const void* kernel, int grid_x, int grid_y, int grid_z, int block_x,
                      int block_y, int block_z, size_t smem_bytes, sci::Stream* stream,
                      void** args);

// Enables dynamic shared memory above the default limit for `kernel`; the largest requested size
// per function is remembered so repeated launches issue no redundant cudaFuncSetAttribute call.
sci::Status EnsureSmemLimit(const void* kernel, size_t smem_bytes);

// Resolves a stream: nullptr means sci::Stream::GetCurrent(); returns the raw handle.
void* ResolveStreamHandle(sci::Stream* stream);

}  // namespace runtime
}  // namespace sca

