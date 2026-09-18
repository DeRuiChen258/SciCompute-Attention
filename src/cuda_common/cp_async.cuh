// cp.async helpers (prompt §7.1).
//
// Policy: 16 B (cp.async.cg) for the bulk of a tile, 4 B fallbacks for the tail. The pipeline in
// flash_fwd_kernel.cuh commits one group per K/V tile and waits with
// `wait_group(stages - 1)`, which keeps the buffer being computed on already complete.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_common/ldmatrix.cuh"

namespace sca {
namespace cuda {

__device__ __forceinline__ void CpAsync16(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(SmemAddress(smem_dst)),
                 "l"(gmem_src));
}

__device__ __forceinline__ void CpAsync8(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(SmemAddress(smem_dst)),
                 "l"(gmem_src));
}

__device__ __forceinline__ void CpAsync4(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(SmemAddress(smem_dst)),
                 "l"(gmem_src));
}

__device__ __forceinline__ void CpAsyncCommit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int kPending>
__device__ __forceinline__ void CpAsyncWaitGroup() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void CpAsyncWaitAll() { asm volatile("cp.async.wait_all;\n" ::); }

}  // namespace cuda
}  // namespace sca
