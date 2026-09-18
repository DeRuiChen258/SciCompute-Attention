// mma.sync m16n8k16 wrappers (prompt §7.1).
//
// Why this instruction and not wgmma/tcgen05:
//   * wgmma is rejected by ptxas on sm_120 (verified: tools/arch_probe/wgmma_probe.cu);
//   * tcgen05 only exists on sm_100/sm_103.
//   The fragment layout below is the one verified at runtime by tools/arch_probe/arch_probe.cu
//   (identity-GEMM check C == B over all 32 lanes), so the mapping is measured, not remembered.
//
// A fragment (16x16, 4 x b32 = 8 halves):
//   a0,a1 = A[g][2c+0..1]    a2,a3 = A[g+8][2c+0..1]
//   a4,a5 = A[g][2c+8..9]    a6,a7 = A[g+8][2c+8..9]
// B fragment (16x8, 2 x b32 = 4 halves):
//   b0,b1 = B[2c+0..1][g]    b2,b3 = B[2c+8..9][g]
// C fragment (16x8, 4 x f32):
//   c0 = C[g][2c]  c1 = C[g][2c+1]  c2 = C[g+8][2c]  c3 = C[g+8][2c+1]
// with g = lane >> 2, c = lane & 3.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sca {
namespace cuda {

// Packs two floats into one 32-bit register holding 2 elements of T.
template <typename T>
__device__ __forceinline__ uint32_t PackPair(float lo, float hi);

template <>
__device__ __forceinline__ uint32_t PackPair<__half>(float lo, float hi) {
    const __half2 pair = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
    return *reinterpret_cast<const uint32_t*>(&pair);
}

template <>
__device__ __forceinline__ uint32_t PackPair<__nv_bfloat16>(float lo, float hi) {
    const __nv_bfloat162 pair =
        __halves2bfloat162(__float2bfloat16(lo), __float2bfloat16(hi));
    return *reinterpret_cast<const uint32_t*>(&pair);
}

// D = A * B + C for m16n8k16 with FP32 accumulation.
template <typename T>
__device__ __forceinline__ void MmaM16N8K16(float c[4], const uint32_t a[4], uint32_t b0,
                                            uint32_t b1);

template <>
__device__ __forceinline__ void MmaM16N8K16<__half>(float c[4], const uint32_t a[4],
                                                    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <>
__device__ __forceinline__ void MmaM16N8K16<__nv_bfloat16>(float c[4], const uint32_t a[4],
                                                           uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Lane decomposition used by every fragment mapping in this project.
__device__ __forceinline__ int LaneGroup(int lane) { return lane >> 2; }
__device__ __forceinline__ int LaneCol(int lane) { return lane & 3; }

}  // namespace cuda
}  // namespace sca
