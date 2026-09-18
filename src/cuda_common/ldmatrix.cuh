// ldmatrix wrappers (prompt §7.1).
//
// Semantics (verified at runtime by tools/arch_probe/arch_probe.cu, probe [3]):
//   non-trans: register m holds matrix m; lane L receives matrix[g][2c] and matrix[g][2c+1]
//   trans    : register m holds matrix m^T; lane L receives matrix[2c][g] and matrix[2c+1][g]
//   Address supply: lanes 0-7 give matrix 0's 8 row addresses, lanes 8-15 matrix 1, etc.
//   Each address must be 16 B aligned (8 halves); the row padding used by this project
//   (kQSplit = 8 halves) keeps every row 16 B aligned while removing the 128 B stride aliasing.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sca {
namespace cuda {

__device__ __forceinline__ uint32_t SmemAddress(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void LdMatrixX4(uint32_t (&regs)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(address));
}

__device__ __forceinline__ void LdMatrixX4Trans(uint32_t (&regs)[4], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(address));
}

__device__ __forceinline__ void LdMatrixX2(uint32_t (&regs)[2], uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(regs[0]), "=r"(regs[1])
                 : "r"(address));
}

// Unpacks the two elements of a 32-bit register into floats.
template <typename T>
__device__ __forceinline__ void UnpackPair(uint32_t packed, float* lo, float* hi);

template <>
__device__ __forceinline__ void UnpackPair<__half>(uint32_t packed, float* lo, float* hi) {
    const __half2 pair = *reinterpret_cast<const __half2*>(&packed);
    *lo = __half2float(__low2half(pair));
    *hi = __half2float(__high2half(pair));
}

template <>
__device__ __forceinline__ void UnpackPair<__nv_bfloat16>(uint32_t packed, float* lo, float* hi) {
    const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&packed);
    *lo = __bfloat162float(__low2bfloat16(pair));
    *hi = __bfloat162float(__high2bfloat16(pair));
}

}  // namespace cuda
}  // namespace sca
