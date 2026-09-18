// Numeric helpers shared by every attention kernel (prompt §7.1).
//
// Policy: the softmax core uses exp2f (exp(x) = exp2(x * log2e)) because scaling by log2(e) can be
// folded into the scale constant, saving one multiply per element. Accumulators are always FP32.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math_constants.h>

namespace sca {
namespace cuda {

inline constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ float FastExp2(float x) { return exp2f(x); }

// exp(x) expressed through exp2f (see docs/online_softmax.md).
__device__ __forceinline__ float FastExp(float x) { return exp2f(x * kLog2e); }

// -inf sentinel used for masked logits. Using -CUDART_INF_F (not a large negative constant) keeps
// exp(-inf - m) exactly 0, which is what makes fully masked rows well defined.
__device__ __forceinline__ float NegInf() { return -CUDART_INF_F; }

__device__ __forceinline__ bool IsFinite(float x) { return isfinite(x); }

// Cast helpers: float -> T and T -> float for the three compute dtypes.
__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float FromFloat(float v) { return v; }
__device__ __forceinline__ __half FromFloatH(float v) { return __float2half_rn(v); }
__device__ __forceinline__ __nv_bfloat16 FromFloatB(float v) { return __float2bfloat16(v); }

template <typename T>
__device__ __forceinline__ T FromFloatT(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ float FromFloatT<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ __half FromFloatT<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 FromFloatT<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// Block-wide max/sum over one value per thread.
//
// Layout: one entry per warp in `scratch` (needs >= 32 floats). The second stage reduces those
// entries in warp 0 and *broadcasts* the result through scratch[0], because every thread needs the
// block-wide value (a reduction that leaves other warps with their partial value silently breaks
// the softmax caller - it was the root cause of a "row reduced to all zeros" bug).
__device__ __forceinline__ float BlockReduceMax(float value, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        // Fully qualified: including <cuda_fp16.h>/<cuda_bf16.h> first makes unqualified lookup
        // ambiguous with the half/bfloat16 overloads.
        value = fmaxf(value, ::__shfl_xor_sync(0xffffffffu, value, offset));
    }
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        const int num_warps = static_cast<int>((blockDim.x + 31) >> 5);
        float block_value = lane < num_warps ? scratch[lane] : -CUDART_INF_F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_value =
                fmaxf(block_value, ::__shfl_xor_sync(0xffffffffu, block_value, offset));
        }
        if (lane == 0) scratch[0] = block_value;
    }
    __syncthreads();
    const float result = scratch[0];
    // Leave the scratch buffer free of in-flight readers before the caller reuses it.
    __syncthreads();
    return result;
}

__device__ __forceinline__ float BlockReduceSum(float value, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += ::__shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        const int num_warps = static_cast<int>((blockDim.x + 31) >> 5);
        float block_value = lane < num_warps ? scratch[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_value += ::__shfl_xor_sync(0xffffffffu, block_value, offset);
        }
        if (lane == 0) scratch[0] = block_value;
    }
    __syncthreads();
    const float result = scratch[0];
    __syncthreads();
    return result;
}

}  // namespace cuda
}  // namespace sca
