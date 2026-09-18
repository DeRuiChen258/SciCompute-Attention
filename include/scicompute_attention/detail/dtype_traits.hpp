#pragma once

#include <cstdint>

#include "core/types.hpp"

namespace sca {
namespace detail {

// Host-side dtype mapping. The CUDA type / MMA type mapping lives in
// src/cuda_common/dtype_cuda.cuh so that this header stays CPU-compilable.
template <sci::DType kDtype>
struct DTypeTraits;

template <>
struct DTypeTraits<sci::DType::kFloat32> {
    using HostType = float;
    static constexpr int64_t kBytes = 4;
    static constexpr bool kSupportsMma = false;  // FP32 is the reference path (naive backend)
};

template <>
struct DTypeTraits<sci::DType::kFloat16> {
    using HostType = uint16_t;  // opaque on the host; converted through CUDA helpers
    static constexpr int64_t kBytes = 2;
    static constexpr bool kSupportsMma = true;
};

template <>
struct DTypeTraits<sci::DType::kBFloat16> {
    using HostType = uint16_t;
    static constexpr int64_t kBytes = 2;
    static constexpr bool kSupportsMma = true;
};

// Tile/stage granularity used by the MMA kernels (m16n8k16 -> K multiples of 16).
inline constexpr int64_t kMmaK = 16;
inline constexpr int64_t kMmaM = 16;
inline constexpr int64_t kMmaN = 8;

// 16 B vectorization granularity for both cp.async and ldmatrix.
inline constexpr int64_t kVectorBytes = 16;

}  // namespace detail
}  // namespace sca

