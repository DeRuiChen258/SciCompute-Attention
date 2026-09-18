#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "core/status.hpp"

namespace sca {
namespace detail {

// Integer helpers shared by host-side code. Overflow is detected, never ignored.

template <typename T>
inline constexpr T CeilDiv(T value, T divisor) noexcept {
    return (value + divisor - 1) / divisor;
}

template <typename T>
inline constexpr bool IsAligned(T value, T alignment) noexcept {
    return alignment != 0 && (value % alignment) == 0;
}

template <typename T>
inline constexpr T AlignUp(T value, T alignment) noexcept {
    return CeilDiv(value, alignment) * alignment;
}

template <typename T>
inline constexpr T Min(T a, T b) noexcept {
    return a < b ? a : b;
}

template <typename T>
inline constexpr T Max(T a, T b) noexcept {
    return a > b ? a : b;
}

// Returns false (and the caller reports an error) when the product does not fit in int64_t.
inline bool CheckedMul(int64_t a, int64_t b, int64_t* out) noexcept {
    if (a < 0 || b < 0) return false;
    if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

inline bool CheckedMulSize(size_t a, size_t b, size_t* out) noexcept {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

}  // namespace detail
}  // namespace sca

