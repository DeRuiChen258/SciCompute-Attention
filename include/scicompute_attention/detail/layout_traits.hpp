#pragma once

#include <cstdint>

#include "scicompute_attention/attention_types.hpp"
#include "scicompute_attention/detail/host_utils.hpp"

namespace sca {
namespace detail {

// Single source of truth for layout arithmetic (DRY). Every kernel and every host-side helper
// obtains its strides from here; no other translation unit may re-derive them.
struct Strides {
    int64_t s_b{0};   // batch stride
    int64_t s_h{0};   // head stride
    int64_t s_s{0};   // sequence stride
    int64_t s_d{0};   // head-dim stride (always 1 for supported layouts)
};

// For kBHSD: [B, H, S, D]  -> s_b = H*S*D, s_h = S*D, s_s = D, s_d = 1
// For kBSHD: [B, S, H, D]  -> s_b = S*H*D, s_s = H*D, s_h = D, s_d = 1
inline Strides StridesFor(AttnLayout layout, int64_t seq, int64_t heads, int64_t head_dim) {
    Strides s{};
    s.s_d = 1;
    if (layout == AttnLayout::kBHSD) {
        s.s_s = head_dim;
        s.s_h = seq * head_dim;
        s.s_b = heads * s.s_h;
    } else {
        s.s_h = head_dim;
        s.s_s = heads * head_dim;
        s.s_b = seq * s.s_s;
    }
    return s;
}

inline int64_t Offset(const Strides& s, int64_t b, int64_t h, int64_t t, int64_t d) noexcept {
    return b * s.s_b + h * s.s_h + t * s.s_s + d * s.s_d;
}

inline int64_t OffsetBHSD(int64_t b, int64_t h, int64_t t, int64_t d, int64_t seq,
                          int64_t heads, int64_t head_dim) noexcept {
    return ((b * heads + h) * seq + t) * head_dim + d;
}

inline int64_t OffsetBSHD(int64_t b, int64_t t, int64_t h, int64_t d, int64_t seq,
                          int64_t heads, int64_t head_dim) noexcept {
    return ((b * seq + t) * heads + h) * head_dim + d;
}

// Total element count of Q/K/V for the given layout (they all share B/S/H/D semantics; K/V use
// num_kv_heads instead of num_heads).
inline bool LayoutSupported(AttnLayout layout) noexcept {
    return layout == AttnLayout::kBHSD || layout == AttnLayout::kBSHD;
}

}  // namespace detail
}  // namespace sca

