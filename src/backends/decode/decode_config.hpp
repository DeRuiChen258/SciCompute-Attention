// Split-K rules for decode attention (prompt §7.5).
//
// Decode reads the whole KV cache once per step and does almost no arithmetic per byte
// (S_q = 1), so the kernel is HBM bound. Split-K exists to (a) expose enough parallelism to fill
// 36 SMs and (b) keep each partial inside one L2-friendly window.
#pragma once

#include <cstdint>

#include "scicompute_attention/capability.hpp"

namespace sca {
namespace decode {

// Tokens handled by one split before the count is clamped. Calibrated in Phase 6/8; the value also
// lives in src/runtime/dispatch_table.inc so that Explain() reports the same number the kernel uses.
inline constexpr int64_t kTokensPerSplit = 512;
inline constexpr int64_t kMaxSplits = 16;

// kNativeSeqQ: the kernel handles one query token per (batch, head) - the real decode case. Larger
// seq_q is intentionally routed to the flash backend (documented in docs/prefill_decode.md).
inline constexpr int64_t kNativeSeqQ = 1;

inline int64_t SplitsFor(int64_t seq_kv, int64_t sm_count) {
    if (seq_kv <= 0) return 1;
    int64_t splits = (seq_kv + kTokensPerSplit - 1) / kTokensPerSplit;
    const int64_t concurrency_cap = sm_count > 0 ? sm_count * 2 : kMaxSplits;
    if (splits > concurrency_cap) splits = concurrency_cap;
    if (splits > kMaxSplits) splits = kMaxSplits;
    return splits < 1 ? 1 : splits;
}

// Elements of FP32 workspace needed for the partials of one request batch.
//   m/l : [B * H_q * num_splits]
//   O   : [B * H_q * num_splits * D]
inline int64_t PartialElements(int64_t batch, int64_t heads_q, int64_t num_splits,
                              int64_t head_dim) {
    const int64_t base = batch * heads_q * num_splits;
    return base * (2 + head_dim);
}

}  // namespace decode
}  // namespace sca

