// AUTO-GENERATED FILE - DO NOT EDIT BY HAND.
//
// Producer : tools/dispatch_threshold_sweep.py
// Revision : r0 (seed values from the implementation prompt §9.1)
// Regenerate:
//   python tools/dispatch_threshold_sweep.py --emit-inc src/runtime/dispatch_table.inc
//
// The values below are the seeds used until the Phase 6/8 threshold sweep replaces them with
// measured numbers. tests/integration/test_dispatch_explain.cpp snapshots the resulting
// behaviour, so a table change can never pass unnoticed.

namespace sca {
namespace detail {

struct DispatchThresholds {
    // seq_q <= this value (or seq_q * 8 <= seq_kv) selects the decode kernel.
    int64_t decode_max_seq_q;
    // decode is only considered for sequences at least this long.
    int64_t decode_min_seq_kv;
    // seq_kv <= this value prefers the tiled kernel (short sequences, less register pressure).
    int64_t tiled_max_seq_kv;
    // split-K granularity seed: tokens handled by one split before clamping.
    int64_t tokens_per_split;
    // upper bound for the automatic split count.
    int64_t max_splits;
    // Revision string, surfaced by Explain() so reports can cite the exact table.
    const char* revision;
};

inline constexpr DispatchThresholds kDispatchThresholds{
    /* decode_max_seq_q   */ 1,
    /* decode_min_seq_kv  */ 256,
    /* tiled_max_seq_kv   */ 128,
    /* tokens_per_split   */ 512,
    /* max_splits         */ 16,
    /* revision           */ "r0",
};

}  // namespace detail
}  // namespace sca
