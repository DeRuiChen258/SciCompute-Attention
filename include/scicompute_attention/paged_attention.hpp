#pragma once

#include <cstdint>

#include "scicompute_attention/attention_config.hpp"
#include "scicompute_attention/attention_result.hpp"
#include "scicompute_attention/export.hpp"
#include "scicompute_attention/paged_kv_cache.hpp"

namespace sca {

struct PagedAttentionParams {
    const int32_t* page_table{nullptr};  // [num_seqs, max_blocks_per_seq], device-resident
    const int32_t* seq_lens{nullptr};    // [num_seqs], device-resident, includes the new token
    int64_t num_seqs{0};
    int64_t block_size{16};
    int64_t max_blocks_per_seq{0};
    // Which KV layer to attend over. The prompt's struct omits it, but a multi-layer KV cache makes
    // the layer ambiguous (see .agent/decisions.md D-010); 0 keeps single-layer call sites working.
    int64_t layer{0};
};

// Level 5: paged attention over non-contiguous KV pages.
//   q: [num_seqs, H_q, seq_q, D] (BHSD, seq_q == 1 for the decode form)
// When `params` is null the page table owned by `kv` is used (the common serving case).
SCI_ATTENTION_API sci::Result<AttentionResult> paged_attention(
    const sci::Tensor& q, const PagedKVCache& kv, const PagedAttentionParams& params,
    const AttentionConfig& cfg);

SCI_ATTENTION_API sci::Result<AttentionResult> paged_attention(const sci::Tensor& q,
                                                               const PagedKVCache& kv,
                                                               const AttentionConfig& cfg);

}  // namespace sca
