#pragma once

// Umbrella header. Upper layers (vLLM(C++), RLHF, Python bindings) include this file — or one of
// the specific sub-headers below — and nothing else from this project.
//
// namespace layout:
//   sca::          public API (this directory)
//   sca::detail::  header-only helpers that are exported because the public API mentions their
//                  types (TileConfig) or because upper layers need the same arithmetic
//   sca::cuda::    device-side implementation, never included by upper layers

#include "scicompute_attention/attention_config.hpp"
#include "scicompute_attention/attention_result.hpp"
#include "scicompute_attention/attention_types.hpp"
#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/block_manager.hpp"
#include "scicompute_attention/capability.hpp"
#include "scicompute_attention/decode_attention.hpp"
#include "scicompute_attention/dispatcher.hpp"
#include "scicompute_attention/export.hpp"
#include "scicompute_attention/flash_attention.hpp"
#include "scicompute_attention/kv_cache.hpp"
#include "scicompute_attention/naive_attention.hpp"
#include "scicompute_attention/paged_attention.hpp"
#include "scicompute_attention/paged_kv_cache.hpp"
#include "scicompute_attention/status.hpp"
#include "scicompute_attention/tiled_attention.hpp"
#include "scicompute_attention/version.hpp"
#include "scicompute_attention/workspace.hpp"

namespace sca {

// Highest-level entry points. They validate, dispatch and execute; no exceptions escape.
SCI_ATTENTION_API sci::Result<AttentionResult> attention(const sci::Tensor& q, const sci::Tensor& k,
                                                         const sci::Tensor& v,
                                                         const AttentionConfig& cfg = {});

inline sci::Result<AttentionResult> flash(const sci::Tensor& q, const sci::Tensor& k,
                                          const sci::Tensor& v, const AttentionConfig& cfg = {}) {
    AttentionConfig c = cfg;
    c.backend = BackendKind::kFlash;
    return attention(q, k, v, c);
}

SCI_ATTENTION_API sci::Result<AttentionResult> decode(const sci::Tensor& q, const sci::Tensor& k,
                                                      const sci::Tensor& v,
                                                      const AttentionConfig& cfg = {});

SCI_ATTENTION_API sci::Result<AttentionResult> paged(const sci::Tensor& q,
                                                     const PagedKVCache& kv,
                                                     const AttentionConfig& cfg = {});

// Dispatcher-inspection helpers for debugging and tests.
SCI_ATTENTION_API std::string ExplainAttention(const sci::Tensor& q, const sci::Tensor& k,
                                               const sci::Tensor& v,
                                               const AttentionConfig& cfg = {});

}  // namespace sca

