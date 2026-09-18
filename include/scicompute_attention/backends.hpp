#pragma once

#include <cstddef>
#include <string>

#include "core/status.hpp"
#include "tensor/tensor.hpp"

#include "scicompute_attention/attention_config.hpp"
#include "scicompute_attention/attention_result.hpp"
#include "scicompute_attention/attention_types.hpp"
#include "scicompute_attention/detail/tile_config.hpp"
#include "scicompute_attention/export.hpp"

namespace sca {

// Capability answer of a backend for one (config, shape) pair. `reason` must always explain "why"
// with concrete numbers when supported == false (prompt §6.4).
struct CapabilityReport {
    bool supported{false};
    std::string reason;
    BackendKind recommended_backend{BackendKind::kAuto};
    TileConfig tile{};
    size_t workspace_bytes{0};
};

class SCI_ATTENTION_API IAttentionBackend {
public:
    virtual ~IAttentionBackend() = default;
    virtual BackendKind Kind() const noexcept = 0;
    virtual const char* Name() const noexcept = 0;
    virtual CapabilityReport Supports(const AttentionConfig& cfg,
                                      const AttentionShape& shape) const = 0;
    virtual size_t WorkspaceBytes(const AttentionConfig& cfg,
                                  const AttentionShape& shape) const = 0;
    virtual sci::Result<AttentionResult> Forward(const sci::Tensor& q, const sci::Tensor& k,
                                                 const sci::Tensor& v,
                                                 const AttentionConfig& cfg) const = 0;
    virtual sci::Result<AttentionResult> ForwardVarlen(
        const sci::Tensor& q, const sci::Tensor& k, const sci::Tensor& v,
        int64_t num_seqs, const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_kv,
        int64_t max_seq_q, int64_t max_seq_kv, const AttentionConfig& cfg) const = 0;
};

// Backend lookup. Never returns nullptr: unknown kinds resolve to the naive backend, and
// unsupported requests surface through Supports()/Forward() instead of a null pointer.
SCI_ATTENTION_API const IAttentionBackend* GetBackend(BackendKind kind) noexcept;

// Stable backend names ("naive", "tiled", "flash", "decode", "paged", "auto").
SCI_ATTENTION_API const char* BackendTypeName(BackendKind kind) noexcept;

// Parses a backend name; returns false and leaves *out untouched when the name is unknown.
SCI_ATTENTION_API bool ParseBackendName(const char* name, BackendKind* out) noexcept;

}  // namespace sca
