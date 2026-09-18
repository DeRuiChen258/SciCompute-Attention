// Internal hand-off between src/runtime (dispatcher) and src/backends.
//
// One accessor per backend keeps the backend implementations in their own translation units
// (prompt §6.4: "each backend gets one translation unit") while the dispatcher remains the only
// place that knows how to resolve a BackendKind.
#pragma once

#include "scicompute_attention/backends.hpp"

namespace sca {
namespace runtime {

const IAttentionBackend* NaiveBackendInstance() noexcept;
const IAttentionBackend* TiledBackendInstance() noexcept;
const IAttentionBackend* FlashBackendInstance() noexcept;
const IAttentionBackend* DecodeBackendInstance() noexcept;
const IAttentionBackend* PagedBackendInstance() noexcept;

}  // namespace runtime
}  // namespace sca

