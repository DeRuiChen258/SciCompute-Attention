// STUB-mode implementation of the KV data-movement entry points.
//
// A CPU-only build has no device kernels, so every append/reset/gather reports
// kUnsupportedFeature with an explicit message instead of silently doing nothing. This keeps the
// host layer linkable (unit tests build and run) while making it impossible to mistake a STUB build
// for a working GPU build.

#include "kv_cache/kv_cache_kernels.cuh"

#include <string>

#include "scicompute_attention/status.hpp"

namespace sca {
namespace cuda {
namespace {

inline sci::Status StubStatus(const char* what) {
    return MakeStatus(AttnStatusCode::kUnsupportedFeature,
                      std::string(what) +
                          " is unavailable in a CPU-only (INFRA_MODE=STUB) build; "
                          "rebuild with -DSCI_ATTENTION_INFRA_MODE=SOURCE");
}

}  // namespace

sci::Status LaunchKvAppend(void*, void*, const void*, const void*, const int32_t*, int64_t,
                           const KVCacheShape&, int64_t, sci::Stream*) {
    return StubStatus("KVCache::Append");
}

sci::Status LaunchKvResetBlock(void*, void*, const int32_t*, int64_t, const KVCacheShape&, int64_t,
                               sci::Stream*) {
    return StubStatus("KVCache::Reset");
}

sci::Status LaunchKvGather(const void*, const int32_t*, int64_t, void*, const KVCacheShape&,
                           int64_t, sci::Stream*) {
    return StubStatus("KVCache::Gather");
}

}  // namespace cuda
}  // namespace sca

