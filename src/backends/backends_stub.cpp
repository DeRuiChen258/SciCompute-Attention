// STUB-mode implementations of every kernel launch entry point.
//
// A CPU-only build (INFRA_MODE=STUB) compiles the API/runtime/backend-adapter layer but no device
// code. These stubs keep that layer linkable while guaranteeing that no caller can mistake a STUB
// build for a working GPU build: every launch returns kUnsupportedFeature with an actionable
// message. Unit tests that need kernels skip themselves via DeviceCapability::CudaAvailable().

#include <string>

#include "backends/decode/decode_splitk.cuh"
#include "backends/flash/flash_fwd_kernel.cuh"
#include "backends/naive/attention_naive.cuh"
#include "backends/paged/paged_attention.cuh"
#include "backends/tiled/attention_tiled.cuh"
#include "scicompute_attention/status.hpp"

namespace sca {
namespace cuda {
namespace {

inline sci::Status StubLaunch(const char* what) {
    return MakeStatus(AttnStatusCode::kUnsupportedFeature,
                      std::string("sca::") + what +
                          " requires the CUDA build; this binary was configured with "
                          "-DSCI_ATTENTION_INFRA_MODE=STUB (CPU-only)");
}

}  // namespace

sci::Status LaunchNaiveForward(const NaiveFwdParams&, sci::Stream*) {
    return StubLaunch("naive_attention");
}

int64_t TiledSmemBytes(int32_t, int64_t, int32_t, int32_t) { return -1; }

sci::Status LaunchTiledForward(const TiledFwdParams&, sci::Stream*, int32_t, int32_t, int32_t) {
    return StubLaunch("tiled_attention");
}

sci::Status LaunchFlashFwd(const FlashFwdParams&, sci::Stream*) {
    return StubLaunch("flash_attention");
}

int64_t FlashSmemBytesFor(int32_t, int64_t, int32_t, int32_t, int32_t) { return -1; }

sci::Status LaunchDecodeSplitK(const DecodeFwdParams&, sci::Stream*) {
    return StubLaunch("decode_attention");
}

sci::Status LaunchDecodeSplitKPaged(const DecodeFwdParams&, sci::Stream*) {
    return StubLaunch("decode_attention (paged)");
}

sci::Status LaunchPagedAttention(const PagedDecodeParams&, sci::Stream*) {
    return StubLaunch("paged_attention");
}

}  // namespace cuda
}  // namespace sca

