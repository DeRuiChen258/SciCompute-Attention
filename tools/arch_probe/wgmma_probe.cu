// Expected-to-FAIL probe: wgmma is not supported on sm_120 by ptxas.
// `run_probe.sh` compiles this file separately and records the ptxas diagnostic; the failure
// is the *expected* result and is quoted in docs/env_report.md.
//
// This file must never be added to any CMake target.

#include <cuda_runtime.h>

__global__ void probe_wgmma(float* out, const float* in) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.fence.sync.aligned;\n" ::);
    out[threadIdx.x] = in[threadIdx.x];
#else
    out[threadIdx.x] = in[threadIdx.x];
#endif
}

int main() {
    probe_wgmma<<<1, 32>>>(nullptr, nullptr);
    return 0;
}

