// STUB-mode implementation of the device_probe C ABI: reports "no CUDA device".
// Compiled only when SCI_ATTENTION_INFRA_MODE=STUB, so that capability.cpp links without CUDA.

#include "runtime/device_probe.h"

#include <cstring>

extern "C" int sca_probe_device(int device_id, ScaDeviceProbe* out) {
    (void)device_id;
    if (out == nullptr) return 1;
    std::memset(out, 0, sizeof(ScaDeviceProbe));
    std::strncpy(out->name, "stub (no CUDA)", SCA_DEVICE_NAME_MAX - 1);
    out->cuda_available = 0;
    return 1;
}

extern "C" int sca_mem_get_info(unsigned long long* free_bytes, unsigned long long* total_bytes) {
    if (free_bytes != nullptr) *free_bytes = 0;
    if (total_bytes != nullptr) *total_bytes = 0;
    return 1;
}

