#include "scicompute_attention/capability.hpp"

#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "runtime/device_probe.h"

namespace sca {
namespace {

std::mutex& CacheMutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<int, DeviceCapability>& Cache() {
    static std::unordered_map<int, DeviceCapability> cache;
    return cache;
}

DeviceCapability Probe(int device_id) {
    DeviceCapability cap;
    ScaDeviceProbe raw{};
    const int rc = sca_probe_device(device_id, &raw);
    if (rc != 0 || raw.cuda_available == 0) {
        cap.device_name = "cuda-unavailable";
        return cap;
    }
    cap.device_name = raw.name;
    cap.major = raw.major;
    cap.minor = raw.minor;
    cap.sm_count = raw.sm_count;
    cap.smem_per_sm = raw.smem_per_sm;
    cap.smem_per_block_default = raw.smem_per_block_default;
    cap.smem_per_block_optin = raw.smem_per_block_optin;
    cap.regs_per_sm = raw.regs_per_sm;
    cap.max_threads_per_sm = raw.max_threads_per_sm;
    cap.l2_bytes = raw.l2_bytes;
    cap.total_mem_bytes = raw.total_mem_bytes;
    cap.free_mem_bytes = raw.free_mem_bytes;

    const int cc = cap.major * 10 + cap.minor;
    cap.has_mma_m16n8k16 = (cc >= 80) && (raw.binary_has_sm80 != 0);
    cap.has_cp_async = (cc >= 80) && (raw.binary_has_sm80 != 0);
    cap.has_tma_bulk = (cc >= 90) && (raw.binary_has_sm90 != 0);
    // wgmma exists from sm_90 onwards, but this project never emits it: ptxas rejects it on
    // sm_120 (see docs/env_report.md), and the ISA probe asserts the failure. Reported as false so
    // that no code path can select it by accident.
    cap.has_wgmma = false;
    // FP8 MMA is an explicitly out-of-scope experimental path (SCI_ATTENTION_ENABLE_FP8=OFF).
    cap.has_fp8_mma = false;
    return cap;
}

}  // namespace

const DeviceCapability& DeviceCapability::ForDevice(int device_id) {
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& cache = Cache();
    const auto it = cache.find(device_id);
    if (it != cache.end()) return it->second;
    return cache.emplace(device_id, Probe(device_id)).first->second;
}

sci::Status DeviceCapability::Refresh(int device_id) {
    const DeviceCapability fresh = Probe(device_id);
    std::lock_guard<std::mutex> lock(CacheMutex());
    Cache()[device_id] = fresh;
    return sci::Status::Ok();
}

size_t DeviceCapability::SmemLimitPerBlock(const DeviceCapability& cap) noexcept {
    size_t limit = cap.smem_per_block_optin != 0 ? cap.smem_per_block_optin
                                                 : cap.smem_per_block_default;
    if (const char* env = std::getenv("SCI_ATTENTION_MAX_SMEM_BYTES")) {
        const long long parsed = std::strtoll(env, nullptr, 10);
        if (parsed > 0 && static_cast<size_t>(parsed) < limit) {
            limit = static_cast<size_t>(parsed);
        }
    }
    // Never exceed what a single SM can provide.
    if (cap.smem_per_sm != 0 && limit > cap.smem_per_sm) limit = cap.smem_per_sm;
    return limit;
}

bool DeviceCapability::CudaAvailable() noexcept {
    ScaDeviceProbe raw{};
    return sca_probe_device(0, &raw) == 0 && raw.cuda_available != 0;
}

}  // namespace sca

