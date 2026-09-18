#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/status.hpp"
#include "scicompute_attention/export.hpp"

namespace sca {

// Runtime device capability report. Probed once per device and cached (Refresh() forces a
// re-probe, which tests use). Values come from cudaDeviceProp plus the ISA probe results archived
// in docs/env_report.md; nothing here is hardcoded for a specific architecture.
struct DeviceCapability {
    std::string device_name;
    int major{0};
    int minor{0};
    int sm_count{0};
    size_t smem_per_sm{0};
    size_t smem_per_block_default{0};
    size_t smem_per_block_optin{0};
    size_t regs_per_sm{0};
    size_t max_threads_per_sm{0};
    size_t l2_bytes{0};
    size_t total_mem_bytes{0};
    size_t free_mem_bytes{0};
    bool has_mma_m16n8k16{false};
    bool has_cp_async{false};
    bool has_tma_bulk{false};
    bool has_wgmma{false};
    bool has_fp8_mma{false};

    // First call probes and caches; later calls return the cached snapshot without any CUDA call.
    static const DeviceCapability& ForDevice(int device_id = 0);

    // Forces a re-probe and refreshes the cache.
    static sci::Status Refresh(int device_id = 0);

    // Effective shared-memory ceiling per block, honouring SCI_ATTENTION_MAX_SMEM_BYTES when set.
    static size_t SmemLimitPerBlock(const DeviceCapability& cap) noexcept;

    static bool CudaAvailable() noexcept;
};

}  // namespace sca

