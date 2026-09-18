// Device capability probing and caching (prompt §9.2).

#include <gtest/gtest.h>

#include <cstdlib>

#include "scicompute_attention/capability.hpp"

TEST(DeviceCapabilityTest, ProbeIsConsistentWithTheEnvironment) {
    const sca::DeviceCapability& cap = sca::DeviceCapability::ForDevice(0);
    if (!sca::DeviceCapability::CudaAvailable()) {
        GTEST_SKIP() << "no CUDA device in this build/environment (infra mode STUB?)";
    }
    EXPECT_FALSE(cap.device_name.empty());
    EXPECT_GE(cap.sm_count, 1);
    EXPECT_GT(cap.smem_per_sm, 0u);
    EXPECT_GT(cap.regs_per_sm, 0u);
    EXPECT_GT(cap.max_threads_per_sm, 0u);
    EXPECT_GT(cap.total_mem_bytes, 0u);
    EXPECT_GT(cap.major, 0);
}

TEST(DeviceCapabilityTest, MmaAndCpAsyncFollowComputeCapability) {
    if (!sca::DeviceCapability::CudaAvailable()) {
        GTEST_SKIP() << "no CUDA device";
    }
    const sca::DeviceCapability& cap = sca::DeviceCapability::ForDevice(0);
    const int cc = cap.major * 10 + cap.minor;
    if (cc >= 80) {
        EXPECT_TRUE(cap.has_mma_m16n8k16);
        EXPECT_TRUE(cap.has_cp_async);
    }
    // wgmma is never advertised: the project forbids it (ptxas rejects it on sm_120).
    EXPECT_FALSE(cap.has_wgmma);
    EXPECT_FALSE(cap.has_fp8_mma);
    EXPECT_EQ(cap.has_tma_bulk, cc >= 90);
}

TEST(DeviceCapabilityTest, RefreshIsIdempotentAndCachedAccessIsStable) {
    if (!sca::DeviceCapability::CudaAvailable()) {
        GTEST_SKIP() << "no CUDA device";
    }
    const sca::DeviceCapability before = sca::DeviceCapability::ForDevice(0);
    EXPECT_TRUE(sca::DeviceCapability::Refresh(0).ok());
    const sca::DeviceCapability after = sca::DeviceCapability::ForDevice(0);
    EXPECT_EQ(before.device_name, after.device_name);
    EXPECT_EQ(before.sm_count, after.sm_count);
    EXPECT_EQ(before.smem_per_block_optin, after.smem_per_block_optin);
}

TEST(DeviceCapabilityTest, SmemLimitHonoursEnvironmentOverride) {
    if (!sca::DeviceCapability::CudaAvailable()) {
        GTEST_SKIP() << "no CUDA device";
    }
    const sca::DeviceCapability cap = sca::DeviceCapability::ForDevice(0);
    const size_t natural = sca::DeviceCapability::SmemLimitPerBlock(cap);
    EXPECT_GT(natural, 0u);
    EXPECT_LE(natural, cap.smem_per_sm);

    setenv("SCI_ATTENTION_MAX_SMEM_BYTES", "49152", 1);
    const size_t capped = sca::DeviceCapability::SmemLimitPerBlock(cap);
    EXPECT_EQ(capped, 49152u);
    unsetenv("SCI_ATTENTION_MAX_SMEM_BYTES");
    EXPECT_EQ(sca::DeviceCapability::SmemLimitPerBlock(cap), natural);
}

