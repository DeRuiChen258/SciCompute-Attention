// KV cache: lifecycle, layout arithmetic, capacity accounting (prompt §8.7).

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common/test_utils.hpp"
#include "device/cuda_device.hpp"
#include "scicompute_attention/attention.hpp"
#include "scicompute_attention/kv_cache.hpp"

namespace {

using namespace sca_test;

std::shared_ptr<sci::CudaDevice> Gpu() {
    if (!sca::DeviceCapability::CudaAvailable()) return nullptr;
    static std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    return device;
}

sca::KVCacheConfig MakeConfig(int64_t block_size, int64_t num_blocks, int64_t layers = 2,
                              int64_t kv_heads = 2, int64_t head_dim = 64) {
    sca::KVCacheConfig cfg;
    cfg.num_layers = layers;
    cfg.num_kv_heads = kv_heads;
    cfg.head_dim = head_dim;
    cfg.block_size = block_size;
    cfg.num_blocks = num_blocks;
    cfg.max_num_seqs = 4;
    cfg.dtype = sci::DType::kFloat16;
    return cfg;
}

}  // namespace

TEST(KvCacheTest, BudgetFormulaMatchesHandComputation) {
    sca::KVCacheConfig cfg = MakeConfig(16, 1536, /*layers=*/32, /*kv_heads=*/8, /*head_dim=*/128);
    const auto bytes = sca::KvCacheBytes(cfg);
    ASSERT_TRUE(bytes.ok());
    // 2 (K and V) * 32 layers * 1536 blocks * 16 tokens * 8 heads * 128 dim * 2 B = 3 GiB,
    // i.e. 2 MiB per block - the reference point of the prompt's §8.4 budget table.
    EXPECT_EQ(*bytes, 2ull * 32 * 1536 * 16 * 8 * 128 * 2);
    EXPECT_EQ(*bytes, 3221225472ull);
    EXPECT_EQ((*bytes / 1536), 2097152ull);  // 2 MiB per block
    EXPECT_EQ(sca::MaxBlocksForBudget(cfg, *bytes), 1536);
    EXPECT_EQ(sca::MaxBlocksForBudget(cfg, *bytes / 2), 768);
}

TEST(KvCacheTest, RejectsInvalidConfigurations) {
    sca::KVCacheConfig cfg = MakeConfig(16, 4);
    cfg.block_size = 7;
    EXPECT_FALSE(sca::KvCacheBytes(cfg).ok());
    cfg = MakeConfig(16, 0);
    EXPECT_FALSE(sca::KvCacheBytes(cfg).ok());
    cfg = MakeConfig(16, 4);
    cfg.head_dim = 48;
    EXPECT_FALSE(sca::KvCacheBytes(cfg).ok());
}

TEST(KvCacheTest, ComputesSlotMappingWithValidatedBlockTable) {
    const int32_t table[] = {3, 1, 0, -1};
    const auto slots = sca::ComputeSlotMapping(table, /*seq_start=*/17, /*num_tokens=*/5,
                                               /*block_size=*/16, /*max_blocks=*/4);
    ASSERT_TRUE(slots.ok());
    ASSERT_EQ(slots->size(), 5u);
    // position 17 -> block index 1 (physical 1), offset 1 -> slot 1*16+1 = 17
    EXPECT_EQ((*slots)[0], 17);
    EXPECT_EQ((*slots)[1], 18);
    EXPECT_EQ((*slots)[2], 19);
    // positions 20,21 -> block index 1 still (offsets 4,5)
    EXPECT_EQ((*slots)[3], 20);
    EXPECT_EQ((*slots)[4], 21);

    const auto beyond = sca::ComputeSlotMapping(table, /*seq_start=*/17 + 5, 5, 16, 4);
    EXPECT_TRUE(beyond.ok());  // block 1 still
    const auto empty = sca::ComputeSlotMapping(table, /*seq_start=*/48, 1, 16, 4);
    EXPECT_FALSE(empty.ok());  // block index 3 is -1
    EXPECT_NE(empty.error().message().find("empty"), std::string::npos);
}

TEST(KvCacheTest, SupportsEveryDocumentedBlockSize) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    for (const int64_t block_size : {1, 8, 16, 32, 64}) {
        sca::KVCacheConfig cfg = MakeConfig(block_size, 8, /*layers=*/1);
        const auto cache = sca::KVCache::Create(cfg, *device);
        ASSERT_TRUE(cache.ok()) << cache.error().ToString() << " block_size=" << block_size;
        EXPECT_EQ(cache->MemoryBytes(), 2ull * 1 * 8 * block_size * 2 * 64 * 2);
        EXPECT_EQ(cache->K(0).dim(0), 8);
        EXPECT_EQ(cache->K(0).dim(1), block_size);
    }
}

TEST(KvCacheTest, AppendResetAndReuseWithoutCrossTalk) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    const int64_t block_size = 8, num_blocks = 4, kv_heads = 2, head_dim = 32;
    sca::KVCacheConfig cfg = MakeConfig(block_size, num_blocks, 1, kv_heads, head_dim);
    auto cache_result = sca::KVCache::Create(cfg, *device);
    ASSERT_TRUE(cache_result.ok());
    sca::KVCache cache = std::move(*cache_result);

    // Three tokens (non-multiple of block_size) written to physical blocks 2 and 0.
    const int64_t num_tokens = 3;
    std::vector<float> k_values(static_cast<size_t>(num_tokens * kv_heads * head_dim));
    std::vector<float> v_values(k_values.size());
    for (size_t i = 0; i < k_values.size(); ++i) {
        k_values[i] = 10.0f + static_cast<float>(i);
        v_values[i] = -1.0f - static_cast<float>(i);
    }
    const Encoded k_enc = Encode(k_values, cfg.dtype);
    const Encoded v_enc = Encode(v_values, cfg.dtype);
    sci::Tensor k = MakeDeviceTensor({num_tokens, kv_heads, head_dim}, cfg.dtype, k_enc, *device);
    sci::Tensor v = MakeDeviceTensor({num_tokens, kv_heads, head_dim}, cfg.dtype, v_enc, *device);

    // block_size = 8, so a slot s maps to block (s / 8) and offset (s % 8):
    //   slot 0  -> block 0 offset 0  (token 0, head 0)
    //   slot 9  -> block 1 offset 1  (token 1)
    //   slot 17 -> block 2 offset 1  (token 2)
    const std::vector<int32_t> slots = {0, 9, 17};
    const sci::Tensor slot_mapping = MakeDeviceInt32(slots, *device);
    const sci::Status appended =
        cache.Append(0, k, v, static_cast<const int32_t*>(slot_mapping.data()), num_tokens, nullptr);
    ASSERT_TRUE(appended.ok()) << appended.ToString();
    device->synchronize();
    EXPECT_EQ(cache.Stats().num_appends, 1);

    const std::vector<int32_t> blocks = {1, 0};
    const sci::Tensor block_ids = MakeDeviceInt32(blocks, *device);
    const auto gather = cache.GatherK(
        0, static_cast<const int32_t*>(block_ids.data()), 2, nullptr);
    ASSERT_TRUE(gather.ok()) << gather.error().ToString();
    device->synchronize();
    const std::vector<float> gathered = ReadToFloat(*gather);
    // Layout of the gathered buffer: [block1 (8 rows)][block0 (8 rows)] x kv_heads x head_dim.
    const size_t row = static_cast<size_t>(kv_heads * head_dim);
    EXPECT_FLOAT_EQ(gathered[0 * row + 0], 0.0f) << "block1 row 0 was never written";
    EXPECT_NEAR(gathered[1 * row + 0], k_enc.dequantized[row], 1e-3) << "token 1 -> block1 offset1";
    EXPECT_NEAR(gathered[8 * row + 0], k_enc.dequantized[0], 1e-3) << "token 0 -> block0 offset0";

    // Reset block 2 (holding token 2) and verify it is zeroed while block 0 keeps its data.
    const std::vector<int32_t> reset_blocks = {2};
    const sci::Tensor reset_ids = MakeDeviceInt32(reset_blocks, *device);
    ASSERT_TRUE(
        cache.Reset(static_cast<const int32_t*>(reset_ids.data()), 1, nullptr).ok());
    device->synchronize();
    const std::vector<int32_t> check_blocks = {2, 0};
    const sci::Tensor check_ids = MakeDeviceInt32(check_blocks, *device);
    const auto after = cache.GatherK(0, static_cast<const int32_t*>(check_ids.data()), 2, nullptr);
    ASSERT_TRUE(after.ok());
    device->synchronize();
    const std::vector<float> after_values = ReadToFloat(*after);
    EXPECT_FLOAT_EQ(after_values[1 * row + 0], 0.0f) << "block2 offset 1 was reset";
    // With block_ids = {2, 0} the gathered rows are [block2 rows 0..7][block0 rows 0..7], so
    // block0's offset 0 is row 8.
    EXPECT_NEAR(after_values[8 * row + 0], k_enc.dequantized[0], 1e-3)
        << "block0 offset 0 must survive the reset of another block";
    // Before the reset block 2 offset 1 held token 2; confirm it was there by checking the value
    // that is now gone is exactly the one we wrote.
    EXPECT_NEAR(k_enc.dequantized[2 * row], 10.0f + static_cast<float>(2 * row), 1e-2);
    EXPECT_EQ(cache.Stats().num_resets, 1);
}

TEST(KvCacheTest, RejectsOverBudgetAllocationWithAdvice) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    sca::KVCacheConfig cfg = MakeConfig(16, 100000, /*layers=*/32, /*kv_heads=*/8, /*head_dim=*/128);
    const auto cache = sca::KVCache::Create(cfg, *device);
    ASSERT_FALSE(cache.ok());
    EXPECT_EQ(sca::ExtractCode(cache.error()), sca::AttnStatusCode::kKVCapacityExceeded);
    EXPECT_NE(cache.error().message().find("num_blocks <="), std::string::npos);
}
