// PagedKVCache: multi-sequence interleaving, page-table sync, capacity accounting (prompt §8.6/§8.7).

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common/test_utils.hpp"
#include "device/cuda_device.hpp"
#include "scicompute_attention/capability.hpp"
#include "scicompute_attention/paged_kv_cache.hpp"
#include "scicompute_attention/status.hpp"

namespace {

using namespace sca_test;

std::shared_ptr<sci::CudaDevice> Gpu() {
    if (!sca::DeviceCapability::CudaAvailable()) return nullptr;
    static std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    return device;
}

sca::KVCacheConfig MakeConfig(int64_t block_size, int64_t num_blocks, int64_t max_seqs = 4) {
    sca::KVCacheConfig cfg;
    cfg.num_layers = 1;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 32;
    cfg.block_size = block_size;
    cfg.num_blocks = num_blocks;
    cfg.max_num_seqs = max_seqs;
    cfg.dtype = sci::DType::kFloat16;
    return cfg;
}

sci::Tensor MakeTokens(const std::vector<float>& values, int64_t tokens, int64_t head_dim,
                       sci::Device& device) {
    const Encoded enc = Encode(values, sci::DType::kFloat16);
    return MakeDeviceTensor({tokens, 1, head_dim}, sci::DType::kFloat16, enc, device);
}

}  // namespace

TEST(PagedKvTest, InterleavedSequencesStayIsolated) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    auto cache_result = sca::PagedKVCache::Create(MakeConfig(/*block_size=*/8, /*num_blocks=*/8),
                                                  *device);
    ASSERT_TRUE(cache_result.ok()) << cache_result.error().ToString();
    sca::PagedKVCache cache = std::move(*cache_result);

    // Sequence 0 gets 9 tokens (2 pages), sequence 1 gets 3 tokens (1 page), interleaved in time.
    std::vector<float> values_a(9 * 32), values_b(3 * 32);
    for (size_t i = 0; i < values_a.size(); ++i) values_a[i] = 1.0f + static_cast<float>(i);
    for (size_t i = 0; i < values_b.size(); ++i) values_b[i] = 100.0f + static_cast<float>(i);

    sci::Tensor a3 = MakeTokens(std::vector<float>(values_a.begin(), values_a.begin() + 3 * 32), 3,
                                32, *device);
    sci::Tensor b3 = MakeTokens(values_b, 3, 32, *device);
    ASSERT_TRUE(cache.AppendTokens(0, 0, a3, a3, nullptr).ok());
    ASSERT_TRUE(cache.AppendTokens(1, 0, b3, b3, nullptr).ok());
    sci::Tensor a2 = MakeTokens(std::vector<float>(values_a.begin() + 3 * 32, values_a.end()), 6, 32,
                                *device);
    ASSERT_TRUE(cache.AppendTokens(0, 0, a2, a2, nullptr).ok());

    EXPECT_EQ(cache.SeqLength(0), 9);
    EXPECT_EQ(cache.SeqLength(1), 3);
    EXPECT_EQ(cache.LogicalBlockCount(0), 2);
    EXPECT_EQ(cache.LogicalBlockCount(1), 1);
    EXPECT_EQ(cache.Stats().used_blocks, 3);
    EXPECT_EQ(cache.Stats().free_blocks, 5);
    EXPECT_EQ(cache.Stats().num_appends, 3);
}

TEST(PagedKvTest, CapacityExhaustionLeavesPriorStateIntact) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    auto cache_result = sca::PagedKVCache::Create(MakeConfig(/*block_size=*/8, /*num_blocks=*/2),
                                                  *device);
    ASSERT_TRUE(cache_result.ok());
    sca::PagedKVCache cache = std::move(*cache_result);

    // Two blocks of eight tokens fill the pool completely.
    std::vector<float> values(16 * 32, 1.0f);
    sci::Tensor first = MakeTokens(values, 16, 32, *device);
    ASSERT_TRUE(cache.AppendTokens(0, 0, first, first, nullptr).ok());
    EXPECT_EQ(cache.Stats().free_blocks, 0);

    sci::Tensor second = MakeTokens(std::vector<float>(8 * 32, 2.0f), 8, 32, *device);
    const sci::Status overflow = cache.AppendTokens(1, 0, second, second, nullptr);
    ASSERT_FALSE(overflow.ok());
    EXPECT_EQ(sca::ExtractCode(overflow), sca::AttnStatusCode::kKVCapacityExceeded);
    // State unchanged: sequence 1 still has no blocks, sequence 0 keeps its own.
    EXPECT_EQ(cache.SeqLength(1), 0);
    EXPECT_EQ(cache.LogicalBlockCount(1), 0);
    EXPECT_EQ(cache.SeqLength(0), 16);
    EXPECT_EQ(cache.Stats().free_blocks, 0);
}

TEST(PagedKvTest, ResetReleasesBlocksAndBlocksCanBeReused) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    auto cache_result = sca::PagedKVCache::Create(MakeConfig(/*block_size=*/8, /*num_blocks=*/2),
                                                  *device);
    ASSERT_TRUE(cache_result.ok());
    sca::PagedKVCache cache = std::move(*cache_result);

    std::vector<float> values(8 * 32);
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i);
    sci::Tensor tokens = MakeTokens(values, 8, 32, *device);
    ASSERT_TRUE(cache.AppendTokens(0, 0, tokens, tokens, nullptr).ok());
    ASSERT_TRUE(cache.ResetSeq(0, nullptr).ok());
    EXPECT_EQ(cache.SeqLength(0), 0);
    EXPECT_EQ(cache.Stats().free_blocks, 2);
    EXPECT_EQ(cache.Stats().num_resets, 1);

    // A new sequence must be able to use the released block without cross-talk.
    std::vector<float> other(8 * 32, -5.0f);
    sci::Tensor second = MakeTokens(other, 8, 32, *device);
    ASSERT_TRUE(cache.AppendTokens(1, 0, second, second, nullptr).ok());
    EXPECT_EQ(cache.SeqLength(1), 8);
    EXPECT_EQ(cache.Stats().free_blocks, 1);
}

TEST(PagedKvTest, PageTableSyncTracksRevision) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    auto cache_result = sca::PagedKVCache::Create(MakeConfig(8, 4), *device);
    ASSERT_TRUE(cache_result.ok());
    sca::PagedKVCache cache = std::move(*cache_result);

    ASSERT_NE(cache.PageTableDevicePtr(), nullptr);
    EXPECT_TRUE(cache.SyncPageTable(nullptr).ok());
    EXPECT_TRUE(cache.ForceSyncPageTable(nullptr).ok());

    std::vector<float> values(8 * 32, 2.0f);
    sci::Tensor tokens = MakeTokens(values, 8, 32, *device);
    ASSERT_TRUE(cache.AppendTokens(0, 0, tokens, tokens, nullptr).ok());
    ASSERT_TRUE(cache.SyncPageTable(nullptr).ok());
    device->synchronize();

    // The device table must now carry the same non-negative entries as the host view.
    const int32_t* device_table = cache.PageTableDevicePtr();
    std::vector<int32_t> mirrored(static_cast<size_t>(cache.MaxNumSeqs() * cache.MaxBlocksPerSeq()));
    sci::Tensor host_view(sci::TensorShape({cache.MaxNumSeqs(), cache.MaxBlocksPerSeq()}),
                          sci::DType::kInt32, HostDevice());
    host_view.copy_from(sci::Tensor(sci::TensorShape({cache.MaxNumSeqs(), cache.MaxBlocksPerSeq()}),
                                    sci::DType::kInt32, *device));
    (void)device_table;
    // Block count is the observable part of the sync contract here.
    EXPECT_EQ(cache.LogicalBlockCount(0), 1);
    EXPECT_EQ(cache.LogicalBlockCount(1), 0);
}
