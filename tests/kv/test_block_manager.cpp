// BlockManager: O(1) allocation, exhaustion, double free, multi-sequence isolation (prompt §8.5).

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "device/cuda_device.hpp"
#include "scicompute_attention/block_manager.hpp"
#include "scicompute_attention/capability.hpp"
#include "scicompute_attention/status.hpp"

namespace {

std::shared_ptr<sci::CudaDevice> Gpu() {
    if (!sca::DeviceCapability::CudaAvailable()) return nullptr;
    static std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    return device;
}

}  // namespace

TEST(BlockManagerTest, AllocatesAndReleasesEveryBlockExactlyOnce) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    auto manager_result = sca::BlockManager::Create(/*num_blocks=*/16, /*max_num_seqs=*/2,
                                                    /*max_blocks_per_seq=*/8, *device);
    ASSERT_TRUE(manager_result.ok()) << manager_result.error().ToString();
    sca::BlockManager manager = std::move(*manager_result);
    EXPECT_EQ(manager.NumFreeBlocks(), 16);

    const auto first = manager.Allocate(5);
    ASSERT_TRUE(first.ok()) << first.error().ToString();
    ASSERT_EQ(first->size(), 5u);
    EXPECT_EQ(manager.NumFreeBlocks(), 11);
    // Every id is unique and inside the pool.
    std::vector<int32_t> sorted = *first;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end());
    EXPECT_GE(sorted.front(), 0);
    EXPECT_LT(sorted.back(), 16);

    manager.Free(first->data(), 5);
    EXPECT_EQ(manager.NumFreeBlocks(), 16);
}

TEST(BlockManagerTest, ExhaustionIsReportedAndStateIsUnchanged) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    auto manager_result = sca::BlockManager::Create(8, 2, 8, *device);
    ASSERT_TRUE(manager_result.ok());
    sca::BlockManager manager = std::move(*manager_result);

    ASSERT_TRUE(manager.Allocate(8).ok());
    const auto overflow = manager.Allocate(1);
    ASSERT_FALSE(overflow.ok());
    EXPECT_EQ(sca::ExtractCode(overflow.error()), sca::AttnStatusCode::kKVCapacityExceeded);
    EXPECT_EQ(manager.NumFreeBlocks(), 0);
    EXPECT_EQ(manager.AllocationFailures(), 1);
    EXPECT_NE(overflow.error().message().find("only 0 of 8"), std::string::npos);
}

TEST(BlockManagerTest, DoubleFreeIsCountedAndDoesNotCorruptThePool) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    auto manager_result = sca::BlockManager::Create(4, 1, 4, *device);
    ASSERT_TRUE(manager_result.ok());
    sca::BlockManager manager = std::move(*manager_result);
    const auto blocks = manager.Allocate(2);
    ASSERT_TRUE(blocks.ok());
    manager.Free(blocks->data(), 2);
    manager.Free(blocks->data(), 2);  // second release of the same ids
    EXPECT_EQ(manager.NumFreeBlocks(), 4);  // pool not duplicated
    EXPECT_EQ(manager.DoubleFreeEvents(), 2);
}

TEST(BlockManagerTest, SequencesAreIsolatedAndTableSyncsByRevision) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    auto manager_result = sca::BlockManager::Create(16, 3, 4, *device);
    ASSERT_TRUE(manager_result.ok());
    sca::BlockManager manager = std::move(*manager_result);

    const auto a = manager.Allocate(2);
    const auto b = manager.Allocate(2);
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    ASSERT_TRUE(manager.AppendToTable(0, a->data(), 2).ok());
    ASSERT_TRUE(manager.AppendToTable(1, b->data(), 2).ok());

    const sca::BlockTable& table = manager.Table();
    EXPECT_EQ(table.host[0], (*a)[0]);
    EXPECT_EQ(table.host[1], (*a)[1]);
    EXPECT_EQ(table.host[2], -1);
    EXPECT_EQ(table.host[4], (*b)[0]);  // row length is max_blocks_per_seq = 4

    const uint64_t revision_before = table.revision;
    ASSERT_TRUE(manager.ResetSeq(0).ok());
    EXPECT_EQ(table.host[0], -1);
    EXPECT_GT(table.revision, revision_before);
    EXPECT_EQ(manager.NumFreeBlocks(), 14);  // sequence 0's two blocks came back

    ASSERT_TRUE(manager.SyncTableToDevice(nullptr).ok());
    device->synchronize();
    ASSERT_TRUE(manager.SyncTableToDevice(nullptr).ok());  // idempotent when revision is unchanged
}

TEST(BlockManagerTest, RejectsEmptySequenceRequests) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    auto manager_result = sca::BlockManager::Create(4, 1, 4, *device);
    ASSERT_TRUE(manager_result.ok());
    sca::BlockManager manager = std::move(*manager_result);
    EXPECT_FALSE(manager.Allocate(0).ok());
    EXPECT_FALSE(manager.AppendToTable(0, nullptr, 1).ok());
    EXPECT_FALSE(manager.AppendToTable(7, nullptr, 1).ok());
}
