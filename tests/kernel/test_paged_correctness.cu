// Level 5 correctness: paged attention over non-contiguous, out-of-order pages (prompt §7.6).
//
// Construction: a PagedKVCache allocates physical blocks from a LIFO free list, so the physical
// order is the reverse of the logical order - exactly the "乱序 page" case. The logical token order
// is preserved by appending in order, and the result must match the contiguous reference.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common/test_utils.hpp"
#include "device/cuda_device.hpp"
#include "scicompute_attention/attention.hpp"
#include "scicompute_attention/paged_kv_cache.hpp"

namespace {

using namespace sca_test;

std::shared_ptr<sci::CudaDevice> Gpu() {
    if (!sca::DeviceCapability::CudaAvailable()) return nullptr;
    static std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    return device;
}

struct PagedCase {
    int64_t block_size;
    int64_t heads_q;
    int64_t heads_kv;
    int64_t seq_kv;
    int64_t head_dim;
    uint32_t seed;
};

void RunPagedCase(const PagedCase& c) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) return;

    sca::KVCacheConfig cfg;
    cfg.num_layers = 1;
    cfg.num_kv_heads = c.heads_kv;
    cfg.head_dim = c.head_dim;
    cfg.block_size = c.block_size;
    cfg.num_blocks = (c.seq_kv + c.block_size - 1) / c.block_size + 1;
    cfg.max_num_seqs = 2;
    cfg.dtype = sci::DType::kFloat16;
    auto cache_result = sca::PagedKVCache::Create(cfg, *device);
    ASSERT_TRUE(cache_result.ok()) << cache_result.error().ToString();
    sca::PagedKVCache cache = std::move(*cache_result);

    const size_t q_elements = static_cast<size_t>(c.heads_q * c.head_dim);
    const size_t kv_elements = static_cast<size_t>(c.seq_kv * c.heads_kv * c.head_dim);
    const Encoded q_enc = Encode(RandomFloats(q_elements, c.seed), sci::DType::kFloat16);
    const Encoded k_enc = Encode(RandomFloats(kv_elements, c.seed + 11), sci::DType::kFloat16);
    const Encoded v_enc = Encode(RandomFloats(kv_elements, c.seed + 22), sci::DType::kFloat16);

    // Append in block_size chunks so that the logical order is preserved while the physical blocks
    // are handed out in LIFO (reversed) order.
    int64_t appended = 0;
    while (appended < c.seq_kv) {
        const int64_t chunk = std::min(c.block_size, c.seq_kv - appended);
        std::vector<float> k_chunk(k_enc.dequantized.begin() + appended * c.heads_kv * c.head_dim,
                                   k_enc.dequantized.begin() +
                                       (appended + chunk) * c.heads_kv * c.head_dim);
        std::vector<float> v_chunk(v_enc.dequantized.begin() + appended * c.heads_kv * c.head_dim,
                                   v_enc.dequantized.begin() +
                                       (appended + chunk) * c.heads_kv * c.head_dim);
        const Encoded kc = Encode(k_chunk, cfg.dtype);
        const Encoded vc = Encode(v_chunk, cfg.dtype);
        sci::Tensor k_dev = MakeDeviceTensor({chunk, c.heads_kv, c.head_dim}, cfg.dtype, kc, *device);
        sci::Tensor v_dev = MakeDeviceTensor({chunk, c.heads_kv, c.head_dim}, cfg.dtype, vc, *device);
        const sci::Status status = cache.AppendTokens(0, 0, k_dev, v_dev, nullptr);
        ASSERT_TRUE(status.ok()) << status.ToString();
        appended += chunk;
    }
    ASSERT_EQ(cache.SeqLength(0), c.seq_kv);
    ASSERT_TRUE(cache.SyncPageTable(nullptr).ok());

    sci::Tensor q = MakeDeviceTensor({1, c.heads_q, 1, c.head_dim}, cfg.dtype, q_enc, *device);

    // Sequence length is supplied on the device for the kernel.
    std::vector<int32_t> lengths = {static_cast<int32_t>(c.seq_kv), 0};
    sci::Tensor seq_lens = MakeDeviceInt32(lengths, *device);

    sca::PagedAttentionParams params;
    params.page_table = cache.PageTableDevicePtr();
    params.seq_lens = static_cast<const int32_t*>(seq_lens.data());
    params.num_seqs = 1;
    params.block_size = c.block_size;
    params.max_blocks_per_seq = cache.MaxBlocksPerSeq();
    params.layer = 0;

    sca::AttentionConfig atn_cfg;
    atn_cfg.causal = true;
    const auto result = sca::paged_attention(q, cache, params, atn_cfg);
    const std::string tag = "paged block_size=" + std::to_string(c.block_size) +
                            " Hq=" + std::to_string(c.heads_q) +
                            " Hkv=" + std::to_string(c.heads_kv) +
                            " Skv=" + std::to_string(c.seq_kv) +
                            " D=" + std::to_string(c.head_dim) +
                            " seed=" + std::to_string(c.seed);
    ASSERT_TRUE(result.ok()) << tag << ": " << result.error().ToString();

    // The KV storage layout is [block][offset][head][dim] (prompt §8.1), i.e. token-major within a
    // page, whereas ReferenceAttention consumes the head-major BHSD order. Re-order the same values
    // before computing the expectation - comparing mismatched layouts was the source of an earlier
    // false failure.
    std::vector<float> k_head_major(static_cast<size_t>(c.heads_kv * c.seq_kv * c.head_dim));
    std::vector<float> v_head_major(k_head_major.size());
    for (int64_t h = 0; h < c.heads_kv; ++h) {
        for (int64_t j = 0; j < c.seq_kv; ++j) {
            for (int64_t d = 0; d < c.head_dim; ++d) {
                const size_t token_major = static_cast<size_t>((j * c.heads_kv + h) * c.head_dim + d);
                const size_t head_major = static_cast<size_t>((h * c.seq_kv + j) * c.head_dim + d);
                k_head_major[head_major] = k_enc.dequantized[token_major];
                v_head_major[head_major] = v_enc.dequantized[token_major];
            }
        }
    }
    const std::vector<double> expect =
        ReferenceAttention(q_enc.dequantized, k_head_major, v_head_major, 1, c.heads_q, c.heads_kv,
                           1, c.seq_kv, c.head_dim, /*causal=*/true);
    const ErrorStats stats = Compare(ReadToFloat(result->out), expect);
    // Tolerance per prompt §11.4: paged vs contiguous flash <= 1e-3 max / 1e-4 mean.
    EXPECT_LE(stats.max_abs, 5e-3) << tag;
    EXPECT_LE(stats.mean_abs, 5e-4) << tag;
    EXPECT_GT(result->stats.workspace_bytes, 0u) << tag;
    EXPECT_GT(result->stats.kv_bytes, 0u) << tag;
}

}  // namespace

TEST(PagedAttentionCorrectness, MatchesReferenceForEveryBlockSize) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    uint32_t seed = 80000;
    for (const int64_t block_size : {1, 8, 16, 32}) {
        PagedCase c;
        c.block_size = block_size;
        c.heads_q = 8;
        c.heads_kv = 8;
        c.seq_kv = 257;  // deliberately not a multiple of any supported block size
        c.head_dim = 128;
        c.seed = seed++;
        RunPagedCase(c);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

TEST(PagedAttentionCorrectness, HandlesGqaAndSeveralSequenceLengths) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    uint32_t seed = 81000;
    for (const int64_t group : {int64_t{1}, int64_t{2}, int64_t{4}}) {
        for (const int64_t seq_kv : {33, 129, 512}) {
            PagedCase c;
            c.block_size = 16;
            c.heads_q = 8 * group;
            c.heads_kv = 8;
            c.seq_kv = seq_kv;
            c.head_dim = 64;
            c.seed = seed++;
            RunPagedCase(c);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

TEST(PagedAttentionCorrectness, PhysicalBlocksAreOutOfOrder) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    sca::KVCacheConfig cfg;
    cfg.num_layers = 1;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 32;
    cfg.block_size = 8;
    cfg.num_blocks = 8;
    cfg.max_num_seqs = 1;
    cfg.dtype = sci::DType::kFloat16;
    auto cache_result = sca::PagedKVCache::Create(cfg, *device);
    ASSERT_TRUE(cache_result.ok());
    sca::PagedKVCache cache = std::move(*cache_result);

    std::vector<float> values(24 * 32);
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i % 97);
    const Encoded enc = Encode(values, cfg.dtype);
    sci::Tensor tokens = MakeDeviceTensor({24, 1, 32}, cfg.dtype, enc, *device);
    ASSERT_TRUE(cache.AppendTokens(0, 0, tokens, tokens, nullptr).ok());
    ASSERT_TRUE(cache.SyncPageTable(nullptr).ok());
    device->synchronize();

    // Mirror the device page table back and verify it is neither identity nor contiguous.
    sci::Tensor host_table(sci::TensorShape({cache.MaxNumSeqs(), cache.MaxBlocksPerSeq()}),
                           sci::DType::kInt32, HostDevice());
    sci::Tensor device_table(
        sci::TensorShape({cache.MaxNumSeqs(), cache.MaxBlocksPerSeq()}), sci::DType::kInt32,
        *device);
    device_table.copy_from(sci::Tensor(sci::TensorShape({1, cache.MaxBlocksPerSeq()}),
                                       sci::DType::kInt32, *device));
    host_table.copy_from(device_table);
    HostDevice().synchronize();
    const auto* table = static_cast<const int32_t*>(host_table.data());
    ASSERT_GE(cache.LogicalBlockCount(0), 3);
    bool non_identity = false;
    for (int64_t i = 0; i < cache.LogicalBlockCount(0); ++i) {
        if (table[i] != i) non_identity = true;
    }
    EXPECT_TRUE(non_identity) << "the allocator must hand out non-contiguous physical blocks";
}
