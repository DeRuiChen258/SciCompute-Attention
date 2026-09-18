// Level 4 (decode) correctness: split-K merge over num_splits in {1,2,3,4,8,16} (prompt §7.5).
//
// Decode is the S_q = 1 case: the causal mask is bottom-right aligned, so every cached token is
// visible and the reference reduces to a plain softmax over the KV axis.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common/test_utils.hpp"
#include "device/cuda_device.hpp"
#include "scicompute_attention/attention.hpp"

namespace {

using namespace sca_test;

std::shared_ptr<sci::CudaDevice> Gpu() {
    if (!sca::DeviceCapability::CudaAvailable()) return nullptr;
    static std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    return device;
}

struct DecodeCase {
    sci::DType dtype;
    int64_t heads_q;
    int64_t heads_kv;
    int64_t seq_kv;
    int64_t head_dim;
    int64_t num_splits;
    uint32_t seed;
};

void RunDecodeCase(const DecodeCase& c) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) return;
    const int64_t seq_q = 1;
    const size_t q_elements = static_cast<size_t>(c.heads_q * seq_q * c.head_dim);
    const size_t kv_elements = static_cast<size_t>(c.heads_kv * c.seq_kv * c.head_dim);
    const Encoded q_enc = Encode(RandomFloats(q_elements, c.seed), c.dtype);
    const Encoded k_enc = Encode(RandomFloats(kv_elements, c.seed + 101), c.dtype);
    const Encoded v_enc = Encode(RandomFloats(kv_elements, c.seed + 202), c.dtype);

    sci::Tensor q = MakeDeviceTensor({1, c.heads_q, seq_q, c.head_dim}, c.dtype, q_enc, *device);
    sci::Tensor k = MakeDeviceTensor({1, c.heads_kv, c.seq_kv, c.head_dim}, c.dtype, k_enc, *device);
    sci::Tensor v = MakeDeviceTensor({1, c.heads_kv, c.seq_kv, c.head_dim}, c.dtype, v_enc, *device);

    sca::AttentionConfig cfg;
    cfg.backend = sca::BackendKind::kDecode;
    cfg.causal = true;
    cfg.num_splits = c.num_splits;
    const auto result = sca::decode_attention(q, k, v, cfg);
    const std::string tag = "decode dtype=" + std::string(sci::kDTypeName(c.dtype)) +
                            " Hq=" + std::to_string(c.heads_q) + " Hkv=" + std::to_string(c.heads_kv) +
                            " Skv=" + std::to_string(c.seq_kv) + " D=" + std::to_string(c.head_dim) +
                            " splits=" + std::to_string(c.num_splits) +
                            " seed=" + std::to_string(c.seed);
    ASSERT_TRUE(result.ok()) << tag << ": " << result.error().ToString();

    const std::vector<double> expect =
        ReferenceAttention(q_enc.dequantized, k_enc.dequantized, v_enc.dequantized, 1, c.heads_q,
                           c.heads_kv, seq_q, c.seq_kv, c.head_dim, /*causal=*/true);
    const ErrorStats stats = Compare(ReadToFloat(result->out), expect);
    const Tolerance tol = ToleranceFor(c.dtype, false);
    EXPECT_LE(stats.max_abs, tol.max_abs) << tag;
    EXPECT_LE(stats.mean_abs, tol.mean_abs) << tag;
    EXPECT_EQ(result->stats.num_splits, c.num_splits) << tag;
    EXPECT_GT(result->stats.workspace_bytes, 0u) << tag;
}

}  // namespace

TEST(DecodeAttentionCorrectness, EverySplitCountMatchesReference) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    uint32_t seed = 70000;
    for (const int64_t splits : {1, 2, 3, 4, 8, 16}) {
        for (const int64_t seq_kv : {1, 127, 1024, 4096}) {
            DecodeCase c;
            c.dtype = sci::DType::kFloat16;
            c.heads_q = 8;
            c.heads_kv = 8;
            c.seq_kv = seq_kv;
            c.head_dim = 128;
            c.num_splits = splits;
            c.seed = seed++;
            RunDecodeCase(c);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

TEST(DecodeAttentionCorrectness, GroupedQueryAttentionAcrossSplitCounts) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    uint32_t seed = 71000;
    for (const int64_t group : {int64_t{1}, int64_t{2}, int64_t{4}, int64_t{8}}) {
        for (const int64_t splits : {1, 4, 16}) {
            DecodeCase c;
            c.dtype = sci::DType::kFloat16;
            c.heads_q = 8 * group;
            c.heads_kv = 8;
            c.seq_kv = 2048;
            c.head_dim = 128;
            c.num_splits = splits;
            c.seed = seed++;
            RunDecodeCase(c);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

TEST(DecodeAttentionCorrectness, SupportsBf16AndEveryHeadDim) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    uint32_t seed = 72000;
    for (const int64_t head_dim : {32, 64, 96, 128, 160, 192, 256}) {
        DecodeCase c;
        c.dtype = sci::DType::kBFloat16;
        c.heads_q = 4;
        c.heads_kv = 2;
        c.seq_kv = 512;
        c.head_dim = head_dim;
        c.num_splits = 4;
        c.seed = seed++;
        RunDecodeCase(c);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

TEST(DecodeAttentionCorrectness, RejectsMultiTokenQueries) {
    std::shared_ptr<sci::CudaDevice> device = Gpu();
    if (!device) GTEST_SKIP() << "no CUDA device";
    const int64_t seq_q = 4, seq_kv = 64, head_dim = 64;
    const size_t q_elements = static_cast<size_t>(2 * seq_q * head_dim);
    const size_t kv_elements = static_cast<size_t>(2 * seq_kv * head_dim);
    const Encoded q_enc = Encode(RandomFloats(q_elements, 5), sci::DType::kFloat16);
    const Encoded kv_enc = Encode(RandomFloats(kv_elements, 7), sci::DType::kFloat16);
    sci::Tensor q = MakeDeviceTensor({1, 2, seq_q, head_dim}, sci::DType::kFloat16, q_enc, *device);
    sci::Tensor k = MakeDeviceTensor({1, 2, seq_kv, head_dim}, sci::DType::kFloat16, kv_enc, *device);
    sci::Tensor v = MakeDeviceTensor({1, 2, seq_kv, head_dim}, sci::DType::kFloat16, kv_enc, *device);
    sca::AttentionConfig cfg;
    cfg.backend = sca::BackendKind::kDecode;
    cfg.causal = true;
    const auto result = sca::decode_attention(q, k, v, cfg);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(sca::ExtractCode(result.error()), sca::AttnStatusCode::kUnsupportedFeature);
    EXPECT_NE(result.error().message().find("flash"), std::string::npos);
}

