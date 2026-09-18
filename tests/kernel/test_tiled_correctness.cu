// Level 1 correctness: tiled attention (smem-resident S, online softmax, register accumulation)
// against the double-precision host reference.
//
// Additional assertions required by the prompt:
//   * workspace_bytes == 0 and materialized_score_bytes == 0  -> "N x N is never materialized";
//   * identical results for BHSD and BSHD inputs;
//   * head_dim > 128 is rejected with kDeviceCapability and a reason that points at flash.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common/test_utils.hpp"
#include "device/cuda_device.hpp"
#include "scicompute_attention/attention.hpp"

namespace {

using namespace sca_test;

struct TiledCase {
    sci::DType dtype;
    bool causal;
    int64_t batch;
    int64_t heads_q;
    int64_t heads_kv;
    int64_t seq_q;
    int64_t seq_kv;
    int64_t head_dim;
    uint32_t seed;
};

std::vector<TiledCase> BuildCases() {
    std::vector<TiledCase> cases;
    uint32_t seed = 5000;
    const sci::DType dtypes[] = {sci::DType::kFloat32, sci::DType::kFloat16,
                                 sci::DType::kBFloat16};
    const int64_t head_dims[] = {32, 64, 96, 128};
    const int64_t seqs[] = {1, 17, 64, 65, 129, 256};
    for (const sci::DType dtype : dtypes) {
        for (const bool causal : {false, true}) {
            for (const int64_t head_dim : head_dims) {
                for (const int64_t seq : seqs) {
                    cases.push_back({dtype, causal, 1, 4, 4, seq, seq, head_dim, seed++});
                }
            }
        }
    }
    return cases;
}

void RunCase(const TiledCase& c, bool check_not_materialized = true) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);

    const size_t q_elements = static_cast<size_t>(c.batch * c.heads_q * c.seq_q * c.head_dim);
    const size_t kv_elements = static_cast<size_t>(c.batch * c.heads_kv * c.seq_kv * c.head_dim);
    const Encoded q_enc = Encode(RandomFloats(q_elements, c.seed), c.dtype);
    const Encoded k_enc = Encode(RandomFloats(kv_elements, c.seed + 7919), c.dtype);
    const Encoded v_enc = Encode(RandomFloats(kv_elements, c.seed + 104729), c.dtype);

    sci::Tensor q =
        MakeDeviceTensor({c.batch, c.heads_q, c.seq_q, c.head_dim}, c.dtype, q_enc, *device);
    sci::Tensor k =
        MakeDeviceTensor({c.batch, c.heads_kv, c.seq_kv, c.head_dim}, c.dtype, k_enc, *device);
    sci::Tensor v =
        MakeDeviceTensor({c.batch, c.heads_kv, c.seq_kv, c.head_dim}, c.dtype, v_enc, *device);

    sca::AttentionConfig cfg;
    cfg.backend = sca::BackendKind::kTiled;
    cfg.causal = c.causal;

    const sci::Result<sca::AttentionResult> result = sca::tiled_attention(q, k, v, cfg);
    const std::string tag = CaseTag("tiled", c.dtype, c.causal, c.batch, c.heads_q, c.heads_kv,
                                    c.seq_q, c.seq_kv, c.head_dim, c.seed);
    ASSERT_TRUE(result.ok()) << tag << " failed: " << result.error().ToString();

    const std::vector<double> expect =
        ReferenceAttention(q_enc.dequantized, k_enc.dequantized, v_enc.dequantized, c.batch,
                           c.heads_q, c.heads_kv, c.seq_q, c.seq_kv, c.head_dim, c.causal);
    const ErrorStats stats = Compare(ReadToFloat(result->out), expect);
    const Tolerance tol = ToleranceFor(c.dtype, false);
    EXPECT_LE(stats.max_abs, tol.max_abs) << tag;
    EXPECT_LE(stats.mean_abs, tol.mean_abs) << tag;
    if (check_not_materialized) {
        EXPECT_EQ(result->stats.workspace_bytes, 0u) << tag;
        EXPECT_EQ(result->stats.materialized_score_bytes, 0) << tag;
    }
}

}  // namespace

TEST(TiledAttentionCorrectness, MatchesDoubleReferenceAcrossDtypesAndShapes) {
    for (const TiledCase& c : BuildCases()) {
        RunCase(c);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

TEST(TiledAttentionCorrectness, SupportsGqaGroupsAndNonSquareSequences) {
    for (const int64_t group : {int64_t{1}, int64_t{2}, int64_t{4}, int64_t{8}}) {
        for (const bool causal : {false, true}) {
            TiledCase c;
            c.dtype = sci::DType::kFloat16;
            c.causal = causal;
            c.batch = 2;
            c.heads_q = 8 * group;
            c.heads_kv = 8;
            c.seq_q = causal ? 64 : 128;  // causal requires S_kv >= S_q
            c.seq_kv = 128;
            c.head_dim = 128;
            c.seed = 9000 + static_cast<uint32_t>(group) * 7 + (causal ? 1u : 0u);
            RunCase(c);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

TEST(TiledAttentionCorrectness, BshdMatchesBhsd) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    const int64_t batch = 1, heads_q = 8, heads_kv = 2, seq = 96, head_dim = 64;
    const sci::DType dtype = sci::DType::kFloat16;
    const size_t q_elements = static_cast<size_t>(batch * heads_q * seq * head_dim);
    const size_t kv_elements = static_cast<size_t>(batch * heads_kv * seq * head_dim);
    const Encoded q_enc = Encode(RandomFloats(q_elements, 31), dtype);
    const Encoded k_enc = Encode(RandomFloats(kv_elements, 37), dtype);
    const Encoded v_enc = Encode(RandomFloats(kv_elements, 41), dtype);

    sci::Tensor q_bhsd = MakeDeviceTensor({batch, heads_q, seq, head_dim}, dtype, q_enc, *device);
    sci::Tensor k_bhsd = MakeDeviceTensor({batch, heads_kv, seq, head_dim}, dtype, k_enc, *device);
    sci::Tensor v_bhsd = MakeDeviceTensor({batch, heads_kv, seq, head_dim}, dtype, v_enc, *device);

    // BSHD staging: rebuild the same logical tensors with the sequence/head axes swapped.
    std::vector<float> q_bshd(q_elements, 0.0f), k_bshd(kv_elements, 0.0f), v_bshd(kv_elements, 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads_q; ++h) {
            for (int64_t t = 0; t < seq; ++t) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    const size_t bhsd = static_cast<size_t>(((b * heads_q + h) * seq + t) * head_dim + d);
                    const size_t bshd = static_cast<size_t>(((b * seq + t) * heads_q + h) * head_dim + d);
                    q_bshd[bshd] = q_enc.dequantized[bhsd];
                }
            }
        }
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads_kv; ++h) {
            for (int64_t t = 0; t < seq; ++t) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    const size_t bhsd = static_cast<size_t>(((b * heads_kv + h) * seq + t) * head_dim + d);
                    const size_t bshd = static_cast<size_t>(((b * seq + t) * heads_kv + h) * head_dim + d);
                    k_bshd[bshd] = k_enc.dequantized[bhsd];
                    v_bshd[bshd] = v_enc.dequantized[bhsd];
                }
            }
        }
    }
    const Encoded q_b = Encode(q_bshd, dtype);
    const Encoded k_b = Encode(k_bshd, dtype);
    const Encoded v_b = Encode(v_bshd, dtype);
    sci::Tensor q_bshd_dev = MakeDeviceTensor({batch, seq, heads_q, head_dim}, dtype, q_b, *device);
    sci::Tensor k_bshd_dev = MakeDeviceTensor({batch, seq, heads_kv, head_dim}, dtype, k_b, *device);
    sci::Tensor v_bshd_dev = MakeDeviceTensor({batch, seq, heads_kv, head_dim}, dtype, v_b, *device);

    sca::AttentionConfig cfg_a;
    cfg_a.backend = sca::BackendKind::kTiled;
    cfg_a.causal = true;
    sca::AttentionConfig cfg_b = cfg_a;
    cfg_b.layout = sca::AttnLayout::kBSHD;

    const auto out_a = sca::tiled_attention(q_bhsd, k_bhsd, v_bhsd, cfg_a);
    const auto out_b = sca::tiled_attention(q_bshd_dev, k_bshd_dev, v_bshd_dev, cfg_b);
    ASSERT_TRUE(out_a.ok()) << out_a.error().ToString();
    ASSERT_TRUE(out_b.ok()) << out_b.error().ToString();
    const ErrorStats stats = Compare(ReadToFloat(out_a->out), ReadToFloat(out_b->out));
    EXPECT_LE(stats.max_abs, 1e-6);
}

TEST(TiledAttentionCorrectness, RejectsLargeHeadDimWithReason) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    const int64_t seq = 32, head_dim = 256;
    const size_t elements = static_cast<size_t>(1 * 2 * seq * head_dim);
    const Encoded q_enc = Encode(RandomFloats(elements, 61), sci::DType::kFloat16);

    sci::Tensor q = MakeDeviceTensor({1, 2, seq, head_dim}, sci::DType::kFloat16, q_enc, *device);
    sci::Tensor k = MakeDeviceTensor({1, 2, seq, head_dim}, sci::DType::kFloat16, q_enc, *device);
    sci::Tensor v = MakeDeviceTensor({1, 2, seq, head_dim}, sci::DType::kFloat16, q_enc, *device);

    sca::AttentionConfig cfg;
    cfg.backend = sca::BackendKind::kTiled;
    const auto result = sca::tiled_attention(q, k, v, cfg);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(sca::ExtractCode(result.error()), sca::AttnStatusCode::kDeviceCapability);
    EXPECT_NE(result.error().message().find("flash"), std::string::npos);
}

