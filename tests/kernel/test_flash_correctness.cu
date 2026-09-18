// Level 3 correctness: flash attention (mma.sync + cp.async + register online softmax) against the
// double-precision host reference.
//
// Coverage: fp16/bf16 x causal/non-causal x D in {32,64,96,128,160,192,256} x S in
// {1,7,63,64,65,127,128,129,1024} x GQA group {1,2,4,8}, plus non-tile-multiple sequence lengths.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common/test_utils.hpp"
#include "device/cuda_device.hpp"
#include "scicompute_attention/attention.hpp"

namespace {

using namespace sca_test;

struct FlashCase {
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

bool RunCase(const FlashCase& c, bool check_lse = false) {
    if (!sca::DeviceCapability::CudaAvailable()) return false;  // callers GTEST_SKIP first
    static std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    if (!device) return false;

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
    cfg.backend = sca::BackendKind::kFlash;
    cfg.causal = c.causal;
    cfg.return_lse = check_lse;

    const sci::Result<sca::AttentionResult> result = sca::flash_attention(q, k, v, cfg);
    const std::string tag = CaseTag("flash", c.dtype, c.causal, c.batch, c.heads_q, c.heads_kv,
                                    c.seq_q, c.seq_kv, c.head_dim, c.seed);
    if (!result.ok()) {
        ADD_FAILURE() << tag << " failed: " << result.error().ToString();
        return false;
    }

    const std::vector<double> expect =
        ReferenceAttention(q_enc.dequantized, k_enc.dequantized, v_enc.dequantized, c.batch,
                           c.heads_q, c.heads_kv, c.seq_q, c.seq_kv, c.head_dim, c.causal);
    const ErrorStats stats = Compare(ReadToFloat(result->out), expect);
    const Tolerance tol = ToleranceFor(c.dtype, false);
    EXPECT_LE(stats.max_abs, tol.max_abs) << tag;
    EXPECT_LE(stats.mean_abs, tol.mean_abs) << tag;
    EXPECT_EQ(result->stats.workspace_bytes, 0u) << tag;
    EXPECT_EQ(result->stats.materialized_score_bytes, 0) << tag;

    if (check_lse) {
        EXPECT_EQ(result->lse.num_elements(), c.batch * c.heads_q * c.seq_q) << tag;
        // LSE = log-sum-exp of the (masked) scaled logits, verified against the host reference.
        const std::vector<float> lse = ReadToFloat(result->lse);
        const int64_t group = c.heads_q / c.heads_kv;
        const int64_t diagonal = c.causal ? (c.seq_kv - c.seq_q) : 0;
        double worst = 0.0;
        for (int64_t b = 0; b < c.batch; ++b) {
            for (int64_t h = 0; h < c.heads_q; ++h) {
                for (int64_t i = 0; i < c.seq_q; ++i) {
                    double m = -std::numeric_limits<double>::infinity();
                    for (int64_t j = 0; j < c.seq_kv; ++j) {
                        if (c.causal && j > i + diagonal) continue;
                        double dot = 0.0;
                        for (int64_t d = 0; d < c.head_dim; ++d) {
                            dot += static_cast<double>(q_enc.dequantized[static_cast<size_t>(
                                       ((b * c.heads_q + h) * c.seq_q + i) * c.head_dim + d)]) *
                                   static_cast<double>(k_enc.dequantized[static_cast<size_t>(
                                       ((b * c.heads_kv + h / group) * c.seq_kv + j) * c.head_dim +
                                       d)]);
                        }
                        m = std::max(m, dot / std::sqrt(static_cast<double>(c.head_dim)));
                    }
                    double sum = 0.0;
                    for (int64_t j = 0; j < c.seq_kv; ++j) {
                        if (c.causal && j > i + diagonal) continue;
                        double dot = 0.0;
                        for (int64_t d = 0; d < c.head_dim; ++d) {
                            dot += static_cast<double>(q_enc.dequantized[static_cast<size_t>(
                                       ((b * c.heads_q + h) * c.seq_q + i) * c.head_dim + d)]) *
                                   static_cast<double>(k_enc.dequantized[static_cast<size_t>(
                                       ((b * c.heads_kv + h / group) * c.seq_kv + j) * c.head_dim +
                                       d)]);
                        }
                        sum += std::exp(dot / std::sqrt(static_cast<double>(c.head_dim)) - m);
                    }
                    const double expect_lse = m + std::log(sum);
                    const double got_lse =
                        static_cast<double>(lse[static_cast<size_t>((b * c.heads_q + h) * c.seq_q + i)]);
                    worst = std::max(worst, std::fabs(got_lse - expect_lse));
                }
            }
        }
        EXPECT_LE(worst, 1e-2) << tag << " LSE max abs error";
    }
    return true;
}

}  // namespace

TEST(FlashAttentionCorrectness, MatchesReferenceAcrossDtypesAndHeadDims) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    uint32_t seed = 20000;
    for (const sci::DType dtype : {sci::DType::kFloat16, sci::DType::kBFloat16}) {
        for (const bool causal : {false, true}) {
            for (const int64_t head_dim : {32, 64, 96, 128, 160, 192, 256}) {
                FlashCase c;
                c.dtype = dtype;
                c.causal = causal;
                c.batch = 1;
                c.heads_q = 8;
                c.heads_kv = 8;
                c.seq_q = 128;
                c.seq_kv = 128;
                c.head_dim = head_dim;
                c.seed = seed++;
                if (!RunCase(c)) return;
            }
        }
    }
}

TEST(FlashAttentionCorrectness, HandlesNonTileMultiplesAndShortSequences) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    uint32_t seed = 30000;
    for (const int64_t seq : {1, 7, 63, 64, 65, 127, 128, 129, 1024}) {
        for (const bool causal : {false, true}) {
            FlashCase c;
            c.dtype = sci::DType::kFloat16;
            c.causal = causal;
            c.batch = 1;
            c.heads_q = 8;
            c.heads_kv = 8;
            c.seq_q = seq;
            c.seq_kv = seq;
            c.head_dim = 128;
            c.seed = seed++;
            if (!RunCase(c)) return;
        }
    }
}

TEST(FlashAttentionCorrectness, SupportsGqaGroups) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    uint32_t seed = 40000;
    for (const int64_t group : {int64_t{1}, int64_t{2}, int64_t{4}, int64_t{8}}) {
        FlashCase c;
        c.dtype = sci::DType::kFloat16;
        c.causal = true;
        c.batch = 2;
        c.heads_q = 8 * group;
        c.heads_kv = 8;
        c.seq_q = 96;
        c.seq_kv = 160;
        c.head_dim = 128;
        c.seed = seed++;
        if (!RunCase(c)) return;
    }
}

TEST(FlashAttentionCorrectness, EmitsLseMatchingLogSumExp) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    FlashCase c;
    c.dtype = sci::DType::kFloat16;
    c.causal = false;
    c.batch = 1;
    c.heads_q = 4;
    c.heads_kv = 2;
    c.seq_q = 64;
    c.seq_kv = 64;
    c.head_dim = 64;
    c.seed = 50123;
    RunCase(c, /*check_lse=*/true);
}

TEST(FlashAttentionCorrectness, RejectsFp32WithPointerToReferenceBackends) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    static std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    const int64_t seq = 32, head_dim = 64;
    const size_t elements = static_cast<size_t>(1 * 2 * seq * head_dim);
    const Encoded enc = Encode(RandomFloats(elements, 11), sci::DType::kFloat32);
    sci::Tensor q = MakeDeviceTensor({1, 2, seq, head_dim}, sci::DType::kFloat32, enc, *device);
    sci::Tensor k = MakeDeviceTensor({1, 2, seq, head_dim}, sci::DType::kFloat32, enc, *device);
    sci::Tensor v = MakeDeviceTensor({1, 2, seq, head_dim}, sci::DType::kFloat32, enc, *device);

    sca::AttentionConfig cfg;
    cfg.backend = sca::BackendKind::kFlash;
    const auto result = sca::flash_attention(q, k, v, cfg);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(sca::ExtractCode(result.error()), sca::AttnStatusCode::kUnsupportedDtype);
    EXPECT_NE(result.error().message().find("naive"), std::string::npos);
}
