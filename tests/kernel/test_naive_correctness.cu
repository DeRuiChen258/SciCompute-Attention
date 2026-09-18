// Level 0 correctness: naive attention versus a double-precision host reference.
//
// Coverage: fp32/fp16/bf16 x causal/non-causal x GQA group {1,2,4} x S in
// {1,7,63,64,65,127,128,129} x D in {64,128} (the remaining head dims are covered by
// test_head_dims.cu once the flash path lands).
//
// The reference is computed from the *quantized* inputs, so narrow-dtype runs are not penalised
// for storage rounding - only for arithmetic.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "device/cuda_device.hpp"
#include "scicompute_attention/attention.hpp"
#include "common/test_utils.hpp"

namespace {

using namespace sca_test;

std::shared_ptr<sci::CudaDevice> GpuOrNull() {
    if (!sca::DeviceCapability::CudaAvailable()) return nullptr;
    return sci::CudaDevice::Create(0);
}

struct NaiveCase {
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

std::vector<NaiveCase> BuildCases() {
    std::vector<NaiveCase> cases;
    const sci::DType dtypes[] = {sci::DType::kFloat32, sci::DType::kFloat16,
                                 sci::DType::kBFloat16};
    const int64_t seqs[] = {1, 7, 63, 64, 65, 127, 128, 129};
    uint32_t seed = 1234;
    for (const sci::DType dtype : dtypes) {
        for (const bool causal : {false, true}) {
            for (const int64_t seq : seqs) {
                for (const int64_t head_dim : {int64_t{64}, int64_t{128}}) {
                    cases.push_back({dtype, causal, 1, 4, 4, seq, seq, head_dim, seed++});
                }
            }
        }
    }
    return cases;
}

void RunAndCompare(const NaiveCase& c) {
    std::shared_ptr<sci::CudaDevice> device = GpuOrNull();
    if (device == nullptr) GTEST_SKIP() << "no CUDA device";

    const size_t q_elements = static_cast<size_t>(c.batch * c.heads_q * c.seq_q * c.head_dim);
    const size_t kv_elements = static_cast<size_t>(c.batch * c.heads_kv * c.seq_kv * c.head_dim);

    const Encoded q_enc = Encode(RandomFloats(q_elements, c.seed), c.dtype);
    const Encoded k_enc = Encode(RandomFloats(kv_elements, c.seed + 7919), c.dtype);
    const Encoded v_enc = Encode(RandomFloats(kv_elements, c.seed + 104729), c.dtype);

    sci::Tensor q = MakeDeviceTensor({c.batch, c.heads_q, c.seq_q, c.head_dim}, c.dtype, q_enc,
                                     *device);
    sci::Tensor k = MakeDeviceTensor({c.batch, c.heads_kv, c.seq_kv, c.head_dim}, c.dtype, k_enc,
                                     *device);
    sci::Tensor v = MakeDeviceTensor({c.batch, c.heads_kv, c.seq_kv, c.head_dim}, c.dtype, v_enc,
                                     *device);

    sca::AttentionConfig cfg;
    cfg.backend = sca::BackendKind::kNaive;
    cfg.causal = c.causal;

    const sci::Result<sca::AttentionResult> result = sca::naive_attention(q, k, v, cfg);
    const std::string tag = CaseTag("naive", c.dtype, c.causal, c.batch, c.heads_q, c.heads_kv,
                                    c.seq_q, c.seq_kv, c.head_dim, c.seed);
    ASSERT_TRUE(result.ok()) << tag << " failed: " << result.error().ToString();

    const std::vector<double> expect =
        ReferenceAttention(q_enc.dequantized, k_enc.dequantized, v_enc.dequantized, c.batch,
                           c.heads_q, c.heads_kv, c.seq_q, c.seq_kv, c.head_dim, c.causal);
    const std::vector<float> got = ReadToFloat(result->out);
    ASSERT_EQ(got.size(), expect.size()) << tag;
    const ErrorStats stats = Compare(got, expect);
    const Tolerance tol = ToleranceFor(c.dtype, false);
    EXPECT_LE(stats.max_abs, tol.max_abs) << tag;
    EXPECT_LE(stats.mean_abs, tol.mean_abs) << tag;
    EXPECT_EQ(result->stats.used_backend, sca::BackendKind::kNaive) << tag;
    EXPECT_GT(result->stats.materialized_score_bytes, 0) << tag;
}

}  // namespace

TEST(NaiveAttentionCorrectness, MatchesDoubleReferenceAcrossDtypesAndShapes) {
    for (const NaiveCase& c : BuildCases()) {
        RunAndCompare(c);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

TEST(NaiveAttentionCorrectness, SupportsGqaGroups) {
    std::shared_ptr<sci::CudaDevice> device = GpuOrNull();
    if (device == nullptr) GTEST_SKIP() << "no CUDA device";

    for (const int64_t group : {int64_t{1}, int64_t{2}, int64_t{4}, int64_t{8}}) {
        const int64_t heads_q = 8 * group / 8 * 8;  // keep H_q a multiple of 8
        const int64_t heads_kv = heads_q / group;
        const int64_t seq = 96;
        const int64_t head_dim = 128;
        const sci::DType dtype = sci::DType::kFloat16;
        const uint32_t seed = 999 + static_cast<uint32_t>(group);

        const size_t q_elements = static_cast<size_t>(1 * heads_q * seq * head_dim);
        const size_t kv_elements = static_cast<size_t>(1 * heads_kv * seq * head_dim);
        const Encoded q_enc = Encode(RandomFloats(q_elements, seed), dtype);
        const Encoded k_enc = Encode(RandomFloats(kv_elements, seed + 13), dtype);
        const Encoded v_enc = Encode(RandomFloats(kv_elements, seed + 29), dtype);

        sci::Tensor q = MakeDeviceTensor({1, heads_q, seq, head_dim}, dtype, q_enc, *device);
        sci::Tensor k = MakeDeviceTensor({1, heads_kv, seq, head_dim}, dtype, k_enc, *device);
        sci::Tensor v = MakeDeviceTensor({1, heads_kv, seq, head_dim}, dtype, v_enc, *device);

        sca::AttentionConfig cfg;
        cfg.backend = sca::BackendKind::kNaive;
        cfg.causal = true;

        const sci::Result<sca::AttentionResult> result = sca::naive_attention(q, k, v, cfg);
        ASSERT_TRUE(result.ok()) << result.error().ToString();
        const std::vector<double> expect =
            ReferenceAttention(q_enc.dequantized, k_enc.dequantized, v_enc.dequantized, 1, heads_q,
                               heads_kv, seq, seq, head_dim, true);
        const ErrorStats stats = Compare(ReadToFloat(result->out), expect);
        EXPECT_LE(stats.max_abs, 5e-3) << "group=" << group;
    }
}

TEST(NaiveAttentionCorrectness, SupportsBshdLayoutIdentically) {
    std::shared_ptr<sci::CudaDevice> device = GpuOrNull();
    if (device == nullptr) GTEST_SKIP() << "no CUDA device";

    const int64_t batch = 2, heads_q = 8, heads_kv = 2, seq_q = 65, seq_kv = 65, head_dim = 64;
    const sci::DType dtype = sci::DType::kFloat16;
    const uint32_t seed = 4242;
    const size_t q_elements = static_cast<size_t>(batch * heads_q * seq_q * head_dim);
    const size_t kv_elements = static_cast<size_t>(batch * heads_kv * seq_kv * head_dim);
    const Encoded q_enc = Encode(RandomFloats(q_elements, seed), dtype);
    const Encoded k_enc = Encode(RandomFloats(kv_elements, seed + 5), dtype);
    const Encoded v_enc = Encode(RandomFloats(kv_elements, seed + 11), dtype);

    // BHSD staging.
    sci::Tensor q_bhsd =
        MakeDeviceTensor({batch, heads_q, seq_q, head_dim}, dtype, q_enc, *device);
    sci::Tensor k_bhsd =
        MakeDeviceTensor({batch, heads_kv, seq_kv, head_dim}, dtype, k_enc, *device);
    sci::Tensor v_bhsd =
        MakeDeviceTensor({batch, heads_kv, seq_kv, head_dim}, dtype, v_enc, *device);

    // BSHD staging: same values, transposed index order (built on the host to avoid depending on
    // a transpose kernel that lands with the flash backend).
    std::vector<float> q_bshd_float(q_elements), k_bshd_float(kv_elements),
        v_bshd_float(kv_elements);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads_q; ++h) {
            for (int64_t t = 0; t < seq_q; ++t) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    q_bshd_float[static_cast<size_t>(((b * seq_q + t) * heads_q + h) * head_dim + d)] =
                        q_enc.dequantized[static_cast<size_t>(((b * heads_q + h) * seq_q + t) * head_dim + d)];
                }
            }
        }
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads_kv; ++h) {
            for (int64_t t = 0; t < seq_kv; ++t) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    const size_t bhsd =
                        static_cast<size_t>(((b * heads_kv + h) * seq_kv + t) * head_dim + d);
                    const size_t bshd =
                        static_cast<size_t>(((b * seq_kv + t) * heads_kv + h) * head_dim + d);
                    k_bshd_float[bshd] = k_enc.dequantized[bhsd];
                    v_bshd_float[bshd] = v_enc.dequantized[bhsd];
                }
            }
        }
    }
    const Encoded q_bshd = Encode(q_bshd_float, dtype);
    const Encoded k_bshd = Encode(k_bshd_float, dtype);
    const Encoded v_bshd = Encode(v_bshd_float, dtype);

    sci::Tensor q_b =
        MakeDeviceTensor({batch, seq_q, heads_q, head_dim}, dtype, q_bshd, *device);
    sci::Tensor k_b =
        MakeDeviceTensor({batch, seq_kv, heads_kv, head_dim}, dtype, k_bshd, *device);
    sci::Tensor v_b =
        MakeDeviceTensor({batch, seq_kv, heads_kv, head_dim}, dtype, v_bshd, *device);

    sca::AttentionConfig cfg_bhsd;
    cfg_bhsd.backend = sca::BackendKind::kNaive;
    cfg_bhsd.causal = true;
    sca::AttentionConfig cfg_bshd = cfg_bhsd;
    cfg_bshd.layout = sca::AttnLayout::kBSHD;

    const sci::Result<sca::AttentionResult> out_bhsd =
        sca::naive_attention(q_bhsd, k_bhsd, v_bhsd, cfg_bhsd);
    ASSERT_TRUE(out_bhsd.ok()) << out_bhsd.error().ToString();
    const sci::Result<sca::AttentionResult> out_bshd =
        sca::naive_attention(q_b, k_b, v_b, cfg_bshd);
    ASSERT_TRUE(out_bshd.ok()) << out_bshd.error().ToString();

    const std::vector<float> a = ReadToFloat(out_bhsd->out);
    const std::vector<float> b = ReadToFloat(out_bshd->out);
    ASSERT_EQ(a.size(), b.size());
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    EXPECT_LE(max_diff, 1e-6) << "BHSD and BSHD must produce identical results";
}

TEST(NaiveAttentionCorrectness, LseMatchesLogSumExp) {
    std::shared_ptr<sci::CudaDevice> device = GpuOrNull();
    if (device == nullptr) GTEST_SKIP() << "no CUDA device";

    const int64_t batch = 1, heads = 2, seq = 64, head_dim = 64;
    const sci::DType dtype = sci::DType::kFloat32;
    const uint32_t seed = 77;
    const size_t elements = static_cast<size_t>(batch * heads * seq * head_dim);
    const Encoded q_enc = Encode(RandomFloats(elements, seed), dtype);
    const Encoded k_enc = Encode(RandomFloats(elements, seed + 3), dtype);
    const Encoded v_enc = Encode(RandomFloats(elements, seed + 7), dtype);

    sci::Tensor q = MakeDeviceTensor({batch, heads, seq, head_dim}, dtype, q_enc, *device);
    sci::Tensor k = MakeDeviceTensor({batch, heads, seq, head_dim}, dtype, k_enc, *device);
    sci::Tensor v = MakeDeviceTensor({batch, heads, seq, head_dim}, dtype, v_enc, *device);

    sca::AttentionConfig cfg;
    cfg.backend = sca::BackendKind::kNaive;
    cfg.causal = false;
    cfg.return_lse = true;

    const sci::Result<sca::AttentionResult> result = sca::naive_attention(q, k, v, cfg);
    ASSERT_TRUE(result.ok()) << result.error().ToString();
    ASSERT_EQ(result->lse.num_elements(), batch * heads * seq);

    const std::vector<float> lse = ReadToFloat(result->lse);
    // Recompute log-sum-exp on the host for the same inputs.
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t i = 0; i < seq; ++i) {
            double m = -std::numeric_limits<double>::infinity();
            for (int64_t j = 0; j < seq; ++j) {
                double dot = 0.0;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(q_enc.dequantized[static_cast<size_t>(
                               ((h * seq + i) * head_dim + d))]) *
                           static_cast<double>(k_enc.dequantized[static_cast<size_t>(
                               ((h * seq + j) * head_dim + d))]);
                }
                m = std::max(m, dot * scale);
            }
            double sum = 0.0;
            for (int64_t j = 0; j < seq; ++j) {
                double dot = 0.0;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(q_enc.dequantized[static_cast<size_t>(
                               ((h * seq + i) * head_dim + d))]) *
                           static_cast<double>(k_enc.dequantized[static_cast<size_t>(
                               ((h * seq + j) * head_dim + d))]);
                }
                sum += std::exp(dot * scale - m);
            }
            const double expect = m + std::log(sum);
            EXPECT_NEAR(lse[static_cast<size_t>(h * seq + i)], expect, 1e-4);
        }
    }
}
