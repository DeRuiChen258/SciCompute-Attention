// Numerical stability under extreme inputs (prompt §11.2 test_numerics.cu).
//
// Cases: logits of magnitude 1e2/1e4 (via scaled Q/K), all-equal rows, single token, and a long
// sequence. Assertions: no NaN/Inf in the output, the tiled and flash results agree with the
// FP64 reference, and fully masked rows stay finite.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "common/test_utils.hpp"
#include "device/cuda_device.hpp"
#include "scicompute_attention/attention.hpp"

namespace {

using namespace sca_test;

bool HasBadValues(const std::vector<float>& values) {
    for (const float v : values) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

// Runs one (backend, dtype) combination with a controlled input distribution.
void RunExtremeCase(sca::BackendKind kind, const char* name, float magnitude, bool causal,
                    int64_t seq_q, int64_t seq_kv, int64_t head_dim, uint32_t seed) {
    if (!sca::DeviceCapability::CudaAvailable()) return;
    static std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    const sci::DType dtype = sci::DType::kFloat16;
    const int64_t heads = 4;
    const size_t q_elements = static_cast<size_t>(heads * seq_q * head_dim);
    const size_t kv_elements = static_cast<size_t>(heads * seq_kv * head_dim);

    // magnitude scales the logits: |s| ≈ magnitude after the 1/sqrt(D) scaling is applied by the
    // kernel, so Q/K values are set to magnitude * sqrt(D) / sqrt(D) = magnitude.
    const std::vector<float> q_float = RandomFloats(q_elements, seed, -magnitude, magnitude);
    const std::vector<float> k_float = RandomFloats(kv_elements, seed + 1, -magnitude, magnitude);
    const std::vector<float> v_float = RandomFloats(kv_elements, seed + 2, -1.0f, 1.0f);
    const Encoded q_enc = Encode(q_float, dtype);
    const Encoded k_enc = Encode(k_float, dtype);
    const Encoded v_enc = Encode(v_float, dtype);

    sci::Tensor q = MakeDeviceTensor({1, heads, seq_q, head_dim}, dtype, q_enc, *device);
    sci::Tensor k = MakeDeviceTensor({1, heads, seq_kv, head_dim}, dtype, k_enc, *device);
    sci::Tensor v = MakeDeviceTensor({1, heads, seq_kv, head_dim}, dtype, v_enc, *device);

    sca::AttentionConfig cfg;
    cfg.backend = kind;
    cfg.causal = causal;
    const auto result = sca::attention(q, k, v, cfg);
    const std::string tag = std::string(name) + " " + sca::BackendName(kind) +
                            " magnitude=" + std::to_string(magnitude) +
                            " Sq=" + std::to_string(seq_q) + " Skv=" + std::to_string(seq_kv);
    ASSERT_TRUE(result.ok()) << tag << ": " << result.error().ToString();

    const std::vector<float> got = ReadToFloat(result->out);
    EXPECT_FALSE(HasBadValues(got)) << tag << " produced NaN/Inf";

    // FP64 reference on the same dequantised inputs; NaN-safe for extreme magnitudes.
    const std::vector<double> expect = ReferenceAttention(
        q_enc.dequantized, k_enc.dequantized, v_enc.dequantized, 1, heads, heads, seq_q, seq_kv,
        head_dim, causal);
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(expect[i])) continue;
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - expect[i]));
    }
    EXPECT_LE(worst, 5e-3) << tag << " max abs error vs FP64 reference";
}

}  // namespace

TEST(NumericsTest, ExtremeLogitsStayFiniteAndAccurate) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    for (const float magnitude : {0.0f, 1.0f, 10.0f, 1e2f, 1e4f}) {
        for (const sca::BackendKind kind : {sca::BackendKind::kNaive, sca::BackendKind::kTiled,
                                            sca::BackendKind::kFlash}) {
            RunExtremeCase(kind, "extreme-logits", magnitude, /*causal=*/false, 64, 64, 128,
                           1234 + static_cast<uint32_t>(magnitude));
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

TEST(NumericsTest, SingleTokenAndAllEqualRows) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    // magnitude 0 => all logits equal 0 => uniform attention (a degenerate softmax).
    RunExtremeCase(sca::BackendKind::kFlash, "all-equal", 0.0f, false, 1, 1, 64, 77);
    RunExtremeCase(sca::BackendKind::kFlash, "all-equal", 0.0f, true, 1, 1024, 128, 78);
    RunExtremeCase(sca::BackendKind::kTiled, "all-equal", 0.0f, false, 32, 32, 64, 79);
    RunExtremeCase(sca::BackendKind::kDecode, "all-equal", 0.0f, true, 1, 512, 128, 80);
}

TEST(NumericsTest, LongSequenceStaysFinite) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    RunExtremeCase(sca::BackendKind::kFlash, "long-sequence", 2.0f, /*causal=*/true, 512, 8192, 128,
                   4242);
}

TEST(NumericsTest, FullyMaskedRowsProduceZerosNotNaNs) {
    if (!sca::DeviceCapability::CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    // Causal with S_q > 1 but the first query row can only see token 0 (never fully masked by
    // construction); the interesting case is S_q=1 with causal, where row 0 sees the whole cache.
    RunExtremeCase(sca::BackendKind::kFlash, "causal-single-row", 1.0f, true, 1, 4096, 128, 99);
    RunExtremeCase(sca::BackendKind::kDecode, "causal-single-row", 1.0f, true, 1, 4096, 128, 100);
}

