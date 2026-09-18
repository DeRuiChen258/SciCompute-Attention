// Validate() / EffectiveScale() / ValidateVarlenHost() rules (prompt §6.2).

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "scicompute_attention/attention_config.hpp"
#include "scicompute_attention/status.hpp"

namespace {

sca::AttentionShape MakeShape(int64_t b = 1, int64_t sq = 128, int64_t skv = 128, int64_t hq = 8,
                              int64_t hkv = 8, int64_t d = 128) {
    sca::AttentionShape shape;
    shape.batch = b;
    shape.seq_q = sq;
    shape.seq_kv = skv;
    shape.num_heads = hq;
    shape.num_kv_heads = hkv;
    shape.head_dim = d;
    return shape;
}

}  // namespace

TEST(AttentionConfigTest, AcceptsEverySupportedHeadDim) {
    for (int64_t i = 0; i < sca::kNumSupportedHeadDims; ++i) {
        const sca::AttentionShape shape = MakeShape(1, 64, 64, 8, 8, sca::kSupportedHeadDims[i]);
        const sca::AttentionConfig cfg;
        EXPECT_TRUE(sca::Validate(cfg, shape).ok())
            << "head_dim=" << sca::kSupportedHeadDims[i] << " should be supported";
    }
}

TEST(AttentionConfigTest, RejectsUnsupportedHeadDimAndNamesTheValue) {
    const sca::AttentionShape shape = MakeShape(1, 64, 64, 8, 8, 48);
    std::string detail;
    const sci::Status status = sca::Validate(sca::AttentionConfig{}, shape, &detail);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(sca::ExtractCode(status), sca::AttnStatusCode::kUnsupportedHeadDim);
    EXPECT_NE(detail.find("head_dim=48"), std::string::npos);
    EXPECT_NE(detail.find("128"), std::string::npos);  // the supported list is printed too
}

TEST(AttentionConfigTest, RejectsCausalWhenKvShorterThanQuery) {
    sca::AttentionConfig cfg;
    cfg.causal = true;
    std::string detail;
    const sci::Status status = sca::Validate(cfg, MakeShape(1, 256, 128), &detail);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(detail.find("causal=true"), std::string::npos);
    EXPECT_NE(detail.find("seq_kv(128)"), std::string::npos);
    EXPECT_NE(detail.find("seq_q(256)"), std::string::npos);
}

TEST(AttentionConfigTest, RejectsNonDivisibleGqa) {
    const sci::Status status =
        sca::Validate(sca::AttentionConfig{}, MakeShape(1, 64, 64, 32, 6, 128));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(sca::ExtractCode(status), sca::AttnStatusCode::kShapeMismatch);
    EXPECT_NE(status.message().find("num_heads=32"), std::string::npos);
    EXPECT_NE(status.message().find("num_kv_heads=6"), std::string::npos);
}

TEST(AttentionConfigTest, RejectsSlidingWindowAsUnimplemented) {
    sca::AttentionConfig cfg;
    cfg.sliding_window = 512;
    const sci::Status status = sca::Validate(cfg, MakeShape());
    EXPECT_EQ(sca::ExtractCode(status), sca::AttnStatusCode::kUnsupportedFeature);
    EXPECT_NE(status.message().find("sliding_window=512"), std::string::npos);
}

TEST(AttentionConfigTest, SoftcapOnlyOnFlashPath) {
    sca::AttentionConfig cfg;
    cfg.softcap = 50.0f;
    cfg.backend = sca::BackendKind::kNaive;
    EXPECT_FALSE(sca::Validate(cfg, MakeShape()).ok());

    cfg.backend = sca::BackendKind::kFlash;
    EXPECT_TRUE(sca::Validate(cfg, MakeShape()).ok());
}

TEST(AttentionConfigTest, EvaluatesDefaultScaleFromHeadDim) {
    sca::AttentionShape shape = MakeShape(1, 64, 64, 8, 8, 128);
    sca::AttentionConfig cfg;
    EXPECT_NEAR(sca::EffectiveScale(cfg, shape), 1.0f / std::sqrt(128.0f), 1e-7f);
    cfg.scale = 0.5f;
    EXPECT_FLOAT_EQ(sca::EffectiveScale(cfg, shape), 0.5f);
    shape.head_dim = 64;
    cfg.scale = 0.0f;
    EXPECT_NEAR(sca::EffectiveScale(cfg, shape), 0.125f, 1e-7f);
}

TEST(AttentionConfigTest, ValidatesVarlenHostMetadata) {
    const int32_t good[] = {0, 5, 9, 20};
    int64_t max_seq = 0;
    EXPECT_TRUE(sca::ValidateVarlenHost(good, 3, &max_seq).ok());
    EXPECT_EQ(max_seq, 11);

    const int32_t not_starting_at_zero[] = {1, 5, 9};
    EXPECT_FALSE(sca::ValidateVarlenHost(not_starting_at_zero, 2).ok());

    const int32_t not_monotonic[] = {0, 5, 5};
    EXPECT_FALSE(sca::ValidateVarlenHost(not_monotonic, 2).ok());
    EXPECT_EQ(sca::ExtractCode(sca::ValidateVarlenHost(not_monotonic, 2)),
              sca::AttnStatusCode::kShapeMismatch);

    EXPECT_FALSE(sca::ValidateVarlenHost(nullptr, 2).ok());
}

