// Error-code mapping and message formatting (prompt §6.11).

#include <gtest/gtest.h>

#include <string>

#include "scicompute_attention/attention_types.hpp"
#include "scicompute_attention/status.hpp"
#include "scicompute_attention/version.hpp"

TEST(AttentionStatusTest, EveryCodeHasAStableString) {
    const sca::AttnStatusCode codes[] = {
        sca::AttnStatusCode::kOk,
        sca::AttnStatusCode::kUnsupportedHeadDim,
        sca::AttnStatusCode::kUnsupportedDtype,
        sca::AttnStatusCode::kUnsupportedLayout,
        sca::AttnStatusCode::kLayoutMismatch,
        sca::AttnStatusCode::kShapeMismatch,
        sca::AttnStatusCode::kSeqTooLong,
        sca::AttnStatusCode::kKVCapacityExceeded,
        sca::AttnStatusCode::kWorkspaceExceeded,
        sca::AttnStatusCode::kUnsupportedFeature,
        sca::AttnStatusCode::kDeviceCapability,
    };
    for (const sca::AttnStatusCode code : codes) {
        const char* text = sca::ToString(code);
        ASSERT_NE(text, nullptr);
        EXPECT_GT(std::string(text).size(), 0u);
    }
    EXPECT_STREQ(sca::ToString(sca::AttnStatusCode::kKVCapacityExceeded), "KVCapacityExceeded");
}

TEST(AttentionStatusTest, MakeStatusCarriesBothCodes) {
    const sci::Status status = sca::MakeStatus(sca::AttnStatusCode::kKVCapacityExceeded,
                                               "requested 40 blocks, free 8");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), sci::StatusCode::kResourceExhausted);
    EXPECT_EQ(sca::ExtractCode(status), sca::AttnStatusCode::kKVCapacityExceeded);
    EXPECT_NE(status.message().find("[SCA-2007]"), std::string::npos);
    EXPECT_NE(status.message().find("requested 40 blocks"), std::string::npos);
}

TEST(AttentionStatusTest, ToSciCodeIsTotal) {
    EXPECT_EQ(sca::ToSciCode(sca::AttnStatusCode::kUnsupportedHeadDim),
              sci::StatusCode::kInvalidArgument);
    EXPECT_EQ(sca::ToSciCode(sca::AttnStatusCode::kWorkspaceExceeded),
              sci::StatusCode::kResourceExhausted);
    EXPECT_EQ(sca::ToSciCode(sca::AttnStatusCode::kUnsupportedFeature),
              sci::StatusCode::kNotImplemented);
    EXPECT_EQ(sca::ToSciCode(sca::AttnStatusCode::kDeviceCapability),
              sci::StatusCode::kInvalidOperation);
    EXPECT_EQ(sca::ExtractCode(sci::Status::Ok()), sca::AttnStatusCode::kOk);
}

TEST(AttentionStatusTest, NamesAndVersionAreExposed) {
    EXPECT_STREQ(sca::BackendName(sca::BackendKind::kFlash), "flash");
    EXPECT_STREQ(sca::LayoutName(sca::AttnLayout::kBSHD), "bshd");
    EXPECT_STREQ(sca::VersionString(), SCI_ATTENTION_VERSION_STRING);
    EXPECT_EQ(sca::DtypeSize(sci::DType::kFloat16), 2);
    EXPECT_EQ(sca::DtypeSize(sci::DType::kBFloat16), 2);
    EXPECT_TRUE(sca::IsSupportedComputeDtype(sci::DType::kFloat32));
    EXPECT_FALSE(sca::IsSupportedComputeDtype(sci::DType::kInt8));
    EXPECT_EQ(sca::SupportedHeadDimIndex(128), 3);
    EXPECT_EQ(sca::SupportedHeadDimIndex(48), -1);
}

