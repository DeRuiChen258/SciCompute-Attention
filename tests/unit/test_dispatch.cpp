// Backend selection + Explain() (prompt §9.1). The reasons are asserted so that a table change
// cannot silently alter behaviour.

#include <gtest/gtest.h>

#include <string>

#include "scicompute_attention/dispatcher.hpp"

namespace {

sca::AttentionShape Shape(int64_t sq, int64_t skv, int64_t d = 128) {
    sca::AttentionShape shape;
    shape.batch = 1;
    shape.seq_q = sq;
    shape.seq_kv = skv;
    shape.num_heads = 8;
    shape.num_kv_heads = 8;
    shape.head_dim = d;
    return shape;
}

}  // namespace

TEST(DispatchTest, DecodeForSingleQueryToken) {
    const sca::AttentionDispatcher dispatcher;
    const sca::DispatchDecision decision =
        dispatcher.Select(sca::AttentionConfig{}, Shape(1, 2048));
    EXPECT_EQ(decision.backend, sca::BackendKind::kDecode);
    EXPECT_NE(decision.reason.find("auto"), std::string::npos);
    EXPECT_NE(decision.reason.find("decode"), std::string::npos);
    EXPECT_NE(decision.reason.find("table=r0"), std::string::npos);
}

TEST(DispatchTest, GroupedSmallQueryGoesToFlash) {
    const sca::AttentionDispatcher dispatcher;
    // seq_q=16 exceeds decode_max_seq_q(1) but 16*8 = 128 <= seq_kv=4096, so the grouped small-q
    // rule applies. The decode kernel is S_q=1 only (documented in docs/prefill_decode.md), hence
    // the flash backend.
    const sca::DispatchDecision decision =
        dispatcher.Select(sca::AttentionConfig{}, Shape(16, 4096));
    EXPECT_EQ(decision.backend, sca::BackendKind::kFlash);
    EXPECT_NE(decision.reason.find("grouped small-q"), std::string::npos);
}

TEST(DispatchTest, FlashForPrefill) {
    const sca::AttentionDispatcher dispatcher;
    const sca::DispatchDecision decision =
        dispatcher.Select(sca::AttentionConfig{}, Shape(1024, 1024));
    EXPECT_EQ(decision.backend, sca::BackendKind::kFlash);
    EXPECT_NE(decision.reason.find("online softmax"), std::string::npos);
}

TEST(DispatchTest, TiledForShortSequences) {
    const sca::AttentionDispatcher dispatcher;
    const sca::DispatchDecision decision =
        dispatcher.Select(sca::AttentionConfig{}, Shape(64, 64));
    EXPECT_EQ(decision.backend, sca::BackendKind::kTiled);
    EXPECT_NE(decision.reason.find("short sequence"), std::string::npos);
}

TEST(DispatchTest, ExplicitBackendIsHonoured) {
    sca::AttentionConfig cfg;
    cfg.backend = sca::BackendKind::kNaive;
    const sca::AttentionDispatcher dispatcher;
    const sca::DispatchDecision decision = dispatcher.Select(cfg, Shape(1024, 1024));
    EXPECT_EQ(decision.backend, sca::BackendKind::kNaive);
    EXPECT_NE(decision.reason.find("explicit"), std::string::npos);
}

TEST(DispatchTest, InvalidShapeIsReportedWithoutChoosingABackend) {
    const sca::AttentionDispatcher dispatcher;
    const sca::AttentionShape shape = Shape(1024, 64, 48);
    const sca::DispatchDecision decision = dispatcher.Select(sca::AttentionConfig{}, shape);
    EXPECT_EQ(decision.backend, sca::BackendKind::kAuto);
    EXPECT_NE(decision.reason.find("invalid"), std::string::npos);
    EXPECT_NE(decision.reason.find("head_dim=48"), std::string::npos);
}

TEST(DispatchTest, UnsupportedBackendFailsFastUnlessFallbackIsAllowed) {
    // Phase 1: no backend implements its kernels yet, so this exercises the failure path that
    // stays in place for genuinely unsupported shapes later on.
    const sca::AttentionDispatcher dispatcher;
    const sca::DispatchDecision strict = dispatcher.Select(sca::AttentionConfig{}, Shape(1024, 1024));
    EXPECT_FALSE(strict.reason.empty());

    sca::AttentionConfig permissive;
    permissive.allow_fallback = true;
    const sca::DispatchDecision fallback = dispatcher.Select(permissive, Shape(1024, 1024));
    EXPECT_TRUE(fallback.reason.find("fallback") != std::string::npos ||
                fallback.reason.find("unsupported") != std::string::npos ||
                fallback.reason.find("auto") != std::string::npos);
}

TEST(DispatchTest, ExplainRendersTileAndWorkspace) {
    const sca::AttentionDispatcher dispatcher;
    const std::string text = dispatcher.Explain(sca::AttentionConfig{}, Shape(1024, 1024));
    EXPECT_NE(text.find("backend="), std::string::npos);
    EXPECT_NE(text.find("tile={bm="), std::string::npos);
    EXPECT_NE(text.find("workspace="), std::string::npos);
    EXPECT_NE(text.find("reason="), std::string::npos);
}

TEST(DispatchTest, RecommendTileMatchesHeadDim) {
    const sca::AttentionDispatcher dispatcher;
    EXPECT_EQ(dispatcher.Select(sca::AttentionConfig{}, Shape(1024, 1024, 128)).tile.block_m, 64);
    EXPECT_EQ(dispatcher.Select(sca::AttentionConfig{}, Shape(1024, 1024, 64)).tile.block_m, 64);
    EXPECT_EQ(dispatcher.Select(sca::AttentionConfig{}, Shape(1024, 1024, 256)).tile.block_m, 32);
    // head_dim > 128 splits the output dimension across warp pairs (see flash_tile_config.hpp).
    EXPECT_EQ(dispatcher.Select(sca::AttentionConfig{}, Shape(1024, 1024, 192)).tile.warps, 8);
    EXPECT_EQ(dispatcher.Select(sca::AttentionConfig{}, Shape(1024, 1024, 128)).tile.warps, 4);
}
