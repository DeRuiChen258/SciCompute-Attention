// TileConfig smem arithmetic. The same formula is mirrored by tools/smem_calc.py; this test pins
// the C++ side so that both stay in sync (prompt §7.7).

#include <gtest/gtest.h>

#include "scicompute_attention/detail/tile_config.hpp"

namespace {

int64_t Smem(int64_t head_dim, int64_t elem_bytes, int bm, int bn, int warps, int stages) {
    return sca::TileConfig::SmemBytesFlash(static_cast<int32_t>(head_dim), elem_bytes, bm, bn,
                                           warps, stages);
}

}  // namespace

TEST(TileConfigTest, MatchesHandComputedLayout) {
    // D=128, BM=64, BN=64, stages=2, fp16, pad=8 halves (16 B):
    //   Q  = 64 * (128+8) * 2 = 17408 B
    //   KV = 64 * (128+8) * 2 = 17408 B each
    //   total = Q + 2*stages*(K+V)?
    // The layout stores `stages` K tiles and `stages` V tiles:
    //   total = 17408 + 2 * 2 * 17408 = 87040 B
    EXPECT_EQ(Smem(128, 2, 64, 64, 4, 2), 87040);
}

TEST(TileConfigTest, EveryRecommendedTileFitsTheDeviceLimit) {
    const int64_t smem_limit = 101376;  // measured opt-in limit, docs/env_report.md
    struct Case {
        int64_t head_dim;
        int bm, bn, warps, stages;
    };
    // Mirrors the primary recommendation table in src/api/api_validate.cpp.
    const Case cases[] = {
        {32, 128, 128, 4, 2}, {64, 128, 128, 4, 2},  {96, 128, 64, 4, 2},
        {128, 64, 64, 4, 2},  {160, 64, 32, 4, 2},   {192, 64, 32, 4, 2},
        {256, 32, 32, 4, 2},
    };
    for (const Case& c : cases) {
        const int64_t bytes = Smem(c.head_dim, 2, c.bm, c.bn, c.warps, c.stages);
        EXPECT_GT(bytes, 0);
        EXPECT_LE(bytes, smem_limit)
            << "D=" << c.head_dim << " bm=" << c.bm << " bn=" << c.bn
            << " stages=" << c.stages << " needs " << bytes << " B";
    }
}

TEST(TileConfigTest, RejectsStructurallyInvalidConfigurations) {
    EXPECT_LT(Smem(0, 2, 64, 64, 4, 2), 0);
    EXPECT_LT(Smem(128, 2, 0, 64, 4, 2), 0);
    EXPECT_LT(Smem(128, 2, 64, 0, 4, 2), 0);
    EXPECT_LT(Smem(128, 2, 64, 64, 0, 2), 0);
    EXPECT_LT(Smem(128, 2, 64, 64, 4, 0), 0);
}

TEST(TileConfigTest, StringifiesForReports) {
    sca::TileConfig tile;
    tile.block_m = 64;
    tile.block_n = 64;
    tile.warps = 4;
    tile.stages = 2;
    EXPECT_EQ(tile.ToString(), "bm=64,bn=64,warps=4,stages=2,tma=off");
    EXPECT_EQ(tile.Threads(), 128);
    EXPECT_FALSE(tile.Empty());
    EXPECT_TRUE(sca::TileConfig{}.Empty());
}

