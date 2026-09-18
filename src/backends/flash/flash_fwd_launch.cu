// Flash forward: template instantiation + launch (prompt §7.4 "flash_fwd_launch.cu").
//
// One calibrated geometry per supported head dimension (see flash_tile_config.hpp). Every
// instantiation is checked at compile time against the measured shared-memory limit, so a table
// edit that does not fit the device fails the build instead of failing at runtime.

#include "backends/flash/flash_fwd_kernel.cuh"

#include <cuda_runtime.h>

#include <string>

#include "backends/flash/flash_fwd_impl.cuh"
#include "backends/flash/flash_tile_config.hpp"
#include "cuda_common/arch_features.cuh"
#include "cuda_common/cuda_check.cuh"
#include "device/stream.hpp"
#include "runtime/launcher.hpp"
#include "scicompute_attention/status.hpp"

namespace sca {
namespace cuda {
namespace {

// Compile-time smem of the instantiated layout (independent of the runtime table).
template <typename T, int kHeadDim, int kBlockM, int kBlockN, int kStages, int kSplitsD>
constexpr int64_t FlashTileSmemBytesConst() {
    constexpr int64_t kRowStride = kHeadDim + kRowPadElems;
    constexpr int64_t q = static_cast<int64_t>(kBlockM) * kRowStride * sizeof(T);
    constexpr int64_t kv = static_cast<int64_t>(kBlockN) * kRowStride * sizeof(T) * kStages;
    return q + 2 * kv;
}

template <typename T, int kHeadDim, int kBlockM, int kBlockN, int kStages, int kSplitsD>
sci::Status LaunchTyped(const FlashFwdParams& p, sci::Stream* stream, bool causal) {
    constexpr int kWarps = (kBlockM / 16) * kSplitsD;
    constexpr int64_t kSmemBytes =
        FlashTileSmemBytesConst<T, kHeadDim, kBlockM, kBlockN, kStages, kSplitsD>();
    const size_t smem = static_cast<size_t>(kSmemBytes);
    const int grid_x = static_cast<int>((p.seq_q + kBlockM - 1) / kBlockM);
    const int grid_y = static_cast<int>(p.num_heads);
    const int grid_z = static_cast<int>(p.batch);
    void* args[] = {const_cast<FlashFwdParams*>(&p)};
    if (causal) {
        return runtime::LaunchRaw(
            reinterpret_cast<const void*>(
                &flash_fwd_kernel<T, kHeadDim, kBlockM, kBlockN, kStages, kSplitsD, true>),
            grid_x, grid_y, grid_z, kWarps * 32, 1, 1, smem, stream, args);
    }
    return runtime::LaunchRaw(
        reinterpret_cast<const void*>(
            &flash_fwd_kernel<T, kHeadDim, kBlockM, kBlockN, kStages, kSplitsD, false>),
        grid_x, grid_y, grid_z, kWarps * 32, 1, 1, smem, stream, args);
}

template <typename T, int kHeadDim, int kBlockM, int kBlockN, int kStages, int kSplitsD>
sci::Status LaunchChecked(const FlashFwdParams& p, sci::Stream* stream, const flash::FlashTile& t) {
    static_assert(FlashTileSmemBytesConst<T, kHeadDim, kBlockM, kBlockN, kStages, kSplitsD>() <=
                      kMaxDynamicSmemBytes,
                  "flash instantiation exceeds the measured shared-memory limit");
    if (t.block_m != kBlockM || t.block_n != kBlockN || t.stages != kStages ||
        t.splits_d != kSplitsD) {
        return sci::Status::InvalidArgument(
            "flash: tile table and compiled instantiation disagree for head_dim=" +
            std::to_string(kHeadDim) + " (table " + t.ToString() + ")");
    }
    return LaunchTyped<T, kHeadDim, kBlockM, kBlockN, kStages, kSplitsD>(p, stream, p.causal);
}

template <typename T, int kHeadDim>
sci::Status LaunchForHeadDim(const FlashFwdParams& p, sci::Stream* stream) {
    const flash::FlashTile* tile = flash::FindFlashTile(kHeadDim);
    if (tile == nullptr) {
        return sci::Status::InvalidArgument("flash: no tile entry for head_dim=" +
                                            std::to_string(kHeadDim));
    }
    // Keep the geometry table and the instantiation in sync at compile time (if constexpr so that
    // only the matching geometry is instantiated - otherwise the smem static_assert fires for the
    // large-head-dim instantiations of every small head dim).
    if constexpr (kHeadDim <= 128) {
        return LaunchChecked<T, kHeadDim, 64, 64, 2, 1>(p, stream, *tile);
    } else if constexpr (kHeadDim <= 192) {
        return LaunchChecked<T, kHeadDim, 64, 32, 2, 2>(p, stream, *tile);
    } else {
        return LaunchChecked<T, kHeadDim, 32, 32, 2, 2>(p, stream, *tile);
    }
}

}  // namespace

int64_t FlashSmemBytesFor(int32_t head_dim, int64_t elem_bytes, int32_t block_m, int32_t block_n,
                          int32_t stages) {
    if (head_dim <= 0 || block_m <= 0 || block_n <= 0 || stages <= 0) return -1;
    const int64_t row_stride = head_dim + kRowPadElems;
    const int64_t q = static_cast<int64_t>(block_m) * row_stride * elem_bytes;
    const int64_t kv = static_cast<int64_t>(block_n) * row_stride * elem_bytes * stages;
    return q + 2 * kv;
}

sci::Status LaunchFlashFwd(const FlashFwdParams& params, sci::Stream* stream) {
    if (params.q == nullptr || params.k == nullptr || params.v == nullptr ||
        params.out == nullptr) {
        return sci::Status::InvalidArgument("flash attention: null tensor pointer");
    }
    switch (params.dtype_id) {
        case 1:  // fp16
            switch (params.head_dim) {
                case 32: return LaunchForHeadDim<__half, 32>(params, stream);
                case 64: return LaunchForHeadDim<__half, 64>(params, stream);
                case 96: return LaunchForHeadDim<__half, 96>(params, stream);
                case 128: return LaunchForHeadDim<__half, 128>(params, stream);
                case 160: return LaunchForHeadDim<__half, 160>(params, stream);
                case 192: return LaunchForHeadDim<__half, 192>(params, stream);
                case 256: return LaunchForHeadDim<__half, 256>(params, stream);
                default: break;
            }
            break;
        case 2:  // bf16
            switch (params.head_dim) {
                case 32: return LaunchForHeadDim<__nv_bfloat16, 32>(params, stream);
                case 64: return LaunchForHeadDim<__nv_bfloat16, 64>(params, stream);
                case 96: return LaunchForHeadDim<__nv_bfloat16, 96>(params, stream);
                case 128: return LaunchForHeadDim<__nv_bfloat16, 128>(params, stream);
                case 160: return LaunchForHeadDim<__nv_bfloat16, 160>(params, stream);
                case 192: return LaunchForHeadDim<__nv_bfloat16, 192>(params, stream);
                case 256: return LaunchForHeadDim<__nv_bfloat16, 256>(params, stream);
                default: break;
            }
            break;
        default:
            return sci::Status::InvalidArgument(
                "flash attention: only fp16/bf16 are instantiated (dtype_id=" +
                std::to_string(params.dtype_id) + "); use the naive or tiled backend for FP32");
    }
    return sci::Status::InvalidArgument("flash attention: head_dim=" +
                                        std::to_string(params.head_dim) + " is not instantiated");
}

}  // namespace cuda
}  // namespace sca
