#pragma once

#include <cstdint>

#include "core/status.hpp"
#include "core/types.hpp"
#include "scicompute_attention/export.hpp"

namespace sca {

// Tensor layout of Q/K/V as provided by the caller.
//   kBHSD = [B, H, S, D]  (internal compute layout; also the fastest path)
//   kBSHD = [B, S, H, D]  (vLLM(C++) convention; strides are handled by layout_traits)
enum class AttnLayout : int32_t { kBHSD = 0, kBSHD = 1 };

// KV storage layout. v1 implements exactly one layout (see docs/kv_cache.md §8.1).
enum class KvLayout : int32_t { kBlockMajor = 0 };

// Backend selector. kAuto asks the dispatcher to choose (see dispatcher.hpp).
enum class BackendKind : int32_t {
    kAuto = 0,
    kNaive,
    kTiled,
    kFlash,
    kDecode,
    kPaged,
};

struct AttentionShape {
    int64_t batch{0};
    int64_t seq_q{0};
    int64_t seq_kv{0};
    int64_t num_heads{0};     // H_q
    int64_t num_kv_heads{0};  // H_kv (GQA/MQA when H_kv < H_q)
    int64_t head_dim{0};

    int64_t QueryElements() const noexcept {
        return batch * seq_q * num_heads * head_dim;
    }
    int64_t KvElements() const noexcept {
        return batch * seq_kv * num_kv_heads * head_dim;
    }
    int64_t OutputElements() const noexcept { return QueryElements(); }
    bool IsGqa() const noexcept {
        return num_kv_heads > 0 && num_kv_heads != num_heads;
    }
    int64_t GroupSize() const noexcept {
        return num_kv_heads > 0 ? num_heads / num_kv_heads : 0;
    }
};

inline constexpr int64_t kMinSupportedHeadDim = 32;
inline constexpr int64_t kMaxSupportedHeadDim = 256;
inline constexpr int64_t kSupportedHeadDims[] = {32, 64, 96, 128, 160, 192, 256};
inline constexpr int64_t kNumSupportedHeadDims = 7;

// Yields the position of `head_dim` inside kSupportedHeadDims, or -1 when unsupported.
SCI_ATTENTION_API int SupportedHeadDimIndex(int64_t head_dim) noexcept;

SCI_ATTENTION_API const char* LayoutName(AttnLayout layout) noexcept;
SCI_ATTENTION_API const char* BackendName(BackendKind kind) noexcept;
SCI_ATTENTION_API const char* KvLayoutName(KvLayout layout) noexcept;

// Number of bytes per element for the compute dtypes (FP16/BF16 = 2, FP32 = 4).
SCI_ATTENTION_API bool IsSupportedComputeDtype(sci::DType dtype) noexcept;
SCI_ATTENTION_API int64_t DtypeSize(sci::DType dtype) noexcept;

}  // namespace sca

