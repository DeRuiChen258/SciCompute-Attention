// Level 5 (paged) backend adapter.
//
// Paged attention serves the decode step of a *batch of sequences* whose KV lives in non-contiguous
// pages. It reuses the split-K decode kernel with page-table addressing (see
// backends/paged/paged_attention.cuh) so that the merge formula and numerics stay in one place.

#include "scicompute_attention/paged_attention.hpp"

#include <string>
#include <vector>

#include "backends/decode/decode_config.hpp"
#include "backends/decode/decode_splitk.cuh"
#include "backends/paged/paged_attention.cuh"
#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/capability.hpp"
#include "scicompute_attention/detail/dtype_traits.hpp"
#include "scicompute_attention/detail/layout_traits.hpp"
#include "scicompute_attention/dispatcher.hpp"
#include "scicompute_attention/status.hpp"
#include "runtime/backend_registry.hpp"
#include "runtime/scratch_pool.hpp"

namespace sca {
namespace {

using cuda::DecodeDtype;
using cuda::DecodeFwdParams;
using cuda::DecodeStrides;

bool DtypeIdFor(sci::DType dtype, DecodeDtype* out) {
    switch (dtype) {
        case sci::DType::kFloat32: *out = DecodeDtype::kFp32; return true;
        case sci::DType::kFloat16: *out = DecodeDtype::kFp16; return true;
        case sci::DType::kBFloat16: *out = DecodeDtype::kBf16; return true;
        default: return false;
    }
}

sci::Result<AttentionResult> RunPaged(const sci::Tensor& q, const PagedKVCache& kv,
                                      const PagedAttentionParams& params,
                                      const AttentionConfig& cfg) {
    if (q.ndims() != 4) {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kShapeMismatch,
            "paged_attention expects q=[num_seqs, H_q, seq_q, D]; got ndims=" +
                std::to_string(q.ndims())));
    }
    const int64_t num_seqs = q.dim(0);
    const int64_t num_heads = q.dim(1);
    const int64_t seq_q = q.dim(2);
    const int64_t head_dim = q.dim(3);
    if (seq_q != decode::kNativeSeqQ) {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedFeature,
            "paged attention implements the decode form only (seq_q=1); got seq_q=" +
                std::to_string(seq_q)));
    }
    if (num_seqs <= 0 || num_seqs > kv.MaxNumSeqs()) {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kShapeMismatch,
            "paged_attention num_seqs=" + std::to_string(num_seqs) +
                " exceeds the cache's max_num_seqs=" + std::to_string(kv.MaxNumSeqs())));
    }
    if (params.layer < 0 || params.layer >= kv.Storage().Config().num_layers) {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kShapeMismatch,
            "paged_attention layer=" + std::to_string(params.layer) + " out of range [0," +
                std::to_string(kv.Storage().Config().num_layers) + ")"));
    }
    DecodeDtype dtype_id = DecodeDtype::kFp16;
    if (!DtypeIdFor(q.dtype(), &dtype_id)) {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedDtype,
            std::string("dtype=") + sci::kDTypeName(q.dtype()) + " is not a compute dtype"));
    }
    if (head_dim % 32 != 0 || SupportedHeadDimIndex(head_dim) < 0) {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedHeadDim,
            "paged attention requires head_dim % 32 == 0 within {32..256}; got " +
                std::to_string(head_dim)));
    }

    const int32_t* page_table =
        params.page_table != nullptr ? params.page_table : kv.PageTableDevicePtr();
    if (page_table == nullptr) {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kShapeMismatch,
            "paged_attention has no page table (call PagedKVCache::SyncPageTable first)"));
    }
    const int64_t block_size = params.block_size > 0 ? params.block_size : kv.BlockSize();
    const int64_t max_blocks =
        params.max_blocks_per_seq > 0 ? params.max_blocks_per_seq : kv.MaxBlocksPerSeq();

    const KVCacheConfig& kv_cfg = kv.Storage().Config();
    const DeviceCapability& cap = DeviceCapability::ForDevice(0);
    AttentionShape shape;
    shape.batch = num_seqs;
    shape.seq_q = seq_q;
    shape.seq_kv = kv_cfg.num_blocks * block_size;  // upper bound; per-sequence lengths come from
                                                    // `seq_lens` when provided
    shape.num_heads = num_heads;
    shape.num_kv_heads = kv_cfg.num_kv_heads;
    shape.head_dim = head_dim;
    const int64_t splits = cfg.num_splits > 0
                               ? detail::Min<int64_t>(cfg.num_splits, decode::kMaxSplits)
                               : decode::SplitsFor(shape.seq_kv, cap.sm_count);
    const size_t partial_bytes =
        static_cast<size_t>(decode::PartialElements(num_seqs, num_heads, splits, head_dim)) *
        sizeof(float);
    const sci::Result<float*> partials = runtime::AcquireScratchFp32(
        runtime::ScratchKind::kDecodePartials, partial_bytes / sizeof(float), q.device());
    if (!partials.ok()) return sci::MakeUnexpected<AttentionResult>(partials.error());
    const int64_t base = num_seqs * num_heads * splits;

    AttentionResult result;
    result.out = sci::Tensor(sci::TensorShape({num_seqs, num_heads, seq_q, head_dim}), q.dtype(),
                             q.device());
    if (result.out.data() == nullptr) {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kWorkspaceExceeded, "allocation of the paged output failed"));
    }

    const sci::Tensor& k_layer = kv.Storage().K(params.layer);
    const sci::Tensor& v_layer = kv.Storage().V(params.layer);
    DecodeStrides q_stride;
    q_stride.s_b = num_heads * seq_q * head_dim;
    q_stride.s_h = seq_q * head_dim;
    q_stride.s_s = head_dim;
    DecodeStrides o_stride = q_stride;
    DecodeStrides kv_stride;
    kv_stride.s_b = 0;  // unused in the paged path (the page table supplies the row)
    kv_stride.s_h = head_dim;
    kv_stride.s_s = kv_cfg.num_kv_heads * head_dim;  // element stride between physical rows

    DecodeFwdParams p;
    p.q = q.data();
    p.k = k_layer.data();
    p.v = v_layer.data();
    p.out = result.out.data();
    p.partial_m = *partials;
    p.partial_l = *partials + base;
    p.partial_o = *partials + 2 * base;
    p.batch = num_seqs;
    p.num_heads = num_heads;
    p.num_kv_heads = kv_cfg.num_kv_heads;
    p.seq_kv = shape.seq_kv;
    p.head_dim = head_dim;
    p.num_splits = splits;
    p.q_stride = q_stride;
    p.k_stride = kv_stride;
    p.v_stride = kv_stride;
    p.o_stride = o_stride;
    p.scale = EffectiveScale(cfg, shape);
    p.dtype = dtype_id;
    p.page_table = page_table;
    p.block_size = block_size;
    p.max_blocks_per_seq = max_blocks;
    p.num_kv_blocks = kv_cfg.num_blocks;
    p.seq_lens = params.seq_lens;

    cuda::PagedDecodeParams paged;
    paged.base = p;
    paged.page_table = page_table;
    paged.block_size = block_size;
    paged.max_blocks_per_seq = max_blocks;
    paged.num_kv_blocks = kv_cfg.num_blocks;
    const sci::Status status = cuda::LaunchPagedAttention(paged, cfg.stream);
    if (!status.ok()) return sci::MakeUnexpected<AttentionResult>(status);

    result.stats.used_backend = BackendKind::kPaged;
    result.stats.workspace_bytes = partial_bytes;
    result.stats.num_splits = splits;
    result.stats.kv_bytes = kv.Storage().MemoryBytes();
    result.stats.note = "paged decode: " + std::to_string(splits) + " splits, block_size=" +
                        std::to_string(block_size) + ", layer=" + std::to_string(params.layer);
    return sci::Ok(std::move(result));
}

class PagedBackend final : public IAttentionBackend {
public:
    BackendKind Kind() const noexcept override { return BackendKind::kPaged; }
    const char* Name() const noexcept override { return "paged"; }

    CapabilityReport Supports(const AttentionConfig&, const AttentionShape& shape) const override {
        CapabilityReport report;
        report.recommended_backend = BackendKind::kDecode;
        report.workspace_bytes = 0;
        std::string reason;
        if (!DeviceCapability::CudaAvailable()) {
            reason = "no CUDA device available";
        } else if (shape.seq_q != decode::kNativeSeqQ) {
            reason = "paged attention implements the decode form only (seq_q=1)";
        } else if (shape.head_dim % 32 != 0) {
            reason = "paged attention requires head_dim % 32 == 0";
        }
        report.supported = reason.empty();
        report.reason = report.supported
                            ? "paged decode over non-contiguous pages (page-table addressing)"
                            : reason;
        return report;
    }

    size_t WorkspaceBytes(const AttentionConfig& cfg, const AttentionShape& shape) const override {
        const DeviceCapability& cap = DeviceCapability::ForDevice(0);
        const int64_t splits = cfg.num_splits > 0 ? cfg.num_splits
                                                  : decode::SplitsFor(shape.seq_kv, cap.sm_count);
        return static_cast<size_t>(
                   decode::PartialElements(shape.batch, shape.num_heads, splits, shape.head_dim)) *
               sizeof(float);
    }

    sci::Result<AttentionResult> Forward(const sci::Tensor&, const sci::Tensor&, const sci::Tensor&,
                                        const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedFeature,
            "paged attention needs the KV cache: call sca::paged_attention(q, kv, params, cfg)"));
    }

    sci::Result<AttentionResult> ForwardVarlen(const sci::Tensor&, const sci::Tensor&,
                                               const sci::Tensor&, int64_t, const int32_t*,
                                               const int32_t*, int64_t, int64_t,
                                               const AttentionConfig&) const override {
        return sci::MakeUnexpected<AttentionResult>(MakeStatus(
            AttnStatusCode::kUnsupportedFeature,
            "paged backend does not implement varlen; use flash_attention_varlen"));
    }
};

const PagedBackend kPagedBackend;

}  // namespace

namespace runtime {
const IAttentionBackend* PagedBackendInstance() noexcept { return &kPagedBackend; }
}  // namespace runtime

sci::Result<AttentionResult> paged_attention(const sci::Tensor& q, const PagedKVCache& kv,
                                             const PagedAttentionParams& params,
                                             const AttentionConfig& cfg) {
    return RunPaged(q, kv, params, cfg);
}

sci::Result<AttentionResult> paged_attention(const sci::Tensor& q, const PagedKVCache& kv,
                                             const AttentionConfig& cfg) {
    PagedAttentionParams params;
    params.page_table = kv.PageTableDevicePtr();
    params.num_seqs = kv.MaxNumSeqs();
    params.block_size = kv.BlockSize();
    params.max_blocks_per_seq = kv.MaxBlocksPerSeq();
    params.layer = 0;
    return RunPaged(q, kv, params, cfg);
}

}  // namespace sca

