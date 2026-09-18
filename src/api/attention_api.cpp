// Public API implementation (prompt §6): shape inference -> validation -> dispatch -> execute.
// No exceptions escape these functions; every failure is a sci::Status carried by Result<T>.

#include "scicompute_attention/attention.hpp"

#include <string>

#include "scicompute_attention/backends.hpp"
#include "scicompute_attention/dispatcher.hpp"
#include "scicompute_attention/status.hpp"

namespace sca {
namespace {

AttentionConfig WithBackend(const AttentionConfig& cfg, BackendKind kind) {
    AttentionConfig out = cfg;
    out.backend = kind;
    return out;
}

}  // namespace

sci::Result<AttentionResult> attention(const sci::Tensor& q, const sci::Tensor& k,
                                       const sci::Tensor& v, const AttentionConfig& cfg) {
    const AttentionDispatcher dispatcher(cfg);
    return dispatcher.Forward(q, k, v, cfg);
}

sci::Result<AttentionResult> decode(const sci::Tensor& q, const sci::Tensor& k,
                                    const sci::Tensor& v, const AttentionConfig& cfg) {
    return attention(q, k, v, WithBackend(cfg, BackendKind::kDecode));
}

sci::Result<AttentionResult> paged(const sci::Tensor& q, const PagedKVCache& kv,
                                   const AttentionConfig& cfg) {
    PagedAttentionParams params;
    const AttentionConfig resolved = WithBackend(cfg, BackendKind::kPaged);
    return paged_attention(q, kv, params, resolved);
}

std::string ExplainAttention(const sci::Tensor& q, const sci::Tensor& k, const sci::Tensor& v,
                             const AttentionConfig& cfg) {
    const sci::Result<AttentionShape> shape = InferShape(q, k, v, cfg.layout);
    if (!shape.ok()) return "invalid: " + shape.message();
    const AttentionDispatcher dispatcher(cfg);
    return dispatcher.Explain(cfg, *shape);
}

}  // namespace sca

