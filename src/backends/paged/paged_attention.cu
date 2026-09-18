// Paged attention launch: shares the split-K decode kernels (see paged_attention.cuh).

#include "backends/paged/paged_attention.cuh"

#include <string>

#include "backends/decode/decode_splitk.cuh"

namespace sca {
namespace cuda {

sci::Status LaunchPagedAttention(const PagedDecodeParams& params, sci::Stream* stream) {
    if (params.page_table == nullptr) {
        return sci::Status::InvalidArgument("paged attention: null page table");
    }
    DecodeFwdParams p = params.base;
    p.page_table = params.page_table;
    p.block_size = params.block_size;
    p.max_blocks_per_seq = params.max_blocks_per_seq;
    p.num_kv_blocks = params.num_kv_blocks;
    return LaunchDecodeSplitKPaged(p, stream);
}

}  // namespace cuda
}  // namespace sca

