// Minimal rollout engine: batch prefill + continuous decode over the paged KV cache (prompt §16.4).
//
// What is real here:
//   * prefill runs the flash kernel (mma.sync + cp.async) on synthetic Q/K/V per request;
//   * every prompt token's K/V is appended to a PagedKVCache (block allocation + page table);
//   * each decode step appends one more token and runs the paged attention kernel;
//   * the reported metrics (prefill P50/P99, decode step P50/P99, tokens/s, KV peak) are measured,
//     not simulated.
// What is synthetic: the token generation (no model runs behind it) - this is a *stub* that shows
// the interface an RLHF rollout would use, exactly as the prompt requires.
//
// Usage:
//   ./build/examples/rollout_engine_stub [--n 4] [--max-new 16] [--json out.json]

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "device/cuda_device.hpp"
#include "device/device.hpp"
#include "device/stream.hpp"
#include "scicompute_attention/attention.hpp"
#include "scicompute_attention/capability.hpp"
#include "scicompute_attention/paged_kv_cache.hpp"
#include "tensor/tensor.hpp"

namespace {

constexpr int64_t kHeadsQ = 32;   // Qwen3-4B-Thinking-2507 shape
constexpr int64_t kHeadsKv = 8;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kLayers = 36;
constexpr int64_t kBlockSize = 16;

// Raw CUDA events: sci::Event defaults to a null handle, and this example should not depend on
// the upstream event API to report numbers.
struct EventPair {
    cudaEvent_t begin{};
    cudaEvent_t end{};
    EventPair() {
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
    }
    ~EventPair() {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
    void Record() {
        cudaEventRecord(begin);
        cudaEventRecord(end);
    }
};

struct Args {
    int64_t num_requests{4};
    int64_t max_new_tokens{16};
    int64_t max_kv_len{1024};
    std::string json;
};

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (key == "--n") args.num_requests = std::atoll(next().c_str());
        else if (key == "--max-new") args.max_new_tokens = std::atoll(next().c_str());
        else if (key == "--max-kv") args.max_kv_len = std::atoll(next().c_str());
        else if (key == "--json") args.json = next();
    }
    return args;
}

float Filler(int64_t index, uint32_t seed) {
    uint32_t state = static_cast<uint32_t>(index) * 2654435761u + seed;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (static_cast<float>(state) / static_cast<float>(0xffffffffu) - 0.5f) * 0.2f;
}

uint16_t ToHalf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                          (mantissa >> 13));
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        half = static_cast<uint16_t>(half + 1);
    }
    return half;
}

sci::Tensor MakeHalfTensor(const std::vector<sci::index_t>& dims, sci::Device& device,
                           uint32_t seed, float scale) {
    const sci::TensorShape shape(dims);
    const size_t elements = static_cast<size_t>(shape.num_elements());
    std::vector<uint16_t> host_half(elements);
    for (size_t i = 0; i < elements; ++i) {
        host_half[i] = ToHalf(Filler(static_cast<int64_t>(i), seed) * scale);
    }
    sci::Tensor host(shape, sci::DType::kFloat16, device);
    host.copy_from(host_half.data(), host_half.size() * sizeof(uint16_t));
    sci::Tensor dev(shape, sci::DType::kFloat16, device);
    dev.copy_from(host);
    return dev;
}

// Non-owning 4-D view (BHSD/BSHD) over an existing 3-D buffer. The prefill path uses BSHD, the KV
// append path uses the packed [tokens, heads, dim] form - same bytes, two shapes.
sci::Tensor View4D(const sci::Tensor& source, const std::vector<sci::index_t>& dims) {
    return sci::Tensor(sci::TensorShape(dims), source.dtype(), source.device(),
                       const_cast<void*>(source.data()), /*owns_data=*/false);
}

double Percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double rank = p * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(rank);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] * (1.0 - (rank - lo)) + values[hi] * (rank - lo);
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);
    if (!sca::DeviceCapability::CudaAvailable()) {
        std::printf("[rollout] no CUDA device available; nothing to run\n");
        return 2;
    }
    std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);

    // KV budget check first: the prompt forbids OOM-driven failures (8 GB machine).
    sca::KVCacheConfig kv_cfg;
    kv_cfg.num_layers = kLayers;
    kv_cfg.num_kv_heads = kHeadsKv;
    kv_cfg.head_dim = kHeadDim;
    kv_cfg.block_size = kBlockSize;
    kv_cfg.max_num_seqs = args.num_requests;
    kv_cfg.dtype = sci::DType::kFloat16;
    const int64_t tokens_needed = args.num_requests * (args.max_kv_len + args.max_new_tokens);
    kv_cfg.num_blocks = (tokens_needed + kBlockSize - 1) / kBlockSize + args.num_requests;
    const auto kv_bytes = sca::KvCacheBytes(kv_cfg);
    if (!kv_bytes.ok()) {
        std::printf("[rollout] KV budget rejected: %s\n", kv_bytes.error().ToString().c_str());
        return 3;
    }
    const sca::DeviceCapability& cap = sca::DeviceCapability::ForDevice(0);
    std::printf("[rollout] device=%s free=%.0f MiB, KV plan: %lld blocks = %.1f MiB\n",
                cap.device_name.c_str(), static_cast<double>(cap.free_mem_bytes) / (1024 * 1024),
                static_cast<long long>(kv_cfg.num_blocks),
                static_cast<double>(*kv_bytes) / (1024 * 1024));
    if (*kv_bytes > cap.free_mem_bytes / 2) {
        std::printf("[rollout] refusing to allocate: KV cache would exceed half of free memory\n");
        return 3;
    }

    auto cache_result = sca::PagedKVCache::Create(kv_cfg, *device);
    if (!cache_result.ok()) {
        std::printf("[rollout] PagedKVCache::Create failed: %s\n",
                    cache_result.error().ToString().c_str());
        return 3;
    }
    sca::PagedKVCache cache = std::move(*cache_result);

    // Per-request prompt lengths (uneven on purpose, like a real rollout batch).
    std::vector<int64_t> prompt_lengths(args.num_requests);
    for (int64_t r = 0; r < args.num_requests; ++r) {
        prompt_lengths[static_cast<size_t>(r)] = 32 + (r * 37) % 96;
    }

    std::vector<double> prefill_ms;
    std::vector<double> decode_ms;
    int64_t generated_tokens = 0;
    size_t kv_peak_bytes = 0;

    EventPair prefill_timer;
    EventPair decode_timer;

    for (int64_t r = 0; r < args.num_requests; ++r) {
        const int64_t prompt_len = prompt_lengths[static_cast<size_t>(r)];
        // ---- prefill -------------------------------------------------------------------------
        // Packed [tokens, heads, dim] buffers: the KV append consumes them directly, and the
        // prefill uses a non-owning BSHD view of the same bytes.
        sci::Tensor q_packed = MakeHalfTensor({prompt_len, kHeadsQ, kHeadDim}, *device,
                                              1000 + static_cast<uint32_t>(r), 1.0f);
        sci::Tensor k_packed = MakeHalfTensor({prompt_len, kHeadsKv, kHeadDim}, *device,
                                              2000 + static_cast<uint32_t>(r), 1.0f);
        sci::Tensor v_packed = MakeHalfTensor({prompt_len, kHeadsKv, kHeadDim}, *device,
                                              3000 + static_cast<uint32_t>(r), 0.5f);
        sci::Tensor q = View4D(q_packed, {1, prompt_len, kHeadsQ, kHeadDim});
        sci::Tensor k = View4D(k_packed, {1, prompt_len, kHeadsKv, kHeadDim});
        sci::Tensor v = View4D(v_packed, {1, prompt_len, kHeadsKv, kHeadDim});
        sca::AttentionConfig cfg;
        cfg.backend = sca::BackendKind::kFlash;
        cfg.causal = true;
        cfg.layout = sca::AttnLayout::kBSHD;  // packed buffers are token-major

        cudaEventRecord(prefill_timer.begin);
        const auto prefill = sca::flash_attention(q, k, v, cfg);
        cudaEventRecord(prefill_timer.end);
        cudaEventSynchronize(prefill_timer.end);
        if (!prefill.ok()) {
            std::printf("[rollout] prefill failed: %s\n", prefill.error().ToString().c_str());
            return 4;
        }
        {
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, prefill_timer.begin, prefill_timer.end);
            prefill_ms.push_back(static_cast<double>(elapsed));
        }

        const sci::Status appended = cache.AppendTokens(r, 0, k_packed, v_packed, nullptr);
        if (!appended.ok()) {
            std::printf("[rollout] KV append failed: %s\n", appended.ToString().c_str());
            return 4;
        }
        kv_peak_bytes = std::max(kv_peak_bytes, cache.Storage().MemoryBytes());
    }

    // ---- decode loop (batched, one paged attention call per step) ----------------------------
    std::vector<int32_t> host_lens(prompt_lengths.begin(), prompt_lengths.end());
    sci::Tensor seq_lens(sci::TensorShape({args.num_requests}), sci::DType::kInt32, *device);
    sca::AttentionConfig decode_cfg;
    decode_cfg.backend = sca::BackendKind::kPaged;
    decode_cfg.causal = true;
    for (int64_t step = 0; step < args.max_new_tokens; ++step) {
        for (int64_t r = 0; r < args.num_requests; ++r) {
            sci::Tensor k_step = MakeHalfTensor({1, kHeadsKv, kHeadDim}, *device,
                                                 6000 + static_cast<uint32_t>(r * 97 + step), 1.0f);
            sci::Tensor v_step = MakeHalfTensor({1, kHeadsKv, kHeadDim}, *device,
                                                 7000 + static_cast<uint32_t>(r * 97 + step), 0.5f);
            const sci::Status step_append = cache.AppendTokens(r, 0, k_step, v_step, nullptr);
            if (!step_append.ok()) {
                std::printf("[rollout] decode append failed: %s\n", step_append.ToString().c_str());
                return 4;
            }
        }
        cache.SyncPageTable(nullptr);
        device->copy_to_device(seq_lens.data(), host_lens.data(),
                               host_lens.size() * sizeof(int32_t));

        sci::Tensor q_step = MakeHalfTensor({args.num_requests, kHeadsQ, 1, kHeadDim}, *device,
                                            5000 + static_cast<uint32_t>(step), 1.0f);
        sca::PagedAttentionParams params;
        params.page_table = cache.PageTableDevicePtr();
        params.seq_lens = static_cast<const int32_t*>(seq_lens.data());
        params.num_seqs = args.num_requests;
        params.block_size = kBlockSize;
        params.max_blocks_per_seq = cache.MaxBlocksPerSeq();
        params.layer = 0;

        cudaEventRecord(decode_timer.begin);
        const auto decoded = sca::paged_attention(q_step, cache, params, decode_cfg);
        cudaEventRecord(decode_timer.end);
        cudaEventSynchronize(decode_timer.end);
        if (!decoded.ok()) {
            std::printf("[rollout] decode failed: %s\n", decoded.error().ToString().c_str());
            return 4;
        }
        {
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, decode_timer.begin, decode_timer.end);
            decode_ms.push_back(static_cast<double>(elapsed));
        }
        for (int64_t r = 0; r < args.num_requests; ++r) {
            host_lens[static_cast<size_t>(r)] += 1;
            ++generated_tokens;
        }
        kv_peak_bytes = std::max(kv_peak_bytes, cache.Storage().MemoryBytes());
    }

    const double prefill_p50 = Percentile(prefill_ms, 0.5);
    const double prefill_p99 = Percentile(prefill_ms, 0.99);
    const double decode_p50 = Percentile(decode_ms, 0.5);
    const double decode_p99 = Percentile(decode_ms, 0.99);
    const double total_ms = [&]() {
        double sum = 0.0;
        for (const double v : prefill_ms) sum += v;
        for (const double v : decode_ms) sum += v;
        return sum;
    }();
    const double tokens_per_second = generated_tokens / (total_ms / 1000.0);
    const auto stats = cache.Stats();

    std::printf("[rollout] requests=%lld prompt_len=[%lld..%lld] max_new=%lld\n",
                static_cast<long long>(args.num_requests),
                static_cast<long long>(prompt_lengths.front()),
                static_cast<long long>(prompt_lengths.back()),
                static_cast<long long>(args.max_new_tokens));
    std::printf("[rollout] prefill  P50=%.3f ms P99=%.3f ms (flash, causal)\n", prefill_p50,
                prefill_p99);
    std::printf("[rollout] decode   P50=%.3f ms P99=%.3f ms (paged, split-K)\n", decode_p50,
                decode_p99);
    std::printf("[rollout] tokens/s=%.2f generated=%lld kv_peak=%.1f MiB used_blocks=%lld/%lld\n",
                tokens_per_second, static_cast<long long>(generated_tokens),
                static_cast<double>(kv_peak_bytes) / (1024 * 1024),
                static_cast<long long>(stats.used_blocks),
                static_cast<long long>(stats.num_blocks));

    if (!args.json.empty()) {
        std::ofstream out(args.json);
        out << "{\n";
        out << "  \"engine\": \"rollout_engine_stub\",\n";
        out << "  \"requests\": " << args.num_requests << ",\n";
        out << "  \"max_new_tokens\": " << args.max_new_tokens << ",\n";
        out << "  \"prefill_latency_ms\": {\"p50\": " << prefill_p50 << ", \"p99\": " << prefill_p99
            << "},\n";
        out << "  \"decode_step_latency_ms\": {\"p50\": " << decode_p50 << ", \"p99\": "
            << decode_p99 << "},\n";
        out << "  \"tokens_per_second\": " << tokens_per_second << ",\n";
        out << "  \"generated_tokens\": " << generated_tokens << ",\n";
        out << "  \"kv_peak_bytes\": " << kv_peak_bytes << ",\n";
        out << "  \"kv_blocks_used\": " << stats.used_blocks << ",\n";
        out << "  \"kv_blocks_total\": " << stats.num_blocks << ",\n";
        out << "  \"shape\": {\"layers\": " << kLayers << ", \"Hq\": " << kHeadsQ
            << ", \"Hkv\": " << kHeadsKv << ", \"D\": " << kHeadDim << "}\n";
        out << "}\n";
        std::printf("[rollout] json written: %s\n", args.json.c_str());
    }
    return 0;
}
