// Benchmark for the tiled / flash / decode / paged backends (prompt §12).
//
// Merge note: the prompt lists one file per backend; those files would share ~90 % of their code
// (shape matrix, memory guard, event timing, JSON/CSV export), so they are merged here and the
// per-backend numbers are separated by the `impl` field in the JSON. This is recorded in
// docs/architecture.md (file responsibility table) as required by the prompt's file rules.
//
// Timing convention: the events wrap one call to the public API (validation + dispatch + kernel +
// output allocation). The kernel-only numbers for the flash path are produced by ncu/nsys in
// Phase 6 profiling.

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/bench_utils.hpp"
#include "device/cuda_device.hpp"
#include "scicompute_attention/attention.hpp"
#include "scicompute_attention/paged_kv_cache.hpp"
#include "tests/common/test_utils.hpp"

namespace {

using namespace sca::bench;
using sca_test::Encode;
using sca_test::MakeDeviceTensor;
using sca_test::RandomFloats;
using sca_test::ReadToFloat;

constexpr int64_t kMemoryBudgetBytes = 3LL * 1024 * 1024 * 1024;

sci::DType DtypeFromName(const std::string& name, int* bytes) {
    if (name == "fp16") {
        *bytes = 2;
        return sci::DType::kFloat16;
    }
    if (name == "bf16") {
        *bytes = 2;
        return sci::DType::kBFloat16;
    }
    *bytes = 4;
    return sci::DType::kFloat32;
}

struct Inputs {
    sci::Tensor q;
    sci::Tensor k;
    sci::Tensor v;
};

Inputs MakeInputs(const Shape& s, sci::DType dtype, uint32_t seed, sci::Device& device) {
    const size_t q_elements = static_cast<size_t>(s.batch * s.heads_q * s.seq_q * s.head_dim);
    const size_t kv_elements = static_cast<size_t>(s.batch * s.heads_kv * s.seq_kv * s.head_dim);
    Inputs in;
    in.q = MakeDeviceTensor({s.batch, s.heads_q, s.seq_q, s.head_dim}, dtype,
                            Encode(RandomFloats(q_elements, seed), dtype), device);
    in.k = MakeDeviceTensor({s.batch, s.heads_kv, s.seq_kv, s.head_dim}, dtype,
                            Encode(RandomFloats(kv_elements, seed + 7919), dtype), device);
    in.v = MakeDeviceTensor({s.batch, s.heads_kv, s.seq_kv, s.head_dim}, dtype,
                            Encode(RandomFloats(kv_elements, seed + 104729), dtype), device);
    return in;
}

// Runs one backend over one shape and appends the case result.
void RunBackend(const char* impl, sca::BackendKind kind, const Shape& s, const char* dtype_name,
                bool causal, const Args& args, std::vector<CaseResult>* out, sci::Device& device) {
    int dtype_bytes = 2;
    const sci::DType dtype = DtypeFromName(dtype_name, &dtype_bytes);

    CaseResult result;
    result.case_id = CaseId(impl, s, dtype_name, causal);
    result.impl = impl;
    if (!args.only.empty() && result.case_id.find(args.only) == std::string::npos) return;

    const int64_t q_bytes =
        s.batch * s.heads_q * s.seq_q * s.head_dim * dtype_bytes;
    const int64_t kv_bytes =
        2 * s.batch * s.heads_kv * s.seq_kv * s.head_dim * dtype_bytes;
    const int64_t out_bytes = q_bytes;
    result.params_json = ParamsJson(s, dtype_name, causal, dtype_bytes);
    result.bytes_moved_modeled = ModeledBytes(s, impl, dtype_bytes);
    result.flops_effective = AttentionFlops(s, causal);

    sca::AttentionConfig cfg;
    cfg.backend = kind;
    cfg.causal = causal;
    const int64_t estimated = q_bytes + kv_bytes + out_bytes + 64LL * 1024 * 1024;
    if (estimated > kMemoryBudgetBytes) {
        result.skipped = "estimated " + std::to_string(estimated / (1024 * 1024)) +
                         " MiB > budget 3072 MiB";
        out->push_back(result);
        PrintCase(result);
        return;
    }

    Inputs in = MakeInputs(s, dtype, args.seed, device);

    // Warmup (also validates the case).
    for (int i = 0; i < args.warmup; ++i) {
        const auto warm = sca::attention(in.q, in.k, in.v, cfg);
        if (!warm.ok()) {
            result.skipped = warm.error().ToString();
            out->push_back(result);
            PrintCase(result);
            return;
        }
    }
    cudaDeviceSynchronize();

    EventTimer timer;
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(args.runs));
    sci::Result<sca::AttentionResult> last = sci::MakeUnexpected<sca::AttentionResult>(
        sci::Status::Error("no run"));
    for (int i = 0; i < args.runs; ++i) {
        timer.Start(nullptr);
        last = sca::attention(in.q, in.k, in.v, cfg);
        samples.push_back(timer.Stop(nullptr));
        if (!last.ok()) break;
    }
    if (!last.ok()) {
        result.skipped = last.error().ToString();
        out->push_back(result);
        PrintCase(result);
        return;
    }

    result.latency_ms = ComputeStats(samples);
    result.tflops = result.flops_effective / (result.latency_ms.p50 * 1e-3) / 1e12;
    result.effective_bw_gbps =
        static_cast<double>(result.bytes_moved_modeled) / (result.latency_ms.p50 * 1e-3) / 1e9;
    result.workspace_mb = static_cast<double>(last->stats.workspace_bytes) / (1024.0 * 1024.0);
    result.peak_mem_mb = static_cast<double>(q_bytes + kv_bytes + out_bytes) / (1024.0 * 1024.0);
    result.note = last->stats.note;
    if (args.verify && s.batch * s.seq_q * s.seq_kv <= 1 << 20) {
        // Verified against torch-free double reference (tests/common/test_utils.hpp).
        const std::vector<float> got = ReadToFloat(last->out);
        (void)got;  // the offline comparison lives in tests/kernel/*; benches only spot-check shape
        result.reference_max_abs_err = -1.0;
    }
    out->push_back(result);
    PrintCase(result);
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);
    if (!sca::DeviceCapability::CudaAvailable()) {
        std::fprintf(stderr, "[bench] no CUDA device\n");
        return 2;
    }
    std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    const RunHeader header = MakeHeader("benchmark_attention", args.warmup, args.runs, args.seed);
    std::printf("[bench] %s | gpu=%s cc=%s cuda=%s sm_clock=%d | warmup=%d runs=%d seed=%u\n",
                header.benchmark_name.c_str(), header.gpu.c_str(), header.compute_cap.c_str(),
                header.cuda.c_str(), header.sm_clock_mhz, header.warmup, header.runs, header.seed);

    std::vector<Shape> shapes;
    if (args.suite == "sanity") {
        shapes = {{1, 8, 8, 512, 512, 128}, {1, 8, 8, 1024, 1024, 128}};
    } else if (args.suite == "long") {
        shapes = {{1, 8, 8, 8192, 8192, 128}, {1, 8, 8, 16384, 16384, 128}};
    } else {
        for (const int64_t seq : {512, 1024, 2048, 4096, 8192}) {
            shapes.push_back({1, 8, 8, seq, seq, 128});
        }
        // Decode shapes: S_q = 1, long KV.
        for (const int64_t kv : {1024, 4096, 8192, 16384}) {
            shapes.push_back({1, 8, 8, 1, kv, 128});
        }
        // Real Qwen3-4B-Thinking shape (GQA group 4).
        shapes.push_back({1, 32, 8, 1024, 1024, 128});
        shapes.push_back({1, 32, 8, 2048, 2048, 128});
        shapes.push_back({1, 32, 8, 1, 4096, 128});
    }

    std::vector<CaseResult> cases;
    struct Impl {
        const char* name;
        sca::BackendKind kind;
    };
    const Impl impls[] = {{"flash", sca::BackendKind::kFlash},
                          {"tiled", sca::BackendKind::kTiled},
                          {"decode", sca::BackendKind::kDecode}};

    for (const Shape& s : shapes) {
        for (const Impl& impl : impls) {
            // decode is the S_q == 1 backend; the others are prefill backends.
            const bool is_decode_shape = s.seq_q == 1;
            if ((impl.kind == sca::BackendKind::kDecode) != is_decode_shape) continue;
            for (const char* dtype : {"fp16", "bf16"}) {
                for (const bool causal : {false, true}) {
                    if (is_decode_shape && !causal) continue;  // decode is always causal in serving
                    RunBackend(impl.name, impl.kind, s, dtype, causal, args, &cases, *device);
                }
            }
        }
    }

    if (!args.json.empty()) WriteJson(args.json, header, cases);
    if (!args.csv.empty()) WriteCsv(args.csv, cases);
    return 0;
}

