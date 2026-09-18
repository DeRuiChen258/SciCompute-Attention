// Level 0 benchmark: the IO baseline.
//
// Two measurements per shape:
//   kernel : events wrap the three naive kernels only (buffers pre-allocated) - comparable with
//            the flash/decode kernel numbers in the other benchmark targets;
//   api    : events wrap sca::naive_attention(), i.e. including the (contract-mandated) output
//            allocation and the validation/dispatch path. The difference is the wrapper overhead
//            reported in docs/results/benchmark_report.md.

#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "backends/naive/attention_naive.cuh"
#include "common/bench_utils.hpp"
#include "device/cuda_device.hpp"
#include "device/device.hpp"
#include "runtime/scratch_pool.hpp"
#include "scicompute_attention/attention.hpp"
#include "tests/common/test_utils.hpp"

namespace {

using namespace sca::bench;

constexpr int64_t kMemoryBudgetBytes = 3LL * 1024 * 1024 * 1024;  // 3 GiB guard (prompt §12.3)

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

std::string TileJson(sca::TileConfig tile) {
    return "{\"m\": " + std::to_string(tile.block_m) + ", \"n\": " + std::to_string(tile.block_n) +
           ", \"warps\": " + std::to_string(tile.warps) +
           ", \"stages\": " + std::to_string(tile.stages) + "}";
}

struct Inputs {
    sci::Tensor q;
    sci::Tensor k;
    sci::Tensor v;
    std::vector<float> q_ref;
    std::vector<float> k_ref;
    std::vector<float> v_ref;
};

Inputs MakeInputs(const Shape& s, sci::DType dtype, uint32_t seed, sci::Device& device) {
    const size_t q_elements =
        static_cast<size_t>(s.batch * s.heads_q * s.seq_q * s.head_dim);
    const size_t kv_elements =
        static_cast<size_t>(s.batch * s.heads_kv * s.seq_kv * s.head_dim);
    const sca_test::Encoded q_enc = sca_test::Encode(sca_test::RandomFloats(q_elements, seed), dtype);
    const sca_test::Encoded k_enc =
        sca_test::Encode(sca_test::RandomFloats(kv_elements, seed + 7919), dtype);
    const sca_test::Encoded v_enc =
        sca_test::Encode(sca_test::RandomFloats(kv_elements, seed + 104729), dtype);

    Inputs in;
    in.q = sca_test::MakeDeviceTensor({s.batch, s.heads_q, s.seq_q, s.head_dim}, dtype, q_enc,
                                      device);
    in.k = sca_test::MakeDeviceTensor({s.batch, s.heads_kv, s.seq_kv, s.head_dim}, dtype, k_enc,
                                      device);
    in.v = sca_test::MakeDeviceTensor({s.batch, s.heads_kv, s.seq_kv, s.head_dim}, dtype, v_enc,
                                      device);
    in.q_ref = q_enc.dequantized;
    in.k_ref = k_enc.dequantized;
    in.v_ref = v_enc.dequantized;
    return in;
}

void RunKernelOnly(const Shape& s, const char* dtype_name, bool causal, uint32_t seed,
                   const Args& args, std::vector<CaseResult>* out, sci::Device& device) {
    int dtype_bytes = 2;
    const sci::DType dtype = DtypeFromName(dtype_name, &dtype_bytes);
    CaseResult result;
    result.case_id = CaseId("naive", s, dtype_name, causal);
    result.impl = "naive";
    result.params_json = ParamsJson(s, dtype_name, causal, dtype_bytes, TileJson(sca::TileConfig{})
                                                                        .c_str());
    if (!args.only.empty() && result.case_id.find(args.only) == std::string::npos) return;

    const int64_t score_elements = s.batch * s.heads_q * s.seq_q * s.seq_kv;
    const int64_t score_bytes = score_elements * 4;
    const int64_t total_estimate = score_bytes * 2 + (s.batch * s.heads_q * s.seq_q * s.head_dim +
                                                     2 * s.batch * s.heads_kv * s.seq_kv *
                                                         s.head_dim) *
                                                         dtype_bytes;
    if (total_estimate > kMemoryBudgetBytes) {
        result.skipped = "estimated " + std::to_string(total_estimate / (1024 * 1024)) +
                         " MiB > budget 3072 MiB";
        out->push_back(result);
        PrintCase(result);
        return;
    }

    Inputs in = MakeInputs(s, dtype, seed, device);
    sci::Tensor out_tensor(
        sci::TensorShape({s.batch, s.heads_q, s.seq_q, s.head_dim}), dtype, device);
    const sci::Result<float*> scores_result =
        sca::runtime::AcquireScratchFp32(sca::runtime::ScratchKind::kScoresFp32,
                                         static_cast<size_t>(score_elements), device);
    if (!scores_result.ok()) {
        result.skipped = scores_result.error().ToString();
        out->push_back(result);
        PrintCase(result);
        return;
    }

    sca::cuda::NaiveFwdParams params;
    params.q = in.q.data();
    params.k = in.k.data();
    params.v = in.v.data();
    params.out = out_tensor.data();
    params.scores = *scores_result;
    params.batch = s.batch;
    params.num_heads = s.heads_q;
    params.num_kv_heads = s.heads_kv;
    params.seq_q = s.seq_q;
    params.seq_kv = s.seq_kv;
    params.head_dim = s.head_dim;
    params.q_stride = {s.heads_q * s.seq_q * s.head_dim, s.seq_q * s.head_dim, s.head_dim};
    params.k_stride = params.v_stride = {s.heads_kv * s.seq_kv * s.head_dim,
                                         s.seq_kv * s.head_dim, s.head_dim};
    params.o_stride = params.q_stride;
    params.scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
    params.causal = causal;
    params.diagonal = causal ? (s.seq_kv - s.seq_q) : 0;
    params.dtype = dtype == sci::DType::kFloat32
                       ? sca::cuda::NaiveDtype::kFp32
                       : (dtype == sci::DType::kFloat16 ? sca::cuda::NaiveDtype::kFp16
                                                        : sca::cuda::NaiveDtype::kBf16);

    // Warmup.
    for (int i = 0; i < args.warmup; ++i) {
        sca::cuda::LaunchNaiveForward(params, nullptr);
    }
    cudaDeviceSynchronize();

    EventTimer timer;
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(args.runs));
    for (int i = 0; i < args.runs; ++i) {
        timer.Start(nullptr);
        sca::cuda::LaunchNaiveForward(params, nullptr);
        samples.push_back(timer.Stop(nullptr));
    }

    result.latency_ms = ComputeStats(samples);
    result.flops_effective = AttentionFlops(s, causal);
    result.tflops = result.flops_effective / (result.latency_ms.p50 * 1e-3) / 1e12;
    result.bytes_moved_modeled = ModeledBytes(s, "naive", dtype_bytes);
    result.effective_bw_gbps =
        static_cast<double>(result.bytes_moved_modeled) / (result.latency_ms.p50 * 1e-3) / 1e9;
    result.workspace_mb = static_cast<double>(score_bytes) / (1024.0 * 1024.0);
    result.peak_mem_mb = result.workspace_mb * 2;  // scores + one extra copy during softmax

    if (args.verify && s.seq_q * s.seq_kv <= 1024 * 1024) {
        const std::vector<float> got = sca_test::ReadToFloat(out_tensor);
        const std::vector<double> expect = sca_test::ReferenceAttention(
            in.q_ref, in.k_ref, in.v_ref, s.batch, s.heads_q, s.heads_kv, s.seq_q, s.seq_kv,
            s.head_dim, causal);
        result.reference_max_abs_err = sca_test::Compare(got, expect).max_abs;
    }
    out->push_back(result);
    PrintCase(result);
}

void RunApi(const Shape& s, const char* dtype_name, bool causal, uint32_t seed, const Args& args,
            std::vector<CaseResult>* out, sci::Device& device) {
    int dtype_bytes = 2;
    const sci::DType dtype = DtypeFromName(dtype_name, &dtype_bytes);
    CaseResult result;
    result.case_id = CaseId("naive_api", s, dtype_name, causal);
    result.impl = "naive_api";
    result.params_json = ParamsJson(s, dtype_name, causal, dtype_bytes);
    if (!args.only.empty() && result.case_id.find(args.only) == std::string::npos) return;

    const int64_t score_bytes = s.batch * s.heads_q * s.seq_q * s.seq_kv * 4;
    if (score_bytes * 2 > kMemoryBudgetBytes) {
        result.skipped = "estimated score storage exceeds 3 GiB budget";
        out->push_back(result);
        PrintCase(result);
        return;
    }

    Inputs in = MakeInputs(s, dtype, seed, device);
    sca::AttentionConfig cfg;
    cfg.backend = sca::BackendKind::kNaive;
    cfg.causal = causal;

    for (int i = 0; i < args.warmup; ++i) {
        const auto warm = sca::naive_attention(in.q, in.k, in.v, cfg);
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
    for (int i = 0; i < args.runs; ++i) {
        timer.Start(nullptr);
        const auto call = sca::naive_attention(in.q, in.k, in.v, cfg);
        const double ms = timer.Stop(nullptr);
        if (!call.ok()) {
            result.skipped = call.error().ToString();
            out->push_back(result);
            PrintCase(result);
            return;
        }
        samples.push_back(ms);
    }

    result.latency_ms = ComputeStats(samples);
    result.flops_effective = AttentionFlops(s, causal);
    result.tflops = result.flops_effective / (result.latency_ms.p50 * 1e-3) / 1e12;
    result.bytes_moved_modeled = ModeledBytes(s, "naive", dtype_bytes);
    result.effective_bw_gbps =
        static_cast<double>(result.bytes_moved_modeled) / (result.latency_ms.p50 * 1e-3) / 1e9;
    result.workspace_mb = static_cast<double>(score_bytes) / (1024.0 * 1024.0);
    result.note = "includes output allocation + validation/dispatch (API contract)";
    out->push_back(result);
    PrintCase(result);
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);
    if (!sca::DeviceCapability::CudaAvailable()) {
        std::fprintf(stderr, "[bench] no CUDA device; benchmark cannot run\n");
        return 2;
    }

    std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    const RunHeader header = MakeHeader("benchmark_naive", args.warmup, args.runs, args.seed);
    std::printf("[bench] %s | gpu=%s cc=%s cuda=%s sm_clock=%d MHz | warmup=%d runs=%d seed=%u\n",
                header.benchmark_name.c_str(), header.gpu.c_str(), header.compute_cap.c_str(),
                header.cuda.c_str(), header.sm_clock_mhz, header.warmup, header.runs, header.seed);

    std::vector<Shape> shapes;
    if (args.suite == "sanity") {
        shapes = {{1, 8, 8, 256, 256, 64}, {1, 8, 8, 256, 256, 128}};
    } else if (args.suite == "long") {
        shapes = {{1, 8, 8, 4096, 4096, 128}, {1, 8, 8, 8192, 8192, 64}};
    } else {
        for (const int64_t seq : {128, 256, 512, 1024, 2048, 4096}) {
            shapes.push_back({1, 8, 8, seq, seq, 64});
            shapes.push_back({1, 8, 8, seq, seq, 128});
        }
        shapes.push_back({1, 32, 8, 1024, 1024, 128});  // real Qwen3-4B shape (GQA group 4)
    }

    std::vector<CaseResult> cases;
    for (const Shape& s : shapes) {
        for (const char* dtype : {"fp16", "bf16"}) {
            for (const bool causal : {false, true}) {
                RunKernelOnly(s, dtype, causal, args.seed, args, &cases, *device);
                if (causal) RunApi(s, dtype, causal, args.seed, args, &cases, *device);
            }
        }
    }

    if (!args.json.empty()) WriteJson(args.json, header, cases);
    if (!args.csv.empty()) WriteCsv(args.csv, cases);
    std::printf("[bench] scratch high-water mark: %.1f MiB\n",
                static_cast<double>(sca::runtime::ScratchHighWaterMarkBytes()) / (1024.0 * 1024.0));
    return 0;
}

