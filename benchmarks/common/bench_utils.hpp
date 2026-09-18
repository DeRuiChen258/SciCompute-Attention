// Benchmark harness shared by every benchmark_*.cu target.
//
// Methodology (fixed by the prompt §12.1, never re-invented per report):
//   warmup 20 iterations, 100 measured iterations, CUDA-event timing around the pure kernel call,
//   mean/median/std/P50/P90/P95/P99, memory delta recorded, seed fixed, one process per suite.
//
// Output: JSON (field set from §12.4) + CSV, both bound to the git sha and the GPU state.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "scicompute_attention/capability.hpp"

namespace sca {
namespace bench {

// ---------------------------------------------------------------------------------------------
// latency statistics
// ---------------------------------------------------------------------------------------------
struct LatencyStats {
    double mean{0.0};
    double median{0.0};
    double std{0.0};
    double p50{0.0};
    double p90{0.0};
    double p95{0.0};
    double p99{0.0};
    double min{0.0};
    double max{0.0};
};

inline double Percentile(std::vector<double> sorted, double p) {
    if (sorted.empty()) return 0.0;
    std::sort(sorted.begin(), sorted.end());
    const double rank = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = rank - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

inline LatencyStats ComputeStats(const std::vector<double>& samples_ms) {
    LatencyStats stats;
    if (samples_ms.empty()) return stats;
    double sum = 0.0;
    for (const double v : samples_ms) sum += v;
    stats.mean = sum / static_cast<double>(samples_ms.size());
    double variance = 0.0;
    for (const double v : samples_ms) variance += (v - stats.mean) * (v - stats.mean);
    stats.std = std::sqrt(variance / static_cast<double>(samples_ms.size()));
    stats.median = Percentile(samples_ms, 0.5);
    stats.p50 = stats.median;
    stats.p90 = Percentile(samples_ms, 0.90);
    stats.p95 = Percentile(samples_ms, 0.95);
    stats.p99 = Percentile(samples_ms, 0.99);
    stats.min = *std::min_element(samples_ms.begin(), samples_ms.end());
    stats.max = *std::max_element(samples_ms.begin(), samples_ms.end());
    return stats;
}

// ---------------------------------------------------------------------------------------------
// CUDA-event timer
// ---------------------------------------------------------------------------------------------
class EventTimer {
public:
    EventTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void Start(cudaStream_t stream) { cudaEventRecord(start_, stream); }
    double Stop(cudaStream_t stream) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return static_cast<double>(ms);
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// ---------------------------------------------------------------------------------------------
// result bookkeeping
// ---------------------------------------------------------------------------------------------
struct CaseResult {
    std::string case_id;
    std::string impl;
    std::string params_json;   // {"B":1,"H":8,...}
    LatencyStats latency_ms;
    double flops_effective{0.0};
    double tflops{0.0};
    int64_t bytes_moved_modeled{0};
    double effective_bw_gbps{0.0};
    double peak_mem_mb{0.0};
    double workspace_mb{0.0};
    int registers_per_thread{-1};
    int64_t smem_bytes{0};
    double achieved_occupancy_pct{-1.0};
    double reference_max_abs_err{-1.0};
    std::string skipped;  // non-empty => the case was not run (memory guard or unsupported)
    std::string note;
};

struct RunHeader {
    std::string benchmark_name;
    std::string git_sha;
    std::string build_type;
    std::string gpu;
    std::string compute_cap;
    std::string driver;
    std::string cuda;
    int sm_clock_mhz{0};
    std::string timestamp;
    int warmup{20};
    int runs{100};
    uint32_t seed{1234};
};

// ---------------------------------------------------------------------------------------------
// FLOPs / traffic models (identical formula in docs/benchmarking.md and docs/roofline.md)
// ---------------------------------------------------------------------------------------------
struct Shape {
    int64_t batch{1};
    int64_t heads_q{8};
    int64_t heads_kv{8};
    int64_t seq_q{512};
    int64_t seq_kv{512};
    int64_t head_dim{128};
};

inline double AttentionFlops(const Shape& s, bool causal) {
    const double qk = 2.0 * static_cast<double>(s.batch) * static_cast<double>(s.heads_q) *
                      static_cast<double>(s.head_dim);
    const double pv = qk;
    if (!causal) {
        const double pairs = static_cast<double>(s.seq_q) * static_cast<double>(s.seq_kv);
        return (qk + pv) * pairs;
    }
    // Bottom-right aligned causal mask: row i sees min(Skv, i + Skv - Sq + 1) tokens.
    const int64_t diagonal = s.seq_kv - s.seq_q;
    double pairs = 0.0;
    for (int64_t i = 0; i < s.seq_q; ++i) {
        const int64_t visible = std::min<int64_t>(s.seq_kv, i + diagonal + 1);
        pairs += static_cast<double>(std::max<int64_t>(visible, 0));
    }
    pairs *= static_cast<double>(s.batch) * static_cast<double>(s.heads_q);
    return (qk + pv) * pairs;
}

// "Modeled" traffic = the minimum a perfect implementation of that algorithm must move.
inline int64_t ModeledBytes(const Shape& s, const char* impl, int64_t dtype_bytes) {
    const int64_t q_bytes = s.batch * s.heads_q * s.seq_q * s.head_dim * dtype_bytes;
    const int64_t kv_bytes = s.batch * s.heads_kv * s.seq_kv * s.head_dim * dtype_bytes;
    const int64_t out_bytes = q_bytes;
    const int64_t score_bytes = s.batch * s.heads_q * s.seq_q * s.seq_kv * 4;  // FP32 scores
    const std::string name(impl);
    if (name == "naive") {
        // Q,K,V read + S written + S read + P written + P read + O written
        return q_bytes + kv_bytes * 2 + out_bytes + 4 * score_bytes;
    }
    if (name == "tiled" || name == "flash") {
        // K/V are re-read per Q tile by construction; the Q-tile count is the reuse factor.
        return q_bytes + out_bytes + 2 * kv_bytes;
    }
    if (name == "decode" || name == "paged") {
        return q_bytes + 2 * kv_bytes + out_bytes;
    }
    return q_bytes + 2 * kv_bytes + out_bytes;
}

// ---------------------------------------------------------------------------------------------
// environment collection
// ---------------------------------------------------------------------------------------------
inline std::string RunCommand(const char* command) {
    std::string out;
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) return out;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) out += buffer;
    pclose(pipe);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

inline std::string TimestampNow() {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
    localtime_r(&now, &tm_buf);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S%z", &tm_buf);
    return buffer;
}

inline RunHeader MakeHeader(const char* benchmark_name, int warmup, int runs, uint32_t seed,
                            const char* build_type = "Release") {
    RunHeader header;
    header.benchmark_name = benchmark_name;
    header.warmup = warmup;
    header.runs = runs;
    header.seed = seed;
    header.timestamp = TimestampNow();
    header.git_sha = RunCommand("git -C " SCA_PROJECT_ROOT " rev-parse --short HEAD 2>/dev/null");
    if (header.git_sha.empty()) header.git_sha = "nogit";
    header.build_type = build_type;
    const DeviceCapability& cap = DeviceCapability::ForDevice(0);
    header.gpu = cap.device_name;
    header.compute_cap = std::to_string(cap.major) + "." + std::to_string(cap.minor);
    header.driver = RunCommand("nvidia-smi --query-gpu=driver_version --format=csv,noheader");
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        header.cuda = std::to_string(runtime_version / 1000) + "." +
                      std::to_string((runtime_version % 1000) / 10);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        header.driver += " (cuda driver api " + std::to_string(driver_version / 1000) + "." +
                         std::to_string((driver_version % 1000) / 10) + ")";
    }
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    header.sm_clock_mhz = clock_khz / 1000;
    return header;
}

// ---------------------------------------------------------------------------------------------
// exporters
// ---------------------------------------------------------------------------------------------
inline std::string EscapeJson(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

inline void WriteJson(const std::string& path, const RunHeader& header,
                      const std::vector<CaseResult>& cases) {
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "[bench] cannot write %s\n", path.c_str());
        return;
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"benchmark_name\": \"" << EscapeJson(header.benchmark_name) << "\",\n";
    out << "  \"git_sha\": \"" << EscapeJson(header.git_sha) << "\",\n";
    out << "  \"build_type\": \"" << EscapeJson(header.build_type) << "\",\n";
    out << "  \"gpu\": \"" << EscapeJson(header.gpu) << "\",\n";
    out << "  \"compute_cap\": \"" << EscapeJson(header.compute_cap) << "\",\n";
    out << "  \"driver\": \"" << EscapeJson(header.driver) << "\",\n";
    out << "  \"cuda\": \"" << EscapeJson(header.cuda) << "\",\n";
    out << "  \"sm_clock_mhz\": " << header.sm_clock_mhz << ",\n";
    out << "  \"timestamp\": \"" << EscapeJson(header.timestamp) << "\",\n";
    out << "  \"warmup\": " << header.warmup << ",\n";
    out << "  \"runs\": " << header.runs << ",\n";
    out << "  \"seed\": " << header.seed << ",\n";
    out << "  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const CaseResult& c = cases[i];
        out << "    {\n";
        out << "      \"case_id\": \"" << EscapeJson(c.case_id) << "\",\n";
        out << "      \"impl\": \"" << EscapeJson(c.impl) << "\",\n";
        out << "      \"params\": " << (c.params_json.empty() ? "{}" : c.params_json) << ",\n";
        if (!c.skipped.empty()) {
            out << "      \"skipped\": \"" << EscapeJson(c.skipped) << "\",\n";
        }
        out << "      \"latency_ms\": {\"mean\": " << c.latency_ms.mean
            << ", \"median\": " << c.latency_ms.median << ", \"std\": " << c.latency_ms.std
            << ", \"p50\": " << c.latency_ms.p50 << ", \"p90\": " << c.latency_ms.p90
            << ", \"p95\": " << c.latency_ms.p95 << ", \"p99\": " << c.latency_ms.p99 << "},\n";
        out << "      \"flops_effective\": " << c.flops_effective << ",\n";
        out << "      \"tflops\": " << c.tflops << ",\n";
        out << "      \"bytes_moved_modeled\": " << c.bytes_moved_modeled << ",\n";
        out << "      \"effective_bw_gbps\": " << c.effective_bw_gbps << ",\n";
        out << "      \"peak_mem_mb\": " << c.peak_mem_mb << ",\n";
        out << "      \"workspace_mb\": " << c.workspace_mb << ",\n";
        out << "      \"registers_per_thread\": " << c.registers_per_thread << ",\n";
        out << "      \"smem_bytes\": " << c.smem_bytes << ",\n";
        out << "      \"achieved_occupancy_pct\": " << c.achieved_occupancy_pct << ",\n";
        out << "      \"reference_max_abs_err\": " << c.reference_max_abs_err << ",\n";
        out << "      \"note\": \"" << EscapeJson(c.note) << "\"\n";
        out << "    }" << (i + 1 == cases.size() ? "" : ",") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("[bench] json written: %s\n", path.c_str());
}

inline void WriteCsv(const std::string& path, const std::vector<CaseResult>& cases) {
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "[bench] cannot write %s\n", path.c_str());
        return;
    }
    out << "name,case_id,warmup,runs,mean_ms,median_ms,std_ms,p50_ms,p90_ms,p95_ms,p99_ms,tflops,"
           "effective_bw_gbps,peak_mem_mb,unit\n";
    out << std::fixed << std::setprecision(6);
    for (const CaseResult& c : cases) {
        if (!c.skipped.empty()) continue;
        out << c.impl << "," << c.case_id << ",20,100," << c.latency_ms.mean << ","
            << c.latency_ms.median << "," << c.latency_ms.std << "," << c.latency_ms.p50 << ","
            << c.latency_ms.p90 << "," << c.latency_ms.p95 << "," << c.latency_ms.p99 << ","
            << c.tflops << "," << c.effective_bw_gbps << "," << c.peak_mem_mb << ",ms\n";
    }
    std::printf("[bench] csv written: %s\n", path.c_str());
}

// ---------------------------------------------------------------------------------------------
// argument parsing
// ---------------------------------------------------------------------------------------------
struct Args {
    std::string suite{"main"};
    std::string only;
    std::string json;
    std::string csv;
    int runs{100};
    int warmup{20};
    uint32_t seed{1234};
    bool verify{false};
    bool allow_large_kv{false};
};

inline Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (key == "--suite") args.suite = next();
        else if (key == "--only") args.only = next();
        else if (key == "--json") args.json = next();
        else if (key == "--csv") args.csv = next();
        else if (key == "--runs") args.runs = std::atoi(next().c_str());
        else if (key == "--warmup") args.warmup = std::atoi(next().c_str());
        else if (key == "--seed") args.seed = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (key == "--verify") args.verify = true;
        else if (key == "--allow-large-kv") args.allow_large_kv = true;
        else if (key == "--help") {
            std::printf("usage: %s [--suite sanity|main|long] [--only id] [--runs N] [--warmup N] "
                        "[--seed N] [--verify] [--json path] [--csv path]\n", argv[0]);
            std::exit(0);
        }
    }
    return args;
}

inline std::string CaseId(const char* impl, const Shape& s, const char* dtype, bool causal) {
    return std::string(impl) + "_B" + std::to_string(s.batch) + "_Hq" + std::to_string(s.heads_q) +
           "_Hkv" + std::to_string(s.heads_kv) + "_Sq" + std::to_string(s.seq_q) + "_Skv" +
           std::to_string(s.seq_kv) + "_D" + std::to_string(s.head_dim) + "_" + dtype +
           "_causal" + (causal ? "1" : "0");
}

inline std::string ParamsJson(const Shape& s, const char* dtype, bool causal, int dtype_bytes,
                              const char* tile = nullptr) {
    std::ostringstream out;
    out << "{\"B\": " << s.batch << ", \"H\": " << s.heads_q << ", \"Hkv\": " << s.heads_kv
        << ", \"Sq\": " << s.seq_q << ", \"Skv\": " << s.seq_kv << ", \"D\": " << s.head_dim
        << ", \"dtype\": \"" << dtype << "\", \"dtype_bytes\": " << dtype_bytes
        << ", \"causal\": " << (causal ? "true" : "false");
    if (tile != nullptr) out << ", \"tile\": " << tile;
    out << "}";
    return out.str();
}

inline void PrintCase(const CaseResult& c) {
    if (!c.skipped.empty()) {
        std::printf("[bench] %-60s SKIPPED (%s)\n", c.case_id.c_str(), c.skipped.c_str());
        return;
    }
    std::printf("[bench] %-60s p50=%8.4f ms p99=%8.4f ms %8.3f TFLOPS %8.2f GB/s\n",
                c.case_id.c_str(), c.latency_ms.p50, c.latency_ms.p99, c.tflops,
                c.effective_bw_gbps);
}

}  // namespace bench
}  // namespace sca
