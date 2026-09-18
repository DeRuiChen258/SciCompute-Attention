// Minimal C++ usage example.
//
// Flow: build Q/K/V on the device -> ask the dispatcher what it would do (ExplainAttention) ->
// execute with allow_fallback=false -> if the primary backend is not implemented yet, retry each
// backend explicitly so the example reports exactly which Level is available in this build.
//
// Exit codes: 0 = at least one backend produced a result, 2 = no CUDA device, 3 = no backend
// implemented yet (expected until the corresponding phase lands).

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "device/cuda_device.hpp"
#include "scicompute_attention/attention.hpp"

namespace {

float Filler(int64_t i) { return static_cast<float>((i % 17) - 8) * 0.0625f; }

// Minimal IEEE-754 binary32 -> binary16 conversion (round-to-nearest-even) so that this example
// stays free of CUDA headers; the kernels themselves use the CUDA intrinsics.
uint16_t ToHalf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);            // underflow -> signed zero
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);  // overflow -> inf
    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                          (mantissa >> 13));
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u) != 0u)) {
        half = static_cast<uint16_t>(half + 1);
    }
    return half;
}

sci::Tensor MakeFilledTensor(const std::vector<sci::index_t>& dims, sci::DType dtype,
                             sci::Device& device) {
    const sci::TensorShape shape(dims);
    sci::Tensor host(shape, dtype, device);
    std::vector<float> staging(static_cast<size_t>(shape.num_elements()));
    for (size_t i = 0; i < staging.size(); ++i) staging[i] = Filler(static_cast<int64_t>(i));
    // dtype conversion is delegated to the caller path in tests; the example only needs values.
    if (dtype == sci::DType::kFloat16 || dtype == sci::DType::kBFloat16) {
        std::vector<uint16_t> half_staging(staging.size());
        for (size_t i = 0; i < staging.size(); ++i) {
            uint32_t bits;
            std::memcpy(&bits, &staging[i], sizeof(bits));
            half_staging[i] = dtype == sci::DType::kFloat16
                                  ? ToHalf(staging[i])
                                  : static_cast<uint16_t>(bits >> 16);  // bf16 = high 16 bits
        }
        host.copy_from(half_staging.data(), half_staging.size() * sizeof(uint16_t));
    } else {
        host.copy_from(staging.data(), staging.size() * sizeof(float));
    }
    sci::Tensor device_tensor(shape, dtype, device);
    device_tensor.copy_from(host);
    return device_tensor;
}

std::string StatusLine(const sci::Status& status) {
    return status.ok() ? std::string("OK") : status.ToString();
}

}  // namespace

int main() {
    const sca::DeviceCapability& cap = sca::DeviceCapability::ForDevice(0);
    std::printf("[example_attention] device=%s cc=%d.%d sm=%d smem/block(optin)=%zu B\n",
                cap.device_name.c_str(), cap.major, cap.minor, cap.sm_count,
                cap.smem_per_block_optin);
    if (!sca::DeviceCapability::CudaAvailable()) {
        std::printf("[example_attention] no CUDA device available; nothing to run\n");
        return 2;
    }

    std::shared_ptr<sci::CudaDevice> device = sci::CudaDevice::Create(0);
    if (device == nullptr) {
        std::printf("[example_attention] CudaDevice::Create(0) failed\n");
        return 2;
    }

    // Real-model-like shape: B=1, H_q=32, H_kv=8 (GQA group 4), D=128, causal.
    const sci::index_t b = 1, hq = 32, hkv = 8, d = 128, sq = 256;
    const sci::TensorShape q_shape({b, hq, sq, d});
    const sci::TensorShape kv_shape({b, hkv, sq, d});

    sci::Tensor q = MakeFilledTensor({b, hq, sq, d}, sci::DType::kFloat16, *device);
    sci::Tensor k = MakeFilledTensor({b, hkv, sq, d}, sci::DType::kFloat16, *device);
    sci::Tensor v = MakeFilledTensor({b, hkv, sq, d}, sci::DType::kFloat16, *device);
    (void)q_shape;
    (void)kv_shape;

    sca::AttentionConfig cfg;
    cfg.causal = true;
    cfg.layout = sca::AttnLayout::kBHSD;

    std::printf("[example_attention] explain: %s\n",
                sca::ExplainAttention(q, k, v, cfg).c_str());

    const sci::Result<sca::AttentionResult> auto_result = sca::attention(q, k, v, cfg);
    if (auto_result.ok()) {
        std::printf("[example_attention] auto backend=%s out=%lldx%lldx%lldx%lld lse_empty=%d\n",
                    sca::BackendName(auto_result->stats.used_backend),
                    static_cast<long long>(auto_result->out.dim(0)),
                    static_cast<long long>(auto_result->out.dim(1)),
                    static_cast<long long>(auto_result->out.dim(2)),
                    static_cast<long long>(auto_result->out.dim(3)),
                    auto_result->lse.num_elements() == 0 ? 1 : 0);
        return 0;
    }
    std::printf("[example_attention] auto path failed: %s\n", auto_result.error().ToString().c_str());

    const sca::BackendKind candidates[] = {sca::BackendKind::kFlash, sca::BackendKind::kDecode,
                                           sca::BackendKind::kTiled, sca::BackendKind::kNaive};
    for (const sca::BackendKind kind : candidates) {
        sca::AttentionConfig explicit_cfg = cfg;
        explicit_cfg.backend = kind;
        const sci::Result<sca::AttentionResult> result = sca::attention(q, k, v, explicit_cfg);
        std::printf("[example_attention] backend=%-6s -> %s\n", sca::BackendName(kind),
                    StatusLine(result.ok() ? sci::Status::Ok() : result.error()).c_str());
        if (result.ok()) {
            std::printf("[example_attention] first working backend: %s (workspace %zu B)\n",
                        sca::BackendName(kind), result->stats.workspace_bytes);
            return 0;
        }
    }
    std::printf("[example_attention] no backend is implemented yet in this build\n");
    return 3;
}
