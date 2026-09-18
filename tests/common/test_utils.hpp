// Shared helper for the C++ kernel tests.
//
// Design: the reference is computed on the host in double precision from the *quantized* inputs
// (the fp16/bf16 values cast back to double). Comparing against dequantized inputs is the only
// fair comparison for narrow dtypes and mirrors the rule in docs/model_integration.md §23.6.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "device/cuda_device.hpp"
#include "device/device.hpp"
#include "tensor/tensor.hpp"

namespace sca_test {

// ---------------------------------------------------------------------------------------------
// deterministic input generation (seed is part of every failure message)
// ---------------------------------------------------------------------------------------------
inline std::vector<float> RandomFloats(size_t count, uint32_t seed, float low = -1.0f,
                                       float high = 1.0f) {
    std::vector<float> out(count);
    uint32_t state = seed == 0 ? 1u : seed;
    for (size_t i = 0; i < count; ++i) {
        // xorshift32: deterministic across libstdc++ versions (unlike std::mt19937 with
        // std::uniform_real_distribution, whose implementation is unspecified).
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const float unit = static_cast<float>(state) / static_cast<float>(0xffffffffu);
        out[i] = low + unit * (high - low);
    }
    return out;
}

inline std::vector<float> RandomLogits(size_t count, uint32_t seed, float magnitude) {
    return RandomFloats(count, seed, -magnitude, magnitude);
}

// ---------------------------------------------------------------------------------------------
// float <-> fp16 / bf16 conversion (host side, no CUDA headers needed in the test translation
// units that only need to stage data)
// ---------------------------------------------------------------------------------------------
inline uint16_t FloatToHalfBits(float value) {
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
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u) != 0u)) {
        half = static_cast<uint16_t>(half + 1);
    }
    return half;
}

inline float HalfBitsToFloat(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    const uint32_t exponent = (half >> 10) & 0x1fu;
    const uint32_t mantissa = half & 0x3ffu;
    uint32_t bits = sign;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            // subnormal
            float value = std::ldexp(static_cast<float>(mantissa), -24);
            return sign ? -value : value;
        }
    } else if (exponent == 31) {
        bits |= 0x7f800000u | (mantissa << 13);
    } else {
        bits |= ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline uint16_t FloatToBf16Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    bits += rounding;
    return static_cast<uint16_t>(bits >> 16);
}

inline float Bf16BitsToFloat(uint16_t bf16) {
    const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Encodes a host float buffer into the storage representation of `dtype` and returns it as raw
// bytes; also returns the dequantized values (the reference domain).
struct Encoded {
    std::vector<uint8_t> bytes;
    std::vector<float> dequantized;
};

inline Encoded Encode(const std::vector<float>& values, sci::DType dtype) {
    Encoded out;
    out.dequantized.resize(values.size());
    if (dtype == sci::DType::kFloat32) {
        out.bytes.resize(values.size() * sizeof(float));
        std::memcpy(out.bytes.data(), values.data(), out.bytes.size());
        out.dequantized = values;
        return out;
    }
    out.bytes.resize(values.size() * sizeof(uint16_t));
    auto* as_u16 = reinterpret_cast<uint16_t*>(out.bytes.data());
    for (size_t i = 0; i < values.size(); ++i) {
        if (dtype == sci::DType::kFloat16) {
            as_u16[i] = FloatToHalfBits(values[i]);
            out.dequantized[i] = HalfBitsToFloat(as_u16[i]);
        } else {
            as_u16[i] = FloatToBf16Bits(values[i]);
            out.dequantized[i] = Bf16BitsToFloat(as_u16[i]);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// device tensor plumbing
// ---------------------------------------------------------------------------------------------
// Host-side device handle (CPU). Cross-device Tensor::copy_from goes through
// CopyAcrossDevices, so staging buffers must live on a real CPU device.
inline sci::Device& HostDevice() {
    static std::shared_ptr<sci::Device> device =
        sci::DeviceManager::Instance().get_device(sci::DeviceType::kCPU, 0);
    return *device;
}

inline sci::Tensor MakeDeviceTensorFromBytes(const std::vector<sci::index_t>& dims,
                                             sci::DType dtype, const std::vector<uint8_t>& bytes,
                                             sci::Device& device) {
    const sci::TensorShape shape(dims);
    sci::Tensor host(shape, dtype, HostDevice());
    host.copy_from(bytes.data(), bytes.size());
    sci::Tensor dev(shape, dtype, device);
    dev.copy_from(host);
    return dev;
}

inline sci::Tensor MakeDeviceTensor(const std::vector<sci::index_t>& dims, sci::DType dtype,
                                    const Encoded& encoded, sci::Device& device) {
    return MakeDeviceTensorFromBytes(dims, dtype, encoded.bytes, device);
}

// Uploads an int32 vector as a 1-D device tensor (for slot mappings / block id lists, which the KV
// kernels consume as device pointers).
inline sci::Tensor MakeDeviceInt32(const std::vector<int32_t>& values, sci::Device& device) {
    const sci::TensorShape shape({static_cast<sci::index_t>(values.size())});
    sci::Tensor host(shape, sci::DType::kInt32, HostDevice());
    host.copy_from(values.data(), values.size() * sizeof(int32_t));
    sci::Tensor dev(shape, sci::DType::kInt32, device);
    dev.copy_from(host);
    return dev;
}

// Reads back a device tensor (or any tensor) and dequantizes it to float.
inline std::vector<float> ReadToFloat(const sci::Tensor& tensor) {
    const size_t elements = static_cast<size_t>(tensor.num_elements());
    sci::Tensor host(tensor.shape(), tensor.dtype(), HostDevice());
    host.copy_from(tensor);
    HostDevice().synchronize();

    std::vector<float> out(elements);
    if (tensor.dtype() == sci::DType::kFloat32) {
        const auto* src = static_cast<const float*>(host.data());
        std::copy(src, src + elements, out.begin());
    } else {
        const auto* src = static_cast<const uint16_t*>(host.data());
        for (size_t i = 0; i < elements; ++i) {
            out[i] = tensor.dtype() == sci::DType::kFloat16 ? HalfBitsToFloat(src[i])
                                                            : Bf16BitsToFloat(src[i]);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// error statistics
// ---------------------------------------------------------------------------------------------
struct ErrorStats {
    double max_abs{0.0};
    double mean_abs{0.0};
};

template <typename TGot, typename TExpect>
inline ErrorStats Compare(const std::vector<TGot>& got, const std::vector<TExpect>& expect) {
    ErrorStats stats;
    double sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - expect[i]);
        stats.max_abs = std::max(stats.max_abs, diff);
        sum += diff;
    }
    stats.mean_abs = got.empty() ? 0.0 : sum / static_cast<double>(got.size());
    return stats;
}

// String used in every assertion message so a failing case is directly reproducible.
inline std::string CaseTag(const char* backend, sci::DType dtype, bool causal, int64_t batch,
                           int64_t heads_q, int64_t heads_kv, int64_t seq_q, int64_t seq_kv,
                           int64_t head_dim, uint32_t seed) {
    return std::string(backend) + " dtype=" + sci::kDTypeName(dtype) +
           " causal=" + (causal ? "1" : "0") + " B=" + std::to_string(batch) +
           " Hq=" + std::to_string(heads_q) + " Hkv=" + std::to_string(heads_kv) +
           " Sq=" + std::to_string(seq_q) + " Skv=" + std::to_string(seq_kv) +
           " D=" + std::to_string(head_dim) + " seed=" + std::to_string(seed) +
           " (use tools/gen_qkv.py to regenerate the same inputs)";
}

// ---------------------------------------------------------------------------------------------
// reference implementation (double precision, host)
// ---------------------------------------------------------------------------------------------
// Bottom-right aligned causal masking: j is visible from i when j <= i + (seq_kv - seq_q).
inline std::vector<double> ReferenceAttention(const std::vector<float>& q,
                                              const std::vector<float>& k,
                                              const std::vector<float>& v, int64_t batch,
                                              int64_t heads_q, int64_t heads_kv, int64_t seq_q,
                                              int64_t seq_kv, int64_t head_dim, bool causal) {
    const int64_t group = heads_q / heads_kv;
    const int64_t diagonal = causal ? (seq_kv - seq_q) : 0;
    std::vector<double> out(static_cast<size_t>(batch * heads_q * seq_q * head_dim), 0.0);
    std::vector<double> scores(static_cast<size_t>(seq_kv));

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads_q; ++h) {
            const int64_t h_kv = h / group;
            for (int64_t i = 0; i < seq_q; ++i) {
                double m = -std::numeric_limits<double>::infinity();
                for (int64_t j = 0; j < seq_kv; ++j) {
                    if (causal && j > i + diagonal) {
                        scores[j] = -std::numeric_limits<double>::infinity();
                        continue;
                    }
                    double dot = 0.0;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        const size_t qi =
                            static_cast<size_t>((((b * heads_q + h) * seq_q) + i) * head_dim + d);
                        const size_t ki =
                            static_cast<size_t>((((b * heads_kv + h_kv) * seq_kv) + j) * head_dim +
                                                d);
                        dot += static_cast<double>(q[qi]) * static_cast<double>(k[ki]);
                    }
                    dot /= std::sqrt(static_cast<double>(head_dim));
                    scores[j] = dot;
                    m = std::max(m, dot);
                }
                double sum = 0.0;
                for (int64_t j = 0; j < seq_kv; ++j) {
                    if (std::isinf(scores[j])) continue;
                    sum += std::exp(scores[j] - m);
                }
                for (int64_t d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (int64_t j = 0; j < seq_kv; ++j) {
                        if (std::isinf(scores[j])) continue;
                        const double p = std::exp(scores[j] - m) / sum;
                        const size_t vi =
                            static_cast<size_t>((((b * heads_kv + h_kv) * seq_kv) + j) * head_dim +
                                                d);
                        acc += p * static_cast<double>(v[vi]);
                    }
                    const size_t oi =
                        static_cast<size_t>((((b * heads_q + h) * seq_q) + i) * head_dim + d);
                    out[oi] = acc;
                }
            }
        }
    }
    return out;
}

// Tolerance table (prompt §11.4 / docs/numerical_stability.md).
struct Tolerance {
    double max_abs;
    double mean_abs;
};

inline Tolerance ToleranceFor(sci::DType dtype, bool against_fp32_reference) {
    if (dtype == sci::DType::kFloat32) {
        return against_fp32_reference ? Tolerance{1e-4, 1e-6} : Tolerance{1e-4, 1e-6};
    }
    if (dtype == sci::DType::kFloat16) return Tolerance{5e-3, 5e-4};
    return Tolerance{2e-2, 2e-3};  // bf16
}

}  // namespace sca_test
