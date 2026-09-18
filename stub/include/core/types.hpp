#pragma once

// STUB-mode replacement for SciComputeInfra's core/types.hpp. It implements exactly the subset of
// the upstream API that the SciCompute-Attention *host* layer uses, so that the API surface and
// the CPU unit tests stay compilable without CUDA. No kernel, benchmark or performance conclusion
// may be derived from a STUB build.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sci {

using index_t = int64_t;

enum class DType : uint8_t {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
    kInt8 = 3,
    kInt32 = 4,
    kInt64 = 5,
    kBool = 6,
    kFloat64 = 7,
    kUInt8 = 8,
};

constexpr size_t kDTypeSize(DType dtype) {
    switch (dtype) {
        case DType::kFloat32: return 4;
        case DType::kFloat16: return 2;
        case DType::kBFloat16: return 2;
        case DType::kInt8: return 1;
        case DType::kInt32: return 4;
        case DType::kInt64: return 8;
        case DType::kBool: return 1;
        case DType::kFloat64: return 8;
        case DType::kUInt8: return 1;
        default: return 0;
    }
}

constexpr const char* kDTypeName(DType dtype) {
    switch (dtype) {
        case DType::kFloat32: return "float32";
        case DType::kFloat16: return "float16";
        case DType::kBFloat16: return "bfloat16";
        case DType::kInt8: return "int8";
        case DType::kInt32: return "int32";
        case DType::kInt64: return "int64";
        case DType::kBool: return "bool";
        case DType::kFloat64: return "float64";
        case DType::kUInt8: return "uint8";
        default: return "unknown";
    }
}

enum class DeviceType : uint8_t { kCPU = 0, kCUDA = 1, kCPUPinned = 2 };

constexpr const char* kDeviceTypeName(DeviceType type) {
    switch (type) {
        case DeviceType::kCPU: return "cpu";
        case DeviceType::kCUDA: return "cuda";
        case DeviceType::kCPUPinned: return "cpu_pinned";
        default: return "unknown";
    }
}

enum class Layout : uint8_t { kContiguous = 0, kStrided = 1 };

class TensorShape {
public:
    TensorShape() = default;

    TensorShape(std::initializer_list<index_t> dims) {
        for (index_t d : dims) {
            if (ndims_ < 8) dims_[ndims_++] = d;
        }
    }

    explicit TensorShape(const std::vector<index_t>& dims) {
        for (index_t d : dims) {
            if (ndims_ < 8) dims_[ndims_++] = d;
        }
    }

    index_t ndims() const { return ndims_; }
    index_t dim(size_t i) const { return dims_[i]; }
    index_t& dim(size_t i) { return dims_[i]; }
    index_t num_elements() const {
        index_t n = ndims_ == 0 ? 0 : 1;
        for (index_t i = 0; i < ndims_; ++i) n *= dims_[i];
        return n;
    }

private:
    index_t dims_[8]{};
    index_t ndims_{0};
};

}  // namespace sci

