#pragma once

// STUB-mode tensor: host memory, contiguous only. Semantics intentionally minimal.

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "core/status.hpp"
#include "core/types.hpp"
#include "device/device.hpp"

namespace sci {

class Tensor {
public:
    Tensor() = default;

    Tensor(const TensorShape& shape, DType dtype, Device& device)
        : shape_(shape), dtype_(dtype), device_(&device) {
        const size_t bytes = static_cast<size_t>(shape.num_elements()) * kDTypeSize(dtype);
        data_ = device.allocate(bytes);
        owns_ = true;
        bytes_ = bytes;
    }

    Tensor(const TensorShape& shape, DType dtype, Device& device, void* external_data,
           bool owns_data = false)
        : shape_(shape), dtype_(dtype), device_(&device), data_(external_data), owns_(owns_data) {
        bytes_ = static_cast<size_t>(shape.num_elements()) * kDTypeSize(dtype);
    }

    ~Tensor() {
        if (owns_ && data_ != nullptr && device_ != nullptr) device_->deallocate(data_);
    }

    Tensor(Tensor&& o) noexcept { MoveFrom(std::move(o)); }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            Release();
            MoveFrom(std::move(o));
        }
        return *this;
    }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    const TensorShape& shape() const { return shape_; }
    index_t ndims() const { return shape_.ndims(); }
    index_t dim(size_t i) const { return shape_.dim(i); }
    index_t num_elements() const { return shape_.num_elements(); }
    DType dtype() const { return dtype_; }
    Device& device() const { return *device_; }
    DeviceType device_type() const { return device_->type(); }
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t num_bytes() const { return bytes_; }
    bool is_contiguous() const { return true; }

    // Minimal copy surface used by the KV cache host code in STUB builds.
    void copy_from(const void* source, size_t bytes) {
        if (data_ != nullptr && source != nullptr) std::memcpy(data_, source, bytes);
    }
    void copy_from(const Tensor& other) {
        if (data_ != nullptr && other.data_ != nullptr) {
            std::memcpy(data_, other.data_, std::min(bytes_, other.bytes_));
        }
    }

private:
    void Release() {
        if (owns_ && data_ != nullptr && device_ != nullptr) device_->deallocate(data_);
        data_ = nullptr;
        owns_ = false;
    }
    void MoveFrom(Tensor&& o) {
        shape_ = o.shape_;
        dtype_ = o.dtype_;
        device_ = o.device_;
        data_ = o.data_;
        owns_ = o.owns_;
        bytes_ = o.bytes_;
        o.data_ = nullptr;
        o.owns_ = false;
    }

    TensorShape shape_{};
    DType dtype_{DType::kFloat32};
    Device* device_{nullptr};
    void* data_{nullptr};
    bool owns_{false};
    size_t bytes_{0};
};

}  // namespace sci
