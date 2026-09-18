#pragma once

#include <cstddef>
#include <memory>

#include "core/status.hpp"
#include "scicompute_attention/export.hpp"

namespace sci {
class Device;
}  // namespace sci

namespace sca {

// Pre-allocated, reusable scratch memory (prompt §9.4).
//
//   * one allocation per Create() call, 256 B aligned slices;
//   * no cudaMalloc / cudaFree on the inference path;
//   * Slice() records (offset, bytes) so out-of-range requests are detected instead of silently
//     aliasing.
class SCI_ATTENTION_API Workspace {
public:
    Workspace();
    Workspace(const Workspace&) = default;
    Workspace& operator=(const Workspace&) = default;
    ~Workspace();

    static sci::Result<Workspace> Create(size_t bytes, sci::Device& device);

    void* Data() noexcept;
    const void* Data() const noexcept;
    size_t Bytes() const noexcept;
    bool Valid() const noexcept;

    // Borrows a 256 B aligned region. Returns kWorkspaceExceeded (with the required size) when
    // the request does not fit.
    sci::Result<void*> Slice(size_t offset, size_t bytes, size_t align = 256);

    // Logical reset: slices become invalid again, memory is not released.
    sci::Status Reset();

    size_t PeakBytes() const noexcept;
    size_t AllocatedSlices() const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace sca

