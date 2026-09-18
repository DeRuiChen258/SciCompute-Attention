#include "scicompute_attention/workspace.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "tensor/tensor.hpp"

#include "scicompute_attention/status.hpp"

namespace sca {

struct Workspace::Impl {
    sci::Tensor buffer;
    size_t bytes{0};
    size_t peak_bytes{0};
    std::vector<std::pair<size_t, size_t>> slices;  // (offset, end)
};

Workspace::Workspace() : impl_(std::make_shared<Impl>()) {}
Workspace::~Workspace() = default;

sci::Result<Workspace> Workspace::Create(size_t bytes, sci::Device& device) {
    if (bytes == 0) {
        return sci::MakeUnexpected<Workspace>(
            MakeStatus(AttnStatusCode::kWorkspaceExceeded, "workspace bytes=0"));
    }
    Workspace ws;
    ws.impl_->buffer =
        sci::Tensor(sci::TensorShape({static_cast<sci::index_t>(bytes)}), sci::DType::kUInt8,
                    device);
    if (ws.impl_->buffer.data() == nullptr) {
        return sci::MakeUnexpected<Workspace>(
            MakeStatus(AttnStatusCode::kWorkspaceExceeded,
                       "allocation of " + std::to_string(bytes) + " B failed"));
    }
    ws.impl_->bytes = bytes;
    return sci::Ok(std::move(ws));
}

void* Workspace::Data() noexcept { return impl_->buffer.data(); }

const void* Workspace::Data() const noexcept { return impl_->buffer.data(); }

size_t Workspace::Bytes() const noexcept { return impl_->bytes; }

bool Workspace::Valid() const noexcept { return impl_ != nullptr && impl_->buffer.data() != nullptr; }

size_t Workspace::PeakBytes() const noexcept { return impl_->peak_bytes; }

size_t Workspace::AllocatedSlices() const noexcept { return impl_->slices.size(); }

sci::Result<void*> Workspace::Slice(size_t offset, size_t bytes, size_t align) {
    if (!Valid()) {
        return sci::MakeUnexpected<void*>(
            MakeStatus(AttnStatusCode::kWorkspaceExceeded, "workspace is not initialised"));
    }
    if (bytes == 0) {
        return sci::MakeUnexpected<void*>(
            MakeStatus(AttnStatusCode::kWorkspaceExceeded, "slice bytes=0"));
    }
    if (align == 0 || offset % align != 0) {
        return sci::MakeUnexpected<void*>(MakeStatus(
            AttnStatusCode::kWorkspaceExceeded,
            "slice offset=" + std::to_string(offset) + " is not " + std::to_string(align) +
                " B aligned"));
    }
    if (offset > impl_->bytes || bytes > impl_->bytes - offset) {
        return sci::MakeUnexpected<void*>(MakeStatus(
            AttnStatusCode::kWorkspaceExceeded,
            "slice [" + std::to_string(offset) + ", " + std::to_string(offset + bytes) +
                ") exceeds workspace bytes=" + std::to_string(impl_->bytes)));
    }

    const size_t begin = offset;
    const size_t end = offset + bytes;
    bool overlaps = false;
    bool identical = false;
    for (const auto& s : impl_->slices) {
        if (begin == s.first && end == s.second) identical = true;
        const bool disjoint = end <= s.first || begin >= s.second;
        if (!disjoint) overlaps = true;
    }
    if (overlaps && !identical) {
        return sci::MakeUnexpected<void*>(MakeStatus(
            AttnStatusCode::kWorkspaceExceeded,
            "slice [" + std::to_string(begin) + ", " + std::to_string(end) +
                ") partially overlaps an existing slice (either reuse it exactly or Reset())"));
    }
    if (!identical) impl_->slices.emplace_back(begin, end);
    impl_->peak_bytes = std::max(impl_->peak_bytes, end);

    auto* base = static_cast<unsigned char*>(impl_->buffer.data());
    return sci::Ok(static_cast<void*>(base + offset));
}

sci::Status Workspace::Reset() {
    if (!Valid()) return MakeStatus(AttnStatusCode::kWorkspaceExceeded, "workspace is not initialised");
    impl_->slices.clear();
    return sci::Status::Ok();
}

}  // namespace sca

