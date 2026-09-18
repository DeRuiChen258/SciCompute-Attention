#include "scicompute_attention/status.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

#include "scicompute_attention/attention_types.hpp"
#include "scicompute_attention/version.hpp"

namespace sca {

const char* VersionString() noexcept { return SCI_ATTENTION_VERSION_STRING; }

const char* ToString(AttnStatusCode code) noexcept {
    switch (code) {
        case AttnStatusCode::kOk: return "OK";
        case AttnStatusCode::kUnsupportedHeadDim: return "UnsupportedHeadDim";
        case AttnStatusCode::kUnsupportedDtype: return "UnsupportedDtype";
        case AttnStatusCode::kUnsupportedLayout: return "UnsupportedLayout";
        case AttnStatusCode::kLayoutMismatch: return "LayoutMismatch";
        case AttnStatusCode::kShapeMismatch: return "ShapeMismatch";
        case AttnStatusCode::kSeqTooLong: return "SeqTooLong";
        case AttnStatusCode::kKVCapacityExceeded: return "KVCapacityExceeded";
        case AttnStatusCode::kWorkspaceExceeded: return "WorkspaceExceeded";
        case AttnStatusCode::kUnsupportedFeature: return "UnsupportedFeature";
        case AttnStatusCode::kDeviceCapability: return "DeviceCapability";
        default: return "UnknownAttnStatus";
    }
}

sci::StatusCode ToSciCode(AttnStatusCode code) noexcept {
    switch (code) {
        case AttnStatusCode::kOk: return sci::StatusCode::kOk;
        case AttnStatusCode::kUnsupportedHeadDim:
        case AttnStatusCode::kUnsupportedDtype:
        case AttnStatusCode::kUnsupportedLayout:
        case AttnStatusCode::kLayoutMismatch:
        case AttnStatusCode::kShapeMismatch: return sci::StatusCode::kInvalidArgument;
        case AttnStatusCode::kSeqTooLong: return sci::StatusCode::kInvalidArgument;
        case AttnStatusCode::kKVCapacityExceeded:
        case AttnStatusCode::kWorkspaceExceeded: return sci::StatusCode::kResourceExhausted;
        case AttnStatusCode::kUnsupportedFeature: return sci::StatusCode::kNotImplemented;
        case AttnStatusCode::kDeviceCapability: return sci::StatusCode::kInvalidOperation;
        default: return sci::StatusCode::kError;
    }
}

sci::Status MakeStatus(AttnStatusCode code, std::string detail) {
    const std::string message =
        "[SCA-" + std::to_string(static_cast<int32_t>(code)) + "] " + detail;
    return sci::Status(ToSciCode(code), message);
}

AttnStatusCode ExtractCode(const sci::Status& status) noexcept {
    const std::string& msg = status.message();
    const size_t pos = msg.find("[SCA-");
    if (pos == std::string::npos) return AttnStatusCode::kOk;
    const int value = std::atoi(msg.c_str() + pos + 5);
    return static_cast<AttnStatusCode>(value);
}

int SupportedHeadDimIndex(int64_t head_dim) noexcept {
    for (int64_t i = 0; i < kNumSupportedHeadDims; ++i) {
        if (kSupportedHeadDims[i] == head_dim) return static_cast<int>(i);
    }
    return -1;
}

const char* LayoutName(AttnLayout layout) noexcept {
    switch (layout) {
        case AttnLayout::kBHSD: return "bhsd";
        case AttnLayout::kBSHD: return "bshd";
        default: return "unknown-layout";
    }
}

const char* KvLayoutName(KvLayout layout) noexcept {
    switch (layout) {
        case KvLayout::kBlockMajor: return "block-major";
        default: return "unknown-kv-layout";
    }
}

const char* BackendName(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::kAuto: return "auto";
        case BackendKind::kNaive: return "naive";
        case BackendKind::kTiled: return "tiled";
        case BackendKind::kFlash: return "flash";
        case BackendKind::kDecode: return "decode";
        case BackendKind::kPaged: return "paged";
        default: return "unknown-backend";
    }
}

bool IsSupportedComputeDtype(sci::DType dtype) noexcept {
    return dtype == sci::DType::kFloat32 || dtype == sci::DType::kFloat16 ||
           dtype == sci::DType::kBFloat16;
}

int64_t DtypeSize(sci::DType dtype) noexcept { return static_cast<int64_t>(sci::kDTypeSize(dtype)); }

}  // namespace sca

