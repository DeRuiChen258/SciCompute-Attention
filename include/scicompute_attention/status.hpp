#pragma once

#include <cstdint>
#include <string>

#include "core/status.hpp"  // upstream: sci::Status / sci::StatusCode / sci::Result
#include "scicompute_attention/export.hpp"

namespace sca {

// Attention-domain error codes.
//
// The upstream enum sci::StatusCode must not be extended (reuse rule, prompt §2.3). Error codes
// in this range are therefore carried inside the message as "[SCA-<code>]" while the underlying
// sci::StatusCode keeps the nearest general meaning, so callers can branch on either.
enum class AttnStatusCode : int32_t {
    kOk = 0,
    kUnsupportedHeadDim = 2001,
    kUnsupportedDtype = 2002,
    kUnsupportedLayout = 2003,
    kLayoutMismatch = 2004,
    kShapeMismatch = 2005,
    kSeqTooLong = 2006,
    kKVCapacityExceeded = 2007,
    kWorkspaceExceeded = 2008,
    kUnsupportedFeature = 2009,
    kDeviceCapability = 2010,
};

SCI_ATTENTION_API const char* ToString(AttnStatusCode code) noexcept;

// Maps the attention code onto the nearest upstream code (never returns an unknown value).
SCI_ATTENTION_API sci::StatusCode ToSciCode(AttnStatusCode code) noexcept;

// Builds a sci::Status carrying "[SCA-<code>] <detail>".
SCI_ATTENTION_API sci::Status MakeStatus(AttnStatusCode code, std::string detail);

// Extracts the sca error code from a status produced by MakeStatus (kOk when absent).
SCI_ATTENTION_API AttnStatusCode ExtractCode(const sci::Status& status) noexcept;

inline bool IsOk(const sci::Status& status) noexcept { return status.ok(); }

}  // namespace sca

