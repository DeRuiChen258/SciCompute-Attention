#pragma once

#define SCI_ATTENTION_VERSION_MAJOR 0
#define SCI_ATTENTION_VERSION_MINOR 1
#define SCI_ATTENTION_VERSION_PATCH 0
#define SCI_ATTENTION_VERSION_STRING "0.1.0"

namespace sca {

// Human readable version of the library, e.g. "0.1.0".
const char* VersionString() noexcept;

}  // namespace sca

