#pragma once

// STUB-mode stream: a handle-less, synchronous no-op.

#include "core/types.hpp"

namespace sci {

class Stream {
public:
    void* handle() const { return nullptr; }
    DeviceType device_type() const { return DeviceType::kCPU; }
    void synchronize() {}

    static Stream& GetCurrent() {
        static Stream stream;
        return stream;
    }
    static void SetCurrent(const Stream&) {}
};

}  // namespace sci

