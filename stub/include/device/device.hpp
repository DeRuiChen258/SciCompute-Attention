#pragma once

// STUB-mode device: host heap only. Exists so that the API layer links in a CPU-only build.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

#include "core/types.hpp"

namespace sci {

class Stream;  // defined in device/stream.hpp

class Device {
public:
    virtual ~Device() = default;
    virtual DeviceType type() const = 0;
    virtual int id() const { return 0; }
    virtual std::string name() const = 0;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void copy_to_device(void* dst, const void* src, size_t bytes) = 0;
    virtual void copy_to_host(void* dst, const void* src, size_t bytes) = 0;
    virtual void copy_within(void* dst, const void* src, size_t bytes) {
        std::memcpy(dst, src, bytes);
    }
    virtual void memset(void* ptr, int value, size_t bytes) = 0;
    virtual void synchronize() = 0;
    // Signature matches the upstream Device::copy_async(dst, src, bytes, Stream&).
    virtual void copy_async(void* dst, const void* src, size_t bytes, Stream& stream);
    virtual size_t total_memory() const { return 0; }
    virtual size_t free_memory() const { return 0; }
};

inline void Device::copy_async(void* dst, const void* src, size_t bytes, Stream&) {
    std::memcpy(dst, src, bytes);
}

// Process-wide host device used by STUB builds.
inline Device& StubHostDevice() {
    class HostDevice final : public Device {
    public:
        DeviceType type() const override { return DeviceType::kCPU; }
        std::string name() const override { return "stub-host"; }
        void* allocate(size_t bytes) override { return std::calloc(1, bytes == 0 ? 1 : bytes); }
        void deallocate(void* ptr) override { std::free(ptr); }
        void copy_to_device(void* dst, const void* src, size_t bytes) override {
            std::memcpy(dst, src, bytes);
        }
        void copy_to_host(void* dst, const void* src, size_t bytes) override {
            std::memcpy(dst, src, bytes);
        }
        void memset(void* ptr, int value, size_t bytes) override {
            std::memset(ptr, value, bytes);
        }
        void synchronize() override {}
    };
    static HostDevice device;
    return device;
}

}  // namespace sci
