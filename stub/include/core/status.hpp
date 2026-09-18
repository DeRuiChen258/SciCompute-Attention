#pragma once

// STUB-mode replacement for SciComputeInfra's core/status.hpp: sci::Status + Result<T> with the
// same call sites used by the host layer (ok/code/message/error/message).

#include <cstdint>
#include <new>
#include <string>
#include <utility>

namespace sci {

enum class StatusCode : int32_t {
    kOk = 0,
    kError = 1,
    kNotFound = 2,
    kInvalidArgument = 4,
    kNotImplemented = 5,
    kOutOfMemory = 6,
    kResourceExhausted = 7,
    kInvalidOperation = 10,
    kInternal = 12,
    kCudaError = 100,
    kCudaNotAvailable = 101,
    kCudaOutOfMemory = 102,
    kDeviceError = 500,
};

constexpr const char* StatusCodeToString(StatusCode code) {
    switch (code) {
        case StatusCode::kOk: return "OK";
        case StatusCode::kError: return "Error";
        case StatusCode::kNotFound: return "Not Found";
        case StatusCode::kInvalidArgument: return "Invalid Argument";
        case StatusCode::kNotImplemented: return "Not Implemented";
        case StatusCode::kOutOfMemory: return "Out of Memory";
        case StatusCode::kResourceExhausted: return "Resource Exhausted";
        case StatusCode::kInvalidOperation: return "Invalid Operation";
        case StatusCode::kInternal: return "Internal Error";
        case StatusCode::kCudaError: return "CUDA Error";
        case StatusCode::kCudaNotAvailable: return "CUDA Not Available";
        case StatusCode::kCudaOutOfMemory: return "CUDA Out of Memory";
        case StatusCode::kDeviceError: return "Device Error";
        default: return "Unknown";
    }
}

class Status {
public:
    Status() = default;
    Status(StatusCode code, std::string message = {})
        : code_(code), message_(std::move(message)) {}

    static Status Ok() { return Status(); }
    static Status Error(const std::string& m = "Unknown error") { return Status(StatusCode::kError, m); }
    static Status InvalidArgument(const std::string& m = "Invalid argument") {
        return Status(StatusCode::kInvalidArgument, m);
    }
    static Status NotImplemented(const std::string& m = "Not implemented") {
        return Status(StatusCode::kNotImplemented, m);
    }
    static Status InvalidOperation(const std::string& m = "Invalid operation") {
        return Status(StatusCode::kInvalidOperation, m);
    }
    static Status OutOfMemory(const std::string& m = "Out of memory") {
        return Status(StatusCode::kOutOfMemory, m);
    }
    static Status CudaError(const std::string& m = "CUDA error") {
        return Status(StatusCode::kCudaError, m);
    }
    static Status CudaNotAvailable(const std::string& m = "CUDA not available") {
        return Status(StatusCode::kCudaNotAvailable, m);
    }
    static Status CudaOutOfMemory(const std::string& m = "CUDA out of memory") {
        return Status(StatusCode::kCudaOutOfMemory, m);
    }

    bool ok() const { return code_ == StatusCode::kOk; }
    StatusCode code() const { return code_; }
    const std::string& message() const { return message_; }
    explicit operator bool() const { return ok(); }
    bool operator==(const Status& o) const { return code_ == o.code_; }
    bool operator!=(const Status& o) const { return !(*this == o); }
    std::string ToString() const {
        return ok() ? std::string("OK")
                    : std::string(StatusCodeToString(code_)) + ": " + message_;
    }

private:
    StatusCode code_{StatusCode::kOk};
    std::string message_;
};

struct Unexpected {
    explicit Unexpected(Status s) : error_(std::move(s)) {}
    const Status& error() const { return error_; }
    Status error_;
};

template <typename T>
class Expected {
public:
    Expected() : has_value_(true) { new (&storage_) T(); }
    explicit Expected(T value) : has_value_(true) { new (&storage_) T(std::move(value)); }
    explicit Expected(Unexpected u) : has_value_(false) { new (&storage_) Status(u.error_); }
    Expected(const Expected& o) : has_value_(o.has_value_) {
        if (has_value_) new (&storage_) T(*o.Ptr());
        else new (&storage_) Status(*o.Err());
    }
    Expected& operator=(const Expected& o) {
        if (this != &o) {
            Destroy();
            has_value_ = o.has_value_;
            if (has_value_) new (&storage_) T(*o.Ptr());
            else new (&storage_) Status(*o.Err());
        }
        return *this;
    }
    ~Expected() { Destroy(); }

    bool ok() const noexcept { return has_value_; }
    explicit operator bool() const noexcept { return has_value_; }
    T& value() & { return *Ptr(); }
    const T& value() const& { return *Ptr(); }
    T* operator->() { return Ptr(); }
    const T* operator->() const { return Ptr(); }
    T& operator*() { return *Ptr(); }
    const T& operator*() const { return *Ptr(); }
    Status error() const { return has_value_ ? Status::Ok() : *Err(); }
    const std::string& message() const {
        static const std::string empty;
        return has_value_ ? empty : Err()->message();
    }

private:
    void Destroy() {
        if (has_value_) Ptr()->~T();
        else Err()->~Status();
    }
    T* Ptr() { return reinterpret_cast<T*>(&storage_); }
    const T* Ptr() const { return reinterpret_cast<const T*>(&storage_); }
    Status* Err() { return reinterpret_cast<Status*>(&storage_); }
    const Status* Err() const { return reinterpret_cast<const Status*>(&storage_); }

    alignas(T) alignas(Status) unsigned char storage_[sizeof(T) > sizeof(Status) ? sizeof(T)
                                                                                 : sizeof(Status)];
    bool has_value_{true};
};

template <typename T>
using Result = Expected<T>;

template <typename T>
Expected<T> MakeUnexpected(Status s) {
    return Expected<T>(Unexpected(std::move(s)));
}

template <typename T>
inline Result<T> Ok(T value) {
    return Result<T>(std::move(value));
}

}  // namespace sci

