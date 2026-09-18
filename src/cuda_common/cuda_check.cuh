// CUDA error handling for device translation units (prompt §7.1).
//
// Every CUDA call and every kernel launch must be checked. Failures are converted into
// sci::Status instead of thrown exceptions.
#pragma once

#include <cuda_runtime.h>

#include <string>

#include "core/status.hpp"
#include "scicompute_attention/status.hpp"

namespace sca {
namespace cuda {

inline sci::Status CudaStatus(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return sci::Status::Ok();
    const std::string message =
        std::string(what) + ": " + cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ")";
    if (err == cudaErrorMemoryAllocation) return sci::Status::CudaOutOfMemory(message);
    if (err == cudaErrorInvalidDevice || err == cudaErrorNoDevice) {
        return sci::Status::CudaNotAvailable(message);
    }
    return sci::Status::CudaError(message);
}

}  // namespace cuda
}  // namespace sca

#define SCI_CUDA_CHECK(expr)                                                                  \
    do {                                                                                      \
        const cudaError_t sca_err__ = (expr);                                                  \
        if (sca_err__ != cudaSuccess) {                                                        \
            return ::sca::cuda::CudaStatus(sca_err__, #expr " at " __FILE__ ":" SCI_STR(__LINE__)); \
        }                                                                                     \
    } while (0)

#define SCI_CUDA_CHECK_LAST()                                                                  \
    do {                                                                                       \
        const cudaError_t sca_err__ = cudaGetLastError();                                      \
        if (sca_err__ != cudaSuccess) {                                                        \
            return ::sca::cuda::CudaStatus(sca_err__, "kernel launch at " __FILE__ ":" SCI_STR(__LINE__)); \
        }                                                                                      \
    } while (0)

#define SCI_STR_IMPL(x) #x
#define SCI_STR(x) SCI_STR_IMPL(x)

