// Device probing (see device_probe.h). Compiled only into the CUDA library.

#include "runtime/device_probe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

namespace {

// Compile-time contribution: a probe that the binary was actually built for an architecture that
// has the instructions. CMake defines SCI_ATTENTION_ARCH_<NN> for every entry of
// SCI_ATTENTION_ARCH. Nothing here hardcodes a device name.
#if defined(SCI_ATTENTION_ARCH_80) || defined(SCI_ATTENTION_ARCH_86) || \
    defined(SCI_ATTENTION_ARCH_89) || defined(SCI_ATTENTION_ARCH_90) || \
    defined(SCI_ATTENTION_ARCH_100) || defined(SCI_ATTENTION_ARCH_103) || \
    defined(SCI_ATTENTION_ARCH_110) || defined(SCI_ATTENTION_ARCH_120) || \
    defined(SCI_ATTENTION_ARCH_121)
constexpr int kBinaryHasSm80 = 1;
#else
constexpr int kBinaryHasSm80 = 0;
#endif

#if defined(SCI_ATTENTION_ARCH_90) || defined(SCI_ATTENTION_ARCH_100) || \
    defined(SCI_ATTENTION_ARCH_103) || defined(SCI_ATTENTION_ARCH_110) || \
    defined(SCI_ATTENTION_ARCH_120) || defined(SCI_ATTENTION_ARCH_121)
constexpr int kBinaryHasSm90 = 1;
#else
constexpr int kBinaryHasSm90 = 0;
#endif

}  // namespace

extern "C" int sca_probe_device(int device_id, ScaDeviceProbe* out) {
    if (out == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    std::memset(out, 0, sizeof(ScaDeviceProbe));

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0 || device_id < 0 || device_id >= count) {
        out->cuda_available = 0;
        return static_cast<int>(err == cudaSuccess ? cudaErrorInvalidDevice : err);
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        out->cuda_available = 0;
        return static_cast<int>(err);
    }

    const size_t name_len = std::min(std::strlen(prop.name), static_cast<size_t>(SCA_DEVICE_NAME_MAX - 1));
    std::memcpy(out->name, prop.name, name_len);
    out->name[name_len] = '\0';
    out->major = prop.major;
    out->minor = prop.minor;
    out->sm_count = prop.multiProcessorCount;
    out->smem_per_sm = prop.sharedMemPerMultiprocessor;
    out->smem_per_block_default = prop.sharedMemPerBlock;
    out->smem_per_block_optin = prop.sharedMemPerBlockOptin;
    out->regs_per_sm = static_cast<unsigned long long>(prop.regsPerMultiprocessor);
    out->max_threads_per_sm = static_cast<unsigned long long>(prop.maxThreadsPerMultiProcessor);
    out->l2_bytes = static_cast<unsigned long long>(prop.l2CacheSize);
    out->total_mem_bytes = static_cast<unsigned long long>(prop.totalGlobalMem);
    out->cuda_available = 1;
    out->binary_has_sm80 = kBinaryHasSm80;
    out->binary_has_sm90 = kBinaryHasSm90;

    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        out->free_mem_bytes = static_cast<unsigned long long>(free_bytes);
    }
    return 0;
}

extern "C" int sca_mem_get_info(unsigned long long* free_bytes, unsigned long long* total_bytes) {
    size_t free_b = 0, total_b = 0;
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) return static_cast<int>(err);
    if (free_bytes != nullptr) *free_bytes = static_cast<unsigned long long>(free_b);
    if (total_bytes != nullptr) *total_bytes = static_cast<unsigned long long>(total_b);
    return 0;
}
