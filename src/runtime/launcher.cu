#include "runtime/launcher.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "device/stream.hpp"

namespace sca {
namespace runtime {
namespace {

std::mutex& SmemMutex() {
    static std::mutex m;
    return m;
}

// function pointer -> largest dynamic smem size successfully enabled so far
std::unordered_map<const void*, size_t>& SmemCache() {
    static std::unordered_map<const void*, size_t> cache;
    return cache;
}

bool DebugSyncEnabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("SCI_ATTENTION_DEBUG_SYNC");
        return v != nullptr && v[0] == '1';
    }();
    return enabled;
}

sci::Status StatusFromCuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return sci::Status::Ok();
    const std::string msg = std::string(what) + ": " + cudaGetErrorName(err) + " (" +
                            cudaGetErrorString(err) + ")";
    if (err == cudaErrorMemoryAllocation) return sci::Status::CudaOutOfMemory(msg);
    return sci::Status::CudaError(msg);
}

}  // namespace

void* ResolveStreamHandle(sci::Stream* stream) {
    if (stream != nullptr) return stream->handle();
    return sci::Stream::GetCurrent().handle();
}

sci::Status EnsureSmemLimit(const void* kernel, size_t smem_bytes) {
    if (kernel == nullptr) return sci::Status::InvalidArgument("EnsureSmemLimit: null kernel");
    if (smem_bytes <= 48 * 1024) return sci::Status::Ok();

    {
        std::lock_guard<std::mutex> lock(SmemMutex());
        const auto it = SmemCache().find(kernel);
        if (it != SmemCache().end() && it->second >= smem_bytes) return sci::Status::Ok();
    }

    const cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    if (err != cudaSuccess) {
        return sci::Status(
            sci::StatusCode::kInvalidOperation,
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=" + std::to_string(smem_bytes) +
            ") failed: " + cudaGetErrorString(err));
    }
    {
        std::lock_guard<std::mutex> lock(SmemMutex());
        size_t& cached = SmemCache()[kernel];
        if (cached < smem_bytes) cached = smem_bytes;
    }
    return sci::Status::Ok();
}

sci::Status LaunchRaw(const void* kernel, int grid_x, int grid_y, int grid_z, int block_x,
                      int block_y, int block_z, size_t smem_bytes, sci::Stream* stream,
                      void** args) {
    if (kernel == nullptr) return sci::Status::InvalidArgument("LaunchRaw: null kernel");
    if (grid_x <= 0 || grid_y <= 0 || grid_z <= 0) {
        return sci::Status::InvalidArgument("LaunchRaw: non-positive grid dimension");
    }
    if (block_x <= 0 || block_y <= 0 || block_z <= 0) {
        return sci::Status::InvalidArgument("LaunchRaw: non-positive block dimension");
    }

    const sci::Status smem_status = EnsureSmemLimit(kernel, smem_bytes);
    if (!smem_status.ok()) return smem_status;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(ResolveStreamHandle(stream));
    const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                    static_cast<unsigned>(grid_z));
    const dim3 block(static_cast<unsigned>(block_x), static_cast<unsigned>(block_y),
                     static_cast<unsigned>(block_z));
    const cudaError_t err = cudaLaunchKernel(kernel, grid, block, args, smem_bytes, cuda_stream);
    if (err != cudaSuccess) return StatusFromCuda(err, "cudaLaunchKernel");

    const cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        return StatusFromCuda(launch_err, "kernel launch (deferred)");
    }

    if (DebugSyncEnabled()) {
        const cudaError_t sync_err = cudaStreamSynchronize(cuda_stream);
        if (sync_err != cudaSuccess) {
            return StatusFromCuda(sync_err, "debug synchronise after launch");
        }
        const cudaError_t post = cudaGetLastError();
        if (post != cudaSuccess) return StatusFromCuda(post, "debug post-launch check");
    }
    return sci::Status::Ok();
}

}  // namespace runtime
}  // namespace sca
