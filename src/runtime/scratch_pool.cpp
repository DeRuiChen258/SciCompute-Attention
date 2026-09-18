#include "runtime/scratch_pool.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "scicompute_attention/status.hpp"

namespace sca {
namespace runtime {
namespace {

struct Entry {
    sci::Tensor tensor;
    size_t elements{0};
};

std::mutex& PoolMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<int, Entry>& Pool() {
    // Intentionally leaked: destroying device buffers from a static destructor would call cudaFree
    // after the CUDA runtime has been torn down, which segfaults during process exit (observed as
    // SIGSEGV in sci::BufferHandle::deallocate). ReleaseScratch() frees the buffers explicitly when
    // a caller actually wants the memory back; otherwise the OS reclaims everything at exit.
    static std::unordered_map<int, Entry>* pool = new std::unordered_map<int, Entry>();
    return *pool;
}

size_t g_high_water_bytes = 0;

int KeyOf(int device_id, ScratchKind kind) { return device_id * 16 + static_cast<int>(kind); }

}  // namespace

sci::Result<float*> AcquireScratchFp32(ScratchKind kind, size_t elements, sci::Device& device) {
    if (elements == 0) {
        return sci::MakeUnexpected<float*>(
            MakeStatus(AttnStatusCode::kWorkspaceExceeded, "scratch request with 0 elements"));
    }
    std::lock_guard<std::mutex> lock(PoolMutex());
    Entry& entry = Pool()[KeyOf(device.id(), kind)];
    if (!entry.tensor.data() || entry.elements < elements) {
        // Grow with slack so that a slightly larger request does not trigger a new allocation.
        const size_t grown = static_cast<size_t>(static_cast<double>(elements) * 1.25) + 1024;
        entry.tensor = sci::Tensor(sci::TensorShape({static_cast<sci::index_t>(grown)}),
                                   sci::DType::kFloat32, device);
        if (entry.tensor.data() == nullptr) {
            entry.elements = 0;
            return sci::MakeUnexpected<float*>(MakeStatus(
                AttnStatusCode::kWorkspaceExceeded,
                "scratch allocation of " + std::to_string(grown * sizeof(float)) + " B failed"));
        }
        entry.elements = grown;
        g_high_water_bytes = std::max(g_high_water_bytes, grown * sizeof(float));
    }
    return sci::Ok(static_cast<float*>(entry.tensor.data()));
}

sci::Status ReleaseScratch() {
    std::lock_guard<std::mutex> lock(PoolMutex());
    Pool().clear();
    return sci::Status::Ok();
}

size_t ScratchHighWaterMarkBytes() {
    std::lock_guard<std::mutex> lock(PoolMutex());
    return g_high_water_bytes;
}

}  // namespace runtime
}  // namespace sca
