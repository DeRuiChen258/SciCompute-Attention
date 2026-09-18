#include "scicompute_attention/backends.hpp"

#include <cstring>

#include "runtime/backend_registry.hpp"

namespace sca {

const IAttentionBackend* GetBackend(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::kNaive: return runtime::NaiveBackendInstance();
        case BackendKind::kTiled: return runtime::TiledBackendInstance();
        case BackendKind::kFlash: return runtime::FlashBackendInstance();
        case BackendKind::kDecode: return runtime::DecodeBackendInstance();
        case BackendKind::kPaged: return runtime::PagedBackendInstance();
        case BackendKind::kAuto:
        default: return runtime::NaiveBackendInstance();
    }
}

const char* BackendTypeName(BackendKind kind) noexcept { return BackendName(kind); }

bool ParseBackendName(const char* name, BackendKind* out) noexcept {
    if (name == nullptr || out == nullptr) return false;
    struct Entry {
        const char* text;
        BackendKind kind;
    };
    static constexpr Entry kEntries[] = {
        {"auto", BackendKind::kAuto},     {"naive", BackendKind::kNaive},
        {"tiled", BackendKind::kTiled},   {"flash", BackendKind::kFlash},
        {"decode", BackendKind::kDecode}, {"paged", BackendKind::kPaged},
    };
    for (const Entry& e : kEntries) {
        if (std::strcmp(name, e.text) == 0) {
            *out = e.kind;
            return true;
        }
    }
    return false;
}

}  // namespace sca

