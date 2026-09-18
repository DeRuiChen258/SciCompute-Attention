# Target layout:
#   sci_attention_kernels - CUDA translation units (device code + host-side launches)
#   sci_attention         - C++17/C++20 host API, runtime, backends and KV cache
#   scicompute_attention  - ALIAS of sci_attention (the name upper layers link against)

set(SCI_ATTENTION_HOST_SOURCES
    api/status.cpp
    api/api_validate.cpp
    api/attention_api.cpp
    runtime/capability.cpp
    runtime/workspace.cpp
    runtime/dispatcher.cpp
    runtime/backend_lookup.cpp
    backends/naive/naive_backend.cpp
    backends/tiled/tiled_backend.cpp
    backends/flash/flash_backend.cpp
    backends/decode/decode_backend.cpp
    backends/paged/paged_backend.cpp
)

set(SCI_ATTENTION_KERNEL_SOURCES "")
if(SCI_ATTENTION_HAVE_CUDA)
    set(SCI_ATTENTION_KERNEL_SOURCES
        runtime/device_probe.cu
        runtime/launcher.cu
    )
else()
    list(APPEND SCI_ATTENTION_HOST_SOURCES runtime/device_probe_stub.cpp)
endif()

if(SCI_ATTENTION_HAVE_CUDA)
    add_library(sci_attention_kernels STATIC ${SCI_ATTENTION_KERNEL_SOURCES})
    target_include_directories(sci_attention_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(sci_attention_kernels PUBLIC ${SCI_ATTENTION_ARCH_DEFS})
    target_compile_options(sci_attention_kernels PRIVATE ${SCI_ATTENTION_CUDA_WARNINGS})
else()
    # CPU-only (STUB) build: no device translation units exist, so keep the target name alive as an
    # INTERFACE library to avoid conditionals at every consumer.
    add_library(sci_attention_kernels INTERFACE)
endif()
if(NOT SCI_ATTENTION_INFRA_MODE STREQUAL "STUB")
    target_link_libraries(sci_attention_kernels PUBLIC sci_attention_infra)
endif()

add_library(sci_attention STATIC ${SCI_ATTENTION_HOST_SOURCES})
add_library(scicompute_attention ALIAS sci_attention)

target_include_directories(sci_attention PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}")   # internal headers (runtime/*.hpp) for our own TUs only

target_compile_options(sci_attention PRIVATE ${SCI_ATTENTION_CXX_WARNINGS})
target_link_libraries(sci_attention PUBLIC sci_attention_kernels)
target_link_libraries(sci_attention PUBLIC sci_attention_infra)
target_link_libraries(sci_attention PUBLIC Threads::Threads)

set_target_properties(sci_attention PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
