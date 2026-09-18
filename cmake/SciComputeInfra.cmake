# Locates SciComputeInfra (the infrastructure layer this project builds upon) and exposes it as
# the single INTERFACE target `sci_attention_infra`.
#
# Modes (SCI_ATTENTION_INFRA_MODE):
#   SOURCE  - default. Compile the upstream *source files* (read-only, never copied) into the
#             `sca_infra_*` targets and link them.
#   PACKAGE - find_package(SciComputeInfra); only usable once upstream installs/exports itself.
#   STUB    - no CUDA, no upstream: a header-only shim under stub/include keeps the API layer and
#             the CPU unit tests compilable. GPU results are INVALID in this mode.
#
# Why SOURCE does not use add_subdirectory(): every upstream CMakeLists uses
# ${CMAKE_SOURCE_DIR}/... for include dirs and source paths, and a nested project() call resets
# CMAKE_SOURCE_DIR to *this* project's root, so a nested build silently resolves to
# <SciCompute-Attention>/include and <SciCompute-Attention>/cuda/kernels/*.cu. That was verified
# experimentally (see upstream/notes.md UP-008). Compiling the upstream sources into dedicated
# targets avoids the hazard without modifying, patching or copying upstream.

set(_sca_infra_mode "${SCI_ATTENTION_INFRA_MODE}")
string(TOUPPER "${_sca_infra_mode}" _sca_infra_mode)

if(_sca_infra_mode STREQUAL "STUB")
    add_library(sci_attention_infra INTERFACE)
    target_include_directories(sci_attention_infra INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/stub/include")
    target_compile_definitions(sci_attention_infra INTERFACE SCI_ATTENTION_INFRA_STUB=1)
    message(WARNING
        "SciCompute-Attention: STUB mode selected.\n"
        "  * no upstream code is compiled and no CUDA kernel is built;\n"
        "  * GPU kernels, benchmarks and all performance/numerical conclusions are INVALID;\n"
        "  * this mode exists so that the API layer and CPU unit tests stay compilable.")
    set(SCI_ATTENTION_HAVE_INFRA OFF)
    set(SCI_ATTENTION_INFRA_SHA "STUB")
    return()
endif()

if(_sca_infra_mode STREQUAL "PACKAGE")
    find_package(SciComputeInfra QUIET)
    if(NOT SciComputeInfra_FOUND)
        message(FATAL_ERROR
            "SCI_ATTENTION_INFRA_MODE=PACKAGE but find_package(SciComputeInfra) failed.\n"
            "  Use the source mode instead:\n"
            "    -DSCI_ATTENTION_INFRA_MODE=SOURCE "
            "-DSCI_ATTENTION_SCICOMPUTE_INFRA_ROOT=/abs/path/to/SciComputeInfra")
    endif()
    add_library(sci_attention_infra INTERFACE)
    target_link_libraries(sci_attention_infra INTERFACE SciComputeInfra::SciComputeInfra)
    set(SCI_ATTENTION_HAVE_INFRA ON)
    set(SCI_ATTENTION_INFRA_SHA "PACKAGE")
    return()
endif()

# ---------------------------------------------------------------------------------------------
# SOURCE mode
# ---------------------------------------------------------------------------------------------
set(_infra_root "${SCI_ATTENTION_SCICOMPUTE_INFRA_ROOT}")

if(NOT EXISTS "${_infra_root}/CMakeLists.txt" OR
   NOT EXISTS "${_infra_root}/include/tensor/tensor.hpp")
    message(FATAL_ERROR
        "SciComputeInfra not found at '${_infra_root}'.\n"
        "  Expected <root>/CMakeLists.txt and <root>/include/tensor/tensor.hpp.\n"
        "  Set -DSCI_ATTENTION_SCICOMPUTE_INFRA_ROOT=/abs/path/to/SciComputeInfra, or use\n"
        "  -DSCI_ATTENTION_INFRA_MODE=STUB for a CPU-only API build (GPU results invalid).")
endif()

execute_process(
    COMMAND git -C "${_infra_root}" rev-parse HEAD
    OUTPUT_VARIABLE SCI_ATTENTION_INFRA_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT SCI_ATTENTION_INFRA_SHA)
    set(SCI_ATTENTION_INFRA_SHA "unknown")
    message(STATUS "SciCompute-Attention: upstream git sha unavailable (not a git checkout?)")
endif()

set(_infra_include "${_infra_root}/include")

# Interface target carrying include dirs and CUDA defines for every upstream-derived target.
add_library(sca_infra_options INTERFACE)
# SYSTEM: upstream headers are third-party from this project's point of view, so their warnings
# must not poison SCI_ATTENTION_WERROR (e.g. device.hpp's unused parameters).
target_include_directories(sca_infra_options SYSTEM INTERFACE "${_infra_include}")
target_compile_definitions(sca_infra_options INTERFACE SCI_USE_CUDA=1 SCI_ENABLE_CUDA=1)

function(sca_add_infra_target target_name)
    # Reuse an existing upstream target when the consumer already provided one.
    if(TARGET ${target_name})
        return()
    endif()
    add_library(${target_name} STATIC ${ARGN})
    target_link_libraries(${target_name} PUBLIC sca_infra_options)
    set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

sca_add_infra_target(sci_device
    "${_infra_root}/src/device/device.cpp"
    "${_infra_root}/src/device/stream.cpp"
    "${_infra_root}/src/device/cuda_device.cpp")
target_link_libraries(sci_device PUBLIC Threads::Threads CUDA::cudart)

sca_add_infra_target(sci_memory
    "${_infra_root}/src/memory/allocator.cpp"
    "${_infra_root}/src/memory/buffer_handle.cpp"
    "${_infra_root}/src/memory/memory_pool.cpp")
target_link_libraries(sci_memory PUBLIC sci_device)

sca_add_infra_target(sci_tensor "${_infra_root}/src/tensor/tensor.cpp")
target_link_libraries(sci_tensor PUBLIC sci_memory sci_device)

sca_add_infra_target(sci_math
    "${_infra_root}/src/math/elementwise.cpp"
    "${_infra_root}/src/math/reduction.cpp"
    "${_infra_root}/src/math/softmax.cpp"
    "${_infra_root}/src/math/normalization.cpp")
target_link_libraries(sci_math PUBLIC sci_tensor sci_device)

sca_add_infra_target(sci_benchmark "${_infra_root}/src/benchmark/benchmark_runner.cpp")
target_link_libraries(sci_benchmark PUBLIC sci_tensor sci_math)

add_library(sci_attention_infra INTERFACE)
target_link_libraries(sci_attention_infra INTERFACE
    sci_tensor sci_memory sci_device sci_math sci_benchmark CUDA::cudart Threads::Threads)

set(SCI_ATTENTION_HAVE_INFRA ON)
message(STATUS "SciCompute-Attention: SciComputeInfra source mode")
message(STATUS "  root : ${_infra_root}")
message(STATUS "  sha  : ${SCI_ATTENTION_INFRA_SHA}")
message(STATUS "  note : upstream CMakeLists use \${CMAKE_SOURCE_DIR}; compiled from an explicit "
               "source list instead (see upstream/notes.md UP-008)")
