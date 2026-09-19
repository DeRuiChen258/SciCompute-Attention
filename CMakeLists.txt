cmake_minimum_required(VERSION 3.24)
project(SciComputeAttention VERSION 0.1.0 LANGUAGES CXX)

# C++20 host code mirrors SciComputeInfra; CUDA translation units stay at C++17.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SciAttentionOptions)

find_package(Threads REQUIRED)

set(SCI_ATTENTION_HAVE_CUDA OFF)
if(NOT SCI_ATTENTION_INFRA_MODE STREQUAL "STUB")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(SCI_ATTENTION_HAVE_CUDA ON)
endif()

# Architecture detection needs the CUDA language, so it runs after enable_language(); it returns
# early for CPU-only builds (STUB mode).
include(SciAttentionArch)

include(SciComputeInfra)
include(SciAttentionDeps)

# ---------------------------------------------------------------------------------------------
# Warning policy (applied per target: upstream sources are compiled but not re-linted)
# ---------------------------------------------------------------------------------------------
set(SCI_ATTENTION_CXX_WARNINGS -Wall -Wextra -Wpedantic)
# -Wpedantic is deliberately omitted for CUDA translation units: nvcc's generated cudafe1.cpp
# emits `#line` directives that GCC reports as "style of line directive is a GCC extension",
# which would make SCI_ATTENTION_WERROR=ON unusable for reasons unrelated to our code.
set(SCI_ATTENTION_CUDA_WARNINGS -Xcompiler=-Wall,-Wextra --expt-relaxed-constexpr)
if(SCI_ATTENTION_WERROR)
    list(APPEND SCI_ATTENTION_CXX_WARNINGS -Werror)
    list(APPEND SCI_ATTENTION_CUDA_WARNINGS -Xcompiler=-Werror)
endif()

add_subdirectory(src)

if(SCI_ATTENTION_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SCI_ATTENTION_BUILD_TESTS AND SCI_ATTENTION_HAVE_GTEST)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SCI_ATTENTION_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(SCI_ATTENTION_BUILD_VLLM_ADAPTER)
    add_subdirectory(vllm_backend)
endif()

if(SCI_ATTENTION_BUILD_PYTHON AND SCI_ATTENTION_HAVE_PYBIND11
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/CMakeLists.txt")
    add_subdirectory(src/bindings)
endif()

# ---------------------------------------------------------------------------------------------
# Install rules (used by a future find_package(SciComputeAttention))
# ---------------------------------------------------------------------------------------------
include(GNUInstallDirs)
set(_sca_install_targets sci_attention)
if(SCI_ATTENTION_HAVE_CUDA)
    list(APPEND _sca_install_targets sci_attention_kernels)
endif()
install(TARGETS ${_sca_install_targets}
        EXPORT SciComputeAttentionTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ---------------------------------------------------------------------------------------------
# Configuration summary (prompt §5.1)
# ---------------------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "SciCompute-Attention ${PROJECT_VERSION} configuration")
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Infra mode        : ${SCI_ATTENTION_INFRA_MODE}")
message(STATUS "  Infra root        : ${SCI_ATTENTION_SCICOMPUTE_INFRA_ROOT}")
message(STATUS "  Infra git sha     : ${SCI_ATTENTION_INFRA_SHA}")
message(STATUS "  CUDA              : ${SCI_ATTENTION_HAVE_CUDA}")
message(STATUS "  CUDA arch         : ${SCI_ATTENTION_ARCH}")
message(STATUS "  TMA policy        : ${SCI_ATTENTION_ENABLE_TMA}")
message(STATUS "  smem/block limit  : ${SCI_ATTENTION_MAX_SMEM_BYTES} B")
message(STATUS "  Tests             : ${SCI_ATTENTION_BUILD_TESTS} (GTest: ${SCI_ATTENTION_HAVE_GTEST})")
message(STATUS "  Benchmarks        : ${SCI_ATTENTION_BUILD_BENCHMARKS} (Google Benchmark: ${SCI_ATTENTION_HAVE_BENCHMARK})")
message(STATUS "  Python bindings   : ${SCI_ATTENTION_BUILD_PYTHON} (pybind11: ${SCI_ATTENTION_HAVE_PYBIND11})")
message(STATUS "  vLLM adapter      : ${SCI_ATTENTION_BUILD_VLLM_ADAPTER}")
message(STATUS "  Werror            : ${SCI_ATTENTION_WERROR}")
message(STATUS "")
