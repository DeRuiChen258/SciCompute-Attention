# All project options plus the mutual-exclusion checks (prompt §5.3).

option(SCI_ATTENTION_BUILD_TESTS "Build GoogleTest targets and register them with CTest" ON)
option(SCI_ATTENTION_BUILD_BENCHMARKS "Build benchmark targets" ON)
option(SCI_ATTENTION_BUILD_PYTHON "Build the pybind11 module" ON)
option(SCI_ATTENTION_BUILD_VLLM_ADAPTER "Build the vLLM(C++) adapter layer" OFF)
option(SCI_ATTENTION_BUILD_EXAMPLES "Build the example programs" ON)
option(SCI_ATTENTION_WERROR "Treat warnings as errors (CI)" OFF)
option(SCI_ATTENTION_ENABLE_FP8 "Enable the experimental FP8 probe path" OFF)

set(SCI_ATTENTION_VLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../vllm"
    CACHE PATH "Path to the vLLM(C++) repository (only used with the adapter)")

set(SCI_ATTENTION_ENABLE_TMA "AUTO"
    CACHE STRING "TMA usage policy: ON | OFF | AUTO (AUTO = runtime probe decides)")
set_property(CACHE SCI_ATTENTION_ENABLE_TMA PROPERTY STRINGS ON OFF AUTO)

set(SCI_ATTENTION_MAX_SMEM_BYTES "102400"
    CACHE STRING "Shared-memory ceiling per block in bytes (defaults to the device value)")

set(SCI_ATTENTION_ARCH "" CACHE STRING "Target compute capabilities, e.g. \"120\"")

# Defined here (not in SciComputeInfra.cmake) because the top-level project() must know whether it
# has to enable the CUDA language at all.
set(SCI_ATTENTION_INFRA_MODE "SOURCE"
    CACHE STRING "How to obtain SciComputeInfra: SOURCE | PACKAGE | STUB")
set_property(CACHE SCI_ATTENTION_INFRA_MODE PROPERTY STRINGS SOURCE PACKAGE STUB)

set(SCI_ATTENTION_SCICOMPUTE_INFRA_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../SciComputeInfra"
    CACHE PATH "Path to SciComputeInfra (upstream infrastructure layer)")

set(SCI_ATTENTION_QUIET OFF CACHE BOOL "Reduce configure-time output")

if(SCI_ATTENTION_ENABLE_TMA STREQUAL "ON" AND SCI_ATTENTION_INFRA_MODE STREQUAL "STUB")
    message(FATAL_ERROR "SCI_ATTENTION_ENABLE_TMA=ON requires a CUDA build (STUB mode has no TMA)")
endif()

if(SCI_ATTENTION_BUILD_VLLM_ADAPTER)
    if(NOT EXISTS "${SCI_ATTENTION_VLLM_ROOT}/include/vllm/attention/flash_attention.hpp")
        message(FATAL_ERROR
            "SCI_ATTENTION_BUILD_VLLM_ADAPTER=ON but '${SCI_ATTENTION_VLLM_ROOT}' does not contain "
            "include/vllm/attention/flash_attention.hpp.\n"
            "  Configure with -DSCI_ATTENTION_VLLM_ROOT=/abs/path/to/vllm")
    endif()
    if(SCI_ATTENTION_INFRA_MODE STREQUAL "STUB")
        message(FATAL_ERROR "The vLLM adapter needs the CUDA build (INFRA_MODE != STUB)")
    endif()
endif()
