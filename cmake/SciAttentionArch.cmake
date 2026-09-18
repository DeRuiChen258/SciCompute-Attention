# Architecture detection and validation.
#
# Rules (prompt §5.4):
#   * never hardcode sm_120 or a CUDA install path;
#   * refuse targets below sm_80 (mma.sync + cp.async are required);
#   * export SCI_ATTENTION_ARCH_<NN> definitions so device code can branch at compile time.

if(NOT DEFINED SCI_ATTENTION_ARCH OR SCI_ATTENTION_ARCH STREQUAL "")
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(SCI_ATTENTION_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Prefer the installed driver's compute capability, then map it onto the arch list that
        # this nvcc can actually emit code for.
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _sca_compute_cap
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_sca_compute_cap)
            string(REGEX REPLACE "[^0-9]" "" _sca_cc_digits "${_sca_compute_cap}")
            set(SCI_ATTENTION_ARCH "${_sca_cc_digits}")
        else()
            # Last resort: build for the highest arch the toolkit knows.
            execute_process(
                COMMAND nvcc --list-gpu-arch
                OUTPUT_VARIABLE _sca_arch_list
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            string(REGEX MATCHALL "[0-9]+" _sca_arch_numbers "${_sca_arch_list}")
            list(SORT _sca_arch_numbers COMPARE NATURAL ORDER DESCENDING)
            list(GET _sca_arch_numbers 0 _sca_top_arch)
            set(SCI_ATTENTION_ARCH "${_sca_top_arch}")
        endif()
    endif()
endif()

set(SCI_ATTENTION_ARCH "${SCI_ATTENTION_ARCH}"
    CACHE STRING "Target compute capabilities, e.g. \"90;120\" (must be >= 80)")

set(_sca_arch_defs "")
foreach(_arch IN LISTS SCI_ATTENTION_ARCH)
    if(_arch LESS 80)
        message(FATAL_ERROR
            "SCI_ATTENTION_ARCH=${SCI_ATTENTION_ARCH} contains sm_${_arch}, but this project "
            "requires sm_80+ (mma.sync m16n8k16 + cp.async). Supported: 80,86,89,90,100,103,110,120+")
    endif()
    list(APPEND _sca_arch_defs "SCI_ATTENTION_ARCH_${_arch}=1")
endforeach()

set(CMAKE_CUDA_ARCHITECTURES "${SCI_ATTENTION_ARCH}")
set(SCI_ATTENTION_ARCH_DEFS "${_sca_arch_defs}")

if(NOT SCI_ATTENTION_QUIET)
    message(STATUS "SciCompute-Attention: CUDA architectures = ${SCI_ATTENTION_ARCH}")
endif()

