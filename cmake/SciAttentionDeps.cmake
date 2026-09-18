# Third-party dependencies with explicit degradation (prompt §5.5). Nothing is silently skipped.

set(SCI_ATTENTION_HAVE_GTEST OFF)
set(SCI_ATTENTION_HAVE_BENCHMARK OFF)
set(SCI_ATTENTION_HAVE_PYBIND11 OFF)

if(SCI_ATTENTION_BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        set(SCI_ATTENTION_HAVE_GTEST ON)
    else()
        message(STATUS "SciCompute-Attention: GoogleTest not found -> test targets disabled "
                       "(install libgtest-dev to enable them)")
    endif()
endif()

if(SCI_ATTENTION_BUILD_BENCHMARKS)
    # libbenchmark is not installed system-wide on this machine; the vcpkg build is the source of
    # truth (see docs/env_report.md).
    set(_sca_vcpkg_benchmark "/home/violet/Workspace/IDE/vcpkg-2026.06.01/packages/benchmark_x64-linux")
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND AND EXISTS "${_sca_vcpkg_benchmark}")
        list(APPEND CMAKE_PREFIX_PATH "${_sca_vcpkg_benchmark}")
        find_package(benchmark QUIET)
    endif()
    if(benchmark_FOUND)
        set(SCI_ATTENTION_HAVE_BENCHMARK ON)
    else()
        message(STATUS "SciCompute-Attention: Google Benchmark not found -> only the built-in "
                       "CUDA-event benchmark harness is built (no benchmark::State targets)")
    endif()
endif()

if(SCI_ATTENTION_BUILD_PYTHON AND NOT SCI_ATTENTION_INFRA_MODE STREQUAL "STUB")
    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND AND Python3_EXECUTABLE)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _sca_pybind11_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_sca_pybind11_dir AND EXISTS "${_sca_pybind11_dir}")
            list(APPEND CMAKE_PREFIX_PATH "${_sca_pybind11_dir}")
            find_package(pybind11 QUIET)
        endif()
    endif()
    if(pybind11_FOUND AND Python3_Development.Module_FOUND)
        set(SCI_ATTENTION_HAVE_PYBIND11 ON)
        message(STATUS "SciCompute-Attention: pybind11 ${pybind11_VERSION} -> Python module enabled")
    else()
        message(STATUS "SciCompute-Attention: pybind11/Python dev headers not found -> Python "
                       "module disabled. Install with: pip install pybind11")
    endif()
endif()

