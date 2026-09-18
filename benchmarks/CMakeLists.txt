# Benchmark targets. The CUDA-event harness lives in benchmarks/common/bench_utils.hpp and works
# without Google Benchmark; benchmark::State based targets are only added when it is available.

if(SCI_ATTENTION_HAVE_BENCHMARK)
    message(STATUS "SciCompute-Attention: Google Benchmark found -> benchmark::State targets enabled")
endif()

if(SCI_ATTENTION_HAVE_CUDA)
    foreach(_bench benchmark_naive benchmark_attention)
        add_executable(${_bench} ${_bench}.cu)
        target_link_libraries(${_bench} PRIVATE sci_attention)
        target_include_directories(${_bench} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/.."
            "${CMAKE_CURRENT_SOURCE_DIR}/../src"
            "${CMAKE_CURRENT_SOURCE_DIR}/../tests")
        target_compile_definitions(${_bench} PRIVATE
            ${SCI_ATTENTION_ARCH_DEFS}
            SCA_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
    endforeach()
endif()
