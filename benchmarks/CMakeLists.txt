# Benchmark targets. The CUDA-event harness lives in benchmarks/common/bench_utils.hpp and works
# without Google Benchmark; benchmark::State based targets are only added when it is available.

if(SCI_ATTENTION_HAVE_BENCHMARK)
    message(STATUS "SciCompute-Attention: Google Benchmark found -> benchmark::State targets enabled")
endif()

