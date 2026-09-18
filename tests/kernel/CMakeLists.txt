# Kernel correctness tests (Phase 2+). Each executable is a standalone CUDA program that can also
# be handed to compute-sanitizer directly (see tests/CMakeLists.txt).

set(SCA_KERNEL_TESTS
    test_naive_correctness.cu
    test_tiled_correctness.cu
    test_flash_correctness.cu
    test_decode_correctness.cu
    test_paged_correctness.cu
)

foreach(_test ${SCA_KERNEL_TESTS})
    get_filename_component(_name ${_test} NAME_WE)
    add_executable(${_name} ${_test})
    target_link_libraries(${_name} PRIVATE sca_test_common)
    target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
    target_compile_definitions(${_name} PRIVATE ${SCI_ATTENTION_ARCH_DEFS})
    add_test(NAME ${_name} COMMAND ${_name})
    set_tests_properties(${_name} PROPERTIES LABELS "kernel")
endforeach()
