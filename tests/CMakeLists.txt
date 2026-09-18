# Test layout (prompt §11): unit / kernel / kv / integration, plus the sanitizer convenience
# targets. gtest_discover_tests registers each TEST() as an individual CTest case.

find_package(GTest REQUIRED)

add_library(sca_test_common INTERFACE)
target_include_directories(sca_test_common INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(sca_test_common INTERFACE sci_attention GTest::gtest GTest::gtest_main)

add_subdirectory(unit)

if(SCI_ATTENTION_HAVE_CUDA)
    add_subdirectory(kernel)
    add_subdirectory(kv)
    add_subdirectory(integration)
endif()

# ---------------------------------------------------------------------------------------------
# Runtime-check targets (compute-sanitizer). Registered as CTest tests so that
# `ctest -R sanitize` runs them; they are excluded from the default run because they are slow.
# ---------------------------------------------------------------------------------------------
if(SCI_ATTENTION_HAVE_CUDA)
    foreach(_target test_flash_correctness test_tiled_correctness test_decode_correctness)
        if(TARGET ${_target})
            add_test(NAME sanitize_${_target}
                     COMMAND compute-sanitizer --tool memcheck --launch-timeout 120
                             $<TARGET_FILE:${_target}>)
            set_tests_properties(sanitize_${_target} PROPERTIES
                LABELS "sanitize" TIMEOUT 900)
        endif()
    endforeach()
endif()

