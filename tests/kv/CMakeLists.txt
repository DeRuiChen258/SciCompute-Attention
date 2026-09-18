set(SCA_KV_TESTS
    test_kv_cache.cpp
    test_block_manager.cpp
    test_paged_kv.cpp
)

foreach(_test ${SCA_KV_TESTS})
    get_filename_component(_name ${_test} NAME_WE)
    add_executable(${_name} ${_test})
    target_link_libraries(${_name} PRIVATE sca_test_common)
    add_test(NAME ${_name} COMMAND ${_name})
    set_tests_properties(${_name} PROPERTIES LABELS "kv")
endforeach()
