set(SCA_UNIT_TESTS
    test_config.cpp
    test_status.cpp
    test_dispatch.cpp
    test_capability.cpp
    test_tile_smem_calc.cpp
)

foreach(_test ${SCA_UNIT_TESTS})
    get_filename_component(_name ${_test} NAME_WE)
    add_executable(${_name} ${_test})
    target_link_libraries(${_name} PRIVATE sca_test_common)
    add_test(NAME ${_name} COMMAND ${_name})
    set_tests_properties(${_name} PROPERTIES LABELS "unit")
endforeach()

