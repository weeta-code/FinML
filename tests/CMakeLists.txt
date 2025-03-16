file(GLOB_RECURSE TEST_SOURCES "*.cpp")

# Add test executable
add_executable(finml_tests ${TEST_SOURCES})
target_link_libraries(finml_tests finml)

# Add tests
add_test(NAME finml_tests COMMAND finml_tests) 