cmake_minimum_required(VERSION 3.18)
project(raw_int8_restore_clear_fixture LANGUAGES CXX)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# CPU-only: the planner header's include closure pulls in no CUDA header (verified), so this exercises the
# production planner/preflight/dispatch over host buffers.  The executor is the only seam.
add_executable(raw_int8_restore_clear_test test.cpp)
target_compile_features(raw_int8_restore_clear_test PRIVATE cxx_std_20)
# Never let a Release/NDEBUG build silence the gate: every check is an explicit counted comparison, and the
# binary exits 1.  -UNDEBUG is belt-and-braces for any assert() added later.
target_compile_options(raw_int8_restore_clear_test PRIVATE -UNDEBUG)
target_include_directories(raw_int8_restore_clear_test PRIVATE "${source_root}/include")

add_test(NAME raw_int8_restore_clear_source
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check_source.py
    ${source_root}/src/program/suffix_raw_runtime.cpp
    ${source_root}/include/strata/core/raw_int8_restore_clear.hpp
    ${CMAKE_CURRENT_LIST_DIR}/test.cpp
    ${source_root}/src/core/conversation_snapshot.cpp)
add_test(NAME raw_int8_restore_clear_exec COMMAND raw_int8_restore_clear_test)
set_tests_properties(raw_int8_restore_clear_source PROPERTIES FIXTURES_SETUP raw_int8_restore_clear_source TIMEOUT 30)
set_tests_properties(raw_int8_restore_clear_exec PROPERTIES FIXTURES_REQUIRED raw_int8_restore_clear_source TIMEOUT 60)
