cmake_minimum_required(VERSION 3.18)
project(suffix_edit_span_fixture LANGUAGES CXX)
enable_testing()
get_filename_component(source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# CPU-only classification gate for the P12 suffix-reuse edit span. Proves no import, no certificate, no engine state.
add_executable(suffix_edit_span_test test.cpp)
target_compile_features(suffix_edit_span_test PRIVATE cxx_std_20)
target_compile_options(suffix_edit_span_test PRIVATE -UNDEBUG -Wno-unused-function)
target_include_directories(suffix_edit_span_test PRIVATE "${source_root}/include")
add_test(NAME suffix_edit_span_test COMMAND suffix_edit_span_test)
set_tests_properties(suffix_edit_span_test PROPERTIES TIMEOUT 30)
