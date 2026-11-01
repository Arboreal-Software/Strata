cmake_minimum_required(VERSION 3.18)
project(suffix_import_policy_fixture LANGUAGES CXX)
enable_testing()
get_filename_component(source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# CPU-only admission gate for P12 serving suffix reuse. Proves the verdict and its reasons only: no import, no
# restore, no certificate, no engine state, no GPU.
add_executable(suffix_import_policy_test test.cpp)
target_compile_features(suffix_import_policy_test PRIVATE cxx_std_20)
target_compile_options(suffix_import_policy_test PRIVATE -UNDEBUG -Wno-unused-function)
target_include_directories(suffix_import_policy_test PRIVATE "${source_root}/include")
add_test(NAME suffix_import_policy_test COMMAND suffix_import_policy_test)
set_tests_properties(suffix_import_policy_test PROPERTIES TIMEOUT 30)
