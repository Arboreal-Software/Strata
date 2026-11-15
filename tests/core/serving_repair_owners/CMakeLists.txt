cmake_minimum_required(VERSION 3.18)
project(serving_repair_owners_fixture LANGUAGES CXX)
enable_testing()
get_filename_component(source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# CPU-only test for the retained repair owner registry gates, activation identity, and
# incoming-prompt identity predicate. No GPU, no CUDA, no engine call.
add_executable(serving_repair_owners_test
    ${source_root}/src/program/serving_repair_owners_test.cpp
    ${source_root}/src/core/window_hash.cpp
    ${source_root}/src/core/window_budget.cpp)
target_compile_features(serving_repair_owners_test PRIVATE cxx_std_20)
target_compile_options(serving_repair_owners_test PRIVATE -UNDEBUG)
target_include_directories(serving_repair_owners_test PRIVATE
    "${source_root}/include"
    "${source_root}/src/program")
add_test(NAME serving_repair_owners_test COMMAND serving_repair_owners_test)
set_tests_properties(serving_repair_owners_test PROPERTIES TIMEOUT 120)
