cmake_minimum_required(VERSION 3.18)
project(conversation_suffix_provenance_fixture LANGUAGES CXX)
enable_testing()
get_filename_component(source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# CPU-only matching-policy gate (conversation_cache.hpp includes no CUDA header).  Proves only that approximate
# provenance can never certify an exact reuse; it does NOT prove any import path, serving behavior or CUDA state.
add_executable(conversation_suffix_provenance_test test.cpp)
target_compile_features(conversation_suffix_provenance_test PRIVATE cxx_std_20)
target_compile_options(conversation_suffix_provenance_test PRIVATE -UNDEBUG)
target_include_directories(conversation_suffix_provenance_test PRIVATE "${source_root}/include")
add_test(NAME conversation_suffix_provenance_test COMMAND conversation_suffix_provenance_test)
set_tests_properties(conversation_suffix_provenance_test PROPERTIES TIMEOUT 30)
