cmake_minimum_required(VERSION 3.18)
project(suffix_bundle_record_fixture LANGUAGES CXX)
enable_testing()
get_filename_component(source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# CPU gate for the qualification bundle record codec: export/load fidelity and refusal of truncated, tampered,
# foreign or inventory-incomplete records. Proves transport integrity only, never capture or comparison results.
add_executable(suffix_bundle_record_test test.cpp "${source_root}/src/core/window_hash.cpp")
target_compile_features(suffix_bundle_record_test PRIVATE cxx_std_20)
target_compile_options(suffix_bundle_record_test PRIVATE -UNDEBUG)
target_include_directories(suffix_bundle_record_test PRIVATE "${source_root}/include")
add_test(NAME suffix_bundle_record_test COMMAND suffix_bundle_record_test)
set_tests_properties(suffix_bundle_record_test PROPERTIES TIMEOUT 60)
