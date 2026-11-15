cmake_minimum_required(VERSION 3.18)
project(serving_raw_contracts_fixture LANGUAGES CXX)
enable_testing()
get_filename_component(source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# CPU-only lifecycle gate for the serving RAW adapter contract: phase permissions, completion-cut acceptance,
# destruction/failstop rules, and the capture-metadata / tracked-state predicates the transaction and the serving
# path call (include/strata/core/serving_raw_metadata.hpp).  window_hash.cpp is the production digest the gate
# derives certificate histories with; it is CPU-only and has no device dependency.  Proves no device behavior, no
# import and no certificate.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_executable(serving_raw_contracts_test test.cpp "${source_root}/src/core/window_hash.cpp")
target_compile_features(serving_raw_contracts_test PRIVATE cxx_std_20)
target_compile_options(serving_raw_contracts_test PRIVATE -UNDEBUG)
target_include_directories(serving_raw_contracts_test PRIVATE "${source_root}/include")
add_test(NAME serving_raw_contracts_test COMMAND serving_raw_contracts_test)
set_tests_properties(serving_raw_contracts_test PROPERTIES TIMEOUT 30)
# The call-site ordering guard over the real serving entry: the work phase must open before the request's native
# mutations and the drain must follow its last real commit.  Source order only - it proves no device behaviour.
add_test(NAME serving_raw_ordinary_lifecycle_test
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/test_ordinary_lifecycle.py --build-dir ${CMAKE_CURRENT_BINARY_DIR}/ordinary-admission)
set_tests_properties(serving_raw_ordinary_lifecycle_test PROPERTIES TIMEOUT 30)
# Executes the real activation oracle admission/rollback helper with recording CPU operations, including
# postimage mismatch and omitted-production-final-restore negative controls. No native/GPU proof.
add_test(NAME serving_raw_activation_qualification_test
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/test_activation_qualification.py
                 --build-dir ${CMAKE_CURRENT_BINARY_DIR}/activation-qualification)
# Includes the full outer adapter/registry/controller and old-bug negative-control compilations.
# The combined CPU gate takes about 45 seconds; allow bounded headroom on shared build hosts.
# Includes the actual two-stage pending-input/group/observer fixture. Root's
# focused recording run was25s in addition to the existing activation fixtures.
set_tests_properties(serving_raw_activation_qualification_test PROPERTIES TIMEOUT 150)
