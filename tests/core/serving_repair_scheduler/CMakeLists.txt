cmake_minimum_required(VERSION 3.18)
project(serving_repair_scheduler_fixture LANGUAGES CXX)
enable_testing()
get_filename_component(source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
# Production hard admission, weighted selection and authentic ancestry-age projection.
# CPU only: no device completion or scheduling latency is inferred.
add_executable(serving_repair_admission_test
  "${source_root}/src/program/serving_repair_admission_test.cpp"
  "${source_root}/src/core/window_repair_ancestry.cpp")
target_compile_features(serving_repair_admission_test PRIVATE cxx_std_20)
target_include_directories(serving_repair_admission_test PRIVATE "${source_root}/include")
target_compile_options(serving_repair_admission_test PRIVATE -UNDEBUG)
add_test(NAME serving_repair_admission_test COMMAND serving_repair_admission_test)
set_tests_properties(serving_repair_admission_test PROPERTIES TIMEOUT 30)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME serving_repair_scheduler_seams_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/test_serving_seams.py
          --build-dir ${CMAKE_CURRENT_BINARY_DIR}/scheduler-seams)
set_tests_properties(serving_repair_scheduler_seams_test PROPERTIES TIMEOUT 30)
