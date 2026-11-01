cmake_minimum_required(VERSION 3.18)
project(owned_batch_slot_sessions_fixture LANGUAGES CXX)
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME owned_batch_slot_sessions_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/test.py
          --build-dir ${CMAKE_CURRENT_BINARY_DIR}/actual-seam)
set_tests_properties(owned_batch_slot_sessions_test PROPERTIES TIMEOUT 60)

add_executable(owned_admission_cancel_test admission_cancel.cpp)
target_compile_features(owned_admission_cancel_test PRIVATE cxx_std_20)
target_include_directories(owned_admission_cancel_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../src/program)
add_test(NAME owned_admission_cancel_test COMMAND owned_admission_cancel_test)
set_tests_properties(owned_admission_cancel_test PROPERTIES TIMEOUT 10)
