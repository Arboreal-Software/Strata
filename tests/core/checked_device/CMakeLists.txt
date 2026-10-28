cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(checked_device_public_fixture LANGUAGES CXX)
  enable_testing()
endif()
# The snapshot preserves the exact standalone CPU fixture that was exercised.
# This repository wrapper additionally binds its copied production files to HEAD.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(checked_device_repository_freshness
  COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_LIST_DIR}/check_repository.py"
  VERBATIM)
add_subdirectory(snapshot)
add_dependencies(checked_device_fixture checked_device_repository_freshness)

# Assertions execute parts of this fixture: Release must not erase them.
# Apply after the unchanged historical snapshot defines its target.
if(MSVC)
  target_compile_options(checked_device_fixture PRIVATE /UNDEBUG)
else()
  target_compile_options(checked_device_fixture PRIVATE -UNDEBUG)
endif()
target_sources(checked_device_fixture PRIVATE "${CMAKE_CURRENT_LIST_DIR}/assertions_enabled.cpp")
