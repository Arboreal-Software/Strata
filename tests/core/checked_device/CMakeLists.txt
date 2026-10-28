# The snapshot preserves the exact standalone CPU fixture that was exercised.
# This repository wrapper additionally binds its copied production files to HEAD.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(checked_device_repository_freshness
  COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_LIST_DIR}/check_repository.py"
  VERBATIM)
add_subdirectory(snapshot)
add_dependencies(checked_device_fixture checked_device_repository_freshness)
