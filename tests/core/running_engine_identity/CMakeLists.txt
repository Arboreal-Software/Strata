# CPU-only source closure: no engine archive, CUDA/HIP runtime, /proc image read
# or model payload is needed by these synthetic fixtures.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT STRATA_BUILD_WINDOW_STATE OR
   NOT STRATA_BUILD_CONVERSATION_TESTS)
  message(FATAL_ERROR "running engine identity fixtures require Linux optional conversation tests")
endif()
find_package(Threads REQUIRED)
get_filename_component(_running_image_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(running_engine_fixture_options INTERFACE)
target_compile_features(running_engine_fixture_options INTERFACE cxx_std_20)
target_include_directories(running_engine_fixture_options INTERFACE "${_running_image_root}/include")
target_compile_options(running_engine_fixture_options INTERFACE
  -O0 -g0 -UNDEBUG -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(running_engine_fixture_options INTERFACE -fsanitize=address,undefined)
target_link_libraries(running_engine_fixture_options INTERFACE Threads::Threads)

# Exact five support units from the frozen standalone fixture. These compile
# without the syscall seam macro, independently of the production libraries.
add_library(running_engine_fixture_support OBJECT
  "${_running_image_root}/src/core/verified_input_manifest.cpp"
  "${_running_image_root}/src/core/verified_inputs.cpp"
  "${_running_image_root}/src/core/verified_artifact.cpp"
  "${_running_image_root}/src/core/window_budget.cpp"
  "${_running_image_root}/src/core/window_hash.cpp")
target_link_libraries(running_engine_fixture_support PRIVATE running_engine_fixture_options)

add_executable(running_engine_identity_test
  "${_running_image_root}/src/core/running_engine_identity.cpp"
  "${_running_image_root}/src/core/running_engine_identity_test.cpp"
  $<TARGET_OBJECTS:running_engine_fixture_support>)
# PRIVATE on this executable only: the production collector in
# strata_window_state is a separate compilation with no test syscall routing.
target_compile_definitions(running_engine_identity_test PRIVATE STRATA_RUNNING_ENGINE_IDENTITY_TEST)
target_link_libraries(running_engine_identity_test PRIVATE running_engine_fixture_options)

# Also preserve the standalone strict-manifest regression from the same frozen
# closure; the existing engine-linked verified_input_manifest_test is unchanged.
add_executable(running_engine_manifest_test
  "${_running_image_root}/src/core/verified_input_manifest_test.cpp"
  $<TARGET_OBJECTS:running_engine_fixture_support>)
target_link_libraries(running_engine_manifest_test PRIVATE running_engine_fixture_options)
foreach(_test IN ITEMS running_engine_identity_test running_engine_manifest_test)
  add_test(NAME ${_test} COMMAND ${_test})
  set_tests_properties(${_test} PROPERTIES TIMEOUT 45
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1")
endforeach()
