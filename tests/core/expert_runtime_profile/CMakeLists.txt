# This fixture uses CUDA runtime declarations and explicit CUDA API mocks. It
# does not qualify HIP's compatibility layer, SYCL, Metal, or device execution.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT STRATA_ENABLE_CUDA OR
   STRATA_ENABLE_HIP OR STRATA_HIP_GFX906 OR NOT STRATA_BUILD_WINDOW_STATE OR
   NOT STRATA_BUILD_CONVERSATION_TESTS)
  message(FATAL_ERROR "expert runtime profile CPU fixture requires Linux CUDA window/conversation tests")
endif()
find_package(Threads REQUIRED)
get_filename_component(_expert_profile_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
# Exact eleven production units from the standalone synthetic fixture closure.
# Never link strata_engine/strata_kernels_cpu: the fixture supplies native_fmt
# and CUDA seams; linking those libraries would change/duplicate that authority.
set(_expert_profile_units
  core/expert_runtime_profile.cpp
  core/expert_source.cpp
  core/pinned.cu
  kernels/cpu/expert_layout.cpp
  kernels/cpu/hadamard.cpp
  core/verified_inputs.cpp
  core/verified_artifact.cpp
  core/window_budget.cpp
  core/window_hash.cpp
  platform/memory.cpp
  platform/direct_file.cpp)
list(TRANSFORM _expert_profile_units PREPEND "${_expert_profile_root}/src/")
# Source properties are directory-scoped: the normal parent pinned.cu target
# remains CUDA. CMP0119 NEW (project CMake>=3.24) compiles this copy as host C++,
# matching the standalone fixture's explicit -x c++ command.
set_source_files_properties("${_expert_profile_root}/src/core/pinned.cu" PROPERTIES LANGUAGE CXX)
add_executable(expert_runtime_profile_test
  "${_expert_profile_root}/src/core/expert_runtime_profile_test.cpp"
  ${_expert_profile_units})
target_compile_features(expert_runtime_profile_test PRIVATE cxx_std_20)
target_compile_definitions(expert_runtime_profile_test PRIVATE STRATA_NATIVE_EXPERTS=1)
target_include_directories(expert_runtime_profile_test PRIVATE
  "${_expert_profile_root}/include" "${_expert_profile_root}/src/core")
target_compile_options(expert_runtime_profile_test PRIVATE
  -O0 -g0 -UNDEBUG -Wall -Wextra -ffunction-sections -fdata-sections
  -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(expert_runtime_profile_test PRIVATE
  -Wl,--gc-sections -fsanitize=address,undefined)
target_link_libraries(expert_runtime_profile_test PRIVATE CUDA::cudart Threads::Threads)
add_test(NAME expert_runtime_profile_test COMMAND expert_runtime_profile_test)
set_tests_properties(expert_runtime_profile_test PROPERTIES
  TIMEOUT 45
  ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1;CUDA_VISIBLE_DEVICES=-1")
