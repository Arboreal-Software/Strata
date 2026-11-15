find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
get_filename_component(STRATA_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(units core/expert_source.cpp core/expert_cache.cpp core/pinned.cu
 kernels/cpu/expert_layout.cpp kernels/cpu/hadamard.cpp
 core/verified_inputs.cpp core/verified_artifact.cpp core/window_budget.cpp core/window_hash.cpp
 platform/memory.cpp platform/direct_file.cpp core/expert_policy_inventory.cpp core/expert_policy_runtime.cpp)
list(TRANSFORM units PREPEND "${STRATA_SOURCE_ROOT}/src/")
set_source_files_properties("${STRATA_SOURCE_ROOT}/src/core/pinned.cu" PROPERTIES LANGUAGE CXX)
add_executable(remote_policy_publication_test remote_policy_publication_test.cpp
 "${STRATA_SOURCE_ROOT}/src/core/remote_policy_publication.cpp" ${units})
target_include_directories(remote_policy_publication_test PRIVATE
 "${STRATA_SOURCE_ROOT}/include" "${STRATA_SOURCE_ROOT}/src/core")
target_compile_features(remote_policy_publication_test PRIVATE cxx_std_20)
target_compile_definitions(remote_policy_publication_test PRIVATE STRATA_NATIVE_EXPERTS=1)
target_compile_options(remote_policy_publication_test PRIVATE -O0 -g0 -UNDEBUG -Wall -Wextra
 -ffunction-sections -fdata-sections -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(remote_policy_publication_test PRIVATE -Wl,--gc-sections -fsanitize=address,undefined)
target_link_libraries(remote_policy_publication_test PRIVATE CUDA::cudart Threads::Threads)
foreach(mode 0 2 3 4 5 6)
 add_test(NAME remote_policy_publication_${mode} COMMAND remote_policy_publication_test ${mode})
 set_tests_properties(remote_policy_publication_${mode} PROPERTIES TIMEOUT 45
 ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1;CUDA_VISIBLE_DEVICES=-1")
endforeach()
