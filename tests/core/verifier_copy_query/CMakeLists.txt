# Include via add_subdirectory(tests/core/verifier_copy_query) in an opted-in CPU test build.
# No CUDA SDK, engine library, GPU, model or native kernel linkage is required.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(copy_query_dir "${CMAKE_CURRENT_LIST_DIR}")
get_filename_component(copy_query_root "${copy_query_dir}/../../.." ABSOLUTE)
add_custom_target(verifier_copy_query_freshness
  COMMAND "${Python3_EXECUTABLE}" "${copy_query_dir}/check_freshness.py"
  DEPENDS "${copy_query_root}/src/core/verify.cpp"
          "${copy_query_root}/src/core/verify_drain.cpp"
          "${copy_query_root}/include/strata/core/verify.hpp"
          "${copy_query_dir}/fixed-methods.inc" "${copy_query_dir}/prefix.inc"
          "${copy_query_dir}/fixed.cpp" "${copy_query_dir}/before.cpp"
  VERBATIM)
foreach(variant IN ITEMS before fixed)
  add_executable(verifier_copy_query_${variant} "${copy_query_dir}/${variant}.cpp")
  target_compile_features(verifier_copy_query_${variant} PRIVATE cxx_std_20)
  add_dependencies(verifier_copy_query_${variant} verifier_copy_query_freshness)
endforeach()
add_test(NAME verifier_copy_query_freshness_test
  COMMAND "${Python3_EXECUTABLE}" "${copy_query_dir}/check_freshness.py")
set_tests_properties(verifier_copy_query_freshness_test PROPERTIES
  FIXTURES_SETUP verifier_copy_query_current TIMEOUT 20)
add_test(NAME verifier_copy_query_expected_red_test
  COMMAND "${Python3_EXECUTABLE}" "${copy_query_dir}/expect_red.py" $<TARGET_FILE:verifier_copy_query_before>)
foreach(mode IN ITEMS copy-error allocation success)
  add_test(NAME verifier_copy_query_${mode}_test COMMAND verifier_copy_query_fixed --${mode})
  set_tests_properties(verifier_copy_query_${mode}_test PROPERTIES
    FIXTURES_REQUIRED verifier_copy_query_current TIMEOUT 30)
endforeach()
set_tests_properties(verifier_copy_query_expected_red_test PROPERTIES
  FIXTURES_REQUIRED verifier_copy_query_current TIMEOUT 30)
