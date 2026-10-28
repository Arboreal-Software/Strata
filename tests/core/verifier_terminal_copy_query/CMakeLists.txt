# Standalone CPU mock fixture: no engine, CUDA SDK or runtime library.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(terminal_dir "${CMAKE_CURRENT_LIST_DIR}")
add_custom_target(verifier_terminal_copy_freshness
  COMMAND "${Python3_EXECUTABLE}" "${terminal_dir}/check_freshness.py" VERBATIM)
foreach(variant IN ITEMS before fixed)
  add_executable(verifier_terminal_copy_${variant} "${terminal_dir}/${variant}.cpp")
  target_compile_features(verifier_terminal_copy_${variant} PRIVATE cxx_std_20)
  target_compile_options(verifier_terminal_copy_${variant} PRIVATE -UNDEBUG)
  add_dependencies(verifier_terminal_copy_${variant} verifier_terminal_copy_freshness)
endforeach()
add_test(NAME verifier_terminal_copy_freshness_test
  COMMAND "${Python3_EXECUTABLE}" "${terminal_dir}/check_freshness.py")
set_tests_properties(verifier_terminal_copy_freshness_test PROPERTIES FIXTURES_SETUP terminal_copy_current TIMEOUT 20)
add_test(NAME verifier_terminal_copy_expected_red_test
  COMMAND "${Python3_EXECUTABLE}" "${terminal_dir}/expect_red.py" $<TARGET_FILE:verifier_terminal_copy_before>)
set_tests_properties(verifier_terminal_copy_expected_red_test PROPERTIES FIXTURES_REQUIRED terminal_copy_current TIMEOUT 30)
foreach(mode IN ITEMS copy-error allocation pending success)
  add_test(NAME verifier_terminal_copy_${mode}_test COMMAND verifier_terminal_copy_fixed --${mode})
  set_tests_properties(verifier_terminal_copy_${mode}_test PROPERTIES FIXTURES_REQUIRED terminal_copy_current TIMEOUT 30)
endforeach()
