# Serial adaptation lifetime fixture

Uses the production `JoinedThreadScope` header and exact serving/nonserving guard, thread-entry and EOS extracts. Adaptation, thread naming and affinity are mocked; no model, CUDA or real adaptation work runs.

Thirteen forked scenarios cover callback success/failure/exception, caller exception unwind, default entry, no launched job and a forbidden serving pipeline fence. Condition variables coordinate delayed callbacks borrowing tracked state. A separate disabled-guard case verifies it does not join. The nonserving EOS branch must return failure for an owned callback failure rather than report successful completion.

The original thirteen scenarios and disabled-guard check passed their standalone ASan/UBSan harness. The added controller-entry coverage below is prepared but has not been compiled or run. This does not qualify public build registration or prove full controller/GPU closure. The two-window speculative controller is outside this guard's contract.

Before building, run `check_extract.py` with `src/program/generate.cpp` and this fixture directory. The build registration must depend on that check; stale extraction is an error. Compile `test.cpp` as C++20 with assertions, pthread and `include/strata/core` in the include path. This fixture is POSIX-only; no CUDA library is needed.

## Run-to-launch coverage

Nine additional forked cases compile contiguous production extracts from each preceding `ver.run` call through worker launch, including failure exits, intervening diagnostics and the serving fence check. The mock verifier synchronously services three pending caller-owned work items; no helper thread releases that work. A fixture-only namespace indirection observes thread construction before scheduling and delegates to a real `std::thread`. The actual production `JoinedThreadScope` joins that thread through its real thread reference.

Normal entry and caller unwind require completed preceding work. Verifier failure and expert-pool failure must create no worker. A forbidden serving fence must exit with status1 and a shared construction counter still zero; checking the constructor avoids confusing an unscheduled callback with an uncreated worker. Non-serving has no fence parameter.

The freshness checker compares complete run-to-launch spans and requires the unique original thread declarations to remain within those spans. It detects stale extracts and movement across the selected run boundary; it is not a C++ control-flow proof. Diagnostic device copies, verifier numerics, real doorbell workers, model loading and the two-window controller remain unqualified. The existing public target already compiles `test.cpp` and depends on the checker; no CMake or production change is needed.
