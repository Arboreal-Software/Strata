# Serial adaptation lifetime fixture

Uses the production `JoinedThreadScope` header and exact serving/nonserving guard, thread-entry and EOS extracts. Adaptation, thread naming and affinity are mocked; no model, CUDA or real adaptation work runs.

Thirteen forked scenarios cover callback success/failure/exception, caller exception unwind, default entry, no launched job and a forbidden serving pipeline fence. Condition variables coordinate delayed callbacks borrowing tracked state. A separate disabled-guard case verifies it does not join. The nonserving EOS branch must return failure for an owned callback failure rather than report successful completion.

The exact fixture passed its standalone ASan/UBSan harness. This does not qualify public build registration or prove full controller/GPU closure. The two-window speculative controller is outside this guard's contract.

Before building, run `check_extract.py` with `src/program/generate.cpp` and this fixture directory. The build registration must depend on that check; stale extraction is an error. Compile `test.cpp` as C++20 with assertions, pthread and `include/strata/core` in the include path. This fixture is POSIX-only; no CUDA library is needed.
