# Submitted-work controller teardown fixture

Uses the production `SubmittedWorkDrainGuard` header and exact driver pump-wrapper/drain extraction. The underlying pump and doorbell/controller lifetime markers are mocks; no verifier, GPU, or model is executed.

Eight forked cases cover normal completion, preexisting uncertainty, false/throwing pump, exception unwind, disabled guard, refused retry, and early input rejection followed by draining submitted work. Shared markers check that nested doorbells join first and uncertain teardown cannot destroy callback targets.

Run `check_extract.py` against `src/program/generate.cpp` and `controller.inc` before compiling; it refuses a stale extraction. Compile `test.cpp` as C++20 with assertions enabled, pthread, and include directories for this folder and `include/strata/core`. The fixture is POSIX-only. No CUDA library is required. ASan/UBSan are suitable; failure cases intentionally use process exit.

The exact fixture passed its standalone sanitizer harness. Public build registration is separate; this directory does not claim full driver/backend qualification.
