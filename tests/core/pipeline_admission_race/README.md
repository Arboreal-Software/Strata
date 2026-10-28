# Pipeline input admission race fixture

Uses the production submitted-work guard, exact driver admission prefix and exact uncertainty wrapper/drain. A controlled input load returns the admitted value and then flips terminal input before the actual step prefix. This represents the precise interleaving without scheduling sleeps; backend work is mocked.

The fixed fixture covers the racing admission, clean rejection before admission, a real backend failure remaining sticky, and legacy step admission. The old-prefix variant removes only the owned guard around the second check; it must fail the precise admitted-race assertion with exit1. The exact standalone sanitizer fixture passed both expectations. This does not qualify full pipeline or GPU behavior.

Before compilation, `check_extract.py` must compare the current `src/program/generate.cpp` with this fixture directory. It checks both actual extracts and the exact old-prefix delta. The build target must depend on this freshness check. Compile as C++20 with assertions, pthread and `include/strata/core` on the include path. POSIX-only; no CUDA library required. Build the expected-old variant with `OLD_ADMISSION` defined and invoke it with `race`; crashes or sanitizer errors are not expected-red evidence.
