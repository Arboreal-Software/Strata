# Retirement worker CPU seams

These four standalone fixtures preserve the exact source used by the corrected worker-close candidate which the coordinating session reported compiled and passed under ASan/UBSan. Promotion itself has not been rebuilt. They use actual extracted lifecycle methods with explicitly mocked surrounding services, not the complete engine.

- workers: RouterLookahead/ForesightSwap declarations and lifecycle methods; mock source/device functions, stub numeric worker and fixture-only destructors.
- file: complete FileExpertSource stop method and reentry helper; mocked thread joins/retained storage sentinels.
- pool: actual dispatch guard, stop and public wrappers; substituted numeric dispatch/wake service with real CPU threads.
- driver: actual background-close helper; retained stub owners test order, refusal/exception short circuit and repeated calls.

No fixture proves external controller/doorbell closure, real filesystem IO completion, GPU drain, numeric output, full production destructor behavior or backend integration. The subsequent pipeline guard outside the background helper is NOT covered.

`check_extracts.py SOURCE_ROOT` is a non-writing freshness/composition checker. It must be a mandatory build dependency of every fixture target, not only an optional test. It compares selected production declarations/control fields, complete methods/helpers, and test composition. Pool fragments are complete verbatim contiguous blocks. The mock substitutes are intentionally visible and remain review obligations; unchanged production excerpts cannot prove mocks faithfully model every runtime effect.

Build with C++17, Threads and assertions enabled. CPU sanitizer runs add ASan/UBSan without GPU libraries. Fixtures should have a30-second CTest timeout. A stale checker failure requires reviewing and regenerating the actual source extract and rerunning qualification; do not merely update a source hash. Source scans use balanced braces and exact text, so unsupported formatting/syntax changes fail closed.

Pool opt-in follow-up (prepared, not rerun): actual legacy/checked destructor now included with mocked diagnostic globals. Constructor mock models the immutable flag; checker verifies actual default argument/initializer and owned-driver construction fragment. Disabled wrapper tests preseed failed bookkeeping to expose any guard access, preserve original exception and subsequent dispatch, and require checked stop refusal without waking. Enabled cases preserve active/failure/retention semantics. A fixture-only reset of a failed checked pool to joined occurs after assertions because the numeric mock retains no borrower; production offers no such recovery. No numerical/default performance claim.

Fresh default coverage: the checked-stop refusal is exercised from initial idle state before poisoned-state cases, checking no stop/wake/state effect and successful subsequent dispatch. This prevents an already-failed checked CAS from masking removal of the default mode guard. The revised CPU fixture passed with ASan/UBSan; removing only the actual default-mode guard triggers the exact fresh-stop refusal assertion. An earlier mutation attempt stopped during preparation because its whitespace-sensitive guard needle was absent; the corrected mutation completed. This validates mocked stop/dispatch control, not worker numerics or GPU execution.
