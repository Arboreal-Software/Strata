# Retirement worker CPU seams

These four standalone fixtures preserve the exact source used by the corrected worker-close candidate which the coordinating session reported compiled and passed under ASan/UBSan. Promotion itself has not been rebuilt. They use actual extracted lifecycle methods with explicitly mocked surrounding services, not the complete engine.

- workers: RouterLookahead/ForesightSwap declarations and lifecycle methods; mock source/device functions, stub numeric worker and fixture-only destructors.
- file: complete FileExpertSource stop method and reentry helper; mocked thread joins/retained storage sentinels.
- pool: actual dispatch guard, stop and public wrappers; substituted numeric dispatch/wake service with real CPU threads.
- driver: actual background-close helper; retained stub owners test order, refusal/exception short circuit and repeated calls.

No fixture proves external controller/doorbell closure, real filesystem IO completion, GPU drain, numeric output, full production destructor behavior or backend integration. The subsequent pipeline guard outside the background helper is NOT covered.

`check_extracts.py SOURCE_ROOT` is a non-writing freshness/composition checker. It must be a mandatory build dependency of every fixture target, not only an optional test. It compares selected production declarations/control fields, complete methods/helpers, and test composition. Pool fragments are complete verbatim contiguous blocks. The mock substitutes are intentionally visible and remain review obligations; unchanged production excerpts cannot prove mocks faithfully model every runtime effect.

Build with C++17, Threads and assertions enabled. CPU sanitizer runs add ASan/UBSan without GPU libraries. Fixtures should have a30-second CTest timeout. A stale checker failure requires reviewing and regenerating the actual source extract and rerunning qualification; do not merely update a source hash. Source scans use balanced braces and exact text, so unsupported formatting/syntax changes fail closed.
