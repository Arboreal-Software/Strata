# Serial verifier copy completion

Prepared source fixtures, not executed. The production change is based on PR25
390356e09b3bff0e9d433c1aa3f280f7b1a88283. Both unchecked serial copy-stream
synchronizations are covered: the overlapped stage-chain return and the ordinary
serial return. `fixed.inc` contains the exact new helper and exact completion
blocks from `Verifier::run`; `before.inc` preserves both original blocks from that base; its reviewed SHA-256
is included in `source-pins.json` so changes to the old control require review. The
wrappers retain the existing entry refusal and omit already-completed GPU/kernel
work. This does not test the full verifier, remote controller, CUDA or HIP runtime.

The mandatory freshness checker compares the helper/callsite blocks, actual header
guard, atomic declaration, source hashes, and existing run/init/batch/commit/drain
refusal sites. Source changes require refreshed, re-reviewed fixture pins. It does
not establish numerical correctness or cleanup completion.

The old expected-red runner requires exact assertion exit 1 and only its expected
line, separately for terminal-first and chain-first execution; crashes, timeouts,
other assertions and sanitizer output cannot qualify. Fixed scenarios cover hard
copy errors, sticky retry without device calls, persistent allocation failure in
the diagnostic, retained pending/copy flags, successful copy, unused copy, and a
non-success pending status. The chain matrix combines downstream success/failure,
primary success/error, and copy success/error; it checks call counts, independent
flag transitions, sticky copy failure, and diagnostic ordering. CUDA synchronization and handoff/resident-plan helpers
are CPU fakes; no device or model is involved. Allocation injection affects real
C++ string allocation with global operator new, not a fake returned error string.

After a fresh CPU grant, configure this directory as a standalone CMake project,
build both targets, then run all three CTests. The fixture is also registered
through the root CMake test configuration.
Successful-path arithmetic and kernels were not modified; backend builds and
numerical qualification have not been performed for this change.
