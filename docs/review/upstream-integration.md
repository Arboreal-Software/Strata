# Owned-serving integration review

The published code snapshot `9e275416af8063debf50bd722690de8cfea64e29` has single public upstream parent `13eedf552417a6a791a7d13c6597e24300ee3a09`. The fixed-family design correction is published as `5696fa7c1ec1db10233be268f80093c244f76013`, a single-parent child of that snapshot. The serving correction `7a37d42adb18342abd1710192c136be9b62e1a9e` is a child of that design commit. Its shutdown and stale-notification follow-up is `24df796bd3ac62df595bfba61a5d9fa85fdf1210`; the present failed-READY repair is based on that follow-up. The snapshot combines that upstream with work derived from local commit `17efb9e1480ec07232c124b27e2ad8f56f51301b`; it is not a completed two-parent merge. The source snapshot manifest is `4c85b275c8d177b12995747bcfa9847a0313c4f3bd4945051f77e017fdaa9418`. Existing repository attributes govern Git line-ending normalization. These review documents are the only additions to that snapshot. Two documentation-only files are omitted: `docs/qualification/owned-suffix-repair/README.md` and `docs/qualification/owned-suffix-repair/evidence-index.json`. They describe an older recovery checkpoint and index local operational receipts; they are not current qualification evidence. No product, test or runtime DATA file is omitted. The originals and both unrelated dirty-document versions remain local. No direct runtime or test reference to either omitted file was found in the bounded static audit; generic Markdown checks would otherwise enumerate the README.

The implemented changes concern owned serving, capture/restore lifetimes, paid startup/prefill and position storage, image request transfer, and verified model/input loading. Review the actual diff from the parent, including tests. The separate [request/result proposal](owned-request-result-proposal.md) remains unimplemented; it is not a claim that current Python logprob results have final-free payment. [API compatibility](logprob-api-compatibility.md) describes that boundary.

Independent exact-GitHub review of that head returned **DESIGN_CHANGES_REQUIRED**: `max_rounds=64` can retain 65 decisions, exceeding 48 shared claims under the old per-decision design. The correction proposes upfront fixed-claim family admission and claim-free decision subaccounts. It is UNIMPLEMENTED and unaccepted; all eight pre-edit bootstrap/import/launch/final-free/semantics/API/install/native proofs stay open. Existing APIs do not prove payment before new construction.

## Recorded validation

Separate lead-run checks used the same source snapshot. They were not rerun while preparing this branch. Private operational receipts remain local.

| Check | Recorded result | Boundary |
| --- | --- | --- |
| Default OFF and optional ON, fault OFF engines | Both compiled and linked with GNU 13.3 / CUDA build profile | No engine/model/GPU execution |
| `conversation_validation_test`, `conversation_transfer_test`, `window_state_test`, `window_budget_test`, `native_owned_batch_serving_test` | Five selected controls passed | Host/recording scope |
| `owned_batch_slot_sessions_test` | Original full 60-second gate passed in 35.4992 seconds | Fourteen fresh primary builds, thirteen negative controls; recording native boundary |
| `window_hash_test`, `thread_priority_test`, `thread_dependencies_test` | Three independent cold builds and original checks passed | Host-only; no cache reuse |
| Ordinary Python coverage | 134 cases accounted for: 87 current-snapshot passes and 47 source-bound retained passes | Separate runs, not one aggregate run |
| App native-control and native-gateway | Seven and four original cases passed respectively | Supplied fake control/HTTP engines; no provider/model |
| App request-flow | Six original cases passed in a separate 8.84463-second supervised run; independent cleanup closed | Genuine GPU command unavailable in the bound CPU profile; normal Docker init and conclusive esbuild-child reap observed; no full-app/provider qualification |

Failures are not erased by later passes. Earlier build/link and fixture failures were corrected in later immutable source/fixture versions. The first host-hash preparation refused a source-mode mismatch before compiler/container execution; its successor bound actual immutable metadata without changing source bytes.

Native completion, real-device model behavior, HIP/Metal/SYCL compatibility, whole-project acceptance, allocator/RSS proofs, and the unimplemented request/result architecture remain open. Compilation or a recording backend does not close them. Required independent commit, subsystem and final reviews remain separate gates. Keep this work in review; no merge, deployment or default-performance claim follows.

## Focused serving correction

The full hosted server suite at the original snapshot failed: 1,743 tests in 649.719 seconds, with five failures, seventeen errors and twenty-four skips. Those traces remain a failed full-suite result. They establish a never-started ordinary-epoch restart error and fixture/lifecycle failures.

The correction limits empty-epoch reaping to never-started ordinary engines, closes the captured pipe on the existing bounded shutdown worker when the child exits between polls, and repairs the named fixture preconditions. Captured-slot fixtures update their existing lists in place; the final single-group subcase starts with a separate ready fixture configured before its first capture. Existing assertions and expected values are preserved. Four added ordinary regressions cover missing channels, strict/live fencing, retained descriptors and the EOF poll race. This narrow EOF fix does not establish universal close ordering for every shutdown path.

After inherited nonce/terminal checks and raw diagnostic callbacks, strict-owned mode consumes only the seven known dedicated owner-notification prefixes. Unknown/error records retain their old routing and strict image-reply rejection. Group, stop and admission-cancel notifications remain informational, without new schema authority. Five added regressions cover all three image waits in both terminal orders, subsequent control traffic, scheduler matching, diagnostic order and uncertain descriptor retention. No new parser, assigned locals or claim is introduced. Immutable code/constants and global module payment remain unproved; existing quotes are not assumed spare.

One bounded CPU run passed all 36 selected methods in 2.550 seconds, with no skips, errors or failures: eighteen methods that failed in the hosted run, nine existing guards, four ordinary regressions and five notification regressions. It used CPython 3.12.3 and the bound genuine package profile, with GPU-driver loading denied, private HTTP/fork fixtures and real sealed descriptors. Source/package/startup guards passed before and after, and independent process/container absence plus input readback closed in 10.616 seconds. Native notification producers and scheduler sinks remain recording boundaries. Earlier pretest-profile and captured-fixture failures are preserved separately. This focused pass does not replace the full hosted suite or establish native execution, allocation payment, ABI/RSS, architecture or release acceptance. Independent review of the later `24df796bd3ac62df595bfba61a5d9fa85fdf1210` follow-up locally closed the shutdown and stale-notification mechanisms but required the inherited failed-READY repair below. Independent review of this new repair remains open.


## Shutdown and stale-notification follow-up

This successor retains one captured shutdown owner across uncertain retries: process, stdin, original pump and log, worker, completion event and errors. Buffered stdin close runs on that worker before reader join. A failed writer requires confirmed captured-child exit before buffered close, which can flush a partial frame; no new STOP/QUIT is sent after failure. Retry joins the same unfinished worker, and timeout retains the captured owner and restart fence. Known-closed broken-pipe cleanup remains allowed. Caller deadlines, writer serialization, captured-process checks and uncertain image-descriptor fences remain in place. This is a fixed ordinary-control record under unchanged caps, not a paid cell or a proof of module funding, preconstruction payment, final-free accounting or RSS. It does not establish boundedness of every possible close operation.

Native COMMIT/KNOWN reports observe completed work only and are optional tracing. Their stdout records grant no identity, admission, credit, data, terminal or retirement authority. The real scheduler consumer ignores those prefixes; raw diagnostic callbacks and existing JSON/slot/post-half guards still precede their consumption. ACCEPT/DONE retain their separate lifecycle authority and terminal-pair checks. Late stale diagnostics after a complete pair are tolerated without changing the replacement request. The image protocol registers expected C before BIPAY/BICAPTURED/BICLOSED; native ACCEPT C follows the image handshake and precedes BADM or token data. The regression defers the existing admission/data publisher until that ACCEPT and post-accept diagnostic checks, rather than fabricating an accepted C during image waits. Callback delivery is distinct from artifact inclusion: the two-slot recorder includes BOWN records, while the native recorder filters them.

One bounded CPU run passed all 25 selected methods in 2.646 seconds, with no skips, errors or failures: sixteen existing shutdown/write/restart guards, five notification guards, three new shutdown regressions and one new stale-notification method. Source, package, startup-origin and GPU-driver-denial guards passed before and after; independent process/container absence and input readback closed in 6.247 seconds. The profile used CPython 3.12.3, the bound genuine packages and real sealed descriptors, with no GPU or network access. The four new methods are scoped CPU regressions. They cover initially exited and failed-writer EOF ordering, bounded retry of a genuinely blocked buffered close, and stale A diagnostics during expected C image waits and accepted C continuation, with both old terminal orders. Existing assertions and protocol guards are preserved. The exact parent `7a37d42adb18342abd1710192c136be9b62e1a9e` separately completed hosted CI: 1,752 tests, 24 skips, zero failures/errors, in 650.664 seconds under its scoped mock profile. That result does not qualify this successor; the earlier 1,743-test failure remains separate. Native behavior, payment, ABI/RSS and all eight strict result/release proofs remain open.

## Read the code

Start with [build registration](../../CMakeLists.txt) and [engine target wiring](../../cmake/strata_gpu_engine.cmake). Follow [RAW ownership](../../src/program/suffix_raw_runtime.cpp), [owned slots](../../src/program/owned_batch_slot_sessions.hpp), [verified input manifests](../../src/core/verified_input_manifest.cpp), [native dense loading](../../src/core/native_dense.cpp), and their declarations/tests. Then read [server request and response consumers](../../serve/server.py), [image transport](../../serve/images.py), and the focused tests beside them. The proposal/API documents describe the unimplemented boundary; they are not a replacement for reading this diff.


## Failed READY retirement correction

This successor retains the actual READY reader in the existing engine pump field.
A failed start must confirm captured-child exit and finish the existing bounded
pipe/reader/log cleanup before an outer or recursive startup retry. Uncertain
retirement raises `EngineStuck` and keeps the original owner; it cannot authorize
a second spawn. Cleanup receives the captured process identity and rejects a
mismatched successor before any scheduler or pipe action. Public close calls
retain their prior default behavior.

The new regressions cover unresolved five- and sixty-second wait paths, an
exited child with a blocked READY reader, completely retired outer/internal
retry, and captured-close mismatch. One bounded CPU run passed all 18 selected
methods in 9.296 seconds, with zero skips, errors or failures: thirteen existing
guards and five new regressions. It used the bound CPython 3.12.3 package profile
and actual private Python children, READY threads and pipes. Source, package,
startup-origin and device-audit guards passed before and after; independent
process/container absence and input readback closed in 13.151 seconds. Controlled
wait/kill faults are not actual uninterruptible-child trials. The exact parent
`24df796bd3ac62df595bfba61a5d9fa85fdf1210` separately passed hosted CI: 1,756
tests, 24 skips, zero failures/errors, in 623.936 seconds. That earlier CI did
not cover this inherited failure. This new repair still requires its own hosted
CI and exact-commit review. The existing close worker and
seven-reference retained owner are reused; additional borrowed references and
call-frame costs remain unmeasured, without a new payment or global-bound claim.
Notification routing, strict FD fences, retry policy and production deadlines
are unchanged. No native, device, RSS or whole-engine qualification follows.
