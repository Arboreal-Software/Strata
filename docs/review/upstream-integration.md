# Owned-serving integration review

This feature branch presents a coherent source snapshot for review. Its single parent is public upstream commit `13eedf552417a6a791a7d13c6597e24300ee3a09`. The snapshot combines that upstream with work derived from local commit `17efb9e1480ec07232c124b27e2ad8f56f51301b`; it is not a completed two-parent merge. The source snapshot manifest is `4c85b275c8d177b12995747bcfa9847a0313c4f3bd4945051f77e017fdaa9418`. Existing repository attributes govern Git line-ending normalization. These review documents are the only additions to that snapshot. Two documentation-only files are omitted: `docs/qualification/owned-suffix-repair/README.md` and `docs/qualification/owned-suffix-repair/evidence-index.json`. They describe an older recovery checkpoint and index local operational receipts; they are not current qualification evidence. No product, test or runtime DATA file is omitted. The originals and both unrelated dirty-document versions remain local. No direct runtime or test reference to either omitted file was found in the bounded static audit; generic Markdown checks would otherwise enumerate the README.

The implemented changes concern owned serving, capture/restore lifetimes, paid startup/prefill and position storage, image request transfer, and verified model/input loading. Review the actual diff from the parent, including tests. The separate [request/result proposal](owned-request-result-proposal.md) remains unimplemented; it is not a claim that current Python logprob results have final-free payment. [API compatibility](logprob-api-compatibility.md) describes that boundary.

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

## Read the code

Start with [build registration](../../CMakeLists.txt) and [engine target wiring](../../cmake/strata_gpu_engine.cmake). Follow [RAW ownership](../../src/program/suffix_raw_runtime.cpp), [owned slots](../../src/program/owned_batch_slot_sessions.hpp), [verified input manifests](../../src/core/verified_input_manifest.cpp), [native dense loading](../../src/core/native_dense.cpp), and their declarations/tests. Then read [server request and response consumers](../../serve/server.py), [image transport](../../serve/images.py), and the focused tests beside them. The proposal/API documents describe the unimplemented boundary; they are not a replacement for reading this diff.
