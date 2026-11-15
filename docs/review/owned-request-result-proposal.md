# Proposed owned request/result storage

**Design only; no helper, protocol or positive installed path is implemented.** Current builtin logprob objects can outlive a native decision through collectors, generators, monitor records and independently retained nested values. Native DONE, HTTP completion and process replacement do not establish their last free. The proposal below needs concrete source, pricing, compatibility and failure proofs before implementation.

## Wire path and payment

Keep ordinary Python helpers and their builtin results unchanged. Add distinct private strict wire adapters carrying a sealed opaque native span per admitted result. Aliases retain that span; no payload getter, iterator, buffer export, pickle, subclass factory or uncharged child view is exposed. Accumulator/selection/range nodes have source-enumerated finite construction bounds and explicit payers. Every writer buffer, parser, formatting temporary, envelope field retained by a new cache, and overlapping consumer peak must be quoted before allocation. Refusal preserves existing features under unchanged budgets; it cannot become a silent replacement for delivering supported logprobs. Existing byte or claim pressure may defer or refuse LP preparation or replacement startup. This availability tradeoff promises neither unlimited progress nor a hidden queue cap, and it permits no feature or output-format loss.

A real drained/closed controller family admits a disjoint setup addition before new setup storage. A captured-channel offer and complete installation acknowledgement bind protocol, source/ABI, family, epoch, nonce, bytes and claim before READY or result grants. Existing setup commitments are not spare funding. The fixed table is bounded by total existing 16/48 claims across all claim classes, not a promise of free capacity. Import, module/type, tokenizer snapshot, frame/control, channel, queue, error and teardown storage need actual inventory and last-free payment; a guessed bootstrap fee is insufficient. Existing pre-entry/enclosing growth cannot be described as admitted before construction merely because admission occurs later.

After count/hash/checked quote over borrowed request values, genuine admission precedes arena construction. Replay checks counts and digest, but is not a hostile-writer atomicity certificate. Read framed LP bytes directly into paid native storage, not an intermediate Python line/string. Preserve the original cancel Event, captured epoch/channel and existing control ownership. Each additional model/MCP decision receives its own grant while earlier collected results remain paid.

## Exact semantics and import

Preserve native selected token, row traversal, comparator/ties, duplicate allowed-ID occurrence order, chosen-ID reinsertion, emitted `%.6f` then parse, filtering/error order, UTF-8 tails, token spelling and structured recovery/intentional omission. Match supported CPython float parsing, exp/log, version-specific builtin summation, stable sorting and JSON formatting/escaping. Distribution equivalence or a C++ formatting approximation is not enough. Dedicated count/write, SSE, monitor and error adapters must preserve the wire output and the existing 262144-character pretty-monitor prefix without unpaid full-body intermediates.

On child replacement, parent-held obligations are foreign debt, not surviving child leases. Confirm old-child death and freeze a bounded versioned snapshot before replacement model reads. Pin entries across import, including frees arriving during that transaction. An early checked headroom test and both genuine model-budget paths retain the debt separately from new child-local allocations. Both model roots must install genuine canonical FOREGROUND reservations for imported debt, with imported bytes, full claim multiplicity and additional control storage genuinely priced. Never fabricate an old HostCredit or private_work parent, and never simultaneously pledge aggregate P and per-entry P against the same payment. Session-first reconciliation must subtract only proven child-local reservation bytes, never foreign debt from observed baseline. Install imported bytes and claims transactionally before READY/new grants; imported metadata is additional. Exact receipts map original family/nonces to successor records. Partial transfer failure is not rollback, and process failstop is not process-surviving ledger escrow.

## Last free and supported build

Span/control/arena storage must actually free while its payer remains alive. Only an authenticated one-time retirement receipt can refund; ACK loss/backpressure keeps debt charged. Final deallocation cannot allocate a queue node, construct Python/socket objects, format an exception or depend on a Python worker surviving finalization. Outer cycles may delay frees indefinitely. Every escaping error/frame/type/module/weak/control tail needs a payer through its actual last free.

The proposed helper is private Linux LP64 CPython 3.11–3.14, GIL/full API only. Reject free-threaded and limited-API builds at compile time. Bind matching interpreter/development module, SOABI, tokenizer/protocol/source identities and actual installation before setup acknowledgement. No stable-ABI or free-threaded promise is made.

Open blockers include numeric disjoint quotes, bootstrap/import/enclosing/control payment, finite node/frame construction, real import transactions, exact CPython semantics/loader/finalization, installed ABI matrix and any required strict builtin-return API. Current mode-0 copy has unchecked enqueue limitations; a partial launch or absent effective-child receipt is not completion evidence. Native uncertain-completion whole-executor quarantine is not an implemented guarantee. These are explicit review obligations, not accepted assumptions or hidden refusals.

## Proposed implementation targets

This is the fifteen-path proposal, not implemented code or an edit grant. Existing-file links let reviewers inspect current callers and boundaries; the three explicitly new files are absent today. Exact SDK/build placement and every open proof above must close before implementation.

| Target | State | Proposed purpose |
| --- | --- | --- |
| [serve/server.py](../../serve/server.py) | Existing source | Strict wire adapters, captured bootstrap/restart and ownership through collectors/writers/errors. |
| [src/program/generate.cpp](../../src/program/generate.cpp) | Existing source | Controller-cut grants/retirement and debt import at both genuine model-budget roots. |
| [src/program/suffix_raw_runtime.hpp](../../src/program/suffix_raw_runtime.hpp) | Existing source | Narrow family-bound result/import declarations; no fake open request. |
| [src/program/suffix_raw_runtime.cpp](../../src/program/suffix_raw_runtime.cpp) | Existing source | Real H/global admission, claim debit, setup funding and uncertain retirement. |
| [src/program/owned_image_request_operation.hpp](../../src/program/owned_image_request_operation.hpp) | Existing source | Only a disjoint setup-term accessor if needed; existing producer/TWO quotes unchanged. |
| `include/strata/core/owned_result_protocol.hpp` | Proposed new file | Fixed LP messages, typed sizing and generation state. |
| `src/python/owned_result.cpp` | Proposed new file | Private compiled arena/span, paid table and direct bounded writer. |
| [CMakeLists.txt](../../CMakeLists.txt) | Existing source | Matching CPython helper build/install and supported ABI checks. |
| [cmake/strata_gpu_engine.cmake](../../cmake/strata_gpu_engine.cmake) | Existing source | Only required native/helper and existing-test linkage. |
| [serve/test_logprobs.py](../../serve/test_logprobs.py) | Existing source | Preserved exact UTF-8, allowed-ID, rounding, stream/collect semantics and ownership. |
| [serve/test_logprobs_parallel.py](../../serve/test_logprobs_parallel.py) | Existing source | Original Event/cancel authority, paid construction and slow-client/failure paths. |
| [serve/test_engine_epoch.py](../../serve/test_engine_epoch.py) | Existing source | Death/import/restart/late ACK and retained aliases across epochs. |
| [serve/test_batch_control_guard.py](../../serve/test_batch_control_guard.py) | Existing source | Actual acquired-extent separation and quarantine producer closure. |
| [serve/test_server.py](../../serve/test_server.py) | Existing source | Writer/error/monitor ownership and unchanged public JSON assertions. |
| `src/core/owned_result_protocol_test.cpp` | Proposed new file | Actual protocol conservation, no-allocation refusal, claims/import/final-free faults. |

`tools/strata_tokenizer.py` remains a read-only representation/equivalence input. Any required edit there or beyond these targets needs a named dependency disposition; no generic allocator, queue or default-policy redesign is proposed.
