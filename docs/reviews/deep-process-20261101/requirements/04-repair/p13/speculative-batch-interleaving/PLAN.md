# Speculative decoding interleaved with suffix repair

Date: 2026-10-29. Revision 1. Prepared in an owner-directed side conversation.

[Parent repair plan](../PLAN.md) · [Work queue](../../../../WORK-QUEUE.md) · [Review protocol](../../../REVIEW-PROTOCOL.md)

Execution: not-started. Plan: draft; shared-contract review required before implementation.
Integration owner: main Strata session. No source ownership is assigned by this document.
Evidence: source inspection only; no implementation, build, model run, or performance result.

Main queue disposition (2026-10-29): linked into the work queue and parent repair
plan after genuine suffix reuse and in-memory incremental live repair, including
their existing non-speculative acceptance gates. Implementation remains not-started
and shared-contract review is pending. The active qualification campaign and pinned
source/tools were not changed. See the parent's
`speculative-batch-follow-on-ordering-20261029` finding for the review disposition.

## Owner direction and ordering

Owner favored having the existing controller alternate a complete speculative batch with a repair step, then requested a second assessment against concurrent execution and a durable plan/handoff. Final instruction verbatim: "Okay, write this up as a plan in the appropriate location, then send a handoff to main to order this appropriately with its outstanding work, noting that the two top priorities are unchanged".

The existing top priorities remain:

1. Genuine suffix reuse after an edit, with honest approximation, exact fallback, and actual end-to-end qualification.
2. In-memory incremental live repair, initially bounded exact-frontier advancement interleaved with foreground, followed by final exact catchup.

This is a follow-on compatibility extension, not a new prerequisite for either priority. Main should reconcile its placement with current outstanding work, finish or safely release running campaigns, and complete the existing non-speculative serving acceptance path before switching implementation lanes. Preserve GPU/state, answer-quality, authenticated HTTP, profile/selector, resource-release, and conditional deployment gates. Do not hold a qualified non-speculative milestone hostage to this extension or interpret this plan as a deployment grant.

Native verifier buffer capacity and active speculation are distinct. Any current startup fix that allocates verifier capacity while retaining serial execution is not implementation or qualification of this plan.

## Decision: one controller, alternating operations first

Use the existing controller thread to run one speculative batch to a completed committed state, advance private exact repair by a bounded amount, optionally publish repaired state, and repeat. Keep private repair advancement separate from publication so later concurrent computation can use the same publication protocol.

Both serial and concurrent designs need correct batch commit/rollback, accepted history, receipts, drafter invalidation after publication, snapshot contracts, and final catchup. Alternation additionally avoids simultaneous device work, cross-thread history visibility, and competing shutdown ownership. The inspected implementation already has controller-thread checks, an exclusive active-operation marker, a service-pause scope, and all-stage completion requirements. A private repair runtime alone does not prove all shared model resources, callbacks, scratch, and streams are independently usable.

A mutex held through GPU completion would largely reproduce alternation with extra threads. Releasing it after CUDA submission would require a real concurrency audit. Concurrent private repair may eventually hide useful work, but contention on the same GPUs may erase the benefit; there is no measurement establishing an advantage yet.

## Initial supported profile

- One request, one foreground verification window, greedy verification, fixed bounded width.
- Existing serial routing across all participating GPU stages.
- No overlapping pipeline windows, adaptive placement changes, or speculative work running ahead of the controller.
- Preserve current output/context/host/device limits until explicitly revised and qualified.
- Prefer the existing single-token first-output path initially; begin batching at the next committed boundary.
- Use a draft-provider boundary. Start with the simplest already-supported provider whose state can be reconstructed from committed history (token lookup is the recommendation); add MTP through the same interface after the batch protocol passes. Main should confirm the lowest-cost provider against current code before committing to that sequence.
- Keep unsupported configurations failclosed. Unflagged behavior and the owner's 524288-token default remain unchanged.

## Source anchors and limits of this inspection

Paths below are relative to `remote-policy-ownership/`. Main is actively changing source; recheck these functions against the source pin selected for implementation.

- `src/program/generate.cpp`: `ServingSuffixRepairOperation::generate_one`, `repair_unit`, `finish`, the retained generation loop, and ordinary serial speculative serving loop.
- `src/program/resident_repair_controller.inc`: `foreground`, `Receipt`, `check_idle`, `attach_prefix`, final catchup, and metadata construction.
- `src/program/resident_repair_dispatch.inc`: `run_foreground_one` currently performs real run/commit/wait and per-stage drain checks for T=1.
- `src/core/window_state.cpp`, `window_capture.cpp`, `window_suffix.cpp`, `window_repair_suffix.cpp`, and `src/program/suffix_raw_runtime.cpp`: snapshot/restore contracts currently require disabled speculation.
- `include/strata/core/mtp.hpp`: drafter prefill/draft interfaces need a separate state-synchronization audit.

Observed integration hazards: per-token operation/cut receipt assumptions; private repair advancement gated by publication alignment and attachment limits; ordinary serving commit extent chosen before some output truncation/EOS handling; and snapshot metadata that explicitly declares speculation disabled. These are design inputs, not permission to relax invariants.

## 1. Extract a completed foreground batch operation

Introduce a narrow adapter around the existing speculative verifier, not a call into the entire ordinary serving loop. Keep proposal policy out of the repair scheduler. Proposed interface names are illustrative:

```cpp
BatchResult run_speculative_batch(SelectedState start,
                                 BatchLimits limits,
                                 DraftProvider& provider);
```

Before mutation, validate selected state/cut, request bounds, receipt capacity, rollback capacity, and actual resource admission. Bind the operation to a unique operation identity and selected-state version. No repair publication or next-batch drafting occurs while it runs.

The operation proposes, verifies, selects the permitted accepted output prefix, commits/rolls back, waits for actual completion, and checks every participating stage. Its result identifies the starting state, offered/accepted draft counts, committed output tokens including any correction/bonus token, final forwarded boundary and pending token, terminal reason, and actual completion evidence. No caller-provided boolean substitutes for completion.

The adapter must preserve verifier routing and complete rollback for KV, recurrent, PLE, and any other live state in the native inventory. Multi-token PLE history checks must replace the current T=1-specific expectations. Do not infer that waiting on one stream establishes completion of the entire chain.

## 2. Commit history and receipts as a batch

Preserve `accepted = forwarded + 1` at generation boundaries. Starting at forwarded position F with one pending token, committing k outputs y1...yk forwards the previous pending token plus y1...y(k-1), advances the forwarded boundary to F+k, and leaves yk pending. Rejected drafts enter neither accepted history nor output receipts. Preserve approximation roots and extend their coverage; speculative verification of approximate state does not create an exact certificate.

Determine EOS and remaining-output limits before selecting commit extent. Do not commit more rows than the sequence the controller records. Validate accepted-history and forwarded/context limits independently, including space for the pending token.

Use an immutable batch receipt with operation identity, starting-state identity, final cut, token sequence, and output-index range. Distinguish batch count from generated-token count. A retry must return the original result without rerunning or re-emitting it; reject reuse of an operation ID for different inputs. Do not manufacture completed intermediate cuts for each output merely to satisfy the old one-token receipt shape. Final validation concatenates batch receipts and compares them with the accepted generated suffix.

Controller idempotency is not exactly-once HTTP delivery. Keep committed output accounting distinct from what a disconnected client actually received.

## 3. Separate private repair progress from publication

Refactor the conceptual operations to:

```cpp
advance_exact_repair(fixed_token_budget);
try_publish_repaired_prefix();
```

Private repair can advance through committed history even when the live endpoint is not publication-aligned or the current partial-attachment limit has been reached. Publication remains subject to existing alignment, ancestry, state, resource, and all-stage completion checks. Retain existing attachment limits initially; continuing private work still reduces final catchup.

Start with a fixed bounded repair budget and measure backlog. If a batch commits k tokens and repair advances r, backlog changes by k-r. Do not claim bounded catchup latency or sustained convergence when measured repair throughput cannot keep pace. Final catchup remains mandatory before reusable exact state is released.

Initially publish at eligible boundaries without forcing batch alignment. Variable accepted lengths can delay publication; track that. If progress requirements or measurements demonstrate starvation, cap foreground width toward the next eligible boundary or use a bounded T=1 alignment fallback. Adaptive repair budgeting and alignment-aware batch scheduling are follow-ups, not prerequisites for the first batch protocol.

## 4. Make drafter invalidation explicit

Prepare the next drafts after the repair/publication turn, never in advance. Token-history lookup proposals use committed history. Following a replacement of live verifier state, discard any proposals or cached drafter state not proved valid for the new selection.

For MTP, inspect which cached states depend on verifier representations. Discarding a token array is not a sufficient reset if the drafter retains incompatible hidden state. Define and qualify an explicit reset/rebind/reconstruction path using supported drafter operations. Charge its resources and include its cost in performance measurements. A repair advance that does not publish does not itself change live verifier state.

## 5. Review snapshot and continuation semantics

Current snapshot code explicitly rejects speculation-enabled metadata and expects a new generation policy without a continuation payload. Do not set `speculation_disabled=true` merely to bypass it while speculation is active.

The intended capture boundary has no outstanding window, fully committed/rolled-back verifier state, completed stage work, and disposable draft continuation state. Review whether an explicit suspension/discard boundary legitimately satisfies the existing contract, or whether a new supported capability/schema is needed. Preserve rejection of unsupported speculative snapshots. Capture, restore, serialization, repair preparation, and callers must agree on the interpretation before enablement.

This is also the boundary for future concurrent repair publication; serial scheduling does not remove the contract review.

## 6. Cancellation, failure, and retirement

Check cancellation before each batch. If cancellation arrives during one, complete a bounded operation to a valid committed boundary before changing owners. Define whether that operation commits its permitted result or restores its starting state using an actually supported rollback path; do not promise immediate cancellation or assume zero-row rollback exists. Specify output accounting consistently with that choice. Start no further speculative work and follow the existing exact catchup/retirement policy before session reuse.

Pre-mutation capacity/refusal may use a qualified T=1 fallback. Post-submission failures permit continuation only with positively established rollback and completion; otherwise use established poison/fail-stop behavior. Retain all owners until every consumer is terminal. Pre-admit batch scratch, receipts, and rollback resources so routine limits do not become mid-mutation failures.

## Implementation order and acceptance

1. Reconcile current source and shared contracts; confirm no displacement of the two leading priorities. Select one supported drafting provider and freeze the interface scope.
2. Extract the batch adapter. Width-one behavior must preserve the existing qualified path. Test actual production decision/commit logic, not a duplicate model of it.
3. Integrate batch receipts/history and independent repair advancement. Keep publication limits and fixed budgets initially.
4. Resolve snapshot/quiescence semantics and provider invalidation. Obtain the affected contract reviews before enabling the combined profile.
5. Run focused CPU controls, then an immutable integrated build and a bounded real engine campaign with actual rollback/state checks.
6. Qualify authenticated HTTP, repeated edits, cancellation, fallback, cleanup, exact final catchup, answer quality, and performance for the same candidate/profile. Extend selector support only through the existing qualification/deployment gates.
7. Add MTP, if not the initial provider, through its audited reset/rebind path. Consider concurrent private repair only after serial measurements justify it.

Required cases: all drafts accepted; first and middle rejection; correction/bonus outputs; EOS at each possible output position; output/context/receipt bounds; stale selection and operation retries; cancellation around verify/commit/publication; device/rollback failure; unaligned endpoints; attachment-limit exhaustion with continuing private progress; repeated repair publications and repeated requests; no rejected draft or approximate donor contamination; and unchanged baseline behavior.

Use two distinct state oracles: (a) batch versus serial greedy execution from the same selected starting state, investigating numerical/decision differences rather than assuming multi-token kernels are identical; (b) final exact repaired state versus clean teacher-forced replay of the actual committed sequence. Neither replaces edit-sensitive answer-quality checks. Preserve full native state inventory checks and do not loosen comparators to obtain a pass.

Performance comparison: suffix reuse alone, speculation alone, and the combined path under matched model/profile/request/sampling conditions. Record external TTFT, inter-output/batch pauses, total completion including final catchup, draft acceptance, private backlog, publication/reset costs, and peak host/device usage. Separate oracle/diagnostic overhead. No speedup is assumed.

## Shared-contract review and later concurrency

Before implementation, main records a finding and propagates review requirements under the existing review protocol for batch execution/completion, receipt semantics, snapshot/continuation, resource peaks, and cancellation. Direct consumers include suffix import, live repair, saved-state maintenance, and final qualification; resolve transitive consumers from the actual dependency graph. This proposal does not invalidate completed evidence for the existing speculation-disabled profile and does not mutate its active plan status.

Future concurrent private repair would require immutable committed-history views, independent resource/stream/callback ownership, bounded scheduling/resource contention, cross-thread failure and cancellation coordination, and stale repair-result rejection. Publication must still occur at a completed foreground boundary. Preserve that interface now, without implementing an unused worker framework.

Completion means the selected combined profile has actual engine and HTTP evidence, safe failure/retirement, explicit approximation, measured quality/performance, and qualified launch boundaries. A lock, an extracted API, a build, or mocked tests alone does not complete this extension.
