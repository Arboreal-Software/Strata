# Persistent thread priorities and dependency inheritance (P07)

[Work queue](../../../WORK-QUEUE.md) · [Parent plan](../PLAN.md) · [Review protocol](../../REVIEW-PROTOCOL.md)

## Status and ownership

- Execution: implementing bounded managed thread/activity retirement and dependency inheritance in delegated wave; executor integration pending.
- Plan: ready, revision 4 for managed priority lifecycle and bounded dependency graph; executor/auth/persistence integration remains review-required.
- Owner: Strata implementation session under the owner handoff.
- Dependencies/integration inputs: [P01](../../01-foundations/p01/PLAN.md), [P02](../../01-foundations/p02/PLAN.md)
- Direct dependent plans: [P08](../p08/PLAN.md), [P10](../p10/PLAN.md), [P15](../../05-qualification/p15/PLAN.md)
- Source: Fig issue 30, derived from issues 25–29. Its design pin differs from the deployed engine; reconcile against baseline provenance before editing.

## Local execution plan

1. Reconcile the source proposal below with the pinned local engine and record exact files, existing behavior, missing mechanisms, and unresolved assumptions in FINDINGS.md.
2. Revise this plan with bounded changes and tests using that evidence. Record interfaces and resource budgets consumed by dependent plans. Historical research restrictions describe the source pass; current owner instructions govern execution.
3. Implement only the agreed scope in an isolated checkout, with the feature disabled by default where applicable. Keep the deployed service unchanged during preparation.
4. Run the applicable inert tests below; record commands, source pin, outcomes, and limitations in EVIDENCE.md. Coordinate any GPU/live qualification window separately.
5. Resolve or propagate plan-review triggers before dependent execution; update this plan, the parent, and queue status together. Mark complete only against its acceptance criteria.

## Findings that trigger plan review

Dependency inheritance, cycle handling, priority update, or reconstruction semantics change. Review direct dependent plans listed above and propagate transitively where their contracts change. A changed state schema, shared interface, topology, or resource envelope also requires the final qualification plan to be reviewed. Do not silently assume an unaffected status: record the decision and rationale.

## Preserved detailed proposal and acceptance criteria

**Depends on:** P01; integrates with P02 control epochs.  
**Existing files:** serve/server.py admission and generate.cpp control handling.  
**Proposed files:** include/strata/core/work_scheduler.hpp; src/core/work_scheduler.cpp.

Implement RegisterThread, SetThreadPriority with expected version, AddDependency/RemoveDependency, CancelOperation, and SchedulerSnapshot. Keep base/effective priority separate from service weight. Local dependency edges target exact operation/result/revision milestones, not the external DAG.

Add cycle-safe atomic graph publication and transitive multiwaiter inheritance. Recompute decreases after priority change, edge removal, timeout, completion, and cancellation. Blocked consumers are not runnable. A shared prerequisite runs once and survives one waiter's cancellation when others remain.

**Failures:** STALE_VERSION, UNKNOWN_GENERATION, DEPENDENCY_CYCLE, DEPENDENCY_FAILED, graph-cap overflow.  
**Inert tests:** chains/diamonds, competing additions forming a cycle, highest-waiter removal, priority decrease while active, delete/recreate, restart reconstruction.  
**Acceptance:** updates affect future units of the whole thread, including parked/future requests; accepted-versus-applied epochs are visible.  
**R7:** scheduler mechanism review without selecting the final policy.

## Review record

Initial plan extraction: 2026-10-27. No implementation or runtime qualification implied. No pending review finding has yet been evaluated for this package.

## Owner addition: repair urgency

Increase live repair urgency according to the oldest unresolved region affected by suffix reuse. Preserve that age across later edits/ancestry; track wall-clock age, earliest affected token position, and token debt separately. Review eligibility and maximum-age/debt fallback under continuous foreground load; aging alone does not prove bounded drift. Saved maintenance stays idle-only unless the owner changes that policy. Thresholds require evidence, not guessed constants.

## Review required — 2026-10-27

Owner: Strata implementation session. Findings: [target](../../01-foundations/p00/FINDINGS.md), [complete cut](../../01-foundations/p01/FINDINGS.md). Reconcile common stage cut, accepted pending history, typed owned payloads and private publication against the deployed engine base. Dependent integration is held; independent source inspection and inert test design may proceed. No prior qualification is inherited. Resolution remains pending.

## User activity amendment — review required

Editable persistent base priority and temporary user-activity boosts are separate fields. Engine receives explicit bounded/versioned activity input; UI/manager owns activity detection and thread mapping. Expose base/effective priority, boost reasons, capabilities, accepted/applied revisions. Define per-user expiry/disconnect, concurrent users, priority decreases and dependency inheritance. Activity never bypasses foreground/live-repair/saved-maintenance eligibility.

Add tests for two users on different threads and on one thread, stale refresh/disconnect, expiry without new requests, base decreases during a boost, waiter cancellation and dependency inheritance recomputation. Required policy bounds and transport mapping remain to be implemented; no browser activity collection in engine.

## Observed serving envelope — 2026-10-27

[Target finding](../../01-foundations/p00/FINDINGS.md#actual-serving-envelope): current Flash-Next
serves a 256K INT8 context with MTP and automatic layer split; the research 128K/Q4 worksheet
is only illustrative. Include 256K INT8 and required two-session operation in resource/runtime
qualification. Deterministic first probes disable all speculation; current MTP compatibility is
not implied by the new descriptor adapter. Host-only schema/ledger checks remain useful but do
not settle this integration review. Actual resolved split/free capacity still needs evidence.

## Revision 3: base/activity control checkpoint

Use an engine-generation/thread-generation identity and separate base revision, activity-lease
revision and accepted/applied scheduling revisions. The external manager supplies an opaque
activity lease ID; no browser/user collection occurs inside the engine. Multiple active leases
use the maximum bounded boost, not a sum that rewards extra open tabs. Bounds/TTL are explicit
constructor policy inputs, not measured production defaults. Expiry and disconnect remove only
the temporary contribution; a base decrease persists. Expiry increments the lease revision so
an old refresh cannot revive it. Keep bounded tombstones until thread closure/recreation.

Validate with a supplied monotonic clock, cap actors/threads/IDs, and fence old engine/thread
generations. Executor acknowledgement is a separate call; accepting a change does not pretend
an active kernel applied it. This host checkpoint may proceed independently of P02. P08/P09–P15
must still integrate ordered control, dependency inheritance and work-class checks before any
capability is advertised. P07 is incomplete until the graph and whole-thread runtime behavior pass.

## Runtime drain review — 2026-10-27

[Verifier drain finding](../../01-foundations/p02/FINDINGS.md): a drained stream is not a
common-cut certificate. Runtime integration remains review-required until host producers
are stopped, all stage cuts match, and failed-device leases stay retained. Existing host-only
plan scopes are unaffected; no runtime acceptance is inherited from their tests.

Execution-lifetime gate review (P02 revision 4): preserve reservations through cancellation;
retain completion handles for quarantined failures; hold capture reads until transfer drain;
fence key, append epoch and operation generation. Runtime integration remains held until
these obligations are connected to the scheduler/reactor. See P02/FINDINGS.md.

Native capture contract review (P05 revision 3): bundle copies now retain peak host
reservation; source read holds span every stage copy. Runtime integration remains held
for transfer-failure retention, measured memory bounds and private restore/activation.

Failed-copy retention review: P05/FINDINGS.md adds a retained failed-transaction handle
and unhealthy source gate. Runtime consumers must preserve the handle through actual
device teardown, account quarantined capacity and rebuild execution before reuse.

Private restore preparation review (P05 revision 4): prepared handles are not selected
execution. Preserve destination claims and failed-transaction ownership; private allocation
non-aliasing and atomic all-stage selection must be established before runtime integration.

### Downstream review trigger: ownership/return obligation and native ABI

Reconcile P02/P03 ownership review checkpoint before runtime integration. Completion keys
are fenced even on cancellation; one accepted-history transaction advances append epoch by
one. Maintenance requires an exclusive, nonzero, window-specific return obligation, priced
by the actual allocator. Runtime must redeem the held lease without charging twice. Native
QsaState layout changed: rebuild all dependent objects; prior mixed-baseline transfer evidence
is superseded for this candidate. Private allocator/elastic ownership, canonical process-wide
registry and operation-bound completion proofs remain unresolved integration gates. This
propagates through exact parking, scheduling, repair and final qualification; no item is
marked complete by the bounded source fixes. See P02/REVIEW.md and P03/REVIEW.md for findings.

## Priority source review disposition (2026-10-27)

ChatSwarm receipt: `.local/strata-implementation/reviews/priority-source-result.json`, delivery
slack-result-delivery:slack-C0C2E4KL03X-c3083a0d70231f9348f5. GPT-6 / selected Extra High; all
five visible parts read. Connector-rendered hash mismatch prevents byte-certified review.

Confirmed same-cut acknowledgement protocol gap: a new accepted revision now requires a
strictly later executor cut; same-revision retry remains idempotent. Regression test passes.
Confirmed repeated global expiry scans: advance now skips scanning until the earliest known
expiry deadline. Refresh/disconnect may conservatively leave an earlier deadline, causing
one extra scan; no expiry can be delayed. Exact UINT64_MAX expiry regression passes.
Expiry at revision exhaustion now clears all boosts when closing; result also suppresses
boosts on closed records. No synthetic exhaustion test yet; do not claim that edge qualified.

Open: closed-thread distinct IDs and expired/disconnected activity tombstones consume bounded
lifetime slots. This preserves ABA protection but is inadequate for indefinite runtime churn.
Implement generation-safe retirement/minted actor incarnations before production; never erase
and reuse version zero naively. Parent/UI integration must bind leases to authenticated actors
and tabs, authorize edits and reads, and restrict acknowledgement to the ordered executor.
Scheduler must receive expiry revision changes without requiring user refresh and distinguish
accepted local_effective from applied scheduling state. These remain P07/P08/P15 acceptance
requirements; source tests do not qualify the two-user production path.

Managed priority/dependency scoped review (2026-10-27): see work-items/03-scheduling/p07/MANAGED-LIFECYCLE-REVIEW.md. Component revision4 is ready and sanitizer-tested; runtime consumers must retain full managed identities, refresh at ordered control/timer cuts, account transactional memory, and preserve hard eligibility. Integration/qualification remains incomplete.

## Serialized-control and teardown checkpoint (2026-10-27)

Plan: review-required for remaining dependent runtime integration. The scoped consumer
contract and tests are reviewed in [checkpoint review](../p08/SERIALIZED-CONTROL-REVIEW.md).
Prior evidence retains its original source scope; no feature acceptance is implied.

## Session runtime interface review (2026-10-27)

Scoped interface decision and remaining held integration are recorded in [session runtime review](../p08/SESSION-RUNTIME-INTEGRATION.md). Full acceptance remains open.

Epoch remediation checkpoint: scoped async ownership/control/restart contract reviewed
under 03-scheduling/p08/SERIALIZED-CONTROL-REVIEW.md; final123-test evidence recorded.
Bounded implementation ready; previous ChatSwarm NOT CLEARED until rereview disposition.
No scheduler, native orphan or hardware/deployment acceptance inferred.

Cross-window-routing-20261027: scoped interface reviewed/replanned under
03-scheduling/p08/WINDOW-MANAGER-CONTRACT.md; bounded implementation ready.
Window replace identity/cut invariants unchanged; driver generation authority and actual
lane eligibility/resource proof required. Full executor/hardware/eviction/import gates open.

## Canonical ledger publication integration contract

Exact initial token ledger now compiles in the production state library. Ordinary and
instrumented sanitizer48746 plus aggregate CUDA20422 (PLE/nine selected CTests) pass.
Accepted history includes pending outputs; decode requires exactly one pending input,
so prompt suffix is forwarded before sampling. Candidate binds complete admitted work
registration/selection identity. Cancellation finish returning true is not publication.
If post-finish observation allocates/fails, retain the candidate and published device
state, withhold output, and recover by exact observation without repeating finish.
Successful host-ledger mutation/receipt publication after verified observation is no-throw.
One reactor must serialize control/admission with this handshake. Restored import,
actual serving calls, canonical O(N) hash cost and full resource/hardware evidence remain open.

### Ordered admission implementation review

New admission policy uses existing priorities, dependency graph and exact manager
admission. Timer wake refreshes activity expiry independently of input; inherited
priority never bypasses class/resource/idle-return gates. Deterministic strict ordering
is not fairness or a repair-age convergence bound. Priced authorization remains trusted
caller input; actual driver authentication/resource setup is still required.

A priority acknowledgement can allocate after mutation. Admitted work is retained
before acknowledgement and cannot execute while ack is pending. Recovery never repeats
manager admission. Record retirement queries the exact retained lease, refuses newer
operations and unhealthy uncertainty; stale/missing manager snapshots are not drain proof.
Two-user/two-lane CPU fault tests31346/34796 passed, including150 allocation positions
(30 preadmission refusals/two retained acknowledgement cases) and UINT64_MAX expiry.
Actual serving integration, hardware scheduling and concurrency remain unqualified.
Main CMake wiring is present; next combined source build/review remains pending.

### Admission reactor lifetime review — revision1, 2026-10-27

Review-required for dependent admission enablement. Origin P08 FINDINGS `admission-reactor-identity-and-charge-lifetime-20261027`; main owns review. Require one noncopyable/nonmovable reactor identity and retain record metadata reservations through covered member destruction, including unresolved admission recovery. Prepare type-trait and retirement/deallocation checks; execute in a newly granted CPU window before claiming the repair validated. Independent source work continues. This correction does not qualify serving integration or change prior evidence.

### Idle activity timer review — revision1, 2026-10-27

Review-required for live activity expiry. Origin P07 FINDINGS `idle-activity-registry-timer-20261027`. Main owns integration. Use the whole priority registry deadline, including actors with no graph operation, for admission/reactor wakeups. Advance monotonic time then re-read after a conservative early wake; preserve absent deadline versus UINT64_MAX. Add no-request/no-operation expiry to two-user qualification and retain base priority. Registry/admission source wiring and tests are prepared; actual serving timed wait and CPU validation remain open. Independent source work continues.

### Verified input envelope review — revision1, 2026-10-27

Review-required for verified serving/control enablement. Origin P08 FINDINGS `verified-serving-input-envelope-20261027`; main owns review. Bound queued and incomplete input, retain admission through detached reader/moved consumer lifetime, and treat overflow/read failure/truncated EOF as terminal rather than empty queue or accepted partial request. Future units stop while submitted work and prompt readers drain/join. Validate first/next-unit races, STOP reset, direct control scanning, persistent allocation failure, exact framing and legacy compatibility. Source repair and fixtures are preparing; compiler/hardware gates remain open. Independent source work continues.

### Priority response preparation review — revision1, 2026-10-27

Review-required for priority transport/admission enablement. Origin P07 FINDINGS `priority-response-before-commit-20261027`; main owns review. Complete allocating success receipts before committing the requested mutation, preserve accepted/applied and full incarnation identities, and retain real admitted work on acknowledgement preparation failure. Account overlapping prepared-output/current-record/insertion metadata. Timer expiry is an independent allowed effect; do not claim whole-cut rollback or durable/network idempotency. Add persistent allocation failure sweeps before public transport qualification. Source work continues; tests/builds unrun.

## Control lifetime follow-up — 2026-10-28

Plan for serving integration: **review-required**. The bounded priority algebra remains
unchanged. Source review of batched control found ownership release must cover initial
capture, final cleanup exceptions and generator close/throw/cancellation, while a successor
or another holder of the same lock must not be released. PR3 now carries an explicit guard;
PR4 uses captured slot dimensions. Both revised candidates are untested. Failed protocol
drains can leave slot bookkeeping incomplete; releasing the lock is not a recovered
request or an accepted/applied scheduling acknowledgement. Main owns integration review.
Require actual-method regressions, then two-user priority/dependency/expiry qualification
through the real executor. Consumers P08/P09/P15 inherit this gate; full retirement review
in P02 remains an independent prerequisite. No UI/activity policy change implied.

## Two-phase retirement dependency — 2026-10-28

Dependent lifetime/publication integration: **review-required**. Origin is P02
`native-elastic-two-phase-retirement-20261028` and its `elastic-retirement/PLAN.md`.
Outer device drain does not eliminate nested pool/VMM drains after earlier frees.
This plan's retained capture/return/delete/cancel/dependency/codec/repair borrowers
must follow terminal sealing, complete producer closure, all-owner preparation and
checked no-drain commit; failed preparation retains the gang and charges, while
partial destructive failure preserves exact residual ownership rather than claiming
rollback. Temporary elastic refusal does not remove required growth/donation scope.
Main integration owns review disposition and must reconcile the actual consumers
before enabling dependent operations. Independent policy/format/codec algorithms
and source preparation may continue; no earlier test evidence covers the new path.
Require actual lifetime/failure tests and P15 hardware/concurrency qualification.

## Pending acknowledgement timer maintenance, 2026-10-28

Review required: actual WindowAdmission currently refuses refresh while acknowledgement is unresolved before advancing activity timers. Proposed source repair maintains registry monotonic expiry and next timer hint independently of new-admission permission. Recovery preserves the original accepted revisions and cut; unrelated expiry can recover at current maintained time, while changed decision-source revisions block and retain the exact lease until cancellation and actual drain. This does not authenticate user/thread identity or wire the serving reactor. Candidate is isolated in admission-expiry-implementation; first compile failed a frozen-snapshot header omission with zero test execution, corrected v2 subsequently passed the exact old-source assertion/exit1 and fixed public ASan/UBSan suite in20.69s, with independent resource release. Source fix is draftPR11; ChatSwarm exact-commit review pending. Two-user qualification must cover an idle actor with no operation expiring while another user has unresolved acknowledgement, independent base edit preservation, no retry/re-admission, stale decision refusal and exact lease retention. Do not clear this review gate until source review and actual public fixture evidence are reconciled.

## Scheduling wire and reactor review — 2026-10-28

Dependent serving integration: **review-required**, finding `scheduling-wire-epoch-reactor-20261028` in P07/P08 FINDINGS. Main owns review. The bounded parser/controller source slice is reviewed and may proceed (interface revision 1); it must expose control acceptance separately from actual executor application, retain epoch-owned replies, bound replay/resource storage, process idle activity expiry, and reject uncertain or stale retries. Two-user tests must cover independent base/activity, restart-stale replies, expiry during pending work and retained cancellation ownership. Actual executor cuts and full concurrent qualification remain required. This amendment does not clear earlier review gates or claim source-only tests qualify serving. Codec/state formats are unchanged; their consumers review only scheduling ownership and admission boundaries.

## Review amendment: scheduling-worker-admission-20261028

Plan revision: additive 2026-10-28 worker-lifetime amendment; prior decisions retained. Plan remains review-required for scheduling-dependent runtime integration. See P03 FINDINGS.md, scheduling-worker-admission-20261028. Main owns resolution. Isolated logical stack implementation may proceed; actual serving enablement requires separately bounded registry/graph/snapshot ownership, safe stop/submit ordering, independent timer progress, and source-linked lifetime/failure tests. Supplied stack charge does not certify total process RSS. No change to this item's full feature or qualification scope; independent source work continues.

## Review amendment: canonical-reservation-owner-20261028

Additive revision 2026-10-28: resource/lifetime-dependent integration remains review-required under P03 FINDINGS. Private canonical reservation ownership identity and each external dedicated-admission consumer must be joined, source-rebuilt and counterfactually tested; use_count alone is insufficient. Preserve prior pins/qualification, full feature scope and independent policy work. Main owns disposition. Isolated reviewed fix implementation allowed; enablement held until actual consumer and resource evidence completes.
