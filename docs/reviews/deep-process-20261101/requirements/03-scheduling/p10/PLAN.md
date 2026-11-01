# Per-chat-session expert accounting (P10)

[Work queue](../../../WORK-QUEUE.md) · [Parent plan](../PLAN.md) · [Review protocol](../../REVIEW-PROTOCOL.md)

## Codec admission review gate — 2026-10-27

Plan status for codec-dependent admission/integration: **review-required**.
Origin: P11 FINDINGS.md, `codec-planner-peak-stream-20261027`. Owner: main
Strata integration. Do not rely on the existing planner as an executable peak
bound. Review shared encode/decode layout, stream binding, overlapping result/control
charges and revised evidence before resuming dependent enablement. Independent RAW
implementation, disabled-capability source work and bounded repair tests continue.
Existing plan revisions/evidence below retain their original scope.

## Status and ownership

- Execution: Not started; proposal requires source reconciliation before implementation.
- Plan: review-required, revision 2; integration held pending complete-cut and target reconciliation.
- Owner: Strata implementation session under the owner handoff.
- Dependencies/integration inputs: [P01](../../01-foundations/p01/PLAN.md), [P02](../../01-foundations/p02/PLAN.md), [P07](../p07/PLAN.md), [P09](../p09/PLAN.md)
- Direct dependent plans: [P11](../../04-repair/p11/PLAN.md), [P15](../../05-qualification/p15/PLAN.md)
- Source: Fig issue 30, derived from issues 25–29. Its design pin differs from the deployed engine; reconcile against baseline provenance before editing.

## Local execution plan

1. Reconcile the source proposal below with the pinned local engine and record exact files, existing behavior, missing mechanisms, and unresolved assumptions in FINDINGS.md.
2. Revise this plan with bounded changes and tests using that evidence. Record interfaces and resource budgets consumed by dependent plans. Historical research restrictions describe the source pass; current owner instructions govern execution.
3. Implement only the agreed scope in an isolated checkout, with the feature disabled by default where applicable. Keep the deployed service unchanged during preparation.
4. Run the applicable inert tests below; record commands, source pin, outcomes, and limitations in EVIDENCE.md. Coordinate any GPU/live qualification window separately.
5. Resolve or propagate plan-review triggers before dependent execution; update this plan, the parent, and queue status together. Mark complete only against its acceptance criteria.

## Findings that trigger plan review

Attribution, speculative-work classification, counter completeness, or telemetry overhead changes. Review direct dependent plans listed above and propagate transitively where their contracts change. A changed state schema, shared interface, topology, or resource envelope also requires the final qualification plan to be reviewed. Do not silently assume an unaffected status: record the decision and rationale.

## Preserved detailed proposal and acceptance criteria

**Depends on:** P01/P02/P07; P09 for full topology coverage.  
**Existing files:** prefill.cpp route production and GPU-only/host branches; verify.cpp final route and commit paths; generate.cpp session/slot mapping.  
**Proposed files:** include/strata/core/session_expert_stats.hpp; src/core/session_expert_stats.cpp; bounded device reduction helper if required.

Implement Configure/Get/Export/Reset/CloseExpertStats with session IDs, epochs, row attribution, purpose, model family, execution IDs, and cut watermarks. Distinguish selections, confirmed completed work, accepted/discarded/pending state rows, and unknown execution after failure. Shared experts are separate. No routing changes.

Cover fused GPU paths as well as the existing host callback. Capture/reduce IDs before reuse, classify via actual verifier keep masks, aggregate owned global layers once, and seal exports across devices or declare partiality. Fork defaults to fresh executed-work counters with an optional labeled parent baseline. Restore does not erase lifetime execution; event redelivery deduplicates while real re-execution counts again.

**Failures:** bounded queue backpressure or explicit incomplete/error; no silent “exact” result after drops; checked counter overflow.  
**Inert tests:** mixed sessions in one kernel/window, reset during commit, accepted/discarded draft/main rows, fused versus host path, fork/restore/retry, stage lag.  
**Acceptance:** exact aggregates or truthful partial flags, bounded memory, no inference behavior change; overhead threshold remains a proposal until measured/reviewed.  
**R10:** accounting semantics and overhead review.

## Review record

Initial plan extraction: 2026-10-27. No implementation or runtime qualification implied. No pending review finding has yet been evaluated for this package.

## Review required — 2026-10-27

Owner: Strata implementation session. Findings: [target](../../01-foundations/p00/FINDINGS.md), [complete cut](../../01-foundations/p01/FINDINGS.md). Reconcile common stage cut, accepted pending history, typed owned payloads and private publication against the deployed engine base. Dependent integration is held; independent source inspection and inert test design may proceed. No prior qualification is inherited. Resolution remains pending.

## User activity amendment — review required

Editable persistent base priority and temporary user-activity boosts are separate fields. Engine receives explicit bounded/versioned activity input; UI/manager owns activity detection and thread mapping. Expose base/effective priority, boost reasons, capabilities, accepted/applied revisions. Define per-user expiry/disconnect, concurrent users, priority decreases and dependency inheritance. Activity never bypasses foreground/live-repair/saved-maintenance eligibility.

Review bounded-unit admission and attribution for base/activity/inherited priority changes; preserve existing class limits and cancellation/publication contracts. Resolve this transitive review when scheduler interfaces and affected work-unit checks are integrated.

## Observed serving envelope — 2026-10-27

[Target finding](../../01-foundations/p00/FINDINGS.md#actual-serving-envelope): current Flash-Next
serves a 256K INT8 context with MTP and automatic layer split; the research 128K/Q4 worksheet
is only illustrative. Include 256K INT8 and required two-session operation in resource/runtime
qualification. Deterministic first probes disable all speculation; current MTP compatibility is
not implied by the new descriptor adapter. Host-only schema/ledger checks remain useful but do
not settle this integration review. Actual resolved split/free capacity still needs evidence.

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

Atomic selection contract review (P09 revision 3): publication serial plus full key/cut
CAS, old-gate retirement, prepared-destination single use and retained readers are required.
Scheduler/disk/real multi-device prerequisites remain unresolved.

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

Managed priority/dependency scoped review (2026-10-27): see work-items/03-scheduling/p07/MANAGED-LIFECYCLE-REVIEW.md. Component revision4 is ready and sanitizer-tested; runtime consumers must retain full managed identities, refresh at ordered control/timer cuts, account transactional memory, and preserve hard eligibility. Integration/qualification remains incomplete.

## Serialized-control and teardown checkpoint (2026-10-27)

Plan: review-required for remaining dependent runtime integration. The scoped consumer
contract and tests are reviewed in [checkpoint review](../p08/SERIALIZED-CONTROL-REVIEW.md).
Prior evidence retains its original source scope; no feature acceptance is implied.

## Session runtime interface review (2026-10-27)

Scoped interface decision and remaining held integration are recorded in [session runtime review](../p08/SESSION-RUNTIME-INTEGRATION.md). Full acceptance remains open.

### hipcomp-dax-target-20261027 — scoped review required

Owner adds hipCOMP alongside nvCOMP for RX 6800 XT. Dax currently has GTX 1080;
SSH qualification tests are authorized once the RX 6800 XT is installed, coordinated
with parent/active workloads. See 04-repair/p11/PLAN.md for authoritative target,
codec compatibility and test requirements. Review this item's AMD codec capability,
resource/format/lifetime assumptions and qualification consumers before codec-dependent
integration. Existing CUDA/RAW work is unaffected and continues; no hardware pass claimed.

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

### Async compression downstream contract review (2026-10-27)

P11 revision4 authorizes immutable owned-input jobs with pre-enqueue peak reservations,
verified private output and retained quarantine on uncertain drain/free. RAW fallback
requires no enqueue or confirmed drain. No vendor framing interoperability assumed.
Review-required: this item's encoded-state serving/persistence/multi-stage acceptance;
use P11 revision4's explicit integration gates before enabling. Independent RAW work
continues. Main owns integration resolution; CPU/SDK compilation is not hardware proof.

### Weight screening / exact document-prefix investigations (2026-10-27)

Separate owner-added child plans: P11/weight-stream-screen and P04/shared-document-prefix.
Review-required only for any resulting weight format/resource/streaming change or exact
prefix branch acceptance affecting this item. Source/metadata and bounded synthetic tests
may proceed; model SSD reads and all GPU runs require separate root coordination. Existing
state codec, approximate suffix and deferred immutable chunk-sharing scopes are unchanged.
Use actual engine-specific source and measurements; DeepSeek and Qwen results do not transfer.

### Actual profile receipt contract — revision 1, 2026-10-27

Plan: **review-required** for complete verified-profile publication and dependent managed enablement. Origin: P00 FINDINGS.md `actual-profile-receipt-retention-20261027`; main integration owns review. Prior loader/owner evidence retains its original source scope. New effective-profile parser capture has a dedicated bounded host reservation and retained immutable descriptor authority, surviving independent stage/gang lifetime. It records requested ranking only; sizing reads, peer reorder, capacity/stage trimming and adaptive placement still need actual effective-policy accounting. PLE startup metadata must be checked against the retained table generation at publication.

Required evidence: current parser failure/identity/close/allocation tests, driver admitted-path/refusal checks, owner/quarantine lifetime, and touched-backend builds before acceptance. Current source is uncompiled; no concurrent or hardware qualification follows. Independent source work is allowed during scheduling hold; publication remains deferred pending these gates.

### Owned adaptation completion review — revision1

Review-required for dynamic expert placement/retirement: P02 FINDINGS `owned-adaptation-completion-20261027`. Main checked owned primary/stage event completion before unpin/publication; no compile or hardware evidence yet. Validate success/NotReady/error and device scope, then close peer completion/upload publication and effective-policy generation gaps. Legacy default unchanged. Existing managed enablement gates remain; independent source work may proceed.

### Placement and actual execution decision review — revision1

Review-required for complete managed identity/cuts/replay. Origin P00 FINDINGS `placement-versus-execution-decisions-20261027`; main owns integration. Startup placement and host-residency hashes do not cover timing-dependent CPU/GPU/foresight/prefill choices. Add bounded owner transaction state and actual consumer decisions, bind exact replay/capture to appropriate revision and dependencies, then test delayed copies/partial failures/compulsory fills/prompt loans/all tiers. Adaptive features remain required. Existing scoped evidence remains valid only for its original components; full publication remains held while source work proceeds.

### Initial expert tier join review — revision1, 2026-10-27

Review-required for managed identity/publication. Origin P10 FINDINGS `initial-expert-tier-join-20261027`; main owns integration. Require actual completion of every declared initial tier at the post-helper-load cut, then invalidate before untracked dynamic setup/execution. Future tracker must seed at that cut and retain exact owners, not reuse a stale initial receipt. Test missing helpers, partial producer failure and successful join/invalidation; retain absent-profile/dynamic/execution gates. Source-only, no backend/numerical qualification.

### Explicit disabled profile review — revision1, 2026-10-27

Review-required for complete profile/managed publication. Origin P00 FINDINGS `explicit-disabled-expert-profile-20261027`; main owns review. Retain the genuine manifest-disabled claim and actual model geometry without manufacturing file header/path/ranking evidence. Price metadata before construction, revoke on claim/read-generation change, and join real initial cache capacity even without a ranking file. Test forged/public-only flags, lease conflicts, absent geometry, insufficient quota, output preservation and actual read invalidation; rebuild all affected inventory consumers before qualification. Source-only; existing dynamic/executor gates stay open.

### Producer CPU evidence update — 2026-10-27

P01 `EXPERT-PRODUCER-CPU-CHECKPOINT.md` records current-source layout/Hadamard sanitizer and File/Arena CPU producer PASS, manifest520713ea..., with source-only driver ordering checks. Evidence scope reviewed: no contract/acceptance change; existing review-required integration gates remain. This does not qualify actual driver execution, backend linkage, device retirement, dynamic placement or concurrency.

### Review-required: verified KV refill submission — 2026-10-27

Origin P02 FINDINGS `verified-kv-refill-submission-20261027`. Main-owned source repair stops publication after a rejected copy even if later synchronization succeeds; earlier accepted copies must drain. Review partial topology/failure ownership before managed runtime enablement; no rollback claim. Exact failure fixtures, full backend builds and device qualification pending. Existing review gates remain; independent work continues.

### Review-required: retained CPU execution sink — 2026-10-27

P08 FINDINGS `retained-cpu-execution-sink-20261027`: selected-work-bound sink source is partial CPU diagnostics only, uncompiled/unwired. Review genuine source/runtime geometry, prepaid work/record lifetime, raw callback removal+join and quarantine contract. Do not use these records as complete GPU/kernel/reduction or numerical-profile authority. Main owns integration; source review and genuine success fixture in progress. Dependent publication remains held, independent work continues.

### Review-required: physical KV transfer disposition — 2026-10-27

P02 FINDINGS `physical-kv-transfer-disposition-20261027` requires actual consumed-handle disposition, bounded detached ownership and zeroing/completion proof before topology publication or retry. map_range false is not retained caller ownership; mapped is not necessarily initialized. Main integration pending; source-only evidence, existing managed gates retained.

Retained CPU sink review update: selected-runtime membership now checked by actual DriverWindowRuntime-owned dispatch/source/pool interface, with source/layout geometry match. P08 retained-cpu-execution-sink-20261027 records this revision and exact pins. Prepared tests unrun; driver installation, authenticated policy revision and complete execution coverage remain open. Existing review-required status retained.

Checked VMM source review update: P02 physical-kv-transfer-disposition-20261027 records owner-fence bypass, vector-capacity charge lifetime and released-range health defects. Repair and exact failure fixtures required before caller integration; optional CMake inclusion is not compiled evidence. Keep prior topology/accounting enablement gates.

### Frozen policy CPU evidence — 2026-10-27

P10 CPU-POLICY-FIXTURES-20261027.md records six frozen source/mock sanitizer fixtures PASS, manifest4a6e3cfe..., session11059. Evidence-only addition; current-source deltas, actual backend/driver/GPU/concurrency and existing dependent enablement reviews remain unqualified. No acceptance criterion changed.

Verified refill review update: exceptions after submitted copies now reach common verified drain before failure/no publication; legacy propagation preserved. See P02 verified-kv-refill-submission-20261027. Twelve-case fixture is prepared, not run; quota/physical ownership/backend/device gates unchanged.

### Review-required: CPU receipt work bound — 2026-10-27

P08 FINDINGS cpu-receipt-linear-bookkeeping-20261027 identifies quadratic row scans and a prepaid lookup replacement in progress. Review added metadata envelope, exact row/job ordering and no-allocation failure semantics before current sink enablement. Frozen prior fixture candidate remains distinct; no new performance/runtime claim.

### Review-required: prepaid elastic KV attachment — 2026-10-27

P03 elastic-kv-prepaid-attachments-20261027 requires distinct arena/private-work and physical/private-attachment quotes, all reservations before first device effect, exclusive batch consumption and retained parent quota through donation. Source implementation in progress; no gang/backend/runtime validation. Existing dependent publication/restore/transfer gates retained.

Prepaid elastic integration review remains required: P03 FINDINGS elastic-kv-prepaid-attachments-20261027 now records host metadata/handle allocation ordering and error-drain gaps in the initial owner. Optional build wiring and PLE host preallocation are source-only. Main owns dependent integration; validate revised VMM/owner source and backend compatibility before enablement. Existing restore, growth/donation, capacity-frontier and concurrency acceptance stays open.

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

### Review-required: actual expert runtime profile join — 2026-10-28

P01 FINDINGS `actual-expert-runtime-profile-join-20261028` changes source receipt ownership and metadata lifetime. Main must review this consumer’s authority/invalidation/accounting assumptions before complete-profile-dependent activation. Source-only collector and driver wiring are untested; no previous build/fixture evidence covers them. Existing execution/qualification gates remain.

### Review-required: actual running main image — 2026-10-28

P00 FINDINGS `actual-running-main-image-20261028` introduces schema3 startup observation and distinct transient IO/retained metadata charges. Main must review this consumer’s identity/schema and lifetime dependencies; producer/driver source is untested. Complete execution identity remains gated on library/JIT closure. Independent source work may continue.


### Review-required: remote routing consumer lifetime — 2026-10-28

Origin: P02 FINDINGS `remote-routing-consumer-lifetime-20261028`. Main integration must reconcile actual primary-output consumption, helper/copy-reader ownership, partial-open cleanup and retained-resource charges before dependent activation. Add actual controller and partial-failure qualification; helper fixture evidence alone is insufficient. Current driver pre-open retirement wiring is uncompiled; no routing binding enabled. Independent source work may continue. Resolution: review-required; full feature scope retained.


### Review-required: terminal copy failure — 2026-10-28

Origin P02 FINDINGS `terminal-copy-query-sticky-failure-20261028`. Actual completion/snapshot reuse must respect the terminal latch and external checked teardown. Main reviews dependent activation; seven-method/first-guard CPU evidence does not qualify the full controller. Preserve failure state and pending ownership until actual drain. Review-required; independent source work permitted.

Review-required: P02 `remote-optimizer-checked-lifecycle-20261028`: strict optimizer partial-init/consumer-drain/free/restoration and retained quota lifetime must be reconciled before dependent activation. Main owns integration; full multi-backend/concurrency scope retained.

Review-required: P02 `chained-host-launch-completion-20261028`; checked actual asynchronous host-launch completion must precede successful cut/retirement publication. Main owns driver integration; activation held, independent source work continues.

Review-required: P02 `owned-verifier-device-selection-20261028`; nested/thread-local device selection/restoration must be checked before owned completion or retirement certification. Main integrates; independent source work continues.

Review-required: P02 `remote-unit-certification-review-20261028` adds final-device, actual topology, recursive terminal-health/publication and queued-owner gates; separate all-resident remote-sum correctness fix requires integration evidence. PR38 focused CPU pass does not qualify these consumers. Main owns downstream reconciliation; scope remains required.

Review-required: P02 `lookahead-residency-borrow-20261028` requires bounded reversible closure of the asynchronous residency/source reader before adaptation writes. Keep lookahead and full async scope; source-only gate/serial completion is not this reader join.

## Scheduling wire and reactor review — 2026-10-28

Dependent serving integration: **review-required**, finding `scheduling-wire-epoch-reactor-20261028` in P07/P08 FINDINGS. Main owns review. The bounded parser/controller source slice is reviewed and may proceed (interface revision 1); it must expose control acceptance separately from actual executor application, retain epoch-owned replies, bound replay/resource storage, process idle activity expiry, and reject uncertain or stale retries. Two-user tests must cover independent base/activity, restart-stale replies, expiry during pending work and retained cancellation ownership. Actual executor cuts and full concurrent qualification remain required. This amendment does not clear earlier review gates or claim source-only tests qualify serving. Codec/state formats are unchanged; their consumers review only scheduling ownership and admission boundaries.

### Integrated source closure review — 2026-10-28

Dependent combined qualification is **review-required**, origin P00 FINDINGS integrated-source-closure-20261028. Restore missing previously intended producer/test/CMake changes additively while retaining newer fixes, re-freeze exact source and complete required target/test mapping. Do not remove failed targets to make the build pass. Main owns review; prior frozen qualification remains valid only for its original source. Independent isolated source work may continue.

## Review amendment: scheduling-worker-admission-20261028

Plan revision: additive 2026-10-28 worker-lifetime amendment; prior decisions retained. Plan remains review-required for scheduling-dependent runtime integration. See P03 FINDINGS.md, scheduling-worker-admission-20261028. Main owns resolution. Isolated logical stack implementation may proceed; actual serving enablement requires separately bounded registry/graph/snapshot ownership, safe stop/submit ordering, independent timer progress, and source-linked lifetime/failure tests. Supplied stack charge does not certify total process RSS. No change to this item's full feature or qualification scope; independent source work continues.

## Review amendment: source-map-foresight-serialization-20261028

Additive plan revision 2026-10-28. Scheduling/placement-dependent activation remains review-required under P02 FINDINGS. Main reviewed isolated source implementation with actual startup quiescence, setup sealing and FS-before-map lock order; external borrower and DMA lifetime still require genuine joins. Preserve full feature scope and old source-pinned evidence. Source work may proceed; changed source needs failure/lifetime/actual-driver qualification before enablement. Main owns downstream disposition.

## Review amendment: canonical-reservation-owner-20261028

Additive revision 2026-10-28: resource/lifetime-dependent integration remains review-required under P03 FINDINGS. Private canonical reservation ownership identity and each external dedicated-admission consumer must be joined, source-rebuilt and counterfactually tested; use_count alone is insufficient. Preserve prior pins/qualification, full feature scope and independent policy work. Main owns disposition. Isolated reviewed fix implementation allowed; enablement held until actual consumer and resource evidence completes.
