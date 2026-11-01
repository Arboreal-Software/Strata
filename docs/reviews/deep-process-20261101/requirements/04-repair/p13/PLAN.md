> Current cross-plan qualification: two-cycle GPU test remains FAILED. The 2GiB cap correctly refused a measured excess; 4GiB allowed cycle1 complete native136/137/138 + logical retirement/budget PASS, but cycle2 native137 had48 KV/scale mismatches from stale future-page destination rows. Full-byte oracle remains strict. Actual supported RAW restore canonicalization, CPU controls and a fresh repeated-cycle GPU run are required; no human/concurrency/full-feature acceptance. Evidence: `.local/strata-implementation/suffix-first/incremental-repair/RESIDENT-REPEATED-RESTORE-FAILURE-20261029.md`. Attribution/prepared-cancel/deferred-fault revisions are applied but UNBUILT/UNQUALIFIED. BothGPU/CPU resources released; liveStrata stays down under owner authorization.

# Live incremental causal repair and final-tail publication (P13)

## Main ordering update — 2026-10-29

Genuine suffix reuse remains first and in-memory incremental live repair second.
Main has queued [speculative batches interleaved with repair](speculative-batch-interleaving/PLAN.md)
after their existing non-speculative acceptance path. The follow-on remains draft,
not-started and subject to shared-contract review; it is not a prerequisite for
current acceptance or conditional deployment. Prefer one controller alternating a
fully committed batch with bounded private repair, with publication only between
completed batches. MTP and simultaneous device execution remain later work.

This ordering changes no current snapshot, receipt, completion or resource
contract. Existing serial qualification can continue without waiting for the new
plan. Before its implementation, propagate reviews for the proposed batch contracts
under the [review protocol](../../REVIEW-PROTOCOL.md); see the
`speculative-batch-follow-on-ordering-20261029` finding. The earlier checkpoint
banner above retains its historical failure evidence; it is not a current service
status or source pin. Main's live campaign/release checkpoint controls those facts.

CPU evidence update: ancestry, native reattachment data, named verifier allocation layout and extracted strict capture-tail fixtures passed in the frozen corrected batch; both negative controls were detected. See `.local/strata-implementation/suffix-first/incremental-repair/CPU-QUALIFICATION-20261028.md`. Actual private executor/graph admission/current-cut witness/ledger/publication and periodic foreground repair remain required; no full-engine or native repair qualification follows from these helper tests.

First native suffix-reuse mechanism experiment completed/released on2026-10-28; report `.local/strata-implementation/suffix-first/GPU-QUALIFICATION-20261028.md`. It does not execute incremental repair or qualify answer quality. Continue the owner-prioritized resident repair/checkpoint/publication integration; ancestry/reattachment/quote/capture-fix sources prepared since the frozen candidate have no inherited GPU qualification. No new resource grant is implied.

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

- Execution: validating actual dedicated resident controller integration; generic manager/server/concurrent repair remains implementing.
- Plan: revision 4; dedicated RAW serial qualification READY under REVIEW-resident-serial-20261029.md. Full generic/concurrent integration and codec-dependent enablement remain review-required. Prior revision3 scope/evidence is retained below.
- Owner: Strata implementation session under the owner handoff.
- Dependencies/integration inputs: [P12](../p12/PLAN.md)
- Direct dependent plans: [P14](../p14/PLAN.md), [P15](../../05-qualification/p15/PLAN.md)
- Source: Fig issue 30, derived from issues 25–29. Its design pin differs from the deployed engine; reconcile against baseline provenance before editing.

## Current milestone: in-memory incremental repair immediately after suffix reuse

Owner priority received 2026-10-28: repair the invalidated region by re-prefilling from a valid exact prefix checkpoint, then reuse the remaining suffix. The first implementation target is bounded contiguous exact-frontier advancement interleaved with foreground execution. This is the first step toward concurrent live repair, not a claim of simultaneous device execution. Disk parking, optional codecs and saved-window maintenance are not prerequisites for this in-memory milestone. Full P13 acceptance below remains required.

Keep a separate complete exact checkpoint at frontier R. Teacher-force the actual accepted history through a bounded all-layer unit, joining all producers before certifying R'. Published tokens and output effects are not regenerated or rewritten. A hybrid foreground endpoint with exact_frontier=R is not itself that complete checkpoint: its GDN/PLE endpoint still reflects the old-conditioned suffix. Preserve the private exact checkpoint across foreground switches.

Reattaching the remaining approximate suffix needs a dedicated reviewed operation. Do not merely remove the existing exact-donor refusal in window_suffix.cpp: that API also requires a new edit revision, append identity and fixed aligned boundaries. Repair within the same content revision must reconcile append growth, selection generation, history prefix and complete cut independently. Preserve unresolved causal roots and their original ages even when their original spatial intervals have been repaired: the retained recurrent endpoint can still depend on them. Advancing R cannot make that endpoint exact. Keep causal roots, spatial coverage, age, earliest affected position and token debt separate.

The first native milestone requires at least two actual partial prefix attachments before full catch-up, with foreground work between them using the retained suffix. Private checkpoint progress followed only by a final exact replacement does not satisfy it. Each attachment preserves the current accepted tokens, pending suffix, content revision, append epoch and output watermark, and publishes through the real manager/ledger transaction. A data-only ancestry helper does not authenticate execution or authorize publication.

Before publishing any repair result, recheck revision/append/selection/provenance/history and all-stage cut. A stale publication may retain only a compatible private exact checkpoint; it cannot overwrite newer foreground history. An edit before R invalidates incompatible exact work and selects an earlier complete checkpoint or replays from zero. On catch-up, freeze a bounded final tail, settle commits, replay it exactly and publish atomically while preserving pending accepted tokens and output watermark. Failure, cancellation or missed catch-up limits use declared exact fallback/suspension and retained failure ownership.

Source review must price foreground state, private exact state, capture/splice/restore overlap and switching work. The current standalone experiment resets one runtime and uses complete RAW copies; it does not yet provide a live repair executor. Measure bytes copied and restore/join latency per unit, foreground interference and catch-up under actual idle fraction. No generic prefill throughput estimate establishes convergence.

Required cases: interruption between units and during attempted device work; edits before/at/after frontier; repeated edits and inherited age; append faster/slower than repair; selection ABA; stale final publication; failure at capture/restore/forward; pending tail; exact-frontier metadata without a matching exact checkpoint; insufficient memory; bounded scheduling with base/activity/dependency changes and two users. First bounded experiments remain opt-in, with speculation disabled and one supported native format; broader concurrency/backend/serving acceptance is not waived.

Main owns integration. Sol source audits cover runtime execution, ancestry/reattachment and control/race cases under suffix-first/incremental-repair/. No current frozen suffix candidate is edited, and no new resource grant is implied. See FINDINGS.md, in-memory-incremental-repair-next-20261028. Plan readiness follows the audits and explicit downstream dispositions, not this priority amendment alone.

Scoped source implementation disposition: main accepted the reviewed immutable bounded ancestry DATA helper, source fixtures and optional CMake registration. All are uncompiled/unrun. A separate native reattachment DATA producer may now be implemented using real budgeted immutable copies and the existing native layout validator; it must pair the retained full root table with spatial metadata and does not authenticate completion, restore devices or publish. The actual controller must bind source capture, complete exact checkpoint, current selected history and checked restore before using its output. Resident runtime construction additionally requires verifier/first-graph allocation accounting beyond NativeSessionAllocation; retirement-only configuration is not admission. These isolated approvals do not close dependent runtime or qualification gates.

## Local execution plan

1. Reconcile the source proposal below with the pinned local engine and record exact files, existing behavior, missing mechanisms, and unresolved assumptions in FINDINGS.md.
2. Revise this plan with bounded changes and tests using that evidence. Record interfaces and resource budgets consumed by dependent plans. Historical research restrictions describe the source pass; current owner instructions govern execution.
3. Implement only the agreed scope in an isolated checkout, with the feature disabled by default where applicable. Keep the deployed service unchanged during preparation.
4. Run the applicable inert tests below; record commands, source pin, outcomes, and limitations in EVIDENCE.md. Coordinate any GPU/live qualification window separately.
5. Resolve or propagate plan-review triggers before dependent execution; update this plan, the parent, and queue status together. Mark complete only against its acceptance criteria.

## Findings that trigger plan review

Repair convergence, oldest unresolved ancestry, age/debt bounds, or atomic final-tail publication changes. Review direct dependent plans listed above and propagate transitively where their contracts change. A changed state schema, shared interface, topology, or resource envelope also requires the final qualification plan to be reviewed. Do not silently assume an unaffected status: record the decision and rationale.

## Preserved detailed proposal and acceptance criteria

**Depends on:** P02/P04/P05/P08/P09/P12; P10 accounting.  
**Existing files:** prefill pipeline, checkpoint/park transactions, window registry and scheduler.

Implement RepairStep on a private exact state at F. Teacher-force actual accepted history through real all-layer computation; share weights with foreground. Resume only from a certified common cut. Check demand/identity before installation/staging and each unit.

Track moving a/C; when near the tail, freeze C*, settle commits, replay the bounded remainder, and CAS revision/append/selection/provenance/history/cut. Preserve H[C*..L*) as pending. No partial early-KV publication may be certified as a complete exact state while retaining an old recurrent endpoint. Explicitly approximate partial attachments are required by the newer owner milestone above and preserve all unresolved causal roots; emitted output/effects stay fixed.

**Failures:** edit invalidates earliest affected frontier; stale final CAS retains only certified historical work; missed catch-up bounds trigger declared exact fallback/suspend approximation.  
**Inert tests:** tail grows faster/slower than repair, edit before/after F, repeated selection ABA, stage failure, cancellation, pending token.  
**Acceptance:** F advances only after complete units; catch-up claims include idle fraction and switching cost; no generic prefill-speed substitution.  
**R13:** causal/publication and foreground interference review.

## Review record

Initial plan extraction: 2026-10-27. No implementation or runtime qualification implied. No pending review finding has yet been evaluated for this package.

## Owner addition: repair urgency

Increase live repair urgency according to the oldest unresolved region affected by suffix reuse. Preserve that age across later edits/ancestry; track wall-clock age, earliest affected token position, and token debt separately. Review eligibility and maximum-age/debt fallback under continuous foreground load; aging alone does not prove bounded drift. Saved maintenance stays idle-only unless the owner changes that policy. Thresholds require evidence, not guessed constants.

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

## Private allocation resource-plan checkpoint

Native qsa_private_host_plan/session_private_plan now price the actual session device carve,
all globally indexed QSA metadata, each owned state's pinned staging, streamed authoritative
KV host backing, and retained RoPE/page-table upload buffers. Each pin allocation rounds
conservatively to 64 KiB; fresh upload vector storage reserves twice requested bytes plus
a per-session metadata margin. The supported builder and host page size must be verified
during qualification. Global KV configuration must remain serialized across sizing/allocation.
Trusted model geometry validation remains a caller prerequisite, matching session_bytes.

This is a component plan, not the entire window reservation: private PLE gather/scratch,
graph bindings, expert handoff and executor scratch must be added by the allocator. A dual
stage plan pays two primary RoPE uploads and two metadata margins; do not price it as one
whole-model host plan. Elastic allocation ownership remains unresolved and is refused by
this private path until implemented. Full scope still requires required configurations.

Frozen candidate frozen-runtime-private-plan (manifest in .local/strata-implementation) is
building in session85371; full-private-plan-build.log. Earlier ownership candidate full
compile/link and both conversation tests passed; artifacts preserved under
ownership-build-artifacts and hashes in full-ownership-build-evidence.json. Neither is
hardware qualification. Resource-plan tests are pending in the new full candidate.

## Private binding isolation correction

Inspection found private initialization registered its completed RoPE table in the device-wide
lookup, changing selected execution before atomic activation. Removed registration from the
private upload path; successful preparation owns the private table but never publishes global
lookup state. Sanitizer private-init tests pass, including zero registrations on success.
Legacy successful initialization remains unchanged. Scheduler activation/private repair execution
must bind the correct lookup only at a drained device execution switch and restore/select the
proper binding afterward. A fresh SessionState alone is not proof of private global bindings.

Related audit: best-effort OnDevice can discard an initial CUDA error and ignores restoration
failure. Capture/restore now use a checked sticky scope and require successful restoration
before publishing success, retaining failed transactions. Fault tests cover first query,
second-stage switch and caller-device restoration errors; validation is running. Extend the
same check to verifier drain/commit paths before trusting their drain result.

## Fresh native allocation owner checkpoint

Implemented allocate_native_sessions: validate whole-model stage topology, price each native
session plus optional PLE embedding/scratch buffers, atomically reserve all devices/host/pin
capacity, then allocate fresh arenas and run checked private initialization. One shared owner
retains the entire gang and its reservation through stage consumers. Last-owner cleanup selects
each owning device, releases QSA host allocations and RoPE ownership, frees arenas/PLE buffers,
and deletes state arrays. Any failed initialization or cleanup quarantines remaining storage
and its budget until worker restart; further private allocation is refused. No live restart
is performed by the API. Quarantine count exposes the required unhealthy-worker signal.

Sanitizer injected-CUDA tests passed for fresh non-aliasing state, PLE pointer rebinding with
reader deliberately unbound, retained stage owner lifetime, successful release on correct
devices, final-device admission shortage before any allocation, partial second-stage failure,
and failed-free accounting retention. Native object compilation and dependent tests are
running in allocator-owner-build.log (session55195). No GPU qualification yet.

PLE table objects contain mutable pending issue/collect IO, so copying the active PleRun/table
would violate isolation even if model weight bytes are immutable. Private allocation currently
creates PLE buffers only. Runtime must supply an independently owned or explicitly serialized
PLE reader, immutable weight bindings, graph/expert/handoff resources and their reservations
before execution. Those are still required integration work, as are elastic pool ownership,
canonical execution registry, global RoPE binding at drained switches and operation-bound
completion/recovery. The factory is not a ready-to-run executor or full P05/P09 completion.

## Per-window PLE IO context checkpoint

Implemented PleReadContext with a retained shared PleTable owner, optional reservation/lifetime
guard, independent rows/raw destination/ticket, and table generation fencing. Direct and
mapped reads can interleave with each other and legacy gather without overwriting a window's
pending rows. Collect/discard joins IO; close/reopen invalidates old contexts without output
publication. Exceptional join retains destinations/table/guard until process restart. Table
open/close are serialized with contexts. The legacy issue/collect API remains available.

Real synthetic GGUF/file tests passed under ASan/UBSan for direct/mmap and threaded/synchronous
IO, reverse collection, duplicate issue/collect rejection, concurrent legacy gather, close
with pending read, reopen fencing, retained table lifetime and pending-read destruction.
Existing eight-format tests and both 90/110-byte reader selftests also passed. These exercise
real CPU/file IO, not injected transfers. No GPU or model execution is claimed. Final is_open
locking addition is included in the full build now running, not the earlier sanitizer run.

Native PleRun has an optional shared read_context; token issue/finish route through it when
present and reject mismatched table binding. Default path remains legacy. Allocator cleanup
releases the context before native buffers. This changes SessionState layout: all engine
objects must rebuild together. Full frozen candidate frozen-runtime-ple-context is building
in session72523, full-ple-context-build.log, including same-source PLE and conversation tests.
Previous private-plan binaries preserved under private-plan-build-artifacts.

Next integration gates: create one context per private window with an actual budget guard;
charge bounded pending ticket metadata/IO slots while treating shared cache/backing once;
bind immutable PLE weights and table ownership; wire graph/expert resources and scheduler IO
drain. PleReader still uses wrapping 32-bit ticket IDs; prevent collision with live tickets
before indefinite runtime qualification. New context lifetime does not make scheduler or
full P05/P09/P15 qualification complete.

Managed priority/dependency scoped review (2026-10-27): see work-items/03-scheduling/p07/MANAGED-LIFECYCLE-REVIEW.md. Component revision4 is ready and sanitizer-tested; runtime consumers must retain full managed identities, refresh at ordered control/timer cuts, account transactional memory, and preserve hard eligibility. Integration/qualification remains incomplete.

## Serialized-control and teardown checkpoint (2026-10-27)

Plan: review-required for remaining dependent runtime integration. The scoped consumer
contract and tests are reviewed in [checkpoint review](../../03-scheduling/p08/SERIALIZED-CONTROL-REVIEW.md).
Prior evidence retains its original source scope; no feature acceptance is implied.

## Session runtime interface review (2026-10-27)

Scoped interface decision and remaining held integration are recorded in [session runtime review](../../03-scheduling/p08/SESSION-RUNTIME-INTEGRATION.md). Full acceptance remains open.

## Disk-v2 implementation review (2026-10-27)

Scoped contract ready; full integration remains open. See [disk-v2 contract](../../02-exact-parking/p06/DISK-V2-IMPLEMENTATION.md).

Scoped plan amendment owned-mapped-runtime-20261027: reviewed/replanned per
`03-scheduling/p08/MAPPED-RUNTIME-REVIEW.md` under work-items. Ready for bounded
implementation; full runtime/hardware acceptance remains open. Earlier decisions and
evidence retained; no serialization or scheduling scope change.

### hipcomp-dax-target-20261027 — scoped review required

Owner adds hipCOMP alongside nvCOMP for RX 6800 XT. Dax currently has GTX 1080;
SSH qualification tests are authorized once the RX 6800 XT is installed, coordinated
with parent/active workloads. See 04-repair/p11/PLAN.md for authoritative target,
codec compatibility and test requirements. Review this item's AMD codec capability,
resource/format/lifetime assumptions and qualification consumers before codec-dependent
integration. Existing CUDA/RAW work is unaffected and continues; no hardware pass claimed.

Disk-owner-retirement-20261027 scoped contract reviewed/replanned per P06/FINDINGS.md:
arbitrary output/source owners retire outside store lock/IO lease; unknown durability
retains charges and requires reconciliation. Tested component ready for integration;
independent review and full feature/runtime acceptance remain open.

Cross-window-routing-20261027: scoped interface reviewed/replanned under
03-scheduling/p08/WINDOW-MANAGER-CONTRACT.md; bounded implementation ready.
Window replace identity/cut invariants unchanged; driver generation authority and actual
lane eligibility/resource proof required. Full executor/hardware/eviction/import gates open.

## Disk review correction gate (2026-10-27)

Exact-source independent review returned revision-required (reviews/disk-c55544-result.json).
Do not infer durable retirement from post-restart observed absence without a successful
directory durability barrier. Refused retries must preserve verified historical operation
phase; read IO failures preserve errno and uncertainty instead of asserting malformed data.
Disk implementation/test fixes are delegated; no new acceptance claimed. Downstream
publication/recovery/eviction acceptance is held pending reproduction/fix evidence.
Main policy requires retained+transient host bounds and IO admission before filesystem
effects; implementation/accounting proposal remains review-required. Unrelated driver
construction and CPU integration work continues.

Disk correction evidence checkpoint: F1/F2/F3 reds46912 reproduced; final F1–F4
sanitizer1899 passed and exact revised component320532a3 compiled/linked in CUDA94808
with PLE/eight selected CPU tests passing. Local host reservations now enforce retained
plus transient charges and operations admit IO/workspace before filesystem effects.
This is a conservative reservation envelope, not an exact allocator/RSS cap. HIP and
independent rereview remain pending; no physical power-cut/selected-runtime acceptance.

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

## Verified artifact and raw-token identity amendment

Profile schema2 distinguishes rendered text from raw model-vocabulary token IDs. Raw mode
rejects tokenizer/template declarations instead of inventing rendering semantics; it binds
full model content, vocabulary and the versioned token-ID ABI. Server rendering/decoding
and authenticated handshake remain separate. Schema1 evidence is preserved.

VerifiedArtifact now creates a private unlinked Linux snapshot, verifies its entire expected
digest, drops writable descriptors and retains only read-only ownership and logical disk
charge. Actual tiny-file full-copy and explicit two-shard GGUF loader tests73805 passed;
FICLONE success was unavailable on the test filesystem and is unqualified. This is source-
independent API-level immutability inside a trusted process, not kernel seals or protection
against same-UID/root writable reopen. No actual model snapshots/reads or loader integration
have occurred. Non-Linux capability, complete tensor/setting inventory, resource admission
and source-pinned numerical evidence remain required. Schema2 sanitizer16530 passed;
combined source/backend integration pending.

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

### Async compression downstream contract review (2026-10-27)

P11 revision4 authorizes immutable owned-input jobs with pre-enqueue peak reservations,
verified private output and retained quarantine on uncertain drain/free. RAW fallback
requires no enqueue or confirmed drain. No vendor framing interoperability assumed.
Review-required: this item's encoded-state serving/persistence/multi-stage acceptance;
use P11 revision4's explicit integration gates before enabling. Independent RAW work
continues. Main owns integration resolution; CPU/SDK compilation is not hardware proof.

### Placement and actual execution decision review — revision1

Review-required for complete managed identity/cuts/replay. Origin P00 FINDINGS `placement-versus-execution-decisions-20261027`; main owns integration. Startup placement and host-residency hashes do not cover timing-dependent CPU/GPU/foresight/prefill choices. Add bounded owner transaction state and actual consumer decisions, bind exact replay/capture to appropriate revision and dependencies, then test delayed copies/partial failures/compulsory fills/prompt loans/all tiers. Adaptive features remain required. Existing scoped evidence remains valid only for its original components; full publication remains held while source work proceeds.

### Admission reactor lifetime review — revision1, 2026-10-27

Review-required for dependent admission enablement. Origin P08 FINDINGS `admission-reactor-identity-and-charge-lifetime-20261027`; main owns review. Require one noncopyable/nonmovable reactor identity and retain record metadata reservations through covered member destruction, including unresolved admission recovery. Prepare type-trait and retirement/deallocation checks; execute in a newly granted CPU window before claiming the repair validated. Independent source work continues. This correction does not qualify serving integration or change prior evidence.

### Idle activity timer review — revision1, 2026-10-27

Review-required for live activity expiry. Origin P07 FINDINGS `idle-activity-registry-timer-20261027`. Main owns integration. Use the whole priority registry deadline, including actors with no graph operation, for admission/reactor wakeups. Advance monotonic time then re-read after a conservative early wake; preserve absent deadline versus UINT64_MAX. Add no-request/no-operation expiry to two-user qualification and retain base priority. Registry/admission source wiring and tests are prepared; actual serving timed wait and CPU validation remain open. Independent source work continues.

### Review-required: verified KV refill submission — 2026-10-27

Origin P02 FINDINGS `verified-kv-refill-submission-20261027`. Main-owned source repair stops publication after a rejected copy even if later synchronization succeeds; earlier accepted copies must drain. Review partial topology/failure ownership before managed runtime enablement; no rollback claim. Exact failure fixtures, full backend builds and device qualification pending. Existing review gates remain; independent work continues.

### Review-required: retained CPU execution sink — 2026-10-27

P08 FINDINGS `retained-cpu-execution-sink-20261027`: selected-work-bound sink source is partial CPU diagnostics only, uncompiled/unwired. Review genuine source/runtime geometry, prepaid work/record lifetime, raw callback removal+join and quarantine contract. Do not use these records as complete GPU/kernel/reduction or numerical-profile authority. Main owns integration; source review and genuine success fixture in progress. Dependent publication remains held, independent work continues.

### Review-required: physical KV transfer disposition — 2026-10-27

P02 FINDINGS `physical-kv-transfer-disposition-20261027` requires actual consumed-handle disposition, bounded detached ownership and zeroing/completion proof before topology publication or retry. map_range false is not retained caller ownership; mapped is not necessarily initialized. Main integration pending; source-only evidence, existing managed gates retained.

### Review-required: CPU receipt work bound — 2026-10-27

P08 FINDINGS cpu-receipt-linear-bookkeeping-20261027 identifies quadratic row scans and a prepaid lookup replacement in progress. Review added metadata envelope, exact row/job ordering and no-allocation failure semantics before current sink enablement. Frozen prior fixture candidate remains distinct; no new performance/runtime claim.

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

## Review amendment: scheduling-worker-admission-20261028

Plan revision: additive 2026-10-28 worker-lifetime amendment; prior decisions retained. Plan remains review-required for scheduling-dependent runtime integration. See P03 FINDINGS.md, scheduling-worker-admission-20261028. Main owns resolution. Isolated logical stack implementation may proceed; actual serving enablement requires separately bounded registry/graph/snapshot ownership, safe stop/submit ordering, independent timer progress, and source-linked lifetime/failure tests. Supplied stack charge does not certify total process RSS. No change to this item's full feature or qualification scope; independent source work continues.

## Review amendment: source-map-foresight-serialization-20261028

Additive plan revision 2026-10-28. Scheduling/placement-dependent activation remains review-required under P02 FINDINGS. Main reviewed isolated source implementation with actual startup quiescence, setup sealing and FS-before-map lock order; external borrower and DMA lifetime still require genuine joins. Preserve full feature scope and old source-pinned evidence. Source work may proceed; changed source needs failure/lifetime/actual-driver qualification before enablement. Main owns downstream disposition.

## Review amendment: canonical-reservation-owner-20261028

Additive revision 2026-10-28: resource/lifetime-dependent integration remains review-required under P03 FINDINGS. Private canonical reservation ownership identity and each external dedicated-admission consumer must be joined, source-rebuilt and counterfactually tested; use_count alone is insufficient. Preserve prior pins/qualification, full feature scope and independent policy work. Main owns disposition. Isolated reviewed fix implementation allowed; enablement held until actual consumer and resource evidence completes.

## Suffix-first integration review — 2026-10-28

Status: review-required, not execution-qualified. Main reconciliation is recorded in `.local/strata-implementation/suffix-first/RECONCILIATION.md`. First experiment uses one session, RAW, same-length position-preserving four-token boundaries and speculation off. Preserve complete all-stage endpoint state and real retained/peak resource admission. Old GDN/PLE/index endpoint remains old-conditioned; suffix import must carry approximate ancestry and never enter the exact cache. Changed token history needs both content revision and append identity advancement; fresh tail forwarding must produce new logits. Unsupported forms use exact recomputation/refusal. Driver selection between dedicated fixed-runtime standalone path and owned APIs is under actual-source review; no prerequisite waived or broader feature marked complete. General scheduling/concurrency/disk/codec expansion is deferred, not removed. P13/P14 repair/maintenance remain later requirements; exact replay may serve the bounded experiment fallback.

## Review amendment: suffix-standalone-admission-20261028

Additive plan revision 2026-10-28; enablement remains review-required. Origin P12 FINDINGS.md. Main owns disposition: retain actual same-ledger global plus aggregate H charges through all source/splice/staging/input/diagnostic lifetimes; late input growth quote must cover old+new string overlap. Dedicated opaque fixed-runtime identity is process-local only, never a complete portable profile or exact-cache certificate. CPU adapter/budget/native/edit/metrics passed both modes; RAW helper and actual driver/GPU integration remain unqualified. Reviewed source work may proceed; dependent execution/enablement held until actual contracts and evidence close. Full scope and prior evidence unchanged.

Review amendment `suffix-diagnostic-admission-20261028` (P12 FINDINGS.md): main confirmed and excludes optional verifier diagnostics that add unadmitted buffers, copies, events or file output from the dedicated experiment. Dependent admission/execution qualification remains review-required; ordinary-engine diagnostic support unchanged. New actual-binary preflight tests are authored, UNRUN; include the fixed environmental profile and refusal evidence in the next bounded build/qualification. Main owns disposition.


Review amendment `suffix-frontier-reporting-20261028` (P12 FINDINGS/EVIDENCE): scoped ChatSwarm review and independent actual-consumer audit agree that exact_frontier=B describes retained exact-prefix provenance, not a truncatable complete GDN/PLE checkpoint inside hybrid endpointF. Repair must select a separately compatible complete exact checkpoint or replay; freshly forwarded descendants remain approximate. Reuse/fresh counts concern the successful prompt path, not all process work; imported recurrent/PLE endpoint still summarizes old whole history. Qualify decoded edited-fact correctness/stale behavior without assigning component cause, and distinguish one warm ordered endpoint timing pair from net latency or speedup. See suffix-first/gpu-proposal/{CHATSWARM-RESULT.json,FRONTIER-REVIEW.md,QUALITY-PROPOSAL.md}. Main owns disposition; source-contract review closes only this narrow question. Actual runtime, repair/convergence and production gates remain open. Current optional build linked, firstCLI help timedout5s; no complete combined qualification or GPU grant.


## Integration review update — 2026-10-28 late UTC

Actual checked verifier named layout is integrated, both CUDA86 engines link and four repair CMake tests pass (sourcee2b167,307.732s; report incremental-repair/INTEGRATED-QUALIFICATION-20261028.md). Newer setup/gate credit bridge and genuine stable geometry ownership are source-only after this freeze. Private dispatch proposal now orders preinit topology, actual completion-gate seal, init and postinit validation before readiness; initialized retirement and full two-attachment controller remain required. Successful data fixtures are not actual resident repair.


Owner dual RTX3090 acceptance requirement (2026-10-29): single-GPU complete-answer probe passed but cannot close this plan. Require correct outputs/suffix reuse across split model, repeated prefix repairs with real foreground between, coordinated two-device capture/restore/publication, failure cleanup and per-GPU bounds. Single-GPU evidence: suffix-first/gpu-quality-followup/QUALIFICATION-20261029.md. All dual-device gates remain open.

## Resident admission lifetime evidence — 2026-10-29

The real admission bridge and retained gate allocator passed Release and ASan/UBSan source/mock fixtures. See `.local/strata-implementation/suffix-first/incremental-repair/RESIDENT-ADMISSION-QUALIFICATION-20261029.md`. Retained allocator/deleter charges survive actual object and shared-control-block frees, including weak-last-owner lifetime. Earlier member-only and deleter-only approaches do not prove this invariant.

Review disposition: this validates the existing retained-storage contract; it does not change the budget or completion API. Downstream actual resident constructor, publication and retirement remain dependent on genuine admission, actual all-stage joins and separately bounded runtime overhead. No inference from these CPU fixtures to resident GPU/concurrent repair. P03, P13 and P15 preserve those gates. New integrated prerequisite build is in progress and its evidence must remain distinct from earlier binaries.

## Owner gate: eventual exactness before human testing (2026-10-29)

Require two actual partial repairs with foreground generation between, private catchup to the CURRENT endpoint, full native-state comparison against a clean reset/recomputation of identical accepted history, and continuation checks. Also qualify interrupted/stale publication, cancellation, allocation/device failures, correct resource lifetimes, repeated-cycle leaks and independent teardown. Initial single/dual reuse and component passes do not qualify this sequence.

Impact review: current dedicated serial controller implementation may continue, but human testing and full P13/P09/P15 acceptance remain held until all relevant gates pass. Full-state comparator and clean-replay instrumentation are being implemented; runtime failures remain explicit outstanding cells. This refines qualification evidence without dropping generic manager, server or concurrency requirements.

## Resident serial qualification review — 2026-10-29

Reviewed scoped outcome: `04-repair/p13/REVIEW-resident-serial-20261029.md` (work-items-relative). Dedicated RAW serial qualification is ready subject to exact final candidate/build/parser/resource checks. Full feature acceptance and the listed generic/concurrent/fault/cleanup gates remain incomplete; no inherited GPU qualification or codec enablement.

Measured resident review update (2026-10-29): dual3090 normal endpoint137/continuation138 and drained cancellation PASS/released; see P13 REVIEW-resident-serial-20261029.md. Pre-forward136clean native proof subsequently PASS/released with pinned5e6f/source51af; full138tensors compared,0mismatches. Reset/capture residual proof remains separate. Actualfault/leak/genericconcurrent/human gates remainOPEN. No default or fullfeature enablement follows.
