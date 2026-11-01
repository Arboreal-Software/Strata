# Local disk v2, receipts, and crash recovery (P06)

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

- Execution: RAW-v2 component implemented, sanitizer/CMake tests passed and CUDA/HIP compiled; independent review dispatched. Serving/Windows/GC remain open.
- Plan: revision 3 ready for bounded RAW-v2 contract; full serving integration remains review-required.
- Owner: Strata implementation session under the owner handoff.
- Dependencies/integration inputs: [P03](../../01-foundations/p03/PLAN.md), [P05](../p05/PLAN.md)
- Direct dependent plans: [P08](../../03-scheduling/p08/PLAN.md), [P09](../../03-scheduling/p09/PLAN.md), [P11](../../04-repair/p11/PLAN.md), [P14](../../04-repair/p14/PLAN.md), [P15](../../05-qualification/p15/PLAN.md)
- Source: Fig issue 30, derived from issues 25–29. Its design pin differs from the deployed engine; reconcile against baseline provenance before editing.

## Local execution plan

1. Reconcile the source proposal below with the pinned local engine and record exact files, existing behavior, missing mechanisms, and unresolved assumptions in FINDINGS.md.
2. Revise this plan with bounded changes and tests using that evidence. Record interfaces and resource budgets consumed by dependent plans. Historical research restrictions describe the source pass; current owner instructions govern execution.
3. Implement only the agreed scope in an isolated checkout, with the feature disabled by default where applicable. Keep the deployed service unchanged during preparation.
4. Run the applicable inert tests below; record commands, source pin, outcomes, and limitations in EVIDENCE.md. Coordinate any GPU/live qualification window separately.
5. Resolve or propagate plan-review triggers before dependent execution; update this plan, the parent, and queue status together. Mark complete only against its acceptance criteria.

## Findings that trigger plan review

Persistence format, post-rename outcomes, durability, idempotency, or quota accounting changes. Review direct dependent plans listed above and propagate transitively where their contracts change. A changed state schema, shared interface, topology, or resource envelope also requires the final qualification plan to be reviewed. Do not silently assume an unaffected status: record the decision and rationale.

## Preserved detailed proposal and acceptance criteria

**Depends on:** P03/P05.  
**Existing files:** conversation_file.hpp/cpp, generate.cpp session command adapter, serve/server.py file adapter.  
**Proposed files:** include/strata/core/window_file.hpp; src/core/window_file.cpp; a bounded local catalog/operation journal implementation.

Define v2 bounded manifest/chunk records and RAW payloads first. Preserve v1 reader constraints. SaveWindow obtains immutable bytes, writes operation-owned temp, verifies/flushes, renames to an immutable artifact, confirms directory durability where supported, then updates local catalog by CAS. No raw device pointer is persisted.

LoadWindow prepares privately and commits explicitly. QueryOperation reconciles duplicate/ambiguous requests by idempotency key and input digest. Opaque artifact IDs stay within the configured local cache area. Quota includes old leased artifacts, new temp/published bytes, and reserve.

**Failures:** distinguish not published, file published/durability unknown, catalog linked, and selected-state committed. A post-rename error cannot be treated as no effect.  
**Inert tests:** fault injection at write/file flush/rename/directory flush/catalog receipt, crash before/after each step, disk full, malicious lengths/overflows, lease-aware delete, same-size model middle-content change.  
**Acceptance:** exact retry reconciliation and no old-good artifact destruction; current sampled model fingerprint cannot substitute for the required identity.  
**R6:** persistence/compatibility review.

## Review record

Initial plan extraction: 2026-10-27. No implementation or runtime qualification implied. No pending review finding has yet been evaluated for this package.

## Review required — 2026-10-27

Owner: Strata implementation session. Findings: [target](../../01-foundations/p00/FINDINGS.md), [complete cut](../../01-foundations/p01/FINDINGS.md). Reconcile common stage cut, accepted pending history, typed owned payloads and private publication against the deployed engine base. Dependent integration is held; independent source inspection and inert test design may proceed. No prior qualification is inherited. Resolution remains pending.

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

Managed priority/dependency scoped review (2026-10-27): see work-items/03-scheduling/p07/MANAGED-LIFECYCLE-REVIEW.md. Component revision4 is ready and sanitizer-tested; runtime consumers must retain full managed identities, refresh at ordered control/timer cuts, account transactional memory, and preserve hard eligibility. Integration/qualification remains incomplete.

## Serialized-control and teardown checkpoint (2026-10-27)

Plan: review-required for remaining dependent runtime integration. The scoped consumer
contract and tests are reviewed in [checkpoint review](../../03-scheduling/p08/SERIALIZED-CONTROL-REVIEW.md).
Prior evidence retains its original source scope; no feature acceptance is implied.

## Session runtime interface review (2026-10-27)

Scoped interface decision and remaining held integration are recorded in [session runtime review](../../03-scheduling/p08/SESSION-RUNTIME-INTEGRATION.md). Full acceptance remains open.

## Disk-v2 implementation review (2026-10-27)

Scoped contract ready; full integration remains open. See [disk-v2 contract](DISK-V2-IMPLEMENTATION.md).

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

### Capture and disk recovery integration review (2026-10-27)

Ledger capture export now retains an exact selected read plus separately admitted immutable
accepted/pending history, preserving pending EOS and output watermark. Approximate cuts
require matching validated ancestry; no inferred exactness. Actual capture consumes
sessions/verifiers from that same retained runtime; failed transfer owners remain retained.
CPU ASan/UBSan evidence10748 covers actual capture adapter with mocked native saves, not
GPU copies. Driver serving connection remains required.

Disk rereview reproduced post-link stat-error misclassification and late returned-handle
admission. Fixes retain publication-cut knowledge with tri-state probes and reserve the
512-byte returned handle before first publication. Red12228 and sanitizer7573 evidence
are preserved. This does not make every later filesystem/disk allocation failure
side-effect-free. Combined immutable CUDA87505 is running; HIP/rereview follow-up pending.
Dependent integration must preserve uncertain publication and source read ownership,
with no numerical/concurrent qualification inferred from these component tests.

### Async compression downstream contract review (2026-10-27)

P11 revision4 authorizes immutable owned-input jobs with pre-enqueue peak reservations,
verified private output and retained quarantine on uncertain drain/free. RAW fallback
requires no enqueue or confirmed drain. No vendor framing interoperability assumed.
Review-required: this item's encoded-state serving/persistence/multi-stage acceptance;
use P11 revision4's explicit integration gates before enabling. Independent RAW work
continues. Main owns integration resolution; CPU/SDK compilation is not hardware proof.

### Disk operation-binding review resolution (2026-10-27)

ChatSwarm1793106180.265029 reassembled18/18 parts and verified19/19 file hashes,
reviewing exact disk.cpp76a0a31... with GPT-6 Extra High (requested5.6Pro not
substantiated). No unresolved demonstrated changed-path defect: own-operation intent
binding prevents same-content receipt promotion; prepared success results preserve caller
output on save/catalog OOM; old output retires outside save lock. CPU red36418/green54485
and CUDA95211/HIP11127 are retained. This resolves that bounded source-repair gate only.
Actual selected-runtime disk execution, physical crash recovery, global RSS, non-POSIX
platforms and concurrency remain pending; parent acceptance stays incomplete.

### Priority response preparation review — revision1, 2026-10-27

Review-required for priority transport/admission enablement. Origin P07 FINDINGS `priority-response-before-commit-20261027`; main owns review. Complete allocating success receipts before committing the requested mutation, preserve accepted/applied and full incarnation identities, and retain real admitted work on acknowledgement preparation failure. Account overlapping prepared-output/current-record/insertion metadata. Timer expiry is an independent allowed effect; do not claim whole-cut rollback or durable/network idempotency. Add persistent allocation failure sweeps before public transport qualification. Source work continues; tests/builds unrun.

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

## Review amendment: source-map-foresight-serialization-20261028

Additive plan revision 2026-10-28. Scheduling/placement-dependent activation remains review-required under P02 FINDINGS. Main reviewed isolated source implementation with actual startup quiescence, setup sealing and FS-before-map lock order; external borrower and DMA lifetime still require genuine joins. Preserve full feature scope and old source-pinned evidence. Source work may proceed; changed source needs failure/lifetime/actual-driver qualification before enablement. Main owns downstream disposition.

## Review amendment: canonical-reservation-owner-20261028

Additive revision 2026-10-28: resource/lifetime-dependent integration remains review-required under P03 FINDINGS. Private canonical reservation ownership identity and each external dedicated-admission consumer must be joined, source-rebuilt and counterfactually tested; use_count alone is insufficient. Preserve prior pins/qualification, full feature scope and independent policy work. Main owns disposition. Isolated reviewed fix implementation allowed; enablement held until actual consumer and resource evidence completes.
