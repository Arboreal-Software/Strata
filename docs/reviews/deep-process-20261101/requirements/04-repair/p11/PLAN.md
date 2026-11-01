# nvCOMP and hipCOMP compression with RAW fallback (P11)

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

- Execution: implementing; RAW/planner and real SDK adapter compile gates passed. Async jobs in progress; no codec hardware qualification.
- Plan: revision 5; review-required for corrected planner peak/stream contract; bounded async ownership repair authorized. Persistence/serving integration and hardware acceptance remain review-required.
- Owner: Strata implementation session under the owner handoff.
- Dependencies/integration inputs: [P03](../../01-foundations/p03/PLAN.md), [P06](../../02-exact-parking/p06/PLAN.md), [P08](../../03-scheduling/p08/PLAN.md), [P09](../../03-scheduling/p09/PLAN.md), [P10](../../03-scheduling/p10/PLAN.md)
- Direct dependent plans: [P15](../../05-qualification/p15/PLAN.md)
- Source: Fig issue 30, derived from issues 25–29. Its design pin differs from the deployed engine; reconcile against baseline provenance before editing.

## Owner addition: hipCOMP and Dax hardware qualification (2026-10-27)

Alongside nvCOMP, implement/evaluate hipCOMP support for a potential RX 6800 XT
Strata deployment. Compression is a high-priority capability for this constrained-memory
AMD target. Runtime compression remains optional with independent RAW fallback; the
implementation/evaluation work for both backends is in scope and cannot be silently omitted.

**Test host: Dax.** Claire reports that Dax currently has a GTX 1080 and will receive an
RX 6800 XT soon. Claire explicitly authorizes connecting to Dax over SSH and running the
needed tests once the RX 6800 XT is installed. The installation has not been verified and
no RX 6800 XT result may be inferred from current hardware. Existing coordinator locators
are 10.0.0.131 / Tailscale 100.78.236.127; verify current SSH access and actual device/driver
inventory when the hardware is ready. Coordinate test windows with the parent session
and inspect active workloads first. This grants Strata qualification access, not ownership
of Dax setup/voice or permission to interrupt unrelated services. No hardware run now.

Qualification on Dax must cover actual engine capture/restore and hipCOMP lossless byte
roundtrips, malformed input, RAW fallback, cancellation/drain/error retention, workspace and
peak VRAM/host/pinned memory, compression ratio and end-to-end switching latency. Test
representative complete hybrid state, including poorly compressible/tail chunks. Establish
RX 6800 XT/gfx1030 build/runtime evidence separately from prior gfx1100/gfx1201 builds.
Where supported, test nvCOMP-to-hipCOMP and hipCOMP-to-nvCOMP restore compatibility on
identical canonical payloads; persist codec/framing/version metadata and reject unsupported
encodings explicitly. API similarity is not proof of byte-format interoperability.

Initial primary source: https://github.com/ROCm/hipCOMP-core (read 2026-10-27) describes
hipCOMP v2.3.0 as based on nvCOMP2.2, lists LZ4 encode/decode and tested gfx1030, and marks
it early-access. Our existing nvCOMP6.0.0 proposal therefore requires explicit API/version
comparison, not a presumed drop-in substitution. Pin source/package/compiler variants
before integration; source support claims are not Dax measurements.

## Local execution plan

1. Reconcile the source proposal below with the pinned local engine and record exact files, existing behavior, missing mechanisms, and unresolved assumptions in FINDINGS.md.
2. Pin and compare nvCOMP and hipCOMP API/ABI, algorithm, framing and supported-target contracts; design a shared backend-neutral codec interface with explicit backend implementations. Revise this plan with bounded changes and tests using that evidence. Record interfaces and resource budgets consumed by dependent plans. Historical research restrictions describe the source pass; current owner instructions govern execution.
3. Implement only the agreed scope in an isolated checkout, with the feature disabled by default where applicable. Keep the deployed service unchanged during preparation.
4. Run the applicable inert tests below; record commands, source pin, outcomes, and limitations in EVIDENCE.md. Coordinate any GPU/live qualification window separately.
5. Resolve or propagate plan-review triggers before dependent execution; update this plan, the parent, and queue status together. Mark complete only against its acceptance criteria.

## Findings that trigger plan review

Codec compatibility, compression benefit, scratch/transition peaks, or RAW fallback changes. Review direct dependent plans listed above and propagate transitively where their contracts change. A changed state schema, shared interface, topology, or resource envelope also requires the final qualification plan to be reviewed. Do not silently assume an unaffected status: record the decision and rationale.

## Preserved detailed proposal and acceptance criteria

**Depends on:** P03/P06/P08/P09; uses P10 for attribution during later evaluation.  
**Existing files:** native part registry and v2 file encoder/decoder.  
**Proposed files:** include/strata/core/window_codec.hpp; src/core/window_codec_nvcomp.cu; src/core/window_codec_hipcomp.cpp (HIP compilation); separate optional build feature definitions.

Bind a matching nvCOMP 6.0.0 header/library package and explicit CUDA variant only under new installation/build authority. Official API/source evidence is in #28. Add per-device CodecPlan/EncodeBatch/DecodeBatch/Poll/Cancel/Drain with queried alignment, maximum output, workspace, actual-size/status arrays, and retained event leases.

Direct decode targets private contiguous native spans; otherwise admit bounded scatter. RAW fallback is independent and selected when compressed storage loses or codec unavailable. Host-authoritative KV takes its distinct cost path. No presumed cross-device decoder, 3090 Blackwell DE, or compression benefit.

**Failures:** SDK mismatch, corrupt chunk, scratch/output cap, actual-size/status error, cancel/drain, device failure; never publish a partial bundle.  
**Inert tests:** native byte equality, mixed RAW/LZ4, incompressible/tail chunks, alignment, version matrix, first staging demand, two-device failure.  
**Acceptance:** codec can be disabled without breaking RAW restore; all byte/lifetime/resource gates pass before performance evaluation.  
**R11:** dependency/provenance, codec and persistence review for both nvCOMP and hipCOMP; actual Dax RX 6800 XT qualification once installed.

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

Scoped contract ready; full integration remains open. See [disk-v2 contract](../../02-exact-parking/p06/DISK-V2-IMPLEMENTATION.md).

Scoped plan amendment owned-mapped-runtime-20261027: reviewed/replanned per
`03-scheduling/p08/MAPPED-RUNTIME-REVIEW.md` under work-items. Ready for bounded
implementation; full runtime/hardware acceptance remains open. Earlier decisions and
evidence retained; no serialization or scheduling scope change.

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

## Async codec ownership review — revision 4 (2026-10-27)

Previous revision3 did not select concrete ownership/API contracts. CPU sanitizer18986
and SDK compile/link81176 establish the RAW/planner/typed adapter slice only. Actual
execution capability stays false until implemented and qualified. nvCOMP6 and hipCOMP2.3
have distinct APIs/framing; no cross-vendor encoded compatibility is assumed.

Approved implementation: immutable retained input descriptors and bytes; private per-job
stream/event; conservative control envelope and all output/workspace/device/pinned/
verification reservations before enqueue. Every beneficial encode roundtrips and validates
bytes/digests before publication. Decode checks actual lengths/statuses/digests; external
compressed-input grammar safety must be resolved before corrupt-input qualification.
Cancel suppresses publication but retains ownership until confirmed drain. Ambiguous
enqueue/completion/free retains a preallocated quarantine node and all charges; no
destructor retry or RAW fallback after uncertain work. RAW fallback is allowed only before
enqueue or after confirmed drain. Vendor provenance stays exact; hipCOMP has no invented
compression-status API. Multi-stage atomic publication remains the manager's responsibility.

Direct/transitive review: P03 reserve peak+quarantine, P05 private decode/generation fences,
P06 distinct encoded framing preserving existing RAW format, P08/P09 async lifetime and
atomic publication, P10 attribution, P12/P13/P14 compatibility, P15 actual CUDA/HIP byte
roundtrip/corrupt/cancel/two-device tests and measured peaks/ratio/latency. Independent
RAW paths remain permitted. These integration/hardware claims remain held. Owner-added
Dax RX6800XT SSH qualification remains pending installation and parent coordination.

## Owner-added separate investigation (2026-10-27)

See [weight-stream-screen](weight-stream-screen/PLAN.md). Source/metadata work authorized; substantial
SSD traffic and all GPU/live runs remain separately coordinated. No qualification
transfers from the serial construction-parity candidate.

### Integration review update — actual load receipts and codec admission

NativeDense and canonical WeightTable inventory producers have returned final CPU sanitizer evidence, including actual model-owner quarantine for native allocations and21 canonical runtime failures. Driver primary/stage metadata admission is wired under separate early verified-input reservations. This is not complete identity coverage or backend qualification: actual Head/Embed/expert/config collection, final provenance joins and aggregate CUDA/HIP/SYCL validation remain required. Reconcile downstream ownership/admission/publication requirements before enabling managed execution.

Async codec review found the standalone planner peak undercounts actual job buffers and lacks its stated intended-stream binding. Shared checked layout and stream-contract repair is assigned; preserve hardware_qualified=false and pending serving/disk integration. Review revised admission peaks, result ownership and cancellation/quarantine contracts in dependent plans; do not treat independent job reservations as proof that the scheduler estimate is correct. Review evidence: .local/strata-implementation/window-codec/review-package/RESULT.txt.

### Admission reactor lifetime review — revision1, 2026-10-27

Review-required for dependent admission enablement. Origin P08 FINDINGS `admission-reactor-identity-and-charge-lifetime-20261027`; main owns review. Require one noncopyable/nonmovable reactor identity and retain record metadata reservations through covered member destruction, including unresolved admission recovery. Prepare type-trait and retirement/deallocation checks; execute in a newly granted CPU window before claiming the repair validated. Independent source work continues. This correction does not qualify serving integration or change prior evidence.

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
