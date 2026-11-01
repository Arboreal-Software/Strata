> Current cross-plan qualification: two-cycle GPU test remains FAILED. The 2GiB cap correctly refused a measured excess; 4GiB allowed cycle1 complete native136/137/138 + logical retirement/budget PASS, but cycle2 native137 had48 KV/scale mismatches from stale future-page destination rows. Full-byte oracle remains strict. Actual supported RAW restore canonicalization, CPU controls and a fresh repeated-cycle GPU run are required; no human/concurrency/full-feature acceptance. Evidence: `.local/strata-implementation/suffix-first/incremental-repair/RESIDENT-REPEATED-RESTORE-FAILURE-20261029.md`. Attribution/prepared-cancel/deferred-fault revisions are applied but UNBUILT/UNQUALIFIED. BothGPU/CPU resources released; liveStrata stays down under owner authorization.

# Define window identity, cuts, and complete state descriptors (P01)

[Work queue](../../../WORK-QUEUE.md) · [Parent plan](../PLAN.md) · [Review protocol](../../REVIEW-PROTOCOL.md)

## Status and ownership

- Execution: implementing host descriptor checkpoint; engine adapter and runtime acceptance pending.
- Plan: ready, revision 2 for host descriptor checkpoint only; independent review and runtime acceptance pending.
- Owner: Strata implementation session under the owner handoff.
- Dependencies/integration inputs: [P00](../p00/PLAN.md)
- Direct dependent plans: [P02](../p02/PLAN.md), [P03](../p03/PLAN.md), [P04](../p04/PLAN.md), [P05](../../02-exact-parking/p05/PLAN.md), [P07](../../03-scheduling/p07/PLAN.md), [P10](../../03-scheduling/p10/PLAN.md), [P15](../../05-qualification/p15/PLAN.md)
- Source: Fig issue 30, derived from issues 25–29. Its design pin differs from the deployed engine; reconcile against baseline provenance before editing.

## Local execution plan

1. Reconcile the source proposal below with the pinned local engine and record exact files, existing behavior, missing mechanisms, and unresolved assumptions in FINDINGS.md.
2. Revise this plan with bounded changes and tests using that evidence. Record interfaces and resource budgets consumed by dependent plans. Historical research restrictions describe the source pass; current owner instructions govern execution.
3. Implement only the agreed scope in an isolated checkout, with the feature disabled by default where applicable. Keep the deployed service unchanged during preparation.
4. Run the applicable inert tests below; record commands, source pin, outcomes, and limitations in EVIDENCE.md. Coordinate any GPU/live qualification window separately.
5. Resolve or propagate plan-review triggers before dependent execution; update this plan, the parent, and queue status together. Mark complete only against its acceptance criteria.

## Findings that trigger plan review

Any missing hybrid-state part, identity field, continuation capability, or cut invariant. Review direct dependent plans listed above and propagate transitively where their contracts change. A changed state schema, shared interface, topology, or resource envelope also requires the final qualification plan to be reviewed. Do not silently assume an unaffected status: record the decision and rationale.

## Preserved detailed proposal and acceptance criteria

**Depends on:** P00.  
**Existing files:** conversation_cache.hpp, session.hpp, conversation_state.cpp.  
**Proposed new files:** include/strata/core/window_state.hpp; src/core/window_state.cpp.

Introduce WindowKey, StateCut, TensorPart, DevicePart, SnapshotDescriptor, and immutable BundleHandle. Include engine/window generations, r/a/s/L/C/F, pending-tail semantics, state layout/geometry, provenance, output watermark, and explicit source leases. Enumerate every required component and every intentionally rebuilt transient object.

Separate a “state usable for a new generation policy” capability from “exact continuation including RNG/parser.” Reject unsupported continuation state instead of dropping it.

**Failure contract:** STALE_GENERATION, INVALID_CUT, MISSING_STATE_PART, INCOMPATIBLE_LAYOUT; no active mutation.  
**Inert tests:** invalid F/C/L ordering, duplicate layer parts, missing GDN/PLE/index tail, reused window ID, absent pending token, unsupported parser state.  
**Acceptance:** no untyped opaque “KV blob” can claim a complete window; every payload has ownership/valid length/layout.  
**R1:** state-inventory and identity review against #25/#27.

## Review record

Initial plan extraction: 2026-10-27. No implementation or runtime qualification implied. No pending review finding has yet been evaluated for this package.

## Revision 2 checkpoint

Implement typed key/cut/layout/part validation and immutable owned bundle construction. Expected layout comes from the runtime adapter, never the imported descriptor. Test stale generation, every invalid frontier ordering, missing/duplicate hybrid components, pending-tail mismatch, stage cut mismatch, unsupported continuation and ownership isolation. This checkpoint does not complete P01; runtime inventory adapter and independent review remain. See [findings](FINDINGS.md).

## Observed serving envelope — 2026-10-27

[Target finding](../../01-foundations/p00/FINDINGS.md#actual-serving-envelope): current Flash-Next
serves a 256K INT8 context with MTP and automatic layer split; the research 128K/Q4 worksheet
is only illustrative. Include 256K INT8 and required two-session operation in resource/runtime
qualification. Deterministic first probes disable all speculation; current MTP compatibility is
not implied by the new descriptor adapter. Host-only schema/ledger checks remain useful but do
not settle this integration review. Actual resolved split/free capacity still needs evidence.

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

## Checked initial reset submission gate

Static audit found legacy session_zero/qsa_state_zero discard individual runtime submission
results; final last-error/synchronization alone is not the direct-call contract. Actual-method
CPU seam reproduced false success with a failing reset submission and clean last-error.
Earlier successful mock/build evidence remains scoped and does not prove this failure path.
New checked reset APIs are being qualified with direct-return errors; legacy void behavior
remains unchanged. Main initialize_sessions and prepare_owned_native_initial now consume
the checked return and poison/refuse before binding/publishing on failure. CPU3730 runs
a direct-false/clean-last-error preparation regression plus existing restore/allocator tests.
Acceptance remains pending final backend/fault evidence; all-device drain/restore and host
producer closure are still required after successful enqueue. No GPU window is available.

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

### Verified-input and checked-reset downstream review

Input groups now retain full expected-digest snapshots, named loader aliases and
metadata/disk reservations; explicit checked retirement seals new leases and retries
ordinary IO admission refusal without deleting uncharged resources. Tiny source
fixtures passed sanitizer68977; actual-library fixture is in pending CUDA3776.
Manifest parser/typed binding remains under implementation; actual driver model reads
are not yet verified input consumption. Early input-budget carveout must reconcile
with later model/session admission before advertising complete instance accounting.

Serving reset fault validation65476 passed eighteen API faults, six preflight refusals
and exception-after-partial-mutation poisoning. In-flight aggregate3776 predates
exception fix; prepared frozen-runtime-reset-exception includes it. Managed executor
still requires fresh generation/private publication instead of these legacy in-place
resets. CPU/mock results do not prove numerical parity or concurrent-window correctness.

### PLE plain-group compatibility review (2026-10-27)

Plan review: review-required for affected runtime numerical acceptance. See P00
FINDINGS.md “PLE neutral plain grouping”. Baseline passed; owned startup refusal
reopens compatibility. Keep numerical qualification pending until neutral0/1-only
validation fix, refusal regressions, source-linked backend builds and parity rerun.
Independent implementation remains authorized. Main owns final review resolution.

PLE plain-group review resolution (2026-10-27): neutral0/1 correction passed
CPU red/green1603, CUDA42161/13CTest/PLE and HIP36629. GPU run47280 passed
four serial default/owned single/dual arms with two requests each, identical
tokens/text/full first-logit bytes and clean exits. Narrow compatibility review
is resolved for pinned candidate a0cf089... only. Later source and complete
executor/concurrency/large-context acceptance remain pending. Exact evidence:
.local/strata-implementation/gpu-qualification/run-20261027T125947Z-strata-window-20.

### Weight screening / exact document-prefix investigations (2026-10-27)

Separate owner-added child plans: P11/weight-stream-screen and P04/shared-document-prefix.
Review-required only for any resulting weight format/resource/streaming change or exact
prefix branch acceptance affecting this item. Source/metadata and bounded synthetic tests
may proceed; model SSD reads and all GPU runs require separate root coordination. Existing
state codec, approximate suffix and deferred immutable chunk-sharing scopes are unchanged.
Use actual engine-specific source and measurements; DeepSeek and Qwen results do not transfer.

### Runtime policy observation review (2026-10-27)

Review-required for exact reusable profile publication: P00 FINDINGS.md records missing
resolved numerical policies and pre-main cached selectors. Actual loader inventory and
read-only consumer observations are being implemented with explicit bounds/retained
owners; partial records cannot certify completeness. Adaptive state/assignment and
reactor mutation fencing remain unresolved. Legacy defaults remain unchanged; source-only
implementation continues. Source audit evidence is not numerical equivalence proof.

### Actual profile receipt contract — revision 1, 2026-10-27

Plan: **review-required** for complete verified-profile publication and dependent managed enablement. Origin: P00 FINDINGS.md `actual-profile-receipt-retention-20261027`; main integration owns review. Prior loader/owner evidence retains its original source scope. New effective-profile parser capture has a dedicated bounded host reservation and retained immutable descriptor authority, surviving independent stage/gang lifetime. It records requested ranking only; sizing reads, peer reorder, capacity/stage trimming and adaptive placement still need actual effective-policy accounting. PLE startup metadata must be checked against the retained table generation at publication.

Required evidence: current parser failure/identity/close/allocation tests, driver admitted-path/refusal checks, owner/quarantine lifetime, and touched-backend builds before acceptance. Current source is uncompiled; no concurrent or hardware qualification follows. Independent source work is allowed during scheduling hold; publication remains deferred pending these gates.

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

### Review-required: CPU receipt work bound — 2026-10-27

P08 FINDINGS cpu-receipt-linear-bookkeeping-20261027 identifies quadratic row scans and a prepaid lookup replacement in progress. Review added metadata envelope, exact row/job ordering and no-allocation failure semantics before current sink enablement. Frozen prior fixture candidate remains distinct; no new performance/runtime claim.

### Review-required: prepaid elastic KV attachment — 2026-10-27

P03 elastic-kv-prepaid-attachments-20261027 requires distinct arena/private-work and physical/private-attachment quotes, all reservations before first device effect, exclusive batch consumption and retained parent quota through donation. Source implementation in progress; no gang/backend/runtime validation. Existing dependent publication/restore/transfer gates retained.

Prepaid elastic integration review remains required: P03 FINDINGS elastic-kv-prepaid-attachments-20261027 now records host metadata/handle allocation ordering and error-drain gaps in the initial owner. Optional build wiring and PLE host preallocation are source-only. Main owns dependent integration; validate revised VMM/owner source and backend compatibility before enablement. Existing restore, growth/donation, capacity-frontier and concurrency acceptance stays open.

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

### Integrated source closure review — 2026-10-28

Dependent combined qualification is **review-required**, origin P00 FINDINGS integrated-source-closure-20261028. Restore missing previously intended producer/test/CMake changes additively while retaining newer fixes, re-freeze exact source and complete required target/test mapping. Do not remove failed targets to make the build pass. Main owns review; prior frozen qualification remains valid only for its original source. Independent isolated source work may continue.

## Review amendment: source-map-foresight-serialization-20261028

Additive plan revision 2026-10-28. Scheduling/placement-dependent activation remains review-required under P02 FINDINGS. Main reviewed isolated source implementation with actual startup quiescence, setup sealing and FS-before-map lock order; external borrower and DMA lifetime still require genuine joins. Preserve full feature scope and old source-pinned evidence. Source work may proceed; changed source needs failure/lifetime/actual-driver qualification before enablement. Main owns downstream disposition.

## Review amendment: canonical-reservation-owner-20261028

Additive revision 2026-10-28: resource/lifetime-dependent integration remains review-required under P03 FINDINGS. Private canonical reservation ownership identity and each external dedicated-admission consumer must be joined, source-rebuilt and counterfactually tested; use_count alone is insufficient. Preserve prior pins/qualification, full feature scope and independent policy work. Main owns disposition. Isolated reviewed fix implementation allowed; enablement held until actual consumer and resource evidence completes.

## Suffix-first integration review — 2026-10-28

Status: review-required, not execution-qualified. Main reconciliation is recorded in `.local/strata-implementation/suffix-first/RECONCILIATION.md`. First experiment uses one session, RAW, same-length position-preserving four-token boundaries and speculation off. Preserve complete all-stage endpoint state and real retained/peak resource admission. Old GDN/PLE/index endpoint remains old-conditioned; suffix import must carry approximate ancestry and never enter the exact cache. Changed token history needs both content revision and append identity advancement; fresh tail forwarding must produce new logits. Unsupported forms use exact recomputation/refusal. Driver selection between dedicated fixed-runtime standalone path and owned APIs is under actual-source review; no prerequisite waived or broader feature marked complete. General scheduling/concurrency/disk/codec expansion is deferred, not removed. P13/P14 repair/maintenance remain later requirements; exact replay may serve the bounded experiment fallback.

## Review amendment: suffix-standalone-admission-20261028

Additive plan revision 2026-10-28; enablement remains review-required. Origin P12 FINDINGS.md. Main owns disposition: retain actual same-ledger global plus aggregate H charges through all source/splice/staging/input/diagnostic lifetimes; late input growth quote must cover old+new string overlap. Dedicated opaque fixed-runtime identity is process-local only, never a complete portable profile or exact-cache certificate. CPU adapter/budget/native/edit/metrics passed both modes; RAW helper and actual driver/GPU integration remain unqualified. Reviewed source work may proceed; dependent execution/enablement held until actual contracts and evidence close. Full scope and prior evidence unchanged.


Review amendment `in-memory-incremental-repair-next-20261028`: P13 revision 3 is now the immediate next milestone after suffix reuse. New dependent repair integration remains review-required: retain a separate complete exact checkpoint, advance it only after bounded real all-layer units, preserve ancestry/age on approximate reattachment, fence edits/appends/selection and final publication, price full-copy/switch overlap, and qualify foreground interference/catch-up. Exact-frontier metadata in a hybrid endpoint is not a complete exact checkpoint. Main owns source/interface disposition; independent source preparation is allowed. Existing qualified suffix candidate, prior evidence and full acceptance remain unchanged. No new resource grant or serving change. See P13 FINDINGS.md for impact mapping and held work.

## Resident serial qualification review — 2026-10-29

Reviewed scoped outcome: `04-repair/p13/REVIEW-resident-serial-20261029.md` (work-items-relative). Dedicated RAW serial qualification is ready subject to exact final candidate/build/parser/resource checks. Full feature acceptance and the listed generic/concurrent/fault/cleanup gates remain incomplete; no inherited GPU qualification or codec enablement.

Measured resident review update (2026-10-29): dual3090 normal endpoint137/continuation138 and drained cancellation PASS/released; see P13 REVIEW-resident-serial-20261029.md. Pre-forward136clean native proof subsequently PASS/released with pinned5e6f/source51af; full138tensors compared,0mismatches. Reset/capture residual proof remains separate. Actualfault/leak/genericconcurrent/human gates remainOPEN. No default or fullfeature enablement follows.
