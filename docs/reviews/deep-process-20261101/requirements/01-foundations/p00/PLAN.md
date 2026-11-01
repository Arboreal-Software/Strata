# Freeze the implementation target and compatibility envelope (P00)

[Work queue](../../../WORK-QUEUE.md) · [Parent plan](../PLAN.md) · [Review protocol](../../REVIEW-PROTOCOL.md)

## Status and ownership

- Execution: Partially complete: build baseline done; compatibility envelope pending.
- Plan: ready, revision 2 for target reconciliation; runtime compatibility acceptance remains pending.
- Owner: Strata implementation session under the owner handoff.
- Dependencies/integration inputs: Completed build baseline; target reconciliation is next.
- Direct dependent plans: [P01](../p01/PLAN.md), [P15](../../05-qualification/p15/PLAN.md)
- Source: Fig issue 30, derived from issues 25–29. Its design pin differs from the deployed engine; reconcile against baseline provenance before editing.

## Local execution plan

1. Reconcile the source proposal below with the pinned local engine and record exact files, existing behavior, missing mechanisms, and unresolved assumptions in FINDINGS.md.
2. Revise this plan with bounded changes and tests using that evidence. Record interfaces and resource budgets consumed by dependent plans. Historical research restrictions describe the source pass; current owner instructions govern execution.
3. Implement only the agreed scope in an isolated checkout, with the feature disabled by default where applicable. Keep the deployed service unchanged during preparation.
4. Run the applicable inert tests below; record commands, source pin, outcomes, and limitations in EVIDENCE.md. Coordinate any GPU/live qualification window separately.
5. Resolve or propagate plan-review triggers before dependent execution; update this plan, the parent, and queue status together. Mark complete only against its acceptance criteria.

## Findings that trigger plan review

Source/model/topology/numeric-profile or compatibility changes. Review direct dependent plans listed above and propagate transitively where their contracts change. A changed state schema, shared interface, topology, or resource envelope also requires the final qualification plan to be reviewed. Do not silently assume an unaffected status: record the decision and rationale.

## Preserved detailed proposal and acceptance criteria

**Depends on:** new explicit implementation authority, none of the technical packages.  
**Inputs:** #24–#29; source pin; actual model pack and intended context/slot targets.  
**Files:** proposed docs/window_state_design.md and a machine-readable compatibility manifest; no runtime changes in this package.

Record actual model/artifact/tokenizer/template/draft digests, engine build and numeric flags, Ubuntu/driver/toolkit/compiler, device IDs/capacities/topology, native KV format, layer split, batch groups, speculation modes, and local disk policy. Reconcile source drift before applying this map. Do not substitute the old dense27B assessment or assume current main remains pinned forever.

Choose an initial deterministic, text-only, all-speculation-disabled correctness profile while retaining planned expansion to required dual/concurrent modes. Inventory every speculation path, not just the MTP flag. Capture applicable rights/provenance review needs separately from engineering.

**Failure/stop:** unknown model/layout or an unsupported required mode is a recorded blocker to that implementation milestone, not an invented serving/billing gate.  
**Inert acceptance:** completeness checklist and source-locator review; no measured capacity claimed until separately authorized inspection.  
**Review gate R0:** owner agrees target/matrix and permitted implementation/review route. No Codex/Copilot review.

## Review record

Initial plan extraction: 2026-10-27. No implementation or runtime qualification implied. No pending review finding has yet been evaluated for this package.

### hipcomp-dax-target-20261027 — scoped review required

Owner adds hipCOMP alongside nvCOMP for RX 6800 XT. Dax currently has GTX 1080;
SSH qualification tests are authorized once the RX 6800 XT is installed, coordinated
with parent/active workloads. See 04-repair/p11/PLAN.md for authoritative target,
codec compatibility and test requirements. Review this item's AMD codec capability,
resource/format/lifetime assumptions and qualification consumers before codec-dependent
integration. Existing CUDA/RAW work is unaffected and continues; no hardware pass claimed.

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

### Integration review update — actual load receipts and codec admission

NativeDense and canonical WeightTable inventory producers have returned final CPU sanitizer evidence, including actual model-owner quarantine for native allocations and21 canonical runtime failures. Driver primary/stage metadata admission is wired under separate early verified-input reservations. This is not complete identity coverage or backend qualification: actual Head/Embed/expert/config collection, final provenance joins and aggregate CUDA/HIP/SYCL validation remain required. Reconcile downstream ownership/admission/publication requirements before enabling managed execution.

Async codec review found the standalone planner peak undercounts actual job buffers and lacks its stated intended-stream binding. Shared checked layout and stream-contract repair is assigned; preserve hardware_qualified=false and pending serving/disk integration. Review revised admission peaks, result ownership and cancellation/quarantine contracts in dependent plans; do not treat independent job reservations as proof that the scheduler estimate is correct. Review evidence: .local/strata-implementation/window-codec/review-package/RESULT.txt.

### Actual profile receipt contract — revision 1, 2026-10-27

Plan: **review-required** for complete verified-profile publication and dependent managed enablement. Origin: P00 FINDINGS.md `actual-profile-receipt-retention-20261027`; main integration owns review. Prior loader/owner evidence retains its original source scope. New effective-profile parser capture has a dedicated bounded host reservation and retained immutable descriptor authority, surviving independent stage/gang lifetime. It records requested ranking only; sizing reads, peer reorder, capacity/stage trimming and adaptive placement still need actual effective-policy accounting. PLE startup metadata must be checked against the retained table generation at publication.

Required evidence: current parser failure/identity/close/allocation tests, driver admitted-path/refusal checks, owner/quarantine lifetime, and touched-backend builds before acceptance. Current source is uncompiled; no concurrent or hardware qualification follows. Independent source work is allowed during scheduling hold; publication remains deferred pending these gates.

### Placement and actual execution decision review — revision1

Review-required for complete managed identity/cuts/replay. Origin P00 FINDINGS `placement-versus-execution-decisions-20261027`; main owns integration. Startup placement and host-residency hashes do not cover timing-dependent CPU/GPU/foresight/prefill choices. Add bounded owner transaction state and actual consumer decisions, bind exact replay/capture to appropriate revision and dependencies, then test delayed copies/partial failures/compulsory fills/prompt loans/all tiers. Adaptive features remain required. Existing scoped evidence remains valid only for its original components; full publication remains held while source work proceeds.

### Initial expert tier join review — revision1, 2026-10-27

Review-required for managed identity/publication. Origin P10 FINDINGS `initial-expert-tier-join-20261027`; main owns integration. Require actual completion of every declared initial tier at the post-helper-load cut, then invalidate before untracked dynamic setup/execution. Future tracker must seed at that cut and retain exact owners, not reuse a stale initial receipt. Test missing helpers, partial producer failure and successful join/invalidation; retain absent-profile/dynamic/execution gates. Source-only, no backend/numerical qualification.

### Explicit disabled profile review — revision1, 2026-10-27

Review-required for complete profile/managed publication. Origin P00 FINDINGS `explicit-disabled-expert-profile-20261027`; main owns review. Retain the genuine manifest-disabled claim and actual model geometry without manufacturing file header/path/ranking evidence. Price metadata before construction, revoke on claim/read-generation change, and join real initial cache capacity even without a ranking file. Test forged/public-only flags, lease conflicts, absent geometry, insufficient quota, output preservation and actual read invalidation; rebuild all affected inventory consumers before qualification. Source-only; existing dynamic/executor gates stay open.

### Review-required: actual expert runtime profile join — 2026-10-28

P01 FINDINGS `actual-expert-runtime-profile-join-20261028` changes source receipt ownership and metadata lifetime. Main must review this consumer’s authority/invalidation/accounting assumptions before complete-profile-dependent activation. Source-only collector and driver wiring are untested; no previous build/fixture evidence covers them. Existing execution/qualification gates remain.

### Review-required: actual running main image — 2026-10-28

P00 FINDINGS `actual-running-main-image-20261028` introduces schema3 startup observation and distinct transient IO/retained metadata charges. Main must review this consumer’s identity/schema and lifetime dependencies; producer/driver source is untested. Complete execution identity remains gated on library/JIT closure. Independent source work may continue.


### Review-required: remote routing consumer lifetime — 2026-10-28

Origin: P02 FINDINGS `remote-routing-consumer-lifetime-20261028`. Main integration must reconcile actual primary-output consumption, helper/copy-reader ownership, partial-open cleanup and retained-resource charges before dependent activation. Add actual controller and partial-failure qualification; helper fixture evidence alone is insufficient. Current driver pre-open retirement wiring is uncompiled; no routing binding enabled. Independent source work may continue. Resolution: review-required; full feature scope retained.

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
