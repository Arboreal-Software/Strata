> Current cross-plan qualification: two-cycle GPU test remains FAILED. The 2GiB cap correctly refused a measured excess; 4GiB allowed cycle1 complete native136/137/138 + logical retirement/budget PASS, but cycle2 native137 had48 KV/scale mismatches from stale future-page destination rows. Full-byte oracle remains strict. Actual supported RAW restore canonicalization, CPU controls and a fresh repeated-cycle GPU run are required; no human/concurrency/full-feature acceptance. Evidence: `.local/strata-implementation/suffix-first/incremental-repair/RESIDENT-REPEATED-RESTORE-FAILURE-20261029.md`. Attribution/prepared-cancel/deferred-fault revisions are applied but UNBUILT/UNQUALIFIED. BothGPU/CPU resources released; liveStrata stays down under owner authorization.

# Required dual-GPU and concurrent-window transactions (P09)

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

- Execution: implementing atomic selection registry; live topology/concurrency qualification incomplete.
- Plan: review-required, revision 4; native/elastic two-phase retirement contract reopened on 2026-10-28; prior scoped evidence retained.
- Owner: Strata implementation session under the owner handoff.
- Dependencies/integration inputs: [P02](../../01-foundations/p02/PLAN.md), [P03](../../01-foundations/p03/PLAN.md), [P05](../../02-exact-parking/p05/PLAN.md), [P06](../../02-exact-parking/p06/PLAN.md), [P08](../p08/PLAN.md)
- Direct dependent plans: [P10](../p10/PLAN.md), [P11](../../04-repair/p11/PLAN.md), [P12](../../04-repair/p12/PLAN.md), [P15](../../05-qualification/p15/PLAN.md)
- Source: Fig issue 30, derived from issues 25–29. Its design pin differs from the deployed engine; reconcile against baseline provenance before editing.

## Local execution plan

1. Reconcile the source proposal below with the pinned local engine and record exact files, existing behavior, missing mechanisms, and unresolved assumptions in FINDINGS.md.
2. Revise this plan with bounded changes and tests using that evidence. Record interfaces and resource budgets consumed by dependent plans. Historical research restrictions describe the source pass; current owner instructions govern execution.
3. Implement only the agreed scope in an isolated checkout, with the feature disabled by default where applicable. Keep the deployed service unchanged during preparation.
4. Run the applicable inert tests below; record commands, source pin, outcomes, and limitations in EVIDENCE.md. Coordinate any GPU/live qualification window separately.
5. Resolve or propagate plan-review triggers before dependent execution; update this plan, the parent, and queue status together. Mark complete only against its acceptance criteria.

## Findings that trigger plan review

Any single/dual-device or concurrent mode requires different cuts, placement, resource peaks, or error handling. Review direct dependent plans listed above and propagate transitively where their contracts change. A changed state schema, shared interface, topology, or resource envelope also requires the final qualification plan to be reviewed. Do not silently assume an unaffected status: record the decision and rationale.

## Preserved detailed proposal and acceptance criteria

**Depends on:** P02/P03/P05/P06/P08.  
**Existing files:** generate.cpp per-stage parking/restore and slot copies; session layer carves; prefill stage callbacks; verifier slot commits; server INFO/mode guards.

Extend RAW Save/Load/Park/Resume/Commit to actual two-device layer bundles and admitted concurrent slots. Common logical cuts, device-local allocations/events, and one engine bundle descriptor are mandatory. Keep head/draft/support allocation placement explicit. Shared host expert arena is counted once physically; device caches remain independent.

Do not remove current disk-mode rejection merely because RAM parking works. Advertise new capabilities only for combinations that pass the replacement contract. Initial cross-topology migration (one→two or two→one) remains explicitly unsupported; same-topology single and dual are both required.

**Failures:** second device/slot fails→entire prepared transaction fails; old selected bundles remain if context healthy.  
**Inert tests:** one and multiple slots on one/two GPUs, leading-stage skew, head/draft placement, batch-group modes, invalid layer ownership, one-device OOM.  
**Acceptance:** required single/dual plus concurrency cells pass, or an exact unimplemented matrix gap remains visible; “single request works” cannot close the task.  
**R9:** topology and concurrency review.

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

## Revision 3

Implement [atomic selection finding](FINDINGS.md) with exact old-state CAS, serialized
admission, quiesced read holds and single-use prepared destinations. No GPU wait under
selection locks. P06/P08 integration prerequisites remain unresolved; this bounded host
component cannot close the dual-GPU/concurrent acceptance matrix.

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

## Checked private initialization checkpoint

Added opt-in session_init_private -> qsa_state_init_private -> retained upload path. The legacy
initializer keeps its successful behavior. Private initialization owns RoPE/page-table upload
sources, checks every copy/launch/drain result, and publishes RoPE registration only after
successful completion. A failed or started upload cannot be retried as a fresh attempt.
Split sessions retain per-QSA upload storage at global ordinals and preserve partial state.
Sanitizer tests pass for each upload failure, final-sync failure, streamed/ring launch errors,
registration ordering, retained sources, split ordinals and first/second-state initialization
failure. Tests inject CUDA calls; no hardware behavior is qualified. CUDA object compilation
and dependent orchestration rerun are in progress (private-init-build.log).

Private initialization currently refuses process-global elastic pools before mutation; this
is an unresolved ownership prerequisite, not a reduction of the full scope. A complete private
allocator still must reserve actual peak memory, retain failure owners, own PLE scratch and
bindings, release device/host/state-array/RoPE resources, and integrate executor admission.
The frozen full build is a prior checkpoint and does not contain this newer initialization path.

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

Private-binding correction validation completed: private upload/session, capture, restore
and selection regressions passed with sanitizer injection; changed CUDA objects compiled.
Verifier drain/commit now also use checked device selection and preserve first error, with
new query/restoration-failure regressions passing. Exact hashes/commands/logs and limitations
are in private-binding-device-evidence.json. Host page size observed 4096 bytes; the pricing
helper conservatively rounds independent pin allocations to 65536 bytes. No GPU qualification.

Next concrete step remains the allocation owner: compose session pricing with PLE/graph/
expert runtime allocations, reserve before allocating the full gang, retain partial allocations
and upload sources on failure, and connect owner lifetime to restore and selected execution.
Do not publish global RoPE bindings during private preparation. Native allocator/activation,
elastic ownership, canonical registry and true completion proofs remain integration gates.

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

## Allocator-backed restore checkpoint

NativeSessionAllocation now seals its owner vector and reservation behind const accessors;
only the allocator populates them. prepare_owned_native_restore accepts that factory handle
and one private binder per stage, validates all binders before transfers, then uses the
existing native validation/materialization/ownership claim path. Combined real allocator +
restore orchestration tests passed with injected CUDA/native copies: correct stage binding,
missing final binder before writes, overlapping restore rejection, retention after caller
drops allocation, and actual final-owner resource release. Exact source hashes and limits
are recorded in owned-restore-evidence.json. CUDA objects and dependent sanitizer tests pass.

Bindings remain an explicit prerequisite, not an implication of successful state transfer.
Real graph/model/expert binding and a PLE reader context must be implemented before execution.
Inspection shows PleReader already supports independent tickets, while PleTable owns one
mutable pending ticket/row buffer. Proposed next implementation: owned per-window read
contexts with retained shared table backing, isolated raw buffers/tickets and close-generation
fencing. Preserve the legacy API; do not copy an active table's mutable pending state or
duplicate the large row cache without resource justification. Review P02/P03/P05/P08/P09/P15
contracts before integrating the new IO ownership; preserve exact accepted/forwarded cuts.

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

PLE exceptional submission semantics: if the shared lower reader throws before returning a
ticket, the context closes/joins the shared reader to protect all possibly submitted raw
destinations. This invalidates EVERY context on that table generation, including foreground
ones. Runtime must treat that as a shared reader failure, fence affected work and recreate
contexts before resuming; it must not silently retry one context as though other windows were
unaffected. If close/join itself throws, context buffers, shared table and the optional
reservation guard remain quarantined until worker restart. Ordinary close/reopen and normal
pending destruction are tested; exceptional allocation/join injection remains unqualified.

## Private PLE binding review — 2026-10-27

Plan: review-required for execution integration; scoped binding implementation ready.
Finding: P02/private-ple-budget-owner (see P02/FINDINGS.md). A private PLE context
must retain its own reservation and immutable weight owner; it must not inherit
mutable native-key or prefill scratch pointers from a selected window. Reserve
16 pending row IO slots and conservative 1 MiB host metadata headroom per context
before constructing it. Shared table cache/slab/weights remain baseline charges,
not multiplied per window. This is a conservative admission policy, not a measured
allocator bound. Actual reader/allocator accounting must be qualified before enabling.

Review resolution: implement and test isolated binding now. Hold serving admission,
canonical model/device ownership validation, native-key scratch binding, graph/expert
bindings and multi-window execution until their prerequisites and qualification pass.
Budget refusal must leave the destination untouched. A retained context must keep
its charge and weight owner through outstanding reads and exceptional quarantine.
Failed restore invalidates all allocation aliases and new private allocation until
worker restart; no recovery may reuse old device pointers. Ticket IDs never recycle
within a reader lifetime: exhaustion fails closed, requiring reader recreation.
Prior test evidence remains scoped to its recorded source; no live execution implied.

### Binding admission correction — 2026-10-27
The binding test exposed that private_work requires one shadow per lease. Added
private_attachment purpose: zero shadow, same-ledger charged private_work or
saved_maintenance parent required, recursively retained until the child drains.
Nested attachments and foreign/foreground parents are refused. Attachment release
must not clear a maintenance parent's exclusive foreground-return claim. This
supersedes the implicit separate-private-work reservation in the binding proposal.
Scoped ledger/binding implementation ready; runtime admission remains review-required.
Revision advanced by this dated amendment. Validate parent retention, no double shadow,
foreign ledger refusal, exhausted IO/host capacity, and maintenance return exclusion.

### Native PLE key scratch amendment — 2026-10-27
Scoped implementation ready: optional private native-key scratch is priced by
native_q8_1_bytes(NG_N_EMBD), allocated on the PLE stage device by the gang factory,
and retained/released with that gang. Binding accepts supported native key formats
42/18/23/8 only with this destination scratch; source mutable scratch and precomputed
projections remain forbidden. Validate optional flag topology, pre-allocation budget
refusal, disjoint buffer ownership and binding rejection without destination scratch.
Actual model execution and format/device identity qualification remain review-required.

### Private PLE weight extraction review — 2026-10-27
Scoped plan ready: validate the six PLE WeightRefs before binding: actual kernel
geometry, BF16 value/key, F32 norms, F16 convolution, canonical S2 planes or supported
native key. Publish only after whole-set validation; never copy native_q8_1 or
prefill mutable pointers from WeightTable. Shared model owner/device identity remains
a caller prerequisite. Default generate binding is unchanged. Tests must reject
missing tensors, wrong dimensions/storage and truncated planes without partial output.
Execution qualification remains review-required; metadata checks are not pointer
provenance, device-placement or numeric model qualification.

### Failed PLE ticket retirement — 2026-10-27
Finding: collect waits for pending==0 but returns on sticky reader error before
erasing the ticket; issue also accepts work after that error. Scoped repair ready:
consume every drained ticket exactly once on success or ordinary IO failure, and
reject new issue on closed/failed readers before allocating a ticket or modifying
output. Do not erase undrained work. Test actual truncated-file failure in threaded
and synchronous readers, duplicate collection refusal, unchanged output on failed
submission, and close/reopen recovery. P02/P03/P05/P08/P09/P15 consumers retain
shared-reader failure fencing; admission/serving integration remains review-required.

### Checked native model release — 2026-10-27
Scoped implementation ready: add explicit NativeDense checked retirement for future
private-model owners. Caller must prove all graph/stream readers drained and select
the owning device. Record successful frees, stop on first error, never retry an
uncertain CUDA free in destructor. Failed retirement is sticky and requires owner
quarantine with baseline accounting until worker restart; it is not completion.
Legacy callers retain existing destructor behavior. Tests inject partial free failure
and verify later allocations are untouched and destructor does not retry. Runtime
owner quarantine, baseline accounting and device-proof integration remain held.

### Shared model-stage owner review — 2026-10-27
Scoped implementation ready: internal adoption API transfers an exclusively owned
canonical arena, WeightTable, NativeDense and already-charged foreground model
reservation into one shared owner. Pre-adoption validation/allocation failure leaves
caller resources untouched. All readers/graphs must retain this owner. Last release
selects owning device, drains device, invokes checked native release, then frees
canonical arena. Any failure retains object and reservation in allocation-free
quarantine and fences further adoption until worker restart. Caller computes complete
model charge and proves pointer ownership/model identity; API checks a lower bound,
not a trusted pricing oracle. Existing stack driver objects cannot be adopted by
no-op aliases; actual driver construction migration remains required.

### PLE metadata review and owned spans — 2026-10-27
ChatSwarm independently decoded exact source hashdc43d02e60fa (receipt in
.local/strata-implementation/reviews/ple-weight-source-result.json). Confirmed
contradictory plain/S2 metadata and unaligned pointers accepted; fixed with negative
CPU regressions (71028 passed). No reachable fixed-geometry arithmetic overflow or
partial failure publication found. Concurrent publication remains caller-serialized.

For allocation extent, metadata alone is insufficient. Added NativeDense original
weight spans recorded from actual load allocations; no scratch/packed-helper spans,
no proof before complete load or after retirement. OwnedModelStage checks canonical
subranges against arena and native spans against retained NativeDense. Owned PLE
binding computes native format bytes using the actual kernel helper and requires
all six tensor spans covered before context publication. Device/model identity and
immutable lifetime still require canonical runtime ownership; direct metadata API
remains explicitly an intermediate descriptor, not execution-ready scratch.

Scoped owner-span tests64159 and initial native-span tests14378 passed. Final
complete-load fence revalidation and PLE adapter regression running. NativeDense
layout changed again; full consistent engine build required, no mixed ABI execution.
Plan ready for bounded fixes; full runtime/driver/qualification remains review-required.

### Native loader publication ownership — 2026-10-27
Source-confirmed: final weights_.reserve(pending.size()) excludes existing HC
allocations and packed copies; push_back(item.release()) releases RAII ownership
before an insertion that can allocate/throw. Packed registration also preceded
recording cleanup ownership. Scoped repair ready: pre-reserve all final vector
capacities with overflow guards; transfer each pointer using get/push/release;
record registration cleanup identity before registration can throw. Keep spans
invalid until entire load completes. Existing failed CUDA cleanup semantics still
require checked loader work; this host-publication repair does not qualify them.
P02/P03/P05/P08/P09/P15 runtime consumers remain review-required for loader use.

### HC ownership and native allocation accounting — 2026-10-27
Source-confirmed: GGUF HC upload allocated a device pointer before recording it,
so upload failure could lose ownership. HC raw push operations could allocate after
cudaMalloc. allocated_bytes omitted packed copies, GGUF HC rows, HC scratch and
native Q8 activation scratch. Scoped repair ready: reserve HC ownership slots before
CUDA, record allocations before upload, and include every native-owned allocation
in the retained allocation estimate. Use checked rounding/addition. weight_bytes_for
remains a weight-only estimate; full reservation must include runtime overhead and
shared event metadata. Failed CUDA-free quarantine/loader transactions remain open.
Affected P02/P03/P05/P08/P09/P15; actual full model pricing/qualification still held.

### Failed-load model adoption fence — 2026-10-27
Model adoption must distinguish untouched NativeDense (canonical-only model), fully
loaded native data, failed/partial load, and retired data. Add one-shot load-attempt
state: failed loads cannot be retried/reused, successful duplicate load refusal does
not invalidate existing spans, and adoption rejects partial/retired state before
resource transfer. Actual loader fault suite must assert this across all cases.
Scoped implementation ready; complete canonical loader and model identity validation
remain caller requirements. Runtime migration/qualification remains review-required.

### Canonical model span admission — 2026-10-27
Scoped implementation ready: model adoption requires nonempty weight metadata and
each nonnull canonical data span contained in the retained arena. Reject zero-byte
resident data, null data marked resident, or separate canonical allocations absent
from this owner's resource inventory before any transfer/device work. Skipped
nonresident canonical rows remain valid for separately owned native paths. This
does not prove every native/head/embedding resource or semantic model identity;
separate WeightTable HC requantization allocations need explicit owner integration.
Tests cover exact arena end, one-byte overflow, outside pointer, empty table and
nonresident skipped rows. P02/P03/P05/P08/P09/P15 full integration stays open.

### Completed canonical load admission — 2026-10-27

Contract amendment: partial or stale WeightTable metadata is insufficient for model
adoption. The latest canonical load must finish its device synchronization successfully.
Every new load attempt revokes readiness, including failures before parsing; a successful
retry can restore it. This preserves existing loader retry behavior. NativeDense remains
subject to its separate one-shot load contract. WeightTable layout changed: rebuild every
consumer consistently; do not mix old objects with the new header.

Scoped plan review: ready for this admission check and real-loader failure tests.
ASan/UBSan test90599 passed actual index/file loading, injected upload/final-sync failure,
stale success followed by early failure, successful retry and existing owner teardown
cases. Evidence: .local/strata-implementation/canonical-load-adoption-evidence.json.
Full CUDA build of frozen-runtime-canonical-load is pending. GPU/serving integration
remains held on complete resource ownership, executor integration and qualification.

### HC auxiliary allocation ownership — scoped amendment 1, 2026-10-27

Review trigger: optional canonical HC requantization allocates a separate device
block that the prior model owner neither retained nor charged. Existing load
failures lost that address and could expose partially converted rows.

Scoped plan status: reviewed and ready for implementation/validation. WeightTable
now records that allocation and device immediately after allocation, publishes HC
rows only after all transfers and final synchronization, and refuses adoption of
a failed load. Successful repeated HC loading is idempotent; failed attempts and
canonical reload after HC ownership are refused. The table is noncopyable.
Checked release requires the owner device and a drained caller; failed frees are
sticky. Model retirement releases HC before the canonical arena, and quarantines
all remaining resources plus the full charge when cleanup is uncertain. Standalone
table destruction attempts owning-device drain/cleanup; uncertain teardown leaks
conservatively, so failed startup remains fatal rather than reusable.

Admission must validate auxiliary device, code and scale spans, and include the
entire padded HC block in the allocation lower bound. Native head/embed, expert
resources, graph lifetimes and actual driver/executor migration remain incomplete.
This change does not claim full allocation pricing or backend/hardware qualification.
WeightTable layout and destruction changed; rebuild all consumers together.

Required evidence: actual canonical+HC loader with synthetic on-disk BF16 rows;
each copy failure, allocation failure, final drain failure, truncated second row,
no partial row publication, unchanged quantized output, idempotent success, wrong
device/foreign scale/undercharge admission refusals, last-owner retirement and
sticky failed-free quarantine. Full CUDA compile/link and existing loader regression
required; HIP/SYCL/Metal and actual hardware remain unqualified.

### Shared model gang lifetime — scoped review, 2026-10-27

Model stages now optionally retain native heads and charge their full reported allocation.
A gang retains all stage owners plus one embedding owner and its separate device/host/pinned
reservation. Retirement drains every referencing stage device before releasing a mapped
embedding; a failed drain or embedding release quarantines the entire retained gang and
charge. Single-stage device drain is not assumed to retire cross-device mapped readers.
All sessions, graphs and asynchronous producers must retain the gang; the owner API alone
does not prove driver holders or producer joins. No global embedding pointer is published
by adoption. Driver binding and unpublication remain explicit integration obligations.

Scoped review outcome: ready for owner integration/tests; P02/P03 resource contracts amended,
P05/P08/P09/P15 inherit gang lifetime and all-device drain. Other transitive consumers inherit
the same shared owner without changing their edit/disk/priority/codec/reuse/repair algorithms.
Model-gang test70972 passes ASan/UBSan using actual owner/canonical loader and mocked native
heads/embedding/device calls (model-gang-evidence.json). Separate production loader tests are
agent-owned and pending aggregate integration. Full CUDA rebuild waits for shared-header wave.
Actual serving, graph/expert inventory and hardware qualification remain incomplete.

Managed priority/dependency scoped review (2026-10-27): see work-items/03-scheduling/p07/MANAGED-LIFECYCLE-REVIEW.md. Component revision4 is ready and sanitizer-tested; runtime consumers must retain full managed identities, refresh at ordered control/timer cuts, account transactional memory, and preserve hard eligibility. Integration/qualification remains incomplete.

Parallel ownership wave review: work-items/01-foundations/p02/PARALLEL-OWNERSHIP-REVIEW.md records accepted checked head/embed/gang interfaces, packed-failure fixes, backend limits and actual driver obligations. Aggregate frozen build27898 pending; no full runtime acceptance implied.

Driver construction checkpoint: see work-items/03-scheduling/p08/DRIVER-MODEL-OWNERSHIP-REVIEW.md. Opt-in actual model ownership is under build/test; this does not resolve executor, allocator inventory, graph/producer seal or hardware gates. Full scope unchanged.

## Serialized-control and teardown checkpoint (2026-10-27)

Plan: review-required for remaining dependent runtime integration. The scoped consumer
contract and tests are reviewed in [checkpoint review](../p08/SERIALIZED-CONTROL-REVIEW.md).
Prior evidence retains its original source scope; no feature acceptance is implied.

## Session runtime interface review (2026-10-27)

Scoped interface decision and remaining held integration are recorded in [session runtime review](../p08/SESSION-RUNTIME-INTEGRATION.md). Full acceptance remains open.

## Disk-v2 implementation review (2026-10-27)

Scoped contract ready; full integration remains open. See [disk-v2 contract](../../02-exact-parking/p06/DISK-V2-IMPLEMENTATION.md).

### Mapped scratch ownership checkpoint (2026-10-27)

The opt-in driver now retains token scratch and mapped doorbell allocations through actual
OwnedWindowBuffer handles and the private-session reservation. Mapped buffers charge host
and pinned bytes without charging their device alias twice; final release drains all declared
consumer devices. Uncertain drain/free/device restore quarantines ownership and charges.
The caller must still close all producers before release; this does not establish selected
execution or a coherent cut. Default allocation paths remain separate. Eleven isolated
actual-source mocked-CUDA failure scenarios passed ASan/UBSan (mapped-evidence.json);
production CUDA build54728 is pending. Prior device-only build66804 passed, not evidence
for mapped changes. GPU execution, complete runtime ownership and exact physical metadata
accounting remain open. Apply this ownership contract to capture/restore, budgeting, dual
GPU publication and qualification consumers; no parent acceptance is closed.

Scoped plan amendment owned-mapped-runtime-20261027: reviewed/replanned per
`03-scheduling/p08/MAPPED-RUNTIME-REVIEW.md` under work-items. Ready for bounded
implementation; full runtime/hardware acceptance remains open. Earlier decisions and
evidence retained; no serialization or scheduling scope change.

Epoch remediation checkpoint: scoped async ownership/control/restart contract reviewed
under 03-scheduling/p08/SERIALIZED-CONTROL-REVIEW.md; final123-test evidence recorded.
Bounded implementation ready; previous ChatSwarm NOT CLEARED until rereview disposition.
No scheduler, native orphan or hardware/deployment acceptance inferred.

Current scoped amendment owned-consumer-retirement-20261027: reviewed/replanned under
03-scheduling/p08/CHECKED-CONSUMER-RETIREMENT.md; implementation ready, verification
and full runtime replacement acceptance pending. No successful cleanup after uncertainty.

### hipcomp-dax-target-20261027 — scoped review required

Owner adds hipCOMP alongside nvCOMP for RX 6800 XT. Dax currently has GTX 1080;
SSH qualification tests are authorized once the RX 6800 XT is installed, coordinated
with parent/active workloads. See 04-repair/p11/PLAN.md for authoritative target,
codec compatibility and test requirements. Review this item's AMD codec capability,
resource/format/lifetime assumptions and qualification consumers before codec-dependent
integration. Existing CUDA/RAW work is unaffected and continues; no hardware pass claimed.

Scoped actual-runtime-owners-20261027 amendment reviewed/replanned: see
03-scheduling/p08/DRIVER-RUNTIME-OWNERSHIP.md. Actual heap callback/stage ownership
compiled; complete private factory, budget pricing and selected execution remain open.

Disk-owner-retirement-20261027 scoped contract reviewed/replanned per P06/FINDINGS.md:
arbitrary output/source owners retire outside store lock/IO lease; unknown durability
retains charges and requires reconciliation. Tested component ready for integration;
independent review and full feature/runtime acceptance remain open.

Cross-window-routing-20261027: scoped interface reviewed/replanned under
03-scheduling/p08/WINDOW-MANAGER-CONTRACT.md; bounded implementation ready.
Window replace identity/cut invariants unchanged; driver generation authority and actual
lane eligibility/resource proof required. Full executor/hardware/eviction/import gates open.

## Manager and checked-runtime integration amendment (2026-10-27)

Downstream review disposition: incorporate exact-cut cancellation/quiescence checked inside
the execution mutex, monotonic engine window generations, and retained exact per-lane work/read
owners from the approved window-manager contract. CPU sanitizer evidence23277/82517 proves
these component fences only; actual scheduler/admission and external authority remain open.

The actual driver now retains remote experts, remote optimization, peer and lookahead objects,
and bounded streams/events in its heap runtime. Drain declared primary/stage/remote/peer
consumer devices after producers close, before member teardown; retain handles until consumers
finish destruction. Strict File/Arena/cache retirement has CUDA/SYCL compile and injected-failure
evidence. Aggregate source is frozen in frozen-runtime-manager-experts; CUDA88483 passed link, PLE and eight selected CPU tests; HIP remains pending.
Foresight cleanup, primary consumers, full producer closure, opaque driver allocation pricing,
process-global VMM multi-device limits, reusable runtime binding and hardware qualification
remain explicit acceptance gates. No parked/executable/runtime completion is inferred.

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

Primary/prompt/graph owner checkpoint reviewed: incorporate the actual ownership and
remaining callback/accounting limits in P08 DRIVER-RUNTIME-OWNERSHIP.md (CUDA23101/12627).
No selected runtime, numerical parity or full feature acceptance inferred.

Review amendment: consume P08 DRIVER-RUNTIME-OWNERSHIP.md prelaunch/Foresight lifetime
contract. Mapped callback buffers require checked drain before free; retaining swap space
past readers does not replace producer joins or full resource accounting. Source integration
and hardware evidence remain separate gates.

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

### Downstream implementation review: hit-path ownership

The owned driver now retains six actual expert-cache hit buffers under runtime scratch
reservations, and names its primary/adaptation handles. Review partial-construction
retention and per-device budget charges when integrating concurrent/private runtimes.
Evidence pending: frozen-runtime-hit-ownership CUDA14661. Existing CPU/mock evidence
does not establish hardware numerical parity; the exclusive DeepSeek window grants
this thread no GPU permission. Full accounting and selected executor remain open.

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

### PLE plain-group compatibility review (2026-10-27)

Plan review: review-required for affected runtime numerical acceptance. See P00
FINDINGS.md “PLE neutral plain grouping”. Baseline passed; owned startup refusal
reopens compatibility. Keep numerical qualification pending until neutral0/1-only
validation fix, refusal regressions, source-linked backend builds and parity rerun.
Independent implementation remains authorized. Main owns final review resolution.

### Async compression downstream contract review (2026-10-27)

P11 revision4 authorizes immutable owned-input jobs with pre-enqueue peak reservations,
verified private output and retained quarantine on uncertain drain/free. RAW fallback
requires no enqueue or confirmed drain. No vendor framing interoperability assumed.
Review-required: this item's encoded-state serving/persistence/multi-stage acceptance;
use P11 revision4's explicit integration gates before enabling. Independent RAW work
continues. Main owns integration resolution; CPU/SDK compilation is not hardware proof.

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

### Owned prompt callback checkpoint (2026-10-27)

Actual owned no-cache serving progress callback now resolves a weak runtime and retains
it during invocation; it no longer captures construction-scope state. Stage no-cache
checkpoint callbacks are absent. Two-runtime/expiration/invalid extent/overflow CPU
seam60831 passed; CUDA62473 PLE/fifteen aggregate tests passed. Cache-enabled, route and
cancellation callbacks and complete repeatable factory remain separate unresolved work.
No selected-executor, concurrent/GPU qualification implied; next hardware gate requires
a new root grant. Snapshot frozen-runtime-owned-progress records exact source.

### Disk operation-binding review resolution (2026-10-27)

ChatSwarm1793106180.265029 reassembled18/18 parts and verified19/19 file hashes,
reviewing exact disk.cpp76a0a31... with GPT-6 Extra High (requested5.6Pro not
substantiated). No unresolved demonstrated changed-path defect: own-operation intent
binding prevents same-content receipt promotion; prepared success results preserve caller
output on save/catalog OOM; old output retires outside save lock. CPU red36418/green54485
and CUDA95211/HIP11127 are retained. This resolves that bounded source-repair gate only.
Actual selected-runtime disk execution, physical crash recovery, global RSS, non-POSIX
platforms and concurrency remain pending; parent acceptance stays incomplete.

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

### Owned adaptation completion review — revision1

Review-required for dynamic expert placement/retirement: P02 FINDINGS `owned-adaptation-completion-20261027`. Main checked owned primary/stage event completion before unpin/publication; no compile or hardware evidence yet. Validate success/NotReady/error and device scope, then close peer completion/upload publication and effective-policy generation gaps. Legacy default unchanged. Existing managed enablement gates remain; independent source work may proceed.

### Placement and actual execution decision review — revision1

Review-required for complete managed identity/cuts/replay. Origin P00 FINDINGS `placement-versus-execution-decisions-20261027`; main owns integration. Startup placement and host-residency hashes do not cover timing-dependent CPU/GPU/foresight/prefill choices. Add bounded owner transaction state and actual consumer decisions, bind exact replay/capture to appropriate revision and dependencies, then test delayed copies/partial failures/compulsory fills/prompt loans/all tiers. Adaptive features remain required. Existing scoped evidence remains valid only for its original components; full publication remains held while source work proceeds.

### Admission reactor lifetime review — revision1, 2026-10-27

Review-required for dependent admission enablement. Origin P08 FINDINGS `admission-reactor-identity-and-charge-lifetime-20261027`; main owns review. Require one noncopyable/nonmovable reactor identity and retain record metadata reservations through covered member destruction, including unresolved admission recovery. Prepare type-trait and retirement/deallocation checks; execute in a newly granted CPU window before claiming the repair validated. Independent source work continues. This correction does not qualify serving integration or change prior evidence.

### Idle activity timer review — revision1, 2026-10-27

Review-required for live activity expiry. Origin P07 FINDINGS `idle-activity-registry-timer-20261027`. Main owns integration. Use the whole priority registry deadline, including actors with no graph operation, for admission/reactor wakeups. Advance monotonic time then re-read after a conservative early wake; preserve absent deadline versus UINT64_MAX. Add no-request/no-operation expiry to two-user qualification and retain base priority. Registry/admission source wiring and tests are prepared; actual serving timed wait and CPU validation remain open. Independent source work continues.

### Verified input envelope review — revision1, 2026-10-27

Review-required for verified serving/control enablement. Origin P08 FINDINGS `verified-serving-input-envelope-20261027`; main owns review. Bound queued and incomplete input, retain admission through detached reader/moved consumer lifetime, and treat overflow/read failure/truncated EOF as terminal rather than empty queue or accepted partial request. Future units stop while submitted work and prompt readers drain/join. Validate first/next-unit races, STOP reset, direct control scanning, persistent allocation failure, exact framing and legacy compatibility. Source repair and fixtures are preparing; compiler/hardware gates remain open. Independent source work continues.

### Priority response preparation review — revision1, 2026-10-27

Review-required for priority transport/admission enablement. Origin P07 FINDINGS `priority-response-before-commit-20261027`; main owns review. Complete allocating success receipts before committing the requested mutation, preserve accepted/applied and full incarnation identities, and retain real admitted work on acknowledgement preparation failure. Account overlapping prepared-output/current-record/insertion metadata. Timer expiry is an independent allowed effect; do not claim whole-cut rollback or durable/network idempotency. Add persistent allocation failure sweeps before public transport qualification. Source work continues; tests/builds unrun.

### Initial expert tier join review — revision1, 2026-10-27

Review-required for managed identity/publication. Origin P10 FINDINGS `initial-expert-tier-join-20261027`; main owns integration. Require actual completion of every declared initial tier at the post-helper-load cut, then invalidate before untracked dynamic setup/execution. Future tracker must seed at that cut and retain exact owners, not reuse a stale initial receipt. Test missing helpers, partial producer failure and successful join/invalidation; retain absent-profile/dynamic/execution gates. Source-only, no backend/numerical qualification.

### Explicit disabled profile review — revision1, 2026-10-27

Review-required for complete profile/managed publication. Origin P00 FINDINGS `explicit-disabled-expert-profile-20261027`; main owns review. Retain the genuine manifest-disabled claim and actual model geometry without manufacturing file header/path/ranking evidence. Price metadata before construction, revoke on claim/read-generation change, and join real initial cache capacity even without a ranking file. Test forged/public-only flags, lease conflicts, absent geometry, insufficient quota, output preservation and actual read invalidation; rebuild all affected inventory consumers before qualification. Source-only; existing dynamic/executor gates stay open.

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

Verified stdin CPU evidence: P08 STDIN-CPU-CHECKPOINT-20261027.md records exact-extract sanitizer fixture PASS, manifestbfee395e...,4.69seconds, explicit release. Main-loop/piped_decode and actual pipeline/device/concurrency gates remain; evidence-only addition, no acceptance change.

Prepaid elastic integration review remains required: P03 FINDINGS elastic-kv-prepaid-attachments-20261027 now records host metadata/handle allocation ordering and error-drain gaps in the initial owner. Optional build wiring and PLE host preallocation are source-only. Main owns dependent integration; validate revised VMM/owner source and backend compatibility before enablement. Existing restore, growth/donation, capacity-frontier and concurrency acceptance stays open.


## Native/elastic retirement revision — 2026-10-28

Plan: **review-required**, retirement subplan revision 1. Origin: [native-elastic-two-phase-retirement-20261028](../../01-foundations/p02/FINDINGS.md#native-elastic-two-phase-retirement-20261028). [Concrete hierarchical implementation and file/API dependencies](../../01-foundations/p02/elastic-retirement/PLAN.md). Review owner: main Strata integration; this source-only update grants no build or GPU window.

Topology integration requires one gang-wide retirement transaction across every actual consumer device, including peer and donated physical chunks. No first-stage free before later-stage drain; no unqualified single-domain shortcut for required dual-device or two-user operation.

Current all-stage allocator predrain and early elastic refusal are bounded containment only. Full elastic growth, shrink, cache donation, restore/capture, capacity barriers and concurrent-window scope remain required. No complete retirement, driver UAF reproduction or new passing test result is claimed. Dependent enablement stays held; independent source implementation/fixture preparation may continue after file leases and interface review. Main retains WORK-QUEUE and remaining transitive review propagation ownership.

Previous top-level status retained for history: `- Plan: ready, revision 3 for host atomic selection only; disk/scheduler/runtime integration held.`. It does not override this reopened retirement gate. Resolution: replanned, not yet reviewed-ready; exact API/closure proof, resource accounting, public test linkage, failure matrix and touched-backend qualification remain prerequisites.

### PLE terminal owner-retention review — 2026-10-28

Review-required for this lifetime boundary: P02 finding `ple-terminal-owner-retention-20261028`. A successful context IO close must not release immutable weight/table owners or attachment charges before gang completion. Source reset removal7572ec7c awaits combined allocator/context expected-red/fixed regression; earlier separate sanitizer evidence does not cover it. Preserve whole Storage ownership on later-close or device-drain failure. Main owns disposition; existing downstream enablement holds remain.

## Submitted pipeline teardown dependency, 2026-10-28

Review required: background worker source/mock closure now passes bounded fixtures, but actual submitted pipeline/controller closure is a separate source-only guard under review. Do not infer safe capture, release, re-admission, publication or concurrent execution from background joins alone. Retain callbacks/owners while servicing already submitted groups, preserve uncertain completion, and require actual controller/backend qualification. Combined PLE owner-retention regression passed; public registrations and complete runtime remain unqualified. See P02 elastic-retirement plan and persistent main todo.

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


Review amendment `in-memory-incremental-repair-next-20261028`: P13 revision 3 is now the immediate next milestone after suffix reuse. New dependent repair integration remains review-required: retain a separate complete exact checkpoint, advance it only after bounded real all-layer units, preserve ancestry/age on approximate reattachment, fence edits/appends/selection and final publication, price full-copy/switch overlap, and qualify foreground interference/catch-up. Exact-frontier metadata in a hybrid endpoint is not a complete exact checkpoint. Main owns source/interface disposition; independent source preparation is allowed. Existing qualified suffix candidate, prior evidence and full acceptance remain unchanged. No new resource grant or serving change. See P13 FINDINGS.md for impact mapping and held work.


## Owner dual RTX3090 acceptance update — 2026-10-29

Single-GPU complete-answer suffix reuse passed on one synthetic edit; see suffix-first/gpu-quality-followup/QUALIFICATION-20261029.md. This does NOT satisfy the explicitly required two-3090 setup: qualify correct outputs/reuse across actual model split; repeated prefix repairs with real foreground between; coordinated capture/restore/publication across both devices; stage failure handling, cleanup and per-GPU memory bounds. Keep dual-device gate open until actual exact pinned two-device evidence. Source-only split support or independent single-device passes are insufficient.

## Owner gate: eventual exactness before human testing (2026-10-29)

Require two actual partial repairs with foreground generation between, private catchup to the CURRENT endpoint, full native-state comparison against a clean reset/recomputation of identical accepted history, and continuation checks. Also qualify interrupted/stale publication, cancellation, allocation/device failures, correct resource lifetimes, repeated-cycle leaks and independent teardown. Initial single/dual reuse and component passes do not qualify this sequence.

Impact review: current dedicated serial controller implementation may continue, but human testing and full P13/P09/P15 acceptance remain held until all relevant gates pass. Full-state comparator and clean-replay instrumentation are being implemented; runtime failures remain explicit outstanding cells. This refines qualification evidence without dropping generic manager, server or concurrency requirements.

## Resident serial qualification review — 2026-10-29

Reviewed scoped outcome: `04-repair/p13/REVIEW-resident-serial-20261029.md` (work-items-relative). Dedicated RAW serial qualification is ready subject to exact final candidate/build/parser/resource checks. Full feature acceptance and the listed generic/concurrent/fault/cleanup gates remain incomplete; no inherited GPU qualification or codec enablement.

Measured resident review update (2026-10-29): dual3090 normal endpoint137/continuation138 and drained cancellation PASS/released; see P13 REVIEW-resident-serial-20261029.md. Pre-forward136clean native proof subsequently PASS/released with pinned5e6f/source51af; full138tensors compared,0mismatches. Reset/capture residual proof remains separate. Actualfault/leak/genericconcurrent/human gates remainOPEN. No default or fullfeature enablement follows.
