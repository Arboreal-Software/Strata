# Strata feature backlog: progress, missing integration, and qualification

Status snapshot: 27 October 2026. This report covers the requested engine improvements, including conversation switching, context edits, reuse, repair, scheduling, and compression. It describes development progress; it is not a release announcement.

## Overall state

The goal is to make Strata spend less time rebuilding conversations: preserve complete conversation state, switch between sessions, restore parked state, reuse work after edits, and repair approximations while keeping interactive requests responsive.

Substantial foundations and component tests exist. The main gap is connecting those components into a complete serving path and proving that path works under real memory limits, cancellation, multiple conversations, and both GPUs. None of the major feature workstreams is recorded as fully complete against its acceptance criteria.

The closest practical milestone is an integrated private restore-and-execute path: prepare a conversation separately, validate its complete state and memory ownership, then safely make it active. Approximate suffix reuse and background repair depend on that foundation and remain later milestones. There is no defensible overall percentage or delivery estimate yet.

## What is built, tested, and installed?

- **Built:** a frozen development aggregate completed linking, its PLE self-test, and 17 selected CPU tests. Earlier candidates also passed CUDA and HIP build checks.
- **GPU-tested:** an earlier, narrower candidate passed four serial numerical-parity test arms. This does not qualify concurrency or later source changes.
- **Component-tested:** separate CPU fixtures cover parts of loading, allocation failure, expert layout, policy handling, and input cancellation. Some use mocked GPU or consumer interfaces.
- **Still source-only:** later changes include portions of owned elastic KV memory, execution observations, scheduling, and driver integration. Earlier test results do not automatically cover them.
- **Live deployment:** the coordinator restored the preserved Strata and voice serving configuration. The evidence reviewed for this report does not establish deployment of these feature improvements into that live service.

Passing a mock fixture, building a candidate, passing a numerical test, and deploying a feature are separate milestones throughout this report.

## Feature-by-feature progress

### 1. Defined compatibility and reproducible correctness settings

**Requested:** record exactly which model, tokenizer, prompt template, cache formats, numerical settings, and device layouts can share or restore state safely.

**Progress:** the deployed source baseline and isolated build environment are identified. Compatibility/profile machinery and loader identity work exist, with component validation. Plans account for the actual 256K INT8 serving configuration rather than relying on the original illustrative sizing assumptions.

**Remaining:** complete the runtime profile and compatibility matrix, validate current source across affected backends, and qualify supported speculation and topology combinations. A successful earlier build is not evidence for every configuration. *(P00)*

### 2. Complete conversation-state ownership

**Requested:** preserve everything needed for a truthful continuation—not just attention KV, but recurrent state, positions, pending accepted input, and the relevant execution identity.

**Progress:** state descriptors, immutable ownership, loader provenance, and private model/session ownership have substantial implementations and targeted tests. Native and canonical loading paths have received failure and lifetime checks.

**Remaining:** finish complete runtime identity coverage and bind it to actual capture, restore, and execution. Newly added ownership paths still need integration and qualification; incomplete descriptors must not certify an exact continuation. *(P01)*

### 3. Safe pause, cancellation, and capture points

**Requested:** stop new work and drain already-started CPU, GPU, and transfer work before saving state or reusing memory. Every stage must represent the same accepted conversation point.

**Progress:** completion ownership, cancellation fencing, and drain handling have been implemented in several paths. The extracted stdin reader and mocked consumer fixture passed sanitizers; policy fixtures cover additional completion cases.

**Remaining:** validate the actual serving loop, pipelined decode, all-stage capture boundary, and device-error handling. A drained stream alone does not prove that every stage describes the same state. *(P02)*

### 4. Memory admission and owned elastic KV capacity

**Requested:** reserve GPU memory per device, ordinary and pinned RAM, and disk capacity before starting work, including temporary copies and the capacity needed to return to foreground work. Support elastic KV capacity with real ownership.

**Progress:** the budget/lease foundation is implemented and tested in bounded cases. Checked physical-memory ownership, prepaid allocation batches, and an initial session-owned elastic KV path exist in source. Follow-up source changes now separate VMM bookkeeping preparation from device activation and make the owned pool attempt draining on device-effect failures. Optional build wiring and explicit SYCL refusal compatibility are also in source; these changes and their prepared failure fixtures have not been compiled or run. SYCL refusal preserves a compatibility boundary, not implemented elastic support.

**Remaining:** current-source builds and failure tests; driver integration; multiple-device support; growth, shrink, transfer, and donation; checks preventing consumers from using uncommitted capacity; complete peak-memory accounting. The initial elastic path is not a qualified growth feature. *(P03; shared prerequisite)*

### 5. Exact context edits and replay

**Requested:** when an earlier prompt section changes, identify the genuinely unchanged exact prefix, restore a suitable checkpoint, and recompute the affected remainder. This is the correctness reference and fallback for approximate reuse.

**Progress:** an immutable exact-edit plan and checkpoint-selection rules are documented/implemented at the planning layer. They distinguish token equality from state that was actually computed under an exact ancestor.

**Remaining:** execute that plan through the real integrated runtime and demonstrate exact results for shorter, longer, ambiguous, and sensitive edits. Preserve accepted output and external effects without replaying them. *(P04)*

### 6. Exact conversation parking and restoration

**Requested:** save complete uncompressed state, prepare a restore in private memory, validate it, and atomically switch to it while retaining old state until its users finish.

**Progress:** native capture, private restore preparation, retained reservations, and failure ownership are implemented in part. Supporting allocation and ownership paths have component evidence.

**Remaining:** complete activation and repeatable execution through the serving driver; bind graphs and runtime state correctly; prove full-state round trips and old-state preservation on failures. This is a central blocker for switching and reuse. *(P05)*

### 7. Disk parking and crash recovery

**Requested:** persist parked conversations with versioned formats, bounded parsing, compatibility checks, and durable operation records so retries can reconcile interrupted or uncertain saves.

**Progress:** disk and operation-receipt components exist. Source review recorded no unresolved demonstrated defect in the reviewed disk change set.

**Remaining:** complete serving integration and qualify physical I/O, crash recovery, repeated requests, malformed data, full disks, and reader lifetimes. Source review is not end-to-end crash qualification. *(P06)*

### 8. Editable base priorities, activity boosts, and dependencies

**Requested:** let users edit a thread's persistent base priority; apply active-user boosts separately and temporarily; allow a blocked task to lend urgency to its prerequisites.

**Progress:** engine priority and dependency machinery exists. Further response-before-commit and idle-deadline repairs are in source, with additional prepared tests. The separation of base priority and activity is recorded in the plans.

**Remaining:** finish serving/control integration, authentication and persistence, activity expiry/disconnect behavior, accepted-versus-applied revisions, and two-user qualification. The parent/UI workstream owns activity detection and the thread editor; this report does not claim that UI is shipped. *(P07)*

### 9. Responsive scheduling and bounded work

**Requested:** check foreground demand between bounded work units, including setup and transfer costs. Run foreground work and necessary prerequisites before live repair; keep saved-state maintenance idle-only.

**Progress:** admission, token accounting, ownership, and bounded-work components are implemented in part. Input and cancellation fixtures passed within their documented scope.

**Remaining:** complete the actual selected-work executor and repeatable private runtime, connect the serving reactor, and measure interruption/return latency under load. Component policy checks do not demonstrate interactive responsiveness. *(P08)*

### 10. Two-GPU transactions and concurrent conversations

**Requested:** switch state coherently across both GPUs and serve multiple conversation windows without mixed publication, aliasing, or unaccounted memory peaks.

**Progress:** atomic-selection and lifetime contracts, private allocation, and earlier numerical work provide foundations.

**Remaining:** real all-device publication, concurrent execution, cancellation/failure races, and two-user resource/latency testing. The four serial numerical arms are explicitly not concurrency qualification. Single-device restrictions in an initial source slice are temporary gates, not a scope reduction. *(P09)*

### 11. Per-session expert-use accounting

**Requested:** report which MoE experts each conversation actually uses, distinguishing prefill, generation, repair, and accepted/discarded speculative work, with bounded overhead.

**Progress:** expert layout and File/Arena producer fixtures passed; policy and placement fixtures also passed in their bounded source/mock configurations. Runtime-bound execution observations and linear bookkeeping are implemented in source.

**Remaining:** install observations in the complete driver path, cover all relevant execution routes, validate dynamic placement and attribution, and measure overhead. Prepared observation tests and source review do not establish full runtime coverage. *(P10)*

### 12. State compression with nvCOMP and hipCOMP

**Requested:** implement and evaluate optional lossless state compression on NVIDIA and AMD, with an uncompressed fallback and honest accounting for temporary memory and transfer costs.

**Progress:** codec adapters/planning have source and build/link evidence, including nvCOMP and hipCOMP checks. Review findings prompted repairs; asynchronous rereview remains outstanding in the reviewed record.

**Remaining:** hardware round trips, failure behavior, memory peaks, actual switching benefit, and current-source integration. RX 6800 XT qualification is required once that card is installed on Dax and testing is coordinated. Its currently recorded GTX 1080 is not evidence for the AMD target. *(P11)*

### 13. Approximate suffix reuse after edits

**Requested:** reuse cached work for an unchanged later span after an earlier context edit, while explicitly acknowledging that the reused state was computed under the old context.

**Progress:** the bounded design and acceptance conditions are recorded. The plan still records implementation as not started, pending prerequisite reconciliation; no end-to-end suffix-reuse result is established.

**Remaining:** first qualify a narrow position-preserving aligned case, retain approximate ancestry labels, prevent contamination of exact caches, and route unsupported or strict edits to exact reconstruction. Measure quality as well as work saved. This depends on exact replay, private restoration, scheduling, and coherent publication. *(P12)*

### 14. Live repair of approximate conversations

**Requested:** recompute actual accepted history into a private exact state while foreground generation continues, then publish a complete repaired replacement at a bounded final boundary.

**Progress:** the design, dependencies, and owner requirements are recorded; feature implementation and runtime qualification remain open.

**Remaining:** implement repair steps, moving-tail catch-up, cancellation, stale-result rejection, and atomic final publication. Preserve the oldest unresolved region's age across edits; track age, affected position, and outstanding work separately. Add an explicit fallback when repair cannot converge under sustained foreground load. Already emitted replies and tool effects remain fixed. *(P13)*

### 15. Idle maintenance of saved conversations

**Requested:** repair or maintain parked state only when no higher-class eligible work exists, while preserving a bounded path back to foreground service.

**Progress:** eligibility and return-capacity requirements are documented. The feature remains planned, with prerequisite integration outstanding.

**Remaining:** implement resumable maintenance, persist progress safely, handle interruption, and prove foreground return and resource retention. It depends on working persistence, scheduling, and repair. *(P14)*

### 16. Qualification, observability, migration, and rollout

**Requested:** establish correctness, quality, responsiveness, resource use, compatibility, rollback, and honest capability reporting across supported single-GPU, dual-GPU, and concurrent configurations.

**Progress:** isolated builds, targeted CPU failure tests, and earlier serial GPU parity provide useful evidence. Results are tied to their specific candidates; later changes remain separate.

**Remaining:** a current integrated build and complete feature acceptance matrix, real concurrency, two-user priority scenarios, suffix quality and repair convergence measurements, migration/rollback checks, and a coordinated enablement decision. No production rollout of this feature set is established. *(P15)*

## Additional requested investigations

**Changed-question shared-document prefix reuse:** a separate DeepSeek engine experiment passed with an explicit document checkpoint: 262,109 cached tokens and 33 fresh tokens for the changed question, with correctness checks and about 3.038 seconds to first token in that run. Warm, continuation, and return-to-earlier-prompt cases also passed. Automatic checkpoint behavior and applicability to Qwen Strata remain separate open questions. This result does not demonstrate approximate suffix reuse or a generic Strata port.

**Quantized expert/Engram weight compression:** a bounded CPU Zstd screen covered 60 samples totaling about 137 MiB and 41,994 exact block round trips. It found no net aligned-storage benefit in the screened data/settings; throughput measurements carry a CPU-overlap caveat. Repeating that screen or proceeding to GPU decompression is not justified by those results alone. This negative weight-compression result does not settle KV/state compression.

**Shared prefix chunks:** the requested investigation/planning handoffs exist, but a completed generic sharing implementation is not established by this snapshot. Automatic checkpoint selection, physical sharing ownership, and Qwen applicability need their own evidence. They must not be marked complete from the DeepSeek explicit-checkpoint result.

**Serving reliability:** restart/shutdown and input-lifetime fixes have targeted test evidence, including a prior 123-test server candidate and newer focused regressions. The newer complete suite and review disposition remain open; this is supporting work, not evidence that every engine feature is integrated.

## Recommended next milestones

1. Converge the current ownership, memory-preparation, and backend-compatibility source changes; build and test one explicitly identified integrated candidate.
2. Demonstrate real private restore, selection, execution, and exact edit replay with complete state and failure handling.
3. Qualify scheduling and concurrent conversations, including two users with independent base priorities and activity boosts.
4. Implement and evaluate the first bounded suffix-reuse case, then live repair and idle maintenance, against the exact reference path.
5. Complete codec hardware evaluation and the acceptance/rollout matrix without treating component tests as deployment readiness.

These are dependency milestones, not calendar promises. Independent work can proceed in parallel, but shared contract changes require downstream plan review. Hardware/build windows remain coordinated separately from this report.
