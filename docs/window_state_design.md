# Window state implementation checkpoint

The new window-state code is under development. It is not yet connected to serving and does
not enable save/restore, scheduling, approximate reuse or repair in the running engine.
Default-path preservation remains a qualification requirement; this checkpoint is not a current-source equivalence claim. The implementation base is engine
`1aff6d14117b88171fcfcec8f2b55fdc47f2b165`; the baseline checkout remains separate.

For the current feature inventory and validation boundaries, see [the backlog report](window_state_backlog.md) and [issue #1](https://github.com/Arboreal-Software/Strata/issues/1).

## State and native layout

`window_state.hpp` separates window identity from a coherent state cut. Accepted length,
forwarded length and exact frontier satisfy `F <= C <= L`; the pending interval is the exact
accepted suffix `[C,L)`. A domain-separated SHA-256 binds the entire accepted token history.
Each device part must share the same cut and retain all mandatory native components.
Approximate state carries contiguous age/ancestry metadata; pending input alone is not debt.

`window_native.hpp` converts already settled text-only, speculation-disabled native host
captures into typed descriptors, and reconstructs private native host images for the existing
restore validator. It covers FP16, INT8, Q4_0, K8V4, rotated FP16 and rotated INT8; pooled spare
rows, raw index tails, GDN recurrence/convolution and PLE history/token metadata are distinct.
Support-only QSA allocation is preserved when a split stage has no executed QSA layer.
Steering mode is explicit. Expected layouts come from resolved runtime settings, never from
an imported file. The first adapter requires little-endian native data.

Exact RNG/parser continuation is explicitly unsupported. The declared mode permits a new
generation policy after state restoration. Draft state is rejected, not silently omitted.
Runtime bindings, graphs and residency maps must be rebuilt by the later private-restore
adapter. Retained historical checkpoint caches may be rebuilt; omitting them does not omit
live causal endpoint state. No GPU completion or safe publication is certified by a host descriptor.

## Resource admission

`window_budget.hpp` atomically reserves the whole vector: each device, total physical host
memory, the pinned subset, disk, IO slots and private shadows. It never pools GPU capacities.
Private work requires a shadow charge. Saved maintenance must hold a foreground-return lease
from the same ledger; that capacity remains charged until its final owner releases it.

The execution layer must supply truthful retained-capacity/peak estimates and hold each shared
reservation through device/IO drain. The ledger cannot infer actual allocation sizes or which
return image matches the selected window. Baseline observations may expose overcommit; new
admission then fails while existing leases remain accounted. The host descriptor/native copy
path currently has transient full-copy overlap that must be budgeted before runtime use.

## User activity priority

`thread_priority.hpp` stores base priority separately from bounded activity leases. The external
manager supplies opaque lease IDs; the engine does not observe browser or user activity itself.
Multiple active leases contribute their maximum boost. Base decreases remain after expiry.
Bounds and durations are constructor policy inputs; test values are not production defaults.

Base updates use a base-version CAS; activity updates and disconnects use their lease-version
CAS. Expiry also increments lease versions, so delayed refreshes cannot revive old activity.
A monotonic control timer must call `advance_time` without waiting for another request.
Registry and lease tombstones are bounded; a full registry returns an explicit limit error.
Thread/engine generations fence stale requests after close/recreation or restart.

Accepted revisions are distinct from executor acknowledgements at safe cuts. The snapshot
reports base, local effective priority, activity reasons, versions and applied cut. Dependency
inheritance and work-class eligibility have separate components; serving integration remains incomplete. An open UI
therefore grants no right to run saved maintenance, and this checkpoint advertises no runtime
scheduling capability. The later engine control interface must preserve these distinctions.

## Remaining integration and qualification

Safe quiescence/async leases, actual allocator accounting, exact edit replay, private GPU
restore/publication, durable disk receipts, the dependency graph, bounded serving units,
expert accounting, codec evaluation, approximate reuse and repair remain required work.

Current serving arguments identify 256K INT8 Flash-Next with MTP and automatic layer split.
The research 128K/Q4 resource sheet is only illustrative. Required qualification includes the
actual target envelope, single and dual GPUs, two concurrent users and competing prerequisites.
No CPU fixture establishes model correctness, GPU capacity, foreground latency or quality.
Independent review and owner enablement remain pending. No live pin or service was changed.


The verifier now retains commit failures across repeated waits. Its opt-in
`poll_snapshot_drain` checks compute, copy and side streams, pending commit events,
watchdog release and chained stages. Errors remain terminal for snapshot use. This
probe requires exclusive execution ownership and joined host producers; it does not
establish those conditions, validate logical stage cuts, or retain an operation lease.
The scheduler integration must supply them before using this as a capture gate.
Metal explicitly declines this probe. SYCL has its own verifier and remains unsupported
by this implementation. Earlier frozen candidates have backend build evidence; the full current checkpoint has not been built across all affected backends.


`WindowExecution` supplies single-writer admission, quiescence, common-stage-cut validation,
append fencing, and cancellation lifetime. `WindowRead` keeps the quiesced allocation alive
and prevents resume during capture. Completion requires joined host producers and drained
device/IO work separately. A device error cannot count as completed work. The reactor must
retain each operation's completion handle; losing one quarantines its owner and reservation
rather than freeing memory still referenced by submitted work. This gate is not yet wired
into request execution or native capture/restore, so those paths remain unqualified.


`prepare_native_restore` validates the entire destination gang before writes, reconstructs
native bytes, restores actual PLE token state, runs private binding setup and drains each
device. A prepared handle retains ownership but does not activate execution. Destination
session/reservation claims prevent reuse while a prepared or failed transaction owns them.
The private allocator must still guarantee non-aliasing allocations; scheduler integration,
atomic selection and real hardware qualification are incomplete. Failed transfer/binding
transactions retain their owners and staging until explicit teardown retirement.


`WindowSelection` serializes exact old-key/cut/publication checks with execution admission.
Replacement requires an old quiesced read from that same execution gate and an unselected
prepared destination. The old gate is retired before the replacement is visible; retained
reads and work handles keep its storage alive. Publication replaces the entire prepared
gang in one pointer-level transaction, with no GPU/IO waits under the selection mutex.
Separate components provide crash receipts, scheduler eligibility and private allocation; their complete serving integration and qualification remain open.
