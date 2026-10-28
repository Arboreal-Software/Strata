# Pipeline copy-query terminal failure

The fixture compiles nine complete extracted production methods with mock CUDA,
clock, device scope, sampling and staging dependencies. A mandatory freshness
check compares those methods and the actual terminal latch/helper byte for byte.
Thirty-one small adapters additionally extract the actual **first statement** of
submission/configuration entry points. These adapters check terminal refusal
before effects; they do not qualify the remaining large method bodies.

The expected-red binary changes only the copy-status check back to its defective
form. Only exit 86 with the exact false-completion assertion qualifies; crashes,
timeouts and sanitizer output do not. Fixed scenarios cover pending/success,
terminal error, allocation failure during diagnostics, sticky retries, retained
work flags/counters, output preservation and refused snapshot certification.
The sibling batch-copy fixture retains its previous semantics with the new latch
false and is included in the prepared CPU qualification batch.

The atomic latch is not concurrent execution authorization. The existing
serialized caller/producer-closure contract remains required. Already-issued
callbacks must be joined externally. Cleanup and release_gpu_waits remain usable;
the latch never represents successful release and has no reset setter. Other
CUDA query-error paths, hardware behavior, whole-engine linking, Metal/SYCL and
all-gang cancellation are outside this fixture's proof.

Public registration and this source checkpoint are prepared, not executed.

Follow-up source checkpoint adds complete route_stats_report and release_gpu_waits extracts (no release stub). Terminal report refuses before allocation/device/host resets. Actual release raises all mapped flags and preserves terminal refusal afterward; stream query errors return false rather than drained. Added same-object NotReady→Success→one finish→no further done announcement and unused-copy/no-query modes. pl_finish itself is not an idempotent public API; this fixture uses the existing done-gated caller protocol, not a fabricated second-finish guarantee. CUDA calls, clock/device scope and CPU sfence remain explicit mocks. No production destructor or caller-level join proof. New follow-up cases remain unexecuted.
