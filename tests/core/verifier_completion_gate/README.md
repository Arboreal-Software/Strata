# Serialized verifier completion publication

The actual gate linearizes a trusted serial completion cut against monotonic
release/copy-failure requests. It does not synchronize CUDA, own runtime resources
or authenticate a caller's completion claim. No mutex spans doorbell service,
GPU waits, callbacks, registry operations or owner destruction.

The factory binds the actual immutable next_ chain only before init can publish
any stage to the watchdog registry. Configure strict policy and topology first;
already-initialized, unhealthy, cyclic or previously bound stages refuse without
partial installation. Caller supplies an exclusively moved genuine foreground
host-only ReservationLease, which the gate retains until its last Verifier,
Token or Cut reference dies. The metadata envelope includes one serialized
Token/Cut pair; external storage retaining additional historical cuts needs its
own admission. No runtime owner cycle is introduced.

Begin only after all reversible work admission. Destroying a live Token leaves
the gate busy: no implicit cancel/reuse. Main controller must keep the actual
in-flight owner or fail-stop. Successful commit creates a historical cut; a later
release invalidates future units, not a previously linearized completed cut.
Actual driver integration and output-disposition policy remain separate gates.

Prepared CPU fixture uses the real gate, actual WindowBudget and exact factory/
monotonic hook bodies with explicitly mocked Verifier fields. Eleven groups cover
healthy multistage/no-allocation operations, deterministic release-before-commit,
copy failure, foreign/replay, invalid topology, already-initialized refusal,
abandoned-token refusal, allocation failure, stage limit, last-cut charge retention
and missing/insufficient/shared quota refusal. It does not execute full Verifier
methods or GPU work. Existing copy-query fixtures now include the exact latch
helper with an explicit gate-mark counter seam; actual gate behavior is tested
separately here. Source freshness is mandatory. Public root registration and all
fixtures remain uncompiled/unrun until a fresh resource grant.
