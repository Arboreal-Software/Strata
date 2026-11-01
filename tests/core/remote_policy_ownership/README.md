# Owned remote routing CPU fixture (prepared, UNRUN)

Run `extract.py --source-root <checkout> --output <private-output>/fixture.cpp`
then compile that generated source with a C++20 compiler, assertions and optional
ASan/UBSan (`-pthread -O0 -g`). This is a source-extracted fixture: the class and
complete admission, begin, finish, copy and close methods come from the supplied
checkout at generation time, recorded by SHA256. There is no historical success
receipt and no compiler/test has been run for this candidate.

CUDA calls, expert layout/cache, RemoteExpertOpt and arithmetic kernels are tiny
CPU fakes. The tests exercise the actual admission/retirement and control flow,
not CUDA ordering, numerical equality, real source DMA ownership, device memory
allocation, multi-window controller service or default throughput. One logical
copy/compute stream each is represented; no GPU is used.

The ordinary cases cover admission before routing, duplicate begin/finish,
inner operation retention after outer scope retirement, both writer stream
drains, publication ordering, no-hit pair, copy-read exclusion/reentrant writer
probe, cleanly drained failed submission, all-peer unattempted rollback, move
ownership, generation exhaustion and owner-destructor reentry outside locks.
Fourteen isolated children cover unsafe close, uncertain D2H fail-stop, attempted
writer abandonment, failed compute finish, device restore failure, failed
compute submission and persistent allocation failure during compute/writer error
reporting, plus a real nested managed-copy restore failure during the compute
finish synchronization seam. A clean compute drain cannot hide concurrent helper
poison. A genuine shared runtime containing the helper and backing buffers tests
retention while compute is pending, release after successful finish, and retained
unknown ownership after attempted writer abandonment. Managed allocation-free and
stream-destroy failures also require fail-stop before losing ownership. Assertions exit86; expected fatal cases require exit1 plus
the exact production diagnostic, with no extra assertion/sanitizer output.
Unknown-retention children deliberately exit without pretending teardown worked.

Remaining fixture gates: actual competing
copy threads/capacity, concurrent finish-versus-begin, custom callback exceptions,
real source staging lifetime and backend compiler qualification. Main must wire
all-helper writers before adaptation scans and outer readers before verifier
RemoteExpertOpt::begin / helper holds. Do not enable managed binding by itself.

The parent capture/reap harness is adapted from the hardened pipeline admission
oracle: nonblocking bounded stderr, steady deadline, wait-before-kill, EINTR retry,
ECHILD disarms PID ownership, no allocating diagnostic operations before cleanup.
Negative controls cover open/closed-stderr hangs, lost status, assertion injection
and extra sanitizer-marker output. It requires SIGCHLD exclusive/default reaping.
These new fixture controls remain UNRUN.

Strict cleanup also covers configured-but-unbound partial startup, not just an
owned binding: seeded partial-state fixtures exercise no-device/no-stream
cleanup and free/stream-destroy failure before binding. These use actual close()
but do not simulate the full open() loader path.
