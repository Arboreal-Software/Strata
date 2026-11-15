# Pipeline input admission race fixture

Uses the production submitted-work guard, exact driver admission prefix and exact uncertainty wrapper/drain. A controlled input load returns the admitted value and then flips terminal input before the actual step prefix. This represents the precise interleaving without scheduling sleeps; backend work is mocked.

The fixed fixture covers the racing admission, clean rejection before admission, a real backend failure remaining sticky, and legacy step admission. The old-prefix variant removes only the owned guard around the second check; it must fail the precise admitted-race assertion with assertion exit86. Assertion failure is distinct from the production guard fatal exit1. This does not qualify full pipeline or GPU behavior.

Before compilation, `check_extract.py` must compare the current `src/program/generate.cpp` with this fixture directory. It checks both actual extracts and the exact old-prefix delta. The build target must depend on this freshness check. Compile as C++20 with assertions, pthread and `include/strata/core` on the include path. POSIX-only; no CUDA library required. Build the expected-old variant with `OLD_ADMISSION` defined and invoke it with `race`; crashes or sanitizer errors are not expected-red evidence.

The parent captures each child's stderr and accepts the fatal scenario only with
exit1 and exactly the production teardown diagnostic. Assertions, extra output,
sanitizer diagnostics, signal deaths and timeouts cannot satisfy that oracle.
Normal scenarios require exit0 and empty stderr. Diagnostic capture is capped64KiB.

The default test additionally injects a failing sticky-state assertion before the
guard and requires the parent oracle to reject its exit86/FAIL diagnostic. Another
child emits explicit synthetic assertion/sanitizer marker noise before real fatal
teardown; exit1 with extra diagnostics must also be rejected. These marker strings
are harness controls, not a real sanitizer error.

`--inject-assertion` runs the full parent harness with that assertion forced in the
fatal scenario. It must fail with exit86 and both the child sticky-assertion and
parent rejection diagnostics. This external negative control proves the parent
cannot report success for the injected assertion. The OLD_ADMISSION `race` mode
remains a separate reproduction and now requires exit86 plus exactly the original
admitted-race diagnostic; do not confuse it with expected fatal teardown exit1.
No production guard or driver behavior changes in this fixture correction.


Child supervision uses nonblocking diagnostic reads and `waitpid(WNOHANG)` with a
three-second monotonic deadline. EOF does not establish child completion. Errors,
oversized diagnostics or expiry kill the exact unreaped child and reap it before
reporting failure; no parent-only alarm remains. The fixed64KiB capture buffer
avoids string allocation while the child is active. A scope owner supplies the
same cleanup on exceptional parent unwind. Kernel reap after SIGKILL must finish
before reporting; this is not a guarantee against uninterruptible kernel IO.

Two short negative controls deliberately hang with stderr open and with stderr
closed. Each must time out, be killed by SIGKILL, be reaped, and produce ECHILD on
an immediate second wait; the closed case must have observed EOF before cleanup.
They run in the default fixture and separately with `--timeout-controls`. These
controls prove local cleanup rather than relying on an external container timeout.

The timeout repair and its new candidate are source-only and unrun pending a new
CPU grant. Previous fixture results do not qualify this change. Production guard,
driver extracts and exact assertion86/fatal-exit1 diagnostic oracles are unchanged.


PID ownership is separate from captured exit status. Any `ECHILD` permanently
disarms the child owner, preserves an unqualified failure, and prevents subsequent
cleanup/destructor signals. Cleanup preflights `waitpid(WNOHANG)` before a kill.
The single-threaded fixture refuses inherited SIGCHLD auto-reaping/custom handlers
before forking, rather than assuming it has exclusive status ownership.
A safe status-loss syscall seam first reaps the real child, then returns synthetic
ECHILD; the control requires zero kill attempts, no qualified status, and ECHILD
on a second wait. Its kill guard refuses any attempt to signal the already-reaped
PID, so this negative control never probes a potentially recycled live PID.
