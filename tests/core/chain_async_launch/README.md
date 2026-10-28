# Chained asynchronous launcher completion

The overlapping-stage branch could return success without finishing/checking its
own HostGraphLaunch. Pending initial status is optimistic; its destructor then
releases GPU waits on any joinable unfinished worker and ignores the final launch
error. The fix finishes/checks that worker after downstream host service returns,
before local stream checks and successful commit publication. Downstream exceptions
retain the existing release-and-join cleanup. Nonthreaded CUDA performs only a host
status confirmation; HIP excludes this launcher branch as before.

Prepared, not compiled or run. The fixture includes the actual complete
HostGraphLaunch header and exact complete chain branch. CUDA calls, downstream
Verifier, device scope, copy checker and graph service are explicit CPU seams.
A condition variable holds the real launch thread until downstream service; no
sleeps establish ordering. Cases cover late launch error, successful completion,
copy failure after launcher completion, downstream exception cleanup and ordinary
nonthreaded completion. The preserved old branch must fail exactly the late-error
completion assertion with exit86. This is not whole-Verifier, GPU, multi-device
or controller qualification. The fixture's independent source pin/freshness check
is mandatory, and the existing serial-copy fixture pin is refreshed without
changing its copied methods or assertions.

Standalone CMake registration is prepared only. No source checker, compiler,
fixture or container was run for this change. Full backend compilation and actual
runtime qualification remain required before claiming completion authority.

Root CMake now registers this fixture beside verifier_serial_copy; this new public
registration is unrun. Both old and fixed processes have an inner20-second Python
supervisor; timeout kills/reaps the exact process containing all launch threads.
No fixture forks descendants. CTest30-second limits and the candidate's outer
container deadline provide additional bounds; a blocked actual join cannot leave
a child process after the inner supervisor completes. This bounds failure rather
than pretending the production std::thread::join itself accepts a timeout.

PR40 fixture compatibility: the fixed chain extraction now passes the actual
strict-retirement boolean to the mock device scope. The mock accepts the optional
boolean; this fixture still tests launch/copy ordering, not checked device
semantics. Full-source pins for chain and serial-copy freshness follow PR40.
Historical expected-red extracts are unchanged. Follow-up not yet executed.
