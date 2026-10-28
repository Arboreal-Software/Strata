# Verifier copy-stream completion regression

These CPU fixtures compile complete extracted `batch_launch`, `batch_poll` and
`poll_snapshot_drain` methods. The numerical kernels, device calls, doorbell
service and surrounding object storage are minimal mocks. They test completion
classification and retained failure state, not GPU execution or safe teardown.

The fixed extraction must exactly match current production methods. The mandatory
freshness checker also requires the production failure-latch declaration to remain
initialized false and checks the mock declaration/wrappers. Both build targets
require this check; CTest uses it as a required fixture even for filtered runs.
After changing a covered method, reconcile the extraction and test assumptions.
Do not update extracts just to silence an unexplained behavioral change.

The old-method fixture must exit with the exact false-completion assertion;
crashes, timeouts and sanitizer diagnostics do not count as expected failure.
Fixed cases cover copy-query failure, terminal no-effect poll/launch/snapshot
refusal, diagnostic allocation failure, pending queries, successful publication
and repeated completed polling. The old extraction deliberately retains the
specific regression and is not matched against current production code.

The local CMake fragment supplies both executables, mandatory freshness and five
CTest entries. A parent test configuration must opt into this subdirectory.
Sanitizer flags are inherited from that configuration. No engine/GPU SDK or model
is needed. The standalone mock does not establish full production class ABI,
backend linkage, inter-thread behavior or pipeline-controller ownership safety.
