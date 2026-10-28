# Submitted lookahead drain CPU fixture (prepared, unrun)

Exact production RouterLookahead class, start/close/submit/drain methods and worker
are extracted. Mandatory freshness checks all pieces against current header/source.
The test-only class substitutes WakeCv for the single condition-variable member;
method bodies and scalar router computation remain actual source. Warm/Foresight
callbacks and CPU feature probes are mocked; no CUDA, files or engine is run.

Seven independent counted scenarios cover real blocked host borrow, expired
caller deadline, persistent seal and fair retry, competing/cross-owner guards,
reopening predictions, worker self-refusal, actual worker exception recording,
terminal wake and no reopening, and pending (not busy) submission notification.
The last seam pauses submit AFTER its mutex unlock and BEFORE notification, then
admits a bounded deadline waiter while the worker sleeps. notify_one is permitted
to select only that waiter, leaving the worker asleep. The current broadcast wakes
both; the separate exact previous-notify control must return only assertion70
with the exact diagnostic. Crashes, timeout, sanitizer output and other assertions
are not accepted. All test-owned threads are joined before the expected assertion.

This is a controlled condition-variable scheduling seam, not a probabilistic
stress test or performance measurement. It does not prove completion of tasks
spawned by warm/foresight downstream, actual GPU retirement, or driver activation.
The production worker exception is rethrown (uncaught thread still terminates);
fixture catches actual run on its own thread solely to inspect sticky refusal.

Configure this directory with CMake; no root CMake changes are included. Three
CTest cases: freshness, seven-scenario fixed suite, precise previous-notify control.

PR44 oracle follow-up: after the real worker drains, the SAME retained guard is
held while another submit is attempted. Its skipped count must increase and warm
count must remain unchanged. The test always drains any incorrectly accepted job
before mutation of borrowed input. Failure closes/joins the worker before asserting.
An independent control removes ONLY submitted_drain_ from the actual submit
rejection condition and must hit the exact idle-seal assertion70; it cannot pass
because the earlier busy condition happened to reject submissions. Timeout,
cleanup failure, sanitizer output, other assertions or crash are not accepted.
Freshness verifies this one-expression transformation as well as original source.
The suite now has four CTests and separately counts the idle-seal assertion.
Snapshot follow-up work is not included; production code is unchanged from PR44.
