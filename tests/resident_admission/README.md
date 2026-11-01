# Resident host admission fixture

Compiles exact extracted RAW bootstrap/shared-cap/admission methods and actual WindowBudget. Source equality is a required CTest fixture; assertions remain enabled in Release. Native allocation/configuration and phase fields are explicit CPU seams, not GPU qualification.

Cases cover shared aggregate-H exhaustion, real-ledger rejection, bootstrap failure rollback, admitted later refusal and retained lifetime. Run the standalone CMake directory or the experimental window-state CPU test suite. Sanitizer runs require a separate sanitizer configuration. Initial frozen tests passed Release and ASan/UBSan; CMake registration is newer and requires its own build evidence.

The permanent fixture now also copies the separately CPU-qualified observer fixture unchanged: exact HostUsage declaration/body, owning-thread/drained/no-pending/initialized refusal with unchanged output, real same-H live/child/retained-credit observation and final global/H baseline restoration. All prior admission/gate/bootstrap/exhaustion/failure/lifetime cases remain. host_usage.inc and bookkeeping.inc are materialized from current actual source with fragment hashes; existing source-equality CTest covers both. CMake target names/registration remain unchanged. This source reconciliation was not compiled or executed; the separate CPU receipt does not qualify this new CMake integration.

Current two-slot shared-H reconciliation reviews all extraction entries and pins
`initialize_slot_shared_host` separately from the unchanged resident initializer.
Actual HostCap/admit_host bodies now enforce16 claims for main-only ownership and48
for genuine shared-slot bootstrap. Added cases execute the actual slot initializer
with explicit physical occupancy/geometry/runtime CPU seams: one aggregate cap for
main+two slots, aggregate exhaustion, claim saturation, rejected foreign identity,
and retained root bootstrap credit through descendant teardown. Existing cases and
Release assertions remain intact. Current pinned CPU Release build and both source/
executable CTests pass; native/GPU execution remains outside this fixture.
