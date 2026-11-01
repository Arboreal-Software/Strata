# Retained cancellation qualification slice

Prepared source and CPU helper tests; no build/GPU/HTTP result is implied. This slice does not add allocation/copy/device fault injection.

The disposable test build must have STRATA_WINDOW_STATE_AVAILABLE and STRATA_RESIDENT_FAULT_QUALIFICATION, on supported Linux CUDA. Set STRATA_SUFFIX_REPAIR_QUALIFY=1, its real source manifest/token cap, and STRATA_SUFFIX_REPAIR_CANCEL_AT=2:import (or 2:first_output / 2:private_unit). The selector is canonical request ordinal1..65536 plus one named site. Exact-first/unowned/unqualified/unsupported profiles and every selector presence in fault-OFF builds reject before model startup. Request output remains1..64 under ordinary retained admission.

Hooks are after the real installed import, after first flushed T/committed output receipt, or after a real completed bounded private replay unit in final catchup with unresolved state remaining. For import use a fixture with known D remaining; it must preserve only the actual installed accepted prefix/pending, never pretend unread D was accepted. Private-unit requires a generated immutable receipt and enough actual C that the selected unit ends strictly before the live endpoint. Too-short or missed sites are qualification failures.

Protocol (stderr only):

- suffix_repair_cancel req=2 phase=wait site=... emits actual serial/cut/accepted/forwarded/exact_frontier/watermark/known_forwarded/generated/private_before/private_after/history/stop_sequence. Send exactly one real stdin STOP after this record. Wait is bounded10seconds; timeout, prior STOP or stdin error fails strictly.
- phase=observed binds a private-constructor witness to the new parser STOP sequence and actual request/site state. The helper does not set stop_req or synthesize a STOP. The parser alone advances the sequence for real STOP lines. The wait never exists in fault-OFF code.
- Ordinary retained cancellation cleanup keeps STOP latched, stops sampling/known-tail acceptance, teacher-forces the frozen actual accepted/emitted H, publishes exact CURRENT, captures and checked-retires the private gang. The normal prefill should_stop callback is untouched.
- Before the post-STOP oracle, final accepted/forwarded/watermark/history/known/generated must equal the witness; exact_frontier must now equal forwarded. The oracle independently resets and T1-replays operation.repaired_final's actual accepted H, compares all native bytes at phase=cancelled_frontier and pending continuation/full-vocabulary logits at phase=cancelled_continuation. Neither reset_sessions nor run_serial_remote_unit -> Verifier::run/commit/wait reads the prefill should_stop callback. The explicit oracle abort predicate ignores only the witnessed STOP; a new STOP sequence or sticky input_error aborts. No global stop flag/callback is cleared or replaced.
- Existing suffix_repair_qualification_done cancelled=0 means the oracle itself was not interrupted. The separate phase=oracle_complete record has request_cancelled=1 and the observed sequence. It is emitted only after both comparisons and restored=1 succeed. There are no extra T records from the oracle. A mismatch/abort/restore failure failstops; a missing site/oracle fails before target DONE or at serving exit. Normal successful target completion is DONE ... cancel after checked close. Existing prefix-only cancellation control remains separate.

Acceptance needs real retained GPU runs for all three sites, exact identity/all-stage compare and continuation evidence, output counts matching witnessed receipts, no later output, exact retirement/close before DONE cancel, next-request exact continuation where applicable, and independent process/container/both-GPU release/restored baseline. CPU seams cannot establish any native state, drain or resource-lifetime property.

CPU helper tests execute the actual production header with reader atomics. They cover strict selector/profile/OFF refusal, request/site selection, observed STOP witness, unchanged partial accepted history versus unread planned tail, immutable receipt/watermark/history rejection, stale/duplicate/missing STOP, true private progress, original/new STOP distinction and sticky input-error abort. All checks execute in Release; no assert-only tests.

Suggested bounded CPU gate commands (not executed by the author):

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Wconversion -Werror -pthread -Isrc/program \
  -DSTRATA_WINDOW_STATE_AVAILABLE -DSTRATA_RESIDENT_FAULT_QUALIFICATION \
  tests/core/serving_raw_contracts/test_cancel_qualification.cpp -o /tmp/strata-cancel-on
/tmp/strata-cancel-on
c++ -std=c++20 -O2 -Wall -Wextra -Wconversion -Werror -pthread -Isrc/program \
  tests/core/serving_raw_contracts/test_cancel_qualification.cpp -o /tmp/strata-cancel-off
/tmp/strata-cancel-off
```

Also require generate ON/OFF syntax, actual existing contract fixtures, real fault-OFF process selector refusal before model loading, and source/proof inclusion of the new header. Future campaign/harness wiring is parent-owned and is not changed by this slice.

Import-cancellation bookkeeping correction: a request-local retained_exact_cleanup_completed flag is set only after exact finish/capture/checked private retirement, any required oracle pass/restore, and checked request close. Only that proven path may preserve consumed plus its actual pending token despite cancelled=true. Generic cancelled-prefill bookkeeping remains unchanged. test_cancel_bookkeeping.py --build-dir /tmp/cancel-bookkeeping extracts and executes the actual bookkeeping branch and the actual next REQ_BEGIN diagnostic. It checks retained import cancellation, generic cancellation, normal/nonfinite/uncached behavior, and reconstructs the original branch as a negative control that must reproduce lost-pending exit22. Real GPU acceptance must verify the next request's REQ_BEGIN pending equals the certified actual pending and its native continuation remains exact.
