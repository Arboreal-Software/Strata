# Experimental serving repair policy

This CPU gate covers hard admission, authentic ancestry-age projection, bounded importance selection and the actual serving decision seams. It proves no native execution, device completion or latency bound.

The retained-owner feature is selected by `STRATA_SERVING_REPAIR_OWNER_SLOTS`. Without a scheduling policy, young or unknown-age owners advance only at idle boundaries; active generation and queued foreground win. Idle owners are visited in entry order. No default escalation age is supplied.

The optional experimental policy v1 requires all five environment variables together:

| Variable | Meaning |
| --- | --- |
| `STRATA_SERVING_REPAIR_AGE_NS` | Positive oldest inherited monotonic age required for escalation, in nanoseconds. |
| `STRATA_SERVING_REPAIR_LIVE_WEIGHT` | Positive integer ticket weight per eligible live repair owner. |
| `STRATA_SERVING_REPAIR_FOREGROUND_WEIGHT` | Positive integer ticket weight for waiting foreground. |
| `STRATA_SERVING_REPAIR_MAX_ESCALATED_UNITS` | Positive maximum consecutive repair units before foreground must resume. |
| `STRATA_SERVING_REPAIR_RANDOM_SEED` | Explicit unsigned 64-bit SplitMix64 seed, including zero. |

Weights and consecutive-unit cap are bounded to 1,000,000. Inputs must be unsigned decimal strings with no whitespace, sign or trailing text. Partial/invalid configuration refuses startup. These are caller-supplied experimental weights, not an adopted importance formula. Record the exact policy and seed in the campaign; the engine emits `REPAIR_POLICY` and `REPAIR_SELECT` records with tickets, total weight, actual age, observation time and publication/record versions.

A real causal import records its steady-clock birth in the runtime's incarnation domain. Every selection recomputes age from the controller's actual ancestry. Unknown, foreign and future roots cannot escalate; a cached old duration cannot hide an unknown inherited root. Saved/disk work is excluded whenever foreground or eligible live work exists, irrespective of its weight or age. No saved/disk execution path is introduced by this slice.

`STRATA_SERVING_REPAIR_IDLE_UNIT_TOKENS` remains the positive per-unit cap for parked repair. Active repair uses its existing four-token bound. Each parked selection runs at most one unit and rechecks terminal input before it starts. Aged parked owners can compete before a queued request starts. They are never interleaved into another conversation's running RAW fence. Active owners may compete after a committed output receipt. Mandatory exact-consumer/capacity-refusal drains remain explicit fallback boundaries.

`serving_repair_admission_test` exercises production logic and real ancestry construction/union/partial coverage. `serving_repair_scheduler_seams_test` compiles the actual trusted projection, active decision and selection lambda extracted from `generate.cpp`, with host controller fixtures and a negative control. Source checks pin actual parked refresh→selection→execution ordering and the queued-input scheduling boundary.
