# All-resident graphs must not consume stale remote sums

The zero-doorbell graph bypasses host expert dispatch, including the optimizer's
begin/reset and remote accumulation. Its resident_plan call also passes no skip
pointer. Unconditionally adding the optimizer's mapped sum can therefore reuse a
previous host-served stage/window's remote contribution. The capture callsite now
omits remote combine whenever the effective all-resident variant is selected.
Ordinary and adaptive ar_off fallback variants retain existing combine behavior;
this does not disable remote experts or adaptive residency.

Prepared, not run. The CPU fixture includes the actual optimizer declaration,
complete begin/accumulate/combine methods and actual ar_on expression/callsite.
Storage and arithmetic kernels are CPU seams with small explicit geometry; no
CUDA graph or device execution is claimed. It seeds a nonzero earlier-stage sum,
then verifies an all-resident stage leaves primary output unchanged, including a
stale device-plan skip value. It also covers ordinary remote contribution,
adaptive fallback, existing primary-only skip and absent optimizer. The old
callsite must hit exactly the stale-sum assertion/exit86. Full source and old
callsite hashes plus exact extraction checks are mandatory. Prior serial/chain
source pins are refreshed without changing their tested methods/cases.

Standalone and root CMake registration are prepared only. This is a numerical
CPU regression of the actual host dataflow, not hardware/model output proof or
whole-controller concurrency qualification.
