# Changed-question exact shared-document prefix reuse

Execution: DeepSeek explicit-config worker matrix passed; Qwen source integration/qualification incomplete.
Plan: revision2, ready for source follow-up; additional hardware held for root grant.
Owner: Strata main for Qwen; DeepSeek worker for DeepSeek-specific engine experiments.
Parent: [P04](../PLAN.md). This is exact prefix reuse, not P12 approximate suffix import.

## Evidence and duplicate-work boundary

Owner relay reports automatic DeepSeek changed-question request initiated full prefill and
was cancelled; it did not measure completed answer correctness/latency. Exact replay and
continuation successes do not qualify this case. Read raw request/log receipts before claims.
DeepSeek NEXT-RUN already schedules explicit document prefix via strata_prefix messages2;
coordinate that existing experiment with root/worker, do not launch a duplicate GPU matrix.
Newer root scheduling overrides stale NEXT-RUN timing/lease text.

Shared immutable chunk storage/incremental transfer remains a distinct later workstream,
with GPT-6 Pro planning route unresolved; this investigation neither substitutes its requested
review model nor implements the deferred chunk-sharing package implicitly.

## Steps and bounded proposed experiment

1. Pin each engine/source/settings/template separately. Compare actual serialized/tokenized
   prompts, including reasoning controls/delimiters; measure longest common token prefix.
   Small-metadata/source inspection only initially; no model payload reads or GPU inventory.
2. Trace cache/slot selection, state checkpoints and partial restore. DeepSeek uses its actual
   cache_n logs; Qwen requires complete recurrent/PLE/KV/pending state and compatible cuts.
3. Classify request/config issue versus missing engine capability. Protect immutable branch
   state; preserve exact prefix and replay any checkpoint gap rather than relabeling suffix
   reuse as exact. Produce source-rooted fix proposal and independent ownership boundary.
4. Proposed first authorized hardware matrix: shared document2048–8192 tokens, output<=64,
   one request at a time, <=10min, root-selected engine/caps. A/question1 -> A/question2;
   A -> unrelated B -> A/question2; continue both branches. Compare fresh/cached tokens,
   TTFT, retrieval correctness, eviction/restoration and cross-request contamination.
   Qwen/DeepSeek run independently; compare against uncached controls. No execution yet.
5. Root/DeepSeek owns already-planned256K document-prefix test. Reconcile its results before
   any extra large-context work. Concurrency is a separate qualification after exact serial
   branching works; no construction-parity or identical-prompt-only substitution.

Deliverables: token-LCP evidence, per-engine cache path map, source-backed root cause and
fix, bounded experiment spec, actual result matrix when authorized.

## Downstream review

P00 serialization/settings identity; P01 full hybrid cut; P04 exact replay; P05 immutable
branch state; P08 admission; P09 eventual concurrency; P15 branch correctness/latency.
Only dependent prefix acceptance is held. Existing P00–P15 implementation continues.

## Worker tokenizer evidence incorporated (2026-10-27)

Read preserved results/prefix-boundary-evidence.json: original262144, changed262147,
exact token LCP262115. Automatic system pin25 is below inspected source threshold256;
explicit document pin262114 lies within the common prefix and meets that threshold.
Template/prompt SHA receipts are preserved in worker-prefix-boundary-evidence.json.
This closes the actual token-boundary evidence gap and strongly supports the missing
document-checkpoint hypothesis. It does not enumerate signed-binary checkpoints or
qualify explicit GPU behavior. DeepSeek worker owns that experiment after sweep and
compression scheduling boundary; no duplicate Strata run or payload read.

## Scheduling and owner relay reconciliation

Root acknowledged the completed CPU screen and timed-out CPU build release, independently verified no build process/container, and resumed DeepSeek for explicit-document/native-max measurements. Worker owns current experiments; Strata continues source-only applicability and Qwen integration. No duplicate GPU matrix, compression rerun, model reads or heavy CPU build authorized. Current partial CUDA build250/267 is not qualification. Inspect the worker's final branch/cached-token/correctness receipts when available before proposing new hardware work.

## Revision2 decision: explicit DeepSeek path measured

Worker matrix327680-document-none-20261027T142814Z inspected: changed question262142 input/262109 cached/33fresh,104output,3.038sTTFT, markers+Record19 correct. Warm, continuation and A-B-A also pass. Preserved summary/hashes: .local/strata-implementation/changed-question-prefix/worker-document-pin-results.json. Prior revision1 hardware proposal is superseded for the immediate DeepSeek explicit-config hypothesis; do not rerun it. Automatic default failure remains unresolved. Qwen applicability and concurrent isolation remain independent implementation/qualification requirements. Earlier token-LCP numbers belong to a different calibrated prompt.

Downstream P00/P01/P05/P08/P09/P15 reviewed: no Strata runtime capability, state contract or grant changed. Their existing implementation/qualification gates remain; this external result cannot clear them. P04 source classification is updated to existing DeepSeek explicit capability with automatic-policy follow-up, rather than missing general partial restore. Parent retains hardware scheduling.
