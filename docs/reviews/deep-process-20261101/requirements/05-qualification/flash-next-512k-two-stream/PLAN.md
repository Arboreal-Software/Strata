# Qualify two simultaneous 512K Flash Next streams

Status: queued by Claire on 2026-10-27. Not deployed or qualified. The current
requested deployment is one 512K stream. This item does not resume the paused
broader Dax/benchmark goal or grant the Strata worker a GPU window.

[Backend qualification](../PLAN.md) · [Work queue](../../../WORK-QUEUE.md)

Voice product dependency: https://github.com/Arboreal-Software/voice-agent-webui/blob/main/docs/WORK-QUEUE.md

## Outcome

Determine whether Defiant's two unpeered RTX 3090s and 64 GB host RAM can serve
two independent Flash Next conversations, each with a 524,288-token total
window including a 32,768-token output allowance. Preserve vision support,
conversation isolation, and useful prefix reuse. Do not conflate parked host
snapshots with simultaneously active GPU KV capacity.

## Dependencies

1. Finish the requested single-stream 512K cutover and verify actual YaRN factor
   2, original context 262,144, int8 KV, exact model/engine identity, and native
   Pi context metadata.
2. Qualify the owner-selected 17 GiB conversation-cache ceiling and 6 GiB
   available-memory floor. Those limits apply to all Flash Next profiles;
   they do not guarantee two full 512K snapshots can be admitted physically.
3. Coordinate an exclusive idle GPU window with voice and codex-strata before
   serving tests. Pin candidate scripts/configs and save rollback settings.

## Execution plan

1. Inspect actual engine per-slot KV allocation, scratch/recurrent state,
   draft/MTP resources, batching modes, and host expert placement. A successful
   single-stream 1M test is only a capacity clue, not two-slot proof.
2. Create an opt-in select-model profile with explicit parallel=2 and verified
   context/scaling flags; keep the working single-stream profile available.
3. Ramp both streams through 64K, 128K, 256K, then near 512K total windows.
   Report input and output separately. Test overlapping prefill, simultaneous
   decode, staggered admission, and one request alone in the two-slot profile.
4. Compare supported serial/MTP and pipelined batching on representative paired
   loads. Measure per-user first-token latency and decode speed, aggregate
   throughput, fairness, cancellation, and time until the other lane resumes.
5. Test continuation, shared-prefix changed questions, tool-result turns, and
   A -> optimizer B -> A. Account for both incoming and outgoing snapshots in
   parking admission. Include distinct retrieval markers to detect cross-talk.
6. Record GPU memory peaks, host/cgroup available RAM, swap activity, expert
   residency, cache rejection/eviction/restoration, and disk/PCIe traffic.
7. Test actual sustained long output separately from merely accepting a 32K
   allowance. Run an overlap soak and repeated cancellation/reuse before
   recommending a default.

## Acceptance and stop conditions

- Both contexts fit with the declared output reserve, without OOM or sustained
  paging, and without breaching the configured memory floor.
- Responses are correct for distinct markers; no cross-conversation KV reuse,
  stale outputs, duplicated work, or unintended mutation replay.
- Unchanged prefixes reuse valid checkpoints after returning to a conversation;
  distinguish legitimate prompt edits from parking failure.
- Publish measured latency/throughput against the single-stream baseline. Do
  not label startup success or small-prompt completion as full qualification.
- Stop on OOM, growing swap pressure, unexpected competing workloads, stale
  identity/receipt evidence, or loss of safe rollback. Restore prior serving.

## Evidence and review triggers

Use a dated results directory linked here. Existing evidence:
[single-stream YaRN report](../../../../../experiments/flash-next-yarn/REPORT.md).
The measured 512K parking snapshot was 7.72 GiB; two need approximately15.44 GiB
before extra exchange/state overhead and must still leave physical RAM floor.

Revisit this plan if parking admission, prefix stability, engine ownership,
batching, model quantization, vision allocation, or host hardware changes.
Coordinate any engine change with the Strata implementation owner; it belongs
in that roadmap as well, not as an untracked serving patch.
