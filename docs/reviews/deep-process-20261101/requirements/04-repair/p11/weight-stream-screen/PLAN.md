# Lossless quantized-weight and Engram screening

Execution: implementing bounded CPU harness; model sampling not started.
Plan: revision1, ready for source/metadata work; substantial SSD reads require coordinator window.
Owner: Strata main; DeepSeek worker owns its engine/runtime and source facts.
Parent: [P11](../PLAN.md). This is weight streaming, separate from window/KV compression.
Authority: owner relay HANDOFF-main-to-strata-weight-compression-prefix-reuse-20261027.md.

## Bounded pilot

First identify engine revision, actual reader units and tensor storage metadata independently
for DeepSeek experts/Engram and Qwen experts/PLE. Do not equate DeepSeek Engram with Qwen
PLE or share binary/format assumptions. Read only small metadata/source before coordination.
Select deterministic strata across types/layers/experts/tables; record population weights
separately from observed traffic weights. Exclude headers/padding from tensor slices; report
zero fractions without silently discarding valid zero-rich payloads.

Proposed ceiling:256MiB total sample payload reads, <=16MiB per sample, 2CPU,2GiB RAM,
10min wall,1GiB temporary/output. Coordinate SSD start and total physical traffic with root;
proposed sequential application read throttle8MiB/s does not prove physical-device isolation.
No full-model hashing/recompression; reference existing source hash receipts and hash each
sample actually read. Never describe receipt hashes as independently reverified model hashes.

Zstd levels1/3/6, independent chunks64KiB/256KiB/1MiB/4MiB plus actual bounded reader
units. Decode each block and compare exact bytes+SHA256. Record per-sample distribution,
compressed and alignment/index bytes, compression/decompression time and process peak RSS.
Aggregate by engine/tensor type without hiding incompressible tails. Model random-row and
whole-expert read amplification separately; obtain real traces before traffic-weighted claims.

Deliverables: reproducible script, frozen sample metadata/seed, raw JSONL, ratio distribution
and throughput table, overhead assumptions and recommendation (including negative result).
Synthetic harness validation is not representative screening. No model reads until coordinated.

## Follow-on gates

CPU Zstd results alone establish neither GPU format support nor serving speedup. Proceed to
GPU feasibility only for useful compression after access amplification and storage overhead.
Check exact nvCOMP/HIP capability/frame settings from actual pinned SDK; no equivalence
assumed. Compare faster GPU codec, byte equality, H2D/decode/scratch/compute contention and
end-to-end latency using bounded pinned staging; direct NVMe-to-GPU is separate.
Root grants all hardware/live windows. No new grant from this plan.

## Downstream review

P00 formats/engine identity; P03 staging/workspace/cache budgets; P08 bounded IO admission;
P10 expert residency/accounting; P15 throughput/byte-exact acceptance. Integration remains
review-required until measurements justify a concrete format and pipeline. P06 window-state
format unchanged; P12 approximate suffix semantics unchanged. Dax scheduling remains root's.

## Ready pilot (metadata-only validation)

Harness screen.py SHA5678459720cefe85944fea12b9dd72c05a7acd53a82e6fb4d63b8331210a1584 and
deepseek-manifest.json SHA068f89ed18c8c44fa162ad56e0d1d5ad9b1fad82b68fb2be899dd6ad7f00e90f
are prepared in .local/strata-implementation/weight-stream-screen.60 samples total
143683200bytes:48 expert tensors across disjoint type/layer/projection/expert strata
and12 Engram table strata. Actual installed pack IQ2_XXS; file5 DSpark excluded.
All full-file SHA values are preexisting download receipts, not newly rehashed.
Source snapshot4b9 and signed engine BUILD562a0c2 provenance differ; structural reader
facts are not exact binary attestation. Schema/stat validation read zero payload bytes;
small Engram format metadata read796bytes. Independent root payload validation also
performed metadata/stat only. Payload run remains held for explicit SSD grant.

## Granted screening result — 2026-10-27

Bounded DeepSeek pilot completed PASS under token strata-cpu-zstd-20261027-1411; explicit CPU/SSD release queued through tmux and submission verified. Evidence `.local/strata-implementation/weight-stream-screen/deepseek-run-20261027T1413Z/REPORT.md` and release-receipt.json.60 samples,137.03MiB read once,900 settings,41994 exact roundtrip blocks. Expert encoded/raw slightly exceeds1; Engram best0.981715 encoded/raw but best1.031502 aligned+index/raw. No net stored-byte win for tested layout; no GPU decompression follow-up justified by this pilot. This does not remove nvCOMP/hipCOMP state-codec scope or decide other compression designs.

A pre-existing CPU fixture overlapped initial screen. Throughput retained as contention-affected observations, not isolated timing; ratios/roundtrip results remain usable. No rerun/full-model hash/GPU test performed. Actual cache traffic/read coalescing/H2D/serving performance remain unmeasured. P00/P03/P10/P11/P15: this pilot does not change current state-codec runtime contracts; only a later proposed weight-format/streaming change reopens those interfaces. Decision: keep current weight format, do not schedule GPU decode on these weak ratios. Root owns later scheduling.
