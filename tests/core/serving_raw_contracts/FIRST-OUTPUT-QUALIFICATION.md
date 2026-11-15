# First approximate output distribution diagnostic

Prepared source/actual-method CPU seam fixture only; not compiled or model-tested by this edit. Active frozen campaigns and production/default execution are unchanged.

With STRATA_SUFFIX_REPAIR_QUALIFY=1 on an admitted retained request, the existing qualification host admission now reserves THREE simultaneous FP32 vocabulary rows: one saved real first-output row and the existing two exact-oracle rows. The additional row is actual n_vocab*sizeof(float), currently993280bytes for248320tokens. Saved storage is preallocated after genuine H/global admission and before any request mutation. The credit is declared before the row and outlives it. Exact-first qualification retains two rows/no saved row; no qualification performs no row admission/allocation/readback. A zero-output import cancellation captures no row and produces no approximation record.

After actual generate_one returns its committed immutable receipt and the first T is flushed, a read-only getter retrieves that real controller receipt. The qualification capture binds operation/publication/cut/token and actual selected accepted/pending metadata. It hashes only accepted[0:forwarded], excluding the newly emitted pending token. Actual runtime binding validation precedes readback. The validated serial chain locates the last-stage device; WindowDeviceScope selects it; primary Verifier::copy_logits(0,...) delegates to the true final head; copy status, actual cudaDeviceSynchronize and scope restoration are checked. It verifies finite values and argmax equal to the actual neutral-greedy emitted token. No new sample, history mutation or native snapshot is taken. Capture precedes first_output cancellation WAIT and later foreground/private head work.

At final exact qualification, the actual generated-receipt count is supplied by Outcome. A generated request must have exactly one saved first-output record. Bind its request, emitted token, conditioning-prefix digest, position and watermark to the final actual accepted H. In longer-output cases, later tokens remain immutable but the saved first-output row belongs to its earlier position. The existing clean reset/T1 replay asks forward() for logits exactly when at+1 equals that saved forwarded position. It reuses buffers.exact_logits; there is no fourth row allocation. Import cancellation with generated0 requires no captured row. Post-STOP witness behavior and original STOP latch remain unchanged.

Distinct observational stderr record:

```
suffix_repair_approximation req=... phase=first_output exact_certificate=0 operation=... publication=... cut=... forwarded=... exact_frontier=... watermark=... conditioning=... emitted_token=... vocabulary=... row_bytes=... simultaneous_rows=3 kl_exact_to_approximate=... rms=... max_abs=... exact_argmax=... approximate_argmax=... emitted_matches_exact=... finite=1 reference=reset_replay_same_conditioning capture_ms=... binary=... source=...
```

Finite zero and nonzero divergence are both valid observations. Approximate argmax must match the emitted receipt; clean argmax must match actual clean T1 pick. The record is not exact PASS or a task-quality certificate. No metadata-tag/hash differences are counted as numerical drift, and the exact native comparator is unchanged. Full final native-state and pending-continuation/logit equality plus mandatory original-endpoint restoration remain required. Failures in readback, binding, finite metrics, execution/drain or restoration fail strictly with retained ownership; no exact fallback from uncertain mutation.

The diagnostic is absent in production timing. Correctness receipts include capture_ms; comparison/reset work is within existing oracle timing. Do not infer native-tensor error bounds from one measured output distribution. The current boundary36 T1 prefix matches this exact replay; larger/batched prefix numerical schedules require separate qualification.

Actual-method fixture:

```
python3 -B tests/core/serving_raw_contracts/test_first_output_qualification.py --build-dir /tmp/first-output-controls
```

It extracts current actual admission, capture, binding, distribution report and saved-position replay loop into recording CPU seams, links actual WindowHasher and compare_suffix_logits unchanged, and compiles strict-Werror. Controls cover QUALIFY absent, exact-first2 versus retained3 row overlap, quota refusal before allocation, actual final-device selection/row0/readback/drain/restoration ordering, real receipt/prefix binding, duplicate/wrong request/token/position, longer-output saved position with the existing scratch row, zero-output no-readback, finite zero/nonzero divergence and readback/sync/device/binding/argmax/nonfinite failures. No CUDA/native-state/drain simulation is accepted as GPU evidence. The parent owns later syntax, build, parser/proof closure and real model qualification.
