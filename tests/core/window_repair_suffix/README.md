CPU evidence update (2026-10-28): the frozen sources in repair cpu-ready-v2 manifest `be2f8a550f081783302b8995c2efae0a9f1daa2ace8d4a39c2f4a8d5e1891075` compiled and executed successfully. Capture negative controls also failed as expected. Full-engine/CMake/backend/native-runtime qualification remains separate. See local `suffix-first/incremental-repair/CPU-QUALIFICATION-20261028.md`. Earlier source-preparation notes below describe the pre-run checkpoint.

# Native repair suffix data fixture

Source prepared; configuration, compilation and execution are UNRUN. Registered
as `window_repair_suffix_test` under `STRATA_BUILD_CONVERSATION_TESTS` on Linux,
matching its POSIX allocation-failure hooks. No GPU or model dependency.

Uses real native layouts, immutable buffers and budget reservations. Checks
partial prefix copying, retained current recurrent/PLE/suffix bytes, all causal
roots and ages, unchanged pending/output history, refusals and allocation failure.
The second attachment uses a synthetic updated foreground descriptor and distinct
endpoint bytes. It tests the data producer twice; it does not execute foreground
inference, restore a device, publish a window or qualify concurrent repair.

The paired root data remains necessary even when native bytes travel as a bare
BundleHandle. Retained memory alone does not let a generic region-only consumer
recover the oldest global root. Controller/capture/ledger integration must retain
and validate the pair. No disk or portable ancestry export is qualified here.

Run only in a separately authorized bounded CPU window. Existing staged GPU
candidate and production serving are unchanged.
