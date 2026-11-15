#!/usr/bin/env python3
"""Supplementary source-ordering gate for the RAW INT8 restore suffix clear.

The byte-level regression is executed by raw_int8_restore_clear_exec; this script only pins the ordering and
shape of the PRODUCTION call site that the CPU byte test cannot reach (device scope, stream lifetime, poison
paths), plus a few anti-mirror properties of the test itself.  It proves nothing about CUDA behaviour.
"""
import re
import sys

RUNTIME, HEADER, TEST = sys.argv[1], sys.argv[2], sys.argv[3]
SNAPSHOT = sys.argv[4] if len(sys.argv) > 4 else None


def text(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


runtime, header, test = text(RUNTIME), text(HEADER), text(TEST)
problems = []


def need(condition, why):
    if not condition:
        problems.append(why)


def restore_body(source):
    start = source.index("SuffixRawResult SuffixRawTransaction::restore(")
    end = source.index("SuffixRawResult SuffixRawTransaction::", start + 10) if "SuffixRawResult SuffixRawTransaction::" in source[start + 10:] else len(source)
    return source[start:end]


body = restore_body(runtime)

# 1. the pure preflight runs before any mutation is started
need(body.index("raw_int8_restore_clear_preflight") < body.index("pending_=storage;phase_=Phase::restoring;"),
     "preflight must precede pending_/restoring (no mutation before every range is validated)")

# 2. the clear is per-stage: the stream handle is declared inside the stage loop, never hoisted above it,
#    because private_stream() is per-thread PER DEVICE
loop = body.index("pending_=storage;phase_=Phase::restoring;")
declared = body.index("cudaStream_t clear_stream=nullptr;")
sync = body.index("cudaStreamSynchronize(clear_stream)")
stream_get = body.index("private_stream(clear_stream)")
need(declared > loop, "the clear stream handle must be acquired inside the stage loop, not cached across stages")
need(stream_get > declared and sync > stream_get, "stream acquisition must precede its use and its checked drain")

# 3. the clear lands before that stage's restore copy, and every failure poisons
copy = body.index("conversation_snapshot_restore(image,*stage.session")
need(declared < copy, "the tail clear must precede the restore copy of the serialized prefix")
need(body.count("suffix clear") >= 2 and "poison(\"RAW unserialized INT8 suffix clear failed\")" in body,
     "a failed clear submission must poison, not continue")
need("poison(\"RAW unserialized INT8 suffix clear drain failed\")" in body,
     "a failed clear drain must poison, not continue")
need("poison(\"RAW suffix shape changed after preflight\")" in body,
     "the post-preflight plan recheck must poison on any change")

# 4. the planner itself keeps the exclusivity gates that were reviewed in
need("kv_elastic != -1" in header and "owned_elastic != nullptr" in header,
     "elastic/VMM-backed pools must be refused: reserved range is not necessarily mapped")
need("st.kv_mode != 0" in header, "only fully resident mode-0 states are in scope")
need("st.kv_rot" in header and "st.kv_q4" in header and "st.kv_hybrid" in header,
     "rotated/Q4/hybrid formats must be refused")
need("n_slots != st.n_pages" in header, "the actual allocation capacity requires n_slots == n_pages")
need("page_size" in header and "qsa_real_shapes" in runtime,
     "the page size must come from the real shape, not a hardcoded constant")

# 5. anti-mirror: the CPU test drives the production entry points and does not re-derive the ranges
need("raw_int8_restore_clear_preflight" in test and "apply_raw_int8_restore_clear_plan" in test,
     "the CPU test must call the production preflight and dispatch closure")
need(not re.search(r"\b(serialized|capacity)_cells\s*=[^=]", test),
     "the CPU test must not duplicate the planner's range arithmetic")
need(not re.search(r"\bassert\s*\(", test),
     "the CPU test must not rely on assert(): NDEBUG would silently remove the gate")
need("return 1" in test, "the CPU test must fail loudly")

if SNAPSHOT:
    snap = text(SNAPSHOT)

    def both(name):
        return snap.count(name)

    need("canonicalize_restored_suffix" in snap and both("canonicalize_restored_suffix(") >= 3,
         "the serving snapshot restore path must canonicalize the omitted tail (helper + both restore call sites)")
    need("raw_int8_restore_clear_plan" in snap and "apply_raw_int8_restore_clear_plan" in snap,
         "the serving path must reuse the SAME planner and dispatch closure, not re-derive ranges")
    need("private_stream(stream)" in snap and "cudaStreamSynchronize(stream)" in snap,
         "the serving clear must use the private non-blocking copy stream with a checked drain")
    need("layout(st, g, upto, index, whole, ignored)" in snap,
         "a failed extent computation must skip canonicalization rather than clear from offset 0")

if problems:
    print("RAW INT8 restore clear source check FAILED:")
    for why in problems:
        print("  - " + why)
    sys.exit(1)
print("RAW INT8 restore clear source check PASS (ordering/shape only; no CUDA behavior proven)")
