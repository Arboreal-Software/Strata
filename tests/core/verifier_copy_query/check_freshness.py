"""Refuse stale extracted methods or a changed terminal-latch declaration."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def extract(text, signature):
    if text.count(signature) != 1:
        raise SystemExit("missing/ambiguous method: " + signature)
    begin = text.index(signature)
    cursor = text.index("{", begin) + 1
    depth = 1
    # Match the same complete-method extraction used for this fixture. Any source
    # change (including comments/whitespace) must be explicitly reconciled.
    while depth:
        if cursor >= len(text):
            raise SystemExit("unterminated method: " + signature)
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    return text[begin:cursor] + "\n"


verify = (ROOT / "src/core/verify.cpp").read_text()
drain = (ROOT / "src/core/verify_drain.cpp").read_text()
actual = extract(verify, "bool Verifier::batch_launch(")
actual += extract(verify, "int Verifier::batch_poll(")
actual += extract(drain, "Verifier::SnapshotDrain Verifier::poll_snapshot_drain(")
if actual != (HERE / "fixed-methods.inc").read_text():
    raise SystemExit("stale verifier copy-query methods; reconcile actual full methods and fixtures")
header = (ROOT / "include/strata/core/verify.hpp").read_text()
pattern = r"^\s*bool\s+batch_copy_failed_\s*=\s*false\s*;\s*$"
if len(re.findall(pattern, header, re.MULTILINE)) != 1:
    raise SystemExit("terminal header latch is missing, duplicated or no longer initialized false")
# The mock must model the same declared latch, and wrappers must use the checked
# methods rather than embedding a second independently maintained implementation.
if "bool batch_copy_failed_=false;" not in (HERE / "prefix.inc").read_text():
    raise SystemExit("mock terminal latch does not match actual declaration")
for mode in ("before", "fixed"):
    expected = '#include "prefix.inc"\n#include "' + mode + '-methods.inc"\n#include "cases.inc"\n'
    if (HERE / (mode + ".cpp")).read_text() != expected:
        raise SystemExit("unexpected fixture wrapper: " + mode)
print("verifier full-method and terminal-latch freshness passed")
