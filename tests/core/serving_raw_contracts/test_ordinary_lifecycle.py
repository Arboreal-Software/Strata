#!/usr/bin/env python3
"""Call-site ordering guard for the ordinary request's own RAW work fence (slice 7).

The CPU metadata/lifecycle gate (test.cpp) proves the PREDICATES the serving path calls.  It cannot see WHERE the
serving path calls them, and the whole point of slice 7 is the order: the work phase must be opened BEFORE the
request's first native mutation (slot hand-off, parked/borrowed restore, zeroed or checkpointed start, prompt read,
decode) and drained only AFTER the last real all-stage commit.  This gate reads the real serving entry and pins
that order, plus the two cleanup rules that keep a cancelled or nonfinite request from either faking a drain or
unwinding an open fence.

It proves no device behaviour and no certificate: it is a source-order guard over one file, nothing more.
"""
import hashlib
import sys
import tempfile
from test_ordinary_admission import run as run_admission
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / 'src' / 'program' / 'generate.cpp'

# The fence this request opens for its own work, and the point where it certifies that work.  These are the
# diagnostic lines the serving entry actually emits, so a comment mentioning a marker cannot satisfy the guard.
FENCE_OPEN = 'ORDINARY_FENCE_OPEN req=%llu "'
FENCE_ADMIT = 'const bool ordinary_fence_admissible ='
CERTIFY = 'retained): ORDINARY_CERTIFY "'
RESET_CERTIFY = 'retained): ORDINARY_RESET_CERTIFY "'
CLEARED = 'retained): ORDINARY_TRACK_CLEARED "'
GUARD_UNWIND = 'slice 7: the request unwound while its OWN work fence was still open'

# Every native mutation the request can perform, in the order the serving entry has them.
MUTATIONS = (
    'if (!copy_from_slot(slot_source, slot_ck, err))',                    # a slot gives its conversation back
    'conversation_snapshot_restore_prefix(*donor, parked.tokens',         # a borrowed parked prefix
    'strata::core::conversation_snapshot_restore(*incoming, ss, g,',      # a parked conversation restored whole
    'if(!owned_runtime->reset_sessions(g,err))',                          # the zeroed start
    '!checkpoint_restore(*c, ss, g)',                                     # a checkpointed start
)

# The last real all-stage completion of THIS request, and the conversation bookkeeping that follows it.  The
# comment is the unique marker: the same `wait_commit` call text also appears in the conversation-cache, pipeline
# and non-serving paths, and pinning one of those would make the ordering checks vacuous.
COMMIT = ('// the last commit (set_commit_async): the session is complete before anything reads or copies it')
COMMIT_CALL = 'if (!ver.wait_commit(err))'
LIVE_SWAP = 'live.swap(consumed);'
DECODE_LOOP = 'while (!pl_ran && !cancelled && produced_n < max_new)'
READ_LOOP_CANCEL = 'serving_read_gave_way = true;'

# The slice 7 disposition, read by the downstream live/pending bookkeeping and the batch-slot hand-off.
DISCARD_GUARDS = (') && !serving_session_discarded) {',
                  '} else if (!serving_session_discarded && !nf_hit && pp_exact',
                  'bool cont = !cancelled && !serving_session_discarded')


def positions(source: str) -> dict:
    # `str.rindex` on purpose: the request-tail commit and the live/pending adoption are each written once in the
    # serving entry but the same call also appears in the conversation-cache paths earlier in the file.  Pinning
    # the last occurrence keeps the guard from passing on an unrelated earlier match, and the occurrence counts
    # below make a duplicated marker visible instead of silently choosing one.
    for marker, count in ((COMMIT, 1), (LIVE_SWAP, 1), (FENCE_OPEN, 1), (CERTIFY, 1), (RESET_CERTIFY, 1),
                          (CLEARED, 1), (DECODE_LOOP, 1)):
        found = source.count(marker)
        if found != count:
            raise ValueError(f'expected {count} occurrences of {marker!r}, found {found}')
    marks = {'fence_admit': source.index(FENCE_ADMIT), 'fence_open': source.rindex(FENCE_OPEN),
             'certify': source.rindex(CERTIFY), 'reset_certify': source.rindex(RESET_CERTIFY),
             'cleared': source.rindex(CLEARED), 'guard_unwind': source.index(GUARD_UNWIND),
             'commit': source.rindex(COMMIT),
             'commit_call': source.index(COMMIT_CALL, source.rindex(COMMIT)),
             'live_swap': source.rindex(LIVE_SWAP),
             'decode': source.rindex(DECODE_LOOP)}
    for name, marker in (('mutation_' + str(i), marker) for i, marker in enumerate(MUTATIONS)):
        marks[name] = source.rindex(marker)
    marks['read_cancel'] = source.rindex(READ_LOOP_CANCEL)
    return marks


def main() -> int:
    text = SOURCE.read_text()
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    try:
        at = positions(text)
    except ValueError as missing:
        print(f'FAIL: a slice 7 or serving mutation marker is missing from the real serving entry: {missing}')
        return 1
    checks = []
    # 1. the fence opens before EVERY native mutation the request can do, and before the read and decode.
    for name, position in at.items():
        if name.startswith('mutation_'):
            checks.append((at['fence_open'] < position, f'{name} runs after the fence opened'))
    checks.append((at['fence_open'] < at['read_cancel'], 'the prompt read runs after the fence opened'))
    checks.append((at['fence_open'] < at['decode'], 'the decode runs after the fence opened'))
    # 2. the drain that certifies comes only AFTER the last all-stage commit, and the certificate is recorded
    #    before the conversation bookkeeping adopts the same run.
    checks.append((at['commit_call'] < at['certify'], 'the certification follows the request-tail real commit call'))
    checks.append((at['commit'] < at['certify'], 'the certification follows the request-tail commit marker'))
    checks.append((at['certify'] < at['live_swap'], 'the certificate is settled before the conversation adopts it'))
    # 3. the undescribable-state answer is a deliberate reset drained to the empty certificate, never a silent
    #    drop and never a drain of work that never completed.
    checks.append((at['reset_certify'] > at['commit'], 'the reset certification also follows the real commit'))
    checks.append((at['cleared'] > at['commit'], 'a cleared certificate is stated at the completion point'))
    # 3b. the disposition of a discarded (reset) session reaches the conversation bookkeeping that follows.
    for marker in DISCARD_GUARDS:
        checks.append((marker in text and text.rindex(COMMIT) < text.rindex(marker),
                       f'the discard disposition guards the downstream bookkeeping ({marker!r})'))
    # 4. unwinding a request scope with the fence still open is the failstop, not a cache reset.
    checks.append((at['guard_unwind'] < at['fence_open'],
                   'the request-scope guard is declared before the fence it failstops on'))
    failed = [what for ok, what in checks if not ok]
    print(f'src/program/generate.cpp sha256={digest}')
    for ok, what in checks:
        print(('PASS ' if ok else 'FAIL ') + what)
    if failed:
        print(f'ordinary RAW fence call-site ordering gate FAILED: {len(failed)}')
        return 1
    print('ordinary RAW fence call-site ordering gate PASS (fence before every mutation, drain after the real '
          'commit, reset-or-clear on an undescribable state; source order only, no device behaviour)')
    if '--source-only' not in sys.argv:
        build_dir = None
        if '--build-dir' in sys.argv:
            build_dir = Path(sys.argv[sys.argv.index('--build-dir') + 1])
            build_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='strata-ordinary-admission-', dir=build_dir) as directory:
            run_admission(directory)
    return 0


if __name__ == '__main__':
    sys.exit(main())
