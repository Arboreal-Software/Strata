#!/usr/bin/env python3
"""Actual binary preflight regressions. No model or GPU access is required.

Run only inside the granted CPU build container. This does not exercise the
runtime controller, GPU capture/restore, or generated output.
"""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--window-state', choices=['on', 'off'], required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    env = {key: os.environ[key] for key in ('PATH', 'LD_LIBRARY_PATH') if key in os.environ}
    env.update(STRATA_LOOKAHEAD='0', NVIDIA_VISIBLE_DEVICES='void', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1')
    cases = 0
    with tempfile.TemporaryDirectory(prefix='suffix-cli-') as directory:
        cwd = Path(directory)
        fifo = cwd / 'unopened-token-input'
        os.mkfifo(fifo)
        common = ['--suffix-reuse-experiment', str(cwd / 'unread-manifest'),
                  '--suffix-reuse-host-bytes', '67108864', '--owned-session-lifetime',
                  '--pipeline-windows', '0', '--spec', '0', '--suffix-draft', '0',
                  '--lookup-chain', '0', '--adapt-every', '0', '--max-context', '64',
                  '--max-new', '1', '--prefill', '4']

        def check(name, command, code, expected, extra_env=None, timeout_seconds=5):
            nonlocal cases
            child_env = dict(env)
            child_env.update(extra_env or {})
            # Independent file caps prevent an unexpected child diagnostic loop
            # from filling memory or disk; exact return-code rejects crashes.
            with tempfile.TemporaryFile(dir=cwd) as stdout, tempfile.TemporaryFile(dir=cwd) as stderr:
                try:
                    completed = subprocess.run([str(binary), *command], cwd=cwd, env=child_env,
                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=timeout_seconds,
                        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20)))
                except subprocess.TimeoutExpired as error:
                    # run() has killed and reaped the child. Preserve bounded
                    # diagnostics before TemporaryFile removes its evidence.
                    stdout.seek(0); stderr.seek(0)
                    out, err = stdout.read(131073), stderr.read(131073)
                    raise RuntimeError(
                        f'{name}: timeout={error.timeout}; '
                        f'stdout_first_131073={out!r}; stderr_first_131073={err!r}') from error
                stdout.seek(0); stderr.seek(0)
                out, err = stdout.read(131073), stderr.read(131073)
            if len(out) > 131072 or len(err) > 131072:
                raise RuntimeError(name + ': unexpected output volume')
            if completed.returncode != code or expected.encode() not in out + err:
                raise RuntimeError(f'{name}: exit={completed.returncode}; stdout={out!r}; stderr={err!r}')
            print('PASS', name, flush=True)
            cases += 1

        # A cold CLI process was observed exiting successfully after6.13s,
        # mostly in filesystem page waits. This is a bounded startup allowance,
        # not a CLI latency SLO. Keep conflict/FIFO cases at5s to catch blocking.
        check('help exposes opt-in experiment', ['--help'], 0,
              '--suffix-reuse-experiment', timeout_seconds=20)
        if args.window_state == 'off':
            check('unsupported build refuses before input', common, 2,
                  'suffix experiment first target needs Linux CUDA window-state build')
        else:
            refusal = 'incompatible standalone suffix experiment profile/input/bounds'
            # If an ordinary file were opened while scanning flags, the first
            # command would block on this FIFO. Both argument orders must refuse.
            check('ordinary file before experiment never opens', ['--tokens-file', str(fifo), *common], 2, refusal)
            check('ordinary file after experiment never opens', [*common, '--tokens-file', str(fifo)], 2, refusal)
            check('ordinary inline tokens conflict', [*common, '--tokens', '1,2'], 2, refusal)
            check('serve conflicts', [*common, '--serve'], 2, refusal)
            check('missing explicit context', common[:common.index('--max-context')] + common[common.index('--max-context')+2:], 2, refusal)
            for variable in ('STRATA_DBG_NAN', 'STRATA_TAP_FILE', 'STRATA_NAN_INJECT',
                             'STRATA_VERIFY_PROFILE', 'STRATA_VERIFY_NODES', 'STRATA_COMMIT_TIMING'):
                # Even "0" is present and can activate presence-based diagnostics.
                check('diagnostic refused ' + variable, common, 2,
                      'suffix experiment requires ' + variable + ' unset', {variable: '0'})
            check('duplicate cap refused', [*common, '--suffix-reuse-host-bytes', '1'], 2,
                  'suffix RAW host cap needs one positive bounded integer')
            check('zero cap refused', ['--suffix-reuse-host-bytes', '0'], 2,
                  'suffix RAW host cap needs one positive bounded integer')
        print(f'RESULT suffix CLI preflight PASS cases={cases} window_state={args.window_state}', flush=True)


if __name__ == '__main__':
    main()
