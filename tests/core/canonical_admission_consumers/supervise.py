"""Require the exact intended failure; crashes/timeouts are never passing controls."""
import subprocess
import sys

mode, executable = sys.argv[1:]
expected = {
    "range": "FAIL range rejects reboxed metadata\n",
    "batch": "FAIL batch physical/metadata rebox refusal and canonical positive\n",
    "pool": "FAIL pool owner/range rebox refusal and canonical positive\n",
}
try:
    result = subprocess.run([executable], capture_output=True, text=True, timeout=10)
except subprocess.TimeoutExpired:
    raise SystemExit("canonical consumer fixture timed out")
if mode == "fixed":
    valid = (result.returncode == 0 and not result.stderr and result.stdout ==
             "PASS canonical range/batch/owned-pool host admissions (geometry/backend seams)\n")
else:
    valid = (result.returncode == 70 and not result.stdout and result.stderr == expected[mode])
if not valid:
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(f"unexpected {mode} result: exit {result.returncode}")
print(f"{mode}: exact expected outcome")
