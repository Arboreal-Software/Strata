"""Refuse a stale copied production fixture before the public target builds."""
import hashlib
import json
from pathlib import Path
import runpy
here = Path(__file__).resolve().parent
root = here.parents[2]
snapshot = here / 'snapshot'
for name, expected in json.loads((snapshot / 'source-pins.json').read_text()).items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('checked-device fixture differs from repository source: ' + name)
runpy.run_path(str(snapshot / 'fixture/check_source.py'), run_name='__main__')
