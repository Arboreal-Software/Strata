# Source extraction equality only; invoked solely inside future granted fixture.
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2]
spec=json.loads((Path(__file__).resolve().parent/'extraction-spec.json').read_text())
for entry in spec:
    source=(root/entry['source']).read_text()
    start=source.index(entry['start'])
    end=source.index(entry['end'],start)
    actual=source[start:end]
    assert hashlib.sha256(actual.encode()).hexdigest()==entry['sha256'],entry['source']
    assert (root/entry['output']).read_text()==actual,entry['output']
print('actual raw bookkeeping/bootstrap/admission and gate source equality PASS')
