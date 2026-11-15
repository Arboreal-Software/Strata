import json,pathlib,sys
p=pathlib.Path(__file__).resolve().parent
actual=json.loads(pathlib.Path(sys.argv[1]).read_text())
expected=json.loads((p/'test-plan.json').read_text())
assert [t['name'] for t in actual['tests']]==expected['tests']
