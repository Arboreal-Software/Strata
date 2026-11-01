#!/usr/bin/env python3
# Source extraction only; running this does not compile or execute the fixture.
from pathlib import Path
import argparse, hashlib, json
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('output',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args();
if not a.check:a.output.mkdir(parents=True,exist_ok=True)
def emit(path,text):
 if a.check:
  if not path.is_file() or path.read_text()!=text:raise SystemExit('stale actual selector extraction: '+str(path))
 else:path.write_text(text)
def function(text,signature):
 start=text.index(signature);brace=text.index('{',start);depth=1;i=brace+1
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[start:i]
header=a.repo/'include/strata/kernels/native_rope.hpp'
for label,name in [('cuda','src/kernels/cuda/native_rope.cu'),('sycl','sycl/src/kernels/cuda/native_rope.dp.cpp')]:
 src=a.repo/name;text=src.read_text();bodies=[function(text,s) for s in ['bool rope_table_enabled()', 'void native_rope_set_enabled(bool value)', 'bool native_rope_enabled()', 'bool native_norm_rope_usable(int head_dim, int n_rot)', 'NativeRopePolicyObservation native_rope_policy_observation(int head_dim, int n_rot)']]
 # getenv is the ONLY substitution: no host environment inspection. Selector
 # arithmetic/defaults/caching and producer bodies remain the actual source.
 joined='\n'.join(bodies).replace('std::getenv(', 'fake_getenv(')
 emit(a.output/(label+'.inc'),joined+'\n')
 emit(a.output/(label+'-pins.json'),json.dumps({name:hashlib.sha256(src.read_bytes()).hexdigest(),'include/strata/kernels/native_rope.hpp':hashlib.sha256(header.read_bytes()).hexdigest(),'extracted_sha256':hashlib.sha256((joined+'\n').encode()).hexdigest()},indent=2)+'\n')

# Generator is intentionally NOT run: it invokes toolchain/link inspection.
# Pin its exact rule and require the corresponding generated stub expression.
generator=a.repo/'metal/tools/gen_stubs.py'
stubs=a.repo/'metal/src/engine/stubs.cpp'
rule='body = "return {false, p0, p1, false, false, false};"'
assert rule in generator.read_text()
assert '[t for _, t in params] != ["int", "int"]' in generator.read_text()
assert 'native RoPE policy observation signature changed' in generator.read_text()
stub='NativeRopePolicyObservation native_rope_policy_observation(int p0, int p1) { return {false, p0, p1, false, false, false}; }'
assert stub in stubs.read_text()
emit(a.output/'metal-pins.json',json.dumps({str(p.relative_to(a.repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [generator,stubs]},indent=2)+'\n')
