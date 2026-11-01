#!/usr/bin/env python3
"""Execute exact production MAIN cancellation settlement with CPU recording boundaries.
No GPU/tensor/reset correctness or cancellation latency claim.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def take(text):
 start=text.index('            auto settle_owned_admission_cancel =')
 opening=text.index('{',start);depth=0
 for at in range(opening,len(text)):
  if text[at]=='{':depth+=1
  elif text[at]=='}':
   depth-=1
   if not depth:
    if text[at+1]!=';':raise ValueError('Actual settlement lambda terminator changed')
    return text[start:at+2]+'\n'
 raise ValueError('Actual settlement lambda missing')

def run(build):
 build=Path(build);build.mkdir(parents=True,exist_ok=True)
 actual=take((ROOT/'src/program/generate.cpp').read_text())
 template=(HERE/'owned_admission_settlement_fixture.cpp.in').read_text()
 (build/'settlement.after.inc').write_text(actual)
 cpp=build/'actual-owned-admission-settlement.cpp';cpp.write_text(template)
 binary=build/'actual-owned-admission-settlement'
 command=['c++','-std=c++20','-O0','-Wall','-Wextra',str(cpp),'-o',str(binary)]
 subprocess.run(command,check=True,timeout=60)
 subprocess.run([str(binary)],check=True,timeout=15)
 print('PASS actual owned MAIN reset/drain/close certificate; default-off/failures/no-fence remain silent',flush=True)
 # Negative control: an early report must be caught on a genuine failed reset boundary.
 start=actual.index('                if(strata::program::http2_receipts_enabled())')
 end=actual.index('                return true;',start)
 diagnostic=actual[start:end]
 anchor='                if(!owned_runtime->reset_sessions(g,cancel_error))'
 if actual.count(anchor)!=1:raise ValueError('Actual reset boundary changed')
 mutant=actual[:start]+actual[end:]
 mutant=mutant.replace(anchor,diagnostic+anchor,1)
 (build/'settlement.after.inc').write_text(mutant)
 subprocess.run(command,check=True,timeout=60)
 result=subprocess.run([str(binary)],capture_output=True,timeout=15)
 if result.returncode==0:raise ValueError('Early-reset diagnostic mutant escaped actual failure test')
 (build/'settlement.after.inc').write_text(actual)
 print('PASS diagnostic-before-actual-reset negative control',flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path);args=parser.parse_args()
 if args.build_dir:run(args.build_dir)
 else:
  with tempfile.TemporaryDirectory(prefix='strata-owned-settlement-')as directory:run(Path(directory))
if __name__=='__main__':main()
