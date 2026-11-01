"""Run generated actual-helper CPU fixture; no compilation or model loading."""
import argparse,subprocess
from pathlib import Path
def run(binary):
 for mode in [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16]:
  result=subprocess.run([str(binary),str(mode)],timeout=10);expected=70 if 10<=mode<=15 else 0
  if result.returncode!=expected:raise RuntimeError(f'mode {mode}: exit{result.returncode}, expected{expected}')
 print('PASS16 actual safe-turn/helper+caller CPU cases; recording native boundaries')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);a=p.parse_args();run(a.binary.resolve(strict=True))
