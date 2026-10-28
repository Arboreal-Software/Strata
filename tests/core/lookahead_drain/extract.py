from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def pieces():
 h=(ROOT/'include/strata/core/expert_source.hpp').read_text()
 c=(ROOT/'src/core/expert_source.cpp').read_text()
 result = {
  'class.inc':h[h.index('class RouterLookahead {'):h.index('\n/// Plan v0.3 P6:',h.index('class RouterLookahead {'))],
  'methods.inc':c[c.index('namespace {\n// Detect same-thread lifecycle'):c.index('void RouterLookahead::enable_stats(')],
  'worker.inc':c[c.index('void RouterLookahead::run()'):c.index('void FileExpertSource::begin_layer(')],
 }
 assert result['class.inc'].count('std::condition_variable cv_;')==1
 result['class-test.inc']=result['class.inc'].replace('std::condition_variable cv_;','WakeCv cv_;')
 methods=result['methods.inc'];a=methods.index('void RouterLookahead::submit(')
 assert methods[a:].count('cv_.notify_all();')==1
 result['methods-notify-one.inc']=methods[:a]+methods[a:].replace('cv_.notify_all();','cv_.notify_one();')
 assert methods[a:].count('submitted_drain_ || ')==1
 result['methods-no-submit-seal.inc']=methods[:a]+methods[a:].replace('submitted_drain_ || ','',1)
 return result
if __name__=='__main__':
 for name,text in pieces().items():
  if sys.argv[1:]==['--prepare']:(HERE/name).write_text(text)
  elif (HERE/name).read_text()!=text:raise SystemExit('stale actual-source extraction: '+name)
 if sys.argv[1:]!=['--prepare']:print('PASS actual RouterLookahead class/lifecycle/submit/drain/worker freshness')
