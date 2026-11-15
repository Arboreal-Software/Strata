from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("driver")
parser.add_argument("fixtures")
args=parser.parse_args()
p=Path(args.fixtures)
s=Path(args.driver).read_text()
a=s.index('        auto pump_step = ');a=s.index('\n',a)+1;b=s.index('            ++pipe_tick;',a)
prefix=s[a:b]
if prefix!=(p/'prefix.inc').read_text():raise SystemExit('stale actual pump admission prefix')
a=s.index('        bool pipe_completion_uncertain = false;');b=s.index('\n#if defined(STRATA_WINDOW_STATE_AVAILABLE)',s.index('        auto pipe_drain = ',a))
if s[a:b]+'\n'!=(p/'controller.inc').read_text():raise SystemExit('stale actual controller wrapper/drain')
a=prefix.index('#if defined(STRATA_WINDOW_STATE_AVAILABLE)');b=prefix.index('#endif',a)+len('#endif')
if prefix[:a]+prefix[b:]!=(p/'old-prefix.inc').read_text():raise SystemExit('unexpected old-prefix delta')
