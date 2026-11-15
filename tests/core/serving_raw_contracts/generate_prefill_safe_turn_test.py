"""Generate actual helper/caller source; does not invoke compiler or device APIs."""
import argparse,hashlib
from pathlib import Path
def block(s,marker):
 a=s.index(marker);b=s.index('{',a);i=b+1;depth=1
 while depth:depth+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[a:i]
def generate(root):
 s=(root/'src/program/generate.cpp').read_text();a=s.index('            std::shared_ptr<const void> prefill_turn_credit;');b=s.index('\n#endif',a)
 helper=s[a:b];fallback=block(s,'if(!turned)')
 if 'prefill_turn_unavailable'not in fallback or 'choose_and_service(false)'not in fallback:raise ValueError('Actual ordinary foreground fallback absent')
 here=Path(__file__).resolve().parent
 return (here/'prefill_safe_turn_prelude.inc').read_text()+'\n'+(here/'prefill_safe_turn_driver.inc').read_text().replace('// ACTUAL_HELPER',helper).replace('// ACTUAL_FALLBACK',fallback)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--generate',type=Path,required=True);a=p.parse_args();a.generate.parent.mkdir(parents=True,exist_ok=True);a.generate.write_text(generate(a.source_root.resolve()))
