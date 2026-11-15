from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
a=s.index('constexpr int kLiveMax = 16;')
b=s.index('// #649:',a)
d=s[s.index('Verifier::~Verifier()'):s.index('bool Verifier::init(')]
assert d.index('g_verifier_readers.retire(this)') < d.index('if (cs_)')
assert 'g_diag_verifier' not in s and 'g_live[' not in s
init=s[s.index('bool Verifier::init('):]
assert init.index('g_verifier_readers.publish(this)') < init.index('device_ =')
Path(sys.argv[2]).write_text(s[a:b])
