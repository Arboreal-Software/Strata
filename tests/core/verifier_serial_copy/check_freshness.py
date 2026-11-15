from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[3]
d=Path(__file__).resolve().parent
s=(root/"src/core/verify.cpp").read_text()
h=(root/"include/strata/core/verify.hpp").read_text()
def block(s,start,end): return s[s.index(start):s.index(end,s.index(start))]
def extracts(s,fixed):
 run=s[s.index('bool Verifier::run(int T,'):]
 chain=block(run,'        const cudaError_t se = cudaStreamSynchronize(cs_);','        if (prof_on_ && G == 1) collect_profile();')
 terminal=block(run, '    progress_at("verify window: waiting for the expert copies"' if fixed else '    commit_pending_ = false;\n    progress_at("verify window: waiting for the expert copies"', '    if (h_bal_ != nullptr && !all_resident_')
 helper=block(s,'bool Verifier::finish_serial_copies(', 'bool Verifier::run(int T,') if fixed else ''
 return helper+'bool Verifier::terminal(std::string& err) {\n    if (refuse_copy_query_failure(err)) return false;\n'+terminal+'    return true;\n}\n'+'bool Verifier::chain(std::string& err) {\n    if (refuse_copy_query_failure(err)) return false;\n'+chain+'    return true;\n}\n'

assert (d/"fixed.inc").read_text()==extracts(s,True), "serial method/callsite stale"
assert (d/"guard.inc").read_text()==block(h,"    bool refuse_copy_query_failure(", "\n    bool capture_commit("), "guard stale"
for n,v in json.loads((d/"source-pins.json").read_text()).items():
 assert hashlib.sha256((root/n).read_bytes()).hexdigest()==v, "source pin changed: "+n
assert "std::atomic<bool> terminal_copy_query_failed_{false};" in h

for marker in ['bool Verifier::init(', 'bool Verifier::run(', 'bool Verifier::batch_launch(']:
    body=s[s.index(marker):]
    assert 'if (refuse_copy_query_failure(err)) return false;' in body[:450], marker+' lost terminal refusal'
drain=(root/'src/core/verify_drain.cpp').read_text()
assert 'bool Verifier::wait_commit(std::string& err) {\n    if (refuse_copy_query_failure(err)) return false;' in drain
assert 'Verifier::SnapshotDrain Verifier::poll_snapshot_drain(std::string& err) {\n    if (refuse_copy_query_failure(err)) return SnapshotDrain::unhealthy;' in drain

print("serial copy exact-source freshness PASS")

header_text=(root if "root" in globals() else ROOT)/"include/strata/core/verify.hpp"
hook_header=header_text.read_text()
a=hook_header.index("    void latch_copy_query_failure()")
b=hook_header.index("\n    bool strict_retirement_",a)
assert (Path(__file__).resolve().parent/"completion-latch.inc").read_text()==hook_header[a:b]+"\n", "stale completion latch hook"
