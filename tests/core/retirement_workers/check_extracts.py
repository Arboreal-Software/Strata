"""Fail closed on stale actual-method extracts or fixture composition; never writes."""
from pathlib import Path
import argparse

def extract(text, marker):
    start = text.index(marker)
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        if text[end] == '{': depth += 1
        elif text[end] == '}': depth -= 1
        end += 1
    return text[start:end]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_root', type=Path)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parent
    def read(name): return (fixture / name).read_text()
    def same(actual, expected, name):
        if actual != expected: raise SystemExit('stale retirement worker fixture: ' + name)
    cpp = (args.source_root/'src/core/expert_source.cpp').read_text()
    header = (args.source_root/'include/strata/core/expert_source.hpp').read_text()
    a=header.index('class RouterLookahead {'); b=header.index('\n};',a)+3
    router=header[a:b]
    same(read('workers/actual-router.inc'),router+'\n','Router declaration')
    same(read('workers/foresight_swap.hpp'),(args.source_root/'include/strata/core/foresight_swap.hpp').read_text(),'Foresight declaration')
    markers=['bool RouterLookahead::start(', 'bool RouterLookahead::close_for_retirement()', 'void RouterLookahead::set_foresight(', 'void RouterLookahead::submit(', 'void RouterLookahead::enable_stats(', 'bool ForesightSwap::set_strict_retirement(', 'bool ForesightSwap::close_for_retirement()', 'bool ForesightSwap::take(', 'void ForesightSwap::hold(', 'void ForesightSwap::unhold(', 'void ForesightSwap::note_miss(', 'void ForesightSwap::predict(', 'void ForesightSwap::request_locked(']
    a=cpp.index('namespace {\n// Detect same-thread lifecycle/source callback reentry'); b=cpp.index('} // namespace',a)+len('} // namespace')
    methods=cpp[a:b]+'\n\n'+'\n\n'.join(extract(cpp,m) for m in markers)
    same(read('workers/actual-methods.inc'),methods+'\n','Router/Foresight methods')
    same(read('workers/test.cpp'),read('workers/prefix.inc')+router+'\n'+'thread_local ForesightSwap::ActiveRequest* ForesightSwap::active_request_ = nullptr;\n'+methods+'\n'+read('workers/cases.inc'),'worker composition')
    a=cpp.index('namespace {\nstruct FileBackgroundStopFrame'); b=cpp.index('\nFileExpertSource::~FileExpertSource()',a)
    actual=cpp[a:b]
    same(read('file/actual-method.inc'),actual,'File methods')
    same(read('file/test.cpp'),read('file/prefix.inc')+actual+read('file/cases.inc'),'File composition')
    a=header.index('    std::mutex background_close_mu_;'); b=header.index('    void io_stop();',a)
    same(read('file/actual-control-fields.inc'),header[a:b],'File control fields')
    pool=(args.source_root/'src/kernels/cpu/pool.cpp').read_text()
    # Full contiguous blocks include namespaces/comments and all guard/wrapper code.
    for name in ['actual-methods.inc','actual-wrappers.inc','actual-destructor.inc']:
        if read('pool/'+name).strip() not in pool:
            raise SystemExit('stale retirement pool extract: '+name)
    same(read('pool/test.cpp'),read('pool/harness-before.inc')+read('pool/harness-after.inc'),'pool composition')
    header=(args.source_root/'include/strata/kernels/cpu/pool.hpp').read_text()
    a=header.index('    const bool checked_retirement_;');b=header.index('    int phase_tasks(',a)
    same(read('pool/actual-control-fields.inc'),header[a:b],'pool control fields')
    if 'int tasks = 0, bool checked_retirement = false);' not in header:
        raise SystemExit('stale pool default constructor contract')
    if 'int tasks, bool checked_retirement)\n    : checked_retirement_(checked_retirement), tasks_(tasks)' not in pool:
        raise SystemExit('stale pool construction mode initialization')
    program=(args.source_root/'src/program/generate.cpp').read_text()
    a=program.index('    DriverRuntimeObject<strata::kernels::cpu::ExpertPool> pool_storage(')
    b=program.index('    auto& pool=pool_storage.get();',a)
    same(read('pool/actual-owned-construction.inc'),program[a:b],'owned pool construction')
    driver=extract(program,'bool DriverWindowRuntime::close_background_after_callers_joined()')
    same(read('driver/actual-method.inc'),driver+'\n','driver helper')
    same(read('driver/test.cpp'),read('driver/prefix.inc')+driver+'\n'+read('driver/cases.inc'),'driver composition')

if __name__ == '__main__': main()
