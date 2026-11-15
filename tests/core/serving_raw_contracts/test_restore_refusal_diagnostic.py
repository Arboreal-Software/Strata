#!/usr/bin/env python3
"""Generate/run the exact restore diagnostic site with injected typed outcomes."""
import argparse
from pathlib import Path
import subprocess
from test_serving_thread_bridge_source import body


def actual(source, marker):
    at = source.index(marker)
    block = body(source, marker)
    return source[at:source.index(block, at)] + block


def generate(root):
    source = (root / 'src/program/serving_suffix_qualification.inc').read_text()
    driver = (root / 'src/program/suffix_experiment_driver.hpp').read_text()
    begin = source.index('const auto restore_source_cut=original.cut,restore_destination_cut=session.cut();')
    final = 'if(!restored)failstop("serving repaired restore before continuation refused");'
    end = source.index(final, begin) + len(final)
    site = source[begin:end]
    helpers = '\n'.join(actual(driver, marker) for marker in [
        '[[noreturn]] static void failstop(',
        'static void checked(',
        'template<class Operation> static SuffixRawResult raw_call(',
    ])
    return r'''
#include "strata/core/window_state.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
using strata::core::StateCut;
enum class SuffixRawStatus {ok,invalid,budget_refused,poisoned};
struct SuffixRawResult {
 SuffixRawStatus status;std::string detail;
 explicit operator bool()const{return status==SuffixRawStatus::ok;}
};
struct Session {
 StateCut state{};unsigned phase_=3;bool open=true;SuffixRawStatus outcome;
 Session(SuffixRawStatus value):outcome(value){
  state.id=180;state.forwarded=2860;state.accepted=2860;
  state.exact_frontier=2860;state.output_watermark=5;
 }
 const StateCut& cut()const{return state;}
 unsigned phase()const{return phase_;}bool request_open()const{return open;}
 SuffixRawResult restore(int,int){
  // Deliberately change adapter observations AFTER invocation: the diagnostic
  // must retain the immutable source/destination observations from before it.
  state.id=999;state.forwarded=999;phase_=6;open=false;
  switch(outcome){
   case SuffixRawStatus::invalid:return {outcome,"restore requires bounded immutable private image and idle destination"};
   case SuffixRawStatus::budget_refused:return {outcome,"aggregate suffix H exhausted"};
   case SuffixRawStatus::poisoned:return {outcome,"private RAW restore partially failed"};
   default:return {outcome,{}};
  }
 }
};
struct Driver {
''' + helpers + r'''
 static void run(Session& session){
  const int repaired=1;
  struct {StateCut cut;int key=1;} original;
  original.cut.id=7;original.cut.forwarded=2859;original.cut.accepted=2860;
  original.cut.exact_frontier=2859;original.cut.output_watermark=5;
''' + site + r'''
 }
};
int main(int argc,char** argv){
 if(argc!=2)return 2;
 const std::string mode=argv[1];
 const auto status=mode=="invalid"?SuffixRawStatus::invalid:
  mode=="budget"?SuffixRawStatus::budget_refused:
  mode=="poison"?SuffixRawStatus::poisoned:SuffixRawStatus::ok;
 Session session(status);Driver::run(session);
 std::puts("successful restore continues unchanged");return 0;
}
'''


def run(binary):
    for mode, status, reason in [
        ('invalid', 1, 'restore requires bounded immutable private image and idle destination'),
        ('budget', 2, 'aggregate suffix H exhausted'),
        ('poison', 3, 'private RAW restore partially failed'),
        ('ok', 0, ''),
    ]:
        result = subprocess.run([str(binary), mode], capture_output=True, text=True, timeout=10)
        print(f'{mode}: exit={result.returncode}\n{result.stderr}{result.stdout}', end='')
        if mode == 'ok':
            assert result.returncode == 0 and not result.stderr
            assert result.stdout == 'successful restore continues unchanged\n'
            continue
        assert result.returncode == 70
        expected = (
            'suffix_restore_refusal v=1 site=serving_repaired_continuation '
            f'status={status} source_cut=7 source_forwarded=2859 source_accepted=2860 '
            'source_exact=2859 source_watermark=5 destination_cut=180 '
            'destination_forwarded=2860 destination_accepted=2860 destination_exact=2860 '
            f'destination_watermark=5 destination_phase=3 request_open=1 detail={reason}\n'
        )
        fatal = 'poisoned RAW transaction' if mode == 'poison' else 'serving repaired restore before continuation refused'
        assert result.stderr == expected + f'suffix experiment terminal execution failure: {fatal}\n'
    print('PASS4 actual-site typed restore diagnostics; failstop/success unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--generate', type=Path)
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    if args.generate:
        args.generate.parent.mkdir(parents=True, exist_ok=True)
        args.generate.write_text(generate(args.source_root))
    elif args.binary:
        run(args.binary.resolve(strict=True))
    else:
        parser.error('--generate or --binary required')
