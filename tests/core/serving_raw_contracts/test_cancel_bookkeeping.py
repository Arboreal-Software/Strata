#!/usr/bin/env python3
"""Execute actual serving live/pending bookkeeping and actual next REQ_BEGIN.

CPU control-flow proof only: no native state, real drain or exactness is simulated.
The original-source negative control removes only the new branch guard and must
reproduce loss of the certified pending token after cancelled retained cleanup.
The discard negative control removes only the slice 7 reset/discard guards and must reproduce a discarded
(zeroed) session having its consumed run, a partial prompt prefix or a pending token resurrected - for BOTH a
nonfinite hit and a mid-prefill cancellation.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

PRELUDE = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc,char** argv){
    if(argc!=2)return 10;
    const std::string name=argv[1];
    struct Options {int prompt_cache=6;bool suffix_cache=true;} o;
    std::vector<int32_t> live={99},consumed={1,2,3},cur={1,2,3,7,8,9};
    std::vector<int> live_imgs,req_imgs;
    auto imgs_below=[](const std::vector<int>& images,int64_t){return images;};
    int32_t last_sampled=7,serving_pending_token=-1;
    bool cancelled=true,nf_hit=false,pp_exact=true,req_ckpt=true,live_ok=false;
    [[maybe_unused]] bool retained_exact_cleanup_completed=true;
    // slice 7: the disposition a request sets when it deliberately returned the session to a reset (zeroed)
    // state because the state it had reached was not describable.  The fixture states the effect that block has
    // already had on the conversation metadata; what is under test is that nothing downstream undoes it.
    [[maybe_unused]] bool serving_session_discarded=false;
    int64_t pp_reached=3;uint64_t serving_request_ordinal=2;
    if(name=="generic_cancel")retained_exact_cleanup_completed=false;
    else if(name=="normal"){cancelled=false;retained_exact_cleanup_completed=false;}
    else if(name=="nonfinite")nf_hit=true;
    else if(name=="uncached")req_ckpt=false;
    else if(name=="nonfinite_discard"){nf_hit=true;serving_session_discarded=true;live.clear();serving_pending_token=-1;}
    else if(name=="prefill_cancel_discard"){serving_session_discarded=true;live.clear();serving_pending_token=-1;}
    else if(name!="retained_cancel")return 11;
'''
EPILOGUE = r'''
    if(name=="nonfinite_discard"||name=="prefill_cancel_discard"){
        // A discarded session adopts nothing: no consumed run, no partial prompt prefix, no pending token, and
        // live_ok stays false.  Resurrecting any of these would let the next request continue from, park, or
        // import a device state that was deliberately zeroed.
        if(!live.empty())return 24;
        if(serving_pending_token!=-1)return 25;
        if(live_ok)return 26;
    }else{
    const bool preserve=name!="generic_cancel";
    if(live!=std::vector<int32_t>({1,2,3}))return 21;
    if(serving_pending_token!=(preserve?7:-1))return 22;
    if(live_ok!=(name!="nonfinite"&&name!="uncached"))return 23;
    }
    // Exact next-request diagnostics are extracted from the real serving entry.
NEXT_REQUEST
    std::puts("PASS actual serving bookkeeping and next REQ_BEGIN; CPU seam only");return 0;
}
'''


def extract(source):
    begin=source.index('            if ((!cancelled\n#if defined(STRATA_SERVING_SUFFIX_REPAIR_AVAILABLE)\n                || retained_exact_cleanup_completed')
    end=source.index('            static const bool state_hash',begin)
    block=source[begin:end]
    begin=source.index('            ++serving_request_ordinal;')
    end=source.index('            const strata::core::SuffixCacheFlags',begin)
    request=source[begin:end]
    if 'pending=%d' not in request or 'serving_pending_token' not in request:
        raise RuntimeError('actual next-request pending diagnostic changed')
    return block,request


# Negative control for the slice 7 disposition: remove ONLY the two discard guards and the discard case must
# reproduce the resurrection it prevents (a nonfinite hit adopting the consumed run, or a cancelled prefill
# adopting a partial prompt prefix with live_ok true).
DISCARD_NEGATIVE=(
    (') && !serving_session_discarded) {', ') && !(serving_session_discarded && false)) {'),
    ('} else if (!serving_session_discarded && !nf_hit && pp_exact',
     '} else if (!(serving_session_discarded && false) && !nf_hit && pp_exact'),
)


def run(build):
    root=Path(__file__).resolve().parents[3]
    source_path=root/'src/program/generate.cpp'
    source=source_path.read_text();before=hashlib.sha256(source_path.read_bytes()).hexdigest()
    block,request=extract(source)
    negative=block.replace('                || retained_exact_cleanup_completed','                || false /* original: cancelled retained cleanup takes generic branch */',1)
    discard_negative=block
    for wanted,replacement in DISCARD_NEGATIVE:
        if wanted not in discard_negative:
            raise RuntimeError('actual discard disposition guard changed: '+wanted)
        discard_negative=discard_negative.replace(wanted,replacement,1)
    for name,actual in [('current',block),('original_negative',negative),
                        ('discard_negative',discard_negative)]:
        code=PRELUDE+actual+EPILOGUE.replace('NEXT_REQUEST',request)
        path=build/(name+'.cpp');binary=build/name;path.write_text(code)
        subprocess.run(['c++','-std=c++20','-O2','-Wall','-Wextra','-Wconversion','-Werror',
                        '-DSTRATA_SERVING_SUFFIX_REPAIR_AVAILABLE=1',str(path),'-o',str(binary)],check=True)
        if name=='current':
            cases=['retained_cancel','generic_cancel','normal','nonfinite','uncached',
                   'nonfinite_discard','prefill_cancel_discard']
        elif name=='original_negative':
            cases=['retained_cancel']
        else:
            cases=['nonfinite_discard','prefill_cancel_discard']
        for case in cases:
            result=subprocess.run([str(binary),case],capture_output=True,text=True,timeout=10)
            if name=='original_negative':
                if result.returncode!=22:
                    raise RuntimeError(f'original pending-loss negative control did not fail at pending assertion: {result.returncode}')
                print('PASS original-source pending-loss control reproduced exit22',flush=True)
                continue
            if name=='discard_negative':
                if result.returncode not in (24,25,26):
                    raise RuntimeError(f'{case}: discard negative control did not reproduce the resurrection '
                                       f'it guards against: exit{result.returncode}')
                print(f'PASS discard negative control {case} reproduced exit{result.returncode}',flush=True)
                continue
            if result.returncode:
                raise RuntimeError(f'{case}: actual source fixture exit{result.returncode}: {result.stderr}')
            if case.endswith('_discard'):
                expected='pending=-1'
                if f'REQ_BEGIN req=3 consumed=0 {expected}' not in result.stderr:
                    raise RuntimeError(f'{case}: a discarded session still reports history or pending: {result.stderr}')
            else:
                expected='pending=-1' if case=='generic_cancel' else 'pending=7'
                if f'REQ_BEGIN req=3 consumed=3 {expected}' not in result.stderr:
                    raise RuntimeError(f'{case}: next real REQ_BEGIN lost history/pending: {result.stderr}')
            print(f'PASS {case}: {result.stderr.strip()}',flush=True)
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=before:
        raise RuntimeError('production source changed during actual-method fixture')
    print(f'generate_sha256={before} actual_block_sha256={hashlib.sha256(block.encode()).hexdigest()}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path)
    args=parser.parse_args()
    if args.build_dir:
        args.build_dir.mkdir(parents=True,exist_ok=True);run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-cancel-bookkeeping-') as directory:
            run(Path(directory))


if __name__=='__main__':
    main()
