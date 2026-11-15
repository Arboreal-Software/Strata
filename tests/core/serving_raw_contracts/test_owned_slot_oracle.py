#!/usr/bin/env python3
"""Actual diagnostic native empty save/restore+comparison with recording host transfers.

No GPU, numerical equivalence, allocation fit, or genuine hardware failure claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent/'owned_slot_oracle'

def take(source,marker):
    first=source.index(marker);opening=source.index('{',first);depth=0
    for at in range(opening,len(source)):
        if source[at]=='{':depth+=1
        elif source[at]=='}':
            depth-=1
            if not depth:return source[first:at+1]+'\n'
    raise RuntimeError(marker)

def controls(build,generate):
    start=generate.index('            bool pending_slot_oracle=false;')
    end=generate.index('                if(!generating&&!queued.queued_input_lines',start)
    selection=generate[start:end]
    begin=generate.index('            const size_t required_peer=selected_slot==0?1:0;')
    end=generate.index('            std::array<strata::core::BundleHandle,2> other_before;',begin)
    preflight=generate[begin:end]
    terminal=take(generate,'            if(queued.input_error||queued.end_of_input){')
    assert generate.index('            if(queued.input_error||queued.end_of_input){')<generate.index('            bool pending_slot_oracle=false;')
    prelude='''#include <memory>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <charconv>
#include <cstdlib>
namespace strata::core {[[noreturn]] void retirement_failed(const char*){throw std::runtime_error("refused");}}
inline std::vector<std::string> events;
void serving_release_thread_lane(size_t,bool cancel){assert(cancel);events.push_back("thread");}
struct Raw {bool open=true;uint64_t nonce=1;bool request_open()const{return open;}uint64_t slot_occupancy_incarnation()const{return nonce;}
int cut()const{return 7;}bool close_request(std::string&){events.push_back("close");return true;}bool retire_owned_slot(std::string&){events.push_back("raw");return true;}};
struct Outcome {std::shared_ptr<int> owned_slot_qualification;};
struct Lane {bool repair_finishing=true;std::shared_ptr<Outcome> owned_repair=std::make_shared<Outcome>();std::shared_ptr<Raw> owned_raw=std::make_shared<Raw>();};
'''
    prelude+='''
namespace strata::program {
struct ServingSuffixRepairOperation {static bool retire_slot_controller(Raw&,Outcome&,std::string&){events.push_back("controller");return true;}};
void report_http2_done(const std::string&,size_t,uint64_t,const char*,int){events.push_back("done");}
}
'''
    prelude=prelude.replace('bool repair_finishing=true;', 'std::string http_request="request";bool repair_finishing=true;')
    prelude+=take((ROOT/"src/program/serving_slot_qualification.inc").read_text(),"static bool owned_slot_qualification_target(")
    body=prelude+'size_t first(std::vector<Lane>& bs){\n'+selection+'return i;}return 999;}\n'
    body+='void peer(std::vector<Lane>& bs,size_t selected_slot){\n'+preflight+'}\n'
    body+='bool terminal(std::vector<Lane>& bs,bool eof){struct {bool input_error=false,end_of_input=false;} queued;queued.end_of_input=eof;std::string err;'+terminal+'return false;}\n'
    body+='''int main(){
uint64_t target=99;unsetenv("STRATA_OWNED_BATCH_QUALIFY");assert(owned_slot_qualification_target(target)&&target==0);
for(const char* invalid:{"0","-1","+1","1x","65537","18446744073709551616"}){setenv("STRATA_OWNED_BATCH_QUALIFY",invalid,1);assert(!owned_slot_qualification_target(target));}
setenv("STRATA_OWNED_BATCH_QUALIFY","65536",1);assert(owned_slot_qualification_target(target)&&target==65536);unsetenv("STRATA_OWNED_BATCH_QUALIFY");
std::vector<Lane> bs(2);
bs[1].owned_repair->owned_slot_qualification=std::make_shared<int>(1);assert(first(bs)==1);
bs[1].owned_repair->owned_slot_qualification.reset();assert(first(bs)==0);
bs[0].owned_repair->owned_slot_qualification=std::make_shared<int>(1);assert(first(bs)==0);
peer(bs,0);peer(bs,1);
for(int mode=0;mode<3;++mode){bs[0].owned_raw=std::make_shared<Raw>();
if(mode==0)bs[0].owned_raw.reset();if(mode==1)bs[0].owned_raw->open=false;if(mode==2)bs[0].owned_raw->nonce=0;
bool refused=false;try{peer(bs,1);}catch(const std::runtime_error&){refused=true;}assert(refused);}
bs=std::vector<Lane>(2);bs[1].owned_repair->owned_slot_qualification=std::make_shared<int>(1);
assert(terminal(bs,true));assert(!bs[0].repair_finishing);assert(!bs[0].owned_repair);
assert((events==std::vector<std::string>{"controller","close","raw","thread","done"}));
}
'''
    variants=[('controls',body),('missing_sibling_pin',body.replace('                if(pending_slot_oracle&&(!lane.owned_repair||!lane.owned_repair->owned_slot_qualification))continue;','')),
              ('missing_actual_peer',body.replace('                strata::core::retirement_failed("two-slot numerical qualification requires a genuine occupied quiet peer");','                (void)0;'))]
    for name,code in variants:
        if name!='controls' and code==body:raise RuntimeError('ineffective mutant '+name)
        cpp=build/(name+'.cpp');binary=build/name;cpp.write_text(code)
        subprocess.run(['c++','-std=c++20','-O0',str(cpp),'-o',str(binary)],check=True)
        result=subprocess.run([str(binary)],capture_output=True,timeout=10)
        if (result.returncode==0)!=(name=='controls'):raise RuntimeError(name+' unexpected result')
    print('PASS actual reversed selected-slot retirement pin, EOF controller-before-RAW cleanup, real-peer preflight + 2 negatives')

def run(build):
    build.mkdir(parents=True,exist_ok=True)
    raw=(ROOT/'src/program/suffix_raw_runtime.cpp').read_text()
    pins=json.loads((HERE/'unchanged_public_pins.json').read_text())
    for marker,digest in pins['public'].items():
        if hashlib.sha256(take(raw,marker).encode()).hexdigest()!=digest:raise RuntimeError('ordinary RAW API changed '+marker)
    generate=(ROOT/'src/program/generate.cpp').read_text()
    unsupported=generate[generate.index('#else\n    if (std::getenv("STRATA_SUFFIX_REPAIR_QUALIFY")'):generate.index('#if defined(STRATA_SERVING_CANCEL_QUALIFICATION_AVAILABLE)',generate.index('#else\n    if (std::getenv("STRATA_SUFFIX_REPAIR_QUALIFY")'))]
    assert 'std::getenv("STRATA_OWNED_BATCH_QUALIFY")' in unsupported and 'return 2;' in unsupported
    for marker,digest in pins['shared_helpers'].items():
        if hashlib.sha256(take(generate,marker).encode()).hexdigest()!=digest:raise RuntimeError('lifted helper changed original native logic '+marker)
    callback=take(generate,'        owned_slot_numerical_oracle =')
    assert 'report_oracle_boundary("complete")' in callback and callback.index('if(!pass)')<callback.index('report_oracle_boundary("complete")')
    assert 'report_owned_slot_preserved(*other.owned_raw' in callback
    assert callback.index('const size_t required_peer=')<callback.index('qualify_owned_slot_final(')
    controls(build,generate)
    flags=['c++','-std=c++20','-O0','-Wall','-Wextra','-I'+str(HERE/'stubs'),'-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/core'),'-I'+str(ROOT/'src/program')]
    common=['src/core/conversation_snapshot.cpp','src/core/window_native.cpp','src/core/window_state.cpp','src/core/window_hash.cpp']
    objects=[]
    for index,path in enumerate(common):
        obj=build/('common'+str(index)+'.o');subprocess.run(flags+['-c',str(ROOT/path),'-o',str(obj)],check=True);objects.append(str(obj))
    fixture=build/'fixture.o';subprocess.run(flags+['-c',str(HERE/'empty_physical.cpp'),'-o',str(fixture)],check=True);objects.append(str(fixture))
    native=(ROOT/'src/core/conversation_state.cpp').read_text()
    variants=[('production',native),('lost_preallocated_stage_children',native.replace('    captured.stage_images.swap(image.stage_images);','    /* negative: root replacement discards stage children */')),('omitted_empty_endpoint_restore',native.replace('    if(!conversation_checkpoint_restore(image.live,ss,g,error))return ConversationRestore::transfer_failed;','    /* negative: endpoint copy omitted */'))]
    for name,code in variants:
        if name!='production' and code==native:raise RuntimeError('ineffective empty restore mutant')
        # Keep original local include resolution while compiling the exact full body.
        cpp=build/(name+'.cpp');obj=build/(name+'.o');binary=build/name;cpp.write_text(code)
        subprocess.run(flags+['-c',str(cpp),'-o',str(obj)],check=True)
        subprocess.run(['c++',str(obj),*objects,'-o',str(binary)],check=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
        if (result.returncode==0)!=(name=='production'):raise RuntimeError(name+': '+result.stderr)
        print('PASS '+name+': '+(result.stdout.strip() if name=='production' else 'actual missing endpoint restore is rejected'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path);args=parser.parse_args()
    if args.build_dir:run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-slot-oracle-') as directory:run(Path(directory))
