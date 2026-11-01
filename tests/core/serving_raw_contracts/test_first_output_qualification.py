#!/usr/bin/env python3
"""Execute extracted actual first-output admission/readback/binding/replay-position logic.

Verifier, device, RAW admission and execution are recording CPU seams. Actual
hashing and distribution metrics are linked unchanged. No native state, real
CUDA drain or GPU qualification is claimed.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

PRELUDE = r'''
#include "strata/core/window_state.hpp"
#include "suffix_experiment_metrics.hpp"
#include "serving_prefix_replay.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
namespace seam {
int device=11,copies=0,syncs=0,finishes=0,last_row=-1;
bool copy_error=false,sync_error=false,device_error=false,finish_error=false;
float* scratch=nullptr;
std::vector<uint64_t> positions;
std::vector<float> clean={0,1,4,2};
void reset(){device=11;copies=syncs=finishes=0;last_row=-1;copy_error=sync_error=device_error=finish_error=false;positions.clear();}
}
constexpr int cudaSuccess=0;
int cudaDeviceSynchronize(){++seam::syncs;return seam::sync_error?1:0;}
namespace strata::core {
struct Verifier {
    const Verifier* next=nullptr;int dev=11;std::vector<float> logits={0,3,1,2};
    struct Binding {const Verifier* next;};
    Binding serial_routing_binding()const{return {next};}
    int device()const{return dev;}
    bool copy_logits(int row,float* out)const{
        if(next)return next->copy_logits(row,out);
        ++seam::copies;seam::last_row=row;
        if(seam::copy_error||row!=0||seam::device!=dev)return false;
        std::copy(logits.begin(),logits.end(),out);return true;
    }
};
struct WindowDeviceScope {
    int entry;bool done=false;
    explicit WindowDeviceScope(int selected):entry(seam::device){seam::device=selected;}
    bool ok()const{return !seam::device_error;}
    bool finish(){++seam::finishes;seam::device=entry;done=true;return !seam::finish_error;}
    ~WindowDeviceScope(){if(!done)seam::device=entry;}
};
}
namespace core=strata::core;
using strata::program::SuffixLogitComparison;
using strata::program::PrefixReplayPlan;
using strata::program::compare_suffix_logits;
struct ServingRawSession {
    core::StateCut current;bool opened=true,drained=true,permit=true;uint64_t bytes=0;
    uint64_t max_tokens()const{return 16;}
    bool request_open()const{return opened;}
    bool capture_allowed()const{return drained;}
    const core::StateCut& cut()const{return current;}
    struct Result {bool ok;std::string detail;explicit operator bool()const{return ok;}};
    Result admit_host(uint64_t n,std::shared_ptr<const void>& out){bytes=n;if(!permit)return {false,"actual seam quota refusal"};out=std::make_shared<int>(1);return {true,{}};}
};
struct Context {int options=0;struct Geometry {int64_t n_vocab=4;} geometry;
    core::Verifier& primary_verifier;int stage_count=2;bool valid=true;};
struct Receipt {uint64_t operation=7,publication=2,cut=19;int32_t token=1;};
struct Harness {
    using Clock=std::chrono::steady_clock;
    static double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
    [[noreturn]] static void failstop(const char* why){throw std::runtime_error(why);}
    template<class C> static bool validate_serving_qualification_execution(const C& c,std::string&){return c.valid;}
    template<class O> static bool serving_qualification_policy(const O&,std::string&){return true;}
    static bool qualification_file_hash(const char*,uint64_t,std::array<char,65>& hash){hash[0]='a';return true;}
'''

HARNESS_END = r'''
    struct Step {core::StateCut cut;int32_t next=-1;};
    template<class C> static Step forward(C&,int&,int&,uint64_t&,int32_t,uint64_t at,uint64_t,
        core::StateDigest,uint64_t,std::span<float> row){
        if(!row.empty()){
            if(row.data()!=seam::scratch||row.size()!=seam::clean.size())failstop("fourth or wrong scratch row");
            seam::positions.push_back(at);std::copy(seam::clean.begin(),seam::clean.end(),row.begin());
        }
        Step step;step.cut.forwarded=at+1;step.next=int32_t(std::max_element(seam::clean.begin(),seam::clean.end())-seam::clean.begin());return step;
    }
    static bool replay_position(const ServingQualification& admission,const core::SnapshotDescriptor& original,bool needs_first){
        struct {std::vector<int32_t> history;std::vector<float> exact_logits;} buffers;
        buffers.history=original.accepted_tokens;buffers.exact_logits.resize(4);seam::scratch=buffers.exact_logits.data();
        uint64_t forwarded=original.cut.forwarded,accepted=original.cut.accepted,next_cut=1,ordinal=2;
        uint64_t prefix_at=0;int c=0,serial=0,tx=0;Step step;bool was_cancelled=false,first_compared=!needs_first;
        auto cancelled=[](){return false;};
ACTUAL_LOOP
        return first_compared&&!was_cancelled;
    }
};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
core::SnapshotDescriptor selected(){
    core::SnapshotDescriptor s;s.cut.id=19;s.cut.accepted=4;s.cut.forwarded=3;s.cut.exact_frontier=1;s.cut.output_watermark=1;
    s.accepted_tokens={3,2,0,1};s.pending_tokens={1};s.cut.history=core::window_history_digest(s.accepted_tokens);
    s.approximation.resize(1);s.approximation[0].begin=1;s.approximation[0].end=3;return s;
}
void capture(Harness::ServingQualification& admission,Context& c,ServingRawSession& session,core::SnapshotDescriptor s=selected(),Receipt receipt={}){
    session.current=s.cut;Harness::capture_serving_first_output(c,session,2,admission,s,receipt);
}
template<class F> void refuses(F f,const char* why){bool refused=false;try{f();}catch(const std::runtime_error&){refused=true;}require(refused,why);}
int main(){try{
    setenv("STRATA_SUFFIX_REPAIR_QUALIFY","1",1);setenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST","/fake/source-manifest",1);
    core::Verifier root,tail;tail.dev=22;root.next=&tail;Context c{0,{4},root};ServingRawSession session;
    {
        Harness::ServingQualification off;unsetenv("STRATA_SUFFIX_REPAIR_QUALIFY");std::string error;
        require(Harness::admit_serving_qualification(c,session,4,off,error)&&!off.credit&&!off.first.values&&!session.bytes,
                "QUALIFY absent must not claim or allocate a row");setenv("STRATA_SUFFIX_REPAIR_QUALIFY","1",1);
    }
    {
        Harness::ServingQualification exact;std::string error;setenv("STRATA_SUFFIX_REPAIR_EXACT_FIRST","1",1);
        require(Harness::admit_serving_qualification(c,session,4,exact,error)&&!exact.retained&&!exact.first.values,
                "exact-first retains two rows and no saved approximate row");
        require(session.bytes==(1ULL<<20)+16*128+4*2*sizeof(float),"actual exact-first two-row quote");unsetenv("STRATA_SUFFIX_REPAIR_EXACT_FIRST");
    }
    {
        Harness::ServingQualification denied;std::string error;session.permit=false;
        require(!Harness::admit_serving_qualification(c,session,4,denied,error)&&!denied.first.values,
                "actual H refusal precedes saved-row allocation");session.permit=true;
    }
    Harness::ServingQualification admission;std::string error;
    require(Harness::admit_serving_qualification(c,session,6,admission,error)&&admission.retained&&admission.first.values,
            "actual retained admission preallocates saved row");
    require(session.bytes==(1ULL<<20)+16*128+4*3*sizeof(float),"actual retained three-row overlap quote");
    require(!admission.first.captured,"zero-output/import cancellation has no captured row");
    seam::reset();capture(admission,c,session);
    require(admission.first.captured&&admission.first.emitted==1&&admission.first.forwarded==3,"bind actual emitted receipt and position");
    require(seam::copies==1&&seam::syncs==1&&seam::finishes==1&&seam::last_row==0&&seam::device==11,
            "actual capture method selects final stage, copies row0, syncs and restores entry device");
    refuses([&]{capture(admission,c,session);},"duplicate first-output capture refused");
    auto final=selected();final.cut.exact_frontier=3;final.approximation.clear();
    require(Harness::qualification_first_output_matches(admission.first,2,final,1),"single-output binding");
    auto changed=final;changed.accepted_tokens[0]^=1;
    require(!Harness::qualification_first_output_matches(admission.first,2,changed,1),"conditioning token mutation refused");
    changed=final;changed.accepted_tokens[3]=2;
    require(!Harness::qualification_first_output_matches(admission.first,2,changed,1),"emitted pending replacement refused");
    require(!Harness::qualification_first_output_matches(admission.first,3,final,1),"different request refused");
    require(!Harness::qualification_first_output_matches(admission.first,2,final,0),"zero outputs cannot own first-output evidence");
    final.accepted_tokens.insert(final.accepted_tokens.end(),{2,3});final.pending_tokens={3};
    final.cut.accepted=6;final.cut.forwarded=final.cut.exact_frontier=5;final.cut.output_watermark=3;
    require(Harness::qualification_first_output_matches(admission.first,2,final,3),"longer output preserves conditioning prefix and original emission");
    seam::positions.clear();require(Harness::replay_position(admission,final,true)&&seam::positions==std::vector<uint64_t>{2},
            "actual replay loop copies saved first-output position, not final position, into existing scratch");
    Harness::ServingQualification zero;require(Harness::admit_serving_qualification(c,session,4,zero,error),"zero-output admission");
    auto zero_final=selected();zero_final.cut.exact_frontier=zero_final.cut.forwarded;zero_final.cut.output_watermark=0;
    seam::positions.clear();require(!zero.first.captured&&Harness::replay_position(zero,zero_final,false)&&seam::positions.empty(),
            "actual zero-output replay makes no diagnostic row request");
    std::array<float,4> same{0,3,1,2},different{0,1,4,2},invalid{0,1,INFINITY,2};
    require(Harness::report_serving_approximation(admission,2,3,1,same),"zero measured divergence accepted");
    require(Harness::report_serving_approximation(admission,2,3,2,different),"finite nonzero divergence and different exact pick accepted");
    require(!Harness::report_serving_approximation(admission,2,3,2,invalid),"nonfinite exact row refused");
    require(!Harness::report_serving_approximation(admission,2,4,2,different),"wrong replay position refused");
    require(!Harness::report_serving_approximation(admission,2,3,1,different),"exact pick must match copied exact row");
    for(int mode=0;mode<9;++mode){
        Harness::ServingQualification failed;require(Harness::admit_serving_qualification(c,session,4,failed,error),"negative admission");
        seam::reset();c.valid=true;root.next=&tail;c.stage_count=2;tail.logits={0,3,1,2};auto s=selected();Receipt receipt;
        if(mode==0)seam::copy_error=true;
        if(mode==1)seam::sync_error=true;
        if(mode==2)seam::finish_error=true;
        if(mode==3)receipt.cut=18;
        if(mode==4)c.valid=false;
        if(mode==5)root.next=nullptr;
        if(mode==6)receipt.token=2;
        if(mode==7)tail.logits={0,1,4,2};
        if(mode==8)tail.logits[1]=INFINITY;
        refuses([&]{capture(failed,c,session,s,receipt);},"failed readback/drain/device/receipt/binding/chain must terminate");
        require(!failed.first.captured&&failed.credit,"failed readback cannot publish evidence or discard its credit");
        c.valid=true;root.next=&tail;
    }
    std::puts("PASS actual first-output admission/readback/binding/replay/metrics CPU seams; no native proof");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
'''


def take(source,marker,semicolon=False):
    begin=source.index(marker);opening=source.index('{',begin);depth=0
    for at in range(opening,len(source)):
        if source[at]=='{':depth+=1
        elif source[at]=='}':
            depth-=1
            if not depth:
                return source[begin:at+1]+(';' if semicolon else '')+'\n'
    raise RuntimeError('unclosed actual source extract: '+marker)


def run(build):
    root=Path(__file__).resolve().parents[3]
    paths=[root/'src/program/serving_suffix_qualification.inc',root/'src/program/suffix_experiment_driver.hpp',
           root/'src/program/suffix_experiment_metrics.hpp',root/'src/core/window_hash.cpp']
    before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source=paths[0].read_text();driver=paths[1].read_text()
    markers=[('struct ServingFirstOutputLogits',True),('struct ServingQualification',True),
             ('static bool serving_qualification_enabled()',False),('static bool serving_qualification_limit(',False),
             ('static std::array<char,65> qualification_hex(',False),
             ('template<class Context>\nstatic bool admit_serving_qualification(',False),
             ('static void qualification_u64(',False),('static void qualification_string(',False),
             ('static bool qualification_conditioning_digest(',False),('static bool qualification_first_output_matches(',False),
             ('template<class Context,class Receipt>\nstatic void capture_serving_first_output(',False),
             ('static bool report_serving_approximation(',False)]
    actual=take(driver,'    static bool add_product(')+''.join(take(source,marker,semi) for marker,semi in markers)
    loop=take(source,'            for(uint64_t at=prefix_at;!was_cancelled&&at<forwarded;++at)')
    code=PRELUDE+actual+HARNESS_END.replace('ACTUAL_LOOP',loop)
    path=build/'actual-first-output.cpp';binary=build/'actual-first-output';path.write_text(code)
    subprocess.run(['c++','-std=c++20','-O2','-Wall','-Wextra','-Wconversion','-Werror',
                    '-I'+str(root/'include'),'-I'+str(root/'src/program'),str(path),str(root/'src/core/window_hash.cpp'),
                    '-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=20)
    for p in paths:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=before[str(p.relative_to(root))]:
            raise RuntimeError('production input changed during actual-method fixture')
    print('source_pins='+repr(before),flush=True)
    print('actual_methods_sha256='+hashlib.sha256(actual.encode()).hexdigest()+' actual_replay_loop_sha256='+hashlib.sha256(loop.encode()).hexdigest(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build-dir',type=Path);args=parser.parse_args()
    if args.build_dir:
        args.build_dir.mkdir(parents=True,exist_ok=True);run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-first-output-') as directory:run(Path(directory))


if __name__=='__main__':main()
