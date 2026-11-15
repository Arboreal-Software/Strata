#!/usr/bin/env python3
"""Execute production activation admission/rollback helper with recording CPU native operations.

The real helper is extracted, not copied. Proves control/restoration/negative handling only,
not GPU tensors, numerical correctness, hardware drains, or allocation fit.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
from test_retained_controller_activation import run as run_retained_controller
from test_retained_request_continuation import run as run_retained_request
from test_retained_activation_outer import run as run_retained_outer
from test_activation_request import run as run_activation_request
from test_owned_slot_oracle import run as run_owned_slot_oracle
from test_fresh_request_watermark import run as run_fresh_request_watermark
from test_bridge_qualification import run as run_bridge_qualification
from test_owned_admission_settlement import run as run_owned_admission_settlement
from test_owned_slot_prealignment import run as run_owned_slot_prealignment
from test_owned_slot_prealignment import check_source as check_owned_slot_prealignment_source


def take(text, marker, semi=False):
    start=text.index(marker);parentheses=0;opening=None
    for at in range(start,len(text)):
        if text[at]=='(':parentheses+=1
        elif text[at]==')':parentheses-=1
        elif text[at]=='{' and parentheses==0:
            opening=at;break
    if opening is None:raise ValueError('production body missing: '+marker)
    depth=0
    for at in range(opening,len(text)):
        if text[at]=='{':depth+=1
        elif text[at]=='}':
            depth-=1
            if not depth:return text[start:at+1]+(';' if semi else '')+'\n'
    raise ValueError(marker)


PRELUDE=r'''
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
namespace core {
using StateDigest=std::array<unsigned char,32>;
struct StateCut {uint64_t id=5,accepted=3,forwarded=2,exact_frontier=2,output_watermark=1;
    StateDigest history{};bool operator==(const StateCut&)const=default;};
struct SnapshotDescriptor {int key=1;StateCut cut;std::vector<int32_t> accepted_tokens{1,2,3},pending_tokens{3};
    std::vector<int> approximation;int payload=10;};
struct BundleHandle {std::shared_ptr<SnapshotDescriptor> p;
    explicit operator bool()const{return bool(p);}SnapshotDescriptor* get()const{return p.get();}};
struct Verifier {bool sampling=true;void set_head_sampling(bool x){sampling=x;}};
}
namespace seam {bool change_capture=false,hash_ok=true,permit=true;int forwards=0,captures=0,restores=0;
std::vector<std::string> phases;}
struct PrefixReplayPlan {
    uint64_t boundary=0,request=0;
    static bool quote(uint64_t boundary,uint64_t entries,uint64_t& bytes){
        if(boundary<=64||!entries)return false;bytes+=16384;return true;}
};
struct ServingRawSession {
    struct Transaction {core::StateCut cut_;} tx;
    int payload=10;bool opened=false;
    bool open_request(std::string& error){if(opened){error="already open";return false;}opened=true;return true;}
    bool close_request(std::string&){opened=false;return true;}
    struct Result {bool ok=true;std::string detail;explicit operator bool()const{return ok;}};
    uint64_t max_tokens()const{return 4096;}bool request_open()const{return opened;}
    bool capture_allowed()const{return true;}core::StateCut cut()const{return tx.cut_;}
    Transaction* transaction(){return &tx;}
    Result admit_host(uint64_t,std::shared_ptr<const void>& out){
        if(!opened)return {false,"host admission needs an open serving request"};
        if(!seam::permit)return {false,"quota"};out=std::make_shared<int>(1);return {};}
    static core::SnapshotDescriptor endpoint_metadata(int key,core::StateCut cut,const std::vector<int32_t>& ids){
        core::SnapshotDescriptor d;d.key=key;d.cut=cut;d.accepted_tokens=ids;
        if(cut.accepted==cut.forwarded)d.pending_tokens.clear();return d;}
    Result capture(const core::SnapshotDescriptor& d,core::BundleHandle& out){
        ++seam::captures;out.p=std::make_shared<core::SnapshotDescriptor>(d);out.p->payload=payload;
        if(seam::change_capture&&seam::captures==1)out.p->payload+=1;return {};}
    Result restore(const core::BundleHandle& image,int){++seam::restores;payload=image.get()->payload;tx.cut_=image.get()->cut;return {};}
};
struct Policy {bool serve=true,suffix_cache=true,suffix_cache_unrepaired_experiment=false,
    owned_session_lifetime=true,owned_model_lifetime=true,prefill_head_alias=false,vision=false,
    batch=false,pipeline_windows=false,adapt_every=false,adapt_async=false,kv_grow=false;};
struct Context {Policy options;struct {int n_vocab=4,n_layers=4,n_expert=2;} geometry;core::Verifier primary_verifier;
    ServingRawSession& session;bool valid=true;};
struct SuffixLogitComparison {double max_absolute_error=0;int exact_argmax=3,approximate_argmax=3;};
static bool compare_suffix_logits(const std::vector<float>& a,const std::vector<float>& b,SuffixLogitComparison& out){
    out.max_absolute_error=0;for(size_t i=0;i<a.size();++i)out.max_absolute_error=std::max(out.max_absolute_error,double(std::abs(a[i]-b[i])));return true;}
struct Harness {
    struct Buffers {std::vector<int32_t> history;std::vector<float> exact_logits,approximate_logits;};
    [[noreturn]] static void failstop(const char* why){throw std::runtime_error(why);}
    [[noreturn]] static void failstop(ServingRawSession&,const char* why){throw std::runtime_error(why);}
    static bool add_product(uint64_t& x,uint64_t n,uint64_t k){if(n&&k>(UINT64_MAX-x)/n)return false;x+=n*k;return true;}
    template<class F> static bool raw_call(F&& f){return bool(f());}
    template<class C> static bool validate_serving_qualification_execution(C& c,std::string&){return c.valid;}
    static bool qualification_file_hash(const char*,uint64_t,std::array<char,65>& out){out.fill('a');out[64]=0;return seam::hash_ok;}
'''
MIDDLE=r'''
    using Outcome=int;
    ACTUAL_HOOKS
    static bool activation_admit_preamble(ServingRawSession& session,ActivationQualificationHooks* qualification,std::string& error){
        ACTUAL_ADMISSION_PREAMBLE
        return true;
    }
    static bool report_serving_comparison(ServingRawSession&,const ServingQualification&,uint64_t,const char* phase,
        const core::BundleHandle& a,const core::BundleHandle& b,bool logits=true,bool picks=true,double error=0,
        void* =nullptr,const char* ="reset_replay"){
        seam::phases.push_back(phase);auto left=a.get()->cut,right=b.get()->cut;left.id=right.id=0;
        return a.get()->payload==b.get()->payload&&left==right&&logits&&picks&&error==0;}
    struct Step {int next=3;core::StateCut cut;};
    template<class C> static Step forward(C& c,int&,ServingRawSession::Transaction& tx,uint64_t& next,
        int32_t token,uint64_t at,uint64_t accepted,core::StateDigest history,uint64_t exact,std::vector<float>& logits){
        ++seam::forwards;c.session.payload+=token;tx.cut_.id=++next;tx.cut_.forwarded=at+1;
        tx.cut_.accepted=accepted;tx.cut_.history=history;tx.cut_.exact_frontier=exact;
        std::fill(logits.begin(),logits.end(),float(c.session.payload));return {3,tx.cut_};}
'''
END=r'''
};
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
int main(){try{
    setenv("STRATA_SUFFIX_ACTIVATION_QUALIFY","rollback",1);setenv("STRATA_LOOKAHEAD","0",1);
    setenv("STRATA_SUFFIX_REPAIR_SOURCE_MANIFEST","/measured-source",1);
    ServingRawSession session;Context context{{},{}, {},session};int serial=0;std::string error;
    Harness::ServingQualification admission;
    require(!Harness::admit_activation_qualification(context,session,3,admission,error)&&!admission.credit&&
            error.find("open serving request")!=std::string::npos,"closed-request reservation refuses at host guard");
    const auto original_cut=session.cut();const int original_payload=session.payload;
    Harness::ActivationQualificationHooks hooks;
    hooks.admit=[&](std::string& why){return Harness::admit_activation_qualification(context,session,3,admission,why);};
    require(Harness::activation_admit_preamble(session,&hooks,error),"actual activation preamble opens before host admission");
    require(session.request_open()&&session.cut()==original_cut&&session.payload==original_payload&&
            seam::captures==0&&seam::restores==0,"scratch admission precedes all native capture/restore with B untouched");
    require(session.close_request(error),"close successful preamble fixture");
    Harness::ServingQualification refused_admission;
    hooks.admit=[&](std::string& why){return Harness::admit_activation_qualification(context,session,3,refused_admission,why);};
    seam::permit=false;
    require(!Harness::activation_admit_preamble(session,&hooks,error)&&!session.request_open()&&!refused_admission.credit&&
            session.cut()==original_cut&&session.payload==original_payload&&seam::captures==0&&seam::restores==0,
            "actual admission refusal closes opened fence without displacing B");
    seam::permit=true;require(session.open_request(error),"open actual rollback qualification request");
    require(admission.activation_oracle&&admission.admitted&&admission.credit&&!admission.first.captured,"independent admitted scratch without fake first row");
    Harness::ServingQualification origin;
    require(Harness::admit_activation_prefix_origin(context,session,100,96,2,origin,error),"real durable origin admission");
    require(origin.credit&&origin.durable_prefix&&origin.admitted&&origin.prefix.boundary==96&&origin.prefix.request==2&&
            !origin.activation_oracle&&!origin.first.values&&!origin.first.captured,
            "origin owns metadata holder without borrowing A3 scratch or strict first-output row");
    Harness::ServingQualification wrong_origin;
    require(!Harness::admit_activation_prefix_origin(context,session,100,96,4,wrong_origin,error)&&!wrong_origin.credit,
            "wrong origin ordinal refuses before metadata admission");
    seam::permit=false;
    require(!Harness::admit_activation_prefix_origin(context,session,100,96,2,wrong_origin,error)&&!wrong_origin.durable_prefix,
            "origin budget refusal exposes no durable holder");seam::permit=true;

    core::BundleHandle before;require(bool(session.capture(core::SnapshotDescriptor{},before)),"capture original");
    seam::captures=0;uint64_t next=8;
    require(Harness::qualify_activation_rollback(context,serial,session,4,admission,before,next),"real rollback helper");
    require(session.payload==10&&session.cut()==before.get()->cut,"real final restore");
    require(seam::forwards==2&&context.primary_verifier.sampling,"two actual mock continuations and sampling restored");
    require(seam::phases==std::vector<std::string>{"rollback_frontier","rollback_continuation","rollback_restoration"},"all actual comparisons reached");
    seam::phases.clear();seam::captures=0;seam::change_capture=true;
    require(!Harness::qualify_activation_rollback(context,serial,session,4,admission,before,next),"changed postimage refused");
    require(session.payload==10&&session.cut()==before.get()->cut,"mismatch still restores original");
    seam::change_capture=false;
    bool threw=false;try{Harness::qualify_activation_rollback(context,serial,session,3,admission,before,next);}catch(const std::runtime_error&){threw=true;}
    require(threw,"wrong request accepted");
    for(int bad=0;bad<4;++bad){Harness::ServingQualification refused;
        seam::permit=bad!=0;seam::hash_ok=bad!=1;context.valid=bad!=2;
        setenv("STRATA_SUFFIX_ACTIVATION_QUALIFY",bad==3?"unknown":"rollback",1);
        require(!Harness::admit_activation_qualification(context,session,3,refused,error),"admission negative control");}
    std::puts("PASS extracted activation admission/rollback CPU seams; no native/GPU proof");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
'''


def check_source():
    root=Path(__file__).resolve().parents[3]
    text=(root/'src/program/serving_suffix_qualification.inc').read_text()
    generate=(root/'src/program/generate.cpp').read_text()
    report=take(text,'static bool report_serving_comparison(')
    if 'admission.activation_oracle?"suffix_activation_qualification":"suffix_repair_qualification"' not in report:
        raise ValueError('activation producer prefix is not separated from strict yield grammar')
    for phase in ('activation_frontier','activation_continuation','activation_restoration',
                  'rollback_frontier','rollback_continuation','rollback_restoration'):
        if '"'+phase+'"' not in text:raise ValueError('activation phase missing: '+phase)
    wrapper=take(generate,'    static bool activate_retained_owner(')
    opened=wrapper.index('if (!session.open_request(fence_error))')
    reserved=wrapper.index('!qualification->admit(error)')
    prior_capture=wrapper.index('session.capture(prior_metadata, prior_image)')
    if not opened<reserved<prior_capture:raise ValueError('actual host admission must follow open and precede native capture')
    if 'activation_hooks.admit=[&](std::string& admission_error)' not in generate:
        raise ValueError('serving host reservation is no longer deferred to activation open')
    if 'retire_activation_refusal("activation qualification host admission refused")' not in generate:
        raise ValueError('admission failure leaves parked controllers unretired on early return')
    begin=wrapper.index('const auto installed = session.restore(payload.retained_publication, identity);')
    forced=wrapper.index('const bool forced=qualification&&qualification->refuse_after_publication_restore;')
    activate=wrapper.index('const bool landed = !forced&&ServingSuffixRepairOperation::activate_retained(')
    undo=wrapper.index('const auto back = session.restore(prior_image, prior.key);',activate)
    observe=wrapper.index('qualification->rollback(prior_image)',undo)
    close=wrapper.index('if (!session.close_request(close_error))',observe)
    if not begin<forced<activate<undo<observe<close:
        raise ValueError('qualification refusal/real rollback/observation ordering differs')
    if 'after_evidence=ServingSuffixRepairOperation::owner_evidence(payload)' not in wrapper:
        raise ValueError('forced refusal does not return actual suspended evidence')
    final=take(text,'template<class Context,class SerialUnit,class Cancelled>\nstatic bool qualify_activation_final(')
    if not final.index('qualify_serving_frontier(')<final.index('session.capture(metadata,restored)')<final.index('"activation_restoration"'):
        raise ValueError('actual A3 replay/restoration compare ordering differs')
    if 'activation_prefix_view_matches(admission.required_origin_boundary,prefix,*actual.get(),' not in final:
        raise ValueError('activation final does not require actual durable view for batched origin')
    if 'prefix.plan->origin_binary!=admission.binary' not in final or 'prefix.plan->origin_source!=admission.source' not in final:
        raise ValueError('activation final loses actual origin binary/source provenance')
    for marker in ('sfx_activation_qualification,operation.repaired_final,next,[&]{return stop_req.load();},sfx_prefix_view())',
                   'sfx_activation_qualification,actual_final,next,[&]{return stop_req.load();},sfx_prefix_view())'):
        if marker not in generate:raise ValueError('A3 endpoint call site dropped recorded prefix view')
    if 'qualify_serving_frontier(' not in final or ',prefix);' not in final:
        raise ValueError('activation final does not pass durable prefix into independent replay')
    if generate.index('admit_activation_prefix_origin(')>generate.index('serving_raw_session->begin_work(fence_error)',generate.index('admit_activation_prefix_origin(')):
        raise ValueError('origin metadata admission follows donor mutation')
    print('PASS actual producer/parser vocabulary and publication/refusal/undo/restore observation order')


def run(build):
    run_owned_slot_oracle(build / "owned-slot-oracle")
    root=Path(__file__).resolve().parents[3]
    text=(root/'src/program/serving_suffix_qualification.inc').read_text()
    names=[('struct ServingFirstOutputLogits',True),('struct ServingQualification',True),
           ('static bool activation_qualification_request(',False),
           ('static bool activation_qualification_selects(',False),
           ('static bool serving_qualification_limit(',False),('template<class Policy>\nstatic bool serving_qualification_policy(',False),
           ('static const char* activation_qualification_mode()',False),
           ('template<class Context>\nstatic bool admit_activation_qualification(',False),
           ('template<class Context>\nstatic bool admit_activation_prefix_origin(',False)]
    actual=''.join(take(text,name,semi) for name,semi in names)
    helper=take(text,'template<class Context,class SerialUnit>\nstatic bool qualify_activation_rollback(')
    generate=(root/'src/program/generate.cpp').read_text()
    wrapper=take(generate,'    static bool activate_retained_owner(')
    preamble=wrapper[wrapper.index('        std::string fence_error;'):wrapper.index('        // (2) The prior conversation')]
    # This fixture isolates actual open/refusal-close and qualification host reservation.
    # Generic incoming-request preparation between them is exercised by its own production-tail fixture.
    if '        PreparedContinuationRequest prepared_request;' in preamble:
        preamble=preamble[:preamble.index('        PreparedContinuationRequest prepared_request;')]+preamble[preamble.index('        // Qualification scratch'):]

    hooks=take(generate,'    struct ActivationQualificationHooks',True).replace('strata::core::','core::')
    middle=MIDDLE.replace('ACTUAL_HOOKS',hooks).replace('ACTUAL_ADMISSION_PREAMBLE',preamble)

    path=build/'actual-activation.cpp';binary=build/'actual-activation'
    path.write_text(PRELUDE+actual+middle+helper+END)
    subprocess.run(['c++','-std=c++20','-O0','-Wall','-Wextra',str(path),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=20)
    # Structural negative control: bypass the production final restore. The executable must reject it.
    marker='if(!raw_call([&]{return session.restore(before,original.key);}))failstop("activation rollback final restore refused");'
    if helper.count(marker)!=1:raise ValueError('production restoration anchor changed')
    path.write_text(PRELUDE+actual+middle+helper.replace(marker,'/* negative control: omitted actual final restore */')+END)
    subprocess.run(['c++','-std=c++20','-O0',str(path),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
    if result.returncode==0 or 'FAIL real rollback helper' not in result.stderr:
        raise ValueError('omitted-production-restore negative control did not fail as expected: '+result.stderr)
    print('PASS omitted actual final restore negative control')
    closed_preamble=preamble.replace('if (!session.open_request(fence_error)) { error = fence_error; return false; }',
                                    '/* negative control: omitted request open */')
    if closed_preamble==preamble:raise ValueError('actual request-open anchor changed')
    closed_middle=MIDDLE.replace('ACTUAL_HOOKS',hooks).replace('ACTUAL_ADMISSION_PREAMBLE',closed_preamble)
    path.write_text(PRELUDE+actual+closed_middle+helper+END)
    subprocess.run(['c++','-std=c++20','-O0',str(path),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
    if result.returncode==0 or 'actual activation preamble opens before host admission' not in result.stderr:
        raise ValueError('closed-request production-order negative did not fail: '+result.stderr)
    print('PASS omitted actual request open negative control; host guard stays strict')
    run_retained_controller(build / "actual-controller")
    request_build=build / "actual-request-tail"
    request_build.mkdir(parents=True,exist_ok=True)
    run_retained_request(request_build)
    run_fresh_request_watermark(build / "actual-fresh-request-watermark")
    run_bridge_qualification(build / "actual-cross-owner-bridge")
    run_owned_admission_settlement(build / "actual-owned-admission-settlement")
    run_owned_slot_prealignment(build / "actual-owned-slot-prealignment")
    run_retained_outer(build / "actual-outer-activation")
    run_activation_request(root,build / "actual-activation-request")



def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build-dir',type=Path);parser.add_argument('--source-only',action='store_true');args=parser.parse_args()
    check_source()
    check_owned_slot_prealignment_source()
    if args.source_only:return
    if args.build_dir:
        args.build_dir.mkdir(parents=True,exist_ok=True);run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-activation-') as directory:run(Path(directory))


if __name__=='__main__':main()
