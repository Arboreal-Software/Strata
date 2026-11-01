#!/usr/bin/env python3
"""Execute actual serving request adapter admission/rebase/known/generate/finish using recording CPU controller.
No native numerical/drain/GPU claim. Tests preserve two requests' cumulative receipts and a real59-token tail.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile


def take(text,marker,semi=False):
    start=text.index(marker);opening=text.index('{',start);depth=0
    for i in range(opening,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start:i+1]+(';' if semi else '')+'\n'
    raise ValueError(marker)


PRELUDE=r'''
#include "strata/core/window_state.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::core {[[noreturn]] void retirement_failed(const char* why){throw std::runtime_error(why);}}
struct Controller {
    struct Published {strata::core::SnapshotDescriptor metadata;uint64_t serial=1;};
    struct Receipt {uint64_t operation=0,publication=1,cut=0;int32_t token=0;};
    struct KnownReceipt {int next=0;};
    struct Runtime {struct Service {struct Geometry {int n_vocab=10000;} geometry;
        Geometry* geometry_owner_=&geometry;} service;Service* service_=&service;} runtime;
    Runtime* runtime_=&runtime;
    std::shared_ptr<Published> published_=std::make_shared<Published>();
    std::array<Receipt,64> receipts_{};
    uint64_t receipt_count_=2,known_count_=1,foreground_count_=3,next_cut_=100,private_pos=5;
    unsigned partial_count_=1;
    KnownReceipt known;
    uint64_t private_forwarded()const{return private_pos;}
    void commit(int32_t token){auto& m=published_->metadata;m.accepted_tokens.push_back(token);m.pending_tokens.assign(1,token);
        ++m.cut.forwarded;++m.cut.accepted;++m.key.append_epoch;m.cut.id=next_cut_++;++foreground_count_;
        m.cut.history=strata::core::window_history_digest(m.accepted_tokens);}
    const KnownReceipt* foreground_known(uint64_t,uint64_t,int32_t token,std::string&){commit(token);++known_count_;known.next=token;return &known;}
    const Receipt* foreground(uint64_t,uint64_t operation,std::string&){int32_t token=int32_t(1000+receipt_count_);commit(token);
        ++published_->metadata.cut.output_watermark;receipts_[receipt_count_]={operation,published_->serial,published_->metadata.cut.id,token};
        return &receipts_[receipt_count_++];}
    bool finish_request_current(uint64_t,std::string&){auto& m=published_->metadata;
        if(m.cut.accepted!=3+foreground_count_||m.cut.output_watermark!=receipt_count_)return false;
        m.approximation.clear();m.cut.exact_frontier=m.cut.forwarded;return true;}
    bool retire(std::string&){return true;}
};
struct ServingRawSession {
    bool open=true,permit=true;uint64_t quote=0;std::vector<int32_t> native_B{99,98};
    strata::core::SnapshotDescriptor captured;
    struct Result {bool ok=true;std::string detail;explicit operator bool()const{return ok;}};
    struct Geometry {int n_vocab=10000;} actual_geometry;
    const Geometry& geometry()const{return actual_geometry;}
    bool request_open()const{return open;}uint64_t max_tokens()const{return 512;}
    Result admit_host(uint64_t n,std::shared_ptr<const void>& credit){quote=n;if(!permit)return {false,"quota"};credit=std::make_shared<int>(1);return {};}
    Result capture(const strata::core::SnapshotDescriptor& m,strata::core::BundleHandle&){captured=m;return {};}
};
struct Harness {
    struct Outcome {
        std::shared_ptr<const void> request_credit;
        std::shared_ptr<Controller> controller=std::make_shared<Controller>();
        std::vector<int32_t> request_tokens{10,20,30,40};
        strata::core::SnapshotDescriptor published;strata::core::BundleHandle repaired_final;
        bool retired=false,yielded=true,certified_exact=false;
        uint64_t next_operation=5,next_cut=100,max_generated=2,generated=2,known_forwarded=1,exact_frontier=0;
        uint64_t request_receipt_base=0,request_known_base=0,request_watermark_base=0;
        std::chrono::steady_clock::time_point started;
    };
'''
END=r'''
};
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
int main(){try{
    ServingRawSession session;Harness::Outcome out;auto controller=out.controller;
    auto& initial=controller->published_->metadata;
    initial.accepted_tokens={10,20,30,40,1000,1001};initial.pending_tokens={1001};
    initial.cut.accepted=6;initial.cut.forwarded=5;initial.cut.exact_frontier=5;initial.cut.output_watermark=2;initial.cut.id=50;
    initial.cut.history=strata::core::window_history_digest(initial.accepted_tokens);
    controller->receipts_[0]={2,1,40,1000};controller->receipts_[1]={3,1,50,1001};out.published=initial;
    std::vector<int64_t> incoming(initial.accepted_tokens.begin(),initial.accepted_tokens.end());
    for(int i=0;i<59;++i)incoming.push_back(200+i);
    Harness::PreparedContinuationRequest pending;std::string error;
    // All pre-displacement refusals preserve B and the old immutable owner request/counters.
    auto unchanged=[&]{return out.request_tokens==std::vector<int32_t>{10,20,30,40}&&out.generated==2&&
        controller->receipt_count_==2&&controller->known_count_==1&&controller->published_->metadata.accepted_tokens.size()==6&&
        session.native_B==std::vector<int32_t>{99,98};};
    {auto wrong=incoming;wrong[5]=999;Harness::PreparedContinuationRequest refused;
        require(!Harness::prepare_continuation_request(session,out,wrong,3,refused,error)&&unchanged(),"changed actual pending accepted before displacement");}
    {Harness::PreparedContinuationRequest refused;
        require(!Harness::prepare_continuation_request(session,out,incoming,63,refused,error)&&unchanged(),"cumulative receipt capacity refusal");}
    {auto too_long=incoming;too_long.resize(512,1);Harness::PreparedContinuationRequest refused;
        require(!Harness::prepare_continuation_request(session,out,too_long,1,refused,error)&&unchanged(),"actual window bound refusal");}
    {session.permit=false;Harness::PreparedContinuationRequest refused;
        require(!Harness::prepare_continuation_request(session,out,incoming,3,refused,error)&&unchanged(),"host quota refusal");session.permit=true;}
    {const auto old=out.next_operation;out.next_operation=UINT64_MAX;Harness::PreparedContinuationRequest refused;
        require(!Harness::prepare_continuation_request(session,out,incoming,3,refused,error),"operation overflow refusal");out.next_operation=old;require(unchanged(),"overflow changed prior state");}
    require(Harness::prepare_continuation_request(session,out,incoming,3,pending,error),"actual pre-admission");
    require(out.request_tokens.size()==4&&out.generated==2&&session.native_B==std::vector<int32_t>{99,98},"pre-admission mutated prior owner/B");
    Harness::adopt_continuation_request(out,pending);
    require(out.request_receipt_base==2&&out.request_known_base==1&&out.request_watermark_base==2,"request-relative bases");
    require(out.request_tokens.size()==65&&out.max_generated==3&&out.generated==0&&out.known_forwarded==0,"immutable A3 request/budget");
    require(pending.tokens==std::vector<int32_t>{10,20,30,40}&&bool(out.request_credit),"old vector/new credit lifetime transfer");
    {Harness::Outcome* serving_suffix_active=&out;
    std::vector<int32_t> live{99,98};int32_t serving_pending_token=98;
    std::vector<int> live_imgs{9},checks{8};std::optional<int> incoming{7};bool borrow=true;
    int slot_source=2,slot_tokens=2,lent_slot=2;const int* slot_ck=&slot_source;
    struct {int chunk=0;} resume_plan;resume_plan.chunk=16;int64_t resume_plan_at=1,resume_plan_end=16,resume=0;
    bool from_live=false,live_ok=false,live_approximate=true,want_cvec=false,cvec_cached=true;
ACTUAL_HOST_REBIND
    require(live==std::vector<int32_t>({10,20,30,40,1000})&&serving_pending_token==1001,"actual A host/pending replaces B");
    require(!incoming&&!borrow&&slot_source==-1&&!slot_ck&&slot_tokens==0&&lent_slot==-1&&resume_plan.chunk==0&&
        resume_plan_at==-1&&resume_plan_end==-1&&resume==5&&from_live&&live_ok&&!live_approximate&&!cvec_cached,"stale reset/restore choices vetoed");}
    int32_t token=-1;require(!Harness::generate_one(out,token,error),"generation before actual appended known tail");
    while(out.published.cut.forwarded+1<out.request_tokens.size())require(Harness::forward_known(out,error),"actual known tail");
    require(out.known_forwarded==59&&controller->known_count_==60&&out.published.accepted_tokens.size()==65,"59 real known tokens with cumulative count");
    require(std::equal(out.published.accepted_tokens.begin(),out.published.accepted_tokens.end(),incoming.begin()),"actual appended59/pending preserved");
    for(int i=0;i<3;++i)require(Harness::generate_one(out,token,error),"new request generation");
    require(out.generated==3&&controller->receipt_count_==5&&out.next_operation==67,"new budget versus cumulative receipts/operations");
    require(controller->receipts_[0].token==1000&&controller->receipts_[1].token==1001,"historical A2 receipts unchanged");
    require(!Harness::generate_one(out,token,error),"new request budget not enforced");
    controller->private_pos=out.published.cut.forwarded; // recording seam supplies completed exact private work
    require(Harness::finish(session,out,error),"actual adapter finish with two-request history");
    require(out.retired&&out.certified_exact&&!out.controller&&session.captured.cut.output_watermark==5,"cumulative watermark and retirement");
    std::puts("PASS actual request adapter: immutable tail59, two cumulative receipt histories, request-relative budget/finish; CPU only");return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}}
'''


def run(build):
    root=Path(__file__).resolve().parents[3];text=(root/'src/program/generate.cpp').read_text()
    markers=[('    struct PreparedContinuationRequest {',True),('    static bool prepare_continuation_request(',False),
             ('    static void adopt_continuation_request(',False),('    static bool forward_known(',False),
             ('    static bool generate_one(',False),('    static bool exact_ready(',False),('    static bool finish(',False)]
    actual=''.join(take(text,marker,semi) for marker,semi in markers)
    source=build/'actual-retained-request.cpp';binary=build/'actual-retained-request'
    start=text.index("// Authoritative host state is now A's installed publication")
    end=text.index('live_approximate=!activated_state.approximation.empty();cvec_cached=want_cvec;',start)
    host=text[start:end+len('live_approximate=!activated_state.approximation.empty();cvec_cached=want_cvec;')]
    source.write_text(PRELUDE+actual+END.replace('ACTUAL_HOST_REBIND',host))
    subprocess.run(['c++','-std=c++20','-O0','-Wall','-Wextra','-I'+str(root/'include'),str(source),str(root/'src/core/window_hash.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=20)
    wrapper=take(text,'    static bool activate_retained_owner(')
    if not wrapper.index('prepare_continuation_request(')<wrapper.index('session.capture(prior_metadata, prior_image)')<wrapper.index('session.restore(payload.retained_publication, identity)')<wrapper.index('adopt_continuation_request('):
        raise ValueError('actual request admission/displacement/adoption order changed')
    start=text.index('// Authoritative host state is now A\'s installed publication')
    donor=text.index('auto sux_repaired =',start)
    if not start<donor:raise ValueError('host adoption after donor decision')
    for marker in ('incoming.reset();borrow=false;slot_source=-1;slot_ck=nullptr;', 'resume=int64_t(activated_state.cut.forwarded);from_live=true;live_ok=true;',
                   'while(resumed.published.cut.forwarded+1<resumed.request_tokens.size())'):
        if marker not in text:raise ValueError('actual activation disposition/known tail missing')
    if not text.index('while(resumed.published.cut.forwarded+1<resumed.request_tokens.size())')<text.index('if(serving_suffix_active)break;'):
        raise ValueError('legacy read skip precedes actual new known tail')
    shape_start=text.index('const bool retained_activation_shape_admissible=')
    shape_end=text.index('; // no owner-bound steering identity',shape_start)
    expression=text[shape_start:shape_end+1]
    expression=expression.replace('strata::kernels::cvec().loaded()','steering_loaded')
    shape_code='#include <vector>\nint main(){struct {bool pipeline_windows=false,adapt_every=false,adapt_async=false,prefill_head_alias=false,batch=false,native_serial_serving=true;int spec=0,lookup_chain=0,suffix_draft=0;} o;bool pipe=false,geni=false,use_mtp=false,batch_mtp=false,steering_loaded=false;int req_cvec=0,admit_slot=-1,max_new=3;bool want_cvec=steering_loaded?(req_cvec!=0):true;std::vector<int> bs;std::vector<int> req_imgs,req_bias;double req_temperature=0;int req_penalty_last_n=0;auto admit=[&](){'+expression+'return retained_activation_shape_admissible;};if(!want_cvec||!admit())return 1;req_temperature=.7;if(admit())return 2;req_temperature=0;req_penalty_last_n=8;if(admit())return 3;req_penalty_last_n=0;req_bias.push_back(1);if(admit())return 4;req_bias.clear();steering_loaded=true;if(admit())return 5;steering_loaded=false;want_cvec=true;if(!admit())return 6;geni=true;if(admit())return 7;geni=false;use_mtp=true;if(admit())return 8;use_mtp=false;req_cvec=0;if(!admit())return 9;return 0;}'
    source.write_text(shape_code)
    subprocess.run(['c++','-std=c++20',str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=20)
    if not shape_start<text.index('if (retained_activation_shape_admissible &&',shape_start)<text.index('::activate_retained_owner(',shape_start):
        raise ValueError('product shape guard is not before displacement')
    if 'yield_facts.registry_present=false;' in text:raise ValueError('qualification falsifies registry observation')
    if 'if(serving_repair_owners!=nullptr&&!force_activation_final_oracle){' not in text or 'if(!force_activation_final_oracle){\n                auto yield_refusal=' not in text:
        raise ValueError('explicit qualification final bypass missing; product yield must remain unchanged')
    print('PASS actual generic admission/displacement/adoption/host-reset-veto/known-tail and sampling/steering refusal; normal repeated yield retained')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build-dir',type=Path);args=parser.parse_args()
    if args.build_dir:args.build_dir.mkdir(parents=True,exist_ok=True);run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='retained-request-') as directory:run(Path(directory))


if __name__=='__main__':main()
