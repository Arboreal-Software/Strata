#!/usr/bin/env python3
"""Actual retained producer/capture/maker/publication/activation/resume CPU seam.

Production methods execute over recording dispatch/RAW boundaries, real metadata,
immutable BundleHandle freeze, digests, owner gates and ledger leases. No native
execution, device completion or numerical tensor proof is inferred.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[3]


def take(text,marker):
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
            if not depth:return text[start:at+1]+'\n'
    raise RuntimeError('unclosed production source '+marker)


TEMPLATE=r'''
#include "serving_repair_owners.hpp"
#include "strata/core/serving_raw_metadata.hpp"
#include <cstdio>
#include <functional>
#include <optional>
#include <stdexcept>
#include <thread>
using namespace strata::program;
namespace core=strata::core;
namespace strata::core {[[noreturn]] inline void retirement_failed(const char* why){throw std::runtime_error(why);}}
void require(bool result,const char* why){if(!result)throw std::runtime_error(why);}
core::BundleHandle freeze_recorded(core::SnapshotDescriptor meta,uint8_t value){
    meta.native_layout="cpu-recording";meta.steering_enabled=false;
    core::TensorLayout tensor;tensor.kind=core::StatePartKind::gdn_recurrence_and_convolution;
    tensor.layer=0;tensor.device_part=0;tensor.dtype="u8";tensor.native_layout="cpu-recording";
    tensor.shape={1};tensor.strides={1};tensor.bytes=tensor.valid_elements=1;
    core::WindowStateLayout layout;layout.native_layout=meta.native_layout;layout.model_profile=meta.key.model_profile;
    layout.topology=meta.key.topology;layout.layers={core::WindowStateLayout::LayerKind::gdn};
    layout.max_tokens=128;layout.max_payload_bytes=1;layout.steering_enabled=false;
    core::WindowStateLayout::Stage stage;stage.id=0;stage.layer_end=1;stage.tensors={tensor};layout.stages={stage};
    core::DevicePart device;device.layer_end=1;device.cut=meta.cut;
    device.tensors.push_back(core::TensorPart{tensor,core::ConversationBuffer{value}});meta.devices={device};
    core::BundleHandle image;auto result=core::BundleHandle::freeze(meta,meta.key,layout,image);
    require(bool(result),"real immutable host bundle freeze validates actual producer metadata");return image;
}
uint8_t payload(const core::BundleHandle& image){uint8_t value=0;
    require(image.get()->devices[0].tensors[0].payload.visit(0,1,[&](const uint8_t* p,size_t,size_t){value=*p;return true;}),"read recording payload");return value;}
struct Allocation {bool good=true;std::shared_ptr<const core::ReservationLease> lease;
    bool healthy()const{return good;}auto reservation()const{return lease;}};
struct Service {void* active_=nullptr;std::thread::id controller_=std::this_thread::get_id();};
struct Receipt {uint64_t operation=0;bool committed()const{return operation!=0;}};
struct Runtime;
struct Dispatch {Runtime* owner=nullptr;std::optional<Receipt> last_completion_;uint64_t last_completed_operation_=0;
    std::vector<int32_t> consumed;bool run_private_one(int32_t,uint64_t,int32_t&,std::string&);};
struct Runtime {Allocation allocation_;std::shared_ptr<Service> service_=std::make_shared<Service>();
    std::thread::id controller_=std::this_thread::get_id();std::unique_ptr<Dispatch> dispatch_=std::make_unique<Dispatch>();
    core::StateCut cut_{};uint64_t operation_=0;Runtime(){dispatch_->owner=this;}};
bool Dispatch::run_private_one(int32_t token,uint64_t,int32_t& output,std::string&){
    consumed.push_back(token);++owner->operation_;last_completed_operation_=owner->operation_;
    last_completion_=Receipt{owner->operation_};output=-1;return true;}
struct SuffixRawTransaction {
    enum class Phase {idle,running,drained,capturing,restoring,poisoned};
    Phase phase_=Phase::drained;core::StateCut cut_{};core::WindowKey key_{};Allocation allocation_;
    uint8_t value=11;int captures=0,restores=0;core::SnapshotDescriptor last_capture;
    std::function<void()> on_capture,on_restore;
    bool begin_driver_work(){if(phase_!=Phase::drained)return false;phase_=Phase::running;return true;}
    bool identity_matches(const core::WindowKey& key)const{return same_stable_window(key_,key);}
    core::StateCut cut()const{return cut_;}
    bool capture(const core::SnapshotDescriptor& meta,core::BundleHandle& out){
        if(phase_!=Phase::drained||meta.cut!=cut_||!core::serving_raw::capture_metadata_acceptable(meta))return false;
        last_capture=meta;out=freeze_recorded(meta,value);++captures;if(on_capture)on_capture();return true;}
    bool restore(const core::BundleHandle& image,const core::WindowKey& key){
        if(phase_!=Phase::drained||!image||!same_stable_window(key_,key)||image.get()->key!=key)return false;
        cut_=image.get()->cut;value=payload(image);++restores;if(on_restore)on_restore();return true;}
};
class ResidentRepairController {public:
    struct Published {core::SnapshotDescriptor metadata;uint64_t serial=2;};
    struct ExactCompletion {core::StateCut cut;Receipt receipt;ExactCompletion(core::StateCut c,Receipt&& r):cut(c),receipt(std::move(r)){};};
    std::shared_ptr<Runtime> runtime_=std::make_shared<Runtime>();SuffixRawTransaction& live_;
    std::unique_ptr<SuffixRawTransaction> private_=std::make_unique<SuffixRawTransaction>();
    std::shared_ptr<Published> published_=std::make_shared<Published>();
    std::vector<int32_t> private_history_;core::WindowKey private_key_{};
    std::optional<ExactCompletion> exact_completion_;RetainedRecordFacts retained_record_;PreparedRetainedResult retained_result_;
    bool busy_=false,retired_=false,cancel_requested_=false,finished_=false,suspended_=true;uint64_t next_cut_=100;
    bool resume_observed_idle=false;int resume_calls=0;
    explicit ResidentRepairController(SuffixRawTransaction& live):live_(live){}
    [[noreturn]] static void fatal(const char* why){throw std::runtime_error(why);}
    uint64_t mint_cut(){return next_cut_++;}
    void complete(SuffixRawTransaction& tx,const core::StateCut& cut){
        require(tx.phase_==SuffixRawTransaction::Phase::running,"recorded dispatch completes only running work");
        tx.cut_=cut;tx.phase_=SuffixRawTransaction::Phase::drained;}
ACTUAL_PAUSE
ACTUAL_RAW_PHASE
ACTUAL_OWNER_EVIDENCE
ACTUAL_ADVANCE
ACTUAL_YIELD
ACTUAL_PRIVATE_UNIT
ACTUAL_CAPTURE
ACTUAL_PREPARE
ACTUAL_PUBLISH
ACTUAL_ACTIVATE
ACTUAL_RESUME
    bool resume_retained(uint64_t expected,std::string& error){
        ++resume_calls;resume_observed_idle=!busy_&&runtime_->service_->active_==nullptr;
        return actual_resume_retained(expected,error);
    }
};
struct Owner {
    std::shared_ptr<ResidentRepairController> controller;
    bool retired=false,yielded=true;core::SnapshotDescriptor published;core::StateCut certified_cut;
    uint64_t retained_debt=4,retained_residency_swaps=7,next_cut=0;core::BundleHandle retained_exact;
    bool populated()const{return bool(controller);}
    RepairOwnerEvidence owner_evidence_snapshot()const{return controller->owner_evidence();}
};
struct Adapter {
    using Outcome=Owner;
ACTUAL_ADAPTER
};
struct Hook {std::function<bool(const core::BundleHandle&)> rollback;bool rolled_back=false;};
struct Outer {
    [[noreturn]] static void failstop(SuffixRawTransaction&,const char* why){throw std::runtime_error(why);}
    static bool finish(bool landed,SuffixRawTransaction& session,const ConversationStateFacts& on_publication,
        const ConversationStateFacts& prior,const core::BundleHandle& prior_image,
        ConversationStateFacts& restored_out,Hook* qualification){
ACTUAL_OUTER
    }
};
core::SnapshotDescriptor descriptor(core::WindowKey key,std::vector<int32_t> tokens,uint64_t id,uint64_t forwarded,
    uint64_t exact,uint64_t watermark){core::SnapshotDescriptor out;out.key=std::move(key);out.accepted_tokens=std::move(tokens);
    out.cut={id,out.accepted_tokens.size(),forwarded,exact,watermark,core::window_history_digest(out.accepted_tokens)};
    out.pending_tokens.assign(out.accepted_tokens.begin()+std::ptrdiff_t(forwarded),out.accepted_tokens.end());out.steering_enabled=false;
    if(exact<forwarded){out.approximation.push_back({exact,forwarded,100,out.cut.history});}
    return out;
}
void run_case(bool refuse_resume){
    core::WindowResources capacity{{0},4096,4096,0,0,4},baseline{{0},0,0,0,0,0};core::WindowBudget budget(capacity,baseline);
    auto charge=budget.reserve(core::WindowResources{{0},256,0,0,0,1},core::ReservationPurpose::private_work);
    require(bool(charge),"real private-work reservation");
    core::WindowKey key;key.engine_generation=key.window_generation=key.revision=key.append_epoch=key.selection_epoch=key.provenance_epoch=1;
    key.session_id="owned";key.thread_id="thread";key.window_id="session";key.model_profile[0]=1;key.topology[0]=2;
    std::vector<int32_t> a_tokens;for(int32_t token=1;token<=13;++token)a_tokens.push_back(token);
    auto a=descriptor(key,a_tokens,20,12,4,3);auto b_key=key;b_key.revision=9;
    auto b=descriptor(b_key,{41,42,43,44,45,46,47,48},901,7,7,9);
    SuffixRawTransaction live;live.key_=b.key;live.cut_=b.cut;live.value=73;
    ResidentRepairController c(live);c.runtime_->allocation_.lease=charge.lease;c.published_->metadata=a;
    c.private_->key_=key;c.private_->allocation_.lease=charge.lease;
    c.private_history_.assign(a_tokens.begin(),a_tokens.begin()+8);
    c.private_->cut_={2,8,8,8,3,core::window_history_digest(c.private_history_)};
    c.exact_completion_.emplace(c.private_->cut_,Receipt{1});c.private_key_=key;
    // Exercise the actual yield producer: its mirror must never acquire fabricated registry admission.
    live.key_=key;live.cut_=a.cut;live.value=11;c.suspended_=false;
    std::string error;core::StateCut certified;core::BundleHandle initial_exact;
    ServingRepairOwnerRegistry<Owner> registry(1);
    RetainedActivationResources resources;
    auto image_charge=budget.reserve(core::WindowResources{{0},16,0,0,0,0},core::ReservationPurpose::private_attachment,charge.lease);
    auto restore_charge=budget.reserve(core::WindowResources{{0},16,0,0,0,0},core::ReservationPurpose::private_attachment,charge.lease);
    core::WindowReturnObligation return_obligation{"actual-recording-B-return",core::WindowResources{{0},16,0,0,0,0}};
    auto prior_charge=budget.reserve(return_obligation.required,core::ReservationPurpose::foreground_return,{},&return_obligation);
    require(bool(image_charge)&&bool(restore_charge)&&bool(prior_charge),"actual ledger activation resources admitted with parent/return obligation");
    resources.image_charge=image_charge.lease;resources.restore_charge=restore_charge.lease;resources.prior_capture_charge=prior_charge.lease;
    resources.image_purpose=resources.restore_purpose=core::ReservationPurpose::private_attachment;
    resources.prior_capture_purpose=core::ReservationPurpose::foreground_return;
    resources.image_bytes=resources.prior_capture_bytes=resources.restore_bytes=16;
    RetainedCapacityTicket ticket;
    if(!registry.reserve_retained_capacity(key,resources,ticket,error))throw std::runtime_error("actual registry reserves capacity: "+error);
    require(c.yield_retained(2,certified,initial_exact,error),"actual yield produces suspended owner");
    require(!c.retained_record_.capacity_reserved&&!c.retained_record_.resources_pinned,
        "actual yield mirror deliberately has no registry reservation");
    auto owner=std::make_unique<Owner>();owner->controller=std::shared_ptr<ResidentRepairController>(&c,[](auto*){});
    owner->published=a;owner->certified_cut=certified;
    ParkOwnerFacts park;park.owner=c.owner_evidence();park.session.session_key=key;
    park.session.session_cut=a.cut;park.session.live_phase=core::serving_raw::Phase::drained;
    park.capacity=ticket;park.resources=resources;
    require(registry.park_retained_owner(key,owner,park,error),"actual registry parks genuine yield");
    live.key_=b.key;live.cut_=b.cut;live.value=73;
    LiveSessionFacts idle;idle.session_key=key;idle.session_cut=b.cut;idle.live_phase=core::serving_raw::Phase::drained;
    idle.private_window_open=true;idle.residency_generation=7;
    RepairOwnerEvidence unit_after;uint64_t advanced=0;
    require(registry.advance_parked_owner(key,idle,4,
        [&](Owner& payload,uint64_t& rows,RepairOwnerEvidence& after,std::string& why){
            const bool ok=payload.controller->retained_private_unit(2,4,rows,why);
            after=payload.controller->owner_evidence();return ok;},unit_after,advanced,error)&&advanced==4,
        "actual registry drives controller private unit at authenticated nonzero residency");
    require(c.runtime_->dispatch_->consumed==std::vector<int32_t>({9,10,11,12}),"actual private producer consumed exactly outstanding rows");
    core::BundleHandle exact;require(c.capture_exact(exact,error),"actual capture_exact accepts honest endpoint prefix");
    require(exact.get()->cut.accepted==12&&exact.get()->cut.forwarded==12&&exact.get()->cut.exact_frontier==12&&
        exact.get()->accepted_tokens.size()==12&&exact.get()->pending_tokens.empty()&&
        exact.get()->cut.history==core::window_history_digest(c.private_history_),"private image does not fabricate publication pending history");
    RepairOwnerRefusal why=RepairOwnerRefusal::none;
    auto result=c.prepare_retained_result(2,exact,why);
    require(result.produced()&&result.accepted_sequence()==12&&result.pending_token()==13,
        "actual consumed-only producer capture reaches real maker with publication pending kept separate");
    auto receipt=c.exact_completion_->receipt;c.exact_completion_->receipt={};
    require(!c.prepare_retained_result(2,exact,why).produced(),"missing completion witness refuses");c.exact_completion_->receipt=receipt;
    require(!c.prepare_retained_result(2,{},why).produced(),"missing actual image refuses");
    auto wrong_metadata=*exact.get();++wrong_metadata.cut.id;auto wrong_image=freeze_recorded(wrong_metadata,11);
    require(!c.prepare_retained_result(2,wrong_image,why).produced(),"actual frozen image of another cut refuses");
    ++c.exact_completion_->cut.id;
    require(!c.prepare_retained_result(2,exact,why).produced(),"committed witness for another operation cut refuses");
    --c.exact_completion_->cut.id;
    auto saved_pending=c.published_->metadata.pending_tokens;c.published_->metadata.pending_tokens.clear();
    require(!c.prepare_retained_result(2,exact,why).produced(),"missing genuine publication pending refuses");c.published_->metadata.pending_tokens=saved_pending;
    PreparedRetainedResult published_result;
    require(!c.publish_retained_if_current(1,exact,published_result,error),"actual publication refuses stale owner record version");
    require(c.publish_retained_if_current(2,exact,published_result,error),"actual publication accepts producer's current zero-debt result");
    require(live.cut_==b.cut&&live.value==73&&live.captures==0,"private repair and result publication leave B untouched");
    core::BundleHandle b_image;require(live.capture(b,b_image),"outer actual recording capture retains different B");
    auto a_image=freeze_recorded(a,11);require(live.restore(a_image,key),"outer installs actual A publication before controller activation");
    ConversationStateFacts on_publication{key,a.cut,a.cut},prior{b.key,b.cut,b.cut},restored;
    LiveSessionFacts session;session.session_key=key;session.session_cut=a.cut;session.live_phase=core::serving_raw::Phase::drained;session.request_open=true;
    c.busy_=true;require(!c.resume_retained(2,error)&&c.suspended_,"public resume still rejects busy caller");c.busy_=false;
    live.on_capture=[&]{require(c.busy_&&c.runtime_->service_->active_,"capture owns actual busy/Pause scope");if(refuse_resume)c.cancel_requested_=true;};
    live.on_restore=[&]{require(c.busy_&&c.runtime_->service_->active_,"inner refusal restore owns fresh bounded busy/Pause scope");};
    PreparedRetainedResult registry_result;
    require(registry.publish_retained_result_from_owner(key,
        [&](Owner& payload,uint64_t version,PreparedRetainedResult& result,std::string&){
            RepairOwnerRefusal refusal;result=payload.controller->prepare_retained_result(version,exact,refusal);
            return result.produced();},registry_result,error),"actual registry publishes current result");
    session.residency_generation=7;
    // Give the real serving adapter its actual held endpoint image; no evidence is manufactured by this fixture.
    RetainedActivationAuthority authority;
    std::unique_ptr<Owner> returned;ConversationStateFacts registry_restored;RepairOwnerEvidence registry_after;
    std::vector<OwnerReservationHandle> returned_charges;
    require(!registry.activate_parked_owner(key,session,on_publication,
        [&](Owner& payload,const RetainedActivationFacts& facts,RepairOwnerEvidence& after,
            ConversationStateFacts& back,std::string& why){
            payload.retained_exact=exact;
            authority=facts.authority;after=payload.controller->owner_evidence();back=on_publication;
            why="deliberate pre-mutation refusal";return false;
        },returned,registry_restored,registry_after,returned_charges,error),"issued authority refusal keeps parked owner");
    require(authority.issued()&&authority.record().versions.residency_generation==7&&
        authority.record().capacity_reserved&&authority.record().resources_pinned,
        "authority carries actual admitted registry record and nonzero residency");
    require(!c.activate_retained(2,exact,session,on_publication,a,restored,error),"unissued authority refuses before mutation");
    const auto before_authority_refusals=live.captures;
    ++c.retained_record_.record_version;
    require(!c.activate_retained(2,exact,session,on_publication,a,restored,error,authority),
        "issued stale authority refuses moved controller record");
    --c.retained_record_.record_version;
    auto wrong_session=session;wrong_session.residency_generation=8;
    require(!c.activate_retained(2,exact,wrong_session,on_publication,a,restored,error,authority),
        "issued authority refuses actual residency divergence");
    require(live.captures==before_authority_refusals&&c.suspended_&&live.cut_==a.cut,
        "authority refusals happen before capture or owner/session mutation");
    const bool landed=registry.activate_parked_owner(key,session,on_publication,
        [&](Owner& payload,const RetainedActivationFacts& facts,RepairOwnerEvidence& after,
            ConversationStateFacts& back,std::string& why){
            return Adapter::activate_retained(payload,session,facts,on_publication,a,back,after,why);
        },returned,restored,registry_after,returned_charges,error);
    require(!c.retained_record_.capacity_reserved&&!c.retained_record_.resources_pinned,
        "controller never fabricates registry reservation in mirror");
    live.on_capture=nullptr;live.on_restore=nullptr;
    require(!c.busy_&&!c.runtime_->service_->active_,"all activation busy/Pause scopes ended");
    require(c.resume_observed_idle&&c.resume_calls==2,"actual public resume observes ended busy AND Pause scopes");
    Hook hooks;hooks.rollback=[&](const core::BundleHandle& before){core::BundleHandle after;
        return live.capture(*before.get(),after)&&after.get()->cut==before.get()->cut&&payload(after)==payload(before);};
    if(refuse_resume){
        require(!landed&&c.suspended_&&restored==on_publication,"actual resume refusal preserves suspended owner and captured A report");
        SuspendedOwnerFacts refused_record;
        require(registry.entry_facts(key,refused_record)&&!refused_record.quarantined&&
            error.find("coherence conjunction")!=std::string::npos,
            "actual adapter false-path evidence preserves parked record and original refusal diagnosis");
        require(!Outer::finish(landed,live,on_publication,prior,b_image,restored,&hooks)&&hooks.rolled_back&&
            live.cut_==b.cut&&live.value==73&&restored==prior,"actual outer refusal physically restores B and runs capture oracle");
    }else{
        require(landed&&!c.suspended_,"actual complete publication activation resume succeeds");
        ConversationStateFacts replay_restored;
        require(!c.activate_retained(2,exact,session,on_publication,a,replay_restored,error,authority)&&
            live.cut_==a.cut,"consumed authority cannot replay a second activation of now-live owner");
        require(c.published_->metadata.cut==a.cut&&c.published_->metadata.pending_tokens==std::vector<int32_t>{13}&&
            restored==on_publication&&live.cut_==a.cut,"activation preserves exact publication metadata and genuine pending");
        auto bad=restored;bad.cut.output_watermark++;
        bool refused=false;try{(void)Outer::finish(true,live,on_publication,prior,b_image,bad,&hooks);}catch(const std::runtime_error&){refused=true;}
        require(refused,"actual outer success branch rejects corrupt inner captured-A report");
        require(Outer::finish(true,live,on_publication,prior,b_image,restored,&hooks)&&restored==prior&&
            live.cut_==a.cut&&live.value==11&&!hooks.rolled_back,"actual outer success accounts retained B without falsely restoring physical B");
    }
    require(registry.retire_all("fixture cleanup",[](Owner& payload,std::string&){
        payload.controller.reset();payload.retired=true;return true;},error),"explicit fixture retirement");
}
int main(){try{run_case(false);run_case(true);
    std::puts("PASS actual producer/capture/maker/publication/activation/resume and outer A/B success/refusal seams; CPU recording only");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
'''


def run(build):
    build=Path(build);build.mkdir(parents=True,exist_ok=True)
    controller=(ROOT/'src/program/resident_repair_controller.inc').read_text()
    generate=(ROOT/'src/program/generate.cpp').read_text()
    methods={
        'ACTUAL_ADAPTER':take(generate,'    static bool activate_retained(Outcome& out,'),
        'ACTUAL_PAUSE':take(controller,'    struct Pause ' )+';',
        'ACTUAL_RAW_PHASE':take(controller,'    static core::serving_raw::Phase raw_phase('),
        'ACTUAL_OWNER_EVIDENCE':take(controller,'    RepairOwnerEvidence owner_evidence()const'),
        'ACTUAL_ADVANCE':take(controller,'    void advance_private_to('),
        'ACTUAL_YIELD':take(controller,'    bool yield_retained('),
        'ACTUAL_PRIVATE_UNIT':take(controller,'    bool retained_private_unit('),
        'ACTUAL_CAPTURE':take(controller,'    bool capture_exact('),
        'ACTUAL_PREPARE':take(controller,'    PreparedRetainedResult prepare_retained_result('),
        'ACTUAL_PUBLISH':take(controller,'    bool publish_retained_if_current('),
        'ACTUAL_ACTIVATE':take(controller,'    bool activate_retained('),
        'ACTUAL_RESUME':take(controller,'    bool resume_retained(').replace('bool resume_retained(', 'bool actual_resume_retained(',1),
    }
    wrapper=take(generate,'    static bool activate_retained_owner(')
    begin=wrapper.index('                if (landed) {')
    end=wrapper.index('            },\n',begin)
    methods['ACTUAL_OUTER']=wrapper[begin:end]
    code=TEMPLATE
    for marker,method in methods.items():code=code.replace(marker,method)
    header=(ROOT/'src/program/serving_repair_owners.hpp').read_text()
    old_check='''    if (facts.certified_cut.forwarded == facts.published_cut.forwarded &&
        facts.certified_cut.accepted == facts.certified_cut.forwarded)
        return refuse(RepairOwnerRefusal::refused_prepared_result_invalid);
'''
    marker='    if (facts.accepted_sequence != facts.certified_cut.accepted || facts.pending_token < 0)'
    if header.count(marker)!=1:raise RuntimeError('maker negative-control anchor changed')
    variants=[('production',code,None),
        ('old_missing_refusal_evidence',code.replace('        after = c.owner_evidence();\n        if (!activated) return false;',
            '        if (!activated) return false;\n        after = c.owner_evidence();'),None),
        ('old_zero_pending_refusal',code,'maker'),
        ('old_busy_resume',code.replace('            busy_=false;\n            std::string resume_error;',
                                       '            busy_=true;\n            std::string resume_error;'),None),
        ('wrong_outer_prior',code.replace('                    restored_out = prior;\n                    return true;',
                                         '                    return true;'),None)]
    for label,actual,overlay in variants:
        includes=['-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/program')]
        if overlay:
            directory=build/overlay;directory.mkdir(exist_ok=True)
            (directory/'serving_repair_owners.hpp').write_text(header.replace(marker,old_check+marker,1))
            includes.insert(0,'-I'+str(directory))
        elif label!='production' and actual==code:raise RuntimeError(label+' did not change production')
        cpp=build/(label+'.cpp');binary=build/label;cpp.write_text(actual)
        subprocess.run(['c++','-std=c++20','-O0','-Wall','-Wextra',*includes,str(cpp),
            str(ROOT/'src/core/window_hash.cpp'),str(ROOT/'src/core/window_state.cpp'),
            str(ROOT/'src/core/window_budget.cpp'),'-o',str(binary)],check=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
        if (result.returncode==0)!=(label=='production'):
            raise RuntimeError(f'{label} unexpected exit {result.returncode}: '+result.stderr)
        print('PASS '+label+(' negative control fails' if label!='production' else ': '+result.stdout.strip()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path);args=parser.parse_args()
    if args.build_dir:run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-controller-activation-') as directory:run(directory)
