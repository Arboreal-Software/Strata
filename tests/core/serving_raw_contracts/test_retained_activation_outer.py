#!/usr/bin/env python3
"""Full actual activation outer adapter over recording native boundaries and real registry/controller.
CPU only: no native tensor/numerical admission is inferred.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
import test_retained_controller_activation as chain

ROOT=chain.ROOT

def code():
    controller=(ROOT/'src/program/resident_repair_controller.inc').read_text()
    generate=(ROOT/'src/program/generate.cpp').read_text()
    source=chain.TEMPLATE
    source=source.replace('#include <thread>','#include <thread>\n#include <span>\n#include <algorithm>')
    # Match actual RAW semantics: key() is immutable runtime identity; restore validates the full image key but
    # only installs its native state/cut. The original chain fixture's transaction now preserves that identity.
    source=source.replace('    bool capture(const core::SnapshotDescriptor& meta,core::BundleHandle& out){','''    struct Result {bool ok;std::string detail;operator bool()const{return ok;}};
    bool request=false;bool private_window=false;
    const core::WindowKey& key()const{return key_;}
    Phase phase()const{return phase_;}
    bool request_open()const{return request;}
    bool private_window_open()const{return private_window;}
    bool open_request(std::string&){if(request||phase_!=Phase::drained)return false;request=true;return true;}
    bool close_request(std::string&){if(!request||phase_!=Phase::drained)return false;request=false;return true;}
    static core::serving_raw::Phase raw_phase(Phase p){return p==Phase::drained?core::serving_raw::Phase::drained:core::serving_raw::Phase::poisoned;}
    Result capture(const core::SnapshotDescriptor& meta,core::BundleHandle& out){''')
    source=source.replace('return false;\n        last_capture=meta;out=freeze_recorded(meta,value);++captures;if(on_capture)on_capture();return true;}','return {false,"capture refused"};\n        last_capture=meta;out=freeze_recorded(meta,value);++captures;if(on_capture)on_capture();return {true,{}};}')
    source=source.replace('    bool restore(const core::BundleHandle& image,const core::WindowKey& key){','    Result restore(const core::BundleHandle& image,const core::WindowKey& key){')
    source=source.replace('image.get()->key!=key)return false;','image.get()->key!=key)return {false,"restore exact key refused"};')
    source=source.replace('++restores;if(on_restore)on_restore();return true;}','++restores;if(on_restore)on_restore();return {true,{}};}')
    source=source.replace('core::BundleHandle retained_exact;','core::BundleHandle retained_exact,retained_publication;')
    source=source.replace('struct Adapter {','''using ServingRawSession=SuffixRawTransaction;
class Adapter;
using ServingSuffixRepairOperation=Adapter;
struct Adapter {''')
    source=source.replace('ACTUAL_ADAPTER','''ACTUAL_ADAPTER
    struct PreparedContinuationRequest {bool admitted=false;};
    static bool prepare_continuation_request(ServingRawSession& session,const Owner&,
        std::span<const int64_t>,uint64_t,PreparedContinuationRequest& p,std::string&){p.admitted=session.request_open();return p.admitted;}
    static void adopt_continuation_request(Owner&,PreparedContinuationRequest& p){require(p.admitted,"real wrapper adopts only admitted request");}
    static RepairOwnerEvidence owner_evidence(const Owner& owner){return owner.controller->owner_evidence();}
    [[noreturn]] static void failstop(ServingRawSession&,const char* why){throw std::runtime_error(why);}
ACTUAL_PREPARE_ADAPTER
ACTUAL_CONVERSATION_FACTS
ACTUAL_LIVE_FACTS
ACTUAL_HOOKS;
ACTUAL_OUTER_WRAPPER''')
    begin=source.index('void run_case(bool refuse_resume){');end=source.index('\nint main()',begin)
    prefix=source[begin:source.index('    RepairOwnerRefusal why=',begin)]
    prefix=prefix.replace('void run_case(bool refuse_resume){','void run_case(int failure){')
    prefix=prefix.replace('auto a=descriptor(key,a_tokens,20,12,4,3);','auto stable_key=key;++key.revision;++key.append_epoch;++key.selection_epoch;\n    auto a=descriptor(key,a_tokens,20,12,4,3);')
    # Both runtime transactions keep the stable identity through all actual restores; publication retains epochs.
    prefix=prefix.replace('live.key_=key;live.cut_=a.cut;','live.key_=stable_key;live.cut_=a.cut;')
    prefix=prefix.replace('live.key_=b.key;live.cut_=b.cut;','live.key_=stable_key;live.cut_=b.cut;')
    # The production core sees actual private/full publication identity through own metadata, not runtime key().
    prefix=prefix.replace('owner->published=a;owner->certified_cut=certified;','owner->published=a;owner->certified_cut=certified;owner->retained_publication=freeze_recorded(a,11);')
    tail=r'''
    require(live.key()==stable_key&&live.key()!=key,"fixture distinguishes stable RAW identity from publication epochs");
    registry.inspect_parked_owner(key,[&](const Owner& held){
        auto& payload=const_cast<Owner&>(held);payload.retained_exact=exact;
        if(failure==1)payload.retained_publication={};
        if(failure==2){auto wrong=a;++wrong.key.append_epoch;payload.retained_publication=freeze_recorded(wrong,11);}
        return true;
    });
    bool perturb_restore=true;
    if(failure==3)live.on_restore=[&]{if(perturb_restore&&live.value==11){perturb_restore=false;++live.cut_.id;}};
    if(failure==4)live.on_capture=[&]{if(live.last_capture.key==key)c.cancel_requested_=true;};
    core::serving_raw::TrackedState installed;installed.cut=b.cut;installed.accepted=b.accepted_tokens;
    installed.approximation=b.approximation;
    std::unique_ptr<Owner> returned;
    std::vector<int64_t> actual_request(a_tokens.begin(),a_tokens.end());actual_request.push_back(99);
    const bool activated=Adapter::activate_retained_owner(live,registry,key,installed,7,returned,error,nullptr,actual_request,4);
    if(failure==0){
        require(activated&&returned&&live.request_open()&&!c.suspended_&&registry.size()==0,"full real outer wrapper activates registry owner and keeps fence open");
        require(live.key()==stable_key&&live.cut_==a.cut&&live.value==11,"actual A image installed without rewriting RAW identity");
        require(c.published_->metadata.key==key,"actual published full epochs retained exactly");
        require(live.close_request(error),"successful test request closes real fence");
    }else{
        SuspendedOwnerFacts record;
        require(!activated&&!returned&&!live.request_open()&&registry.entry_facts(key,record)&&!record.quarantined,
            "every clean outer refusal preserves real parked evidence and closes fence");
        require(live.key()==stable_key&&live.cut_==b.cut&&live.value==73,"every outer refusal restores physical B cut and bytes");
        if(failure==4)require(error.find("coherence conjunction")!=std::string::npos,"actual inner resume refusal survives outer rollback diagnosis");
    }
    require(registry.retire_all("fixture cleanup",[](Owner& held,std::string&){held.controller.reset();held.retired=true;return true;},error),"explicit real registry cleanup");
}
'''
    source=source[:begin]+prefix+tail+source[end:]
    main_begin=source.index('int main()');main_end=source.index("\n'''") if "\n'''" in source else len(source)
    source=source[:main_begin]+'''int main(){try{for(int failure=0;failure<=4;++failure)run_case(failure);
    std::puts("PASS full actual outer adapter+registry+controller with stable RAW identity and bumped publication epochs; CPU recording only");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\\n",e.what());return 1;}}
'''
    methods={
        'ACTUAL_PAUSE':chain.take(controller,'    struct Pause ')+';',
        'ACTUAL_RAW_PHASE':chain.take(controller,'    static core::serving_raw::Phase raw_phase('),
        'ACTUAL_OWNER_EVIDENCE':chain.take(controller,'    RepairOwnerEvidence owner_evidence()const'),
        'ACTUAL_ADVANCE':chain.take(controller,'    void advance_private_to('),
        'ACTUAL_YIELD':chain.take(controller,'    bool yield_retained('),
        'ACTUAL_PRIVATE_UNIT':chain.take(controller,'    bool retained_private_unit('),
        'ACTUAL_CAPTURE':chain.take(controller,'    bool capture_exact('),
        'ACTUAL_PREPARE':chain.take(controller,'    PreparedRetainedResult prepare_retained_result('),
        'ACTUAL_PUBLISH':chain.take(controller,'    bool publish_retained_if_current('),
        'ACTUAL_ACTIVATE':chain.take(controller,'    bool activate_retained('),
        'ACTUAL_RESUME':chain.take(controller,'    bool resume_retained(').replace('bool resume_retained(','bool actual_resume_retained(',1),
        'ACTUAL_ADAPTER':chain.take(generate,'    static bool activate_retained(Outcome& out,'),
        'ACTUAL_PREPARE_ADAPTER':chain.take(generate,'    static bool prepare_retained_result(const Outcome& out,'),
        'ACTUAL_CONVERSATION_FACTS':chain.take(generate,'    static strata::program::ConversationStateFacts conversation_facts('),
        'ACTUAL_LIVE_FACTS':chain.take(generate,'    static LiveSessionFacts live_session_facts('),
        'ACTUAL_HOOKS':chain.take(generate,'    struct ActivationQualificationHooks '),
        'ACTUAL_OUTER_WRAPPER':chain.take(generate,'    static bool activate_retained_owner('),
        'ACTUAL_OUTER':'return false;',
    }
    # Longest markers first: ACTUAL_PREPARE is a prefix of ACTUAL_PREPARE_ADAPTER.
    for marker in sorted(methods,key=len,reverse=True):source=source.replace(marker,methods[marker])
    return source

def run(build):
    build=Path(build);build.mkdir(parents=True,exist_ok=True);production=code()
    variants=[('production',production),
        ('old_runtime_key_equals_publication',production.replace('!strata::program::same_stable_window(session.key(), payload.published.key)','session.key() != payload.published.key')),
        ('old_inner_prior_runtime_key',production.replace('                on_publication.key = payload.retained_publication.get()->key;','')),
        ('missing_early_refusal_evidence',production.replace('                after_evidence = ServingSuffixRepairOperation::owner_evidence(payload);\n                restored_out = prior;',''))]
    for name,source in variants:
        if name!='production' and source==production:raise RuntimeError('negative did not alter source: '+name)
        cpp=build/(name+'.cpp');binary=build/name;cpp.write_text(source)
        subprocess.run(['c++','-std=c++20','-O0','-Wall','-Wextra','-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/program'),str(cpp),str(ROOT/'src/core/window_hash.cpp'),str(ROOT/'src/core/window_state.cpp'),str(ROOT/'src/core/window_budget.cpp'),'-o',str(binary)],check=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
        if (result.returncode==0)!=(name=='production'):raise RuntimeError(f'{name}: '+result.stderr)
        print('PASS '+name+(' (old-bug negative fails)' if name!='production' else ': '+result.stdout.strip()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path);args=parser.parse_args()
    if args.build_dir:run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory() as directory:run(directory)
