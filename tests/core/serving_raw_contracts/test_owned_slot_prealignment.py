#!/usr/bin/env python3
"""Extract real RAW preparation/completion and real two-stage group issuer.
Native allocation/math/drain and source-image capture are recording boundaries.
The source anchor is metadata from the first authentic group cut, NOT proof of a
native donor image or a usable suffix operand. No suffix import is certified here.
"""
import argparse,hashlib,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[3]

def load(path):
 spec=importlib.util.spec_from_file_location('actual_owned_slot_fixture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def generate(base):
 fixture=load(base/'tests/core/owned_batch_slot_sessions/test.py')
 fixture.ROOT=base
 fixture.run_stop_parser=lambda directory:None
 class Collected(Exception):pass
 def stop(*args,**kw):raise Collected()
 with tempfile.TemporaryDirectory() as directory:
  try:
   with patch.object(fixture.subprocess,'run',side_effect=stop):fixture.run(Path(directory))
  except Collected:pass
  code=(Path(directory)/'production.cpp').read_text()
 take=fixture.take;raw=(base/'src/program/suffix_raw_runtime.cpp').read_text();header=(base/'src/program/suffix_raw_runtime.hpp').read_text()
 declarations=take(header,'    struct CohortReceiptView ')+''';
    std::shared_ptr<const void> alignment_owner_credit_;
    core::BundleHandle alignment_donor_,slot_source_provenance_,slot_donor_provenance_;
    bool recording_capture_success=false;
    uint64_t recording_capture_calls=0;
    SuffixRawResult capture_owned_slot(core::BundleHandle&);
    core::WindowKey edited_key()const noexcept;
    core::WindowKey recording_install_key(const core::BundleHandle&,bool);
    std::unique_ptr<core::WindowKey> alignment_edit_origin_;
    static void recording_consumed_bit(const core::NativeBatchGroupCompletion& r,size_t slot,bool value){r.consumed_[slot].store(value);}
    const core::StateCut& cut()const{return tx_->cut_;}
    uint64_t max_tokens()const{return tx_->max_tokens_;}
    static core::SnapshotDescriptor endpoint_metadata(const core::WindowKey&,const core::StateCut&,const std::vector<int32_t>&);
    SuffixRawResult admit_host(uint64_t,std::shared_ptr<const void>&);
    const core::WindowKey* suffix_donor_key(const core::BundleHandle&)const noexcept;
    bool prepare_owned_alignment_known(std::shared_ptr<const std::vector<int64_t>>,std::shared_ptr<const void>,std::string&);
    bool cancel_owned_alignment_known(std::string&);
    bool complete_owned_alignment_known(const core::NativeBatchGroupCompletion&,std::string&);
    bool inspect_cohort_receipt(const core::NativeBatchGroupCompletion&,const core::SnapshotDescriptor&,int32_t,CohortReceiptView&,std::string&);
'''
 code=code.replace('SuffixRawResult capture(const SnapshotDescriptor&,BundleHandle&){transfer_capture_reached=true;return invalid("recorded refusal at native capture boundary");}',
     'SuffixRawResult capture(const SnapshotDescriptor&,BundleHandle&);',1)
 code='#include "http_prealignment_receipt.hpp"\n'+code
 code=code.replace('class ServingRawSession {public:','class ServingRawSession {public:\n'+declarations,1)
 bodies='\n'.join(take(raw,marker) for marker in (
  'core::SnapshotDescriptor ServingRawSession::endpoint_metadata(',
  'SuffixRawResult ServingRawSession::admit_host(',
  'const WindowKey* ServingRawSession::suffix_donor_key(',
  'bool ServingRawSession::prepare_owned_alignment_known(',
  'bool ServingRawSession::cancel_owned_alignment_known(',
  'bool ServingRawSession::complete_owned_alignment_known(',
  'bool ServingRawSession::inspect_cohort_receipt(',
  'SuffixRawResult ServingRawSession::capture_owned_slot(',
  'core::WindowKey ServingRawSession::edited_key('))
 install=take(raw,'bool ServingRawSession::install_owned_slot_image(')
 key_line=next(line.strip()for line in install.splitlines()if 'WindowKey destination_key=edited_prefix?'in line)
 adoption=next(line.strip()for line in install.splitlines()if 'slot_current_key_=std::move(destination_key);'in line)
 provenance=next(line.strip()for line in install.splitlines()if 'if(edited_prefix)slot_donor_provenance_='in line)
 bodies+='\n'+'SuffixRawResult ServingRawSession::capture(const SnapshotDescriptor& metadata,BundleHandle& out) {\n    transfer_capture_reached=true;\n    if(!recording_capture_success)return invalid("recorded refusal at native capture boundary");\n    if(!tx_->identity_matches(metadata.key))return invalid("recorded capture stable identity refusal");\n    ++recording_capture_calls;auto* actor=this;\n\n        core::SnapshotDescriptor descriptor=metadata;\n        descriptor.native_layout="recorded-metadata-anchor-not-native-image";\n        core::WindowStateLayout layout;layout.native_layout=descriptor.native_layout;\n        layout.model_profile=descriptor.key.model_profile;layout.topology=descriptor.key.topology;\n        layout.max_tokens=128;layout.max_payload_bytes=4096;layout.steering_enabled=false;\n        // Explicit abstract recording inventory: this cannot materialize through\n        // the native adapter and is NEVER passed to suffix quote/import.\n        layout.layers.assign(4,core::WindowStateLayout::LayerKind::qsa);\n        for(uint32_t stage=0;stage<2;++stage){\n            core::WindowStateLayout::Stage expected;expected.id=stage;expected.layer_begin=stage*2;expected.layer_end=stage*2+2;\n            core::DevicePart image;image.id=stage;image.layer_begin=expected.layer_begin;image.layer_end=expected.layer_end;image.cut=actor->cut();\n            for(int64_t layer=expected.layer_begin;layer<expected.layer_end;++layer)\n                for(auto kind:{core::StatePartKind::qsa_k,core::StatePartKind::qsa_v,core::StatePartKind::qsa_pooled,\n                        core::StatePartKind::qsa_tail,core::StatePartKind::qsa_spare,core::StatePartKind::qsa_block_position}){\n                    core::TensorLayout spec;spec.kind=kind;spec.layer=layer;spec.device_part=stage;\n                    spec.dtype="recording-u8";spec.native_layout="recording-control-only";spec.shape={1};spec.strides={1};spec.bytes=1;\n                    expected.tensors.push_back(spec);core::TensorPart part;part.layout=spec;part.payload.resize(1);\n                    part.payload=core::ConversationBuffer{uint8_t(actor->slot_tokens_.back())};image.tensors.push_back(std::move(part));\n                }\n            layout.stages.push_back(std::move(expected));descriptor.devices.push_back(std::move(image));\n        }\n        std::shared_ptr<const void> paid;require(bool(actor->admit_host(4096,paid)),"metadata anchor is actually charged");\n        core::BundleHandle result;\n        require(bool(core::BundleHandle::freeze(descriptor,descriptor.key,layout,result,paid)),"immutable metadata anchor from actual receipt cut");\n        out=std::move(result);return {SuffixRawStatus::ok,{},{}};\n\n}\n'+'\ncore::WindowKey ServingRawSession::recording_install_key(const BundleHandle& source_image,bool edited_prefix){\n'+key_line+'\n'+adoption+'\n'+provenance+'\nslot_source_provenance_=source_image;return slot_current_key_;\n}\n'
 marker='using namespace strata::program;\nnamespace core=strata::core;'
 assert marker in code
 code=code.replace(marker,'void require(bool,const char*);\nnamespace strata::program {\n'+bodies+'\n}\n'+marker,1)
 begin=code.index('    const auto bytes=std::ftell(emitted);require(!a->drain_owned_slot_group')
 end=code.index('\n}\nvoid actual_group_completion',begin)
 exercise=r'''
    // Recording source capture boundary: preserve metadata from the real first
    // group-produced cut, not an externally supplied completed cut. No native
    // layout/copy or suffix producer evidence is claimed by this fixture.
    auto source_anchor=[&](const std::shared_ptr<ServingRawSession>& actor){
        actor->recording_capture_success=true;core::BundleHandle result;
        require(bool(actor->capture_owned_slot(result)),"actual capture_owned_slot enters recording native capture boundary");return result;
    };
    auto original=source_anchor(a),foreign=source_anchor(b);a->slot_source_provenance_=original;
    require(a->suffix_donor_key(original)&&!a->suffix_donor_key(foreign),"foreign actual actor authority is refused");
    const auto host_before=a->tx_->host_cap_->live.load();auto cap=a->tx_->host_cap_;
    const auto watermark=a->cut().output_watermark;const auto first_source_cut=original.get()->cut;
    uint64_t last_alignment_op=0;
    auto request=std::make_shared<const std::vector<int64_t>>(std::vector<int64_t>{17,18,31,32,33,34});
    std::shared_ptr<const void> request_credit;require(bool(a->admit_host(4096,request_credit)),"actual aggregate request charge");
    ::unsetenv("STRATA_HTTP_TWO_SLOT_RECEIPTS");
    strata::program::report_prealignment_begin("aaaaaaaaaaaa",3,0,a->slot_occupancy_->incarnation(),64,*request,first_source_cut,a->cut(),source_cut,4);
    ::setenv("STRATA_HTTP_TWO_SLOT_RECEIPTS","1",1);
    strata::program::report_prealignment_begin("aaaaaaaaaaaa",3,0,a->slot_occupancy_->incarnation(),64,*request,first_source_cut,a->cut(),source_cut,4);
    auto bad_request=std::make_shared<const std::vector<int64_t>>(std::vector<int64_t>{17,99,31,32,33,34});
    require(!a->prepare_owned_alignment_known(bad_request,request_credit,error),"wrong authentic pending refuses before dispatch");
    const auto events_before_cancel=seam::events.size();
    require(a->prepare_owned_alignment_known(request,request_credit,error)&&a->cancel_owned_alignment_known(error),"pre-dispatch preparation can cancel");
    require(seam::events.size()==events_before_cancel&&a->cut()==first_source_cut,"cancellation submits no native work and changes no cut");
    for(int step=0;step<3;++step){
        require(a->prepare_owned_alignment_known(request,request_credit,error),"real next known input prepared");
        if(step==1){require(a->cancel_owned_alignment_known(error)&&a->prepare_owned_alignment_known(request,request_credit,error),"between-row cancellation is drained and resumable");}
        const int group_rows[]={0,1},group_keep[]={1,1};
        const int32_t group_tokens[]={a->slot_tokens_.back(),b->slot_tokens_.back()};
        const int64_t group_positions[]={int64_t(a->cut().forwarded),int64_t(b->cut().forwarded)};
        require(a->begin_work(error)&&b->begin_work(error),"both actual RAW owners begin only after preparation");
        auto quoted=budget.reserve({{0,0},core::NativeBatchGroupCompletion::metadata_bytes(),0,0,0,0},core::ReservationPurpose::foreground);
        std::shared_ptr<const core::NativeBatchGroupCompletion> group;
        const auto dispatches=std::count(seam::events.begin(),seam::events.end(),"dispatch");
        require(root.run_owned_slot_group(observations.data(),2,uint64_t(3+step),group_rows,2,group_tokens,group_positions,group_keep,nullptr,nullptr,output,std::move(quoted.lease),group,error),"actual two-stage shared group creates real receipt");
        require(std::count(seam::events.begin(),seam::events.end(),"dispatch")==dispatches+2,"one shared all-stage dispatch, no serial per-lane replacement");
        const auto before_output=std::ftell(emitted);
        require(!a->drain_owned_slot_group(*group,emitted,error),"standalone emitter cannot overwrite prepared known authority");
        ServingRawSession::recording_consumed_bit(*group,0,true);
        require(!a->complete_owned_alignment_known(*group,error),"already-consumed real receipt cannot seal pending");
        ServingRawSession::recording_consumed_bit(*group,0,false); // adversarial fixture injection only, never a producer API
        require(a->complete_owned_alignment_known(*group,error),"actual receipt seals the true next known pending");
        require(std::ftell(emitted)==before_output&&a->cut().output_watermark==watermark,"known row emits nothing and preserves inherited watermark");
        require(!a->complete_owned_alignment_known(*group,error),"duplicate actual receipt cannot complete twice");
        if(step==0){core::BundleHandle premature;const auto captures=a->recording_capture_calls;
            require(!a->capture_owned_slot(premature)&&a->recording_capture_calls==captures,"genuine in-progress unaligned endpoint cannot bind an origin before actual U");}

        strata::program::report_prealignment_known("aaaaaaaaaaaa",*group,0,a->cut(),a->slot_tokens_.back());
        last_alignment_op=group->operation();
        require(b->drain_owned_slot_group(*group,emitted,error),"same real group advances the peer response");
    }
    require(a->cut().forwarded==4&&a->cut().accepted==5&&a->slot_tokens_==std::vector<int32_t>({17,18,31,32,33}),"three actual rows seal genuine aligned F4/A5");
    require(original.get()->cut==first_source_cut&&original.get()->accepted_tokens==std::vector<int32_t>({17,18}),"original source remains immutable");
    require(!a->prepare_owned_alignment_known(request,request_credit,error),"adapter cannot advance beyond genuine ceil-aligned endpoint");
    // B reached F4 through real ordinary group receipts and remains identity-keyed.
    // Execute real installer key selection at the explicit transfer-recording boundary.
    core::BundleHandle aligned;require(bool(b->capture_owned_slot(aligned))&&!b->alignment_edit_origin_,"actual aligned donor capture needs no alignment origin");
    b->slot_source_provenance_=aligned;
    const auto aligned_target=b->recording_install_key(aligned,true);
    core::BundleHandle aligned_prefix;require(bool(b->capture_owned_slot(aligned_prefix)),"aligned edited-prefix capture remains permitted");
    require(!b->alignment_edit_origin_&&b->edited_key()==aligned_target&&b->suffix_donor_key(aligned),"aligned prefix capture cannot create a second edit target epoch");

    // Bad epoch/owner/source faults are fixture injections, not producer APIs.
    // They must never gain suffix donor authority from an ordinary capture.
    const auto good_key=a->slot_current_key_;auto good_source=a->slot_source_provenance_;auto good_credit=a->alignment_owner_credit_;
    for(int fault=0;fault<6;++fault){
        a->slot_current_key_=good_key;a->slot_source_provenance_=good_source;a->alignment_owner_credit_=good_credit;
        if(fault==0)++a->slot_current_key_.revision;
        if(fault==1)++a->slot_current_key_.selection_epoch;
        if(fault==2)++a->slot_current_key_.provenance_epoch;
        if(fault==3)++a->slot_current_key_.window_generation;
        if(fault==4)a->alignment_owner_credit_.reset();
        if(fault==5)a->slot_source_provenance_={};
        core::BundleHandle refused;const auto captured=a->capture_owned_slot(refused);
        require(!a->alignment_edit_origin_&&(!captured||!a->suffix_donor_key(refused)),"forged epochs or missing real source/credit cannot bind donor origin");
    }
    a->slot_current_key_=good_key;a->slot_source_provenance_=good_source;a->alignment_owner_credit_=good_credit;
    core::BundleHandle progressed;require(bool(a->capture_owned_slot(progressed))&&a->alignment_edit_origin_&&a->suffix_donor_key(progressed),"actual known progress capture binds actual origin exactly once");
    const auto progressed_origin=*a->alignment_edit_origin_;
    const auto progressed_target=a->recording_install_key(progressed,true);
    core::BundleHandle progressed_prefix;require(bool(a->capture_owned_slot(progressed_prefix)),"progressed path edited-prefix capture remains permitted");
    require(*a->alignment_edit_origin_==progressed_origin&&a->edited_key()==progressed_target&&a->suffix_donor_key(progressed),"progressed prefix capture preserves original donor and actual installer target");
    require(a->close_request(error)&&b->close_request(error)&&a->retire_owned_slot(error)&&b->retire_owned_slot(error),"both real owners close and retire after consumed group");
    original={};foreign={};progressed={};progressed_prefix={};aligned={};aligned_prefix={};
    b->slot_source_provenance_={};b->slot_donor_provenance_={};good_source={};good_credit.reset();request.reset();request_credit.reset();
    const auto actual_retired_cut=a->cut();const auto actual_retired_occupancy=a->slot_occupancy_->incarnation();
    a.reset();
    strata::program::report_prealignment_abort("aaaaaaaaaaaa",0,actual_retired_occupancy,last_alignment_op,"cancel","donor",actual_retired_cut);
    require(cap->live.load()<host_before,"retired actor destruction releases retained alignment/source aggregate credits");
    b.reset();std::fclose(emitted);
    require(slots.retire_after_consumers_joined({},error),"actual all-stage physical retirement follows actor retirement");
'''
 return code[:begin]+exercise+code[end:]

def source_inventory(base):
 # Bind the actual frozen inputs for THIS execution. There is deliberately no
 # pre-change header hash: legitimate typed read-only getters remain testable.
 paths=set()
 for rel in ['include','tests/core/owned_batch_slot_sessions']:
  paths.update(p for p in (base/rel).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 for rel in ['src/program/generate.cpp','src/program/suffix_raw_runtime.cpp','src/program/suffix_raw_runtime.hpp',
             'src/program/http_prealignment_receipt.hpp','src/program/http_two_slot_receipt.hpp','src/program/owned_batch_slot_sessions.hpp',
             'src/core/window_allocator.cpp','src/core/verify.cpp','src/core/window_budget.cpp','src/core/native_retirement.cpp',
             'src/core/window_hash.cpp','src/core/window_state.cpp']:
  paths.add(base/rel)
 paths.update(p for p in (base/'src/core/window_hash_vendor').rglob('*') if p.is_file())
 paths.update(p for p in (HERE/'owned_slot_prealignment').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def check_source():
 raw=(ROOT/'src/program/suffix_raw_runtime.cpp').read_text();gen=(ROOT/'src/program/generate.cpp').read_text()
 fixture=load(ROOT/'tests/core/owned_batch_slot_sessions/test.py');take=fixture.take
 prepare=take(raw,'bool ServingRawSession::prepare_owned_alignment_known(')
 complete=take(raw,'bool ServingRawSession::complete_owned_alignment_known(')
 admit=take(gen,'        auto admit_owned_repair_slot =')
 step=take(gen,'        auto batch_step =')
 assert 'request->at(size_t(cut().forwarded))!=slot_tokens_.back()' in prepare
 assert prepare.index('admit_host(')<prepare.index('std::make_unique<AlignmentKnown>')
 assert complete.index('inspect_cohort_receipt(')<complete.index('compare_exchange_strong')<complete.index('complete_driver_work(')
 assert 'next.cut.output_watermark!=cut().output_watermark' in complete and 'fprintf' not in complete
 assert admit.index('planned_accepted.reserve(')<admit.index('owned_alignment_step()')<admit.index('facts.accepted=std::span<const int32_t>')
 assert admit.index('capture_owned_slot(donor)')<admit.index('report_prealignment_seal(')<admit.index('install_owned_slot_image(prefix_source')
 assert step.count('run_owned_slot_group(')==1 and step.index('prepare_owned_alignment_known(')<step.index('begin_work(err)')
 assert step.index('complete_owned_alignment_known(')<step.index('report_prealignment_known(')
 assert 'session.quote_suffix(donor, *actual_donor_key' in gen and 'session.prepare_suffix(donor, *actual_donor_key' in gen
 authority=take(raw,'const WindowKey* ServingRawSession::suffix_donor_key(')
 assert 'donor.get()!=alignment_donor_.get()' in authority
 refusal=gen[gen.index('const bool canceled=owned_admission_scope.finish_cancelled();',gen.index('admit_owned_repair_slot(target')):]
 assert refusal.index('retire_owned_slot(err)')<refusal.index('slot.owned_raw.reset();')<refusal.index('report_prealignment_abort(')
 assert 'b%4||f%4' in (ROOT/'src/core/window_suffix.cpp').read_text()
 print('PASS canonical prealignment real-operand/group/receipt/retirement source ordering; aligned core gate retained')

def run(build):
 base=ROOT
 check_source()
 initial=source_inventory(base)
 build.mkdir(parents=True,exist_ok=True);source=generate(base)
 fixture=base/'tests/core/owned_batch_slot_sessions'
 flags=['c++','-std=c++20','-O0','-Wall','-Wextra','-Werror','-DSTRATA_OWNED_BATCH_AVAILABLE=1','-I'+str(fixture/'stubs'),'-I'+str(base/'include'),'-I'+str(base/'src/program')]
 objects=[]
 for name in ['window_budget','native_retirement','window_hash','window_state']:
  obj=build/(name+'.o');subprocess.run(flags+['-c',str(base/'src/core'/f'{name}.cpp'),'-o',str(obj)],check=True);objects.append(str(obj))
 old_capture=(HERE/'owned_slot_prealignment/capture_owned_slot_before_origin_guard.inc').read_text()
 current_capture=load(base/'tests/core/owned_batch_slot_sessions/test.py').take((base/'src/program/suffix_raw_runtime.cpp').read_text(),'SuffixRawResult ServingRawSession::capture_owned_slot(')
 variants=[('production',source),
  ('old_broad_capture_origin',source.replace(current_capture.strip(),old_capture.strip())),
  ('consumed_receipt_ignored',source.replace('receipt.consumed_[slot].load(std::memory_order_acquire)','false').replace('if(!receipt.consumed_[view.slot].compare_exchange_strong(unused,true,std::memory_order_acq_rel))return false;','receipt.consumed_[view.slot].store(true);(void)unused;')),
  ('wrong_pending',source.replace('request->at(size_t(cut().forwarded))!=slot_tokens_.back()||','')),
  ('watermark_mutation',source.replace('StateCut next=cut();++next.id;','StateCut next=cut();++next.output_watermark;++next.id;').replace('next.cut.output_watermark!=cut().output_watermark||','')),
  ('cancel_ticket_retained',source.replace('    alignment_known_.reset();return true;','    return true;')),
  ('foreign_actor_authority',source.replace('if(donor.get()->key==tx_->identity_)return &tx_->identity_;','if(donor)return &donor.get()->key;')),
  ('retained_actor_credit',source.replace('    a.reset();','    /* negative: retired actor remains retained */'))]
 for name,code in variants:
  if name!='production' and code==source:raise RuntimeError('ineffective negative '+name)
  cpp=build/(name+'.cpp');binary=build/name;cpp.write_text(code)
  subprocess.run(flags+[str(cpp),*objects,'-o',str(binary)],check=True)
  result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
  if (result.returncode==0)!=(name=='production'):raise RuntimeError(name+': '+result.stderr)
  if name=='production':
   events=[(line.split(' ',1)[0],json.loads(line.split(' ',1)[1])) for line in result.stdout.splitlines() if line.startswith('BPREALIGN_')]
   if [kind for kind,_ in events]!=['BPREALIGN_BEGIN']+['BPREALIGN_KNOWN']*3+['BPREALIGN_ABORT']:raise RuntimeError('actual observer event count/order/disabled gate differs')
   progress=[row for kind,row in events if kind=='BPREALIGN_KNOWN']
   if [x['input'] for x in progress]!=[18,31,32] or [x['next_known'] for x in progress]!=[31,32,33] or [x['actual']['forwarded'] for x in progress]!=[2,3,4] or any(x['actual']['watermark']!=1 for x in progress):raise RuntimeError('actual typed receipt observer projection differs')
   if events[-1][1]['retired']!=progress[-1]['actual'] or events[-1][1]['phase']!='donor':raise RuntimeError('actual retired actor observer differs')
  print('PASS '+name+(': actual group/RAW pending seal+typed observer recording boundary' if name=='production' else ': mutant is rejected'))
 if source_inventory(base)!=initial:raise RuntimeError('actual frozen source changed during production fixture')
 recorded={str(Path(path).relative_to(base)):digest for path,digest in initial.items()}
 (build/'source-pins.json').write_text(json.dumps(recorded,indent=2)+'\n')
 print('PASS actual frozen source inventory unchanged: '+str(len(recorded))+' files; pins in '+str(build/'source-pins.json'))
 for name in ['test_prealignment.py','test_lineage.py']:
  subprocess.run([sys.executable,'-B',str(HERE/'owned_slot_prealignment'/name)],check=True,timeout=20)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path);parser.add_argument('--source-only',action='store_true');args=parser.parse_args();check_source()
 if not args.source_only:
  if args.build_dir:run(args.build_dir.resolve())
  else:
   with tempfile.TemporaryDirectory(prefix='strata-prealignment-') as directory:run(Path(directory))
