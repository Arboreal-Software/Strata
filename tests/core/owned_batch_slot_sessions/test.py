#!/usr/bin/env python3
"""Production wrapper/actual native-retirement/fresh-session factory, real ledger.

Physical allocation/session/CUDA calls record CPU control and lifetime only.
No device allocation fit, native byte ownership or GPU drain is inferred.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def take(text,marker):
    begin=text.index(marker);opening=text.index('{',begin);depth=0
    for at in range(opening,len(text)):
        if text[at]=='{':depth+=1
        elif text[at]=='}':
            depth-=1
            if not depth:return text[begin:at+1]+'\n'
    raise RuntimeError('unclosed source '+marker)


def source_check():
    source=(ROOT/'src/program/generate.cpp').read_text()
    if '(o.batch>0&&!owned_batch_requested)' not in source or 'native_owned_batch_serving::eligible(native_pack,o,true)' not in source:
        raise RuntimeError('owned-session batch execution lacks explicit resolved eligibility guard')
    handoff=source[source.index('const bool admitted=admit_owned_repair_slot'):source.index('goto owned_batch_request_done;')]
    assert handoff.index('reset_main_after_slot_handoff')<handoff.index('finish_cancelled()')<handoff.index('if(!admitted||canceled)')<handoff.index('report_http2_accept')
    assert 'stop_req.load()' not in handoff
    assert source.count('accept_batch_stop(*it);')==2 and source.count('accept_batch_stop(line);')==1
    stop=take(source,'        auto accept_batch_stop =')
    assert stop.index('slot_occupancy_incarnation()!=')<stop.index('retire_slot_controller')<stop.index('retire_owned_slot')<stop.index('report_http2_stop')
    step=take(source,'        auto batch_step =')
    assert step.index('std::vector<int> keep')<step.index('begin_work(err)')
    maintenance=take(source,'        owned_slot_maintenance =')
    assert maintenance.index('queued.input_error||queued.end_of_input')<maintenance.index('serving_repair_idle_unit==0')
    assert 'boundary.consecutive_escalated_units=serving_repair_consecutive_escalated' in maintenance
    baseline=json.loads((HERE/'baseline_fragments.json').read_text())['fragments']
    if baseline['legacy_slot_allocation'] not in source:
        raise RuntimeError('legacy slot allocation body changed')
    constructor=source.index('owned_runtime->batch_slot_sessions.initialize(')
    sizing=source.index('    // Secure MTP',constructor)
    if constructor>sizing:raise RuntimeError('owned slots allocated after cache/model consumers')
    if source.index('OwnedBatchSlotSessions batch_slot_sessions;')>source.index('std::unique_ptr<strata::core::Verifier> primary_verifier;'):
        raise RuntimeError('owned allocation does not outlive borrowing verifier members')
    if 'batch_slot_handles.at(slot)' not in source:
        raise RuntimeError('borrowed views silently reacquire recycled allocation generation')
    verifier=(ROOT/'src/core/verify.cpp').read_text()
    header=(ROOT/'include/strata/core/verify.hpp').read_text()
    if take(header,'    bool snapshot_binding_matches(')!=baseline['solo_binding']:
        raise RuntimeError('solo RAW verifier binding guard changed')
    init=take(verifier,'bool Verifier::init_slots(')
    if not init.index('++slot_inventory_generation_;slot_inventory_ready_=false;')<init.index('if (!mapped(')<init.index('slot_inventory_ready_=true;'):
        raise RuntimeError('actual slot inventory epoch does not revoke before setup/publish only after setup')
    raw=(ROOT/'src/program/suffix_raw_runtime.cpp').read_text()
    for signature in ['SuffixRawResult SuffixRawTransaction::capture(', 'SuffixRawResult SuffixRawTransaction::restore(']:
        if take(raw,signature)!=baseline[signature]:raise RuntimeError('actual native RAW copy pipeline changed unexpectedly '+signature)
    print('PASS isolated production branch, legacy allocator/batch guard/solo RAW guard unchanged; actual verifier epoch publication order')


def run_stop_parser(directory):
    source=(ROOT/'src/program/generate.cpp').read_text()
    parser=take(source,'        auto accept_batch_stop =')
    fragment=parser[parser.index('                unsigned slot='):parser.index('                auto& lane=')]
    cpp=directory/'actual-stop-parser.cpp';binary=directory/'actual-stop-parser'
    cpp.write_text('#include <array>\n#include <charconv>\n#include <string>\n#include <string_view>\n#include <cstdint>\n#include <cassert>\n'
        'bool parse(const std::string& command){std::array<int,2> bs{};bool accepted=false;auto actual=[&](){\n'+fragment+
        'accepted=true;};actual();return accepted;}\nint main(){assert(parse("BSTOP 0 occupancy=1"));assert(parse("BSTOP 1 occupancy=18446744073709551615"));'
        'for(const auto* invalid:{"BSTOP -1 occupancy=1","BSTOP 2 occupancy=1","BSTOP 0","BSTOP 0 occupancy=0","BSTOP 0 occupancy=-1","BSTOP 0 occupancy=+1","BSTOP 0 occupancy=18446744073709551616","BSTOP 0 occupancy=1 trailing","BSTOP 0 occupancy="})assert(!parse(invalid));}\n')
    subprocess.run(['c++','-std=c++20','-Wall','-Wextra','-Werror',str(cpp),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=10)
    print('PASS actual strict native BSTOP parser: complete decimal nonce/bounds/overflow/sign/trailing refusals')


def run(directory):
    run_stop_parser(directory)
    directory=Path(directory)
    allocator=(ROOT/'src/core/window_allocator.cpp').read_text()
    verifier=(ROOT/'src/core/verify.cpp').read_text()
    verifier_header=(ROOT/'include/strata/core/verify.hpp').read_text()
    init=take(verifier,'bool Verifier::init_slots(')
    actual_run=take(verifier,'bool Verifier::run_slot_rows(')
    copy_begin=actual_run.index('        const auto copy_status=cudaStreamSynchronize(copy_);')
    copy_end=actual_run.index('        if (prof_on_)',copy_begin)
    capture=take(verifier,'bool Verifier::capture_batch(')
    upload_begin=capture.index('    const auto upload_status=')
    upload_end=capture.index('    std::string list;',upload_begin)
    parts={
        'ACTUAL_CONVENIENCE_ALLOCATOR':take(allocator,'NativeAllocationResult allocate_native_sessions(const ModelGeometry& g,int64_t cells,int64_t k,\n    const std::vector<NativeSessionRequest>& requests,WindowBudget& budget,NativeSessionAllocation& out)'),
        'ACTUAL_PREPARE_RETIREMENT':take(allocator,'bool NativeRetirementAttempt::prepare('),
        'ACTUAL_DRAIN_RETIREMENT':take(allocator,'bool NativeRetirementAttempt::drain_and_commit('),
        'ACTUAL_GRAPH_UPLOAD_CHECK':capture[upload_begin:upload_end],
        'ACTUAL_COMMON_REFUSAL':take(verifier_header,'    bool refuse_copy_query_failure('),
        'ACTUAL_BATCH_COPY_CHECK':actual_run[copy_begin:copy_end],
        'ACTUAL_OWNED_SETUP':take(verifier,'bool Verifier::prepare_owned_batch_slots('),
        'ACTUAL_OWNED_GROUP':take(verifier,'bool Verifier::run_owned_slot_group('),
        'ACTUAL_SLOT_MATCH':take(verifier,'bool Verifier::owned_batch_slot_matches('),
        'ACTUAL_SLOT_INSPECT':take(verifier,'bool Verifier::inspect_owned_batch_slot('),
        'ACTUAL_SOLO_BINDING':take(verifier_header,'    bool snapshot_binding_matches('),
        'ACTUAL_INVENTORY_BEGIN':init[init.index('    if(slot_inventory_generation_==UINT64_MAX)'):init.index('    const strata::kernels::QsaShapes')],
        'ACTUAL_INVENTORY_READY':'slot_inventory_ready_=true;',

    }
    raw=(ROOT/'src/program/suffix_raw_runtime.cpp').read_text()
    slot_parts={
        'ACTUAL_HOST_CAP':take(raw,'struct SuffixRawTransaction::HostCap '),
        'ACTUAL_HOST_CREDIT':take(raw,'struct SuffixRawTransaction::HostCredit '),
        'ACTUAL_TX_CONSTRUCTOR':'SuffixRawTransaction::SuffixRawTransaction'+raw.split('SuffixRawTransaction::SuffixRawTransaction',1)[1].split('SuffixRawTransaction::~SuffixRawTransaction',1)[0],
    }
    for marker,name in [('TX_INITIALIZE','SuffixRawResult SuffixRawTransaction::initialize'),('TX_SHARED_INITIALIZE','SuffixRawResult SuffixRawTransaction::initialize_slot_shared_host'),('TX_RESIDENT_SHARED_INITIALIZE','SuffixRawResult SuffixRawTransaction::initialize_resident_shared_host'),('SLOT_TRANSFER','bool ServingRawSession::transfer_owned_slot_from'),('TX_ADMIT_HOST','SuffixRawResult SuffixRawTransaction::admit_host'),('TX_ADMIT_CONTROLLER','SuffixRawResult SuffixRawTransaction::admit_controller_host'),('TX_IDENTITY_MATCH','bool SuffixRawTransaction::identity_matches'),('SERVING_FACTORY','std::shared_ptr<ServingRawSession> ServingRawSession::open'),('SERVING_TERMINAL','bool ServingRawSession::terminal'),('SERVING_HEALTHY','bool ServingRawSession::healthy'),('TX_DESTRUCTOR','SuffixRawTransaction::~SuffixRawTransaction'),('TX_RESOLVE','bool SuffixRawTransaction::resolve'),('TX_POISON','SuffixRawResult SuffixRawTransaction::poison'),('TX_BEGIN','SuffixRawResult SuffixRawTransaction::begin_driver_work'),('TX_POLL','SuffixRawResult SuffixRawTransaction::poll_all'),('TX_COMPLETE','SuffixRawResult SuffixRawTransaction::complete_driver_work'),('SERVING_IDENTITY','core::WindowKey ServingRawSession::serving_identity'),('SLOT_FACTORY','std::shared_ptr<ServingRawSession> ServingRawSession::open_owned_slot'),('SLOT_QUEUE','bool ServingRawSession::queue_owned_slot_input'),('SLOT_CONSUME','bool ServingRawSession::drain_owned_slot_group'),('SLOT_RETIRE','bool ServingRawSession::retire_owned_slot'),('SERVING_PHASE','ServingRawSession::Phase ServingRawSession::phase'),('SERVING_OPEN','bool ServingRawSession::open_request'),('SERVING_BEGIN','bool ServingRawSession::begin_work'),('SERVING_CLOSE','bool ServingRawSession::close_request')]:
        slot_parts['ACTUAL_'+marker]=take(raw,name+'(')
    raw_header=(ROOT/'src/program/suffix_raw_runtime.hpp').read_text()
    slot_parts['ACTUAL_ALIGNMENT_KNOWN']=take(raw_header,'    struct AlignmentKnown ')
    slot_fixture=(HERE/'slot_raw_fixture.inc').read_text()
    for marker,part in slot_parts.items():slot_fixture=slot_fixture.replace(marker,part)
    code=(HERE/'test.cpp').read_text().replace('SLOT_RAW_FIXTURE',slot_fixture)
    for name,part in parts.items():code=code.replace(name,part)
    header=(ROOT/'src/program/owned_batch_slot_sessions.hpp').read_text().replace('#pragma once\n','')
    variants=[('production',code),
              ('old_factory_geometry_copy_address',code.replace('aggregate_owner.tx_->geometry_identity_!=&geometry','&aggregate_owner.tx_->geometry_!=&geometry')),
              ('old_shared_geometry_copy_address',code.replace('geometry_identity_!=live.geometry_identity_','&geometry_!=&live.geometry_')),
              ('old_transfer_geometry_copy_address',code.replace('source.tx_->geometry_identity_!=tx_->geometry_identity_','&source.tx_->geometry_!=&tx_->geometry_')),
              ('early_publication',code.replace('#include "owned_batch_slot_sessions.hpp"',header.replace(
                  'if(!result){failed_=true;error=result.detail;return false;}',
                  'if(!result){published_=true;failed_=true;error=result.detail;return false;}'))),
              ('stale_generation',code.replace('#include "owned_batch_slot_sessions.hpp"',header.replace(
                  'handle.generation!=generations_[handle.slot] || ',''))),
              ('omitted_gang_drain',code.replace('if(!context_.drain_after_producers_closed()){','if(false){')),
              ('wrong_slot_table',code.replace('slots_[slot]!=native->session ||','')),
              ('stale_inventory',code.replace('observed.inventory_generation_!=slot_inventory_generation_ ||','')),
              ('missing_group_submission_latch',code.replace('for(const auto& weak:owned_slot_occupancies_)if(auto occupancy=weak.lock();occupancy&&\n        occupancy->submitted_group_.load(std::memory_order_acquire))return false;','').replace('if(!occupancy->submitted_group_.compare_exchange_strong(unused,operation,std::memory_order_acq_rel))return false;','occupancy->submitted_group_.store(operation,std::memory_order_release);(void)unused;')),
              ('missing_raw_slot_verifier_match',code.replace('!verifiers_[i]->owned_batch_slot_matches(slot_bindings_[i])','false')),
              ('missing_slot_occupancy',code.replace('receipt.occupancies_[slot]!=slot_occupancy_||','').replace('slot_occupancy_->submitted_group_.load(std::memory_order_acquire)!=receipt.operation_||','')),
              ('ignored_graph_upload_error',code.replace('owned_batch_setup_&&(upload_status!=cudaSuccess||upload_drain!=cudaSuccess)','false').replace('const auto upload_status=','[[maybe_unused]] const auto upload_status=').replace('const auto upload_drain=','[[maybe_unused]] const auto upload_drain=')),
              ('ignored_first_copy_error',code.replace('owned_batch_setup_&&copy_status!=cudaSuccess','false').replace('const auto copy_status=cudaStreamSynchronize(copy_);','[[maybe_unused]] const auto copy_status=cudaStreamSynchronize(copy_);'))]

    for label,actual in variants:
        if label!='production' and actual==code:raise RuntimeError('negative control did not change source '+label)
        cpp=directory/(label+'.cpp');binary=directory/label;cpp.write_text(actual)
        subprocess.run(['c++','-std=c++20','-O0','-Wall','-Wextra','-Werror','-DSTRATA_OWNED_BATCH_AVAILABLE=1',
            '-I'+str(HERE/'stubs'),'-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/program'),str(cpp),
            str(ROOT/'src/core/window_budget.cpp'),str(ROOT/'src/core/native_retirement.cpp'),str(ROOT/'src/core/window_hash.cpp'),'-o',str(binary)],check=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
        if (result.returncode==0)!=(label=='production'):
            raise RuntimeError(f'{label}: unexpected result {result.returncode}: {result.stderr}')
        print('PASS '+label+(': negative control fails' if label!='production' else ': '+result.stdout.strip()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path);parser.add_argument('--source-only',action='store_true')
    args=parser.parse_args();source_check()
    if not args.source_only:
        if args.build_dir:
            args.build_dir.mkdir(parents=True,exist_ok=True);run(args.build_dir)
        else:
            with tempfile.TemporaryDirectory(prefix='strata-owned-slots-') as directory:run(directory)
