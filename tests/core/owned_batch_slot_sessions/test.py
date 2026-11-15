#!/usr/bin/env python3
"""Production wrapper/actual native-retirement/fresh-session factory, real ledger.

Physical allocation/session/CUDA calls record CPU control and lifetime only.
No device allocation fit, native byte ownership or GPU drain is inferred.
"""
import argparse
from contextlib import contextmanager
import hashlib
import os
import signal
import sys
import time
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


@contextmanager
def compiled_variants(directory,variants,code,flags,common_objects):
    """Two fresh compile groups at most; yield binaries in original order."""
    owned={};pending={};next_variant=0
    launching=False;interrupted_launch=None
    signals={signal.SIGINT,signal.SIGTERM,signal.SIGHUP}
    previous={}

    def interrupted(signum,frame):
        nonlocal interrupted_launch
        if launching:
            interrupted_launch=signum
            return
        raise SystemExit(128+signum)

    def kill_group(child,signum):
        try:os.killpg(child.pid,signum)
        except ProcessLookupError:pass

    def observe():
        for entry in pending.values():
            if entry['done']:continue
            child=entry['child']
            # Observe without reaping: the owned PID/PGID cannot be recycled
            # before group retirement, including compile failure/interrupt.
            status=os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            if status is None:continue
            result=status.si_status if status.si_code==os.CLD_EXITED else -status.si_status
            binary=entry['binary']
            receipt={'event':'variant_compile_exit','label':entry['label'],
                'pid':child.pid,'pgid':child.pid,'argv':entry['argv'],'exit':result}
            if result==0 and binary.is_file():
                receipt.update(binary_bytes=binary.stat().st_size,
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            print(json.dumps(receipt),flush=True)
            if result:raise subprocess.CalledProcessError(result,entry['argv'])
            if not binary.is_file() or not binary.stat().st_size:
                raise RuntimeError('fresh variant binary missing/empty: '+str(binary))
            # A successful c++ driver has finished its workers. Retire any
            # residual group before reaping its identity; never leave workers.
            # No spawn occurs here: block handled interrupts until the
            # reaped identity is removed, then deliver any pending signal.
            mask=signal.pthread_sigmask(signal.SIG_BLOCK,signals)
            try:
                kill_group(child,signal.SIGKILL)
                child.wait(timeout=1)
                del owned[child.pid]
                entry['done']=True
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)

    def fill():
        nonlocal next_variant,launching
        while len(pending)<2 and next_variant<len(variants):
            observe()  # A completed compile failure stops further scheduling.
            index=next_variant;label,actual=variants[index]
            if label!='production' and actual==code:raise RuntimeError('negative control did not change source '+label)
            cpp=directory/(label+'.cpp');binary=directory/label
            binary.unlink(missing_ok=True);cpp.write_text(actual)
            argv=['c++',*flags,str(cpp),*[str(obj) for obj in common_objects],'-o',str(binary)]
            # Defer handled interrupts only across spawn+registration. Unlike
            # blocking before fork, this leaves child signal masks unchanged.
            launching=True
            try:
                child=subprocess.Popen(argv,start_new_session=True)
                owned[child.pid]=child
                pending[index]={'label':label,'binary':binary,'argv':argv,'child':child,'done':False}
            finally:launching=False
            if interrupted_launch is not None:raise SystemExit(128+interrupted_launch)
            print(json.dumps({'event':'variant_compile_start','label':label,
                'pid':child.pid,'pgid':child.pid,'argv':argv,
                'source_sha256':hashlib.sha256(cpp.read_bytes()).hexdigest()}),flush=True)
            next_variant+=1

    def ordered():
        fill()
        for index in range(len(variants)):
            while not pending[index]['done']:
                observe()
                if not pending[index]['done']:time.sleep(0.01)
            entry=pending.pop(index)
            yield entry['label'],entry['binary']
            observe()
            fill()

    try:
        for signum in signals:
            previous[signum]=signal.signal(signum,interrupted)
        yield ordered()
    finally:
        # Keep interrupts blocked until all owned groups are killed and every
        # driver is waited. No grace period can extend the outer test limit.
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,signals)
        failure=sys.exc_info()[1];cleanup_errors=[];cleanup_receipts=[]
        try:
            for signum in (signal.SIGTERM,signal.SIGKILL):
                for child in owned.values():
                    try:kill_group(child,signum)
                    except OSError as error:cleanup_errors.append(str(error))
            for child in owned.values():
                try:
                    result=child.wait(timeout=1)
                    cleanup_receipts.append({'event':'variant_compile_cleanup','pid':child.pid,
                        'pgid':child.pid,'argv':child.args,'exit':result})
                except (OSError,subprocess.TimeoutExpired) as error:cleanup_errors.append(str(error))
            for receipt in cleanup_receipts:
                try:print(json.dumps(receipt),flush=True)
                except OSError as error:cleanup_errors.append(str(error))
        finally:
            for signum,handler in previous.items():signal.signal(signum,handler)
            signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        if cleanup_errors:
            detail='owned compiler cleanup failed: '+'; '.join(cleanup_errors)
            if failure is None:raise RuntimeError(detail)
            failure.add_note(detail)
            print(detail,file=sys.stderr,flush=True)


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
        'ACTUAL_BANK_KEY':take(verifier_header,'    std::vector<int> bkey('),
        'ACTUAL_BATCH_KEY':take(verifier_header,'    static std::vector<int> batch_key('),
        'ACTUAL_BANK_RETIREMENT':take((ROOT/'src/program/generate.cpp').read_text(),'    struct OwnedPositionBankRetirement '),
        'ACTUAL_BANK_REPLAY':take(verifier_header,'    struct OwnedPositionReplay '),
        'ACTUAL_POSITION_BEGIN':take(verifier,'bool Verifier::owned_position_begin_batch('),
        'ACTUAL_POSITION_COMMIT':take(verifier,'bool Verifier::owned_position_commit_matches('),
        'ACTUAL_POSITION_MATCH':take(verifier,'bool Verifier::owned_position_binding_matches('),
        'ACTUAL_POSITION_TABLE':take(verifier,'const int32_t* Verifier::owned_position_table('),
        'ACTUAL_PREFILL_BANK_TARGET':take((ROOT/'src/prefill/prefill.cpp').read_text(),'bool Prefill::owned_position_bank_target('),
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
    # Bridge transitively includes the exact real window_native.hpp. Retire
    # only the old local declaration shadow, never an actual authority/helper.
    native_stage_shadow=take(slot_fixture,'struct NativeWindowStage ').rstrip()+';'
    if slot_fixture.count(native_stage_shadow)!=1:
        raise RuntimeError('expected one legacy native-stage declaration shadow')
    slot_fixture=slot_fixture.replace(native_stage_shadow,'',1)

    # Retain the actual tag/accessors and complete called reset/retire closure.
    # Old groups remain text/bank/recorded-native tests, NOT image execution or
    # numerical oracle qualification; no image branch is silently stubbed out.
    slot_fixture=slot_fixture.replace('bool steering_enabled_=false,initialized_=false;',
        'bool steering_enabled_=false;uint8_t initialized_=0;\n'+take(raw_header,'    bool image_exact()const noexcept'))
    slot_fixture=slot_fixture.replace('std::shared_ptr<void> pending_;','std::shared_ptr<const void> pending_;')
    source_methods='\n'.join(take(raw,name+'(') for name in [
        'std::shared_ptr<OwnedImageSessionRecords> ServingRawSession::image_directory',
        'std::shared_ptr<OwnedImageRequestOperation> ServingRawSession::image_operation',
        'void ServingRawSession::set_image_operation',
        'bool ServingRawSession::image_install_bank',
        'bool ServingRawSession::image_reset',
        'bool ServingRawSession::image_retire_current'])
    declarations=raw_header[raw_header.index('    std::shared_ptr<OwnedImageSessionRecords> image_directory()const;'):
        raw_header.index('    bool image_prepare_transport(')]
    slot_fixture=slot_fixture.replace('    bool slot_controller_bound_=false;',
        '    bool slot_controller_bound_=false;\n    std::shared_ptr<const void> alignment_owner_credit_;\n'+declarations+
        '\n'+take(raw_header,'    const core::WindowKey& key() const')+
        take(raw_header,'    const core::StateCut& cut() const')+
        take(raw_header,'    uint64_t max_tokens() const')+
        take(raw_header,'    uint64_t slot_occupancy_incarnation()const'))
    slot_fixture=slot_fixture.replace('ACTUAL_SERVING_CLOSE','ACTUAL_SERVING_CLOSE\n'+source_methods,1)

    # Add only named new declarations/actual methods; old extracted RAW bodies,
    # the 13 original mutants, flags, order and fresh common-TU build stay intact.
    slot_fixture=slot_fixture.replace('struct HostCap;struct HostCredit;',
        'struct HostCap;struct HostCredit;struct PositionAdmission;')
    slot_fixture=slot_fixture.replace('    SuffixRawResult admit_controller_host(',
        '    SuffixRawResult admit_position_banks(std::span<const uint64_t>,uint64_t,PositionAdmission&);\n    SuffixRawResult admit_controller_host(',1)
    slot_fixture=slot_fixture.replace('ACTUAL_HOST_CREDIT;',
        'ACTUAL_HOST_CREDIT;\n'+take(raw,'struct SuffixRawTransaction::PositionAdmission ')+';\n'+
        take(raw,'SuffixRawResult SuffixRawTransaction::admit_position_banks('),1)
    slot_fixture=slot_fixture.replace('    static std::shared_ptr<ServingRawSession> open_owned_slot(',
        '    static bool open_owned_startup(const ModelGeometry&,NativeSessionAllocation,std::shared_ptr<const void>,std::vector<Verifier*>,WindowBudget&,uint64_t,uint64_t,std::shared_ptr<ServingRawSession>&,std::string&);\n'
        '    bool prepare_owned_position_banks(std::shared_ptr<const NativeBatchSlotBindings>,std::span<prefill::Prefill* const>,uint64_t,OwnedPositionBankSetup&,std::string&);\n'
        '    '+take(raw_header,'bool private_window_open() const noexcept')+'\n'
        '    static std::shared_ptr<ServingRawSession> open_owned_slot(',1)
    slot_fixture=slot_fixture.replace('ACTUAL_SERVING_FACTORY',
        'ACTUAL_SERVING_FACTORY\n'+take(raw,'bool ServingRawSession::open_owned_startup(')+
        take(raw,'bool ServingRawSession::prepare_owned_position_banks('),1)

    # Exact new private host/one-shot/observation method closure. Access is
    # exposed only in this generated CPU TU; no production root API is added.
    image_names=[
        'std::shared_ptr<OwnedImageSessionRecords> ServingRawSession::image_directory',
        'VisionRequestOwner::Keepers ServingRawSession::image_reactor_keepers',
        'std::shared_ptr<OwnedImageRequestOperation> ServingRawSession::image_operation',
        'bool ServingRawSession::image_bind_controller',
        'bool ServingRawSession::image_select_preparation',
        'void ServingRawSession::set_image_operation',
        'bool ServingRawSession::image_authority_current',
        'bool ServingRawSession::image_prepare_transport',
        'bool ServingRawSession::image_capture_transport',
        'bool ServingRawSession::image_close_transport',
        'bool ServingRawSession::image_install_bank',
        'bool ServingRawSession::image_peer_visit',
        'bool ServingRawSession::image_observe_peer',
        'bool ServingRawSession::image_compare_peer',
        'bool ServingRawSession::image_reset',
        'bool ServingRawSession::image_cancel_current',
        'bool ServingRawSession::image_retire_current',
        'bool ServingRawSession::image_bind_prefill',
        'bool ServingRawSession::image_transfer_to',
        'SuffixRawResult ServingRawSession::prepare_vision_request',
        'core::SnapshotDescriptor ServingRawSession::endpoint_metadata']
    full_image_declarations=raw_header[raw_header.index('    std::shared_ptr<OwnedImageSessionRecords> image_directory()const;'):
        raw_header.index('    bool image_transfer_to(')]
    full_image_declarations+=raw_header[raw_header.index('    bool image_transfer_to('):raw_header.index('\n',raw_header.index('    bool image_transfer_to('))]
    slot_fixture=slot_fixture.replace(declarations,full_image_declarations+
        '\n    SuffixRawResult prepare_vision_request(int,std::span<const int64_t>,uint64_t,std::shared_ptr<const VisionRequestOwner>&);\n    static core::SnapshotDescriptor endpoint_metadata(const core::WindowKey&,const core::StateCut&,const std::vector<int32_t>&);')
    slot_fixture=slot_fixture.replace(source_methods,'\n'.join(take(raw,name+'(') for name in image_names))
    slot_fixture=slot_fixture.replace('struct HostCap;struct HostCredit;struct PositionAdmission;',
        'struct HostCap;struct HostCredit;struct PositionAdmission;struct VisionAdmission;')
    slot_fixture=slot_fixture.replace('    SuffixRawResult admit_controller_host(',
        '    SuffixRawResult admit_vision_host(const ServingRawSession&,uint64_t,VisionAdmission&);\n    SuffixRawResult admit_controller_host(',1)
    slot_fixture=slot_fixture.replace('ACTUAL_HOST_CREDIT;', 'ACTUAL_HOST_CREDIT;\n'+
        take(raw,'struct SuffixRawTransaction::VisionAdmission ')+';\n',1)
    slot_fixture+='\nnamespace strata::program {\n'+take(raw,'SuffixRawResult SuffixRawTransaction::admit_vision_host(')+'\n}\n'
    # Keep only the legacy native capture boundary recording; it is not image
    # acceptance. Image generic refusal executes the actual identity guard.
    slot_fixture=slot_fixture.replace('    bool transfer_capture_reached=false;',
        '    bool transfer_capture_reached=false;\n    '+take(raw_header,'    const std::shared_ptr<SuffixRawTransaction>& transaction()'))

    slot_fixture=slot_fixture.replace('inline bool present(',take(raw,'bool bounded_metadata(')+'\n'+take(raw,'bool capture_metadata(')+'\ninline bool present(',1)
    slot_fixture=slot_fixture.replace('    SuffixRawResult initialize();',
        '    SuffixRawResult recorded_capture_guard(const SnapshotDescriptor&,BundleHandle&);\n    bool recorded_capture_native_reached=false;\n    SuffixRawResult initialize();',1)
    full_capture=take(raw,'SuffixRawResult SuffixRawTransaction::capture(')
    capture_prefix=full_capture[:full_capture.index('    try {')].replace('SuffixRawTransaction::capture(', 'SuffixRawTransaction::recorded_capture_guard(',1)
    slot_fixture+='\nnamespace strata::program {\n'+capture_prefix+'    (void)out;recorded_capture_native_reached=true;return invalid("recording native snapshot boundary");\n}\n}\n'
    for marker,part in slot_parts.items():slot_fixture=slot_fixture.replace(marker,part)
    code=(HERE/'test.cpp').read_text().replace('SLOT_RAW_FIXTURE',slot_fixture)
    for name,part in parts.items():code=code.replace(name,part)
    operation=(ROOT/'src/program/owned_image_request_operation.hpp').read_text()
    code=code.replace('ACTUAL_IMAGE_OPERATION_DECL',take(operation,'class OwnedImageRequestOperation final '))
    code=code.replace('ACTUAL_IMAGE_DIRECTORY_DECL',take(operation,'class OwnedImageSessionRecords final '))
    code=code.replace('ACTUAL_IMAGE_WITNESS_DECL',take(operation,'class OwnedImagePeerObservation final '))
    # Use the real header exactly once (pragma once), including both native types.
    code=code.replace('ACTUAL_IMAGE_PLE_DECL;', '#include \"strata/core/window_native.hpp\"')
    code=code.replace('ACTUAL_PREFILL_IMAGE_RETIREMENT',
        take((ROOT/'src/prefill/prefill.cpp').read_text(),'bool Prefill::owned_image_retirement_ready('))

    # Pure physical extents use real production functions, not a fixture
    # accounting/geometry formula. Recording allocation below only owns bytes.
    states=(ROOT/'src/core/conversation_state.cpp').read_text()
    snapshots=(ROOT/'src/core/conversation_snapshot.cpp').read_text()
    state_header=(ROOT/'include/strata/core/conversation_snapshot.hpp').read_text()
    checked=(ROOT/'src/core/conversation_checked.hpp').read_text().replace('#pragma once','')
    q4=(ROOT/'include/strata/kernels/kv_q4.hpp').read_text()
    ng=(ROOT/'include/strata/kernels/ngram.hpp').read_text()
    ng_constants=ng[ng.index('inline constexpr int NG_N_EMBD'):ng.index('// The table:')]
    q4_types=q4[q4.index('inline constexpr int QK4_0'):q4.index('/// Bytes per cell and KV head:')]
    q4_count=take(q4,'inline uint64_t kv_q4_bytes_per_head(')
    formats=(ROOT/'include/strata/kernels/kv_stream.hpp').read_text()
    format_enum=formats[formats.index('enum KvFormat : int'):formats.index(';',formats.index('enum KvFormat : int'))+1]
    qsa=(ROOT/'include/strata/kernels/qsa.hpp').read_text()
    ng_constants+=take(qsa,'enum QsaStep : int ')+';\n'
    geometry_helpers=checked+'\nnamespace strata::kernels {\n'+ng_constants+q4_types+q4_count+format_enum+'\n}\nnamespace strata::core {\n'
    geometry_helpers+=take((ROOT/'include/strata/core/layer.hpp').read_text(),'inline int qsa_kv_format(')
    geometry_helpers+=take(state_header,'struct ConversationStateSizes ')+';\nnamespace {\nusing conversation_detail::add;using conversation_detail::product;\n'
    geometry_helpers+=take(states,'std::array<int64_t, 18> geometry_key(')+take(states,'bool fail(')+'\n}\n'
    geometry_helpers+=take(states,'bool conversation_state_sizes(')+take(states,'bool conversation_session_sizes(')
    geometry_helpers+='namespace {\n'+take(snapshots,'struct Layout ')+';\n'+take(snapshots,'bool valid_extent(')+take(snapshots,'bool layout(')+'\n}\n'
    geometry_helpers+=take(snapshots,'bool conversation_kv_part_sizes(')+'\n}\n'
    code=code.replace('ACTUAL_IMAGE_SIZE_HELPERS',geometry_helpers)
    prefill_source=(ROOT/'src/prefill/prefill.cpp').read_text()
    prefill_header=(ROOT/'include/strata/prefill/prefill.hpp').read_text()
    import re
    producer=re.search(r'constexpr int64_t N =[^;]+;',prefill_source)[0]+'\n'
    producer+=re.search(r'inline constexpr int kMaxStagerThreads =[^;]+;',prefill_header)[0]+'\n'
    producer+=take(prefill_source,'static bool prepare_paid_uploads(')
    code=code.replace('ACTUAL_PREFILL_IMAGE_PRODUCER',producer)
    code=code.replace('ACTUAL_PREFILL_WORKSPACE_ADMISSION',take(prefill_source,'WorkspaceAdmission Prefill::admit_owned_workspace('))
    code=code.replace('ACTUAL_PREFILL_IMAGE_TARGET',take(prefill_source,'bool Prefill::owned_image_target('))
    code=code.replace('ACTUAL_PREFILL_IMAGE_ROWS',take(prefill_source,'bool Prefill::owned_image_rows_present('))
    run_impl=take(prefill_source,'bool Prefill::run_impl(')
    row_prefix=run_impl[run_impl.index('    if(workspace_credit_&&'):run_impl.index('    const int32_t* owned_main_positions=')]
    code=code.replace('ACTUAL_IMAGE_ROWS_PREFIX',row_prefix)

    driver=take((ROOT/'src/program/generate.cpp').read_text(),'class OwnedImageRequestDriver final ')
    code=code.replace('ACTUAL_IMAGE_DRIVER',driver)

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

    # Existing source collectors stop at the first common-TU subprocess.
    (directory/'production.cpp').write_text(code)

    # Compile invariant TUs once in this invocation; never reuse cached objects.
    flags=['-std=c++20','-O0','-Wall','-Wextra','-Werror','-DSTRATA_OWNED_BATCH_AVAILABLE=1',
        '-I'+str(HERE/'stubs'),'-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/program')]
    common_sources=[ROOT/'src/core/window_budget.cpp',ROOT/'src/core/native_retirement.cpp',ROOT/'src/core/window_hash.cpp']
    common_objects=[]
    for source in common_sources:
        obj=directory/('owned_slots_common_'+source.stem+'.o')
        obj.unlink(missing_ok=True)
        subprocess.run(['c++',*flags,'-c',str(source),'-o',str(obj)],check=True)
        if not obj.is_file() or not obj.stat().st_size:
            raise RuntimeError('fresh common object missing/empty: '+str(obj))
        common_objects.append(obj)

    with compiled_variants(directory,variants,code,flags,common_objects) as compiled:
        for label,binary in compiled:
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
