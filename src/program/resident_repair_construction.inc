// Experimental dedicated resident repair; integration and hardware qualification pending.
// SOURCE ONLY / UNCOMPILED. This is a NEW file; existing owner/dispatch fragments
// remain untouched. Forward-declare ResidentConstructionInputs before the real
// controller; include this file AFTER controller and owner/dispatch definitions.
// Required controller declaration:
// static FactoryResult construct(const std::shared_ptr<::DriverWindowRuntime>&,
//   SuffixRawTransaction&,const ResidentConstructionInputs&,
//   std::shared_ptr<ResidentRepairRuntime>&,std::string&);
// Also uses controller_host_requirement(uint64_t max_tokens,uint64_t& out),
// the controller owner's checked sizeofController+8MiB+128T logical envelope.
// No public callback or caller 'completed' boolean is introduced.
#include "strata/core/layout.hpp"
#include "strata/core/resident_fault_receipt.hpp"
#include "strata/core/window_ple.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <algorithm>
#include <limits>
#include <new>
#include <type_traits>

namespace strata::program::resident_repair_detail {
struct ResidentConstructionInputs {
    int64_t max_cells=0;
    bool native_ple_key=false;
    bool fault_second_arena=false;
    int fault_receipt_fd=-1;
    // REVIEWED experimental cold/shared/runtime/graph+controller host envelope.
    // Actual geometry/explicit arena/mapped/native-session bytes are DERIVED
    // below, not accepted from this field. Nonzero per participating device and
    // host are REQUIRED. No claim CUDA obeys an allocator-hard hidden-byte cap.
    // Caller owns this bounded shape under its already admitted metadata lane.
    core::WindowResources platform_and_control;
    // Pinned current window_buffer.cpp metadata quote ceiling PER allocation:
    // actual Storage fields +256 +<=8 consumer int capacity (logical, not RSS).
    // A reviewed 4096B ceiling exceeds that current fixed schema. This must be
    // revisited if that allocator/source/ABI changes; an opaque lease is no proof.
    uint64_t buffer_metadata_host_ceiling=0;
};
namespace resident_construction_detail {
inline bool add(uint64_t& n,uint64_t v) noexcept {if(v>UINT64_MAX-n)return false;n+=v;return true;}
inline bool mul(uint64_t a,uint64_t b,uint64_t& out) noexcept {
    if(b&&a>UINT64_MAX/b)return false;out=a*b;return true;
}
// Retain actual credit through managed-object deletion AND allocator/controlblock
// deallocation, including weak observers. No runtime/service backreference.
template<class T> struct RetainedAllocator {
    using value_type=T;
    std::shared_ptr<const void> credit;
    explicit RetainedAllocator(std::shared_ptr<const void> c):credit(std::move(c)){}
    template<class U> RetainedAllocator(const RetainedAllocator<U>& a)noexcept:credit(a.credit){}
    template<class U>struct rebind{using other=RetainedAllocator<U>;};
    T* allocate(size_t n){return std::allocator<T>{}.allocate(n);}
    void deallocate(T* p,size_t n)noexcept {
        auto keep=credit;std::allocator<T>{}.deallocate(p,n);
    }
    template<class U>bool operator==(const RetainedAllocator<U>& a)const noexcept {
        return credit.get()==a.credit.get()&&!credit.owner_before(a.credit)&&!a.credit.owner_before(credit);
    }
    template<class U>bool operator!=(const RetainedAllocator<U>& a)const noexcept{return !(*this==a);}
};
struct Credits {
    std::shared_ptr<const void> bootstrap; // released last
    std::shared_ptr<const void> explicit_setup;
    std::shared_ptr<const void> allocator_host_cap; // H ONLY, not second global reserve
};
} // namespace resident_construction_detail

FactoryResult ResidentRepairController::construct(
    const std::shared_ptr<::DriverWindowRuntime>& service,SuffixRawTransaction& tx,
    const ResidentConstructionInputs& input,std::shared_ptr<ResidentRepairRuntime>& out,
    std::string& error) {
    using namespace resident_construction_detail;
    auto refused=[&](const char* why){error=why;return FactoryResult{FactoryStatus::invalid_binding,why};};
    // All source handles are real retained service owners. Never copy PleRun,
    // DriverWindowRuntime, GpuStage, populated dispatch vectors or raw weights.
    if(out||!service||!service->geometry_owner||!service->ple_table_owner||
       !service->ple_table_owner->is_open()||!service->model||!service->model->healthy()||
       !service->sessions.healthy()||!service->source||!service->pool||!service->cache||!service->drive||
       !service->owns_expert_dispatch(&service->drive->d)||service->drive->d.failed||
       service->background_retirement.load(std::memory_order_acquire)!=0||
       !tx.initialized_||tx.phase_!=SuffixRawTransaction::Phase::drained||
       tx.controller_!=std::this_thread::get_id()||!tx.allocation_.healthy())
        return refused("actual drained service/geometry/table/RAW binding required");
    if(const char* why=resident_service_refusal(*service))return refused(why);
    const auto& g=*service->geometry_owner;
    const bool slot_bound=bool(tx.slot_occupancy_);
    if(slot_bound&&!checked_slot_live_binding(service,tx,error))
        return refused("actual typed service-owned slot RAW/model binding required");
    const auto& live=slot_bound?tx.allocation_.stages():service->sessions.stages();
    if(input.fault_second_arena) {
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
        if(live.size()!=2||live[0].device==live[1].device||!core::resident_fault_receipt::valid_fd(input.fault_receipt_fd))
            return refused("actual dual-stage fault/receipt scope required");
#else
        return refused("resident allocation fault build unavailable");
#endif
    }else if(input.fault_receipt_fd!=-1)return refused("unarmed resident fault receipt");
    const auto& model=service->model->stages();
    const auto& parent=slot_bound?tx.allocation_.reservation():service->sessions.reservation();
    const auto& tx_parent=tx.allocation_.reservation();
    if(!parent||!tx_parent||parent.get()!=tx_parent.get()||parent.owner_before(tx_parent)||tx_parent.owner_before(parent)||
       live.empty()||live.size()>maximum_stages||model.size()!=live.size()||service->stages.size()+1!=live.size()||
       input.max_cells<=0||input.max_cells>INT32_MAX-7||input.max_cells%4||
       g.n_layers<=0||g.n_layers>4096||g.qsa_interval<=0||g.qsa_interval>g.n_layers||g.n_expert<=0||g.n_expert>INT16_MAX||
       g.idx_top_k<=0||g.idx_top_k>INT32_MAX-3||g.idx_block!=4||g.head_dim!=256||
       input.buffer_metadata_host_ceiling<4096)
        return refused("construction identity/native geometry/metadata envelope");
    for(int64_t value:{g.n_embd,g.hc,g.hc_lr,g.ssm_conv_channels,g.ssm_value_dim,g.ssm_v_heads,
        g.n_head,g.n_head_kv,g.idx_q_heads,g.idx_key_dim,g.n_ff})
        if(value<=0||value>INT32_MAX)return refused("bounded native helper integer dimensions required");
    if(g.n_head>INT32_MAX/256||g.n_embd%32||g.n_ff%32||g.ssm_state_size!=128||g.ssm_d_conv!=4)
        return refused("actual restricted verifier/quantizer geometry required");
    const auto& platform=input.platform_and_control;
    // A genuine bounded experimental envelope is supplied by reviewed driver
    // policy; absent/zero envelopes REFUSE, never fabricate ready or zero-cost.
    uint64_t controller_bytes=0;
    if(!ResidentRepairController::controller_host_requirement(uint64_t(input.max_cells),controller_bytes))
        return refused("actual bounded controller metadata quote refused");
    uint64_t minimum_control=sizeof(ResidentRepairRuntime)+sizeof(ResidentExpertService)+
        sizeof(ResidentDispatchBinding)+sizeof(Credits)+maximum_stages*4096+65536;
    uint64_t verifier_objects=0;
    if(!mul(live.size(),sizeof(core::Verifier),verifier_objects)||
       !add(minimum_control,verifier_objects)||!add(minimum_control,controller_bytes))
        return refused("actual verifier/controller host quote overflow");
    // Price actual fresh dispatch vector element requests before constructing it.
    // Doubling is pinned stdlib growth headroom, not a whole-RSS guarantee.
    const auto& dispatch=service->drive->d;
    auto control_vector=[&](uint64_t count,uint64_t width){uint64_t n=0;return mul(count,width,n)&&mul(n,2,n)&&add(minimum_control,n);};
    if(!control_vector(10,sizeof(typename std::remove_cvref_t<decltype(dispatch.jobs)>::value_type))||
       !control_vector(kernels::cpu::MAXT,sizeof(typename std::remove_cvref_t<decltype(dispatch.act_multi)>::value_type))||
       !control_vector(kernels::cpu::MAXT,kernels::cpu::kNativeActBytes)||
       !control_vector(uint64_t(kernels::cpu::MAXT)*10,sizeof(typename std::remove_cvref_t<decltype(dispatch.jobs_multi)>::value_type))||
       !control_vector(g.n_expert,sizeof(int16_t))||
       !control_vector(uint64_t(kernels::cpu::MAXT)*10,sizeof(typename std::remove_cvref_t<decltype(dispatch.zc_fmt)>::value_type)))
        return refused("actual private dispatch metadata arithmetic overflow");
    if(platform.device_bytes.size()!=parent->resources().device_bytes.size()||
       platform.host_bytes<minimum_control||platform.pinned_bytes>platform.host_bytes||
       platform.io_slots||platform.disk_bytes||platform.shadows)
        return refused("missing reviewed platform/control host and device envelope");
    int64_t end=0;size_t ple_stage=maximum_stages;
    for(size_t i=0;i<live.size();++i){const auto& a=live[i];
        if(!a.owner||!a.session||!model[i]||!model[i]->healthy()||a.device!=model[i]->device()||
           a.device<0||size_t(a.device)>=platform.device_bytes.size()||!platform.device_bytes[size_t(a.device)]||
           a.session->layer_lo!=end||a.session->layer_hi<=end||a.session->layer_hi>g.n_layers||
           !a.session->qsa_states||a.session->qsa_alloc<=0||a.session->qsa_ord0<0||
           a.session->qsa_ord0>g.n_qsa_layers()-a.session->qsa_alloc)
            return refused("actual model/native stage topology and envelope");
        const auto& q=a.session->qsa_states[a.session->qsa_primary()];
        if(q.max_cells!=input.max_cells||core::qsa_kv_format(q)!=1||q.kv_rot||q.kv_q4||q.kv_hybrid)
            return refused("factory first target requires fixed matching native INT8 sessions");
        if(a.session->layer_lo<=1&&1<a.session->layer_hi)ple_stage=i;
        end=a.session->layer_hi;
    }
    if(end!=g.n_layers||ple_stage==maximum_stages)return refused("complete stage/PLE ownership missing");
    const auto& live_ple=live[ple_stage].session->ple;
    if(live_ple.table!=service->ple_table_owner.get()||!live_ple.read_context||
       input.native_ple_key!=(live_ple.w.key_native_data!=nullptr))
        return refused("actual retained shared PLE table/native key binding differs");
    std::shared_ptr<const void> bootstrap;
    // Ledger-operation shape/control transients are existing admitted metadata
    // operations; no new session/consumer/image/handle allocation precedes this.
    auto admitted=tx.admit_resident_setup(platform,bootstrap);
    if(!admitted)return refused("global+H platform/control admission refused");
    bool physical_started=false;
    std::shared_ptr<ResidentRepairRuntime> r; // OUTSIDE try: failstop catch runs before owner destruction
    try {
        auto credits=std::allocate_shared<Credits>(RetainedAllocator<Credits>{bootstrap});
        credits->bootstrap=bootstrap;
        std::vector<core::NativeSessionRequest> requests;requests.reserve(live.size());
        std::array<core::VerifierSetupQuoteLimits,maximum_stages> caps{};
        core::WindowResources explicit_setup;explicit_setup.device_bytes.resize(platform.device_bytes.size());
        core::WindowResources native_expected;native_expected.device_bytes.resize(platform.device_bytes.size());native_expected.shadows=1;
        uint64_t external_host=0,residency_bytes=0;
        if(!mul(uint64_t(g.n_layers),uint64_t(g.n_expert),residency_bytes)||
           !mul(residency_bytes,sizeof(int32_t),residency_bytes))return refused("residency size overflow");
        core::VerifierSerialHandoffQuote handoff;
        if(core::quote_verifier_serial_handoffs(g.n_embd,g.hc,uint32_t(live.size()),UINT64_MAX,handoff)!=core::VerifierSetupQuoteStatus::ok)
            return refused("actual handoff layout refused");
        const auto& native=kernels::cpu::expert_layout();
        if(!native.native||native.max_blob<=0)return refused("actual immutable native expert layout missing");
        for(size_t i=0;i<live.size();++i){const auto& a=live[i];
            core::SessionPrivatePlan session;
            if(!core::session_private_plan(g,input.max_cells,10,a.session->layer_lo,a.session->layer_hi,session))
                return refused("actual native private plan refused");
            core::NativeSessionRequest request;request.device=a.device;request.layer_begin=a.session->layer_lo;
            request.layer_end=a.session->layer_hi;request.ple_buffers=i==ple_stage;
            request.ple_native_key=request.ple_buffers&&input.native_ple_key;
            if(input.fault_second_arena&&i==1){
                request.qualification_fault=core::NativeAllocationQualificationFault::second_arena_cuda_refusal;
                request.qualification_receipt_fd=input.fault_receipt_fd;
            }
            requests.push_back(request);
            if(!add(native_expected.device_bytes[size_t(a.device)],session.device_bytes)||
               !add(native_expected.host_bytes,session.host_bytes)||!add(native_expected.pinned_bytes,session.pinned_bytes))
                return refused("native allocation exact plan aggregate overflow");
            if(request.ple_native_key&&!add(native_expected.device_bytes[size_t(a.device)],kernels::native_q8_1_bytes(kernels::NG_N_EMBD)))
                return refused("actual native PLE quantizer scratch overflow");
            if(request.ple_buffers&&(!add(native_expected.device_bytes[size_t(a.device)],uint64_t(kernels::NG_N_EMBD)*sizeof(float))||
               !add(native_expected.device_bytes[size_t(a.device)],core::ple_run_scratch_bytes())||
               !add(native_expected.host_bytes,uint64_t(kernels::NG_N_EMBD)*sizeof(float)*2)))
                return refused("actual PLE embedding/scratch/host overlap overflow");
            // EXACT native allocator host charge; device/host are reserved by
            // allocate_native_sessions itself, not charged again by setup below.
            if(!add(external_host,session.host_bytes))return refused("native host quote overflow");
            if(request.ple_buffers&&(!add(external_host,uint64_t(kernels::NG_N_EMBD)*sizeof(float)*2)||
               !add(external_host,core::private_ple_host_headroom)))return refused("PLE host quote overflow");
            // Device-only residency buffer still owns separately globally-charged
            // host metadata. H cap includes reviewed metadata ceiling once.
            if(!add(external_host,input.buffer_metadata_host_ceiling))return refused("buffer host overflow");
            const auto* output=model[i]->weights().find("output.weight");
            if(!output||output->ne1<=0)return refused("actual immutable output geometry missing");
            core::VerifierSetupDimensions d;d.n_layers=g.n_layers;d.n_gdn_layers=g.n_gdn_layers();d.n_qsa_layers=g.n_qsa_layers();
            d.n_embd=g.n_embd;d.hc=g.hc;d.hc_lr=g.hc_lr;d.conv_channels=g.ssm_conv_channels;d.value_dim=g.ssm_value_dim;d.v_heads=g.ssm_v_heads;
            d.n_head=g.n_head;d.head_dim=g.head_dim;d.n_head_kv=g.n_head_kv;d.idx_q_heads=g.idx_q_heads;d.idx_key_dim=g.idx_key_dim;
            d.n_ff=g.n_ff;d.n_expert=g.n_expert;d.n_vocab=output->ne1;d.max_cells=input.max_cells;
            d.selection_cap=kernels::qsa_selection_width(kernels::kTopkMaxCells,core::qsa_shapes(g));
            core::VerifierSetupNativeRequests bytes{
                std::max<uint64_t>(kernels::moe_hit_grouped_scratch_bytes(20,g.n_embd,g.n_ff),kernels::native_expert_scratch_bytes(20,g.n_ff)),
                kernels::argmax_rows_scratch_bytes(2),uint64_t(native.max_blob)};
            core::VerifierSetupProfile profile;profile.external_main_stream=true;profile.side_mixer_mode=2;
            for(const auto& format:native.fmt)profile.had_gu_any|=format.had_gu;
            // Reserve optional qfuse counters conservatively even when disabled;
            // actual restricted init still derives its own flags and helper bytes.
            profile.qfuse_counters=true;core::VerifierSetupQuote quote;
            if(core::quote_verifier_explicit_setup(d,bytes,profile,{UINT64_MAX,UINT64_MAX},quote)!=core::VerifierSetupQuoteStatus::ok)
                return refused("actual verifier explicit request quote refused");
            caps[i]={quote.arena_bytes+quote.extra_device_bytes,quote.mapped_host_bytes};
            if(caps[i].max_device_bytes<quote.arena_bytes||!add(explicit_setup.device_bytes[size_t(a.device)],caps[i].max_device_bytes)||
               !add(explicit_setup.host_bytes,quote.mapped_host_bytes)||!add(explicit_setup.pinned_bytes,quote.pinned_bytes))
                return refused("explicit setup aggregate overflow");
        }
        if(!add(external_host,handoff.mapped_host_bytes)||
           !add(external_host,(live.size()-1)*input.buffer_metadata_host_ceiling))return refused("mapped handoff host overflow");
        admitted=tx.admit_resident_setup(explicit_setup,credits->explicit_setup);
        if(!admitted)return refused("actual explicit setup global+H admission refused");
        // Actual native/PLE/buffer APIs below each reserve GLOBAL once. This
        // companion is H-only; it cannot prove any external runtime obeys a cap.
        admitted=tx.claim_external_splice(external_host,credits->allocator_host_cap);
        if(!admitted)return refused("native/PLE/buffer aggregate-H capacity refused");
        std::shared_ptr<const core::ReservationLease> gate_metadata;std::shared_ptr<const void> gate_cap;
        admitted=tx.admit_completion_gate_metadata(gate_metadata,gate_cap);
        if(!admitted)return refused("actual exclusive foreground gate metadata admission refused");
        // Private constructor access stays in real friend controller. Retained
        // allocator+deleter keep all credits through object/controlblock free.
        auto make_runtime=[&]{
            RetainedAllocator<ResidentRepairRuntime> a{credits};auto* p=a.allocate(1);
            try {::new(static_cast<void*>(p)) ResidentRepairRuntime();}catch(...){a.deallocate(p,1);throw;}
            return std::shared_ptr<ResidentRepairRuntime>(p,[a](ResidentRepairRuntime* q)mutable{q->~ResidentRepairRuntime();a.deallocate(q,1);},a);
        };
        r=make_runtime();r->setup_credit_=credits;r->model_=service->model;
        RetainedAllocator<ResidentExpertService> sa{credits};auto* raw=sa.allocate(1);
        try {::new(static_cast<void*>(raw)) ResidentExpertService(service);}catch(...){sa.deallocate(raw,1);throw;}
        r->service_=std::shared_ptr<ResidentExpertService>(raw,[sa](ResidentExpertService* p)mutable{p->~ResidentExpertService();sa.deallocate(p,1);},sa);
        if(slot_bound){
            r->service_->live_slot_=tx.slot_occupancy_;r->service_->live_slot_allocation_=tx.allocation_;r->service_->count_=live.size();
            for(size_t stage=0;stage<live.size();++stage)r->service_->live_slot_bindings_[stage]=tx.slot_bindings_[stage];
        }
        r->stage_count_=live.size(); // cut_ remains UNSET until actual checked reset/all-stage drain.
        physical_started=true;
        auto allocated=core::allocate_native_sessions(g,input.max_cells,10,requests,tx.budget_,r->allocation_);
        if(!allocated){
            if(allocated.status==core::NativeAllocationStatus::unhealthy)core::retirement_failed("private native allocation uncertain");
            physical_started=false;return refused("native allocator genuine global reservation refused before device work");
        }
        if(!r->allocation_.reservation()||r->allocation_.reservation()->resources()!=native_expected)
            core::retirement_failed("native allocator quote changed during actual construction");
        r->phase_=RuntimePhase::sessions_only;
        for(const auto& a:r->allocation_.stages()) {
            if(!a.owner||!a.session||!a.session->qsa_states||a.session->qsa_alloc<=0)
                core::retirement_failed("fresh native allocation inventory missing");
            for(int64_t q=0;q<a.session->qsa_alloc;++q){const auto& state=a.session->qsa_states[a.session->qsa_ord0+q];
                if(core::qsa_kv_format(state)!=1||state.kv_rot||state.kv_q4||state.kv_hybrid)
                    core::retirement_failed("global native configuration changed during private construction");
            }
        }
        // From this point EVERY failure is strict failstop before any local owner
        // can unwind; allocator/session/table/buffer/stream owners remain alive.
        for(size_t i=0;i<r->stage_count_;++i){auto& dst=r->stages_[i];const auto& a=r->allocation_.stages()[i];
            auto* cache=i?&service->stages[i-1]->cache:service->cache.get();
            dst.session_owner=a.owner;dst.session=a.session;
            dst.service={a.device,a.session->layer_lo,a.session->layer_hi,model[i].get(),cache};
            if(!dst.handles.declare_consumer_device(a.device,error)||
               !dst.handles.create_stream(a.device,cudaStreamNonBlocking,dst.stream,error))
                core::retirement_failed("private stream construction failed");
            if(!core::allocate_window_buffer(a.device,residency_bytes,std::vector<int>{a.device},tx.budget_,r->allocation_.reservation(),dst.retained_buffers[0],error))
                core::retirement_failed("private residency buffer allocation failed");
        }
        std::vector<int> consumers;consumers.reserve(live.size());for(const auto& a:live)consumers.push_back(a.device);
        for(size_t i=0;i+1<r->stage_count_;++i)
            if(!core::allocate_window_buffer(live[i].device,handoff.bytes_per_edge,consumers,tx.budget_,r->allocation_.reservation(),r->handoffs_[i],error,core::WindowBufferMemory::mapped_host))
                core::retirement_failed("private mapped handoff allocation failed");
        if(!core::bind_owned_private_ple(r->allocation_,ple_stage,model[ple_stage],service->ple_table_owner,
               input.native_ple_key,kernels::ple_artifact_consts(),tx.budget_,error))
            core::retirement_failed("private actual table/model PLE binding failed");
        if(!ResidentDispatchBinding::prepare_chain(*r,g,std::span<const core::VerifierSetupQuoteLimits>(caps.data(),r->stage_count_),
               std::move(gate_metadata),std::move(gate_cap),error)||!r->dispatch_||!r->dispatch_->initialize_private(error))
            core::retirement_failed("private actual chain prepare/init failed");
        if(r->phase_!=RuntimePhase::ready)core::retirement_failed("private construction did not reach checked ready");
        out=std::move(r);return {FactoryStatus::prepared,"actual private resident chain constructed; no unit submitted/published"};
    }catch(...){
        if(physical_started)core::retirement_failed("private construction exception retains all actual owners");
        return refused("private host construction/admission allocation failed before device work");
    }
}
} // namespace strata::program::resident_repair_detail
