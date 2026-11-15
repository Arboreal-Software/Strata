// Concrete checked execution members. Included in generate.cpp AFTER actual
// Drive/SplitDrive/drive_pool_* declarations and the driver class header,
// OUTSIDE anonymous namespace. Only the explicit disposable experiment uses them.
#include "strata/core/window_device.hpp"
#include <cmath>
#include <cstring>
namespace strata::program::suffix_driver_detail {
struct NativeBinding {
    core::SessionState* session=nullptr;core::Verifier* verifier=nullptr;
    int device=-1;int64_t begin=0,end=0,covered=0;bool ple_commit=false;
};
struct Bindings {
    std::array<NativeBinding,8> native{};
    std::array<core::Verifier*,8> chain{};
    size_t native_count=0,chain_count=0;
};
template<class Context>
bool bind(const Context& c,Bindings& out,const char*& why) noexcept {
    auto refuse=[&](const char* text){why=text;return false;};
    if(c.geometry.family!=::strata::ModelFamily::Qwen4Exp||c.geometry.n_layers<=0||c.geometry.n_layers>4096||
       c.geometry.qsa_interval<=0||c.geometry.head_dim<=0||c.geometry.head_dim%64||
       c.geometry.idx_block!=4||c.geometry.idx_key_dim<=0||c.geometry.n_head_kv<=0)
        return refuse("actual native geometry");
    if(!c.owner||!c.owner->model||!c.owner->primary_verifier||
       c.owner->primary_verifier.get()!=&c.primary_verifier||!c.owner->sessions.healthy()||
       !c.owner->drive||!c.owner->pool||!c.owner->cache||!c.primary_stream||
       c.primary_stream!=c.owner->main_stream||c.primary_device!=c.primary_verifier.device()||
       !c.owner->owns_expert_dispatch(&c.owner->drive->d))return refuse("owned runtime/dispatch binding");
    const auto& allocation=c.owner->sessions.stages();
    if(allocation.empty()||allocation.size()>8||allocation.size()!=c.secondary_stages.size()+1||
       allocation.size()!=c.owner->stages.size()+1||c.stage_count<1||c.stage_count>8||
       allocation.front().session!=&c.primary_session)return refuse("owned native stage inventory");
    const auto& parent=c.owner->sessions.reservation();
    if(!parent||!parent->canonical_owner(parent)||
       (parent->purpose()!=core::ReservationPurpose::private_work&&parent->purpose()!=core::ReservationPurpose::saved_maintenance))return refuse("actual owned session parent");
    if(c.owner->drive->d.failed||c.owner->drive->d.remote_count||c.owner->remote_opt||
       c.owner->drive->d.lookahead||c.owner->foresight||
       (c.owner->adapt_gate&&c.owner->adapt_gate->active.load(std::memory_order_acquire)))return refuse("active incompatible session writer");
    out.native_count=allocation.size();int64_t end=0;
    for(size_t i=0;i<allocation.size();++i){const auto& a=allocation[i];
        if(!a.owner||!a.session||a.device<0||size_t(a.device)>=parent->resources().device_bytes.size()||
           !parent->resources().device_bytes[size_t(a.device)]||
           a.session->layer_lo!=end||a.session->layer_hi<=end||a.session->layer_hi>c.geometry.n_layers||
           !a.session->qsa_states||a.session->qsa_alloc<=0||a.session->qsa_ord0<0||
           a.session->qsa_ord0>c.geometry.n_qsa_layers()-a.session->qsa_alloc)
            return refuse("actual native session/geometry");
        if(i&&(c.owner->stages[i-1].get()!=c.secondary_stages[i-1].get()||
           a.session!=&c.secondary_stages[i-1]->ss||a.device!=c.secondary_stages[i-1]->dev||
           !c.secondary_stages[i-1]->stream))return refuse("secondary runtime ownership");
        for(int64_t q=0;q<a.session->qsa_alloc;++q){const auto& state=a.session->qsa_states[a.session->qsa_ord0+q];
            if(state.kv_rot||state.kv_q4||state.kv_hybrid||core::qsa_kv_format(state)!=1)return refuse("actual native format is not unrotated INT8");}
        out.native[i]={a.session,nullptr,a.device,a.session->layer_lo,a.session->layer_hi,a.session->layer_lo,false};
        end=a.session->layer_hi;
    }
    if(end!=c.geometry.n_layers)return refuse("incomplete native layer partition");
    out.chain_count=size_t(c.stage_count);out.chain[0]=&c.primary_verifier;
    const auto root=c.primary_verifier.serial_routing_binding();
    // Owned initialization already refuses same-device shared-session split.
    // First bridge supports the reachable one-verifier-per-native-stage shape.
    if(out.chain_count!=allocation.size())return refuse("unsupported verifier/native stage topology");
    for(size_t i=1;i<out.chain_count;++i)out.chain[i]=&c.owner->stages[i-1]->ver;
    const bool split=out.chain_count>1;
    if(!split){
        if(c.pool!=&::drive_pool_multi||c.pool_user!=c.owner->drive.get()||
           c.owner->drive->d.plan!=root.plan||c.owner->drive->d.cache_base!=c.owner->cache->device_slot(0)||
           c.owner->drive->d.cache_slot_off!=c.owner->cache->slot_offsets())return refuse("actual direct pool binding");
    }else if(!c.owner->split_drive||c.pool!=&::drive_pool_split||c.pool_user!=c.owner->split_drive.get()||
             c.owner->split_drive->base!=c.owner->drive.get()||c.owner->split_drive->n!=c.stage_count)
        return refuse("actual serial split pool binding");
    end=0;
    for(size_t i=0;i<out.chain_count;++i){auto* verifier=out.chain[i];const auto b=verifier->serial_routing_binding();
        const auto* next=i+1<out.chain_count?out.chain[i+1]:nullptr;
        if(verifier->has_adapt_gate())return refuse("bound adaptive session writer");
        if(verifier->max_t()!=2||verifier->pcie_balance_on()||b.optimizer||b.geometry!=&c.geometry||!b.session||!b.plan||
           b.layer_begin!=end||b.layer_end<=end||b.layer_end>c.geometry.n_layers||b.next!=next||
           b.next_user!=(next?static_cast<const void*>(c.owner->split_drive.get()):nullptr))return refuse("actual serial verifier chain");
        size_t native=allocation.size();for(size_t j=0;j<allocation.size();++j)if(allocation[j].session==b.session){native=j;break;}
        if(native==allocation.size()||verifier->device()!=out.native[native].device||
           b.layer_begin!=out.native[native].covered||b.layer_end>out.native[native].end)return refuse("verifier/session layer ownership");
        auto& n=out.native[native];if(!n.verifier)n.verifier=verifier;n.covered=b.layer_end;
        if(b.layer_begin<=1&&1<b.layer_end){if(n.ple_commit)return refuse("duplicate PLE commit producer");n.ple_commit=true;}
        if(split){const auto& d=*c.owner->split_drive;
            if(d.end[i]!=b.layer_end||d.plan[i]!=b.plan)return refuse("actual split plan/layer binding");
            if(native==0){if(d.cache_base[i]!=c.owner->cache->device_slot(0)||d.cache_slot_off[i]!=c.owner->cache->slot_offsets())return refuse("primary split cache binding");}
            else {const auto& stage=*c.owner->stages[native-1];if(d.cache_base[i]!=stage.cache.device_slot(0)||d.cache_slot_off[i]!=stage.cache.slot_offsets())return refuse("secondary split cache binding");}
        }
        end=b.layer_end;
    }
    if(end!=c.geometry.n_layers)return refuse("incomplete verifier chain");
    for(size_t i=0;i<out.native_count;++i)if(!out.native[i].verifier||out.native[i].covered!=out.native[i].end)return refuse("native stage not covered by actual chain");
    return true;
}
inline bool present(const core::StateDigest& d) noexcept {for(auto b:d)if(b)return true;return false;}
} // namespace strata::program::suffix_driver_detail

namespace strata::program {
template<class Context>
bool SuffixExperimentDriver::validate_serving_qualification_execution(const Context& c,std::string& error) {
    suffix_driver_detail::Bindings bindings;
    const char* why=nullptr;
    if(!suffix_driver_detail::bind(c,bindings,why)){error=why;return false;}
    return true;
}
template<class Context>
bool SuffixExperimentDriver::validate(const Context& c,std::string& error){
    const auto& o=c.options;
    if(!o.owned_session_lifetime||!o.owned_model_lifetime||o.serve||o.spec!=2||
       o.suffix_experiment_requested_spec!=0||o.suffix_draft||o.lookup_chain||o.batch||o.pipeline_windows||
       o.batch_mtp||o.vision||o.kv_grow||o.adapt_every||o.adapt_async||o.no_pool||o.no_capture||
       !o.mtp.empty()||!o.draft_model.empty()||o.draft=="dflash2"||!o.cvec_files.empty()||
       !o.greedy||o.max_new<=0||o.max_new>64||o.max_context<=o.max_new||c.geometry.n_vocab<=0){
        error="unsupported resolved standalone suffix execution policy";return false;
    }
    // DriverRuntimeObject constructs RouterLookahead even when disabled. The
    // one startup start() call is skipped by exact environment0; actual dispatch
    // must also have no pointer to it. Ownership alone is NOT activity.
    const char* lookahead=std::getenv("STRATA_LOOKAHEAD");
    if(!lookahead||std::strcmp(lookahead,"0")!=0){error="explicit disabled lookahead required";return false;}
    // verify.cpp's presence-based diagnostic allocates another vocabulary row
    // and issues an unchecked transfer; it is outside this controller's budget.
    for(const char* diagnostic:{"STRATA_DBG_NAN","STRATA_TAP_FILE","STRATA_NAN_INJECT",
                                "STRATA_VERIFY_PROFILE","STRATA_VERIFY_NODES","STRATA_COMMIT_TIMING"})
        if(std::getenv(diagnostic)!=nullptr){error="unadmitted verifier diagnostic enabled";return false;}
    suffix_driver_detail::Bindings bindings;const char* why=nullptr;
    if(!suffix_driver_detail::bind(c,bindings,why)){error=why;return false;}
    // Exactly this relevant predictor is terminally closed once in validate.
    // It is an always-allocated placeholder when startup env0 skipped start().
    // Failed close is unknown producer ownership: do not unwind the owner.
    if(c.owner->lookahead&&!c.owner->lookahead->close_for_retirement())
        failstop("lookahead retirement uncertain before experiment");
    // Raw greedy head picks, with no per-request penalty/sampling side path.
    c.primary_verifier.set_head_sampling(false);
    // Persistent file fillers/parked pool workers may remain: they own source
    // buffers and cannot submit session work in this dedicated serial branch.
    // Only lookahead was closed above; no pool/file filler/runtime teardown.
    return true;
}

template<class Context>
SuffixExperimentDriver::Step SuffixExperimentDriver::reset(Context& c,SuffixRawTransaction& tx,uint64_t& next_cut,uint64_t output_watermark){
    suffix_driver_detail::Bindings bindings;const char* why=nullptr;
    if(!suffix_driver_detail::bind(c,bindings,why)||!next_cut||next_cut==UINT64_MAX||next_cut<=tx.cut_.id)
        failstop("reset binding/cut precondition");
    // Allocating history hash is prepared BEFORE begin; caller already owns its
    // controller-host H/global credit. Any later exception is failstop.
    const std::vector<int32_t> empty;const auto history=core::window_history_digest(empty);
    std::string error; // bounded controller diagnostic margin, not payload owner
    try {
        const auto begun=tx.begin_driver_work();if(!begun)failstop("reset begin refused");
        const bool reset_ok=c.owner->reset_sessions(c.geometry,error);
        if(!reset_ok||!c.owner->sessions.healthy())failstop("actual all-stage reset failed");
        if(!suffix_driver_detail::bind(c,bindings,why))failstop("reset bindings changed");
        std::array<core::StateCut,8> cuts{};
        core::StateCut cut;cut.id=next_cut;cut.history=history;
        // A clean recomputation resets tensors, not the already recorded output
        // transcript. Ordinary experiment resets retain their zero default.
        cut.output_watermark=output_watermark;
        for(size_t i=0;i<bindings.native_count;++i){const auto& b=bindings.native[i];
            if(b.session->ple_prev[0]!=-1||b.session->ple_prev[1]!=-1||b.session->ple_token!=-1)
                failstop("reset actual PLE endpoint differs");
            if(b.verifier->poll_snapshot_drain(error)!=core::Verifier::SnapshotDrain::drained)
                failstop("reset actual verifier drain incomplete");
            // This logical stage cut derives from checked reset_sessions over
            // the exact allocation plus this stage's actual drain, not input.
            cuts[i]=cut;
        }
        const auto complete=tx.complete_driver_work(reset_ok,cut,{cuts.data(),bindings.native_count});
        if(!complete)failstop("reset common cut refused");++next_cut;return {cut,-1};
    }catch(...){failstop("exception during attempted reset");}
}

template<class Context,class SerialUnit>
SuffixExperimentDriver::Step SuffixExperimentDriver::forward(Context& c,SerialUnit& serial,SuffixRawTransaction& tx,
    uint64_t& next_cut,int32_t token,uint64_t position,uint64_t accepted,core::StateDigest history,
    uint64_t exact_frontier,std::span<float> logits){
    suffix_driver_detail::Bindings bindings;const char* why=nullptr;
    if(!suffix_driver_detail::bind(c,bindings,why)||!next_cut||next_cut==UINT64_MAX||next_cut<=tx.cut_.id||
       token<0||int64_t(token)>=c.geometry.n_vocab||position!=tx.cut_.forwarded||position>=accepted||
       accepted>uint64_t(c.options.max_context)||position>=uint64_t(INT64_MAX)||
       !suffix_driver_detail::present(history)||exact_frontier>position+1||
       (!logits.empty()&&logits.size()!=size_t(c.geometry.n_vocab)))failstop("forward binding/token/cut precondition");
    if(tx.cut_.exact_frontier<tx.cut_.forwarded){
        if(exact_frontier!=tx.cut_.exact_frontier)failstop("forward cannot erase approximate ancestry");
    }else if(exact_frontier!=position+1)failstop("exact forward frontier mismatch");
    std::array<std::array<int32_t,2>,8> old_prev{};
    for(size_t i=0;i<bindings.native_count;++i)old_prev[i]={bindings.native[i].session->ple_prev[0],bindings.native[i].session->ple_prev[1]};
    std::string error;
    try {
        const auto begun=tx.begin_driver_work();if(!begun)failstop("forward begin refused");
        for(size_t i=0;i<bindings.native_count;++i)if(bindings.native[i].ple_commit)
            bindings.native[i].session->ple_token=token; // current input; commit shifts prev once
        auto& root=c.primary_verifier;int32_t picked=-1;
        c.owner->drive->d.layers=0;c.owner->drive->d.experts=0; // failed latch already checked, never clear it
        core::WindowDeviceScope run_device(root.device());
        if(!run_device.ok())failstop("forward owning device select failed");
        const bool ran=serial(root,1,&token,int64_t(position),c.pool,c.pool_user,&picked,error);
        if(!ran||c.owner->drive->d.failed||root.nonfinite_row()>=0)failstop("actual T1 run/dispatch/nonfinite failure");
        int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=root.device())failstop("forward device drift");
        const bool committed=root.commit(1,error);if(!committed)failstop("actual commit1 failed");
        const bool waited=root.wait_commit(error);if(!waited)failstop("actual recursive commit wait failed");
        if(cudaGetDevice(&current)!=cudaSuccess||current!=root.device()||!run_device.finish())failstop("forward caller device restore failed");
        if(!suffix_driver_detail::bind(c,bindings,why))failstop("forward bindings changed");
        if(!logits.empty()){
            core::WindowDeviceScope head(bindings.chain[bindings.chain_count-1]->device());
            if(!head.ok()||!root.copy_logits(0,logits.data())||cudaDeviceSynchronize()!=cudaSuccess||!head.finish())
                failstop("actual final-stage logits transfer failed");
            for(float value:logits)if(!std::isfinite(value))failstop("nonfinite copied logits");
        }
        if(picked<0||int64_t(picked)>=c.geometry.n_vocab)failstop("invalid actual greedy next token");
        core::StateCut cut;cut.id=next_cut;cut.accepted=accepted;cut.forwarded=position+1;
        cut.exact_frontier=exact_frontier;cut.history=history;cut.output_watermark=tx.cut_.output_watermark;
        std::array<core::StateCut,8> cuts{};
        for(size_t i=0;i<bindings.native_count;++i){const auto& b=bindings.native[i];
            if(b.ple_commit&&(b.session->ple_prev[0]!=old_prev[i][1]||b.session->ple_prev[1]!=token||b.session->ple_token!=token))
                failstop("actual committed PLE tokens differ");
            if(b.verifier->poll_snapshot_drain(error)!=core::Verifier::SnapshotDrain::drained)
                failstop("actual native-stage drain incomplete");
            // Authenticated root run+recursive commit/wait covered this exact
            // session/layer partition. Poll confirms its terminal stream cut.
            cuts[i]=cut;
        }
        const bool actual=ran&&committed&&waited&&!c.owner->drive->d.failed&&c.owner->sessions.healthy();
        const auto completed=tx.complete_driver_work(actual,cut,{cuts.data(),bindings.native_count});
        if(!completed)failstop("actual forward common cut refused");++next_cut;return {cut,picked};
    }catch(...){failstop("exception during attempted forward/commit/logits");}
}

// Only the serving oracle supplies this borrowed, sealed-prefix adapter. It has
// no transaction authority: actual owned devices/PLE/drains authenticate cuts.
template<class Context>
SuffixExperimentDriver::Step SuffixExperimentDriver::replay_prefill_piece(Context& c,SuffixRawTransaction& tx,
    uint64_t& next_cut,const PrefixReplayEvent& event,const PrefixReplayCallbacks& calls,
    std::span<const int32_t> history,uint64_t accepted,core::StateDigest digest,PrefixReplayAbort abort,bool& stopped) {
    suffix_driver_detail::Bindings bindings;const char* why=nullptr;
    const bool batch=event.kind==PrefixReplayEvent::Kind::prefill;
    if(!suffix_driver_detail::bind(c,bindings,why)||!calls.run_piece||!calls.refill||!calls.user||
       !next_cut||next_cut==UINT64_MAX||next_cut<=tx.cut_.id||event.from!=tx.cut_.forwarded||
       event.to<event.from||(batch&&event.to==event.from)||event.to>accepted||accepted!=history.size()||
       accepted>uint64_t(c.options.max_context)||!suffix_driver_detail::present(digest)||
       tx.cut_.exact_frontier!=tx.cut_.forwarded)failstop("prefix replay binding/cut bounds");
    if(batch&&abort()){stopped=true;return {tx.cut_,-1};} // no begin, no omitted-row certificate
    int entry_device=-1;
    if(cudaGetDevice(&entry_device)!=cudaSuccess||entry_device!=c.primary_device)
        failstop("prefix replay caller owning device differs");
    const auto original_bindings=bindings;
    const auto original_owner=c.owner; // retain actual executed runtime even if a callback substitutes it
    struct AllocationIdentity {
        const void* qsa=nullptr;const void* qsa_arena=nullptr;const void* qsa_scratch=nullptr;
        const void* gdn=nullptr;const void* ple=nullptr;const void* block=nullptr;const void* moe=nullptr;const void* residual=nullptr;
        int64_t cells=0,k=0,lo=0,hi=0,qo=0,qn=0,go=0,gn=0;
        bool operator==(const AllocationIdentity&)const=default;
    };
    auto allocation_identity=[](const core::SessionState& s){return AllocationIdentity{
        s.qsa_states,s.qsa_state_arena,s.qsa_buf_arena,s.gdn_state,s.ple_hist,s.block_arena,s.moe_arena,s.R,
        s.max_cells,s.k,s.layer_lo,s.layer_hi,s.qsa_ord0,s.qsa_alloc,s.gdn_ord0,s.gdn_alloc};};
    std::array<AllocationIdentity,8> original_allocation{};
    std::array<std::array<int32_t,3>,8> ple{};
    for(size_t i=0;i<bindings.native_count;++i){const auto* s=bindings.native[i].session;
        ple[i]={s->ple_prev[0],s->ple_prev[1],s->ple_token};original_allocation[i]=allocation_identity(*s);}
    std::string error;PrefixReplayProgress progress;
    try {
        if(!tx.begin_driver_work())failstop("prefix replay begin refused");
        const bool ran=batch?calls.run_piece(calls.user,event,history,abort,progress,error):calls.refill(calls.user,error);
        // Public Prefill::run joins every successor even on errors. A false
        // result can hide a second downstream error: never treat it as cancel.
        if(!ran)failstop("prefix actual prefill/refill or joined successor failed");
        int current_device=-1;
        if(cudaGetDevice(&current_device)!=cudaSuccess||current_device!=entry_device)
            failstop("prefix replay callback caller device drift");
        if(!suffix_driver_detail::bind(c,bindings,why)||!c.owner->sessions.healthy()||c.owner->drive->d.failed)
            failstop("prefix replay actual owned bindings/dispatch changed");
        if(c.owner!=original_owner||bindings.native_count!=original_bindings.native_count||
           bindings.chain_count!=original_bindings.chain_count)failstop("prefix replay executed runtime/stage inventory changed");
        for(size_t i=0;i<bindings.chain_count;++i)if(bindings.chain[i]!=original_bindings.chain[i])
            failstop("prefix replay executed verifier chain changed");
        for(size_t i=0;i<bindings.native_count;++i){const auto& before=original_bindings.native[i];const auto& after=bindings.native[i];
            if(before.session!=after.session||before.verifier!=after.verifier||before.device!=after.device||
               before.begin!=after.begin||before.end!=after.end||before.covered!=after.covered||before.ple_commit!=after.ple_commit||
               allocation_identity(*after.session)!=original_allocation[i])
                failstop("prefix replay executed native identity/layer/allocation layout changed");
        }
        uint64_t endpoint=event.to;
        if(batch){
            if(progress.stages!=bindings.native_count)failstop("prefix replay missing actual stage receipt");
            if(progress.stopped_at>=0){
                if(progress.stopped_at==0||uint64_t(progress.stopped_at)>=event.to-event.from)
                    failstop("prefix replay invalid graceful stop endpoint");
                endpoint=event.from+uint64_t(progress.stopped_at);stopped=true;
            }
            for(size_t i=0;i<bindings.native_count;++i)
                if(progress.stage_end[i]!=endpoint)failstop("prefix replay actual stage endpoints disagree");
        }
        core::StateCut cut;cut.id=next_cut;cut.accepted=accepted;cut.forwarded=endpoint;
        cut.exact_frontier=endpoint;cut.history=digest;cut.output_watermark=tx.cut_.output_watermark;
        std::array<core::StateCut,8> cuts{};
        for(size_t i=0;i<bindings.native_count;++i){const auto& b=bindings.native[i];
            core::WindowDeviceScope device(b.device);
            if(!device.ok()||cudaDeviceSynchronize()!=cudaSuccess||!device.finish())
                failstop("prefix replay actual all-device completion/restoration failed");
            if(b.verifier->poll_snapshot_drain(error)!=core::Verifier::SnapshotDrain::drained)
                failstop("prefix replay verifier drain incomplete");
            const int32_t previous0=batch?(endpoint>=2?history[size_t(endpoint-2)]:-1):ple[i][0];
            const int32_t previous1=batch?(endpoint?history[size_t(endpoint-1)]:-1):ple[i][1];
            if(b.session->ple_prev[0]!=previous0||b.session->ple_prev[1]!=previous1||b.session->ple_token!=ple[i][2])
                failstop("prefix replay actual PLE epilogue differs");
            cuts[i]=cut;
        }
        if(!tx.complete_driver_work(true,cut,{cuts.data(),bindings.native_count}))
            failstop("prefix replay actual common cut refused");
        ++next_cut;return {cut,-1};
    }catch(...){failstop("prefix replay exception retains callback and RAW owners");}
}
} // namespace strata::program
