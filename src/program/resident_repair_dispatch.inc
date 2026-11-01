// Experimental dedicated resident repair; integration and hardware qualification pending.
// ISOLATED source: follows resident-repair-runtime.PROPOSAL.inc after real
// generate types. No legacy predicate/callback is modified. No execution here.
#include "strata/core/window_device.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <limits>
#include <optional>
namespace strata::program::resident_repair_detail {
class ResidentDispatchBinding final {
public:
    ResidentDispatchBinding(const ResidentDispatchBinding&)=delete;
    ResidentDispatchBinding& operator=(const ResidentDispatchBinding&)=delete;
    ~ResidentDispatchBinding()noexcept {
        if(callback_depth_ || service_->active_==this)
            core::retirement_failed("resident dispatch destroyed with live operation");
    }
private:
    friend class ResidentRepairController;
    ResidentDispatchBinding(ResidentRepairRuntime& runtime,const core::ModelGeometry& geometry)
        :runtime_(runtime),service_(runtime.service_),geometry_(geometry){}
    bool current_residency_digest(core::StateDigest& output)const noexcept {
        if(!drive_.d.host_res||!runtime_.stage_count_||runtime_.stage_count_>size_t(SplitDrive::kMax)||
           runtime_.stage_count_>runtime_.stages_.size()||geometry_.n_layers<=0||geometry_.n_expert<=0||
           uint64_t(geometry_.n_layers)>SIZE_MAX/sizeof(int32_t)/uint64_t(geometry_.n_expert))return false;
        core::WindowHasher hash;constexpr char domain[]="resident-private-original-cache-map-v1";
        hash.update(domain,sizeof(domain));
        hash.update(drive_.d.host_res,size_t(geometry_.n_layers)*size_t(geometry_.n_expert)*sizeof(int32_t));
        for(size_t i=0;i<runtime_.stage_count_;++i){const auto* cache=runtime_.stages_[i].service.cache;
            if(!cache||cache->slots()<1||cache->slot_offsets()!=split_.cache_slot_off[i]||
               cache->device_slot(0)!=split_.cache_base[i]||!cache->slot_offsets()||
               uint64_t(cache->slots())>SIZE_MAX/sizeof(uint64_t))return false;
            const int64_t count=cache->slots(),bytes=cache->bytes();
            hash.update(&count,sizeof(count));hash.update(&bytes,sizeof(bytes));
            hash.update(cache->slot_offsets(),size_t(count)*sizeof(uint64_t));
        }
        output=hash.digest();return true;
    }
    bool original_residency_restored()const noexcept {
        core::StateDigest actual{};return residency_bound_&&current_residency_digest(actual)&&actual==original_residency_;
    }

    // BEFORE init: bind new callbacks/topology, then seal the real gate. Main
    // creates gate_metadata via actual global foreground admission and retains
    // its aggregate-H companion; private_attachment setup credit is NOT that lease.
    static bool prepare_chain(ResidentRepairRuntime& r,const core::ModelGeometry& g,
        std::span<const core::VerifierSetupQuoteLimits> caps,
        std::shared_ptr<const core::ReservationLease> gate_metadata,
        std::shared_ptr<const void> gate_cap_credit,std::string& error) {
        if(r.dispatch_||!r.setup_credit_||!r.service_||!r.service_->live_slot_valid()||!r.service_->slot_private_allocation_disjoint(r.allocation_)||!r.model_||r.model_!=r.service_->model_||
           !gate_cap_credit||r.service_->geometry_owner_.get()!=&g||
           r.phase_!=RuntimePhase::sessions_only||!r.allocation_.healthy()||
           std::this_thread::get_id()!=r.controller_||r.service_->controller_!=r.controller_||
           r.service_->active_||r.service_->unhealthy_||r.stage_count_==0||r.stage_count_>maximum_stages||
           caps.size()!=r.stage_count_||r.allocation_.stages().size()!=r.stage_count_||
           r.service_->owner_->stages.size()+1!=r.stage_count_){error="preinit private chain owner/phase";return false;}
        auto b=std::shared_ptr<ResidentDispatchBinding>(new ResidentDispatchBinding(r,g));
        auto& old=*r.service_->owner_;auto& source=old.drive->d;
        if(const char* refused=resident_service_refusal(old)){error=refused;return false;}
        if(!old.owns_expert_dispatch(&source)||source.failed||!source.host_res||source.remote_count||source.peer||
           source.lookahead||source.fs||source.pcie_model.on||source.host_execution_observer||
           !source.usage.empty()||source.evict_sync||source.res_dirty||source.dres_check||source.n_expert!=g.n_expert||
           source.cache_blob<=0||(r.stage_count_>1&&(!old.split_drive||old.split_drive->n!=int(r.stage_count_)))){
            error="service is not fixed-profile serial expert service";return false;}
        if(g.n_expert<=0||g.n_expert>INT16_MAX||g.n_layers<=0||g.n_layers>4096){error="dispatch dimensions";return false;}
        auto& d=b->drive_.d;
        d.jobs.resize(10);d.act_multi.resize(kernels::cpu::MAXT);
        d.nact_multi.resize(size_t(kernels::cpu::MAXT)*kernels::cpu::kNativeActBytes);
        d.jobs_multi.resize(size_t(kernels::cpu::MAXT)*10);d.job_of.assign(size_t(g.n_expert),int16_t(-1));
        d.zc_fmt.resize(d.jobs_multi.size());
        d.src=source.src;d.pool=source.pool;d.cache=old.cache.get();d.n_expert=g.n_expert;d.host_res=source.host_res;
        d.cache_blob=source.cache_blob;d.split_rows=source.split_rows;d.hit_cpu_order=source.hit_cpu_order;
        // All other fields retain FRESH defaults, including no legacy hit scratch.
        b->split_.base=&b->drive_;b->split_.n=int(r.stage_count_);
        std::array<std::unique_ptr<core::Verifier>,maximum_stages> fresh;
        core::VerifierSerialHandoffQuote handoffs;
        if(core::quote_verifier_serial_handoffs(uint64_t(g.n_embd),uint64_t(g.hc),uint32_t(r.stage_count_),UINT64_MAX,handoffs)!=core::VerifierSetupQuoteStatus::ok){error="private handoff dimensions";return false;}
        int64_t end=0;
        for(size_t i=0;i<r.stage_count_;++i){const auto& stage=r.stages_[i];const auto& a=r.allocation_.stages()[i];
            auto* cache=i?&old.stages[i-1]->cache:old.cache.get();
            if(stage.verifier||!stage.session_owner||stage.session_owner.owner_before(a.owner)||a.owner.owner_before(stage.session_owner)||
               stage.session!=a.session||stage.service.device!=a.device||stage.service.weights!=r.model_->stages()[i].get()||
               stage.session->layer_lo!=end||stage.session->layer_hi<=end||stage.service.cache!=cache){error="preinit private stage mismatch";return false;}
            for(const auto& live:old.sessions.stages())if(a.session==live.session||
                (!a.owner.owner_before(live.owner)&&!live.owner.owner_before(a.owner))){error="private session aliases foreground";return false;}
            if(r.service_->live_slot_)for(const auto& live:r.service_->live_slot_allocation_.stages())
                if(a.session==live.session||(!a.owner.owner_before(live.owner)&&!live.owner.owner_before(a.owner))){
                    error="private session aliases owned foreground slot";return false;
                }
            if(i&&(!r.handoffs_[i-1].device_data()||!r.handoffs_[i-1].data()||r.handoffs_[i-1].bytes()<handoffs.bytes_per_edge)){error="missing private handoff input";return false;}
            if(i+1<r.stage_count_&&(!r.handoffs_[i].device_data()||!r.handoffs_[i].data()||r.handoffs_[i].bytes()<handoffs.bytes_per_edge)){error="missing private handoff output";return false;}
            fresh[i]=std::make_unique<core::Verifier>();
            if(!fresh[i]->set_strict_retirement(true,error)||!fresh[i]->set_checked_requested_layout(caps[i],error))return false;
            fresh[i]->set_stage(a.session->layer_lo,a.session->layer_hi,
                i?static_cast<const float*>(r.handoffs_[i-1].device_data()):nullptr,
                i+1<r.stage_count_?static_cast<float*>(r.handoffs_[i].device_data()):nullptr);
            if(stage.stream)fresh[i]->set_stream(stage.stream);
            b->split_.end[i]=a.session->layer_hi;b->split_.cache_base[i]=cache->device_slot(0);
            b->split_.cache_slot_off[i]=cache->slot_offsets();b->split_.pcie_num[i]=r.stage_count_>1?old.split_drive->pcie_num[i]:source.pcie_num;
            if(!b->split_.cache_base[i]||!b->split_.cache_slot_off[i]){error="shared cache inventory absent";return false;}
            end=a.session->layer_hi;
        }
        if(end!=g.n_layers){error="private preinit chain incomplete";return false;}
        for(size_t i=0;i<r.stage_count_;++i)fresh[i]->set_next(i+1<r.stage_count_?fresh[i+1].get():nullptr,i+1<r.stage_count_?b.get():nullptr);
        fresh[0]->set_head_sampling(false); // exact greedy head, no per-request sampling buffers
        // This actual API authenticates exclusivity, foreground purpose, host
        // quote and preinit strict topology. Move avoids an extra lease alias.
        if(!core::Verifier::prepare_serial_completion_gate(*fresh[0],std::move(gate_metadata),b->gate_,error,std::move(gate_cap_credit)))return false;
        for(size_t i=0;i<r.stage_count_;++i)r.stages_[i].verifier=std::move(fresh[i]);
        r.dispatch_=std::move(b);r.phase_=RuntimePhase::chain_prepared;return true;
    }
    // Actual init consumes independently owned residency uploads, not legacy
    // unowned d_res stack pointers. All setup/hidden-runtime admissions precede
    // this call. No callback/plan is visible until every init validates below.
    bool initialize_private(std::string& error){
        auto& r=runtime_;
        if(r.phase_!=RuntimePhase::chain_prepared||!gate_||!gate_->healthy_advisory()||service_->active_||
           std::this_thread::get_id()!=r.controller_){error="private init requires sealed preinit chain";return false;}
        r.phase_=RuntimePhase::constructing;
        try {
            for(size_t i=0;i<r.stage_count_;++i){auto& stage=r.stages_[i];auto* cache=stage.service.cache;
                const auto& table=stage.retained_buffers[0];
                const uint64_t layers=uint64_t(geometry_.n_layers),experts=uint64_t(geometry_.n_expert);
                if(layers>UINT64_MAX/experts/sizeof(int32_t)||!table.device_data()||table.bytes()<layers*experts*sizeof(int32_t))
                    core::retirement_failed("private residency allocation missing/undersized");
                core::WindowDeviceScope device(stage.service.device);if(!device.ok())core::retirement_failed("private init device select");
                if(cudaMemcpy(table.device_data(),drive_.d.host_res,size_t(layers*experts*sizeof(int32_t)),cudaMemcpyHostToDevice)!=cudaSuccess)
                    core::retirement_failed("private residency upload failed");
                core::VerifyHits hits;hits.d_res=static_cast<const int32_t*>(table.device_data());hits.h_res=drive_.d.host_res;
                hits.cache_base=cache->device_slot(0);hits.blob=drive_.d.cache_blob;hits.slot_off=cache->slot_offsets();hits.n_slots=cache->slots();
                if(!stage.verifier->init(stage.service.weights->weights(),geometry_,*stage.session,hits,stage.service.weights->head(),2,error)||
                   !device.finish())core::retirement_failed("private verifier init uncertain");
            }
            for(size_t i=0;i<r.stage_count_;++i){auto& stage=r.stages_[i];const auto actual=stage.verifier->serial_routing_binding();
                auto* next=i+1<r.stage_count_?r.stages_[i+1].verifier.get():nullptr;
                if(actual.session!=stage.session||actual.geometry!=&geometry_||actual.optimizer||!actual.plan||actual.next!=next||
                   actual.next_user!=(next?static_cast<const void*>(this):nullptr)||actual.layer_begin!=stage.session->layer_lo||
                   actual.layer_end!=stage.session->layer_hi||stage.verifier->device()!=stage.service.device||
                   stage.verifier->max_t()!=2||stage.verifier->pcie_balance_on())core::retirement_failed("private postinit actual routing mismatch");
                auto* owned_plan=stage.verifier->plan_sink();
                if(owned_plan!=actual.plan)
                    core::retirement_failed("private mutable plan differs from observed routing");
                split_.plan[i]=owned_plan;
                if(stage.verifier->poll_snapshot_drain(error)!=core::Verifier::SnapshotDrain::drained)
                    core::retirement_failed("private postinit native stage drain");
            }
            if(!gate_->healthy_advisory())core::retirement_failed("private postinit completion gate unhealthy");
            if(!current_residency_digest(original_residency_))core::retirement_failed("private original residency binding unavailable");
            residency_bound_=true;
            drive_.d.plan=split_.plan[0];drive_.d.cache_base=split_.cache_base[0];drive_.d.cache_slot_off=split_.cache_slot_off[0];drive_.d.pcie_num=split_.pcie_num[0];
            r.phase_=RuntimePhase::ready;return true;
        }catch(...){core::retirement_failed("private init exception retains admitted owners");}
    }

    static void pool(void* context,const float* x,const int32_t* ids,int64_t tokens,int64_t k,float* out,int64_t layer){
        auto& self=*static_cast<ResidentDispatchBinding*>(context);auto& r=self.runtime_;
        if(self.service_->active_!=&self || std::this_thread::get_id()!=r.controller_ ||
           r.phase_!=RuntimePhase::running || self.callback_depth_ || tokens!=1||k!=10||layer<0||layer>=self.geometry_.n_layers)
            core::retirement_failed("private dispatch callback outside actual operation");
        ++self.callback_depth_;
        try {
            size_t i=0;while(i+1<r.stage_count_&&layer>=self.split_.end[i])++i;
            auto& d=self.drive_.d;d.layers=layer;d.plan=self.split_.plan[i];
            d.cache_base=self.split_.cache_base[i];d.cache_slot_off=self.split_.cache_slot_off[i];d.pcie_num=self.split_.pcie_num[i];
            core::expert_pool_dispatch_multi(d,x,ids,tokens,k,out);
            --self.callback_depth_;
        }catch(...){core::retirement_failed("private expert callback exception; IO/pool ownership uncertain");}
    }
    // Concrete synchronous private forward unit. No callable asserted joins.
    // Foreground must also enter this same service phase before its own run;
    // main controller installs that exact wrapper before exposing either path.
    bool run_private_one(int32_t token,uint64_t position,int32_t& picked,std::string& error){
        auto& r=runtime_;auto& s=*service_;
        if(private_sealed_||!s.live_slot_valid()||std::this_thread::get_id()!=r.controller_||s.active_||s.unhealthy_||
           (r.phase_!=RuntimePhase::ready&&r.phase_!=RuntimePhase::drained)||
           !r.allocation_.healthy()||drive_.d.failed||!original_residency_restored()||position!=r.cut_.forwarded||
           r.cut_.exact_frontier!=r.cut_.forwarded||token<0||token>=geometry_.n_vocab||position>=uint64_t(INT64_MAX)||s.operation_==UINT64_MAX){
            error="private forward operation refused before submission";return false;}
        core::VerifierCompletionGate::Token completion;
        if(!gate_||!gate_->try_begin(completion)){error="private completion gate refused before submission";return false;}
        s.active_=this;++s.operation_;r.operation_=s.operation_;r.phase_=RuntimePhase::running;
        try {
            std::array<std::array<int32_t,2>,maximum_stages> previous{};
            for(size_t i=0;i<r.stage_count_;++i){auto& a=*r.stages_[i].session;previous[i]={a.ple_prev[0],a.ple_prev[1]};
                if(a.layer_lo<=1&&1<a.layer_hi)a.ple_token=token;}
            auto& root=*r.stages_[0].verifier;core::WindowDeviceScope owning(root.device());
            if(!owning.ok())core::retirement_failed("private forward device select");
            if(!root.run(1,&token,int64_t(position),&pool,this,&picked,error)||drive_.d.failed||root.nonfinite_row()>=0||
               !root.commit(1,error)||!root.wait_commit(error))
                core::retirement_failed("private actual run/commit/wait failed");
            int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=root.device())core::retirement_failed("private caller device drift");
            for(size_t i=0;i<r.stage_count_;++i){auto& stage=r.stages_[i];auto& a=*stage.session;
                if(a.layer_lo<=1&&1<a.layer_hi&&(a.ple_prev[0]!=previous[i][1]||a.ple_prev[1]!=token||a.ple_token!=token))
                    core::retirement_failed("private PLE commit mismatch");
                if(stage.verifier->poll_snapshot_drain(error)!=core::Verifier::SnapshotDrain::drained)
                    core::retirement_failed("private stage terminal drain incomplete");
            }
            if(callback_depth_||!owning.finish()||picked<0||picked>=geometry_.n_vocab)
                core::retirement_failed("private callback/device/output completion uncertain");
            // root.run is synchronous: actual expert_pool_dispatch_multi joins
            // pool.run_split_multi[_native], prefetch_end and serial HostGraphLaunch
            // before returning. No async helper/overlap path admitted here.
            // Controller still publishes authenticated history/common StateCut;
            // do not fabricate accepted/history/cut IDs inside this binding.
            core::VerifierCompletionGate::Cut completed;
            if(!gate_->commit_after_checked_completion(completion,completed))core::retirement_failed("private checked completion gate refused");
            last_completion_.reset();last_completion_.emplace(std::move(completed));last_completed_operation_=r.operation_;
            r.phase_=RuntimePhase::drained;s.active_=nullptr;return true;
        }catch(...){core::retirement_failed("private forward exception retains entire service/runtime");}
    }
    // The SAME serialized service gate protects the actual legacy foreground
    // call. Main routes foreground only through this member once installed;
    // retaining an old bypass call is explicitly invalid integration.
    bool run_foreground_one(int32_t token,uint64_t position,int32_t& picked,std::string& error){
        auto& s=*service_;auto& live=*s.owner_;
        if(private_sealed_||s.live_slot_||std::this_thread::get_id()!=s.controller_||s.active_||s.unhealthy_||
           !live.sessions.healthy()||!live.primary_verifier||!live.drive||live.drive->d.failed||
           token<0||token>=geometry_.n_vocab||position>=uint64_t(INT64_MAX)||s.operation_==UINT64_MAX){
            error="foreground service operation refused before submission";return false;}
        const auto& stages=live.sessions.stages();
        if(stages.size()!=runtime_.stage_count_){error="foreground service stage inventory changed";return false;}
        auto* root=live.primary_verifier.get();
        for(size_t i=0;i<stages.size();++i){auto* v=i?&live.stages[i-1]->ver:root;auto b=v->serial_routing_binding();
            auto* next=i+1<stages.size()?&live.stages[i]->ver:nullptr;
            if(b.session!=stages[i].session||b.geometry!=&geometry_||b.optimizer||b.next!=next||
               b.next_user!=(next?static_cast<const void*>(live.split_drive.get()):nullptr)){
                error="foreground actual service chain changed";return false;}}
        std::array<std::array<int32_t,2>,maximum_stages> previous{};
        for(size_t i=0;i<stages.size();++i)previous[i]={stages[i].session->ple_prev[0],stages[i].session->ple_prev[1]};
        s.active_=this;++s.operation_;
        try {
            for(const auto& stage:stages)if(stage.session->layer_lo<=1&&1<stage.session->layer_hi)stage.session->ple_token=token;
            core::WindowDeviceScope device(root->device());if(!device.ok())core::retirement_failed("foreground service device select");
            const auto callback=stages.size()>1?&::drive_pool_split:&::drive_pool_multi;
            void* context=stages.size()>1?static_cast<void*>(live.split_drive.get()):static_cast<void*>(live.drive.get());
            if(stages.size()==1)live.drive->d.plan=root->plan_sink();
            if(!root->run(1,&token,int64_t(position),callback,context,&picked,error)||live.drive->d.failed||root->nonfinite_row()>=0||
               !root->commit(1,error)||!root->wait_commit(error))core::retirement_failed("foreground service run/commit/wait");
            for(size_t i=0;i<stages.size();++i){auto& session=*stages[i].session;
                if(session.layer_lo<=1&&1<session.layer_hi&&(session.ple_prev[0]!=previous[i][1]||session.ple_prev[1]!=token||session.ple_token!=token))
                    core::retirement_failed("foreground actual PLE commit mismatch");
                if((i?&live.stages[i-1]->ver:root)->poll_snapshot_drain(error)!=core::Verifier::SnapshotDrain::drained)
                    core::retirement_failed("foreground service terminal drain");
            }
            int current=-1;if(cudaGetDevice(&current)!=cudaSuccess||current!=root->device()||!device.finish()||picked<0||picked>=geometry_.n_vocab)
                core::retirement_failed("foreground service device/output completion");
            s.active_=nullptr;return true;
        }catch(...){core::retirement_failed("foreground service exception retains both gangs");}
    }
    // Concrete controller-only one-way retirement. No supplied joined bool.
    // Success-path token members above finish their LOCAL HostGraphLaunch and
    // CPU callback scopes before releasing active_. No launch worker is retained
    // in this binding, and no raw submission API is exposed by it.
    bool retire_initialized(std::string& error){
        auto& r=runtime_;auto& service=*service_;
        if(std::this_thread::get_id()!=r.controller_||r.controller_!=service.controller_||
           private_sealed_||service.active_||callback_depth_||service.unhealthy_||
           (r.phase_!=RuntimePhase::ready&&r.phase_!=RuntimePhase::drained)){
            error="initialized retirement requires idle actual controller";return false;
        }
        // Keep self, service, model, geometry and setup credit until the final
        // native retirement AND all consumers below have finished destruction.
        auto self=r.dispatch_;
        if(self.get()!=this||!gate_||!r.model_||!service.geometry_owner_||
           service.geometry_owner_.get()!=&geometry_||!r.allocation_.healthy())
            core::retirement_failed("private initialized retirement owner mismatch");
        private_sealed_=true;r.phase_=RuntimePhase::retiring;service.active_=this;
        core::VerifierCompletionGate::Token retirement;
        if(!gate_->try_begin(retirement))
            core::retirement_failed("private retirement gate still busy or unhealthy");
        try {
            // No CPU producer can be newly admitted after seal. These queries
            // supplement the actual synchronous operation joins, not replace them.
            for(size_t i=0;i<r.stage_count_;++i){auto& stage=r.stages_[i];
                if(!stage.verifier||stage.verifier->poll_snapshot_drain(error)!=core::Verifier::SnapshotDrain::drained)
                    core::retirement_failed("private pre-retirement stage drain incomplete");
            }
            // Destruction marks the ACTUAL completion gate unhealthy before
            // registry-reader join and strict checked GPU/graph/buffer cleanup.
            // Every external stage stream and handoff remains alive throughout.
            for(size_t i=0;i<r.stage_count_;++i)r.stages_[i].verifier.reset();
            if(gate_->healthy_advisory())
                core::retirement_failed("private verifier destruction failed to invalidate gate");
            // Existing strict handle owner seals creation, checked-drains all
            // declared consumer devices, destroys events/streams and restores
            // device selection. Never closes shared source/pool/service handles.
            for(size_t i=0;i<r.stage_count_;++i){auto& stage=r.stages_[i];
                stage.handles.retire_after_producers_closed();stage.stream=nullptr;
            }
            if(core::window_buffer_quarantine_count()!=0)
                core::retirement_failed("private buffer retirement already uncertain");
            for(size_t i=0;i<r.stage_count_;++i)for(auto& buffer:r.stages_[i].retained_buffers){
                buffer={};
                if(core::window_buffer_quarantine_count()!=0)
                    core::retirement_failed("private buffer retirement quarantined");
            }
            for(auto& handoff:r.handoffs_){
                handoff={};
                if(core::window_buffer_quarantine_count()!=0)
                    core::retirement_failed("private handoff retirement quarantined");
            }
            // NativeRetirementAttempt seals the allocation, explicitly closes
            // its actual private PLE read_context CPU/IO, then all-stage consumer
            // barrier before any native arena/rope/PLE allocation is freed.
            // It retains shared table/model/context charges through that barrier.
            core::NativeRetirementAttempt attempt;
            if(!attempt.prepare(r.allocation_,std::span<const int>{},error)||!attempt.drain_and_commit(error))
                core::retirement_failed("private native all-stage terminal retirement failed");
            for(size_t i=0;i<r.stage_count_;++i){r.stages_[i].session=nullptr;r.stages_[i].session_owner.reset();}
            r.allocation_={};r.stage_count_=0;r.phase_=RuntimePhase::retired;
            // Historical Cuts remain valid historical observations only; they
            // retain real metadata/H through gate and weak controlblock owners.
            // Do not commit this retirement Token or resurrect admission.
            r.dispatch_.reset();service.active_=nullptr;
            // Local self/retirement keep binding/gate credit alive until after
            // all actual resource frees. Runtime setup_credit stays with its
            // still-live metadata owner; service/model/geometry stay retained.
            return true;
        }catch(...){core::retirement_failed("private terminal retirement exception retains process owners");}
    }
    ResidentRepairRuntime& runtime_; // runtime owns binding; nonowning avoids cycle
    std::shared_ptr<ResidentExpertService> service_;
    core::StateDigest original_residency_{};
    bool residency_bound_=false;
    const core::ModelGeometry& geometry_;
    std::shared_ptr<core::VerifierCompletionGate> gate_;
    ::Drive drive_{};
    ::SplitDrive split_{};
    unsigned callback_depth_=0;
    bool private_sealed_=false;
    std::optional<core::VerifierCompletionGate::Cut> last_completion_;
    uint64_t last_completed_operation_=0;
};
} // namespace strata::program::resident_repair_detail
