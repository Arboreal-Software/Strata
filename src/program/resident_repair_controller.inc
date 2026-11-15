// Experimental dedicated resident repair; integration and hardware qualification pending.
// Include AFTER ownership/private-dispatch and BEFORE resident-construction.
// Real dedicated serial controller; NO generic WindowManager registration or
// exact-only WindowTokenLedger stage guard is changed/claimed integrated.
#include "strata/core/window_repair_suffix.hpp"
#include "strata/core/serving_raw_contracts.hpp"
#include "strata/core/window_ple.hpp"
#include "strata/core/resident_copy_fault.hpp"
#include "strata/core/resident_deferred_fault.hpp"
#include "strata/core/resident_fault_precondition.hpp"
#include <chrono>
#include <algorithm>
#include <cstdio>
#include "serving_output_capacity.hpp"
#include "suffix_experiment_metrics.hpp"
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
// The retained-repair evidence value and the shared owner gates. They live in a CUDA-free header because the CPU
// gate (src/program/serving_repair_owners_test.cpp) must drive the SAME decision code the controller and the
// serving scope use; the controller below is the only producer of an evidence value.
#include "serving_repair_owners.hpp"
#include "serving_repair_age.hpp"
#include "strata/core/native_batch_group_completion.hpp"
#endif
namespace strata::program::resident_repair_detail {
struct ResidentConstructionInputs;
class ResidentRepairController final {
    friend class ::strata::program::SuffixExperimentDriver;
    // The serving counterpart of that driver: a trusted composition inside the serve entry that runs the same
    // qualified operation over an OWNED serving session.  It is not a general way to reach these internals.
    friend class ::strata::program::ServingSuffixRepairOperation;
    static FactoryResult construct(const std::shared_ptr<::DriverWindowRuntime>&,
        SuffixRawTransaction&,const ResidentConstructionInputs&,
        std::shared_ptr<ResidentRepairRuntime>&,std::string&);
    // Factory construction fragment also calls this actual bounded controller
    // metadata/history envelope before any controller allocation. Logical ABI
    // envelope, not exact malloc/RSS or graph runtime enforcement.
    static bool controller_host_requirement(uint64_t tokens,uint64_t& out)noexcept {
        // Two observer snapshots, each three <=64-device vectors. Logical
        // payload/object bound; allocator bookkeeping remains existing envelope.
        constexpr uint64_t snapshots=2*sizeof(core::WindowBudgetSnapshot)+6*64*sizeof(uint64_t);
        if(tokens>(UINT64_MAX-sizeof(ResidentRepairController)-(8ULL<<20)-snapshots)/128)return false;
        out=sizeof(ResidentRepairController)+(8ULL<<20)+snapshots+128*tokens;return true;
    }
    struct Published {
        core::SnapshotDescriptor metadata; // no tensor payload/devices
        std::shared_ptr<const core::NativeRepairSuffixData> data;
        uint64_t serial=1;
    };
    struct SpliceHold {
        std::shared_ptr<const void> cap; // retained after prepared image dies
        core::PreparedNativeRepairSuffix prepared;
    };
    struct Receipt {
        uint64_t operation=0,publication=0,cut=0;int32_t token=-1;
        // Fixed retained acknowledgement: retry same operation returns this
        // historical receipt without forwarding/emitting again.
    };
    using ReceiptHistory=::strata::program::ServingReceiptHistory<Receipt>;
    struct KnownReceipt {
        uint64_t operation=0,publication=0,cut=0;
        int32_t consumed=-1,next=-1;
    };
    // Retain setup admission through controller object+controlblock deallocation.
    template<class T> struct Alloc {
        using value_type=T;std::shared_ptr<const void> credit;
        explicit Alloc(std::shared_ptr<const void> c):credit(std::move(c)){}
        template<class U>Alloc(const Alloc<U>& a)noexcept:credit(a.credit){}
        template<class U>struct rebind{using other=Alloc<U>;};
        T* allocate(size_t n){return std::allocator<T>{}.allocate(n);}
        void deallocate(T* p,size_t n)noexcept{auto keep=credit;std::allocator<T>{}.deallocate(p,n);}
        template<class U>bool operator==(const Alloc<U>& a)const noexcept{return !credit.owner_before(a.credit)&&!a.credit.owner_before(credit);}
        template<class U>bool operator!=(const Alloc<U>& a)const noexcept{return !(*this==a);}
    };
    struct Delete {
        std::shared_ptr<const void> credit;
        void operator()(ResidentRepairController* p)const noexcept{delete p;}
    };
    ResidentRepairController(std::shared_ptr<ResidentRepairRuntime> runtime,std::shared_ptr<SuffixRawTransaction> live)
        :runtime_(std::move(runtime)),live_owner_(std::move(live)),live_(*live_owner_){}
    ~ResidentRepairController()noexcept {
        if(!retired_)core::retirement_failed("resident controller dropped before terminal private retirement");
    }
    struct Pause {
        ResidentRepairController& c;
        explicit Pause(ResidentRepairController& owner):c(owner){
            auto& s=*c.runtime_->service_;
            if(s.active_||std::this_thread::get_id()!=s.controller_)core::retirement_failed("resident service pause owner mismatch");
            s.active_=this;
        }
        ~Pause(){c.runtime_->service_->active_=nullptr;}
    };
    static core::SnapshotDescriptor metadata_only(const core::SnapshotDescriptor& s){
        core::SnapshotDescriptor m;m.key=s.key;m.cut=s.cut;m.native_layout=s.native_layout;
        m.accepted_tokens=s.accepted_tokens;m.pending_tokens=s.pending_tokens;m.approximation=s.approximation;
        m.speculation_disabled=true;m.steering_enabled=false;return m;
    }
    [[noreturn]] static void fatal(const char* reason){core::retirement_failed(reason);}
    uint64_t mint_cut(){if(!next_cut_||next_cut_==UINT64_MAX)fatal("resident cut identity exhausted");return next_cut_++;}
    void complete(SuffixRawTransaction& tx,const core::StateCut& cut){
        std::array<core::StateCut,maximum_stages> cuts{};for(size_t i=0;i<tx.verifiers_.size();++i)cuts[i]=cut;
        const auto completed=tx.complete_driver_work(true,cut,{cuts.data(),tx.verifiers_.size()});
        if(!completed){
#if defined(STRATA_SERVING_FAULT_QUALIFICATION_AVAILABLE)
            const auto* fault=core::resident_deferred_fault::current;
            if(fault&&fault->serving_.site()==core::resident_serving_fault::Site::private_replay&& &tx==private_.get())
                core::resident_serving_fault::event(fault->serving_,fault->serving_snapshot_,"raw_failure","complete_driver_work",int(completed.status),
                    int(tx.phase_==SuffixRawTransaction::Phase::poisoned),int(!tx.allocation_.healthy()),
                    int(tx.cut_.id==fault->serving_snapshot_.private_cut),int(bool(completed.retained_failure)));
#endif
            fatal("resident actual all-stage completion refused");
        }
    }
    bool check_idle(uint64_t expected)const noexcept {
        // `suspended_` is the retained-owner state (slice 2): the owner's publication is deliberately no longer
        // the live cut, so no live-requiring entry point may accept it. With the selector absent `suspended_` is
        // never set and this conjunction is exactly the reviewed one.
        return !busy_&&!retired_&&!cancel_requested_&&!suspended_&&published_&&expected==published_->serial&&
            live_.phase_==SuffixRawTransaction::Phase::drained&&published_->metadata.cut==live_.cut_&&runtime_->allocation_.healthy();
    }
    uint64_t private_forwarded()const noexcept { return private_?private_->cut_.forwarded:0; }
    // Public view of the private transaction's cut. `SuffixRawTransaction::cut_` is private to that type, so
    // callers outside the controller must not read it directly; this accessor is the controller's own view and
    // returns an empty cut when there is no private transaction. Copy, not reference: the caller must not be
    // able to alias the transaction's live cut through the accessor.
    core::StateCut private_cut()const noexcept { return private_?private_->cut_:core::StateCut{}; }
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
    // A slot-backed controller may borrow the retained model/expert service, never its main native session.
    // Authenticate the physical allocation against that exact service's published owned slot inventory.
    static bool checked_slot_live_binding(const std::shared_ptr<::DriverWindowRuntime>& service,
                                          const SuffixRawTransaction& tx,std::string& error){
        auto refuse=[&](const char* why){error=why;return false;};
        if(!service||!service->geometry_owner||!service->primary_verifier||!service->batch_slot_bindings||
           !service->batch_slot_sessions.published()||!tx.slot_occupancy_||!tx.slot_occupancy_->active()||
           tx.runtime_.get()!=service.get()||tx.runtime_.owner_before(service)||service.owner_before(tx.runtime_)||tx.geometry_identity_!=service->geometry_owner.get()||
           !tx.initialized_||tx.phase_!=SuffixRawTransaction::Phase::drained||
           tx.controller_!=std::this_thread::get_id()||!tx.allocation_.healthy()||
           service->batch_slot_bindings->geometry()!=service->geometry_owner.get())
            return refuse("slot factory requires actual service-owned drained RAW geometry and occupancy");
        const size_t slot=tx.slot_occupancy_->slot();
        if(slot>=2)return refuse("slot factory physical slot outside owned inventory");
        const auto handle=service->batch_slot_sessions.handle(slot);
        const auto* actual=service->batch_slot_sessions.allocation(handle);
        const core::NativeBatchSlotToken token{handle.incarnation,handle.generation,handle.slot};
        if(!actual||!actual->healthy()||token!=service->batch_slot_bindings->token(slot)||
           tx.allocation_.stages().size()!=actual->stages().size()||
           tx.verifiers_.size()!=actual->stages().size()||tx.slot_bindings_.size()!=actual->stages().size()||
           actual->stages().empty()||actual->stages().size()>maximum_stages||
           service->stages.size()+1!=actual->stages().size())
            return refuse("slot factory does not name the actual complete physical service slot");
        const auto& parent=actual->reservation();const auto& claimed=tx.allocation_.reservation();
        if(!parent||!claimed||!parent->canonical_owner(parent)||!claimed->canonical_owner(claimed)||
           parent->purpose()!=core::ReservationPurpose::private_work||parent.get()!=claimed.get()||
           parent.owner_before(claimed)||claimed.owner_before(parent))
            return refuse("slot factory parent is not the actual owned slot reservation");
        for(size_t stage=0;stage<actual->stages().size();++stage){
            const auto& expected=actual->stages()[stage];const auto& bound=tx.allocation_.stages()[stage];
            if(stage&&!service->stages[stage-1])return refuse("slot factory shared stage owner missing");
            auto* verifier=stage?&service->stages[stage-1]->ver:service->primary_verifier.get();
            const auto& observed=tx.slot_bindings_[stage];
            if(tx.verifiers_[stage]!=verifier||!expected.owner||!bound.owner||expected.session!=bound.session||
               expected.device!=bound.device||expected.owner.owner_before(bound.owner)||bound.owner.owner_before(expected.owner)||
               observed.token()!=token||observed.stage_index()!=stage||observed.geometry()!=service->geometry_owner.get()||
               !observed.matches_session(expected.session,expected.device)||!verifier->owned_batch_slot_matches(observed)||
               service->batch_slot_inspections[stage*2+slot].token()!=token||
               !verifier->owned_batch_slot_matches(service->batch_slot_inspections[stage*2+slot]))
                return refuse("slot factory actual verifier/session/generation ownership differs");
        }
        try{
            std::vector<core::NativeWindowStage> config;
            if(!tx.resolve(config,error))return false;
        }catch(...){return refuse("slot factory binding observation allocation refused");}
        return true;
    }
    static bool prepared_slot_runtime_matches(const ResidentRepairRuntime& runtime,const SuffixRawTransaction& live)noexcept {
        if(!runtime.service_)return false;
        const auto& service=*runtime.service_;
        if(bool(live.slot_occupancy_)!=bool(service.live_slot_))return false;
        if(!live.slot_occupancy_)return true; // original serial construction retains its existing checked path
        const auto& actual=live.allocation_.reservation();const auto& retained=service.live_slot_allocation_.reservation();
        return service.live_slot_==live.slot_occupancy_&&service.live_slot_valid()&&actual&&retained&&
            actual.get()==retained.get()&&!actual.owner_before(retained)&&!retained.owner_before(actual)&&
            service.count_==live.verifiers_.size();
    }
    static bool checked_slot_actor_binding(ServingRawSession* actor,const SuffixRawTransaction& live,std::string& error){
        if(!live.slot_occupancy_){
            if(actor){error="serial controller cannot claim a slot actor";return false;}
            return true;
        }
        if(!actor||actor->tx_.get()!=&live||actor->slot_occupancy_!=live.slot_occupancy_||
           actor->slot_controller_bound_||!actor->healthy()||!actor->request_open()){
            error="slot controller requires its genuine unclaimed open actor at creation";return false;
        }
        return true;
    }
#endif
    static bool retire_preconstructed(std::shared_ptr<ResidentRepairRuntime>& runtime,std::string& error){
        if(!runtime)return true;
        if(!runtime->dispatch_||runtime->phase_!=RuntimePhase::ready||
           !runtime->dispatch_->retire_initialized(error))return false;
        runtime.reset();return true;
    }
    static bool initialize_preconstructed_raw(const std::shared_ptr<::DriverWindowRuntime>& service,
        SuffixRawTransaction& live,const std::shared_ptr<ResidentRepairRuntime>& runtime,const core::WindowKey& key,
        std::unique_ptr<SuffixRawTransaction>& output,std::string& error){
        if(output||!runtime||runtime->phase_!=RuntimePhase::ready||!runtime->allocation_.healthy()||
           !runtime->service_||runtime->service_->owner_.get()!=service.get()){
            error="private RAW preinitialization requires genuine prepared prefix runtime";return false;
        }
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
        if((live.slot_occupancy_&&!checked_slot_live_binding(service,live,error))||!prepared_slot_runtime_matches(*runtime,live)){
            error="prepared private RAW factory does not match its actual live slot owner";return false;
        }
#endif
        std::vector<core::Verifier*> verifiers;verifiers.reserve(runtime->stage_count_);
        for(size_t i=0;i<runtime->stage_count_;++i)verifiers.push_back(runtime->stages_[i].verifier.get());
        auto tx=std::unique_ptr<SuffixRawTransaction>(new SuffixRawTransaction(*service->geometry_owner,key,
            runtime->allocation_,runtime,std::move(verifiers),live.budget_,live.max_tokens_,live.max_bytes_,live.host_limit_,false));
        const auto initialized=tx->initialize_resident_shared_host(live);
        if(!initialized){error=initialized.detail;return false;}
        output=std::move(tx);return true;
    }
    struct DataDelete {
        std::shared_ptr<const void> credit;
        void operator()(core::NativeRepairSuffixData* p)const noexcept{delete p;}
    };
    static bool admit_initial_roots(SuffixRawTransaction& live,std::shared_ptr<const void>& credit,std::string& error){
        const uint64_t bytes=sizeof(core::NativeRepairSuffixData)+3*core::RepairAncestry::retained_bytes()+65536;
        const auto admitted=live.admit_controller_host(bytes,credit);
        if(!admitted){error=admitted.detail;return false;}
        return true;
    }
    // Real first-import conversion, not authentication by ancestor digest. Only
    // concrete driver calls this with its actual exact donor + installed splice.
    static bool make_initial_roots(const std::shared_ptr<::DriverWindowRuntime>& service,SuffixRawTransaction& live,
        const core::BundleHandle& actual_exact_donor,const core::SnapshotDescriptor& current,
        std::shared_ptr<const core::NativeRepairSuffixData>& out,std::string& error,std::shared_ptr<const void> prepaid={},
        core::RepairMonotonicBirth first_unresolved_birth={}){
        if(out||!service||!service->remote_policy_incarnation||!actual_exact_donor||current.cut!=live.cut_||
           current.approximation.size()!=1||!live.identity_matches(current.key)||
           !live.identity_matches(actual_exact_donor.get()->key)||!actual_exact_donor.get()->approximation.empty()||
           actual_exact_donor.get()->cut.exact_frontier!=actual_exact_donor.get()->cut.forwarded){error="actual first suffix ancestry operands differ";return false;}
        // Birth is captured by the trusted causal import caller, never here during root reconstruction.
        // Default/legacy/restored inputs stay unknown. A foreign runtime cannot contribute a known birth.
        const auto& region=current.approximation[0];const auto& donor=*actual_exact_donor.get();
        if(region.begin!=current.cut.exact_frontier||region.end!=current.cut.forwarded||region.begin>=region.end||
           donor.cut.history!=region.ancestor||!region.first_unresolved_unix_ms){error="first suffix ancestor DATA binding differs";return false;}
        std::shared_ptr<const void> credit=std::move(prepaid);
        if(!credit&&!admit_initial_roots(live,credit,error))return false;
        core::RepairCausalRoot root;root.original_history=donor.cut.history;root.engine_generation=current.key.engine_generation;
        root.window_generation=current.key.window_generation;root.runtime_incarnation=service->remote_policy_incarnation;
        root.original_cut=donor.cut.id;root.original_begin=region.begin;root.original_count=region.end-region.begin;
        root.first_unresolved_unix_ms=region.first_unresolved_unix_ms;
        if(!bind_initial_serving_repair_birth(root,first_unresolved_birth,service->remote_policy_incarnation)){
            error="first suffix monotonic birth belongs to a different runtime";return false;
        }
        core::WindowHasher hash;const char domain[]="strata-private-first-suffix-root-data-v1";hash.update(domain,sizeof(domain));
        hash.update(root.original_history.data(),root.original_history.size());hash.update(&root.runtime_incarnation,sizeof(root.runtime_incarnation));
        hash.update(&root.original_cut,sizeof(root.original_cut));root.identity=hash.digest();
        core::RepairAncestry ancestry;
        if(core::make_repair_ancestry({&root,1},{64,core::RepairAncestry::retained_bytes()},ancestry)!=core::RepairAncestryStatus::ok){error="first suffix root DATA refused";return false;}
        core::RepairSpatialCoverage coverage;
        if(core::make_repair_spatial_coverage(ancestry,region.begin,region.end,coverage)!=core::RepairAncestryStatus::ok)return false;
        auto data=std::shared_ptr<core::NativeRepairSuffixData>(new core::NativeRepairSuffixData(coverage),DataDelete{credit},Alloc<core::NativeRepairSuffixData>{credit});
        data->region_roots.push_back({0,0});out=std::move(data);return true;
    }
    // Only the existing concrete driver friend creates this controller. Initial
    // metadata comes from its actual current complete cut; capture reads LIVE
    // bytes now, so an old frozen origin is never a current-state witness.
    static bool create(const std::shared_ptr<::DriverWindowRuntime>& service,std::shared_ptr<SuffixRawTransaction> live_owner,
        const ResidentConstructionInputs& inputs,const core::SnapshotDescriptor& actual_current,
        std::shared_ptr<const core::NativeRepairSuffixData> roots,const core::BundleHandle& exact_seed,
        std::shared_ptr<ResidentRepairController>& output,std::string& error,uint64_t minimum_next_cut=0,
        std::shared_ptr<ResidentRepairRuntime> prepared_runtime={},std::unique_ptr<SuffixRawTransaction> prepared_transaction={},ServingRawSession* slot_actor=nullptr){
        // This trusted serving composition supplies both genuine prebuilt owners only after its initial
        // transaction restore completed. Neither this witness nor its matching live cut comes from a caller bool.
        const bool checked_serving_restore=bool(prepared_runtime)&&bool(prepared_transaction);
        if(minimum_next_cut==UINT64_MAX){error="resident initial mint floor exhausted";return false;}
        if(!live_owner){error="genuine retained live RAW owner required";return false;}
        auto& live=*live_owner;
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
        if(!checked_slot_actor_binding(slot_actor,live,error)||(live.slot_occupancy_&&!checked_slot_live_binding(service,live,error)))return false;
#endif
        if(checked_serving_restore&&live.phase_!=SuffixRawTransaction::Phase::drained){
            error="serving prebuilt binding requires checked drained installed restore";return false;
        }
        if(output||!exact_seed||!roots||actual_current.cut.accepted!=actual_current.cut.forwarded+1||actual_current.pending_tokens.size()!=1||!roots->coverage.valid()||actual_current.cut!=live.cut_||
           !actual_current.devices.empty()||actual_current.approximation.size()!=1||
           roots->coverage.begin()!=actual_current.cut.exact_frontier||roots->coverage.end()!=actual_current.cut.forwarded||
           roots->region_roots.size()!=1||exact_seed.get()->cut.exact_frontier!=exact_seed.get()->cut.forwarded||
           !exact_seed.get()->approximation.empty()||exact_seed.get()->cut.accepted!=exact_seed.get()->cut.forwarded||
           !exact_seed.get()->pending_tokens.empty()||exact_seed.get()->cut.forwarded>actual_current.cut.exact_frontier){
            error="resident initial exact seed/current paired cut mismatch";return false;}
        const auto binding=roots->region_roots[0];const auto ancestry=roots->coverage.ancestry().roots();
        if(binding.region!=0||binding.root>=ancestry.size()||
           ancestry[binding.root].original_history!=actual_current.approximation[0].ancestor||
           ancestry[binding.root].first_unresolved_unix_ms!=actual_current.approximation[0].first_unresolved_unix_ms){error="current paired root DATA map differs";return false;}
        const auto& seed=*exact_seed.get();
        if(seed.key.revision!=actual_current.key.revision||!live.identity_matches(seed.key)||
           seed.accepted_tokens.size()>actual_current.accepted_tokens.size()||
           !std::equal(seed.accepted_tokens.begin(),seed.accepted_tokens.end(),actual_current.accepted_tokens.begin())){
            error="resident seed actual token prefix mismatch";return false;}
        std::shared_ptr<ResidentRepairRuntime> runtime;
        if(prepared_runtime){
            if(prepared_runtime->phase_!=RuntimePhase::ready||!prepared_runtime->allocation_.healthy()||
               !prepared_runtime->dispatch_||!prepared_runtime->service_||
               prepared_runtime->service_->owner_.get()!=service.get()||
               prepared_runtime->service_->geometry_owner_.get()!=service->geometry_owner.get()||
               prepared_runtime->stage_count_!=live.verifiers_.size()
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
               ||!prepared_slot_runtime_matches(*prepared_runtime,live)
#endif
               ){
                error="preconstructed resident runtime no longer matches exact-prefix service";return false;
            }
            runtime=std::move(prepared_runtime);
        }else if(!construct(service,live,inputs,runtime,error))return false;
        if(prepared_transaction&&(!prepared_transaction->initialized_||prepared_transaction->phase_!=SuffixRawTransaction::Phase::idle||
           prepared_transaction->runtime_.get()!=runtime.get()||!prepared_transaction->identity_matches(actual_current.key))){
            error="preinitialized private RAW owner no longer matches prepared runtime";return false;
        }
        // From this point any exception is failstop: a real initialized gang
        // exists and may not be unwound as an unbound host object.
        try {
            auto credit=runtime->setup_credit_;
            auto c=std::shared_ptr<ResidentRepairController>(new ResidentRepairController(runtime,std::move(live_owner)),Delete{credit},Alloc<ResidentRepairController>{credit});
            c->published_=std::make_shared<Published>();c->published_->metadata=actual_current;c->published_->data=std::move(roots);
            c->initial_accepted_=actual_current.cut.accepted;c->initial_output_watermark_=actual_current.cut.output_watermark;
            c->published_->metadata.accepted_tokens.reserve(size_t(live.max_tokens_));c->published_->metadata.pending_tokens.reserve(size_t(live.max_tokens_));
            c->next_cut_=std::max(actual_current.cut.id,seed.cut.id);if(c->next_cut_==UINT64_MAX)fatal("resident initial cut exhausted");++c->next_cut_;
            c->next_cut_=std::max(c->next_cut_,minimum_next_cut); // BEFORE initial zero/gate mint
            if(prepared_transaction)c->private_=std::move(prepared_transaction);
            else {
                std::vector<core::Verifier*> verifiers;verifiers.reserve(runtime->stage_count_);
                for(size_t i=0;i<runtime->stage_count_;++i)verifiers.push_back(runtime->stages_[i].verifier.get());
                c->private_.reset(new SuffixRawTransaction(*service->geometry_owner,actual_current.key,runtime->allocation_,runtime,
                    std::move(verifiers),live.budget_,live.max_tokens_,live.max_bytes_,live.host_limit_,false));
                if(!c->private_->initialize_resident_shared_host(live))fatal("resident shared-H RAW setup refused after private construction");
            }
            c->private_key_=seed.key;c->private_history_=seed.accepted_tokens;c->private_history_.reserve(size_t(live.max_tokens_));
            {
                Pause pause(*c);
                core::VerifierCompletionGate::Token initial_operation;
                if(!runtime->dispatch_->gate_->try_begin(initial_operation)||!c->private_->begin_driver_work())fatal("resident actual zero/gate begin refused");
                for(size_t i=0;i<runtime->stage_count_;++i){auto& stage=runtime->stages_[i];core::WindowDeviceScope device(stage.service.device);
                    if(!device.ok()||!stage.stream||!core::session_zero_checked(*stage.session,*service->geometry_owner,nullptr,stage.stream,error)||
                       cudaStreamSynchronize(stage.stream)!=cudaSuccess||!device.finish())fatal("resident actual private zero failed");}
                core::StateCut zero;zero.id=c->mint_cut();zero.history=core::window_history_digest(std::vector<int32_t>{});c->complete(*c->private_,zero);
                if(!c->private_->restore(exact_seed,seed.key))fatal("resident genuine exact seed restore refused");
                runtime->cut_=seed.cut;runtime->phase_=RuntimePhase::drained;
                core::VerifierCompletionGate::Cut initial_witness;
                if(!runtime->dispatch_->gate_->commit_after_checked_completion(initial_operation,initial_witness))fatal("resident actual restored exact gate completion refused");
                c->exact_completion_.emplace(seed.cut,std::move(initial_witness));
                // Scenario mode observes CURRENT with its real capture. Serving already holds the completed
                // all-stage restore witness for exactly actual_current.cut (checked above); capturing then
                // discarding the same image would add a needless 4P host peak after mutation.
                if(!checked_serving_restore){
                    core::BundleHandle now;if(!live.capture(actual_current,now))fatal("resident initial CURRENT donor capture failed");
                }
            }
            output=std::move(c);
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
            if(slot_actor)slot_actor->slot_controller_bound_=true; // genuine completed construction owns actor mode
#endif
            return true;
        }catch(...){fatal("resident controller initialization exception retains process");}
    }
    // The shared private advance: one actual all-layer exact token at a time, up to `target`. BOTH entry points
    // below (`repair_to` for a live-coherent owner, `retained_private_unit` for a suspended parked owner) run
    // exactly this loop, so a parked owner's private work is the same real device path rather than a copy of it.
    // The caller owns the precondition gate and must already hold `busy_`.
    void advance_private_to(uint64_t target){
        while(private_->cut_.forwarded<target){uint64_t pos=private_->cut_.forwarded;
            if(private_history_.size()!=pos||!std::equal(private_history_.begin(),private_history_.end(),published_->metadata.accepted_tokens.begin()))fatal("repair exact prefix changed");
            const int32_t token=published_->metadata.accepted_tokens[size_t(pos)];int32_t unused=-1;
            if(!private_->begin_driver_work())fatal("repair RAW begin refused");
            std::string unit_error;   // a refused private T1 below is terminal either way: see the fatal()
            if(!runtime_->dispatch_->run_private_one(token,pos,unused,unit_error))fatal("repair actual T1 refused after RAW begin");
            private_history_.push_back(token);core::StateCut cut;cut.id=mint_cut();cut.accepted=cut.forwarded=cut.exact_frontier=pos+1;
            cut.history=core::window_history_digest(private_history_);cut.output_watermark=published_->metadata.cut.output_watermark;
#if defined(STRATA_SERVING_FAULT_QUALIFICATION_AVAILABLE)
            if(serving_fault_.site()==core::resident_serving_fault::Site::private_replay&&receipt_count_&&!serving_fault_fired_){
                // Host history/cut/evidence are prepared first. The real T1/commit above completed, but its RAW
                // unit has not been certified. A real owned-stream trap must fail the original completion poll.
                Pause pause(*this);core::resident_deferred_fault::Scope fault;
                const auto snapshot=serving_fault_snapshot(cut.id);
                arm_serving_replay_fault(fault,pause,snapshot);
                complete(*private_,cut);
                fatal("serving replay fault unexpectedly certified a new cut");
            }
#endif
            complete(*private_,cut);runtime_->cut_=cut;private_key_=published_->metadata.key;
            auto& binding=*runtime_->dispatch_;
            if(!binding.last_completion_||!binding.last_completion_->committed()||binding.last_completed_operation_!=runtime_->operation_)fatal("private exact completion witness missing");
            exact_completion_.reset();exact_completion_.emplace(cut,std::move(*binding.last_completion_));binding.last_completion_.reset();
        }
    }
    // Full exact private checkpoint advances one actual all-layer token. It
    // takes accepted history already committed by foreground; no output effects.
    bool repair_to(uint64_t expected,uint64_t target,std::string& error){
        if(finished_||!check_idle(expected)||target>published_->metadata.cut.forwarded||target<private_->cut_.forwarded){error="repair target/selection stale";return false;}
        busy_=true;
        try { advance_private_to(target); busy_=false; return true; }
        catch(...){fatal("repair actual work exception");}
    }
    bool original_private_residency_restored()const noexcept {
        return runtime_&&runtime_->dispatch_&&runtime_->dispatch_->original_residency_restored();
    }
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
    // The transaction phase as the public contract spells it (serving_raw_contracts.hpp:26). Observation only:
    // this reads the transaction's own field, it never sets one.
    static core::serving_raw::Phase raw_phase(SuffixRawTransaction::Phase phase)noexcept {
        switch(phase){
            case SuffixRawTransaction::Phase::idle:return core::serving_raw::Phase::idle;
            case SuffixRawTransaction::Phase::running:return core::serving_raw::Phase::running;
            case SuffixRawTransaction::Phase::drained:return core::serving_raw::Phase::drained;
            case SuffixRawTransaction::Phase::capturing:return core::serving_raw::Phase::capturing;
            case SuffixRawTransaction::Phase::restoring:return core::serving_raw::Phase::restoring;
            case SuffixRawTransaction::Phase::poisoned:return core::serving_raw::Phase::poisoned;
        }
        return core::serving_raw::Phase::absent;
    }
    // THE trusted evidence operation. Everything the registry and the serving scope may conclude about this
    // controller comes from here: the facts are read from the controller's own members, and the maker
    // (serving_repair_owners.hpp) refuses a snapshot that contradicts itself and derives debt/coherence from the
    // cuts instead of accepting them as arguments. There is no caller-facing way to state any of these facts.
    RepairOwnerEvidence owner_evidence()const noexcept {
        RepairOwnerFacts facts;
        facts.live_phase=raw_phase(live_.phase_);
        facts.private_phase=private_?raw_phase(private_->phase_):core::serving_raw::Phase::absent;
        facts.busy=busy_;facts.retired=retired_;facts.cancel_requested=cancel_requested_;
        facts.finished=finished_;facts.suspended=suspended_;
        facts.live_allocation_healthy=live_.allocation_.healthy();
        facts.private_allocation_healthy=private_&&private_->allocation_.healthy();
        facts.service_idle=runtime_->service_&&runtime_->service_->active_==nullptr;
        facts.controller_thread=runtime_->controller_==std::this_thread::get_id()&&
            runtime_->service_&&runtime_->service_->controller_==std::this_thread::get_id();
        if(published_){facts.published_serial=published_->serial;facts.published_key=published_->metadata.key;
            facts.published_cut=published_->metadata.cut;}
        facts.live_cut=live_.cut_;
        if(private_)facts.private_cut=private_->cut_;
        if(exact_completion_){facts.exact_completion_cut=exact_completion_->cut.id;
            facts.exact_completion_committed=exact_completion_->receipt.committed();}
        if(const auto lease=runtime_->allocation_.reservation()){
            facts.private_charge=lease;facts.private_charge_purpose=lease->purpose();
        }
        RepairOwnerRefusal why=RepairOwnerRefusal::none;
        return make_repair_owner_evidence(facts,why);
    }
    // The private gang's own ledger charge (allocate_native_sessions carries ReservationPurpose::private_work,
    // src/core/window_allocator.cpp:317). The registry pins this so the ENTRY, not an anonymous scope, is what
    // keeps the parked owner's memory admitted; an explicit retire releases it. Refusal leaves `out` untouched.
    bool retained_charge(std::shared_ptr<const core::ReservationLease>& out,std::string& error)const {
        const auto lease=runtime_->allocation_.reservation();
        if(!lease){error="private gang has no ledger reservation to retain";return false;}
        if(lease->purpose()!=core::ReservationPurpose::private_work){
            error="retained owner charge must be a private_work reservation";return false;}
        out=lease;return true;
    }
    // THE CERTIFIED YIELD at a non-zero-debt coherent cut (BOUNDED-BREAKDOWN section 3, GAP-A blocker 6). It
    // publishes nothing and retires nothing: it certifies the private cut the controller actually holds a
    // committed all-stage witness for, captures that state as a real frozen image through the private
    // transaction's own drained-phase capture, and moves the controller into the SUSPENDED state from which only
    // bounded private work and an explicit retire are reachable. The published CURRENT state (which is what the
    // live session stands on) keeps its approximation regions: the debt is retained, never claimed closed.
    // `certified` is the private cut; `exact_checkpoint` is its all-stage frozen image, retained by the caller.
    bool yield_retained(uint64_t expected,core::StateCut& certified,core::BundleHandle& exact_checkpoint,
                        std::string& error){
        certified={};
        if(suspended_){error="controller is already suspended";return false;}
        RepairOwnerRefusal why=RepairOwnerRefusal::none;
        core::StateCut gate_cut{};
        // The gate below is the same shared predicate the CPU cell drives; it re-reads nothing but this
        // controller's own facts, so a caller cannot talk its way past any conjunct.
        if(!retained_yield_allowed(owner_evidence(),why,gate_cut)){
            error=std::string("retained yield refused: ")+repair_owner_refusal_name(why);return false;}
        if(expected!=published_->serial){error="retained yield selection stale";return false;}
        busy_=true;
        try {
            // The capture is the certificate: `capture_exact` requires the real committed completion witness for
            // this exact private cut (resident_repair_controller.inc:625-629) and the private transaction's own
            // drained phase, and it charges the ledger through the capture's own private_work reservation.
            core::BundleHandle captured;
            if(!capture_exact(captured,error)){busy_=false;return false;}
            if(!captured||!captured.get()||!(captured.get()->cut==private_->cut_)){
                busy_=false;error="retained yield captured no image of its own private cut";return false;}
            exact_checkpoint=std::move(captured);
            certified=private_->cut_;
            // The versioned owner record starts here, from the controller's own cuts. Every later private unit
            // bumps it, which is what makes a result prepared before that unit stale (slice 5).
            retained_record_={};
            retained_record_.identity=published_->metadata.key;
            retained_record_.published_serial=published_->serial;
            retained_record_.record_version=1;
            retained_record_.published_cut=published_->metadata.cut;
            retained_record_.certified_cut=private_->cut_;
            retained_record_.token_debt=published_->metadata.cut.forwarded-private_->cut_.forwarded;
            retained_record_.versions=make_retained_versions(published_->metadata.key,private_->cut_,0);
            retained_record_.capacity_reserved=false;   // the registry's own copy carries the admission
            retained_record_.resources_pinned=false;
            retained_result_={};
            suspended_=true;   // one-way until resume_retained() or activate_retained(); see the gates at check_idle
            busy_=false;
            return true;
        }catch(...){fatal("retained yield exception retains both gangs");}
    }
    // ONE bounded private unit for a SUSPENDED (parked) owner, through the same `advance_private_to` loop a
    // live-coherent owner uses. The gate is the retained-owner one, not `check_idle`: live coherence is exactly
    // what a parked owner no longer has, and every other conjunct is stricter, not looser - the service must be
    // idle, this thread must be the controller thread, both allocations must be healthy, both gangs must be at a
    // drained cut, and the target is capped by the CALLER's selector-named number (no compiled default here).
    // `advanced` reports the tokens actually replayed. This never touches the live transaction.
    bool retained_private_unit(uint64_t expected,uint64_t cap_tokens,uint64_t& advanced,std::string& error){
        advanced=0;
        RepairOwnerRefusal why=RepairOwnerRefusal::none;
        uint64_t target=0;
        const RepairOwnerEvidence evidence=owner_evidence();
        if(!retained_private_unit_allowed(evidence,cap_tokens,target,why)){
            error=std::string("retained private unit refused: ")+repair_owner_refusal_name(why);return false;}
        if(expected!=published_->serial){error="retained private unit selection stale";return false;}
        const uint64_t before=private_->cut_.forwarded;
        busy_=true;
        try {
            advance_private_to(target);
            // The chain moved: the record version moves with it and any result prepared against the previous
            // certified cut is dropped, exactly as the registry entry does.
            if(!retained_record_.identity.window_id.empty()){
                retained_record_.certified_cut=private_->cut_;
                retained_record_.token_debt=published_->metadata.cut.forwarded-private_->cut_.forwarded;
                retained_record_.versions=make_retained_versions(retained_record_.identity,private_->cut_,
                                                                 retained_record_.versions.residency_generation);
                retained_result_={};
                ++retained_record_.record_version;}
            busy_=false;
            advanced=private_->cut_.forwarded-before;
            return advanced>0;
        }catch(...){fatal("retained private unit exception retains both gangs");}
    }
    // The only way out of the suspended state: full live coherence must be genuine again (the same conjunction
    // every live-requiring operation already demands, `check_idle`), including a matching serial. Nothing here
    // reconciles, rewrites or re-attaches a published cut, which is why a parked owner whose live cut moved on
    // stays suspended and can only be retired deliberately (GAP-A blocker 7: refused, never forced).
    // The only way out of the suspended state: full live coherence must be genuine again (the same conjunction
    // every live-requiring operation already demands, `check_idle`), including a matching serial. Nothing here
    // reconciles, rewrites or re-attaches a published cut, which is why a parked owner whose live cut moved on
    // stays suspended and can only be retired deliberately (GAP-A blocker 7: refused, never forced).
    // ---- the versioned retained RESULT: prepare, publish-if-current, activate (slice 5, A->B->A) --------
    // The path GAP-A blocker 7 left missing. It does NOT touch `resume_retained` above: that comparison stays, and
    // activation is what makes it genuinely true again for the right conversation.
    // (a) The controller states a retained RESULT about itself: its certified private cut, the committed all-stage
    // witness that covers exactly that cut, the accepted sequence length, the one pending token the activated state
    // consumes, the output watermark, and the chain's version stamp (content digest plus identity epochs).
    // `frozen_image` is the all-stage image the caller retains; the controller checks it against its OWN private
    // cut rather than accepting the caller's word, and the maker refuses any set of facts that contradicts itself.
    PreparedRetainedResult prepare_retained_result(uint64_t record_version,const core::BundleHandle& frozen_image,
                                                   RepairOwnerRefusal& why)const noexcept {
        if(!published_||!private_||
           (!retained_record_.identity.window_id.empty()&&retained_record_.identity!=published_->metadata.key)){
            why=RepairOwnerRefusal::refused_identity_mismatch;return {};}
        PreparedRetainedResultFacts facts;
        facts.identity=published_->metadata.key;
        facts.published_serial=published_->serial;
        facts.record_version=record_version;
        facts.published_cut=published_->metadata.cut;
        facts.certified_cut=private_->cut_;
        facts.accepted_sequence=private_->cut_.accepted;
        facts.pending_token=published_->metadata.pending_tokens.size()==1?published_->metadata.pending_tokens.front():-1;
        facts.output_watermark=published_->metadata.cut.output_watermark;
        facts.witness_committed=bool(exact_completion_)&&exact_completion_->receipt.committed()&&
            exact_completion_->cut.id==private_->cut_.id;
        facts.holds_frozen_image=bool(frozen_image)&&frozen_image.get()&&frozen_image.get()->cut==private_->cut_;
        return make_prepared_retained_result(facts,why);
    }
    // (b) Publish-if-current into the OWNER RECORD only. Nothing here mutates the live transaction, the private
    // gang's cut, or the published CURRENT state: it records the controller's own certified result on the
    // controller's mirror of the owner record, behind the versioned current check and the same shared gate the
    // registry runs. A refusal names its rule and changes nothing, so the owner stays parked and resumable.
    bool publish_retained_if_current(uint64_t expected_record_version,const core::BundleHandle& frozen_image,
                                     PreparedRetainedResult& out,std::string& error){
        out={};
        if(!suspended_){error="only a parked (suspended) owner has a retained-result record";return false;}
        if(busy_){error="the controller is busy";return false;}
        if(retired_||cancel_requested_||finished_){error="the owner is retired, cancelled or finished";return false;}
        RepairOwnerRefusal why=RepairOwnerRefusal::none;
        const PreparedRetainedResult result=prepare_retained_result(expected_record_version,frozen_image,why);
        if(!result.produced()){error=std::string("retained result refused: ")+repair_owner_refusal_name(why);return false;}
        if(!retained_result_publishable(result,owner_evidence(),retained_record_,why)){
            error=std::string("retained publication refused: ")+repair_owner_refusal_name(why);return false;}
        retained_result_=result;
        return true;
    }
    // (c) Activate: make this owner's published state the LIVE session state, at a drained all-stage boundary.
    // Order is deliberate and every step is gated before it mutates: (1) the shared activation gate over the
    // controller's own evidence and the live session's facts; (2) capture the live session's PRIOR conversation
    // state through the live transaction, charged by the entry's `foreground_return` pin; (3) restore the owner's
    // certified image and adopt it as the controller's publication; (4) verify the installed cut; (5) only then
    // `resume_retained`, whose published-cut==live-cut conjunction is now genuinely true. Any failure at (3) or (4)
    // restores the captured prior state and verifies it field by field, and the owner stays suspended, parked and
    // resumable. A restore that cannot be verified is the failstop the RAW contract already requires, never a
    // silently continued request.
    bool activate_retained(uint64_t expected_record_version,const core::BundleHandle& frozen_image,
                           const LiveSessionFacts& session,const ConversationStateFacts& prior,
                           const core::SnapshotDescriptor& prior_metadata,
                           ConversationStateFacts& restored_out,std::string& error,
                           const RetainedActivationAuthority& authority=RetainedActivationAuthority()){
        restored_out={};
        if(!suspended_){error="the owner is not parked";return false;}
        if(busy_){error="the controller is busy";return false;}
        if(retired_||cancel_requested_||finished_){error="the owner is retired, cancelled or finished";return false;}
        if(!published_||!private_||published_->metadata.pending_tokens.size()!=1){
            error="the owner has no publication with its own pending token";return false;}
        RepairOwnerRefusal why=RepairOwnerRefusal::none;
        const PreparedRetainedResult result=prepare_retained_result(expected_record_version,frozen_image,why);
        if(!result.produced()){error=std::string("activation refused: ")+repair_owner_refusal_name(why);return false;}
        // Admission belongs to the registry. Check its immutable issued record against this controller's
        // own version/cuts before using its reservation and residency facts in the shared activation gate.
        if(!authority.issued()){
            error="activation refused: registry activation authority absent";return false;}
        const auto& admitted=authority.record();
        if(admitted.identity!=retained_record_.identity||
           admitted.published_serial!=retained_record_.published_serial||
           admitted.record_version!=retained_record_.record_version||
           admitted.record_version!=expected_record_version||
           admitted.published_cut!=retained_record_.published_cut||
           admitted.certified_cut!=retained_record_.certified_cut||
           admitted.token_debt!=retained_record_.token_debt||
           !same_chain_versions(admitted.versions,retained_record_.versions)){
            error="activation refused: registry activation authority does not match controller record";return false;}
        RetainedActivationFacts activation;
        activation.record_version=expected_record_version;
        activation.live_all_stage_drained=live_.phase_==SuffixRawTransaction::Phase::drained&&
            live_.allocation_.healthy();
        activation.request_fence_open=true;      // the caller holds the returning conversation's open fence
        const bool prior_coherent=prior.cut.id!=0&&core::serving_raw::digest_present(prior.cut.history)&&
            live_.identity_matches(prior.key);
        activation.prior_state_captured=prior_coherent&&prior.session_cut==live_.cut_;
        // Slice 6 (reviewed contract extension, not a relaxation): the A->B->A case where the serving scope has
        // ALREADY restored this owner's publication image onto the live transaction, so the prior conversation's
        // state is a retained image captured at an earlier session cut. The controller asserts it only when the
        // live cut is exactly this owner's recorded publication, on both sides of the comparison: a session that
        // was not restored onto the record's publication is not a restored session, and stays refused.
        activation.prior_state_retained_image=prior_coherent&&prior.session_cut!=live_.cut_&&
            live_.cut_==published_->metadata.cut&&live_.cut_==retained_record_.published_cut;
        activation.prior_session_cut=prior.session_cut;
        if(!retained_activation_allowed(result,owner_evidence(),admitted,session,activation,why)){
            error=std::string("activation refused: ")+repair_owner_refusal_name(why);return false;}
        // The supported boundary, checked BEFORE anything mutates: the live transaction has been brought to this
        // owner's own publication cut (the serving scope restores the publication image and completes the
        // all-stage drain at that cut, generate.cpp), the chain has caught it, and the publication still carries
        // this conversation's own pending token and output watermark. Anything else is refused without touching
        // the session, and the owner stays parked - a foreign or moved live state is never overwritten.
        // Slice 6 (reviewed contract change to this boundary): the exactness conjunct applies to the CERTIFIED
        // PRIVATE CHECKPOINT R, whose committed all-stage witness `prepare_retained_result` already speaks for,
        // and NOT to the publication D. D is what gets installed and it stays exactly as published - suffix reuse
        // is approximate by construction, so demanding `D.exact_frontier == D.forwarded` here would make
        // activation unreachable for the feature instead of making it safe. What makes the installation safe is
        // (i) `live_.cut_ == published_->metadata.cut`, a field-by-field StateCut comparison that refuses an
        // installed image relabeling D's exact frontier, history digest, accepted count or watermark, and (ii) the
        // descriptor check after the resume below, which refuses a publication whose approximation regions changed.
        if(!(live_.cut_==published_->metadata.cut)||
           private_->cut_.forwarded!=published_->metadata.cut.forwarded||
           private_->cut_.exact_frontier!=private_->cut_.forwarded||
           (published_->metadata.approximation.empty()&&
            published_->metadata.cut.exact_frontier!=published_->metadata.cut.forwarded)||
           published_->metadata.cut.exact_frontier!=result.published_cut().exact_frontier||
           published_->metadata.cut.accepted!=result.published_cut().accepted||
           !(published_->metadata.cut.history==result.published_cut().history)||
           published_->metadata.cut.output_watermark!=result.output_watermark()||
           published_->metadata.pending_tokens.front()!=result.pending_token()||
           !live_.identity_matches(published_->metadata.key)||
           live_.phase_!=SuffixRawTransaction::Phase::drained||!live_.allocation_.healthy()||
           private_->phase_!=SuffixRawTransaction::Phase::drained){
            error="activation refused before mutation: the live session is not at this owner's drained all-stage "
                  "publication boundary";
            return false;}
        busy_=true;
        // The publication the activation installs, recorded field by field BEFORE anything mutates, so the check
        // after the resume can say what the activation did not do: it never relabels the publication. An
        // approximate D stays approximate, with its approximation regions, its own exact frontier, its accepted
        // run and its pending token carried through; a publication that came back exact, shortened or re-epoch'd
        // is the failstop this contract already is, never a silently upgraded certificate (slice 6).
        const uint64_t installed_serial=published_->serial;
        const core::SnapshotDescriptor installed=published_->metadata;
        try {
            // Capture owns the bounded busy/service scope. Public resume requires that scope to have ended;
            // its idle guard stays unchanged and cannot be bypassed by activation's own bookkeeping.
            core::BundleHandle prior_bundle;
            {
                Pause pause(*this);
                if(!live_.capture(prior_metadata,prior_bundle)){
                    busy_=false;error="activation could not capture the live session's prior conversation state";
                    return false;}
            }
            busy_=false;
            std::string resume_error;
            if(!resume_retained(published_->serial,resume_error)){
                // A refused resume has not selected another publication. Restore the captured operand under a
                // fresh bounded service scope; the outer serving adapter separately owns displaced B's image.
                busy_=true;
                {
                    Pause pause(*this);
                    const auto back=live_.restore(prior_bundle,prior.key);
                    restored_out=prior;
                    if(!back||live_.cut_!=prior.cut)
                        fatal("a refused resume left the live session unrecoverable");
                }
                busy_=false;
                error="activation refused at the coherence conjunction: "+resume_error;return false;}
            if(published_.get()!=nullptr&&
               (published_->serial!=installed_serial||!(published_->metadata.key==installed.key)||
                !(published_->metadata.cut==installed.cut)||
                published_->metadata.approximation.size()!=installed.approximation.size()||
                published_->metadata.accepted_tokens.size()!=installed.accepted_tokens.size()||
                published_->metadata.pending_tokens!=installed.pending_tokens)){
                fatal("an activation changed the publication it installed");}
            retained_result_={};       // consumed: the record has been landed, it is not pending any more
            retained_record_={};       // the owner is live again; the parked record is gone
            restored_out=prior;
            busy_=false;
            return true;
        }catch(...){fatal("retained activation exception retains both gangs");}
    }
    // The old `resume_retained`, kept exactly as reviewed (its published-cut == live-cut refusal performs no
    // reconciliation, which is correct: reconciliation is `activate_retained`'s job).
    bool resume_retained(uint64_t expected,std::string& error){
        if(!suspended_){error="controller is not suspended";return false;}
        if(!published_||expected!=published_->serial){error="resume selection stale";return false;}
        if(busy_||retired_||cancel_requested_||finished_||!private_||!runtime_->allocation_.healthy()){
            error="resume requires an idle healthy unretired owner";return false;}
        if(private_->phase_!=SuffixRawTransaction::Phase::drained||
           live_.phase_!=SuffixRawTransaction::Phase::drained||published_->metadata.cut!=live_.cut_){
            error="resume requires the live session to still stand on this owner's publication";return false;}
        suspended_=false;
        return true;
    }
#endif
    // Explicit trusted approximate continuation, separate from exact-only
    // generic ledger staging. ALL global roots and birth anchors are retained.
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
    // A cohort has one actual native group dispatch. Preparation retains all fallible publication/ancestry/hash
    // state before either lane begins work. The controller stays bound to its original slot RAW owner.
    struct CohortDecodeInput {
        int32_t token()const noexcept{return token_;}
        uint64_t position()const noexcept{return position_;}
        bool is_known()const noexcept{return known_;}
    private:
        friend class ResidentRepairController;
        int32_t token_=-1;uint64_t position_=0;bool known_=false;
    };
    enum class CohortKind {decode,known};
    struct CohortPending {
        std::shared_ptr<Published> before,next;
        ServingRawSession* slot=nullptr;
        CohortDecodeInput input;
        uint64_t operation=0;CohortKind kind=CohortKind::decode;int32_t next_known=-1;
        core::WindowHasher history;
        std::vector<core::StateCut> stage_cuts;
    };
    const CohortDecodeInput* prepare_cohort_decode(uint64_t expected,uint64_t operation,
                                                  ServingRawSession& slot,std::string& error){
        return prepare_cohort_operation(expected,operation,slot,CohortKind::decode,-1,error);
    }
    const CohortDecodeInput* prepare_cohort_known(uint64_t expected,uint64_t operation,
            ServingRawSession& slot,int32_t next_known,std::string& error){
        return prepare_cohort_operation(expected,operation,slot,CohortKind::known,next_known,error);
    }
    const CohortDecodeInput* prepare_cohort_operation(uint64_t expected,uint64_t operation,
            ServingRawSession& slot,CohortKind kind,int32_t next_known,std::string& error){
        if(cohort_pending_||!live_owner_||live_owner_.get()!=&live_||slot.tx_.get()!=&live_||
           !slot.slot_occupancy_||!slot.healthy()||!slot.request_open()||!check_idle(expected)||finished_||
           !operation||operation<=std::max(last_.operation,last_known_.operation)||
           (kind==CohortKind::decode&&receipt_count_==receipts_.size())||
           (kind==CohortKind::known&&(next_known<0||next_known>=live_.geometry_.n_vocab||known_count_==UINT64_MAX))||
           foreground_count_==UINT64_MAX||next_cut_==UINT64_MAX||
           published_->metadata.cut.accepted!=published_->metadata.cut.forwarded+1||
           published_->metadata.accepted_tokens.size()!=published_->metadata.cut.accepted||
           slot.slot_tokens_!=published_->metadata.accepted_tokens||
           published_->metadata.pending_tokens.size()!=1||
           published_->metadata.accepted_tokens.size()>=live_.max_tokens_||
           published_->metadata.key.append_epoch==UINT64_MAX||(kind==CohortKind::decode&&published_->metadata.cut.output_watermark==UINT64_MAX)){
            error="cohort preparation requires its own idle bound slot, fresh operation and admitted history capacity";
            return nullptr;
        }
        try{
            auto pending=std::make_unique<CohortPending>();
            pending->before=published_;pending->slot=&slot;pending->operation=operation;pending->kind=kind;pending->next_known=next_known;
            pending->next=std::make_shared<Published>();pending->next->metadata=published_->metadata;
            pending->next->serial=published_->serial;
            auto& next=pending->next->metadata;const uint64_t position=next.cut.forwarded;
            const int32_t token=next.accepted_tokens[size_t(position)];
            if(next.pending_tokens.front()!=token){error="cohort input differs from controller accepted pending";return nullptr;}
            pending->input.token_=token;pending->input.position_=position;pending->input.known_=kind==CohortKind::known;
            next.accepted_tokens.reserve(size_t(live_.max_tokens_));next.pending_tokens.reserve(size_t(live_.max_tokens_));
            ++next.key.append_epoch;++next.cut.forwarded;++next.cut.accepted;
            if(kind==CohortKind::decode)++next.cut.output_watermark;
            if(next.approximation.empty()){next.cut.exact_frontier=next.cut.forwarded;pending->next->data.reset();}
            else{
                next.approximation.back().end=next.cut.forwarded;
                if(!published_->data){error="cohort approximate publication lacks inherited ancestry";return nullptr;}
                core::RepairSpatialCoverage coverage;
                if(core::make_repair_spatial_coverage(published_->data->coverage.ancestry(),next.cut.exact_frontier,
                        next.cut.forwarded,coverage)!=core::RepairAncestryStatus::ok){
                    error="cohort inherited ancestry coverage refused before dispatch";return nullptr;
                }
                auto data=std::make_shared<core::NativeRepairSuffixData>(coverage);
                data->region_roots=published_->data->region_roots;pending->next->data=std::move(data);
            }
            // Same domain/count/LE-int32 sequence as window_history_digest, with the actual sample appended only
            // after authentic completion. WindowHasher allocates its state here, never after native dispatch.
            constexpr char domain[]="strata-window-history-v1";
            pending->history.update(domain,sizeof(domain));
            std::array<uint8_t,8> count{};
            for(unsigned j=0;j<8;++j)count[j]=uint8_t(next.cut.accepted>>(8*j));
            pending->history.update(count.data(),count.size());
            for(int32_t accepted:next.accepted_tokens){
                std::array<uint8_t,4> bytes{};
                for(unsigned j=0;j<4;++j)bytes[j]=uint8_t(uint32_t(accepted)>>(8*j));
                pending->history.update(bytes.data(),bytes.size());
            }
            if(kind==CohortKind::known){
                next.accepted_tokens.push_back(next_known);next.pending_tokens.assign(1,next_known);
                std::array<uint8_t,4> bytes{};
                for(unsigned j=0;j<4;++j)bytes[j]=uint8_t(uint32_t(next_known)>>(8*j));
                pending->history.update(bytes.data(),bytes.size());next.cut.history=pending->history.digest();
            }
            pending->stage_cuts.resize(live_.allocation_.stages().size());
            if(pending->stage_cuts.empty()){error="cohort bound slot has no complete stage gang";return nullptr;}
            next.cut.id=mint_cut(); // a canceled preparation may burn an ID; it never reuses one
            if(kind==CohortKind::known)for(auto& stage:pending->stage_cuts)stage=next.cut;
            cohort_pending_=std::move(pending);slot.slot_controller_bound_=true;busy_=true;
            return &cohort_pending_->input;
        }catch(...){error="cohort metadata/ancestry/hash preparation refused before native dispatch";return nullptr;}
    }
    bool cancel_cohort_preparation(ServingRawSession& slot,std::string& error){
        if(!cohort_pending_||cohort_pending_->slot!=&slot||slot.tx_.get()!=&live_||
           live_.phase_!=SuffixRawTransaction::Phase::drained||published_.get()!=cohort_pending_->before.get()||
           live_.cut_!=cohort_pending_->before->metadata.cut){
            error="cohort cancellation is allowed only at its untouched drained pre-dispatch cut";return false;
        }
        cohort_pending_.reset();busy_=false;return true;
    }
    const Receipt* complete_cohort_decode(ServingRawSession& slot,const core::NativeBatchGroupCompletion& receipt,
                                          FILE* output,std::string& error){
        if(!cohort_pending_||cohort_pending_->kind!=CohortKind::decode||!busy_||!output||cohort_pending_->slot!=&slot||slot.tx_.get()!=&live_||
           live_owner_.get()!=&live_||published_.get()!=cohort_pending_->before.get()||
           published_->serial!=cohort_pending_->before->serial){
            error="cohort completion has no matching internally prepared controller/slot owner";return nullptr;
        }
        auto& pending=*cohort_pending_;
        ServingRawSession::CohortReceiptView actual;
        if(!slot.inspect_cohort_receipt(receipt,pending.before->metadata,pending.input.token_,actual,error))return nullptr;
        // The private RAW peer validated the actual group sample, physical occupancy and every stage completion.
        // No caller-provided cut, token pick, completed flag or serial can enter the publication below.
        auto& next=pending.next->metadata;
        next.accepted_tokens.push_back(actual.sample);next.pending_tokens.assign(1,actual.sample);
        std::array<uint8_t,4> bytes{};
        for(unsigned j=0;j<4;++j)bytes[j]=uint8_t(uint32_t(actual.sample)>>(8*j));
        pending.history.update(bytes.data(),bytes.size());next.cut.history=pending.history.digest();
        for(auto& stage:pending.stage_cuts)stage=next.cut;
        // Past this point a refused commit cannot be retried with an already-finished hasher or silently unwind
        // over actual group work. Preserve the owner/ticket on uncertainty; the serving composition failstops.
        if(!slot.commit_cohort_receipt(actual,next,pending.stage_cuts,output,error))
            fatal("cohort RAW commit refused after authentic group completion");
        if(live_.cut_!=next.cut)fatal("cohort controller/slot published different all-stage cuts");
        const uint64_t operation=pending.operation;const int32_t sample=actual.sample;
        published_=std::move(pending.next);++foreground_count_;
        last_={operation,published_->serial,published_->metadata.cut.id,sample};
        receipts_[receipt_count_++]=last_;
        cohort_pending_.reset();busy_=false;
        return &receipts_[receipt_count_-1];
    }
    const KnownReceipt* complete_cohort_known(ServingRawSession& slot,const core::NativeBatchGroupCompletion& receipt,
                                               std::string& error){
        if(!cohort_pending_||cohort_pending_->kind!=CohortKind::known||!busy_||cohort_pending_->slot!=&slot||
           slot.tx_.get()!=&live_||live_owner_.get()!=&live_||published_.get()!=cohort_pending_->before.get()||
           published_->serial!=cohort_pending_->before->serial){
            error="known cohort completion has no matching internally prepared owner";return nullptr;
        }
        auto& pending=*cohort_pending_;
        ServingRawSession::CohortReceiptView actual;
        if(!slot.inspect_cohort_receipt(receipt,pending.before->metadata,pending.input.token_,actual,error))return nullptr;
        // The native pick is deliberately ignored: the next pending token was authenticated by the request
        // composition and fully hashed before dispatch. A known prompt row is never an emitted receipt.
        if(!slot.commit_cohort_known_receipt(actual,pending.next->metadata,pending.stage_cuts,error))
            fatal("known cohort RAW commit refused after authentic group completion");
        if(live_.cut_!=pending.next->metadata.cut)fatal("known cohort controller/slot published different cuts");
        const uint64_t operation=pending.operation;const int32_t consumed=pending.input.token_,next=pending.next_known;
        published_=std::move(pending.next);++foreground_count_;++known_count_;
        last_known_={operation,published_->serial,published_->metadata.cut.id,consumed,next};
        cohort_pending_.reset();busy_=false;return &last_known_;
    }
#endif

    const Receipt* foreground(uint64_t expected,uint64_t operation,std::string& error){
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
        if(live_.slot_occupancy_){error="owned slot foreground requires genuine cohort dispatch";return nullptr;}
#endif

        for(size_t i=0;i<receipt_count_;++i)if(operation&&receipts_[i].operation==operation)return &receipts_[i]; // historical retry only
        if(!check_idle(expected)||!operation||operation<=std::max(last_.operation,last_known_.operation)||receipt_count_==receipts_.size()||
           published_->metadata.cut.forwarded>=published_->metadata.accepted_tokens.size()||
           published_->metadata.cut.accepted!=published_->metadata.cut.forwarded+1||
           published_->metadata.accepted_tokens.size()>=live_.max_tokens_||
           published_->metadata.key.append_epoch==UINT64_MAX||published_->metadata.cut.output_watermark==UINT64_MAX){error="foreground operation/history bound";return nullptr;}
        busy_=true;
        try {
            auto next=std::make_shared<Published>();next->metadata=published_->metadata;next->serial=published_->serial;
            next->metadata.accepted_tokens.reserve(size_t(live_.max_tokens_));next->metadata.pending_tokens.reserve(size_t(live_.max_tokens_));
            const uint64_t pos=next->metadata.cut.forwarded;const int32_t token=next->metadata.accepted_tokens[size_t(pos)];int32_t picked=-1;
            if(!live_.begin_driver_work())fatal("foreground RAW begin refused");
            if(!runtime_->dispatch_->run_foreground_one(token,pos,picked,error))fatal("foreground actual T1 refused after RAW begin");
            // Milestone maintains exactly one pending sampled token. Refuse
            // other pending shapes at create; append AFTER real greedy output.
            next->metadata.accepted_tokens.push_back(picked);next->metadata.pending_tokens.assign(1,picked);
            ++next->metadata.key.append_epoch;auto& cut=next->metadata.cut;cut.id=mint_cut();++cut.forwarded;cut.accepted=next->metadata.accepted_tokens.size();++cut.output_watermark;
            cut.history=core::window_history_digest(next->metadata.accepted_tokens); // exact frontier unchanged
            if(next->metadata.approximation.empty())cut.exact_frontier=cut.forwarded;
            else next->metadata.approximation.back().end=cut.forwarded;
            if(!next->metadata.approximation.empty()){
                if(!published_->data)fatal("approximate foreground lacks paired full ancestry");
                core::RepairSpatialCoverage coverage;
                if(core::make_repair_spatial_coverage(published_->data->coverage.ancestry(),cut.exact_frontier,cut.forwarded,coverage)!=core::RepairAncestryStatus::ok)fatal("foreground inherited ancestry coverage refused");
                auto data=std::make_shared<core::NativeRepairSuffixData>(coverage);data->region_roots=published_->data->region_roots;next->data=std::move(data);
            }else next->data.reset(); // whole exact replacement has no unresolved root closure
            complete(live_,cut);published_=std::move(next);++foreground_count_;
            last_={operation,published_->serial,cut.id,picked};receipts_[receipt_count_++]=last_;busy_=false;return &receipts_[receipt_count_-1];
        }catch(...){fatal("foreground approximate continuation exception");}
    }
    // Teacher-force ONE request-owned pending token, accepting the caller's next known prompt token. The trusted
    // serving composition validates both against its immutable request before entering. The computed head pick
    // is discarded: this is prompt work, never an emitted output or generation acknowledgement.
    const KnownReceipt* foreground_known(uint64_t expected,uint64_t operation,int32_t next_token,std::string& error){
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
        if(live_.slot_occupancy_){error="owned slot foreground requires genuine cohort dispatch";return nullptr;}
#endif

        if(operation&&operation==last_known_.operation){
            if(next_token!=last_known_.next){error="known operation retry changed its token";return nullptr;}
            return &last_known_;
        }
        if(!check_idle(expected)||finished_||!operation||operation<=std::max(last_.operation,last_known_.operation)||
           next_token<0||next_token>=runtime_->service_->geometry_owner_->n_vocab||
           published_->metadata.cut.accepted!=published_->metadata.cut.forwarded+1||
           published_->metadata.accepted_tokens.size()!=published_->metadata.cut.accepted||
           published_->metadata.pending_tokens.size()!=1||
           published_->metadata.accepted_tokens.size()>=live_.max_tokens_||
           published_->metadata.key.append_epoch==UINT64_MAX){error="known foreground operation/history bound";return nullptr;}
        busy_=true;
        try {
            auto next=std::make_shared<Published>();next->metadata=published_->metadata;next->serial=published_->serial;
            next->metadata.accepted_tokens.reserve(size_t(live_.max_tokens_));next->metadata.pending_tokens.reserve(size_t(live_.max_tokens_));
            const uint64_t pos=next->metadata.cut.forwarded;
            const int32_t token=next->metadata.accepted_tokens[size_t(pos)];int32_t ignored=-1;
            if(next->metadata.pending_tokens.front()!=token)fatal("known foreground pending history differs");
            next->metadata.accepted_tokens.push_back(next_token);next->metadata.pending_tokens.assign(1,next_token);
            ++next->metadata.key.append_epoch;auto& cut=next->metadata.cut;
            cut.id=mint_cut();++cut.forwarded;cut.accepted=next->metadata.accepted_tokens.size();
            cut.history=core::window_history_digest(next->metadata.accepted_tokens);
            if(next->metadata.approximation.empty())cut.exact_frontier=cut.forwarded;
            else {
                next->metadata.approximation.back().end=cut.forwarded;
                if(!published_->data)fatal("known foreground lacks inherited ancestry");
                core::RepairSpatialCoverage coverage;
                if(core::make_repair_spatial_coverage(published_->data->coverage.ancestry(),cut.exact_frontier,cut.forwarded,coverage)!=core::RepairAncestryStatus::ok)
                    fatal("known foreground ancestry coverage refused");
                auto data=std::make_shared<core::NativeRepairSuffixData>(coverage);data->region_roots=published_->data->region_roots;next->data=std::move(data);
            }
            if(!live_.begin_driver_work())fatal("known foreground RAW begin refused");
            if(!runtime_->dispatch_->run_foreground_one(token,pos,ignored,error))fatal("known foreground actual T1 refused after RAW begin");
            complete(live_,cut);published_=std::move(next);++foreground_count_;++known_count_;
            last_known_={operation,published_->serial,cut.id,token,next_token};busy_=false;return &last_known_;
        }catch(...){fatal("known foreground exception retains both gangs");}
    }
    // Refusal-site observation while donor/prefix/splice owners still exist.
    // Does not certify completion or recover a poisoned/undrained transaction.
    void attachment_refusal_diagnostic(const char* phase,uint64_t quoted,bool quote_available=true)const noexcept {
        std::fprintf(stderr,"resident attachment refusal phase=%s attachments=%u foreground=%llu",
            phase,partial_count_,(unsigned long long)foreground_count_);
        // This is the producer's splice quote, NOT a restore/capture quote.
        if(quote_available)std::fprintf(stderr," splice_quote=%llu",(unsigned long long)quoted);
        else std::fputs(" splice_quote_observed=0",stderr);
        SuffixRawTransaction::HostUsage usage;
        if(live_.observe_host_usage(usage))
            std::fprintf(stderr," h_observed=1 h_limit=%llu h_live=%llu h_claims=%u\n",
                (unsigned long long)usage.limit,(unsigned long long)usage.live,usage.claims);
        else std::fputs(" h_observed=0\n",stderr);
        std::fflush(stderr);
    }
    bool attach_prefix(uint64_t expected,std::string& error){
        if(!check_idle(expected)||partial_count_>=2||(partial_count_&&foreground_count_<=foreground_at_attachment_)){
            error="partial attachment requires current selection and intervening foreground";return false;}
        const auto& old=published_->metadata;const uint64_t p=old.cut.exact_frontier,r=private_->cut_.forwarded,d=old.cut.forwarded;
        if(p>=r||r>=d||p%4||r%4||d%4||d-r<4||old.key.selection_epoch==UINT64_MAX||published_->serial==UINT64_MAX){error="partial attachment aligned P<R<C bound";return false;}
        busy_=true;
        try {
            Pause pause(*this);auto before=published_;
            core::BundleHandle donor,prefix;
            if(!live_.capture(before->metadata,donor))fatal("partial CURRENT donor capture failed");
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
            core::resident_deferred_fault::Scope deferred;
            if(fault_deferred_trap_&&partial_count_==1)arm_deferred_trap(deferred,pause);
#endif
            bool exact_captured=false;
            try { exact_captured=capture_exact(prefix,error); }
            catch(...) {
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
                if(deferred.armed_)fatal("deferred trap capture preparation exception");
#endif
                throw;
            }
            if(!exact_captured){
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
                if(deferred.armed_)deferred.record("raw_failure",int(deferred.observed_status_),
                    int(private_->phase_==SuffixRawTransaction::Phase::poisoned),int(!private_->allocation_.healthy()));
#endif
                fatal("partial actual exact checkpoint capture failed");
            }
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
            if(deferred.armed_)fatal("deferred trap unexpectedly captured normal state");
#endif
            core::WindowKey target=old.key;++target.selection_epoch;const uint64_t cut=mint_cut();
            core::NativeSuffixLimits limits{live_.max_tokens_,live_.max_bytes_,64ULL<<20};uint64_t quoted=0;
            const auto* paired=before->data.get();
            auto q=core::quote_int8_native_repair_suffix(donor,old.key,prefix,private_key_,*runtime_->service_->geometry_owner_,
                live_.initial_config_,paired->coverage,paired->region_roots,{target,cut},limits,quoted);
            if(!q){
                attachment_refusal_diagnostic("quote",quoted,false);
                std::fprintf(stderr,"resident attachment quote status=%u reason=%u\n",unsigned(q.status),unsigned(q.reason));
                busy_=false;error="partial data quote refused before selected mutation";return false;
            }
            auto hold=std::make_shared<SpliceHold>();
            const auto splice_admission=live_.claim_external_splice(quoted,hold->cap);
            if(!splice_admission){
                attachment_refusal_diagnostic("splice_admission",quoted);
                std::fprintf(stderr,"resident attachment splice status=%u detail=%s\n",unsigned(splice_admission.status),splice_admission.detail.c_str());
                busy_=false;error="partial aggregate H admission refused: "+splice_admission.detail;return false;
            }
            auto prepared=core::prepare_int8_native_repair_suffix(donor,old.key,prefix,private_key_,*runtime_->service_->geometry_owner_,
                live_.initial_config_,paired->coverage,paired->region_roots,{target,cut},limits,live_.budget_,hold->prepared);
            if(!prepared){
                attachment_refusal_diagnostic("prepare",quoted);
                std::fprintf(stderr,"resident attachment prepare status=%u reason=%u\n",unsigned(prepared.status),unsigned(prepared.reason));
                busy_=false;error="partial actual data producer refused";return false;
            }
            // Build all publish metadata BEFORE destructive restore. A lost ack
            // retains this selected state in controller; old serial cannot win.
            auto selected=std::make_shared<Published>();selected->metadata=metadata_only(*hold->prepared.bundle().get());selected->serial=before->serial+1;
            selected->metadata.accepted_tokens.reserve(size_t(live_.max_tokens_));selected->metadata.pending_tokens.reserve(size_t(live_.max_tokens_));
            selected->data=std::shared_ptr<const core::NativeRepairSuffixData>(hold,hold->prepared.data());
            if(published_.get()!=before.get()||live_.cut_!=old.cut)fatal("selected CURRENT donor changed inside serial attachment");
            SuffixRawResult restored;
#if defined(STRATA_SERVING_FAULT_QUALIFICATION_AVAILABLE)
            if(serving_fault_.site()==core::resident_serving_fault::Site::second_partial_copy&&!serving_fault_fired_){
                const auto snapshot=serving_fault_snapshot(cut);
                core::resident_serving_fault::binding_record(serving_fault_,snapshot);
                core::ResidentCopyFault fault(serving_fault_,snapshot,live_.allocation_.stages()[1].device);
                serving_fault_fired_=true;
                restored=live_.restore(hold->prepared.bundle(),target,&fault);
                observe_serving_restore_failure(snapshot,restored,fault);
                if(restored)fatal("serving second copy fault unexpectedly restored");
            }else
#endif
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
            if(fault_second_copy_&&partial_count_==1){
                // Actual SECOND partial attachment, after named current donor
                // capture/preparation and before any new selection publication.
                core::ResidentCopyFault fault(fault_receipt_fd_,before->serial,cut,before->metadata.cut.id,
                    before->metadata.cut.forwarded,before->metadata.cut.exact_frontier,foreground_count_,partial_count_,
                    live_.allocation_.stages()[1].device);
                restored=live_.restore(hold->prepared.bundle(),target,&fault);
                if(restored)fatal("requested partial-copy failure unexpectedly restored");
            }else
#endif
            restored=live_.restore(hold->prepared.bundle(),target);
            if(!restored){
                if(restored.status==SuffixRawStatus::poisoned)fatal("partial selected restore uncertain");
                attachment_refusal_diagnostic("restore",quoted);
                std::fprintf(stderr,"resident attachment restore status=%u detail=%s\n",unsigned(restored.status),restored.detail.c_str());
                busy_=false;error=restored.detail;return false;
            }
            if(live_.cut_!=selected->metadata.cut)fatal("partial installed cut differs from prepared paired image");
            const unsigned index=partial_count_;
            attachments_[index]={selected->serial,p,r,d,foreground_count_,selected->metadata.cut.id,selected->data->coverage.ancestry().roots().size()};
            published_=std::move(selected);++partial_count_;foreground_at_attachment_=foreground_count_;busy_=false;return true;
        }catch(...){fatal("partial current-donor reattachment exception");}
    }
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
    void arm_deferred_trap(core::resident_deferred_fault::Scope& scope,const Pause& pause){
        if(!busy_||cancel_requested_||partial_count_!=1||foreground_count_!=4||receipt_count_!=4||
           published_->serial!=2||live_.cut_.forwarded!=132||live_.cut_.exact_frontier!=36||private_->cut_.forwarded!=40||
           runtime_->stage_count_!=2||private_->allocation_.stages().size()!=2||runtime_->service_->active_!=&pause||runtime_->service_->unhealthy_||
           runtime_->dispatch_->callback_depth_||!runtime_->allocation_.healthy()||!private_->allocation_.healthy()||
           private_->phase_!=SuffixRawTransaction::Phase::drained||live_.phase_!=SuffixRawTransaction::Phase::drained||
           (runtime_->phase_!=RuntimePhase::ready&&runtime_->phase_!=RuntimePhase::drained)||
           runtime_->controller_!=std::this_thread::get_id()||fault_receipt_fd_<0)
            fatal("deferred trap controller prerequisite refused");
        auto& stage=runtime_->stages_[1];
        if(runtime_->stages_[0].service.device!=0||stage.service.device!=1||!stage.session||!stage.session_owner||!stage.verifier||!stage.stream||!stage.handles.stream_count()||
           stage.verifier->stream()!=stage.stream||private_->verifiers_.size()!=2||private_->verifiers_[1]!=stage.verifier.get()||
           private_->allocation_.stages()[1].session!=stage.session||private_->allocation_.stages()[1].device!=1)
            fatal("deferred trap stage binding refused");
        core::WindowDeviceScope device(1);
        if(!device.ok())fatal("deferred trap device prerequisite refused");
        core::resident_fault_precondition::check(fault_receipt_fd_,"private_stage1_trap",1,stage.stream);
        scope.arm(fault_receipt_fd_,stage.verifier.get(),stage.stream,published_->serial,live_.cut_.id,
                  live_.cut_.forwarded,live_.cut_.exact_frontier,private_->cut_.forwarded,foreground_count_,partial_count_);
        scope.record("armed",0);
        const auto launched=core::resident_deferred_fault::launch_private_stage_trap(stage.stream);
        scope.record("launch",int(launched));
        if(launched!=cudaSuccess)fatal("deferred trap launch refused");
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        try {
        for(unsigned polls=0;polls<100000&&std::chrono::steady_clock::now()<deadline;++polls){
            std::string error;const auto drained=runtime_->stages_[0].verifier->poll_snapshot_drain(error);
            if(drained==core::Verifier::SnapshotDrain::unhealthy){
                if(!scope.observed_||(scope.observed_status_!=cudaErrorIllegalInstruction&&scope.observed_status_!=cudaErrorLaunchFailure))
                    fatal("deferred trap native query attribution unavailable");
                return; // original capture must ACTUALLY poison; no state set here
            }
            if(drained==core::Verifier::SnapshotDrain::unsupported)fatal("deferred trap drain unsupported");
        }
        fatal("deferred trap query deadline or poll bound");
        }catch(...){fatal("deferred trap query exception retains owners");}
    }
#endif
#if defined(STRATA_SERVING_FAULT_QUALIFICATION_AVAILABLE)
    core::resident_serving_fault::Snapshot serving_fault_snapshot(uint64_t target)const {
        if(!serving_fault_.valid()||serving_fault_fired_||!published_||!private_||!receipt_count_||
           runtime_->stage_count_!=2||live_.allocation_.stages().size()!=2||private_->allocation_.stages().size()!=2||
           live_.allocation_.stages()[0].device!=0||live_.allocation_.stages()[1].device!=1||
           private_->allocation_.stages()[0].device!=0||private_->allocation_.stages()[1].device!=1||
           known_count_>UINT64_MAX-receipt_count_||known_count_+receipt_count_!=foreground_count_)
            fatal("serving fault requires actual two-stage request/output owners");
        const auto& m=published_->metadata;const auto& receipt=receipts_[receipt_count_-1];
        core::resident_serving_fault::Snapshot s;
        s.publication=published_->serial;s.live_cut=m.cut.id;s.accepted=m.cut.accepted;s.forwarded=m.cut.forwarded;
        s.exact=m.cut.exact_frontier;s.watermark=m.cut.output_watermark;s.history=m.cut.history;
        s.private_cut=private_->cut_.id;s.repaired=private_->cut_.forwarded;s.target_cut=target;
        s.known=known_count_;s.generated=receipt_count_;s.foreground=foreground_count_;
        s.foreground_at_attachment=foreground_at_attachment_;s.attachments=partial_count_;
        s.receipt_operation=receipt.operation;s.receipt_publication=receipt.publication;s.receipt_cut=receipt.cut;s.receipt_token=receipt.token;
        if(receipt.token!=m.accepted_tokens.back()||m.pending_tokens.size()!=1||m.pending_tokens.front()!=receipt.token)
            fatal("serving fault last emitted receipt differs from actual pending history");
        return s;
    }
    void observe_serving_restore_failure(const core::resident_serving_fault::Snapshot& snapshot,
        const SuffixRawResult& result,const core::ResidentCopyFault& fault)const noexcept {
        core::resident_serving_fault::event(serving_fault_,snapshot,"raw_failure","restore",int(result.status),
            int(live_.phase_==SuffixRawTransaction::Phase::poisoned),int(!live_.allocation_.healthy()),
            int(live_.cut_.id==snapshot.live_cut&&published_->serial==snapshot.publication&&!finished_),
            int(bool(result.retained_failure)&&bool(live_.pending_)));
        if(!fault.fired_||!fault.stage0_completed_)fatal("serving copy fault did not reach actual intended stage1 API");
    }
    void arm_serving_replay_fault(core::resident_deferred_fault::Scope& scope,const Pause& pause,
                                 const core::resident_serving_fault::Snapshot& snapshot){
        if(!busy_||cancel_requested_||serving_fault_fired_||!serving_fault_.valid()||
           serving_fault_.site()!=core::resident_serving_fault::Site::private_replay||!snapshot.generated||
           private_->phase_!=SuffixRawTransaction::Phase::running||live_.phase_!=SuffixRawTransaction::Phase::drained||
           runtime_->phase_!=RuntimePhase::drained||runtime_->service_->active_!=&pause||runtime_->service_->unhealthy_||
           runtime_->dispatch_->callback_depth_||!runtime_->allocation_.healthy()||!private_->allocation_.healthy()||
           !runtime_->dispatch_->last_completion_||!runtime_->dispatch_->last_completion_->committed()||
           runtime_->dispatch_->last_completed_operation_!=runtime_->operation_||
           private_history_.size()!=snapshot.repaired+1||snapshot.repaired>=snapshot.forwarded||
           runtime_->controller_!=std::this_thread::get_id())fatal("serving replay fault actual T1 boundary refused");
        auto& stage=runtime_->stages_[1];
        if(!stage.session||!stage.session_owner||!stage.verifier||!stage.stream||
           stage.service.device!=1||stage.verifier->stream()!=stage.stream||private_->verifiers_.size()!=2||
           private_->verifiers_[1]!=stage.verifier.get()||private_->allocation_.stages()[1].session!=stage.session)
            fatal("serving replay fault actual stage1 binding refused");
        core::WindowDeviceScope device(1);if(!device.ok())fatal("serving replay fault device prerequisite refused");
        core::resident_serving_fault::binding_record(serving_fault_,snapshot);
        core::resident_serving_fault::event(serving_fault_,snapshot,"private_work_completed","run_private_one",0);
        core::resident_fault_precondition::check(serving_fault_.fd(),core::resident_serving_fault::name(serving_fault_.site()),1,stage.stream);
        scope.arm_serving(serving_fault_,snapshot,stage.verifier.get(),stage.stream);serving_fault_fired_=true;
        scope.record("armed",0);
        const auto launched=core::resident_deferred_fault::launch_private_stage_trap(stage.stream);scope.record("launch",int(launched));
        if(launched!=cudaSuccess)fatal("serving replay fault trap launch refused");
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        for(unsigned polls=0;polls<100000&&std::chrono::steady_clock::now()<deadline;++polls){
            // Already on the checked target device. A generic verifier poll
            // can encounter sticky failure in GetDevice/SetDevice or another
            // stage before it ever queries this bound stream. Attribute the
            // ORIGINAL target stream API first; RAW still owns poisoning below.
            const auto status=cudaStreamQuery(stage.stream);
            core::resident_deferred_fault::observe(stage.verifier.get(),
                core::resident_deferred_fault::QueryApi::stream,stage.stream,status);
            if(status==cudaErrorIllegalInstruction||status==cudaErrorLaunchFailure){
                if(!scope.observed_||scope.observed_status_!=status)
                    fatal("serving replay fault bound query attribution unavailable");
                return; // unchanged actual RAW complete_driver_work must poison
            }
            if(status!=cudaSuccess&&status!=cudaErrorNotReady)
                fatal("serving replay fault bound stream returned unsupported status");
        }
        fatal("serving replay fault query deadline or poll bound");
    }
#endif
    bool capture_exact(core::BundleHandle& out,std::string& error){
        if(!exact_completion_||exact_completion_->cut!=private_->cut_||!exact_completion_->receipt.committed()||
           private_->cut_.exact_frontier!=private_->cut_.forwarded){error="exact cursor lacks actual committed completion witness";return false;}
        core::SnapshotDescriptor meta;meta.key=private_key_;meta.cut=private_->cut_;meta.accepted_tokens=private_history_;
        meta.steering_enabled=false;meta.speculation_disabled=true;
        return bool(private_->capture(meta,out));
    }
    // Preserve the experiment's two-attachment/later-generation trace policy. The serving operation has a
    // separate no-generation policy; both share the same actual CURRENT exact-cut publication below.
    bool finish_current(uint64_t expected,std::string& error){
        return finish_current_with_trace(expected,core::serving_raw::RepairFinishMode::two_attachment_generation,error);
    }
    bool finish_serving_current(uint64_t expected,std::string& error){
        return finish_current_with_trace(expected,core::serving_raw::RepairFinishMode::serving_no_generation,error);
    }
    bool finish_request_current(uint64_t expected,std::string& error){
        return finish_current_with_trace(expected,core::serving_raw::RepairFinishMode::request_continuation,error);
    }
    // Whole exact replacement is allowed ONLY at CURRENT live forwarded D with its genuine private completion
    // witness. Changing trace policy does not relax the idle selection, history, cut, completion or restore gates.
    bool finish_current_with_trace(uint64_t expected,core::serving_raw::RepairFinishMode mode,std::string& error){
        if(!check_idle(expected)||!core::serving_raw::repair_finish_trace_acceptable(mode,
               {partial_count_,foreground_count_,foreground_at_attachment_,uint64_t(receipt_count_),
                initial_accepted_,published_->metadata.cut.accepted,
                initial_output_watermark_,published_->metadata.cut.output_watermark})||
           private_->cut_.forwarded!=published_->metadata.cut.forwarded||!exact_completion_||
           exact_completion_->cut!=private_->cut_||!exact_completion_->receipt.committed()||
           published_->metadata.key.selection_epoch==UINT64_MAX||published_->serial==UINT64_MAX){error="full catchup requires actual exact CURRENT D witness";return false;}
        busy_=true;
        try {
            Pause pause(*this);auto before=published_;auto target=before->metadata.key;++target.selection_epoch;
            const auto d=before->metadata.cut.forwarded;
            if(private_history_.size()!=d||!std::equal(private_history_.begin(),private_history_.end(),before->metadata.accepted_tokens.begin()))fatal("final exact CURRENT prefix mismatch");
            core::VerifierCompletionGate::Token operation;auto& binding=*runtime_->dispatch_;
            if(!binding.gate_->try_begin(operation)||!private_->begin_driver_work())fatal("final pending-history gate admission refused");
            // Accept known current pending history/output watermark without
            // forwarding it. All GPU endpoint buffers still represent exact D.
            auto cut=before->metadata.cut;cut.id=mint_cut();cut.exact_frontier=d;
            complete(*private_,cut);runtime_->cut_=cut;private_key_=target;
            core::VerifierCompletionGate::Cut witness;
            if(!binding.gate_->commit_after_checked_completion(operation,witness))fatal("final exact pending-history completion gate refused");
            exact_completion_.reset();exact_completion_.emplace(cut,std::move(witness));
            core::SnapshotDescriptor meta=before->metadata;meta.key=target;meta.cut=cut;meta.approximation.clear();
            core::BundleHandle exact;if(!private_->capture(meta,exact))fatal("final actual complete exact CURRENT capture failed");
            auto selected=std::make_shared<Published>();selected->metadata=metadata_only(*exact.get());selected->serial=before->serial+1;
            selected->metadata.accepted_tokens.reserve(size_t(live_.max_tokens_));selected->metadata.pending_tokens.reserve(size_t(live_.max_tokens_));
            if(published_.get()!=before.get()||live_.cut_!=before->metadata.cut)fatal("final CURRENT selection changed before install");
            SuffixRawResult restored;
#if defined(STRATA_SERVING_FAULT_QUALIFICATION_AVAILABLE)
            if(serving_fault_.site()==core::resident_serving_fault::Site::final_exact_copy&&!serving_fault_fired_){
                const auto snapshot=serving_fault_snapshot(exact.get()->cut.id);
                core::resident_serving_fault::binding_record(serving_fault_,snapshot);
                core::ResidentCopyFault fault(serving_fault_,snapshot,live_.allocation_.stages()[1].device);
                serving_fault_fired_=true;restored=live_.restore(exact,target,&fault);
                observe_serving_restore_failure(snapshot,restored,fault);
                if(restored)fatal("serving final copy fault unexpectedly restored");
            }else
#endif
            restored=live_.restore(exact,target);
            if(!restored){if(restored.status==SuffixRawStatus::poisoned)fatal("final whole-state exact restore uncertain");busy_=false;error=restored.detail;return false;}
            published_=std::move(selected);finished_=true;busy_=false;return true;
        }catch(...){fatal("final exact CURRENT catchup exception");}
    }
    // Additional exact endpoint check, NOT the independent clean replay oracle.
    // Both gangs consume precisely the same previously accepted pending H[D].
    bool compare_current_pending(std::string& error){
        if(!finished_||!check_idle(published_->serial)||!published_->metadata.approximation.empty()||
           private_->cut_.forwarded!=published_->metadata.cut.forwarded||
           !exact_completion_||exact_completion_->cut!=private_->cut_||!exact_completion_->receipt.committed()||
           published_->metadata.pending_tokens.size()!=1||receipt_count_==receipts_.size()||
           published_->metadata.accepted_tokens.size()>=live_.max_tokens_){error="exact pending comparison phase/history";return false;}
        auto& binding=*runtime_->dispatch_;auto& root=*runtime_->stages_[0].verifier;
        auto& live_root=*runtime_->service_->owner_->primary_verifier;
        const int64_t vocab=root.vocab();
        if(vocab<=0||vocab!=live_root.vocab()||uint64_t(vocab)>(UINT64_MAX-65536)/(2*sizeof(float))||uint64_t(vocab)>SIZE_MAX/sizeof(float)){
            error="actual identical vocabulary comparison bound";return false;}
        // Two fresh host arrays plus bounded metadata. REAL global + SAME H
        // admission BEFORE vectors; buffers die before retained credit.
        std::shared_ptr<const void> credit;
        if(!live_.admit_controller_host(uint64_t(vocab)*2*sizeof(float)+65536,credit)){error="exact logit comparison admission refused";return false;}
        std::vector<float> private_logits,selected_logits;
        private_logits.reserve(size_t(vocab));selected_logits.reserve(size_t(vocab));
        private_logits.resize(size_t(vocab));selected_logits.resize(size_t(vocab));
        const uint64_t d=published_->metadata.cut.forwarded;
        if(private_history_.size()!=d||!std::equal(private_history_.begin(),private_history_.end(),published_->metadata.accepted_tokens.begin())){
            error="exact pending comparison actual prefix changed";return false;}
        const int32_t token=published_->metadata.accepted_tokens[size_t(d)];
        const auto same_history=published_->metadata.cut.history;
        busy_=true;
        try {
            int32_t private_picked=-1;
            if(!private_->begin_driver_work()||!binding.run_private_one(token,d,private_picked,error))fatal("exact pending private actual T1 failed");
            private_history_.push_back(token);
            core::StateCut cut;cut.id=mint_cut();cut.accepted=cut.forwarded=cut.exact_frontier=d+1;
            cut.output_watermark=published_->metadata.cut.output_watermark;cut.history=core::window_history_digest(private_history_);
            if(cut.history!=same_history)fatal("exact pending input history differs between gangs");
            complete(*private_,cut);runtime_->cut_=cut;
            if(!binding.last_completion_||!binding.last_completion_->committed()||binding.last_completed_operation_!=runtime_->operation_)fatal("exact pending private completion witness missing");
            exact_completion_.reset();exact_completion_.emplace(cut,std::move(*binding.last_completion_));binding.last_completion_.reset();
            {
                Pause pause(*this);core::WindowDeviceScope head(runtime_->stages_[runtime_->stage_count_-1].service.device);
                if(!head.ok()||!root.copy_logits(0,private_logits.data())||cudaDeviceSynchronize()!=cudaSuccess||!head.finish())fatal("exact pending private head copy uncertain");
            }
            // Single controller invokes its own real foreground method. No
            // external callback/event loop can enter between these operations.
            busy_=false;
            const Receipt* receipt=foreground(published_->serial,last_.operation+1,error);
            if(!receipt)fatal("exact pending selected actual T1 refused after private forward");
            busy_=true;
            {
                Pause pause(*this);const auto& live_stages=runtime_->service_->owner_->sessions.stages();
                core::WindowDeviceScope head(live_stages.back().device);
                if(!head.ok()||!live_root.copy_logits(0,selected_logits.data())||cudaDeviceSynchronize()!=cudaSuccess||!head.finish())fatal("exact pending selected head copy uncertain");
            }
            if(!compare_suffix_logits(private_logits,selected_logits,comparison_))fatal("exact pending copied logits nonfinite/invalid");
            if(comparison_.exact_argmax!=private_picked||comparison_.approximate_argmax!=receipt->token)fatal("exact pending actual picks differ from copied row argmax");
            comparison_done_=true;comparison_pass_=private_picked==receipt->token;comparison_forwarded_=d+1;comparison_history_=same_history;
            busy_=false;
            if(!comparison_pass_)error="private and selected exact pending-token picks differ";
            return comparison_pass_;
        }catch(...){fatal("exact pending comparison exception retains actual owners");}
    }
    enum class Scenario { normal, cancel_before_second_attachment, cancel_after_second_prepare };
    // Exact bounded observations, no GPU/resource query or fake completion.
    // Published history/root objects are immutable while reachable here.
    struct Observation {
        ReceiptHistory::Snapshot receipt_snapshot; // separately paid; payload before credit
        std::shared_ptr<Published> selected;
        core::WindowKey key,private_key;
        Receipt last;
        std::shared_ptr<const core::NativeRepairSuffixData> ancestry_data;
        uint64_t coverage_begin=0,coverage_end=0;
        std::array<core::RepairRegionRootBinding,64> region_roots{};
        size_t region_count=0;
        core::StateCut live,private_cut,runtime_cut,selected_cut,exact_cut;
        std::vector<int32_t> accepted,pending,private_history;
        std::array<int64_t,maximum_stages> private_windows{},live_windows{};
        std::array<std::array<int32_t,3>,maximum_stages> private_ple{},live_ple{};
        std::array<core::RepairCausalRoot,core::RepairAncestry::capacity> roots{};
        size_t root_count=0,receipt_count=0;
        uint64_t next_cut=0,service_operation=0,private_operation=0,last_completed=0,foreground=0;
        RuntimePhase phase=RuntimePhase::host_only;
        SuffixRawTransaction::Phase live_phase=SuffixRawTransaction::Phase::idle,private_phase=SuffixRawTransaction::Phase::idle;
        const void* service_active=nullptr;
        unsigned callback_depth=0,attachments=0;
        bool exact_receipt=false,binding_receipt=false,gate_healthy=false;
    };
    Observation observe()const {
        Observation o;o.selected=published_;o.key=published_->metadata.key;o.private_key=private_key_;o.last=last_;o.live=live_.cut_;o.private_cut=private_->cut_;
        o.runtime_cut=runtime_->cut_;o.selected_cut=published_->metadata.cut;
        o.accepted=published_->metadata.accepted_tokens;o.pending=published_->metadata.pending_tokens;
        o.private_history=private_history_;o.receipt_count=receipt_count_;
        if(!receipts_.snapshot(receipt_count_,[&](uint64_t bytes,std::shared_ptr<const void>& credit){
            return bool(live_.admit_controller_host(bytes,credit));},o.receipt_snapshot))
            fatal("receipt observation host admission/allocation refused");
        if(published_->data){o.ancestry_data=published_->data;o.coverage_begin=published_->data->coverage.begin();o.coverage_end=published_->data->coverage.end();
            o.region_count=published_->data->region_roots.size();if(o.region_count>o.region_roots.size())fatal("observer root map bound");
            std::copy(published_->data->region_roots.begin(),published_->data->region_roots.end(),o.region_roots.begin());
            auto roots=published_->data->coverage.ancestry().roots();o.root_count=roots.size();std::copy(roots.begin(),roots.end(),o.roots.begin());}
        o.next_cut=next_cut_;o.foreground=foreground_count_;o.attachments=partial_count_;o.phase=runtime_->phase_;
        o.live_phase=live_.phase_;o.private_phase=private_->phase_;auto& b=*runtime_->dispatch_;
        o.service_operation=runtime_->service_->operation_;o.private_operation=runtime_->operation_;o.last_completed=b.last_completed_operation_;
        o.service_active=runtime_->service_->active_;o.callback_depth=b.callback_depth_;o.gate_healthy=b.gate_->healthy_advisory();
        o.exact_receipt=exact_completion_&&exact_completion_->receipt.committed();if(exact_completion_)o.exact_cut=exact_completion_->cut;
        o.binding_receipt=b.last_completion_&&b.last_completion_->committed();
        const auto& live=runtime_->service_->owner_->sessions.stages();
        for(size_t i=0;i<runtime_->stage_count_;++i){o.private_windows[i]=runtime_->stages_[i].verifier->windows;
            o.live_windows[i]=(i?&runtime_->service_->owner_->stages[i-1]->ver:runtime_->service_->owner_->primary_verifier.get())->windows;
            const auto& p=*runtime_->stages_[i].session;const auto& l=*live[i].session;
            o.private_ple[i]={p.ple_prev[0],p.ple_prev[1],p.ple_token};o.live_ple[i]={l.ple_prev[0],l.ple_prev[1],l.ple_token};}
        return o;
    }
    static bool receipt_equal(const Receipt& a,const Receipt& b)noexcept {
        return a.operation==b.operation&&a.publication==b.publication&&a.cut==b.cut&&a.token==b.token;
    }
    bool unchanged(const Observation& before,bool one_unused_mint=false)const {
        const auto after=observe();
        if(before.selected.get()!=after.selected.get()||before.key!=after.key||before.private_key!=after.private_key||!receipt_equal(before.last,after.last)||
           before.ancestry_data.get()!=after.ancestry_data.get()||before.coverage_begin!=after.coverage_begin||before.coverage_end!=after.coverage_end||before.region_count!=after.region_count||before.live!=after.live||before.private_cut!=after.private_cut||
           before.runtime_cut!=after.runtime_cut||before.selected_cut!=after.selected_cut||before.exact_cut!=after.exact_cut||
           before.accepted!=after.accepted||before.pending!=after.pending||before.private_history!=after.private_history||
           before.private_windows!=after.private_windows||before.live_windows!=after.live_windows||before.private_ple!=after.private_ple||before.live_ple!=after.live_ple||
           before.root_count!=after.root_count||before.receipt_count!=after.receipt_count||(one_unused_mint?(before.next_cut==UINT64_MAX||before.next_cut+1!=after.next_cut):before.next_cut!=after.next_cut)||before.foreground!=after.foreground||
           before.attachments!=after.attachments||before.phase!=after.phase||before.live_phase!=after.live_phase||before.private_phase!=after.private_phase||
           before.service_operation!=after.service_operation||before.private_operation!=after.private_operation||before.last_completed!=after.last_completed||
           before.service_active!=after.service_active||before.callback_depth!=after.callback_depth||before.gate_healthy!=after.gate_healthy||
           before.exact_receipt!=after.exact_receipt||before.binding_receipt!=after.binding_receipt)return false;
        for(size_t i=0;i<before.region_count;++i)if(before.region_roots[i].region!=after.region_roots[i].region||before.region_roots[i].root!=after.region_roots[i].root)return false;
        for(size_t i=0;i<before.root_count;++i)if(before.roots[i]!=after.roots[i])return false;
        for(size_t i=0;i<before.receipt_count;++i)if(!receipt_equal(before.receipt_snapshot.entries[i],after.receipt_snapshot.entries[i]))return false;
        return !busy_;
    }
    bool check_stale_serial(std::string& error){
        if(!check_idle(published_->serial)||published_->serial<=1||partial_count_!=1){error="stale self-check requires first published attachment";return false;}
        const auto before=observe();std::string refused;
        if(attach_prefix(published_->serial-1,refused)||refused.empty()||!unchanged(before))fatal("stale selection altered actual resident state/counters");
        ++stale_checks_;return true;
    }
    bool check_historical_receipt(std::string& error){
        if(!check_idle(published_->serial)||!receipt_count_){error="receipt retry self-check requires committed foreground";return false;}
        const auto before=observe();const Receipt expected=receipts_[0];
        const Receipt* retry=foreground(expected.publication,expected.operation,error);
        if(retry!=&receipts_[0]||!receipt_equal(*retry,expected)||!unchanged(before))fatal("historical receipt retry changed actual state/selection");
        ++retry_checks_;return true;
    }
    bool request_cancel(uint64_t expected,std::string& error){
        if(!check_idle(expected)||runtime_->service_->active_||runtime_->dispatch_->callback_depth_||
           private_->phase_!=SuffixRawTransaction::Phase::drained||
           (runtime_->phase_!=RuntimePhase::ready&&runtime_->phase_!=RuntimePhase::drained)){
            error="cancellation requires actual drained controller boundary";return false;}
        cancel_requested_=true;return true; // one-way; never resets admission
    }
    bool cancel_at_drained_boundary(std::string& error,core::BundleHandle before={}){
        if(partial_count_!=1||finished_){error="cancellation scenario requires before second publication";return false;}
        auto selected=published_;const auto cut=live_.cut_;const auto key=selected->metadata.key;const auto foreground=foreground_count_;
        // Owners live OUTSIDE try: uncertain capture failure reaches failstop
        // before captured payload/runtime owners could be unwound.
        core::BundleHandle after;
        try {
            if(!before){
                Pause pause(*this);
                const auto captured=live_.capture(selected->metadata,before);
                if(!captured){if(captured.status==SuffixRawStatus::poisoned)fatal("pre-cancel live capture uncertain");error=captured.detail;return false;}
            }
            if(!request_cancel(selected->serial,error))return false;
            cancelled_=true;
            if(!retire(error))fatal("drained cancellation private retirement failed");
            if(published_.get()!=selected.get()||live_.cut_!=cut||published_->metadata.key!=key||partial_count_!=1||finished_||foreground_count_!=foreground||
               live_.phase_!=SuffixRawTransaction::Phase::drained)fatal("cancellation changed selected committed state");
            {
                // Private gang is retired. Pause retains ONLY the actual shared
                // service exclusion; live foreground owners/streams remain valid.
                Pause pause(*this);
                const auto captured=live_.capture(selected->metadata,after);
                if(!captured){if(captured.status==SuffixRawStatus::poisoned)fatal("post-cancel live capture uncertain");error=captured.detail;return false;}
            }
            core::WindowStateLayout layout;
            const auto valid=core::native_window_layout(live_.geometry_,live_.initial_config_,key,cut,
                live_.max_tokens_,live_.max_bytes_,layout,live_.steering_enabled_);
            if(!valid){error=valid.detail;return false;}
            const bool preserved=clean_recompute_qualification::compare_preserved_selection(before,after,layout,cancel_native_report_);
            if(!preserved)error="drained cancellation changed selected native state or comparison failed";
            return preserved; // terminal cancel SUCCESS requires full native match
        }catch(...){fatal("drained cancellation preservation exception retains owners");}
    }
    // Dedicated synchronous qualification: genuine candidate, never installed.
    // Donor survives as before-native witness; all candidate owners die before
    // the existing idle cancellation/retirement method is entered.
    bool cancel_after_prepare(std::string& error){
        if(!check_idle(published_->serial)||partial_count_!=1||foreground_count_!=4||
           receipt_count_!=4||published_->serial!=2){error="prepared cancellation boundary";return false;}
        core::BundleHandle donor; // outside try: uncertainty failstops with owner retained
        const auto observation=observe();const auto& old=published_->metadata;
        const auto binding=runtime_->dispatch_;const auto gate=binding->gate_;
        std::array<std::uintptr_t,maximum_stages> addresses{};const size_t stages=runtime_->stage_count_;
        if(!stages||stages>maximum_stages)fatal("prepared cancellation stage count");
        for(size_t i=0;i<stages;++i){addresses[i]=reinterpret_cast<std::uintptr_t>(runtime_->stages_[i].verifier.get());
            if(!addresses[i]||!core::Verifier::diagnostic_contains_address(addresses[i]))fatal("prepared cancellation stage not registered");}

        const uint64_t p=old.cut.exact_frontier,r=private_->cut_.forwarded,d=old.cut.forwarded;
        if(p>=r||r>=d||p%4||r%4||d%4||d-r<4||old.key.selection_epoch==UINT64_MAX||next_cut_==UINT64_MAX){
            error="prepared cancellation aligned cut bound";return false;}
        const auto& parent=live_.allocation_.reservation();
        if(!parent||parent->resources().device_bytes.empty()||parent->resources().device_bytes.size()>64){error="prepared cancellation ledger dimension bound";return false;}
        // Bootstrap parent and ledger dimensions are immutable while active.
        core::WindowBudgetSnapshot donor_budget,released_budget;
        SuffixRawTransaction::HostUsage donor_usage,released_usage;
        std::weak_ptr<SpliceHold> discarded;
        busy_=true;
        try {
            {
                Pause pause(*this);
                if(!live_.capture(old,donor))fatal("prepared cancellation current donor capture failed");
                if(!live_.observe_host_usage(donor_usage))fatal("prepared cancellation H observation failed");
                donor_budget=live_.budget_.snapshot();
                if(donor_budget.overcommitted||donor_budget.capacity.device_bytes.empty()||donor_budget.capacity.device_bytes.size()>64||
                   donor_budget.baseline.device_bytes.size()!=donor_budget.capacity.device_bytes.size()||
                   donor_budget.reserved.device_bytes.size()!=donor_budget.capacity.device_bytes.size())fatal("prepared cancellation ledger snapshot bound");
                core::BundleHandle prefix;
                if(!capture_exact(prefix,error))fatal("prepared cancellation exact checkpoint capture failed");
                core::WindowKey target=old.key;++target.selection_epoch;const uint64_t cut=mint_cut();
                core::NativeSuffixLimits limits{live_.max_tokens_,live_.max_bytes_,64ULL<<20};uint64_t quoted=0;
                const auto* paired=published_->data.get();
                auto q=core::quote_int8_native_repair_suffix(donor,old.key,prefix,private_key_,*runtime_->service_->geometry_owner_,
                    live_.initial_config_,paired->coverage,paired->region_roots,{target,cut},limits,quoted);
                if(!q){busy_=false;error="prepared cancellation quote refused";return false;}
                auto hold=std::make_shared<SpliceHold>();discarded=hold;
                if(!live_.claim_external_splice(quoted,hold->cap)){busy_=false;error="prepared cancellation aggregate H refused";return false;}
                auto prepared=core::prepare_int8_native_repair_suffix(donor,old.key,prefix,private_key_,*runtime_->service_->geometry_owner_,
                    live_.initial_config_,paired->coverage,paired->region_roots,{target,cut},limits,live_.budget_,hold->prepared);
                if(!prepared){busy_=false;error="prepared cancellation actual producer refused";return false;}
                auto selected=std::make_shared<Published>();selected->metadata=metadata_only(*hold->prepared.bundle().get());selected->serial=published_->serial+1;
                selected->metadata.accepted_tokens.reserve(size_t(live_.max_tokens_));selected->metadata.pending_tokens.reserve(size_t(live_.max_tokens_));
                selected->data=std::shared_ptr<const core::NativeRepairSuffixData>(hold,hold->prepared.data());
                if(!hold->cap||!hold->prepared.bundle()||!selected->data||!quoted||
                   selected->metadata.key!=target||selected->metadata.cut.id!=cut||
                   published_.get()!=observation.selected.get()||live_.cut_!=old.cut)
                    fatal("prepared cancellation candidate binding failed");
                SuffixRawTransaction::HostUsage charged;
                if(!live_.observe_host_usage(charged)||charged.limit!=donor_usage.limit||charged.live<=donor_usage.live||charged.claims<=donor_usage.claims)
                    fatal("prepared cancellation charged candidate observation failed");
                uint64_t tensor_count=0,payload_bytes=0;
                for(const auto& stage:hold->prepared.bundle().get()->devices)for(const auto& tensor:stage.tensors){
                    if(tensor_count==UINT64_MAX||tensor.layout.bytes>UINT64_MAX-payload_bytes)fatal("prepared cancellation payload count overflow");
                    ++tensor_count;payload_bytes+=tensor.layout.bytes;
                }
                if(!tensor_count||!payload_bytes)fatal("prepared cancellation empty native payload");
                prepared_cancel_tensors_=tensor_count;prepared_cancel_bytes_=payload_bytes;
                prepared_cancel_quote_=quoted;prepared_cancel_target_=cut;prepared_cancel_old_=old.cut.id;
                prepared_cancel_h_delta_=charged.live-donor_usage.live;
                prepared_cancel_claim_delta_=charged.claims-donor_usage.claims;
                // NO restore, completion, publication, attachment increment or
                // cancellation mutation inside Pause. Scope destroys aliases,
                // prepared bundle, H claim and prefix before idle transition.
            }
            busy_=false;
            if(!discarded.expired()||!live_.observe_host_usage(released_usage)||
               released_usage.limit!=donor_usage.limit||released_usage.live!=donor_usage.live||released_usage.claims!=donor_usage.claims)
                fatal("prepared cancellation temporary ownership did not release");
            released_budget=live_.budget_.snapshot();
            if(released_budget.overcommitted||released_budget.capacity!=donor_budget.capacity||
               released_budget.baseline!=donor_budget.baseline||released_budget.reserved!=donor_budget.reserved)
                fatal("prepared cancellation global candidate charge did not release");
            // A real preparation consumes one cut; no rollback/reuse of its ID.
            if(!unchanged(observation,true))fatal("prepared cancellation changed selected history/receipts");
            prepared_cancel_released_=true;
            const bool ok=cancel_at_drained_boundary(error,std::move(donor));
            if(ok){
                if(!retired_||private_||runtime_->phase_!=RuntimePhase::retired||runtime_->stage_count_||runtime_->dispatch_||
                   !runtime_->allocation_.stages().empty()||runtime_->service_->active_||runtime_->service_->unhealthy_||
                   !binding->private_sealed_||binding->callback_depth_||gate->healthy_advisory())fatal("prepared cancellation retirement incomplete");
                for(size_t i=0;i<stages;++i){const auto& stage=runtime_->stages_[i];
                    if(stage.verifier||stage.session||stage.session_owner||stage.stream||core::Verifier::diagnostic_contains_address(addresses[i]))
                        fatal("prepared cancellation stage survived retirement");}
                if(core::native_session_quarantine_count()||core::window_buffer_quarantine_count()||core::model_stage_quarantine_count()||core::model_gang_quarantine_count())
                    fatal("prepared cancellation quarantined resources");
            }
            prepared_cancel_pass_=ok;return ok;
        }catch(...){fatal("prepared cancellation exception retains donor owner");}
    }
    bool cancelled()const noexcept{return cancelled_;}
    // Concrete bounded milestone: CURRENT captures twice, with actual T1
    // foreground units between and after; final exact state catches CURRENT D.
    bool execute_two_attachments(uint64_t first_r,uint64_t second_r,std::string& error,Scenario scenario=Scenario::normal){
        if(!published_||first_r>=second_r||second_r%4||first_r%4||second_r>=published_->metadata.cut.forwarded){error="two attachment target bounds";return false;}
        if(!repair_to(published_->serial,first_r,error)||!attach_prefix(published_->serial,error))return false;
        for(unsigned i=0;i<4;++i)if(!foreground(published_->serial,last_.operation+1,error))return false;
        if(!repair_to(published_->serial,second_r,error)||!check_stale_serial(error)||!check_historical_receipt(error))return false;
        if(scenario==Scenario::cancel_before_second_attachment)return cancel_at_drained_boundary(error);
        if(scenario==Scenario::cancel_after_second_prepare)return cancel_after_prepare(error);
        if(!attach_prefix(published_->serial,error)||!check_historical_receipt(error))return false;
        for(unsigned i=0;i<4;++i)if(!foreground(published_->serial,last_.operation+1,error))return false;
        const uint64_t current_d=published_->metadata.cut.forwarded;
        if(!repair_to(published_->serial,current_d,error)||!finish_current(published_->serial,error))return false;
        // Capture INSTALLED live D before the ninth forward can overwrite any
        // restored endpoint field. capture performs real global+SAME-H admission
        // before native copies/freeze and retains that charge with this handle.
        if(!check_idle(published_->serial)||catchup_snapshot_){error="catchup capture phase";return false;}
        busy_=true;
        try {
            Pause pause(*this);
            auto captured=live_.capture(published_->metadata,catchup_snapshot_);
            if(!captured){
                if(captured.status==SuffixRawStatus::poisoned)fatal("installed catchup capture uncertain");
                busy_=false;error=captured.detail;return false;
            }
            if(catchup_snapshot_.get()->cut!=published_->metadata.cut)fatal("installed catchup captured cut changed");
            busy_=false;
        }catch(...){fatal("installed catchup capture exception retains owners");}
        // Pending receipt8 is inherited, not retroactively resampled by repair.
        // The newly exact ninth prediction remains a separate real operation.
        return compare_current_pending(error)&&receipt_trace_valid(error);
    }
    // ---- serving production repair -------------------------------------------------------------------
    // Exactly ONE attachment and NO foreground step.  The controller's `foreground()` is a GENERATION step (it
    // samples a token, appends the pick to the accepted history and raises the output watermark), so using it to
    // walk a serving request's own prompt would silently substitute generated tokens for request tokens.  Here the
    // exactness chain is: private catch-up to the request's frontier (real forwards of the request's OWN accepted
    // tokens), CURRENT prefix attachment (the donor's rows past the frontier become the selection), private
    // catch-up to the current endpoint, finish, installed catchup capture, and the private/live pending check.
    // Nothing past the current endpoint is touched: the request keeps reading its own prompt from there.
    // The scenario's generation-based self-checks (stale serial, historical receipt retry) cannot run without
    // foreground receipts, so this mode deliberately omits them rather than faking a receipt.
    bool execute_suffix_repair(std::span<const uint64_t> repair_at,unsigned foreground_between,std::string& error){
        auto refuse=[&](const char* why){error=why;return false;};
        if(repair_at.size()!=core::serving_raw::kServingRepairMaxFrontiers)
            return refuse("serving suffix repair admits exactly one attachment");
        if(foreground_between)return refuse("serving suffix repair admits no foreground step: it would generate request tokens");
        if(!published_||finished_||busy_||retired_||cancel_requested_)return refuse("suffix repair needs an idle fresh controller");
        if(!check_idle(published_->serial))return refuse("suffix repair selection changed");
        const uint64_t current=published_->metadata.cut.forwarded;
        const uint64_t exact=published_->metadata.cut.exact_frontier;
        const uint64_t frontier=repair_at[0];
        if(frontier%4||current%4||exact%4||exact>=frontier||frontier>=current||
           current-frontier<core::serving_raw::kServingRepairMinGap)
            return refuse("suffix repair frontier/current/exact bounds");
        if(published_->metadata.accepted_tokens.size()<=current)
            return refuse("suffix repair needs the request's own token at the current endpoint");
        if(!repair_to(published_->serial,frontier,error))return false;
        if(!attach_prefix(published_->serial,error))return false;
        const uint64_t current_d=published_->metadata.cut.forwarded;
        if(current_d!=current)return refuse("suffix repair current endpoint moved during attachment");
        if(!repair_to(published_->serial,current_d,error)||!finish_serving_current(published_->serial,error))return false;
        if(!check_idle(published_->serial)||catchup_snapshot_)return refuse("suffix repair catchup capture phase");
        busy_=true;
        try {
            Pause pause(*this);
            auto captured=live_.capture(published_->metadata,catchup_snapshot_);
            if(!captured){
                if(captured.status==SuffixRawStatus::poisoned)fatal("installed catchup capture uncertain");
                busy_=false;error=captured.detail;return false;
            }
            if(catchup_snapshot_.get()->cut!=published_->metadata.cut)fatal("installed catchup captured cut changed");
            busy_=false;
        }catch(...){fatal("installed catchup capture exception retains owners");}
        // NO generation in this mode, so the scenario's `compare_current_pending` is deliberately NOT called: it
        // ends with its own `foreground(...)` (a sampled pick appended to the accepted history, advancing live
        // forwarded).  The state is pinned instead by facts the caller can verify against the request: the current
        // endpoint has not moved, the published history is exactly the accepted history it was handed, the pending
        // token is the request's own token at that endpoint, and no generation receipt or watermark exists.
        const auto& now=published_->metadata;
        if(now.cut.forwarded!=current||now.cut.exact_frontier!=current||private_->cut_.forwarded!=current)
            return refuse("suffix repair moved the current endpoint: generation is not permitted here");
        if(now.accepted_tokens.size()!=current+1||now.cut.accepted!=current+1)
            return refuse("suffix repair accepted history does not match the current endpoint");
        if(now.pending_tokens.size()!=1||now.pending_tokens.front()!=now.accepted_tokens[size_t(current)])
            return refuse("suffix repair pending token is not the request's own endpoint token");
        if(receipt_count_!=0||now.cut.output_watermark!=0)
            return refuse("suffix repair committed a generation receipt in no-generation mode");
        if(!now.approximation.empty())return refuse("suffix repair left an unresolved approximate region");
        return repair_receipts_consistent(repair_at,error);
    }
    // Structural (not scenario-shaped) receipt discipline: every attachment widened the exact frontier inside the
    // qualified bounds, every receipt is a distinct monotonic operation drawn from the SAME accepted history, and
    // the operation ended with the current selection exact.  Deliberately asserts no fixed token counts.
    bool repair_receipts_consistent(std::span<const uint64_t> repair_at,std::string& error){
        auto refuse=[&](const char* why){error=why;return false;};
        if(!finished_||!check_idle(published_->serial)||!catchup_snapshot_)
            return refuse("repair receipt check needs a finished idle controller with an installed catchup");
        const auto& final=published_->metadata;
        if(!final.approximation.empty())return refuse("repair left an unresolved approximate region");
        if(final.cut.exact_frontier!=final.cut.forwarded||private_->cut_.forwarded!=final.cut.forwarded)
            return refuse("repair did not certify the current selection exact");
        if(final.pending_tokens.size()!=1||final.accepted_tokens.size()<=final.cut.forwarded)
            return refuse("repair published an incomplete accepted history");
        if(final.cut.output_watermark!=receipt_count_)return refuse("repair watermark and receipt count differ");
        if(receipt_count_!=0)return refuse("serving repair committed a generation receipt");
        if(partial_count_!=repair_at.size())return refuse("repair attachment count differs from the request");
        for(unsigned i=0;i<partial_count_;++i){
            const auto& a=attachments_[i];
            if(!core::serving_raw::repair_attachment_publication_acceptable(a.publication,i))
                return refuse("repair publication sequence differs");
            if(a.exact!=repair_at[i])return refuse("repair frontier was not the requested position");
            if(a.previous>=a.exact||a.exact>=a.current||a.current-a.exact<4)return refuse("repair attachment bounds differ");
            if(i){
                if(attachments_[i-1].exact>=a.exact)return refuse("repair frontiers are not increasing");
                if(attachments_[i-1].foreground_before>=a.foreground_before)
                    return refuse("an attachment ran without fresh forwards after the previous one");
                if(attachments_[i-1].current>a.previous)return refuse("repair current state moved backwards");
            }
            if(a.cut>=catchup_snapshot_.get()->cut.id)return refuse("repair attachment cut is not older than catchup");
        }
        for(size_t i=0;i<receipt_count_;++i){
            const auto& receipt=receipts_[i];
            if(!receipt.operation)return refuse("repair receipt lacks an operation identity");
            if(i&&receipts_[i-1].cut>=receipt.cut)return refuse("repair receipt cuts are not monotonic");
            const size_t history=size_t(initial_accepted_)+i;
            if(history>=final.accepted_tokens.size())return refuse("repair receipt exceeds accepted history");
            if(receipt.token<0||receipt.token!=final.accepted_tokens[history])
                return refuse("repair receipt token is not the accepted history token");
        }
        repair_receipts_pass_=true;return true;
    }
    // Independent of the report formatter: bind all nine actual receipts to
    // immutable accepted history. Historical approximate picks need not equal
    // a clean replay argmax. Only the existing ninth exact check demands that.
    bool receipt_trace_valid(std::string& error){
        receipt_trace_checked_=true;receipt_trace_pass_=false;
        receipt_pending_bound_=false;receipt_history_bound_=false;
        auto refuse=[&](const char* why){error=why;return false;};
        if(cancelled_||!finished_||!published_||!check_idle(published_->serial)||!catchup_snapshot_||
           !comparison_done_||!comparison_pass_||initial_accepted_!=129||initial_output_watermark_!=0||
           receipt_count_!=9||foreground_count_!=9||partial_count_!=2||published_->serial!=4)
            return refuse("actual nine-receipt trace phase/count mismatch");
        const auto& final=published_->metadata;const auto& before=*catchup_snapshot_.get();
        if(final.cut.accepted!=138||final.cut.forwarded!=137||final.cut.exact_frontier!=137||
           final.cut.output_watermark!=9||final.accepted_tokens.size()!=138||final.pending_tokens.size()!=1||
           !final.approximation.empty()||before.cut.accepted!=137||before.cut.forwarded!=136||
           before.cut.exact_frontier!=136||before.cut.output_watermark!=8||
           before.accepted_tokens.size()!=137||before.pending_tokens.size()!=1||!before.approximation.empty())
            return refuse("actual catchup/final trace counts or watermark mismatch");
        auto preceding_key=final.key;
        if(!preceding_key.append_epoch)return refuse("actual trace append epoch underflow");
        --preceding_key.append_epoch;
        if(before.key!=preceding_key||before.cut.id>=final.cut.id||final.cut.id!=receipts_[8].cut||
           !std::equal(before.accepted_tokens.begin(),before.accepted_tokens.end(),final.accepted_tokens.begin()))
            return refuse("actual catchup/final trace identity or prefix mismatch");
        if(attachments_[0].publication!=2||attachments_[1].publication!=3||
           attachments_[0].current!=128||attachments_[1].current!=132||
           attachments_[0].foreground_before!=0||attachments_[1].foreground_before!=4||
           attachments_[0].cut>=receipts_[0].cut||receipts_[3].cut>=attachments_[1].cut||
           attachments_[1].cut>=receipts_[4].cut||receipts_[7].cut>=before.cut.id||before.cut.id>=receipts_[8].cut)
            return refuse("actual trace publication/cut boundaries mismatch");
        uint64_t previous_cut=attachments_[0].cut;
        for(size_t i=0;i<9;++i){const auto& receipt=receipts_[i];
            const uint64_t publication=i<4?2:i<8?3:4;
            if(receipt.operation!=i+1||receipt.publication!=publication||receipt.cut<=previous_cut||
               receipt.token<0||receipt.token!=final.accepted_tokens[size_t(initial_accepted_)+i])
                return refuse("actual receipt not bound to accepted output position");
            previous_cut=receipt.cut;
        }
        if(before.pending_tokens[0]!=before.accepted_tokens.back()||before.pending_tokens[0]!=receipts_[7].token||
           final.pending_tokens[0]!=final.accepted_tokens.back()||final.pending_tokens[0]!=receipts_[8].token)
            return refuse("actual trace pending token is not preceding receipt");
        receipt_pending_bound_=true;
        try {
            if(core::window_history_digest(before.accepted_tokens)!=before.cut.history||
               core::window_history_digest(final.accepted_tokens)!=final.cut.history)
                return refuse("actual trace canonical history mismatch");
        }catch(...){return refuse("actual trace history allocation refused");}
        receipt_history_bound_=true;receipt_trace_pass_=true;return true;
    }
    void report(FILE* output)const {
        std::fprintf(output,"suffix_repair_guards stale_refusals=%u historical_retries=%u unchanged_cuts_history_counters=%u scope=actual_controller_methods\n",stale_checks_,retry_checks_,unsigned(stale_checks_>0&&retry_checks_>0));
        if(cancelled_&&prepared_cancel_released_){
            std::fprintf(output,"suffix_repair_prepared_cancel prepared=1 candidate_released=1 restore_entered=0 pass=%u old_cut=%llu target_cut=%llu quote=%llu h_delta=%llu claim_delta=%u prepared_tensors=%llu prepared_bytes=%llu history_receipts_unchanged=1 cleanup_checked=%u global_released=1\n",
                unsigned(prepared_cancel_pass_),(unsigned long long)prepared_cancel_old_,(unsigned long long)prepared_cancel_target_,
                (unsigned long long)prepared_cancel_quote_,(unsigned long long)prepared_cancel_h_delta_,prepared_cancel_claim_delta_,(unsigned long long)prepared_cancel_tensors_,(unsigned long long)prepared_cancel_bytes_,unsigned(prepared_cancel_pass_));
        }
        if(cancelled_){
            for(unsigned i=0;i<partial_count_;++i){const auto& a=attachments_[i];
                std::fprintf(output,"suffix_repair_attachment index=%u publication=%llu previous_exact=%llu exact=%llu current_forwarded=%llu foreground_before=%llu cut=%llu roots=%zu\n",i+1,
                    (unsigned long long)a.publication,(unsigned long long)a.previous,(unsigned long long)a.exact,(unsigned long long)a.current,
                    (unsigned long long)a.foreground_before,(unsigned long long)a.cut,a.roots);}
            std::fprintf(output,"suffix_repair_cancelled terminal=1 private_retired=%u native_preserved=%u tensors=%llu bytes=%llu mismatches=%llu attachments=%u selected_publication=%llu cut=%llu current_forwarded=%llu exact_frontier=%llu pending=%zu output_watermark=%llu foreground=%llu no_second_or_final_publication=1\n",
                unsigned(retired_),unsigned(cancel_native_report_.completed&&cancel_native_report_.equal),
                (unsigned long long)cancel_native_report_.tensors_compared,(unsigned long long)cancel_native_report_.bytes_compared,
                (unsigned long long)cancel_native_report_.mismatches,partial_count_,(unsigned long long)published_->serial,(unsigned long long)live_.cut_.id,
                (unsigned long long)live_.cut_.forwarded,(unsigned long long)live_.cut_.exact_frontier,published_->metadata.pending_tokens.size(),
                (unsigned long long)live_.cut_.output_watermark,(unsigned long long)foreground_count_);
            for(size_t i=0;i<std::min<uint64_t>(cancel_native_report_.mismatches,cancel_native_report_.first.size());++i){const auto& m=cancel_native_report_.first[i];
                std::fprintf(output,"suffix_repair_cancel_difference reason=%u stage=%u layer=%lld kind=%u offset=%llu left=%llu right=%llu\n",
                    unsigned(m.reason),m.stage,(long long)m.layer,unsigned(m.kind),(unsigned long long)m.offset,
                    (unsigned long long)m.left,(unsigned long long)m.right);
            }
            for(size_t i=0;i<receipt_count_;++i){const auto& r=receipts_[i];std::fprintf(output,"suffix_repair_receipt operation=%llu publication=%llu cut=%llu token=%d\n",
                (unsigned long long)r.operation,(unsigned long long)r.publication,(unsigned long long)r.cut,r.token);}
            return; // NEVER print ordinary suffix_repair_result success on cancel
        }
        if(receipt_trace_checked_)std::fprintf(output,"suffix_repair_receipt_trace checked=1 pass=%u receipts=%zu initial_accepted=%llu catchup_forwarded=%llu final_forwarded=%llu pending_bound=%u history_bound=%u scope=actual_controller_history\n",
            unsigned(receipt_trace_pass_),receipt_count_,(unsigned long long)initial_accepted_,
            (unsigned long long)(catchup_snapshot_?catchup_snapshot_.get()->cut.forwarded:0),
            (unsigned long long)published_->metadata.cut.forwarded,unsigned(receipt_pending_bound_),unsigned(receipt_history_bound_));
        if(comparison_done_)std::fprintf(output,"suffix_repair_exact_pending compared=1 pass=%u forwarded=%llu vocab=%lld private_argmax=%d selected_argmax=%d rms=%.12g max_abs=%.12g kl_private_to_selected=%.12g oracle=private_resident_not_clean_replay\n",
            unsigned(comparison_pass_),(unsigned long long)comparison_forwarded_,(long long)runtime_->service_->owner_->primary_verifier->vocab(),
            comparison_.exact_argmax,comparison_.approximate_argmax,comparison_.root_mean_square_error,comparison_.max_absolute_error,comparison_.kl_exact_to_approximate);
        for(unsigned i=0;i<partial_count_;++i){const auto& a=attachments_[i];
            std::fprintf(output,"suffix_repair_attachment index=%u publication=%llu previous_exact=%llu exact=%llu current_forwarded=%llu foreground_before=%llu cut=%llu roots=%zu\n",i+1,
                (unsigned long long)a.publication,(unsigned long long)a.previous,(unsigned long long)a.exact,(unsigned long long)a.current,
                (unsigned long long)a.foreground_before,(unsigned long long)a.cut,a.roots);}
        std::fprintf(output,"suffix_repair_result attachments=%u foreground=%llu receipt_count=%zu final_exact=%u current_forwarded=%llu exact_frontier=%llu pending=%zu output_watermark=%llu scope=dedicated_serial_no_generic_manager\n",
            partial_count_,(unsigned long long)foreground_count_,receipt_count_,unsigned(finished_&&published_->metadata.cut.exact_frontier==published_->metadata.cut.forwarded),
            (unsigned long long)published_->metadata.cut.forwarded,(unsigned long long)published_->metadata.cut.exact_frontier,published_->metadata.pending_tokens.size(),
            (unsigned long long)published_->metadata.cut.output_watermark);
        for(size_t i=0;i<receipt_count_;++i){const auto& r=receipts_[i];std::fprintf(output,"suffix_repair_receipt operation=%llu publication=%llu cut=%llu token=%d\n",
            (unsigned long long)r.operation,(unsigned long long)r.publication,(unsigned long long)r.cut,r.token);}
    }
    // Fixed qualification witnesses only. Weak observations cannot authorize
    // execution or replace checked retirement. No aliasing/no-op owners.
    struct CycleOwners {
        std::weak_ptr<ResidentRepairController> controller;
        std::weak_ptr<ResidentRepairRuntime> runtime;
        std::weak_ptr<ResidentExpertService> service;
        std::weak_ptr<ResidentDispatchBinding> binding;
        std::shared_ptr<ResidentDispatchBinding> inspect_binding; // genuine temporary owner for post-retire fields
        std::weak_ptr<core::VerifierCompletionGate> gate;
        std::weak_ptr<const void> setup,native;
        std::array<std::weak_ptr<const void>,maximum_stages> stages{},ple_contexts{};
        std::array<std::uintptr_t,maximum_stages> addresses{};
        size_t count=0;
        void release_inspection()noexcept{inspect_binding.reset();}
        bool strong_owners_expired()const noexcept {
            if(inspect_binding||!controller.expired()||!runtime.expired()||!service.expired()||!binding.expired()||!gate.expired())return false;
            for(size_t i=0;i<count;++i)if(!stages[i].expired()||!ple_contexts[i].expired())return false;
            return true;
        }
        // Custom deleter/allocator controlblocks retain credits even after the
        // managed object dies. Drop THESE weak groups before testing credits.
        void drop_owner_controlblocks()noexcept {
            release_inspection();controller.reset();runtime.reset();service.reset();binding.reset();gate.reset();
            for(auto& owner:stages)owner.reset();
            for(auto& owner:ple_contexts)owner.reset();
        }
        bool charge_owners_expired()const noexcept{return setup.expired()&&native.expired();}
        void reset()noexcept{drop_owner_controlblocks();setup.reset();native.reset();addresses={};count=0;}
    };
    bool observe_cycle_owners(const std::shared_ptr<ResidentRepairController>& self,CycleOwners& out,std::string& error){
        if(self.get()!=this||!check_idle(published_->serial)||!runtime_->dispatch_||!runtime_->stage_count_||
           runtime_->stage_count_>maximum_stages||out.count){error="cycle owner observation phase";return false;}
        CycleOwners value;value.controller=self;value.runtime=runtime_;value.service=runtime_->service_;
        value.binding=runtime_->dispatch_;value.inspect_binding=runtime_->dispatch_;value.gate=runtime_->dispatch_->gate_;
        value.setup=runtime_->setup_credit_;value.native=runtime_->allocation_.reservation();
        value.count=runtime_->stage_count_;
        for(size_t i=0;i<value.count;++i){
            const auto& stage=runtime_->stages_[i];
            if(!stage.verifier||!stage.session_owner){error="cycle private stage owner missing";return false;}
            value.stages[i]=stage.session_owner;
            if(stage.session&&stage.session->ple.read_context)value.ple_contexts[i]=stage.session->ple.read_context;
            value.addresses[i]=reinterpret_cast<std::uintptr_t>(stage.verifier.get());
            if(!core::Verifier::diagnostic_contains_address(value.addresses[i])){
                error="cycle private verifier was not actually registered";return false;
            }
        }
        out=std::move(value);return true;
    }
    bool observe_retired_cycle(const CycleOwners& owners,std::string& error)const {
        if(!retired_||cancelled_||busy_||private_||!runtime_||runtime_->phase_!=RuntimePhase::retired||
           runtime_->stage_count_||runtime_->dispatch_||!runtime_->allocation_.stages().empty()||
           runtime_->service_->active_||runtime_->service_->unhealthy_||!owners.count){
            error="actual private retirement state not terminal";return false;
        }
        auto binding=owners.binding.lock();auto gate=owners.gate.lock();
        if(!binding||!gate||!binding->private_sealed_||binding->callback_depth_||gate->healthy_advisory()){
            error="actual retired dispatch/gate still admits work";return false;
        }
        for(size_t i=0;i<owners.count;++i){const auto& stage=runtime_->stages_[i];
            if(stage.verifier||stage.session||stage.session_owner||stage.stream||
               core::Verifier::diagnostic_contains_address(owners.addresses[i])){
                error="private stage/registry survived actual retirement";return false;
            }
        }
        if(core::native_session_quarantine_count()||core::window_buffer_quarantine_count()||
           core::model_stage_quarantine_count()||core::model_gang_quarantine_count()){
            error="cycle actual retirement quarantined resources";return false;
        }
        return true;
    }
    bool retire(std::string& error){
        if(retired_&&cancelled_&&!busy_)return true; // terminal idempotent cancellation cleanup only
        if(busy_||retired_||!runtime_->dispatch_){error="resident controller retirement phase";return false;}
        // All RAW/read/capture calls are synchronous and owned by this same
        // private controller. Neither helper is running/capturing/restoring.
        if(private_->phase_!=SuffixRawTransaction::Phase::drained||live_.phase_!=SuffixRawTransaction::Phase::drained)fatal("resident RAW work alive at retirement");
        if(!runtime_->dispatch_->retire_initialized(error))return false;
        private_.reset();retired_=true;return true;
    }
    struct ExactCompletion {
        core::StateCut cut;
        core::VerifierCompletionGate::Cut receipt;
        ExactCompletion(core::StateCut c,core::VerifierCompletionGate::Cut&& w):cut(c),receipt(std::move(w)){}
    };
    std::shared_ptr<ResidentRepairRuntime> runtime_;
    std::shared_ptr<SuffixRawTransaction> live_owner_; // actual owner, never no-op alias
    SuffixRawTransaction& live_;
    std::unique_ptr<SuffixRawTransaction> private_;
    std::shared_ptr<Published> published_;
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
    std::unique_ptr<CohortPending> cohort_pending_;
#endif
    core::BundleHandle catchup_snapshot_; // installed LIVE pre-ninth image + real global/shared-H credit
    uint64_t initial_accepted_=0,initial_output_watermark_=0;
    bool repair_receipts_pass_=false;
    bool receipt_trace_checked_=false,receipt_trace_pass_=false,receipt_pending_bound_=false,receipt_history_bound_=false;
    std::vector<int32_t> private_history_;
    core::WindowKey private_key_;
    std::optional<ExactCompletion> exact_completion_;
    struct Attachment {uint64_t publication=0,previous=0,exact=0,current=0,foreground_before=0,cut=0;size_t roots=0;};
    std::array<Attachment,2> attachments_{};
    SuffixLogitComparison comparison_{};
    core::StateDigest comparison_history_{};
    uint64_t comparison_forwarded_=0;
    bool comparison_done_=false,comparison_pass_=false;
    Receipt last_{};
    KnownReceipt last_known_{};
    ReceiptHistory receipts_;
    size_t receipt_count_=0;
    uint64_t next_cut_=1,foreground_count_=0,foreground_at_attachment_=0,known_count_=0;
    unsigned partial_count_=0;
    unsigned stale_checks_=0,retry_checks_=0;
    clean_recompute_qualification::Report cancel_native_report_{};
    bool fault_second_copy_=false,fault_deferred_trap_=false;int fault_receipt_fd_=-1;
#if defined(STRATA_SERVING_FAULT_QUALIFICATION_AVAILABLE)
    core::resident_serving_fault::Binding serving_fault_{};bool serving_fault_fired_=false;
#endif
    uint64_t prepared_cancel_quote_=0,prepared_cancel_target_=0,prepared_cancel_old_=0,prepared_cancel_h_delta_=0;
    uint64_t prepared_cancel_tensors_=0,prepared_cancel_bytes_=0;
    unsigned prepared_cancel_claim_delta_=0;
    bool prepared_cancel_released_=false,prepared_cancel_pass_=false;
    bool cancel_requested_=false,cancelled_=false;
    bool busy_=false,retired_=false,finished_=false;
    // The retained-owner state (slice 2). Set only by `yield_retained`, cleared only by `resume_retained`; while
    // it is set every `check_idle`-guarded entry point refuses, so a parked owner can run bounded private work
    // and be retired deliberately, and can never publish into a live session it no longer stands on.
    bool suspended_=false;
#if defined(STRATA_WINDOW_STATE_AVAILABLE)
    // The controller's own mirror of the versioned owner record (slice 5). Established at the certified yield,
    // bumped whenever the private chain moves, and the slot the published retained result sits in. The registry
    // holds the authoritative copy; the controller's is what its `publish_retained_if_current` gate checks, so a
    // caller cannot present a result the controller does not currently stand on. Both types come from
    // serving_repair_owners.hpp, which is included under the same guard.
    RetainedRecordFacts retained_record_{};
    PreparedRetainedResult retained_result_{};
#endif
};
} // namespace strata::program::resident_repair_detail
