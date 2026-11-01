// Experimental dedicated resident repair; integration and hardware qualification pending.
// ISOLATED SOURCE PROPOSAL. Include in generate.cpp after actual
// DriverWindowRuntime, GpuStage, Drive, SplitDrive and lifecycle definitions.
// Not compiled/integrated. No existing dispatch ownership predicate is changed.
#include "strata/core/window_allocator.hpp"
#include "strata/core/window_buffer.hpp"
#include "strata/core/window_gpu_handles.hpp"
#include "strata/core/window_model_owner.hpp"
#include "strata/core/strict_retirement.hpp"
#include <array>
#include <memory>
#include <thread>
#include <span>

namespace strata::program::resident_repair_detail {
constexpr size_t maximum_stages=8;
enum class FactoryStatus { invalid_binding, admission_wiring_required, prepared };
enum class RuntimePhase { host_only, sessions_only, chain_prepared, constructing, ready, running, drained, retiring, poisoned, retired };
struct FactoryResult {
    FactoryStatus status=FactoryStatus::invalid_binding;
    const char* reason="invalid resident factory binding";
    explicit operator bool()const noexcept{return status==FactoryStatus::prepared;}
};
// Non-owning observations are usable only while retained service+model live.
// This is NOT a fabricated GpuStage: no owning cache/model object is copied.
struct ServiceStageView {
    int device=-1;
    int64_t begin=0,end=0;
    const core::OwnedModelStage* weights=nullptr;
    core::ExpertCache* cache=nullptr;
};
class ResidentRepairRuntime;
class ResidentDispatchBinding;
class ResidentRepairController;
class ResidentExpertService final {
public:
    ResidentExpertService(const ResidentExpertService&)=delete;
    ResidentExpertService& operator=(const ResidentExpertService&)=delete;
private:
    friend class ResidentRepairRuntime;
    friend class ResidentDispatchBinding;
    friend class ResidentRepairController;
    explicit ResidentExpertService(std::shared_ptr<::DriverWindowRuntime> owner)
        :owner_(std::move(owner)),geometry_owner_(owner_->geometry_owner),model_(owner_->model),controller_(std::this_thread::get_id()){}
    // owner_ comes FIRST and releases LAST; it retains source/pool/caches,
    // background workers and legacy service charges through all private users.
    std::shared_ptr<::DriverWindowRuntime> owner_;
    // Genuine stable typed geometry owner, not a no-op alias to stack storage.
    // Both live and private Verifiers bind this actual object address.
    std::shared_ptr<const core::ModelGeometry> geometry_owner_;
    std::shared_ptr<const core::OwnedModelGang> model_;
    const std::thread::id controller_;
    std::array<ServiceStageView,maximum_stages> stages_{};
    size_t count_=0;
    const void* active_=nullptr; // actual exclusive controller operation
    uint64_t operation_=0; // future actual checked operation scope, never caller bool
    bool unhealthy_=false;
    std::shared_ptr<const core::NativeBatchSlotOccupancy> live_slot_;
    core::NativeSessionAllocation live_slot_allocation_;
    std::array<core::NativeBatchSlotInspection,maximum_stages> live_slot_bindings_{};
    bool slot_private_allocation_disjoint(const core::NativeSessionAllocation& candidate)const noexcept {
        if(!live_slot_)return true; // existing serial checks remain unchanged
        if(!live_slot_valid()||!candidate.healthy()||candidate.stages().size()!=count_)return false;
        for(const auto& native:candidate.stages()){
            if(!native.owner||!native.session)return false;
            for(const auto& main:owner_->sessions.stages())
                if(native.session==main.session||(!native.owner.owner_before(main.owner)&&!main.owner.owner_before(native.owner)))return false;
            for(size_t slot=0;slot<2;++slot){
                const auto* physical=owner_->batch_slot_sessions.allocation(owner_->batch_slot_sessions.handle(slot));
                if(!physical)return false;
                for(const auto& foreground:physical->stages())
                    if(native.session==foreground.session||(!native.owner.owner_before(foreground.owner)&&!foreground.owner.owner_before(native.owner)))return false;
            }
        }
        return true;
    }
    bool live_slot_valid()const noexcept {
        if(!live_slot_)return true;
        if(!owner_||!geometry_owner_||geometry_owner_.get()!=owner_->geometry_owner.get()||
           !live_slot_->active()||!live_slot_allocation_.healthy()||!count_||count_>maximum_stages||
           live_slot_allocation_.stages().size()!=count_||owner_->stages.size()+1!=count_||
           !owner_->primary_verifier||!owner_->batch_slot_sessions.published()||!owner_->batch_slot_bindings||
           owner_->batch_slot_bindings->geometry()!=geometry_owner_.get()||live_slot_->slot()>=2)return false;
        const auto handle=owner_->batch_slot_sessions.handle(live_slot_->slot());
        const auto* actual=owner_->batch_slot_sessions.allocation(handle);
        const core::NativeBatchSlotToken token{handle.incarnation,handle.generation,handle.slot};
        if(!actual||actual->stages().size()!=count_||owner_->batch_slot_bindings->token(live_slot_->slot())!=token)return false;
        const auto& retained=live_slot_allocation_.reservation();const auto& parent=actual->reservation();
        if(!retained||!parent||retained.get()!=parent.get()||retained.owner_before(parent)||parent.owner_before(retained))return false;
        for(size_t stage=0;stage<count_;++stage){
            if(stage&&!owner_->stages[stage-1])return false;
            const auto& native=live_slot_allocation_.stages()[stage];const auto& expected=actual->stages()[stage];
            const auto* verifier=stage?&owner_->stages[stage-1]->ver:owner_->primary_verifier.get();
            if(native.session!=expected.session||native.device!=expected.device||!native.owner||!expected.owner||
               native.owner.owner_before(expected.owner)||expected.owner.owner_before(native.owner)||
               live_slot_bindings_[stage].token()!=token||live_slot_bindings_[stage].stage_index()!=stage||
               live_slot_bindings_[stage].geometry()!=geometry_owner_.get()||
               !verifier->owned_batch_slot_matches(live_slot_bindings_[stage])||
               !live_slot_bindings_[stage].matches_session(native.session,native.device))return false;
        }
        return true;
    }
};
struct RepairStageExecution final {
    // Genuine session owner, not aliasing shared_ptr with a no-op deleter.
    // Released after verifier and scratch consumers (reverse member order).
    std::shared_ptr<const void> session_owner;
    ServiceStageView service;
    core::SessionState* session=nullptr;
    core::OwnedWindowGpuHandles handles{16,8};
    cudaStream_t stream=nullptr; // handle must belong to handles, never raw create
    std::array<core::OwnedWindowBuffer,4> retained_buffers{};
    // FRESH Verifier, never copied from service->primary_verifier/GpuStage.ver.
    std::unique_ptr<core::Verifier> verifier;
    RepairStageExecution()=default;
    RepairStageExecution(const RepairStageExecution&)=delete;
    RepairStageExecution& operator=(const RepairStageExecution&)=delete;
};

class ResidentRepairRuntime final {
public:
    ResidentRepairRuntime(const ResidentRepairRuntime&)=delete;
    ResidentRepairRuntime& operator=(const ResidentRepairRuntime&)=delete;
    ~ResidentRepairRuntime()noexcept {
        // Do not silently unwind live or uncertain resources, or call shared
        // service close while another gang still uses it. Controller must retire.
        if(phase_!=RuntimePhase::host_only && phase_!=RuntimePhase::retired)
            core::retirement_failed("resident repair owner dropped before checked retirement");
    }
    // Read-only real-source preflight, deliberately NOT a public caller-supplied
    // joined/admitted bool. Until actual admission wiring exists no runtime is
    // returned and no sessions/streams/graphs/verifiers are allocated/initialized.
    static FactoryResult prepare(const std::shared_ptr<::DriverWindowRuntime>& service,
        const core::ModelGeometry& geometry,std::unique_ptr<ResidentRepairRuntime>& out) noexcept {
        (void)out; // unchanged on EVERY refusal; no partial owner escapes
        if(!service||!service->geometry_owner||service->geometry_owner.get()!=&geometry||!service->model||!service->model->healthy()||
           !service->sessions.healthy()||!service->source||!service->pool||!service->cache||
           !service->drive||!service->owns_expert_dispatch(&service->drive->d)||
           service->background_retirement.load(std::memory_order_acquire)!=0)
            return {FactoryStatus::invalid_binding,"requires actual live owned file-source service"};
        const auto& allocated=service->sessions.stages();
        const auto& model=service->model->stages();
        if(allocated.empty()||allocated.size()>maximum_stages||
           model.size()!=allocated.size()||service->stages.size()+1!=allocated.size())
            return {FactoryStatus::invalid_binding,"actual complete physical stage inventory"};
        if(service->drive->d.failed||service->drive->d.remote_count||service->remote_opt||
           service->drive->d.lookahead||service->foresight||service->adapt_gate)
            return {FactoryStatus::invalid_binding,"unsupported active service writer/optimizer"};
        int64_t end=0;
        for(size_t i=0;i<allocated.size();++i){const auto& stage=allocated[i];
            if(!stage.owner||!stage.session||!model[i]||!model[i]->healthy()||
               stage.device!=model[i]->device()||stage.session->layer_lo!=end||
               stage.session->layer_hi<=end||stage.session->layer_hi>geometry.n_layers)
                return {FactoryStatus::invalid_binding,"stage owner/model/carve mismatch"};
            if(i&&(!service->stages[i-1]||&service->stages[i-1]->ss!=stage.session||
                   service->stages[i-1]->dev!=stage.device))
                return {FactoryStatus::invalid_binding,"secondary service cache owner mismatch"};
            end=stage.session->layer_hi;
        }
        if(end!=geometry.n_layers)
            return {FactoryStatus::invalid_binding,"native layer partition incomplete"};
        // Intentionally executable refusal, NOT a placeholder success branch.
        // set_checked_requested_layout(caps) alone will NOT remove this gate.
        return {FactoryStatus::admission_wiring_required,
            "missing actual global+aggregate setup/cold-state/graph admission and serialized service operation binding"};
    }

    // Owner controller only. No external work may exist in sessions_only; this
    // phase permits safe rollback after session allocation but before bind/init.
    // A ready/running/drained runtime needs the future concrete operation join,
    // so it is refused here BEFORE marking/freeing anything.
    bool retire_unbound_sessions(std::string& error) {
        if(std::this_thread::get_id()!=controller_ || phase_!=RuntimePhase::sessions_only){
            error="retirement requires actual unbound sessions-only factory phase";return false;
        }
        core::NativeRetirementAttempt attempt;
        if(!attempt.prepare(allocation_,std::span<const int>{},error)){
            allocation_.mark_unhealthy();phase_=RuntimePhase::poisoned;return false;
        }
        if(!attempt.drain_and_commit(error)){
            allocation_.mark_unhealthy();phase_=RuntimePhase::poisoned;return false;
        }
        // No verifier or other consumer has been initialized in this phase.
        // Drop stage aliases only after terminal native retirement succeeds.
        for(auto& stage:stages_){stage.session=nullptr;stage.session_owner.reset();}
        allocation_={};phase_=RuntimePhase::retired;return true;
    }
private:
    friend class ResidentDispatchBinding;
    friend class ResidentRepairController;
    ResidentRepairRuntime()=default;
    // Actual GLOBAL+aggregate setup owner must be retained FIRST/released LAST.
    // Empty until genuine controller admission implementation constructs owner.
    std::shared_ptr<const void> setup_credit_;
    std::shared_ptr<ResidentExpertService> service_;
    std::shared_ptr<const core::OwnedModelGang> model_;
    core::NativeSessionAllocation allocation_;
    const std::thread::id controller_=std::this_thread::get_id();
    RuntimePhase phase_=RuntimePhase::host_only;
    core::StateCut cut_{};
    uint64_t operation_=0;
    // Fresh dispatch receiver has independent vectors, flags, jobs, plan and
    // counters. It borrows src/pool/cache only via retained service_ under gate.
    std::array<RepairStageExecution,maximum_stages> stages_{};
    std::array<core::OwnedWindowBuffer,maximum_stages-1> handoffs_{};
    size_t stage_count_=0;
    std::shared_ptr<ResidentDispatchBinding> dispatch_; // destroyed before targets
};
// Observe actual live bindings before serving installs a splice. Never close or mutate a producer here.
inline const char* resident_service_refusal(::DriverWindowRuntime& service) noexcept {
    if(!service.drive||!service.geometry_owner||!service.primary_verifier||!service.sessions.healthy()||
       service.background_retirement.load(std::memory_order_acquire)!=0)
        return "suffix service has no idle owned runtime";
    auto& source=service.drive->d;const auto& g=*service.geometry_owner;
    if(!service.owns_expert_dispatch(&source)||source.failed||!source.host_res)
        return "suffix service dispatch is unhealthy or unbound";
    if(source.remote_count||service.remote_opt||source.peer)
        return "suffix repair does not support active remote/peer dispatch";
    if(source.lookahead||service.foresight||source.fs)
        return "suffix repair requires lookahead and foresight writers disabled";
    if((service.adapt_gate&&service.adapt_gate->active.load(std::memory_order_acquire))||
       source.pcie_model.on||source.host_execution_observer||!source.usage.empty()||
       source.evict_sync||source.res_dirty||source.dres_check)
        return "suffix repair requires fixed serial dispatch without adaptive or observer writers";
    if(source.n_expert!=g.n_expert||source.cache_blob<=0)
        return "suffix repair fixed dispatch dimensions differ";
    const auto& stages=service.sessions.stages();
    if(stages.empty()||stages.size()>maximum_stages||service.stages.size()+1!=stages.size()||
       (stages.size()>1&&(!service.split_drive||service.split_drive->n!=int(stages.size()))))
        return "suffix repair live serial stage inventory differs";
    for(const auto& stage:service.stages)if(!stage)return "suffix repair live stage owner missing";
    for(size_t i=0;i<stages.size();++i){
        auto* verifier=i?&service.stages[i-1]->ver:service.primary_verifier.get();
        const auto binding=verifier->serial_routing_binding();
        auto* next=i+1<stages.size()?&service.stages[i]->ver:nullptr;
        if(verifier->has_adapt_gate()||binding.optimizer||binding.session!=stages[i].session||binding.geometry!=&g||binding.next!=next||
           binding.next_user!=(next?static_cast<const void*>(service.split_drive.get()):nullptr))
            return "suffix repair live verifier routing/optimizer differs";
    }
    return nullptr;
}
} // namespace strata::program::resident_repair_detail
