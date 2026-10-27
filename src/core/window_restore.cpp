#include "strata/core/window_restore.hpp"
#include "strata/core/window_allocator.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/window_device.hpp"
#include "strata/core/window_lease.hpp"
#include <cuda_runtime.h>
#include <new>
#include <set>
#include <map>
#include <atomic>
#include <optional>

namespace strata::core {
struct NativeRestoreStorage {
    // Owners/accounting outlive host staging and all destination stage references.
    std::shared_ptr<const ReservationLease> destination_budget, host_budget;
    BundleHandle source;
    struct Initial { WindowKey key; StateCut cut{}; };
    std::optional<Initial> initial;
    std::vector<NativePrivateStage> stages;
    // Destroy bound consumers before releasing destination session owners.
    std::shared_ptr<NativeRuntimeAttachment> runtime_attachment;
    SavedConversation native;
    std::shared_ptr<NativeRestoreStorage> pending_owner;
    std::atomic<bool> selected{false};
};
struct NativeRestoreAccess {
    static void prepare(PreparedNativeWindow& out,std::shared_ptr<NativeRestoreStorage> storage) { out.storage_=std::move(storage); }
    static void attach(NativeRestoreFailure& out,std::shared_ptr<NativeRestoreStorage> storage) { out.storage_=std::move(storage); }
};
const SnapshotDescriptor* PreparedNativeWindow::descriptor() const { return storage_ ? storage_->source.get() : nullptr; }
std::shared_ptr<WindowExecution> PreparedNativeWindow::execution() const {
    if(!storage_) return {};
    if(storage_->initial)
        return std::make_shared<WindowExecution>(storage_->initial->key,storage_->initial->cut,uint32_t(storage_->stages.size()),
                                                 storage_,storage_->destination_budget);
    const auto& descriptor=*storage_->source.get();
    return std::make_shared<WindowExecution>(descriptor.key,descriptor.cut,uint32_t(storage_->stages.size()),
                                             storage_,storage_->destination_budget);
}
std::vector<NativeSessionView> PreparedNativeWindow::session_views() const {
    std::vector<NativeSessionView> views;
    if(storage_) for(const auto& stage:storage_->stages) views.push_back({stage.device,stage.session});
    return views;
}
std::shared_ptr<NativeRuntimeAttachment> PreparedNativeWindow::runtime_attachment() const {
    return storage_ ? storage_->runtime_attachment : nullptr;
}
bool PreparedNativeWindow::claim_selection() const {
    if(!storage_) return false;
    bool expected=false;
    return storage_->selected.compare_exchange_strong(expected,true);
}
void PreparedNativeWindow::release_selection_claim() const { storage_->selected.store(false); }
void NativeRestoreFailure::retire_after_teardown() {
    std::lock_guard lock(mutex_);
    if(storage_) { storage_->pending_owner.reset();storage_.reset(); }
}
namespace {
NativeRestoreResult invalid(const std::string& why) { return {NativeRestoreStatus::invalid,why}; }
bool claim_destination(const std::shared_ptr<NativeRestoreStorage>& storage) {
    // A reservation/session already held by a prepared or failed transaction is
    // not private for another restore. The allocator additionally guarantees that
    // different SessionState objects cannot conceal overlapping allocation ranges.
    static std::mutex mutex;
    static std::map<const ReservationLease*,std::weak_ptr<NativeRestoreStorage>> leases;
    static std::map<const SessionState*,std::weak_ptr<NativeRestoreStorage>> sessions;
    std::lock_guard lock(mutex);
    for(auto it=leases.begin();it!=leases.end();) it=it->second.expired()?leases.erase(it):std::next(it);
    for(auto it=sessions.begin();it!=sessions.end();) it=it->second.expired()?sessions.erase(it):std::next(it);
    if(leases.contains(storage->destination_budget.get())) return false;
    for(const auto& stage:storage->stages) if(sessions.contains(stage.session)) return false;
    leases.emplace(storage->destination_budget.get(),storage);
    for(const auto& stage:storage->stages) sessions.emplace(stage.session,storage);
    return true;
}
}
NativeRestoreResult prepare_native_restore(const BundleHandle& source,const WindowKey& expected,
    const ModelGeometry& g,std::vector<NativePrivateStage> stages,
    std::shared_ptr<const ReservationLease> destination_budget,WindowBudget& host_budget,
    uint64_t max_tokens,uint64_t max_bytes,bool steering,PreparedNativeWindow& out,
    std::shared_ptr<NativeRuntimeAttachment> runtime_attachment) {
    if(!source || source.get()->key!=expected || !destination_budget ||
       destination_budget->purpose()!=ReservationPurpose::private_work || stages.empty() ||
       stages.size()>size_t(g.n_layers)) return invalid("restore requires immutable source and reserved private destinations");
    try {
        std::vector<NativeWindowStage> config;
        std::set<const SessionState*> unique;
        const auto& allocated=destination_budget->resources();
        for(const auto& stage:stages) {
            if(!stage.session || !stage.owner || !stage.bind_private || stage.device<0 ||
               size_t(stage.device)>=allocated.device_bytes.size() || !allocated.device_bytes[size_t(stage.device)] ||
               !unique.insert(stage.session).second) return invalid("invalid or repeated private restore stage");
            const auto& s=*stage.session;
            if(!s.qsa_states || s.qsa_alloc<=0 || s.qsa_alloc>g.n_qsa_layers() ||
               s.qsa_ord0<0 || s.qsa_ord0>g.n_qsa_layers()-s.qsa_alloc) return invalid("invalid destination QSA allocation");
            NativeWindowStage resolved{s.layer_lo,s.layer_hi,{},s.ple_hist!=nullptr};
            for(int64_t q=0;q<s.qsa_alloc;++q) {
                const auto& state=s.qsa_states[s.qsa_ord0+q];
                resolved.qsa_formats.push_back((state.kv_hybrid ? 3:qsa_kv_format(state))+
                    (state.kv_rot && !state.kv_q4 && !state.kv_hybrid ? 16:0));
            }
            config.push_back(std::move(resolved));
        }
        WindowStateLayout layout;
        auto checked=native_window_layout(g,config,expected,source.get()->cut,max_tokens,max_bytes,layout,steering);
        if(!checked) return invalid(checked.detail);
        checked=validate_window_state(*source.get(),expected,layout);
        if(!checked) return invalid(checked.detail);
        if(max_tokens>UINT64_MAX/128 || max_bytes>(UINT64_MAX-max_tokens*128)/2)
            return invalid("restore staging budget overflow");
        WindowResources resources;
        resources.device_bytes.resize(host_budget.snapshot().capacity.device_bytes.size());
        resources.host_bytes=max_bytes*2+max_tokens*128;
        const uint64_t directory_headroom=uint64_t(stages.size())*1024*1024;
        if(directory_headroom>UINT64_MAX-resources.host_bytes) return invalid("restore metadata budget overflow");
        resources.host_bytes+=directory_headroom;
        auto host=host_budget.reserve(resources,ReservationPurpose::foreground);
        if(!host) return {NativeRestoreStatus::budget_refused,"restore host staging capacity unavailable"};
        auto storage=std::make_shared<NativeRestoreStorage>();
        storage->destination_budget=std::move(destination_budget);storage->host_budget=host.lease;
        storage->source=source;storage->stages=std::move(stages);
        storage->runtime_attachment=std::move(runtime_attachment);
        if(!claim_destination(storage)) return invalid("private destination already belongs to another transaction");
        std::vector<NativePleTokens> tokens;
        checked=materialize_native_window(*source.get(),expected,g,config,max_tokens,max_bytes,steering,storage->native,tokens);
        if(!checked) return invalid(checked.detail);
        std::string error;
        // Validate the entire gang before any transfer or binding callback.
        for(size_t i=0;i<storage->stages.size();++i) {
            const auto& native=i ? storage->native.stage_images[i-1]:storage->native;
            if(!conversation_snapshot_validate(native,*storage->stages[i].session,g,static_cast<const QsaState*>(nullptr),error))
                return invalid(error);
        }
        auto failure=std::make_shared<NativeRestoreFailure>();NativeRestoreAccess::attach(*failure,storage);
        storage->pending_owner=storage;
        try {
            for(size_t i=0;i<storage->stages.size();++i) {
                auto& stage=storage->stages[i];WindowDeviceScope device(stage.device);
                if(!device.ok())
                    return {NativeRestoreStatus::unhealthy,"restore device selection failed",failure};
                const auto& native=i ? storage->native.stage_images[i-1]:storage->native;
                if(conversation_snapshot_restore(native,*stage.session,g,static_cast<const QsaState*>(nullptr),error)!=ConversationRestore::restored)
                    return {NativeRestoreStatus::unhealthy,error,failure};
                stage.session->ple_prev[0]=tokens[i].previous[0];stage.session->ple_prev[1]=tokens[i].previous[1];
                stage.session->ple_token=tokens[i].current;
                if(!stage.bind_private(error)) return {NativeRestoreStatus::unhealthy,error,failure};
                const auto status=cudaDeviceSynchronize();
                if(status!=cudaSuccess) return {NativeRestoreStatus::unhealthy,cudaGetErrorString(status),failure};
                if(!device.finish()) return {NativeRestoreStatus::unhealthy,"restore could not restore caller device",failure};
            }
        } catch(...) { return {NativeRestoreStatus::unhealthy,"exception during private restore; teardown required",failure}; }
        storage->pending_owner.reset();
        NativeRestoreAccess::prepare(out,std::move(storage));
        return {NativeRestoreStatus::prepared,{}};
    } catch(const std::bad_alloc&) { return {NativeRestoreStatus::budget_refused,"restore allocation failed"}; }
}
NativeRestoreResult prepare_owned_native_restore(const BundleHandle& source,const WindowKey& expected,
    const ModelGeometry& g,const NativeSessionAllocation& allocation,const std::vector<NativeSessionBinder>& binders,
    WindowBudget& budget,uint64_t max_tokens,uint64_t max_bytes,bool steering,PreparedNativeWindow& out,
    std::shared_ptr<NativeRuntimeAttachment> runtime_attachment) {
    if(!allocation.reservation()||allocation.stages().empty()||binders.size()!=allocation.stages().size())
        return invalid("owned restore requires factory allocations and one private binder per stage");
    if(!allocation.healthy())return {NativeRestoreStatus::unhealthy,"owned restore allocation is unhealthy; worker restart required"};
    try {
        std::vector<NativePrivateStage> stages;stages.reserve(allocation.stages().size());
        for(size_t i=0;i<allocation.stages().size();++i) {
            const auto& stage=allocation.stages()[i];
            if(!binders[i])return invalid("owned restore requires all private binders before transfer");
            stages.push_back({stage.device,stage.session,stage.owner,
                [session=stage.session,binder=binders[i]](std::string& error){return binder(*session,error);}});
        }
        auto result=prepare_native_restore(source,expected,g,std::move(stages),allocation.reservation(),budget,
                                           max_tokens,max_bytes,steering,out,std::move(runtime_attachment));
        if(result.status==NativeRestoreStatus::unhealthy)allocation.mark_unhealthy();
        return result;
    } catch(const std::bad_alloc&) { return {NativeRestoreStatus::budget_refused,"owned restore metadata allocation failed"}; }
}
NativeRestoreResult prepare_owned_native_initial(const WindowKey& key,const ModelGeometry& g,
    const NativeSessionAllocation& allocation,const std::vector<NativeSessionBinder>& binders,
    WindowBudget& budget,uint64_t max_tokens,uint64_t max_bytes,bool steering,PreparedNativeWindow& out,
    std::shared_ptr<NativeRuntimeAttachment> runtime_attachment) {
#if defined(STRATA_GPU_METAL)
    // session_zero currently resets the global Metal window, not this private
    // session. Refuse before allocation/device callbacks until that API exists.
    return invalid("private initial windows require a per-session Metal reset; unsupported");
#endif
    const auto& owned=allocation.stages();
    if(!key.engine_generation||!key.window_generation||key.session_id.empty()||key.thread_id.empty()||key.window_id.empty()||
       !allocation.reservation()||allocation.reservation()->purpose()!=ReservationPurpose::private_work||
       owned.empty()||owned.size()>64||owned.size()>size_t(g.n_layers)||binders.size()!=owned.size())
        return invalid("initial window requires identity, factory destinations and one binder per stage");
    if(!allocation.healthy())return {NativeRestoreStatus::unhealthy,"initial allocation is unhealthy; worker restart required"};
    try {
        // Same bounded metadata headroom used by restore; no payload is copied or
        // advertised here. Runtime allocations are reserved by the private binder.
        uint64_t bytes=sizeof(NativeRestoreStorage)+uint64_t(owned.size())*1024*1024;
        for(const auto* text:{&key.session_id,&key.thread_id,&key.window_id}) {
            if(text->size()>max_bytes||text->size()>UINT64_MAX-bytes)return invalid("initial identity metadata exceeds bound");
            bytes+=text->size();
        }
        WindowResources resources;resources.device_bytes.resize(budget.snapshot().capacity.device_bytes.size());
        resources.host_bytes=bytes;
        auto host=budget.reserve(resources,ReservationPurpose::foreground);
        if(!host)return {NativeRestoreStatus::budget_refused,"initial metadata capacity unavailable"};
        auto storage=std::make_shared<NativeRestoreStorage>();
        storage->host_budget=host.lease;storage->destination_budget=allocation.reservation();
        // First successful cut in this fresh window generation: empty token
        // counts are zero, but cut identity and empty-history digest are real.
        StateCut initial_cut;initial_cut.id=1;initial_cut.history=window_history_digest({});
        storage->initial=NativeRestoreStorage::Initial{key,initial_cut};
        storage->runtime_attachment=std::move(runtime_attachment);
        storage->stages.reserve(owned.size());
        std::vector<NativeWindowStage> config;config.reserve(owned.size());
        std::set<const SessionState*> unique;std::set<int> devices;
        const auto& allocated=allocation.reservation()->resources();
        for(size_t i=0;i<owned.size();++i) {
            const auto& stage=owned[i];
            if(!stage.session||!stage.owner||!binders[i]||stage.device<0||
               size_t(stage.device)>=allocated.device_bytes.size()||!allocated.device_bytes[size_t(stage.device)]||
               !unique.insert(stage.session).second||!devices.insert(stage.device).second)
                return invalid("invalid or repeated initial factory stage/binder");
            const auto& session=*stage.session;
            if(!session.qsa_states||session.qsa_alloc<=0||session.qsa_alloc>g.n_qsa_layers()||
               session.qsa_ord0<0||session.qsa_ord0>g.n_qsa_layers()-session.qsa_alloc)
                return invalid("invalid initial QSA allocation");
            NativeWindowStage resolved{session.layer_lo,session.layer_hi,{},session.ple_hist!=nullptr};
            for(int64_t q=0;q<session.qsa_alloc;++q) {
                const auto& state=session.qsa_states[session.qsa_ord0+q];
                resolved.qsa_formats.push_back((state.kv_hybrid?3:qsa_kv_format(state))+
                    (state.kv_rot&&!state.kv_q4&&!state.kv_hybrid?16:0));
            }
            config.push_back(std::move(resolved));
            storage->stages.push_back({stage.device,stage.session,stage.owner,
                [session=stage.session,binder=binders[i]](std::string& error){return binder(*session,error);}});
        }
        WindowStateLayout layout;
        const auto checked=native_window_layout(g,config,key,initial_cut,max_tokens,max_bytes,layout,steering);
        if(!checked)return invalid(checked.detail);
        if(!claim_destination(storage))return invalid("initial destination already belongs to another transaction");
        auto failure=std::make_shared<NativeRestoreFailure>();NativeRestoreAccess::attach(*failure,storage);
        NativeRestoreResult fallback{NativeRestoreStatus::unhealthy,"initial zero/bind failed; teardown required",failure};
        storage->pending_owner=storage;
        auto unhealthy=[&](const std::string& why) {
            allocation.mark_unhealthy();
            try { return NativeRestoreResult{NativeRestoreStatus::unhealthy,why,failure}; }
            catch(...) { return std::move(fallback); }
        };
        try {
            for(auto& stage:storage->stages) {
                WindowDeviceScope device(stage.device);
                if(!device.ok())return unhealthy("initial device selection failed");
                // Do not erase an unrelated preexisting CUDA error as success.
                auto status=cudaGetLastError();
                if(status!=cudaSuccess)return unhealthy(cudaGetErrorString(status));
                std::string error;
                if(!session_zero_checked(*stage.session,g,nullptr,nullptr,error))return unhealthy(error);
                status=cudaGetLastError();
                if(status!=cudaSuccess)return unhealthy(cudaGetErrorString(status));
                if(!stage.bind_private(error))return unhealthy(error);
                status=cudaGetLastError();
                if(status!=cudaSuccess)return unhealthy(cudaGetErrorString(status));
                status=cudaDeviceSynchronize();
                if(status!=cudaSuccess)return unhealthy(cudaGetErrorString(status));
                if(!device.finish())return unhealthy("initial could not restore caller device");
            }
            if(!allocation.healthy())return unhealthy("initial allocation poisoned during preparation");
        } catch(...) { allocation.mark_unhealthy();return fallback; }
        storage->pending_owner.reset();
        NativeRestoreAccess::prepare(out,std::move(storage));
        return {NativeRestoreStatus::prepared,{}};
    } catch(const std::bad_alloc&) { return {NativeRestoreStatus::budget_refused,"initial metadata allocation failed"}; }
}
} // namespace strata::core
