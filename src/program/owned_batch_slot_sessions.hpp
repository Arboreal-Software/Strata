#pragma once
// Allocation dependency only. Does not admit batching, issue RAW certificates,
// bind a verifier, or establish that external graph/IO consumers have stopped.
#include "strata/core/window_allocator.hpp"
#include "strata/core/window_device.hpp"
#include <array>
#include <atomic>
#include <span>

namespace strata::program {
struct OwnedBatchSlotHandle {
    uint64_t incarnation=0,generation=0;
    size_t slot=0;
};
class OwnedBatchSlotSessions final {
public:
    static constexpr size_t capacity=2;
    OwnedBatchSlotSessions():incarnation_(mint_incarnation()){}
    OwnedBatchSlotSessions(const OwnedBatchSlotSessions&)=delete;
    OwnedBatchSlotSessions& operator=(const OwnedBatchSlotSessions&)=delete;
    // Members are retained by the runtime beyond every borrowing verifier and
    // stage. Native allocation deleters retain complete gangs on uncertain drain.
    bool initialize(const core::ModelGeometry& geometry,int64_t cells,int64_t k,
                    const std::vector<core::NativeSessionRequest>& requests,
                    core::WindowBudget& budget,std::string& error) {
        if(!incarnation_ || published_ || failed_ || retirement_attempted_ || requests.empty() || requests.size()>8){
            error="owned two-slot allocation requires fresh bounded stage gangs";return false;
        }
        for(const auto& allocation:allocations_)if(!allocation.stages().empty()){
            error="owned slot allocation cannot replace a retained gang";return false;
        }
        for(uint64_t generation:generations_)if(generation==UINT64_MAX){error="owned slot generation exhausted";return false;}
        for(size_t slot=0;slot<capacity;++slot){
            auto& allocation=allocations_[slot];
            const auto result=core::allocate_native_sessions(geometry,cells,k,requests,budget,allocation);
            if(!result){failed_=true;error=result.detail;return false;}
            // These are fresh private sessions; no slot graph can see them until
            // both complete stage gangs have initialized and publication succeeds.
            for(const auto& stage:allocation.stages()){
                core::WindowDeviceScope device(stage.device);
                if(!device.ok() || !stage.owner || !stage.session ||
                   !core::session_zero_checked(*stage.session,geometry,nullptr,nullptr,error) ||
                   cudaDeviceSynchronize()!=cudaSuccess || !device.finish()){
                    allocation.mark_unhealthy();failed_=true;
                    if(error.empty())error="owned batch slot reset/drain failed";
                    return false;
                }
            }
        }
        stage_count_=requests.size();
        for(auto& generation:generations_)++generation;
        published_=true;return true;
    }
    bool published()const noexcept{return published_;}
    size_t stage_count()const noexcept{return published_?stage_count_:0;}
    OwnedBatchSlotHandle handle(size_t slot)const noexcept {
        return published_&&slot<capacity?OwnedBatchSlotHandle{incarnation_,generations_[slot],slot}:OwnedBatchSlotHandle{};
    }
    const core::NativeSessionAllocation* allocation(OwnedBatchSlotHandle handle)const noexcept {
        if(!published_ || handle.incarnation!=incarnation_ || handle.slot>=capacity ||
           handle.generation!=generations_[handle.slot] || !allocations_[handle.slot].healthy())return nullptr;
        return &allocations_[handle.slot];
    }
    core::SessionState* session(size_t stage,OwnedBatchSlotHandle handle)const noexcept {
        const auto* owned=allocation(handle);
        return owned && stage<owned->stages().size()?owned->stages()[stage].session:nullptr;
    }
    // Read-only lifetime/IO closure access, never permission to execute unpublished
    // or terminal sessions. Includes complete gangs retained after startup failure.
    const core::NativeSessionAllocation& allocation_for_teardown(size_t slot)const {return allocations_.at(slot);}
    // Caller must first revoke and JOIN all graph/dispatch/IO borrowers. Current
    // integration uses runtime member destruction after its graph owners; this
    // explicit terminal API supports later checked slot teardown, never live reuse.
    bool retire_after_consumers_joined(std::span<const int> additional_devices,std::string& error) {
        if(retirement_attempted_){error="owned batch slot retirement cannot be retried";return false;}
        retirement_attempted_=true;published_=false;
        std::array<core::NativeRetirementAttempt,capacity> attempts;
        for(size_t slot=0;slot<capacity;++slot)
            if(!allocations_[slot].stages().empty() &&
               !attempts[slot].prepare(allocations_[slot],additional_devices,error)){
                failed_=true;return false;
            }
        for(size_t slot=0;slot<capacity;++slot)
            if(!allocations_[slot].stages().empty() && !attempts[slot].drain_and_commit(error)){
                failed_=true;return false;
            }
        for(auto& allocation:allocations_)allocation=core::NativeSessionAllocation{};
        stage_count_=0;failed_=false;retirement_attempted_=false;
        return true;
    }
private:
    static uint64_t mint_incarnation()noexcept {
        static std::atomic<uint64_t> next{1};
        uint64_t value=next.load(std::memory_order_relaxed);
        for(;;){
            if(!value || value==UINT64_MAX)return 0;
            if(next.compare_exchange_weak(value,value+1,std::memory_order_relaxed))return value;
        }
    }
    uint64_t incarnation_=0;
    std::array<core::NativeSessionAllocation,capacity> allocations_;
    std::array<uint64_t,capacity> generations_{};
    size_t stage_count_=0;
    bool published_=false,failed_=false,retirement_attempted_=false;
};
} // namespace strata::program
