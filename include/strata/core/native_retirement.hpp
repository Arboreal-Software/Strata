#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>

namespace strata::core {
// Trusted internal owner-reactor contract, not proof of arbitrary host closure.
// Retain the genuine runtime/allocation OUTSIDE its covered resources (no cycle).
// Seal every resource before drain; close/join producers before calling drain.
// No future submission, callback reentry, or concurrent mutation is permitted.
class NativeRetirementContext {
public:
    NativeRetirementContext()=default;
    NativeRetirementContext(const NativeRetirementContext&)=delete;
    NativeRetirementContext& operator=(const NativeRetirementContext&)=delete;
    NativeRetirementContext(NativeRetirementContext&&)=delete;
    NativeRetirementContext& operator=(NativeRetirementContext&&)=delete;
    bool prepare(std::shared_ptr<const void> owner,std::span<const int> devices) noexcept {
        if(phase_!=Phase::empty||!owner||devices.empty()||devices.size()>devices_.size())return false;
        for(size_t i=0;i<devices.size();++i){
            if(devices[i]<0)return false;
            for(size_t j=0;j<i;++j)if(devices[i]==devices[j])return false;
        }
        auto value=next_.load();
        do {if(!value||value==UINT64_MAX)return false;}
        while(!next_.compare_exchange_weak(value,value+1));
        for(size_t i=0;i<devices.size();++i)devices_[i]=devices[i];
        size_=devices.size();generation_=value;owner_=std::move(owner);phase_=Phase::prepared;return true;
    }
    // Actual checked all-device barrier, implemented in native_retirement.cpp.
    // Includes zero-handle owners. Failure is sticky and cannot mint a receipt.
    bool drain_after_producers_closed() noexcept;
    bool prepared()const noexcept{return phase_==Phase::prepared;}
    bool drained()const noexcept{return phase_==Phase::drained;}
    bool covers(int device)const noexcept{for(size_t i=0;i<size_;++i)if(devices_[i]==device)return true;return false;}
    uint64_t generation()const noexcept{return generation_;}
private:
    enum class Phase{empty,prepared,draining,drained,failed};
    inline static std::atomic<uint64_t> next_{1};
    std::shared_ptr<const void> owner_;
    std::array<int,64> devices_{};
    size_t size_=0;
    uint64_t generation_=0;
    Phase phase_=Phase::empty;
};
} // namespace strata::core
