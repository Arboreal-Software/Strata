#pragma once
#include "strata/core/native_batch_slot_binding.hpp"
#include "strata/core/window_device.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <utility>

namespace strata::prefill {class Prefill;}
namespace strata::core {class Verifier;}
namespace strata::program {
class ServingRawSession;
class SuffixRawTransaction;
class OwnedPositionBankBinding;
class OwnedPositionBankSetup;
class OwnedImageRequestOperation;

// Concrete startup storage, not permission to submit work or import an image.
// No root/ordinary-credit/lease aliases are exported, including weak aliases.
class OwnedVisionPositionBanks final {
public:
    OwnedVisionPositionBanks(const OwnedVisionPositionBanks&)=delete;
    OwnedVisionPositionBanks& operator=(const OwnedVisionPositionBanks&)=delete;
    ~OwnedVisionPositionBanks() noexcept {
        // All borrowing graph/worker owners must already have retired. A device
        // error cannot unwind credit or the staging source behind uncertain work.
        for(size_t i=0;i<stages_;++i){
            core::WindowDeviceScope scope(allocation_.stages()[i].device);
            if(!scope.ok()||cudaDeviceSynchronize()!=cudaSuccess)std::terminate();
            for(auto* p:banks_[i])if(p&&cudaFree(p)!=cudaSuccess)std::terminate();
            if(!scope.finish())std::terminate();
        }
        if(staging_)staging_allocator_.deallocate(staging_,cells_*3);
    }
private:
    friend class ServingRawSession;
    friend class SuffixRawTransaction;
    friend class OwnedPositionBankBinding;
    friend class OwnedPositionBankSetup;
    friend class core::Verifier;
    friend class prefill::Prefill;
    struct Keepers {
        std::shared_ptr<const void> family_root;
        std::shared_ptr<const void> local_setup;
        std::shared_ptr<const void> admitted_credit;
    };
    template<class T> class HostAllocator {
        friend class ServingRawSession;
        friend class OwnedVisionPositionBanks;
        template<class> friend class HostAllocator;
        Keepers keepers_;
        explicit HostAllocator(const Keepers& k)noexcept:keepers_(k){}
    public:
        using value_type=T;
        HostAllocator()=delete;
        HostAllocator(const HostAllocator&)noexcept=default;
        HostAllocator& operator=(const HostAllocator&)=delete;
        template<class U>HostAllocator(const HostAllocator<U>& a)noexcept:keepers_(a.keepers_){}
        T* allocate(size_t n){return std::allocator<T>{}.allocate(n);}
        void deallocate(T* p,size_t n)noexcept{
            const Keepers through_last_free=keepers_;
            std::allocator<T>{}.deallocate(p,n);
        }
        template<class U,class... A>void construct(U* p,A&&... a){::new(static_cast<void*>(p))U(std::forward<A>(a)...);}
        template<class U>void destroy(U* p)noexcept{p->~U();}
        template<class U>bool operator==(const HostAllocator<U>& a)const noexcept{
            const auto same=[](const auto& x,const auto& y){return x.get()==y.get()&&!x.owner_before(y)&&!y.owner_before(x);};
            return same(keepers_.family_root,a.keepers_.family_root)&&same(keepers_.local_setup,a.keepers_.local_setup)&&
                same(keepers_.admitted_credit,a.keepers_.admitted_credit);
        }
        template<class U>bool operator!=(const HostAllocator<U>& a)const noexcept{return !(*this==a);}
    };
    HostAllocator<int32_t> staging_allocator_; // keepers outlive all native and host resources
    core::NativeSessionAllocation allocation_; // allocator storage, never DriverWindowRuntime
    std::shared_ptr<const core::NativeBatchSlotBindings> slots_;
    size_t stages_=0,cells_=0;
    uint64_t epoch_=0;
    // Setup identity/address is immutable. Content installs have a separate
    // generation per physical bank; another occupied bank is never overwritten.
    std::array<uint64_t,3> content_generations_{{1,1,1}};
    std::array<std::weak_ptr<const OwnedImageRequestOperation>,3> image_contexts_{};
    std::array<std::array<int32_t*,3>,8> banks_{};
    int32_t* staging_=nullptr;
    OwnedVisionPositionBanks(const Keepers& k,const core::NativeSessionAllocation& allocation,
        std::shared_ptr<const core::NativeBatchSlotBindings> slots,size_t cells,uint64_t epoch)
        :staging_allocator_(k),allocation_(allocation),slots_(std::move(slots)),
         stages_(allocation.stages().size()),cells_(cells),epoch_(epoch){}
};

// Fixed outer setup state, including refusal AFTER admission. Its concrete
// generate retirement scope outlives actual enclosing frees. No ordinary
// admission handle or weak root may be obtained from this object.
class OwnedPositionBankSetup final {
public:
    OwnedPositionBankSetup()=default;
    OwnedPositionBankSetup(const OwnedPositionBankSetup&)=delete;
    OwnedPositionBankSetup& operator=(const OwnedPositionBankSetup&)=delete;
    bool ready()const noexcept{return bool(owner_);}
private:
    friend class ServingRawSession;
    OwnedVisionPositionBanks::Keepers keepers_;
    std::shared_ptr<const OwnedVisionPositionBanks> owner_;
};

// Only the drained startup factory can mint/install this stage-local binding.
// It retains its native owner and typed-control payment through weak last free.
class OwnedPositionBankBinding final {
public:
    OwnedPositionBankBinding(const OwnedPositionBankBinding&)=delete;
    OwnedPositionBankBinding& operator=(const OwnedPositionBankBinding&)=delete;
private:
    friend class ServingRawSession;
    friend class core::Verifier;
    friend class prefill::Prefill;
    template<class>friend class OwnedVisionPositionBanks::HostAllocator;
    std::shared_ptr<const OwnedVisionPositionBanks> owner_;
    size_t stage_=0;
    uint64_t verifier_=0,inventory_=0,epoch_=0;
    OwnedPositionBankBinding(std::shared_ptr<const OwnedVisionPositionBanks> owner,size_t stage,
        uint64_t verifier,uint64_t inventory)
        :owner_(std::move(owner)),stage_(stage),verifier_(verifier),inventory_(inventory),epoch_(owner_->epoch_){}
    const int32_t* table(size_t bank)const noexcept{
        return owner_&&epoch_==owner_->epoch_&&stage_<owner_->stages_&&bank<3?owner_->banks_[stage_][bank]:nullptr;
    }
};
} // namespace strata::program
