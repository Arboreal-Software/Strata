#pragma once
#include "strata/core/window_state.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace strata::program {
class ServingRawSession;
class SuffixRawTransaction;
class OwnedImageRequestOperation;
class OwnedImageSessionRecords;
class OwnedImagePeerObservation;
class OwnedImageRequestDriver;

// Sealed host data only. It is not a completion/transfer certificate and cannot
// be supplied to generic RAW capture/restore as a native image.
class VisionRequestOwner final {
public:
    struct Grid {
        size_t token_begin=0,row_begin=0,rows=0;
        int32_t nx=0,ny=0;
    };
    VisionRequestOwner(const VisionRequestOwner&)=delete;
    VisionRequestOwner& operator=(const VisionRequestOwner&)=delete;
    ~VisionRequestOwner() { if(allocation_)arena_allocator_.deallocate(allocation_,layout_.allocation_bytes); }
    std::span<const int64_t> tokens() const noexcept {return {at<int64_t>(layout_.tokens),layout_.token_count};}
    std::span<const float* const> row_pointers() const noexcept {return {at<const float*>(layout_.pointers),layout_.token_count};}
    std::span<const Grid> grids() const noexcept {return {at<Grid>(layout_.grids),image_count_};}
    std::span<const int32_t> position_triples() const noexcept {return {at<int32_t>(layout_.positions),layout_.cells*3};}
    std::span<const std::byte> captured_file() const noexcept {return {at<std::byte>(layout_.file),layout_.file_bytes};}
    uint64_t output_budget() const noexcept {return layout_.output_budget;}
    size_t embedding_width() const noexcept {return layout_.width;}
    uint64_t admitted_host_bytes() const noexcept {return layout_.host_bytes;}
private:
    friend class ServingRawSession;
    friend class SuffixRawTransaction;
    friend class OwnedImageRequestOperation;
    friend class OwnedImageSessionRecords;
    friend class OwnedImagePeerObservation;
    friend class OwnedImageRequestDriver;
    // No constructor/accessor exports ordinary credit/root/lease handles.
    struct Keepers {
        std::shared_ptr<const void> family_root;
        std::shared_ptr<const void> local_setup;
        std::shared_ptr<const void> admitted_credit;
    };
    template<class T> class HostAllocator {
        friend class ServingRawSession;
        friend class VisionRequestOwner;
        friend class OwnedImagePeerObservation;
        friend class OwnedImageRequestDriver;
        template<class> friend class HostAllocator;
        Keepers keepers_;
        explicit HostAllocator(const Keepers& keepers) noexcept:keepers_(keepers) {}
    public:
        using value_type=T;
        HostAllocator()=delete;
        HostAllocator(const HostAllocator&) noexcept=default;
        HostAllocator& operator=(const HostAllocator&)=delete;
        template<class U> HostAllocator(const HostAllocator<U>& other) noexcept:keepers_(other.keepers_) {}
        T* allocate(size_t n) {return std::allocator<T>{}.allocate(n);}
        void deallocate(T* p,size_t n) noexcept {
            // Copy roots first and credit last; no access to p after its free.
            const Keepers through_last_free=keepers_;
            std::allocator<T>{}.deallocate(p,n);
        }
        template<class U,class... Args> void construct(U* p,Args&&... args) {
            ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
        template<class U> void destroy(U* p) noexcept {p->~U();}
        template<class U> bool operator==(const HostAllocator<U>& other) const noexcept {
            const auto same=[](const auto& a,const auto& b) {
                return a.get()==b.get()&&!a.owner_before(b)&&!b.owner_before(a);
            };
            return same(keepers_.family_root,other.keepers_.family_root)&&
                same(keepers_.local_setup,other.keepers_.local_setup)&&
                same(keepers_.admitted_credit,other.keepers_.admitted_credit);
        }
        template<class U> bool operator!=(const HostAllocator<U>& other) const noexcept {return !(*this==other);}
    };
    struct Layout {
        size_t file=0,decoded=0,tokens=0,pointers=0,grids=0,positions=0;
        size_t file_bytes=0,token_count=0,width=0,row_stride=0,cells=0,allocation_bytes=0;
        uint64_t output_budget=0,host_bytes=0;
    };
    struct Identity {
        const void* source_session=nullptr;
        const void* runtime=nullptr;
        const void* geometry=nullptr;
        uint64_t engine=0,window=0,occupancy=0;
        core::StateDigest model{},topology{};
        core::StateCut preparation_cut{};
    };
    HostAllocator<std::byte> arena_allocator_; // all keepers precede every resource
    Layout layout_;
    Identity identity_;
    std::byte* allocation_=nullptr;
    std::byte* arena_=nullptr;
    size_t image_count_=0,decoded_floats_=0;
    VisionRequestOwner(const Keepers& keepers,const Layout& layout,const Identity& identity)
        :arena_allocator_(keepers),layout_(layout),identity_(identity),
         allocation_(arena_allocator_.allocate(layout.allocation_bytes)),
         arena_(allocation_+(64-reinterpret_cast<std::uintptr_t>(allocation_)%64)%64) {}
    template<class T> T* at(size_t offset) noexcept {
        return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(arena_)+offset);
    }
    template<class T> const T* at(size_t offset) const noexcept {
        return reinterpret_cast<const T*>(reinterpret_cast<const std::byte*>(arena_)+offset);
    }
};
} // namespace strata::program
