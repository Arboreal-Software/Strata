#pragma once
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace strata::program {
// Request budgets are finite, including the native BGEN first-token phase.
inline bool serving_output_fits(uint64_t prompt,uint64_t output,uint64_t context,uint64_t slack=8) noexcept {
    return output && prompt<=context && output<=context-prompt && slack<=context-prompt-output;
}
// Two outer publications, their preparation/clone transients, immutable request
// and generated history. Receipt storage/observations are separately admitted.
inline bool serving_output_history_quote(uint64_t prompt,uint64_t output,uint64_t& bytes) noexcept {
    constexpr uint64_t base=1ULL<<20,per_token=128;
    if(!output||prompt>UINT64_MAX-output||prompt+output>(UINT64_MAX-base)/per_token)return false;
    const auto count=prompt+output;
    if(count>SIZE_MAX/sizeof(int32_t))return false;
    bytes=base+per_token*count;return true;
}
inline bool serving_receipt_requirement(uint64_t committed,uint64_t output,uint64_t comparison_headroom,uint64_t& count)noexcept {
    if(!output||committed>UINT64_MAX-output||comparison_headroom>UINT64_MAX-committed-output)return false;
    count=committed+output+comparison_headroom;return count<=SIZE_MAX;
}

// Serialized controller storage. Inline capacity preserves the old diagnostic
// default. Growth is proportional to ADMITTED cumulative output, never context
// by default. Existing receipts never move, including across continuation growth.
template<class T,size_t Inline=64> class ServingReceiptHistory {
    static_assert(std::is_nothrow_copy_assignable_v<T>);
    template<class U> struct Alloc {
        using value_type=U;std::shared_ptr<const void> credit;
        explicit Alloc(std::shared_ptr<const void> c):credit(std::move(c)){}
        template<class V>Alloc(const Alloc<V>& a)noexcept:credit(a.credit){}
        U* allocate(size_t n){return std::allocator<U>{}.allocate(n);}
        void deallocate(U* p,size_t n)noexcept{auto keep=credit;std::allocator<U>{}.deallocate(p,n);}
        template<class V>bool operator==(const Alloc<V>& a)const noexcept{return !credit.owner_before(a.credit)&&!a.credit.owner_before(credit);}
    };
    struct Block {
        std::shared_ptr<const void> credit; // payload dies before admission
        std::unique_ptr<T[]> values;size_t begin=0,count=0;
    };
    struct Directory {
        std::shared_ptr<const void> credit;
        std::vector<std::shared_ptr<Block>> blocks;
    };
public:
    struct Prepared {
        std::shared_ptr<Directory> directory;
        const void* base=nullptr;size_t before=0,capacity=0;bool valid=false;
    };
    struct Snapshot {
        std::shared_ptr<const void> credit;
        std::vector<T> entries;
        Snapshot()=default;
        Snapshot(const Snapshot&)=delete;
        Snapshot& operator=(const Snapshot&)=delete;
        Snapshot(Snapshot&&)=default;
        Snapshot& operator=(Snapshot&&)=delete;
    };
    ServingReceiptHistory()=default;
    ServingReceiptHistory(const ServingReceiptHistory&)=delete;
    ServingReceiptHistory& operator=(const ServingReceiptHistory&)=delete;
    size_t size()const noexcept{return capacity_;} // available admitted slots
    const T& operator[](size_t i)const noexcept {
        if(i<Inline)return initial_[i];
        for(const auto& block:directory_->blocks)
            if(i>=block->begin&&i-block->begin<block->count)return block->values[i-block->begin];
        std::terminate();
    }
    T& operator[](size_t i)noexcept{return const_cast<T&>(static_cast<const ServingReceiptHistory&>(*this)[i]);}
    bool growth_quote(uint64_t required,uint64_t& bytes)const noexcept {
        if(required<=capacity_){bytes=0;return true;}
        if(required>SIZE_MAX/sizeof(T))return false;
        const uint64_t blocks=directory_?directory_->blocks.size():0;
        constexpr uint64_t fixed=sizeof(Block)+sizeof(Directory)+256; // shared control blocks/envelope
        if(blocks>(UINT64_MAX-fixed)/(2*sizeof(std::shared_ptr<Block>))-1)return false;
        const uint64_t overhead=fixed+2*(blocks+1)*sizeof(std::shared_ptr<Block>);
        if(required-capacity_>(UINT64_MAX-overhead)/sizeof(T))return false;
        bytes=overhead+(required-capacity_)*sizeof(T);return true;
    }
    // admit(bytes, credit) must use the owning ledger. Allocation/refusal leaves
    // this history and the caller's Prepared value unchanged.
    template<class Admit>bool prepare(uint64_t required,Admit&& admit,Prepared& output)const noexcept {
        if(output.valid||!required)return false;
        uint64_t bytes=0;if(!growth_quote(required,bytes))return false;
        try {
            Prepared candidate;candidate.base=directory_.get();candidate.before=capacity_;
            candidate.capacity=capacity_;candidate.valid=true;
            if(bytes){
                // Directory credit must NOT be tied to a historical block:
                // replacing directories would otherwise accumulate O(turns^2)
                // charged pointer tables despite only O(receipts) live storage.
                const uint64_t block_bytes=sizeof(Block)+128+(required-capacity_)*sizeof(T);
                std::shared_ptr<const void> block_credit,directory_credit;
                if(!admit(block_bytes,block_credit)||!block_credit||
                   !admit(bytes-block_bytes,directory_credit)||!directory_credit)return false;
                auto block=std::allocate_shared<Block>(Alloc<Block>{block_credit});block->credit=block_credit;
                block->begin=capacity_;block->count=size_t(required-capacity_);
                block->values=std::make_unique<T[]>(block->count);
                auto directory=std::allocate_shared<Directory>(Alloc<Directory>{directory_credit});directory->credit=directory_credit;
                const size_t old=directory_?directory_->blocks.size():0;
                directory->blocks.reserve(old+1);
                if(directory_)for(const auto& prior:directory_->blocks)directory->blocks.push_back(prior);
                directory->blocks.push_back(std::move(block));candidate.directory=std::move(directory);
                candidate.capacity=size_t(required);
            }
            output=std::move(candidate);return true;
        }catch(...){return false;}
    }
    // Publication is allocation-free. Caller owns a serialized admission cut.
    bool commit(Prepared& prepared)noexcept {
        if(!prepared.valid||prepared.base!=directory_.get()||prepared.before!=capacity_)return false;
        if(prepared.directory)directory_.swap(prepared.directory);
        capacity_=prepared.capacity;prepared.valid=false;return true;
    }
    // Observers own exact copied receipt bytes independently of storage capacity.
    // No assumed number of simultaneous observers or borrowing from old credits.
    template<class Admit>bool snapshot(size_t count,Admit&& admit,Snapshot& output)const noexcept {
        if(count>capacity_||output.credit||!output.entries.empty()||count>(UINT64_MAX-sizeof(Snapshot)-256)/sizeof(T))return false;
        try {
            Snapshot candidate;
            if(count){
                const uint64_t bytes=sizeof(Snapshot)+256+uint64_t(count)*sizeof(T);
                if(!admit(bytes,candidate.credit)||!candidate.credit)return false;
                candidate.entries.resize(count);
                for(size_t i=0;i<count;++i)candidate.entries[i]=(*this)[i];
            }
            output.credit.swap(candidate.credit);output.entries.swap(candidate.entries);return true;
        }catch(...){return false;}
    }
private:
    std::array<T,Inline> initial_{};
    std::shared_ptr<Directory> directory_;
    size_t capacity_=Inline;
};
} // namespace strata::program
