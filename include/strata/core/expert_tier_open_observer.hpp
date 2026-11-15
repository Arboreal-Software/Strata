#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::core {
enum class ExpertTierOpenOutcome : uint8_t { pending_copy, copied, already_resident, skipped_full, skipped_other };
struct ExpertTierOpenDecision {
    uint64_t input_rank=0,source_ordinal=0,bytes=0;
    int32_t layer=0,expert=0,slot=-1;
    ExpertTierOpenOutcome outcome=ExpertTierOpenOutcome::skipped_other;
};
struct ExpertTierOpenCapacity {int device=-1;int64_t layers=0,experts=0;uint64_t slots=0,decision_limit=0,slot_guard_bytes=0;bool per_layer=false,disabled=false;};
// Borrowed only during synchronous open. Caller prepays pending storage and
// retains context/ordinals until return. No C++20/budget dependency in GPU TUs.
struct ExpertTierOpenObserver {
    void* context=nullptr;
    const uint64_t* source_ordinals=nullptr;
    size_t ordinal_count=0;
    ExpertTierOpenDecision* pending=nullptr;
    size_t pending_capacity=0;
    bool (*configure)(void*,const ExpertTierOpenCapacity&) noexcept=nullptr;
    bool (*decision)(void*,const ExpertTierOpenDecision&) noexcept=nullptr;
    bool (*complete)(void*,uint64_t actual_attempts,bool capacity_full) noexcept=nullptr;
    void (*abort)(void*) noexcept=nullptr;
};
// Stack transaction: failed/throwing opens invalidate the observer without
// pretending that this callback drains or repairs device allocations.
class ExpertTierOpenTransaction {
public:
    explicit ExpertTierOpenTransaction(ExpertTierOpenObserver* o) noexcept:o_(o){}
    ~ExpertTierOpenTransaction(){if(o_&&!committed_&&o_->abort)o_->abort(o_->context);}
    bool valid(size_t ranks)const noexcept {
        return !o_||(o_->configure&&o_->decision&&o_->complete&&o_->abort&&o_->ordinal_count==ranks&&
            (ranks==0||(o_->source_ordinals&&o_->pending))&&o_->pending_capacity>=ranks);
    }
    ExpertTierOpenDecision* append(size_t rank,int32_t layer,int32_t expert) noexcept {
        if(!o_)return nullptr;
        if(count_>=o_->pending_capacity||rank>=o_->ordinal_count)return nullptr;
        auto* d=&o_->pending[count_++];*d={uint64_t(rank),o_->source_ordinals[rank],0,layer,expert,-1,ExpertTierOpenOutcome::skipped_other};return d;
    }
    ExpertTierOpenDecision* selected(size_t& cursor) noexcept {
        if(!o_)return nullptr;
        while(cursor<count_){auto* d=&o_->pending[cursor++];if(d->outcome==ExpertTierOpenOutcome::pending_copy)return d;}
        return nullptr;
    }
    bool publish(int device,int64_t layers,int64_t experts,uint64_t slots,bool full,uint64_t guard_bytes=0,bool per_layer=false) noexcept {
        if(!o_){committed_=true;return true;}
        if(!o_->configure(o_->context,{device,layers,experts,slots,uint64_t(count_),guard_bytes,per_layer,false}))return false;
        for(size_t i=0;i<count_;++i){if(o_->pending[i].outcome==ExpertTierOpenOutcome::pending_copy||!o_->decision(o_->context,o_->pending[i]))return false;}
        if(!o_->complete(o_->context,uint64_t(count_),full))return false;
        committed_=true;return true;
    }
private:
    ExpertTierOpenObserver* o_;size_t count_=0;bool committed_=false;
};
}
