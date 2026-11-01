#pragma once
// Host binding observations only. No native submission, drain, common-cut
// certificate, RAW permission or model-weight authority is issued by this type.
#include "strata/core/window_allocator.hpp"
#include <array>
#include "strata/core/verifier_incarnation.hpp"
#include <limits>

namespace strata::core {
class Verifier;
struct NativeBatchSlotToken {
    uint64_t incarnation=0,generation=0;
    size_t slot=0;
    bool operator==(const NativeBatchSlotToken&)const=default;
};
class NativeBatchSlotBindings final {
public:
    static constexpr size_t slot_count=2,max_stages=8;
    struct StageLayout {int device=-1;int64_t begin=0,end=0,cells=0,k=0,gdn_ord0=0,qsa_ord0=0;};
    // Conservative bounded owner/vector/function/control-block metadata envelope.
    // Physical arenas and verifier graph/setup work require separate admission.
    static constexpr uint64_t metadata_bytes()noexcept{return sizeof(NativeBatchSlotBindings)+16384;}
    static std::shared_ptr<const NativeBatchSlotBindings> create(
            const std::array<const NativeSessionAllocation*,slot_count>& allocations,
            const std::array<NativeBatchSlotToken,slot_count>& tokens,
            std::shared_ptr<const ModelGeometry> geometry,
            std::shared_ptr<const ReservationLease> metadata,std::string& error){
        if(!geometry || !metadata || !metadata->canonical_owner(metadata) || metadata.use_count()!=1 ||
           metadata->purpose()!=ReservationPurpose::foreground || metadata->resources().host_bytes<metadata_bytes() ||
           metadata->resources().pinned_bytes || metadata->resources().disk_bytes ||
           metadata->resources().io_slots || metadata->resources().shadows){
            error="owned slot binding metadata/geometry admission refused";return {};
        }
        for(auto bytes:metadata->resources().device_bytes)if(bytes){error="slot binding metadata has device charge";return {};}
        size_t stages=0;
        for(size_t slot=0;slot<slot_count;++slot){
            const auto* allocation=allocations[slot];
            if(!allocation || !allocation->healthy() || allocation->stages().empty() ||
               allocation->stages().size()>max_stages || !allocation->reservation() ||
               !allocation->reservation()->canonical_owner(allocation->reservation()) ||
               allocation->reservation()->purpose()!=ReservationPurpose::private_work ||
               tokens[slot].slot!=slot || !tokens[slot].incarnation || !tokens[slot].generation ||
               (slot && tokens[slot].incarnation!=tokens[0].incarnation)){
                error="owned slot binding requires genuine healthy slot gangs/generations";return {};
            }
            if(slot && stages!=allocation->stages().size()){error="owned slot stage counts differ";return {};}
            stages=allocation->stages().size();int64_t end=0;
            for(size_t stage=0;stage<stages;++stage){const auto& native=allocation->stages()[stage];
                if(!native.owner || !native.session || native.device<0 || native.session->layer_lo!=end ||
                   native.session->layer_hi<=end || native.session->layer_hi>geometry->n_layers){
                    error="owned slot stage inventory is incomplete";return {};
                }
                end=native.session->layer_hi;
                if(slot){const auto& first=allocations[0]->stages()[stage];
                    if(native.device!=first.device || native.session->layer_lo!=first.session->layer_lo ||
                       native.session->layer_hi!=first.session->layer_hi ||
                       native.session->max_cells!=first.session->max_cells || native.session->k!=first.session->k ||
                       native.session->gdn_ord0!=first.session->gdn_ord0 || native.session->qsa_ord0!=first.session->qsa_ord0){
                        error="owned slots disagree on stage/device/layout";return {};
                    }
                }
                for(size_t previous_slot=0;previous_slot<=slot;++previous_slot)
                    for(size_t previous_stage=0;previous_stage<allocations[previous_slot]->stages().size();++previous_stage){
                        if(previous_slot==slot && previous_stage>=stage)break;
                        if(allocations[previous_slot]->stages()[previous_stage].session==native.session){
                            error="owned slot/session aliases another stage";return {};
                        }
                    }
            }
            if(end!=geometry->n_layers){error="owned slot omits model layers";return {};}
        }
        try {
            // The control-block deleter keeps admission through payload destruction
            // and any weak observers. Copies retain actual complete allocation owners.
            auto destroy=[metadata=std::move(metadata)](const NativeBatchSlotBindings* value){delete value;};
            return std::shared_ptr<const NativeBatchSlotBindings>(
                new NativeBatchSlotBindings(allocations,tokens,std::move(geometry)),std::move(destroy));
        }catch(...){error="admitted slot binding owner allocation refused";return {};}
    }
    size_t stage_count()const noexcept{return allocations_[0].stages().size();}
    const ModelGeometry* geometry()const noexcept{return geometry_.get();}
    NativeBatchSlotToken token(size_t slot)const noexcept{return slot<slot_count?tokens_[slot]:NativeBatchSlotToken{};}
    const OwnedNativeSession* stage(NativeBatchSlotToken token,size_t index)const noexcept {
        if(token.slot>=slot_count || token!=tokens_[token.slot] || !allocations_[token.slot].healthy() ||
           index>=allocations_[token.slot].stages().size())return nullptr;
        const auto& native=allocations_[token.slot].stages()[index];const auto& shape=layouts_[index];
        const auto* session=native.session;
        if(!session || native.device!=shape.device || session->layer_lo!=shape.begin || session->layer_hi!=shape.end ||
           session->max_cells!=shape.cells || session->k!=shape.k || session->gdn_ord0!=shape.gdn_ord0 ||
           session->qsa_ord0!=shape.qsa_ord0)return nullptr;
        return &native;
    }
private:
    NativeBatchSlotBindings(const std::array<const NativeSessionAllocation*,slot_count>& allocations,
        const std::array<NativeBatchSlotToken,slot_count>& tokens,std::shared_ptr<const ModelGeometry> geometry)
        :geometry_(std::move(geometry)),tokens_(tokens){
        for(size_t slot=0;slot<slot_count;++slot)allocations_[slot]=*allocations[slot];
        for(size_t stage=0;stage<allocations_[0].stages().size();++stage){
            const auto& native=allocations_[0].stages()[stage];const auto& session=*native.session;
            layouts_[stage]={native.device,session.layer_lo,session.layer_hi,session.max_cells,session.k,session.gdn_ord0,session.qsa_ord0};
        }
    }
    std::shared_ptr<const ModelGeometry> geometry_;
    std::array<NativeSessionAllocation,slot_count> allocations_;
    std::array<NativeBatchSlotToken,slot_count> tokens_{};
    std::array<StageLayout,max_stages> layouts_{};
};
class NativeBatchSlotInspection final {
public:
    bool present()const noexcept{return bool(owner_);}
    NativeBatchSlotToken token()const noexcept{return token_;}
    uint64_t inventory_generation()const noexcept{return inventory_generation_;}
    size_t stage_index()const noexcept{return stage_;}
    const ModelGeometry* geometry()const noexcept{return owner_?owner_->geometry():nullptr;}
    bool matches_session(const SessionState* session,int device)const noexcept{
        const auto* stage=owner_?owner_->stage(token_,stage_):nullptr;
        return stage&&stage->session==session&&stage->device==device;
    }
private:
    friend class Verifier;
    std::shared_ptr<const NativeBatchSlotBindings> owner_;
    const Verifier* verifier_=nullptr;
    NativeBatchSlotToken token_{};
    size_t stage_=0;
    uint64_t inventory_generation_=0;
    uint64_t verifier_incarnation_=0;
};
} // namespace strata::core
