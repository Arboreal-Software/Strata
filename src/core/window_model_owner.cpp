#include "strata/core/window_model_owner.hpp"
#include "strata/core/window_device.hpp"
#include <mutex>

namespace strata::core {
namespace {
std::recursive_mutex mutex;
bool failed=false;
uint64_t quarantine_count=0;
OwnedModelStage* quarantine=nullptr;
OwnedModelGang* gang_quarantine=nullptr;
uint64_t gang_quarantine_count=0;
}
bool OwnedModelStage::healthy() const {std::lock_guard lock(mutex);return !failed;}
bool OwnedModelStage::owns_weight_span(const void* pointer,uint64_t bytes,bool native) const {
    std::lock_guard lock(mutex);
    if(failed||!pointer||!bytes)return false;
    if(native)return resources_.native&&resources_.native->owns_weight_span(pointer,bytes);
    if(resources_.weights&&resources_.weights->owns_auxiliary_span(pointer,bytes))return true;
    const auto base=reinterpret_cast<uintptr_t>(resources_.arena);
    const auto p=reinterpret_cast<uintptr_t>(pointer);
    return p>=base&&p-base<=resources_.arena_bytes&&bytes<=resources_.arena_bytes-(p-base);
}
uint64_t model_stage_quarantine_count(){std::lock_guard lock(mutex);return quarantine_count;}
void OwnedModelStage::destroy(OwnedModelStage* owner) noexcept {
    auto& r=owner->resources_;
    // An empty object can arise if shared_ptr control-block allocation fails.
    if(!r.arena&&!r.native){delete owner;return;}
    bool safe=false;
    try {
        WindowDeviceScope device(r.device);
        std::string error;
        if(device.ok()&&cudaDeviceSynchronize()==cudaSuccess&&
           (!r.head||r.head->release_checked(error))&&
           (!r.native||r.native->release_checked(error))&&
           (!r.weights||r.weights->release_auxiliary_checked(error))) {
            if(!r.arena||cudaFree(r.arena)==cudaSuccess) {
                r.arena=nullptr;
                safe=device.finish();
            }
        }
    } catch(...) {}
    if(safe){delete owner;return;}
    // Retain every remaining allocation AND its reservation. Never allocate on
    // this failure path, and never infer completion from an attempted free.
    std::lock_guard lock(mutex);
    failed=true;owner->next_=quarantine;quarantine=owner;++quarantine_count;
}
bool adopt_model_stage(ModelStageResources& input,std::shared_ptr<const OwnedModelStage>& out,std::string& error) {
    std::lock_guard lock(mutex);
    if(failed){error="model owner quarantined; worker restart required";return false;}
    if(out||!input.arena||!input.arena_bytes||!input.weights||!input.native||!input.native->ready_for_adoption()||!input.reservation||input.device<0||
       input.reservation->purpose()!=ReservationPurpose::foreground) {
        error="model adoption requires exclusive loaded resources and a foreground reservation";return false;
    }
    if(!input.weights->load_complete()||input.weights->all().empty()) {
        error="model adoption requires a completed weight load and nonempty metadata";return false;
    }
    if(input.weights->auxiliary_bytes()&&input.weights->auxiliary_device()!=input.device) {
        error="model auxiliary allocation belongs to another device";return false;
    }
    if(input.head&&(!input.head->ready_for_adoption()||input.head->device()!=input.device)) {
        error="model head is incomplete or belongs to another device";return false;
    }
    const auto arena_base=reinterpret_cast<uintptr_t>(input.arena);
    for(const auto& [name,weight]:input.weights->all()) {
        if(weight.q8_scale) {
            if(!weight.resident||weight.elements<=0||weight.elements%32||
               uint64_t(weight.elements)>UINT64_MAX/4||
               weight.bytes!=uint64_t(weight.elements)+uint64_t(weight.elements)/32*4||
               !input.weights->owns_auxiliary_span(weight.data,uint64_t(weight.elements))||
               !input.weights->owns_auxiliary_span(weight.q8_scale,uint64_t(weight.elements)/32*4)) {
                error="requantized model weight lacks retained allocation: "+name;return false;
            }
            continue;
        }
        if(!weight.data) {
            if(weight.resident){error="resident model weight has no allocation: "+name;return false;}
            continue;
        }
        const auto base=reinterpret_cast<uintptr_t>(weight.data);
        if(!weight.resident||!weight.bytes||base<arena_base||base-arena_base>input.arena_bytes||
           weight.bytes>input.arena_bytes-(base-arena_base)) {
            error="canonical model weight lies outside retained arena: "+name;return false;
        }
    }
    const auto& charge=input.reservation->resources();
    const auto native_bytes=input.native->allocated_bytes();
    const auto aux_bytes=input.weights->auxiliary_bytes();
    const auto head_bytes=input.head?input.head->allocated_bytes():0;
    if(size_t(input.device)>=charge.device_bytes.size()||native_bytes>UINT64_MAX-input.arena_bytes||
       aux_bytes>UINT64_MAX-input.arena_bytes-native_bytes||
       head_bytes>UINT64_MAX-input.arena_bytes-native_bytes-aux_bytes||
       charge.device_bytes[size_t(input.device)]<input.arena_bytes+native_bytes+aux_bytes+head_bytes) {
        error="model reservation does not cover the allocation lower bound";return false;
    }
    try {
        std::shared_ptr<OwnedModelStage> candidate(new OwnedModelStage,OwnedModelStage::destroy);
        candidate->resources_=std::move(input);
        input.device=-1;input.arena=nullptr;input.arena_bytes=0;
        out=std::move(candidate);return true;
    } catch(const std::bad_alloc&) {error="model owner allocation failed";return false;}
}

bool OwnedModelGang::healthy() const {std::lock_guard lock(mutex);return !failed;}
uint64_t model_gang_quarantine_count(){std::lock_guard lock(mutex);return gang_quarantine_count;}
bool drain_model_gang_users(const OwnedModelGang& owner,std::string& error) {
    // No model-registry lock across a device wait: a queued host callback may
    // itself release a model lease or observe model health.
    for(const auto& stage:owner.stages()) {
        WindowDeviceScope device(stage->device());
        if(!device.ok()||cudaDeviceSynchronize()!=cudaSuccess||!device.finish()) {
            error="model consumer drain failed";return false;
        }
    }
    return true;
}
void OwnedModelGang::destroy(OwnedModelGang* owner) noexcept {
    auto& r=owner->resources_;
    bool safe=true;
    try {
        // Do not free even host-mapped storage until every device that can read
        // it is drained. Keeping all stage owners here also retains their charge.
        std::string error;
        safe=drain_model_gang_users(*owner,error);
        if(safe&&r.embedding) {
            WindowDeviceScope device(r.embedding->device());
            std::string error;
            safe=device.ok()&&r.embedding->release_checked(error)&&device.finish();
        }
    } catch(...) {safe=false;}
    if(safe){delete owner;return;}
    std::lock_guard lock(mutex);
    failed=true;owner->next_=gang_quarantine;gang_quarantine=owner;++gang_quarantine_count;
}
bool adopt_model_gang(ModelGangResources& input,std::shared_ptr<const OwnedModelGang>& out,std::string& error) {
    std::lock_guard lock(mutex);
    if(failed||out||input.stages.empty()) {
        error="model gang requires healthy stages and empty output";return false;
    }
    for(size_t i=0;i<input.stages.size();++i) {
        if(!input.stages[i]||!input.stages[i]->healthy()) {error="model gang stage unavailable";return false;}
        for(size_t j=0;j<i;++j)if(input.stages[j]->device()==input.stages[i]->device()) {
            error="model gang has duplicate stage devices";return false;
        }
    }
    if(input.embedding) {
        const auto& e=*input.embedding;
        const auto& lease=input.embedding_reservation;
        bool device_present=false;
        for(const auto& stage:input.stages)device_present|=stage->device()==e.device();
        if(!e.ready_for_adoption()||!device_present||!lease||lease->purpose()!=ReservationPurpose::foreground) {
            error="model gang embedding requires completed load, owning stage and reservation";return false;
        }
        const auto& charge=lease->resources();
        for(const auto& stage:input.stages)if(stage->reservation()==lease) {
            error="embedding needs a separate reservation from stage allocations";return false;
        }
        if(size_t(e.device())>=charge.device_bytes.size()||
           charge.device_bytes[size_t(e.device())]<e.allocated_device_bytes()||
           charge.host_bytes<e.allocated_host_bytes()||charge.pinned_bytes<e.allocated_pinned_bytes()) {
            error="model embedding reservation does not cover retained allocation";return false;
        }
    } else if(input.embedding_reservation) {
        error="embedding reservation has no embedding owner";return false;
    }
    try {
        std::shared_ptr<OwnedModelGang> candidate(new OwnedModelGang,OwnedModelGang::destroy);
        candidate->resources_=std::move(input);
        out=std::move(candidate);return true;
    } catch(const std::bad_alloc&) {error="model gang owner allocation failed";return false;}
}
} // namespace strata::core
