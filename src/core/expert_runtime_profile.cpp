#include "strata/core/expert_runtime_profile.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/window_restore.hpp"
#include <limits>
#include <stdexcept>
namespace strata::core {
namespace {
void need(bool value,const char* why){if(!value)throw std::invalid_argument(why);}
template<class T>bool same(const std::shared_ptr<const T>& a,const std::shared_ptr<const T>& b){
    return a&&b&&a.get()==b.get()&&!a.owner_before(b)&&!b.owner_before(a);
}
const ExpertSourceInventory* receipt(const ExpertProfileSourceView& v) noexcept {
    if(bool(v.file)==bool(v.arena))return nullptr;
    return v.file?v.file->completed_inventory():v.arena->completed_inventory();
}
std::shared_ptr<const ExpertSourceCapture> capture_owner(const ExpertProfileSourceView& v) noexcept {
    if(bool(v.file)==bool(v.arena))return {};
    return v.file?v.file->completed_inventory_owner():v.arena->completed_inventory_owner();
}
uint64_t extent(uint64_t count,uint64_t stride,uint64_t bytes){
    need(count&&bytes&&stride>=bytes,"invalid expert span geometry");
    need(count-1<=(UINT64_MAX-bytes)/stride,"expert span overflow");
    return (count-1)*stride+bytes;
}
uint64_t add(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,"expert metadata sum overflow");return a+b;}
uint64_t multiply(uint64_t a,uint64_t b){need(!a||b<=UINT64_MAX/a,"expert metadata product overflow");return a*b;}
void text(const std::string& s,uint64_t max){need(!s.empty()&&s.size()<=max&&s.find('\0')==std::string::npos,"expert profile text bound");}
}
struct ExpertRuntimeProfile::Storage {
    // Reservation destroyed last, after all priced metadata and runtime refs.
    std::shared_ptr<const ReservationLease> reservation;
    std::shared_ptr<const NativeRuntimeAttachment> runtime;
    ExpertProfileSourceView source;
    std::shared_ptr<const ExpertSourceCapture> capture;
    std::shared_ptr<const ExpertLayoutCapture> layout;
    const ExpertSourceInventory* original=nullptr;
    uint64_t layout_generation=0,hadamard_generation=0;
    std::vector<VerifiedInputLease> inputs;
    std::vector<ProfileArtifact> artifacts;
    std::vector<ExpertRuntimeProfileSpan> spans;
};
ExpertRuntimeProfile::ExpertRuntimeProfile()=default;
ExpertRuntimeProfile::~ExpertRuntimeProfile()=default;
bool ExpertRuntimeProfile::current() const noexcept {
    if(!storage_)return false;
    const auto& s=*storage_;
    const auto* binding=dynamic_cast<const ExpertRuntimeProfileBinding*>(s.runtime.get());
    if(!binding)return false;
    const auto now=binding->expert_profile_source();
    if(now.file!=s.source.file||now.arena!=s.source.arena||!same(now.bindings,s.source.bindings))return false;
    const auto capture=capture_owner(now);
    if(!same(capture,s.capture))return false;
    const auto* r=receipt(now);const auto* l=s.layout?s.layout->completed_inventory():nullptr;
    return r&&r==s.original&&l&&same(r->retained_input_bindings,now.bindings)&&
        same(r->retained_layout,s.layout)&&r->layout_generation==s.layout_generation&&
        r->hadamard_generation==s.hadamard_generation&&l->layout_generation==s.layout_generation&&
        l->hadamard_generation==s.hadamard_generation;
}
const std::vector<ProfileArtifact>* ExpertRuntimeProfile::artifacts()const noexcept{return current()?&storage_->artifacts:nullptr;}
const std::vector<ExpertRuntimeProfileSpan>* ExpertRuntimeProfile::spans()const noexcept{return current()?&storage_->spans:nullptr;}
const ExpertLayoutInventory* ExpertRuntimeProfile::layout()const noexcept{return current()?storage_->layout->completed_inventory():nullptr;}
bool collect_expert_runtime_profile(std::shared_ptr<const NativeRuntimeAttachment> runtime,
    const ExpertRuntimeProfileLimits& limits,WindowBudget budget,ExpertRuntimeProfile& out,std::string& error) {
    try {
        need(limits.max_sources&&limits.max_sources<=1024&&limits.max_layers&&limits.max_layers<=65536&&
            limits.max_string_bytes&&limits.max_string_bytes<=4096&&limits.max_host_bytes,"expert profile bounds invalid");
        const auto* binding=dynamic_cast<const ExpertRuntimeProfileBinding*>(runtime.get());
        need(binding,"runtime has no expert source membership interface");
        const auto view=binding->expert_profile_source();const auto captured=capture_owner(view);const auto* r=receipt(view);
        need(view.bindings&&captured&&r&&captured->completed_inventory()==r&&same(r->retained_input_bindings,view.bindings),"runtime source/binding receipt mismatch");
        need(view.bindings->native.size()<=limits.max_sources,"expert native binding bound");
        const auto* l=r->retained_layout?r->retained_layout->completed_inventory():nullptr;
        need(l&&same(l->retained_input_bindings,view.bindings)&&l->requested_layers>0&&l->effective_experts>0&&
            uint64_t(l->requested_layers)==r->layers&&uint64_t(l->effective_experts)==r->experts&&
            r->layers<=limits.max_layers&&r->sources.size()<=limits.max_sources&&
            r->limits.max_string_bytes<=limits.max_string_bytes&&
            r->spans.size()==r->layers*3&&r->arena==bool(view.arena),"expert layout/source geometry mismatch");
        need(r->layout_generation==l->layout_generation&&r->hadamard_generation==l->hadamard_generation,"expert generation mismatch");
        need(r->pack_path==view.bindings->pack_path()&&r->pack_path==l->claims.expected_pack_path,"expert pack authority mismatch");
        // Quote validated actual counts, with capacity/control-block/string
        // overhead. Existing retained captures keep their own prepaid leases;
        // their parsed layouts/sign buffers are not copied or charged twice.
        const uint64_t per_source=add(sizeof(VerifiedInputLease)+sizeof(ProfileArtifact)+512,
            multiply(4,limits.max_string_bytes));
        const uint64_t envelope=add(add(65536,sizeof(ExpertRuntimeProfile::Storage)),
            add(multiply(r->sources.size(),per_source),
                multiply(multiply(r->layers,3),sizeof(ExpertRuntimeProfileSpan)+64)));
        need(envelope<=limits.max_host_bytes,"expert profile metadata envelope refused");
        const auto snapshot=budget.snapshot();
        WindowResources resources;resources.device_bytes.resize(snapshot.capacity.device_bytes.size());resources.host_bytes=envelope;
        auto admitted=budget.reserve(resources,ReservationPurpose::foreground);
        need(bool(admitted),"expert profile metadata reservation refused");
        ExpertRuntimeProfile next;next.storage_=std::make_unique<ExpertRuntimeProfile::Storage>();
        auto& s=*next.storage_;s.reservation=std::move(admitted.lease);s.runtime=std::move(runtime);
        s.source=view;s.capture=captured;s.layout=r->retained_layout;s.original=r;s.layout_generation=r->layout_generation;s.hadamard_generation=r->hadamard_generation;
        s.inputs.reserve(r->sources.size());s.artifacts.reserve(r->sources.size());s.spans.reserve(r->layers*3);
        // Source-index mapping is bounded and priced by fixed source ceiling.
        std::vector<uint64_t> indices(r->sources.size(),UINT64_MAX);
        for(size_t i=0;i<r->sources.size();++i){
            const auto& f=r->sources[i];if(!f.selected)continue;
            text(f.path,limits.max_string_bytes);
            VerifiedInputLease found;
            auto match=[&](const VerifiedInputLease& lease){
                if(lease&&lease.loader_path()==f.path){
                    need(!found,"ambiguous expert input alias");found=lease;
                }
            };
            match(view.bindings->pack_payloads[2]);
            for(const auto& input:view.bindings->native)match(input);
            need(found&&found.readonly_fd()==f.retained_fd&&found.size()==f.expected_bytes,"selected source is not actual retained verified file");
            const std::string name="expert-source/"+std::to_string(i);text(name,limits.max_string_bytes);
            indices[i]=s.artifacts.size();s.inputs.push_back(found);
            s.artifacts.push_back(found.profile_artifact(ProfileRole::experts,name));
        }
        for(uint64_t layer=0;layer<r->layers;++layer)for(uint64_t role=0;role<3;++role){
            const auto& span=r->spans[size_t(layer*3+role)];const bool required=role<(r->from_gguf?3u:1u);
            need(span.complete==required,"incomplete or unexpected expert source role");if(!required)continue;
            need(span.layer==layer&&span.role==role&&span.experts==r->experts&&span.source<indices.size()&&
                indices[size_t(span.source)]!=UINT64_MAX,"expert span source/ordinal mismatch");
            const auto& f=r->sources[size_t(span.source)];
            const auto bytes=extent(span.experts,span.source_stride,span.bytes_per_expert);
            need(span.source_offset<=f.expected_bytes&&bytes<=f.expected_bytes-span.source_offset,"expert source span outside verified file");
            const auto dst=extent(span.experts,span.destination_stride,span.bytes_per_expert);
            need(span.destination_offset<=UINT64_MAX-dst,"expert destination span overflow");
            const auto expected=r->arena?(r->from_gguf?ExpertSourceTransfer::scatter_roles:ExpertSourceTransfer::read_packed):
                (r->from_gguf?ExpertSourceTransfer::mapped_roles:ExpertSourceTransfer::mapped_packed);
            need(span.transfer==expected,"expert transfer differs from actual source kind");
            s.spans.push_back({span.layer,span.role,indices[size_t(span.source)],span.source_offset,span.source_stride,
                span.bytes_per_expert,span.experts,span.destination_offset,span.destination_stride,span.mapped_file_ordinal,span.transfer});
        }
        need(!s.artifacts.empty()&&next.current(),"expert source changed during profile collection");
        out.storage_.swap(next.storage_);error.clear();return true;
    }catch(const std::exception& e){try{error=e.what();}catch(...){error.clear();}return false;}
}
}
