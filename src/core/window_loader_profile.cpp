#include "strata/core/window_loader_profile.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
namespace strata::core {
namespace {
void need(bool ok,const char* error){if(!ok)throw std::invalid_argument(error);}
uint64_t plus(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,"loader profile size overflow");return a+b;}
uint64_t times(uint64_t a,uint64_t b){need(!a||b<=UINT64_MAX/a,"loader profile product overflow");return a*b;}
bool nonzero(const StateDigest& d){return std::any_of(d.begin(),d.end(),[](uint8_t b){return b!=0;});}
void text_ok(const std::string& s,uint64_t bound){need(!s.empty()&&s.size()<=bound&&s.find('\0')==std::string::npos,"loader profile string bound");}
template<class T>bool owner_is(const std::shared_ptr<const T>& actual,const std::shared_ptr<const VerifiedDriverBindings>& expected){
    return actual.get()==static_cast<const void*>(expected.get())&&!actual.owner_before(expected)&&!expected.owner_before(actual);
}
// Fixed field order and explicit names. Values here are integers/enums ONLY;
// external names/path text never enters these grammar strings.
struct Fields {
    std::string value;uint64_t bound;
    Fields(const char* version,uint64_t b):value(version),bound(b){need(value.size()<=bound,"loader encoding bound");}
    template<class T>void field(const char* name,T v){
        const auto number=std::to_string(v);const auto n=std::char_traits<char>::length(name);
        need(plus(plus(plus(value.size(),n),number.size()),2)<=bound,"loader encoding bound");
        value+=';';value+=name;value+='=';value+=number;
    }
};
void span_ok(const VerifiedInputLease& lease,uint64_t offset,uint64_t bytes){
    need(bool(lease)&&lease.readonly_fd()>=0&&lease.size()&&nonzero(lease.content_digest()),"missing verified file authority");
    need(bytes&&offset<=lease.size()&&bytes<=lease.size()-offset,"loader source outside verified file");
}
ProfileRole role_for(const std::string& name){
    if(name=="token_embd.weight")return ProfileRole::embedding;
    if(name=="output.weight")return ProfileRole::head;
    return ProfileRole::dense;
}
const VerifiedInputLease& payload(const VerifiedDriverBindings& b,int64_t file){
    need(file>=0&&uint64_t(file)<b.pack_payloads.size(),"unknown canonical file id");return b.pack_payloads[size_t(file)];
}
const VerifiedInputLease& native_source(const std::vector<std::string>& paths,uint64_t ordinal,
    const std::vector<VerifiedInputLease>& leases,uint64_t bound){
    need(paths.size()==leases.size()&&ordinal<paths.size(),"native shard membership mismatch");
    for(size_t i=0;i<paths.size();++i){text_ok(paths[i],bound);need(bool(leases[i])&&paths[i]==leases[i].loader_path(),"native shard differs from actual verified argument");}
    return leases[size_t(ordinal)];
}
void shape(ProfileTensor& tensor,const std::vector<uint64_t>& dims,const LoaderProfileLimits& l){
    need(!dims.empty()&&dims.size()<=l.max_rank,"native shape rank bound");uint64_t count=1;
    for(auto d:dims){need(d>0,"zero native dimension");count=times(count,d);}tensor.shape=dims;
}
void canonical_scalars(ProfileTensor& t,const CanonicalTensorLoadRecord& r){
    t.weight_kind=r.normalized_kind;t.code_bits=r.code_bits;t.code_bias=r.code_bias;
    t.group_elems=r.group_elems;t.activation_kind=r.activation_kind;
    t.codes_bytes=r.index_codes_bytes;t.scales_bytes=r.index_scales_bytes;t.offsets_bytes=r.index_offsets_bytes;
    t.codebook_iq4nl=r.index_codebook!=0;t.has_offset=r.index_has_offset!=0;t.source_scales_fp16=r.index_scales_fp16!=0;
}
void dependency_matches(const NativeTensorLoadRecord& n,const CanonicalTensorLoadRecord& c){
    const auto& d=n.canonical;
    need(n.name==c.name&&d.file_id==c.file_id&&n.source_offset==c.source_offset&&n.source_bytes==c.source_bytes&&
        d.source_offset==c.source_offset&&d.source_bytes==c.source_bytes&&d.weight_kind==c.normalized_kind&&
        d.code_bits==c.code_bits&&d.code_bias==c.code_bias&&d.group_elems==c.group_elems&&d.activation_kind==c.activation_kind&&
        d.codes_bytes==c.index_codes_bytes&&d.scales_bytes==c.index_scales_bytes&&d.offsets_bytes==c.index_offsets_bytes&&
        d.source_scales_fp16==bool(c.index_scales_fp16)&&d.has_offset==bool(c.index_has_offset)&&d.codebook_iq4nl==bool(c.index_codebook)&&
        n.source_shape.size()==2&&c.ne0>0&&c.ne1>0&&n.source_shape[0]==uint64_t(c.ne0)&&n.source_shape[1]==uint64_t(c.ne1),
        "native canonical dependency differs from actual canonical receipt");
}
struct Collector {
    const LoaderProfileLimits& limits;
    const VerifiedDriverBindings& bindings;
    std::vector<ProfileArtifact>& artifacts;
    std::vector<ProfileTensor>& tensors;
    std::vector<uint64_t>& metadata_only;
    void artifact(ProfileRole role,const std::string& name,const VerifiedInputLease& lease){
        text_ok(name,limits.max_string_bytes);need(artifacts.size()<limits.max_artifacts,"loader artifact count bound");
        span_ok(lease,0,lease.size());
        need(std::none_of(artifacts.begin(),artifacts.end(),[&](const auto& a){return a.name==name;}),"duplicate loader artifact");
        auto a=lease.profile_artifact(role,name);need(a.verified_full_content&&a.immutable_owner&&nonzero(a.verification_evidence),"invalid verified artifact evidence");
        artifacts.push_back(std::move(a));
    }
    void publish(ProfileTensor t,const VerifiedInputLease& lease){
        text_ok(t.name,limits.max_string_bytes);text_ok(t.source_type,limits.max_string_bytes);text_ok(t.runtime_layout,limits.max_string_bytes);text_ok(t.conversion,limits.max_string_bytes);
        need(tensors.size()<limits.max_tensors,"loader tensor count bound");
        need(std::none_of(tensors.begin(),tensors.end(),[&](const auto& p){return p.role==t.role&&p.name==t.name;}),"duplicate loader tensor");
        span_ok(lease,t.offset,t.bytes);t.artifact=t.name+"/source";artifact(t.role,t.artifact,lease);tensors.push_back(std::move(t));
    }
    void canonical(const CanonicalLoadInventory& inv,const std::string& prefix){
        need(inv.expected_pack_loader_path==bindings.pack_path(),"canonical pack argument differs from verified binding");
        artifact(ProfileRole::dense,prefix+"/pack-index",bindings.pack_index);
        for(size_t i=0;i<inv.records.size();++i){const auto& r=inv.records[i];text_ok(r.name,limits.max_string_bytes);
            need(r.ne0>0&&r.ne1>=0&&r.plane_count<=r.planes.size()&&(!r.chosen?r.plane_count==0:true),"invalid canonical shape/plane receipt");
            need(r.source_index_kind>=0&&r.source_index_kind<=5&&r.normalized_kind>=0&&r.normalized_kind<=3&&r.code_bits>=0&&r.code_bits<=8&&r.group_elems>=0&&r.activation_kind>=0&&r.activation_kind<=1,"unsupported canonical scalar metadata");
            ProfileTensor t;t.role=role_for(r.name);t.name=prefix+"/canonical/"+r.name;t.offset=r.source_offset;t.bytes=r.source_bytes;
            t.shape={uint64_t(r.ne0)};if(r.ne1)t.shape.push_back(uint64_t(r.ne1));need(t.shape.size()<=limits.max_rank,"canonical rank bound");
            times(uint64_t(r.ne0),uint64_t(r.ne1?r.ne1:1));
            t.source_type="canonical-index-kind/"+std::to_string(r.source_index_kind);canonical_scalars(t,r);
            if(!r.chosen&&!r.source_bytes&&!r.runtime_bytes&&!r.plane_count&&r.code_bits!=0){
                t.source_type="canonical-index-metadata-v1";t.offset=0;t.bytes=bindings.pack_index.size();
                Fields metadata("canonical-shape-only-v1",limits.max_string_bytes);
                metadata.field("shape_only",1);metadata.field("chosen",0);metadata.field("executed_planes",0);
                metadata.field("file_id",r.file_id);metadata.field("original_source_offset",r.source_offset);metadata.field("original_source_bytes",r.source_bytes);
                metadata.field("source_index_kind",r.source_index_kind);metadata.field("raw16",r.raw16);
                metadata.field("index_destination_offset",r.index_destination_offset);metadata.field("runtime_offset",r.runtime_offset);metadata.field("runtime_bytes",r.runtime_bytes);
                metadata.field("index_codebook",r.index_codebook);metadata.field("index_has_offset",r.index_has_offset);metadata.field("index_scales_fp16",r.index_scales_fp16);
                t.runtime_layout=std::move(metadata.value);t.conversion="no-execution";
                metadata_only.push_back(tensors.size());publish(std::move(t),bindings.pack_index);continue;
            }
            const auto& lease=payload(bindings,r.file_id);span_ok(lease,r.source_offset,r.source_bytes);
            Fields layout("canonical-loader-v1",limits.max_string_bytes);
            layout.field("chosen",r.chosen);layout.field("file_id",r.file_id);layout.field("raw16",r.raw16);
            layout.field("index_destination_offset",r.index_destination_offset);layout.field("runtime_offset",r.runtime_offset);layout.field("runtime_bytes",r.runtime_bytes);
            layout.field("index_alignment",inv.index_alignment);layout.field("index_pool_bytes",inv.index_pool_bytes);layout.field("canonical_arena_bytes",inv.canonical_arena_bytes);
            layout.field("hc_completed",inv.hc_completed);layout.field("hc_allocation_bytes",inv.hc_allocation_bytes);layout.field("index_codebook",r.index_codebook);layout.field("index_has_offset",r.index_has_offset);layout.field("index_scales_fp16",r.index_scales_fp16);
            t.runtime_layout=std::move(layout.value);
            Fields conv("canonical-planes-v1",limits.max_string_bytes);conv.field("count",r.plane_count);
            for(uint32_t j=0;j<r.plane_count;++j){const auto& plane=r.planes[j];need(uint32_t(plane.conversion)<=uint32_t(CanonicalPlaneConversion::f16_to_f32),"unsupported canonical plane conversion");
                need(plane.source_offset<=r.source_bytes&&plane.source_bytes<=r.source_bytes-plane.source_offset,"plane outside canonical source row");
                span_ok(lease,plus(r.source_offset,plane.source_offset),plane.source_bytes);
                need(plane.runtime_offset<=r.runtime_bytes&&plane.runtime_bytes<=r.runtime_bytes-plane.runtime_offset,"plane outside canonical runtime row");
                const auto absolute_runtime=plus(r.runtime_offset,plane.runtime_offset);
                need(absolute_runtime<=inv.canonical_arena_bytes&&plane.runtime_bytes<=inv.canonical_arena_bytes-absolute_runtime,"plane outside canonical arena");
                conv.field("plane",j);conv.field("row_relative_source_offset",plane.source_offset);conv.field("source_bytes",plane.source_bytes);conv.field("row_relative_runtime_offset",plane.runtime_offset);conv.field("runtime_bytes",plane.runtime_bytes);conv.field("conversion",uint32_t(plane.conversion));
            }t.conversion=std::move(conv.value);publish(std::move(t),lease);
        }
        need(inv.hc_completed||inv.hc_records.empty(),"incomplete canonical HC receipt");
        for(size_t i=0;i<inv.hc_records.size();++i){const auto& h=inv.hc_records[i];need(h.canonical_row<inv.records.size()&&uint32_t(h.conversion)<=uint32_t(CanonicalHcConversion::bf16_in_f32_to_q8_host),"unsupported HC source/conversion");
            const auto& r=inv.records[size_t(h.canonical_row)];need(!r.chosen,"HC source must be actual skipped canonical row");
            need(h.codes_offset<=inv.hc_allocation_bytes&&h.codes_bytes<=inv.hc_allocation_bytes-h.codes_offset&&h.scales_offset<=inv.hc_allocation_bytes&&h.scales_bytes<=inv.hc_allocation_bytes-h.scales_offset,"HC allocation span overflow");
            ProfileTensor t;t.role=role_for(r.name);t.name=prefix+"/canonical-hc/"+std::to_string(i)+"/"+r.name;t.offset=r.source_offset;t.bytes=r.source_bytes;t.shape={uint64_t(r.ne0)};if(r.ne1)t.shape.push_back(uint64_t(r.ne1));canonical_scalars(t,r);t.source_type="canonical-index-kind/"+std::to_string(r.source_index_kind);
            Fields layout("canonical-hc-v1",limits.max_string_bytes);layout.field("row",h.canonical_row);layout.field("codes_offset",h.codes_offset);layout.field("codes_bytes",h.codes_bytes);layout.field("scales_offset",h.scales_offset);layout.field("scales_bytes",h.scales_bytes);layout.field("allocation_bytes",inv.hc_allocation_bytes);t.runtime_layout=std::move(layout.value);
            Fields conversion("canonical-hc-conversion-v1",limits.max_string_bytes);conversion.field("conversion",uint32_t(h.conversion));t.conversion=std::move(conversion.value);publish(std::move(t),payload(bindings,r.file_id));
        }
    }
    ProfileTensor native_tensor(const NativeTensorLoadRecord& r,const std::string& name){
        text_ok(r.name,limits.max_string_bytes);need(uint32_t(r.runtime_layout)<=uint32_t(NativeLoadLayout::hc_q8_0)&&uint32_t(r.conversion)<=uint32_t(NativeLoadConversion::bf16_to_q8_0_device)&&r.runtime_bytes,"unsupported native layout/conversion");
        ProfileTensor t;t.role=role_for(r.name);t.name=name;t.offset=r.source_offset;t.bytes=r.source_bytes;shape(t,r.source_shape,limits);
        t.source_type=r.source==NativeLoadSource::gguf_argument?"gguf-type/"+std::to_string(r.source_type):"canonical-dependency-v1";
        need(r.source==NativeLoadSource::unresolved_canonical||(r.source==NativeLoadSource::gguf_argument&&r.source_type>=0),"unsupported native source");
        t.native_type=r.source_type;
        return t;
    }
    void native(const NativeLoadInventory& inv,const CanonicalLoadInventory& canonical,const std::string& prefix){
        bool unresolved=false;
        for(const auto& r:inv.records){ProfileTensor t=native_tensor(r,prefix+"/native/"+r.name);const VerifiedInputLease* lease=nullptr;
            if(r.source==NativeLoadSource::gguf_argument)lease=&native_source(inv.sources,r.source_ordinal,bindings.dense,limits.max_string_bytes);
            else{unresolved=true;const auto found=std::find_if(canonical.records.begin(),canonical.records.end(),[&](const auto& c){return c.name==r.name;});need(found!=canonical.records.end(),"native missing canonical dependency");dependency_matches(r,*found);canonical_scalars(t,*found);lease=&payload(bindings,found->file_id);}
            Fields layout("native-loader-v1",limits.max_string_bytes);layout.field("source_kind",uint32_t(r.source));layout.field("source_ordinal",r.source_ordinal);layout.field("layout",uint32_t(r.runtime_layout));layout.field("runtime_bytes",r.runtime_bytes);layout.field("q8_packed_helper_bytes",r.q8_packed_helper_bytes);
            layout.field("layer_lo",inv.layer_lo);layout.field("layer_hi",inv.layer_hi);layout.field("global_layer_lo",inv.global_layer_lo);layout.field("global_layer_hi",inv.global_layer_hi);layout.field("include_ple_key",inv.include_ple_key);t.runtime_layout=std::move(layout.value);
            Fields conv("native-conversion-v1",limits.max_string_bytes);conv.field("conversion",uint32_t(r.conversion));t.conversion=std::move(conv.value);publish(std::move(t),*lease);
        }need(unresolved==inv.has_unresolved_canonical,"native unresolved coverage mismatch");
    }
    void head_embed(const HeadEmbedLoadInventory& inv,const std::string& prefix,int device){
        need(inv.generation&&inv.device==device&&uint32_t(inv.packed)<=uint32_t(HeadPackedForm::q8_sm60_qs_d)&&uint32_t(inv.placement)<=uint32_t(EmbedPlacement::device_alias_fallback),"unsupported head/embed state");
        const bool head=inv.role==HeadEmbedRole::head;need(head||inv.role==HeadEmbedRole::embedding,"unknown head/embed role");
        need(inv.tensor.source==NativeLoadSource::gguf_argument&&inv.tensor.name==(head?"output.weight":"token_embd.weight"),"head/embed source identity mismatch");
        const VerifiedInputLease* lease=nullptr;
        if(head){need(!bindings.head_disabled,"native head contradicts binding");lease=&native_source(inv.sources,inv.tensor.source_ordinal,bindings.head,limits.max_string_bytes);}
        else if(bindings.embedding_from_native)lease=&native_source(inv.sources,inv.tensor.source_ordinal,bindings.native,limits.max_string_bytes);
        else{need(inv.sources.size()==1&&inv.tensor.source_ordinal==0&&bindings.embedding&&inv.sources[0]==bindings.embedding.loader_path(),"embedding override differs from verified source");lease=&bindings.embedding;}
        auto t=native_tensor(inv.tensor,prefix+(head?"/head/":"/embedding/")+inv.tensor.name);
        Fields layout("head-embed-loader-v1",limits.max_string_bytes);layout.field("layout",uint32_t(inv.tensor.runtime_layout));layout.field("runtime_bytes",inv.tensor.runtime_bytes);layout.field("source_ordinal",inv.tensor.source_ordinal);layout.field("q8_packed_helper_bytes",inv.tensor.q8_packed_helper_bytes);layout.field("role",uint32_t(inv.role));layout.field("packed",uint32_t(inv.packed));layout.field("placement",uint32_t(inv.placement));layout.field("row_bytes",inv.row_bytes);layout.field("main_bytes",inv.main_bytes);layout.field("scratch_bytes",inv.scratch_bytes);layout.field("helper_bytes",inv.helper_bytes);layout.field("embedding_override",inv.embedding_override);t.runtime_layout=std::move(layout.value);
        Fields conv("native-conversion-v1",limits.max_string_bytes);conv.field("conversion",uint32_t(inv.tensor.conversion));t.conversion=std::move(conv.value);publish(std::move(t),*lease);
    }
};
}
bool collect_window_loader_profile(std::shared_ptr<const OwnedModelGang> gang,
    std::shared_ptr<const VerifiedDriverBindings> bindings,const LoaderProfileLimits& limits,
    WindowBudget budget,WindowLoaderProfile& out,std::string& error) {
    try {
        need(gang&&bindings&&gang->healthy(),"missing or unhealthy loader owner");
        need(limits.max_stages&&limits.max_stages<=64&&limits.max_tensors&&limits.max_tensors<=262144&&
            limits.max_artifacts&&limits.max_artifacts<=262208&&limits.max_rank&&limits.max_rank<=8&&
            limits.max_string_bytes&&limits.max_string_bytes<=4096&&limits.max_host_bytes,"unsupported collector bounds");
        need(!gang->stages().empty()&&gang->stages().size()<=limits.max_stages,"loader stage count bound");
        uint64_t count=0;
        // This pass does not copy inventories. They remain owned and immutable
        // under the caller's startup/reactor exclusion until publication.
        auto authority=[&](const auto* inv){need(inv&&inv->metadata_reservation&&owner_is(inv->retained_input_bindings,bindings),"loader inventory lacks exact binding authority");};
        auto add_count=[&](uint64_t n){count=plus(count,n);need(count<=limits.max_tensors,"loader tensor preflight bound");};
        for(const auto& stage:gang->stages()) {
            need(stage&&stage->healthy(),"unhealthy model stage");
            const auto* c=stage->weights().loaded_inventory();authority(c);
            need(c->records.size()<=c->limits.max_records&&c->hc_records.size()<=c->records.size(),"canonical inventory count invalid");
            add_count(c->records.size());add_count(c->hc_records.size());
            for(const auto& r:c->records)text_ok(r.name,limits.max_string_bytes);
            if(const auto* native=stage->native()) {const auto* n=native->loaded_inventory();authority(n);
                need(n->records.size()<=n->limits.max_records&&n->sources.size()<=n->limits.max_sources,"native inventory count invalid");
                add_count(n->records.size());for(const auto& r:n->records){text_ok(r.name,limits.max_string_bytes);need(r.source_shape.size()<=limits.max_rank,"native preflight rank bound");}}
            if(const auto* head=stage->head()){authority(head->loaded_inventory());add_count(1);}
        }
        if(const auto* embedding=gang->embedding()){authority(embedding->loaded_inventory());add_count(1);}
        const uint64_t artifacts=plus(count,gang->stages().size());
        need(artifacts<=limits.max_artifacts,"loader artifact preflight bound");
        // Conservative logical allocator envelope: retained output, vector
        // capacity, duplicate per-file lease/control blocks and one working row.
        // It is not an allocator-specific physical RSS measurement.
        const uint64_t tensor_bound=plus(sizeof(ProfileTensor)+sizeof(uint64_t)+512,
            plus(times(10,limits.max_string_bytes),times(limits.max_rank,sizeof(uint64_t))));
        const uint64_t artifact_bound=plus(sizeof(ProfileArtifact)+sizeof(VerifiedInputLease)+512,times(2,limits.max_string_bytes));
        const uint64_t host=plus(plus(8192,times(2,tensor_bound)),plus(times(count,tensor_bound),times(artifacts,artifact_bound)));
        need(host<=limits.max_host_bytes,"loader collector host bound");
        auto snapshot=budget.snapshot();need(snapshot.capacity.device_bytes.size()<=64,"collector device bound");
        WindowResources charge;charge.device_bytes.resize(snapshot.capacity.device_bytes.size());charge.host_bytes=host;
        auto reservation=budget.reserve(charge,ReservationPurpose::foreground);
        need(bool(reservation)&&bool(reservation.lease),"loader collector budget refused");
        WindowLoaderProfile candidate;candidate.reservation_=std::move(reservation.lease);candidate.gang_=gang;candidate.bindings_=bindings;
        candidate.artifacts_.reserve(size_t(artifacts));candidate.tensors_.reserve(size_t(count));candidate.metadata_only_.reserve(size_t(count));
        Collector collect{limits,*bindings,candidate.artifacts_,candidate.tensors_,candidate.metadata_only_};
        for(size_t i=0;i<gang->stages().size();++i){const auto& stage=*gang->stages()[i];const auto* c=stage.weights().loaded_inventory();
            const auto prefix="stage/"+std::to_string(i);collect.canonical(*c,prefix);
            if(const auto* n=stage.native())collect.native(*n->loaded_inventory(),*c,prefix);
            if(const auto* h=stage.head())collect.head_embed(*h->loaded_inventory(),prefix,stage.device());
        }
        if(const auto* embedding=gang->embedding()){const auto& inv=*embedding->loaded_inventory();collect.head_embed(inv,"gang",inv.device);}
        need(gang->healthy(),"loader owner became unhealthy during collection");
        need(candidate.tensors_.size()==count&&candidate.artifacts_.size()==artifacts,"loader inventory changed during collection");
        error.clear();
        // All allocations and validation precede this noexcept publication.
        // Old owners retire outside any loader/registry lock.
        using std::swap;swap(out.reservation_,candidate.reservation_);swap(out.gang_,candidate.gang_);
        swap(out.bindings_,candidate.bindings_);swap(out.artifacts_,candidate.artifacts_);swap(out.tensors_,candidate.tensors_);swap(out.metadata_only_,candidate.metadata_only_);
        return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
    catch(const std::exception& e){try{error=e.what();}catch(...){error.clear();}return false;}
    catch(...){try{error="loader profile collection failed";}catch(...){error.clear();}return false;}
}
} // namespace strata::core
