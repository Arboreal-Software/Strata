#include "strata/core/window_profile.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
namespace strata::core {
namespace {
void need(bool b,const char* s){if(!b)throw std::invalid_argument(s);}
bool nonzero(const StateDigest& d){return std::any_of(d.begin(),d.end(),[](auto n){return n!=0;});}
struct Encode {
    WindowHasher h;
    void u64(uint64_t n){uint8_t b[8];for(int i=0;i<8;++i)b[i]=uint8_t(n>>(i*8));h.update(b,8);}
    void text(const std::string& s){u64(s.size());h.update(s.data(),s.size());}
    void digest(const StateDigest& d){h.update(d.data(),d.size());}
    void number(double x){need(std::isfinite(x),"nonfinite numeric profile");if(x==0)x=0;u64(std::bit_cast<uint64_t>(x));}
    template<class T>void list(const std::vector<T>& v){u64(v.size());for(auto n:v)u64(n);}
};
void bounded(const std::string& s,const ProfileLimits& l){need(!s.empty()&&s.size()<=l.max_string_bytes&&s.find('\0')==std::string::npos,"missing/oversize/NUL profile text");}
uint64_t mul(uint64_t a,uint64_t b){need(!a||b<=UINT64_MAX/a,"profile product overflow");return a*b;}
}
const std::vector<std::string>& window_profile_switches(){
    static const std::vector<std::string> keys={"no_ple","native_bf16","native_bf16_extra","native_ple_key","native_moe_combine","native_gdn","native_flash_attn_short","native_qsa_indexer","native_qsa","native_rope","native_ple_postops","native_router","cpu_oracle_q8_0","gr_fp32_activations","gr_native_mmvf","native_preset","shared_late","keep_canonical","no_fused_gr","no_fast_attn","no_fused_gdn","no_fast_select","kv_rot","hc_q8"};
    return keys;
}
bool compose_window_profile(const WindowProfileInput& in,const ProfileLimits& l,WindowProfileIdentity& out,std::string& error){
    try {
        need(l.max_artifacts&&l.max_tensors&&l.max_string_bytes&&l.max_rank&&l.max_stages&&l.max_layers,"zero profile bounds");
        need(in.artifacts.size()<=l.max_artifacts&&in.tensors.size()<=l.max_tensors&&in.stages.size()<=l.max_stages,"profile count bound");
        const auto& g=in.geometry;const auto& r=in.runtime;const auto& p=in.ple;
        need(g.family==ModelFamily::Qwen4Exp&&g.n_layers>0&&uint64_t(g.n_layers)<=l.max_layers&&g.qsa_interval>0&&g.n_layers%g.qsa_interval==0,"unsupported geometry family/layers");
        need(g.n_embd>0&&g.n_vocab>0&&g.rms_eps>0&&g.n_head>0&&g.n_head_kv>0&&g.head_dim>0&&g.n_expert>0&&g.n_expert_used>0&&g.n_expert_used<=g.n_expert,"invalid model geometry");
        need(!r.mtp&&!r.suffix&&!r.lookup&&!r.batch_mtp&&!r.coupled_draft&&!r.pipeline_speculation&&!r.vision&&!r.steering,"profile capability requires text/no speculation/no steering; expansion pending");
        for(auto* s:{&r.engine_version,&r.backend,&r.compiler,&r.toolkit,&r.driver,&r.native_state_layout})bounded(*s,l);
        need(r.backend=="cuda"||r.backend=="hip","unsupported profile backend");
        need(r.native_state_layout=="qwen4exp-native-window-v1","unsupported native state schema");
        need(r.rope_type>=0&&r.rope_type<=2&&r.rope_base>0&&r.rope_factor>0&&r.rope_scale>0,"invalid resolved RoPE semantics");
        need(r.max_context&&r.kv_resident<=r.max_context&&r.batch_groups,"invalid context/batch bounds");
        need(r.kv=="fp16"||r.kv=="int8"||r.kv=="q4_0"||r.kv=="k8v4","unknown KV profile");
        need(r.frontend==ProfileFrontend::rendered_text||r.frontend==ProfileFrontend::raw_token_ids,"unknown frontend identity mode");
        if(r.frontend==ProfileFrontend::rendered_text){need(nonzero(r.tokenizer_implementation)&&nonzero(r.template_overrides)&&r.raw_token_abi.empty(),"missing/mixed resolved tokenizer/template semantics");}
        else need(!nonzero(r.tokenizer_implementation)&&!nonzero(r.template_overrides)&&r.raw_token_abi=="model-vocabulary-index-v1","raw token ABI missing or mixed rendering claim");
        std::map<std::string,int64_t> switches;
        need(r.switches.size()==window_profile_switches().size(),"missing numeric switches");
        for(const auto& [k,v]:r.switches){need(std::find(window_profile_switches().begin(),window_profile_switches().end(),k)!=window_profile_switches().end(),"unknown numeric switch");need(v==0||v==1,"nonboolean numeric switch");need(switches.emplace(k,v).second,"duplicate numeric switch");}
        need(bool(switches.at("no_ple"))!=p.enabled,"PLE enable switch mismatch");
        std::map<std::string,const ProfileArtifact*> artifacts;
        std::set<ProfileRole> roles,tensor_roles;
        WindowProfileIdentity candidate;candidate.artifact_owners.reserve(in.artifacts.size());
        for(const auto& a:in.artifacts){bounded(a.name,l);need(uint32_t(a.role)<=uint32_t(ProfileRole::engine),"unknown artifact role");need(a.bytes&&a.verified_full_content&&nonzero(a.content)&&nonzero(a.verification_evidence)&&a.immutable_owner,"missing immutable full-content verification declaration");need(artifacts.emplace(a.name,&a).second,"duplicate artifact name/role");roles.insert(a.role);candidate.artifact_owners.push_back(a.immutable_owner);}
        for(auto role:{ProfileRole::dense,ProfileRole::embedding,ProfileRole::head,ProfileRole::experts,ProfileRole::engine})need(roles.contains(role),"missing required artifact role");
        need(roles.contains(ProfileRole::tokenizer)==(r.frontend==ProfileFrontend::rendered_text)&&roles.contains(ProfileRole::chat_template)==(r.frontend==ProfileFrontend::rendered_text),"frontend artifact role mismatch");
        need(roles.contains(ProfileRole::ple)==p.enabled,"PLE artifact role mismatch");
        std::map<std::pair<ProfileRole,std::string>,const ProfileTensor*> tensors;
        for(const auto& t:in.tensors){bounded(t.name,l);bounded(t.artifact,l);bounded(t.source_type,l);bounded(t.runtime_layout,l);bounded(t.conversion,l);need(uint32_t(t.role)<=uint32_t(ProfileRole::ple),"invalid tensor role");auto a=artifacts.find(t.artifact);need(a!=artifacts.end()&&a->second->role==t.role,"missing/mismatched tensor source role");need(t.bytes&&t.offset<=a->second->bytes&&t.bytes<=a->second->bytes-t.offset,"tensor source extent out of bounds");need(!t.shape.empty()&&t.shape.size()<=l.max_rank,"invalid tensor rank");uint64_t n=1;for(auto d:t.shape){need(d>0,"zero tensor dimension");n=mul(n,d);}need(t.codes_bytes<=UINT64_MAX-t.scales_bytes&&t.codes_bytes+t.scales_bytes<=UINT64_MAX-t.offsets_bytes,"tensor plane size overflow");need(t.weight_kind>=0&&t.weight_kind<=3&&t.code_bits>=0&&t.code_bits<=8&&t.group_elems>=0&&t.activation_kind>=0&&t.activation_kind<=1&&t.native_type>=-1,"invalid tensor conversion contract");need(tensors.emplace(std::make_pair(t.role,t.name),&t).second,"duplicate tensor role/name");tensor_roles.insert(t.role);}
        for(auto role:{ProfileRole::dense,ProfileRole::embedding,ProfileRole::head,ProfileRole::experts})need(tensor_roles.contains(role),"missing required tensor role");
        need(tensor_roles.contains(ProfileRole::ple)==p.enabled,"missing/unexpected PLE tensors");
        need(p.layers.size()<=l.max_layers&&p.multipliers.size()<=l.max_rank&&p.vocab.size()<=l.max_tensors&&p.offsets.size()<=l.max_tensors,"PLE count bound");
        if(p.enabled){bounded(p.format,l);need(p.ngram_size>=2&&p.ngram_size==p.multipliers.size()&&p.heads_per_ngram&&p.vocab.size()==mul(p.ngram_size-1,p.heads_per_ngram)&&p.vocab.size()==p.offsets.size()&&p.row_width&&p.conv_kernel&&p.scale>0&&!p.layers.empty(),"invalid PLE semantics");need(p.eos_token>=0&&p.eos_token<g.n_vocab&&p.image_token>=0&&p.image_token<g.n_vocab&&p.null_token==-1,"invalid PLE token conventions");uint64_t end=0;for(size_t i=0;i<p.vocab.size();++i){need(p.vocab[i]&&p.offsets[i]==end&&p.vocab[i]<=UINT64_MAX-end,"invalid PLE head offset/vocab");end+=p.vocab[i];}uint64_t prev=0;for(size_t i=0;i<p.layers.size();++i){need(p.layers[i]<uint64_t(g.n_layers)&&(!i||p.layers[i]>prev),"duplicate/unordered PLE layers");prev=p.layers[i];}}
        else need(p.layers.empty()&&p.multipliers.empty()&&p.vocab.empty()&&p.offsets.empty()&&p.format.empty()&&!p.ngram_size&&!p.heads_per_ngram&&!p.row_width&&!p.conv_kernel&&p.scale==0&&p.eos_token==-1&&p.image_token==-1&&p.null_token==-1,"disabled PLE carries semantics");
        need(!in.stages.empty(),"missing topology");uint64_t layer=0,ple_owners=0;std::set<std::string> devices;
        for(size_t i=0;i<in.stages.size();++i){const auto& s=in.stages[i];bounded(s.device_uuid,l);bounded(s.architecture,l);bounded(s.transport,l);need(s.transport=="single"||s.transport=="host-staged"||s.transport=="peer","unknown stage transport");need(s.id==i&&s.layer_begin==layer&&s.layer_end>layer&&s.layer_end<=uint64_t(g.n_layers)&&s.capacity_bytes,"invalid stage partition");need(devices.insert(s.device_uuid).second,"duplicate physical stage device");uint64_t qsa=0;for(auto j=s.layer_begin;j<s.layer_end;++j)if(is_qsa_layer(g,int64_t(j)))++qsa;need(s.qsa_formats.size()==qsa,"missing native QSA formats");for(auto f:s.qsa_formats)need(f<=3||f==16||f==17,"unsupported native QSA format");ple_owners+=s.ple_owner;layer=s.layer_end;}
        need(layer==uint64_t(g.n_layers)&&ple_owners==uint64_t(p.enabled),"incomplete topology/PLE ownership");
        Encode m;m.text("strata-window-model-profile");m.u64(2);m.u64(uint32_t(r.frontend));m.text(r.raw_token_abi);
        m.u64(artifacts.size());for(const auto& [name,a]:artifacts){m.text(name);m.u64(uint32_t(a->role));m.u64(a->bytes);m.digest(a->content);} // proof identity is not model semantics
        m.u64(tensors.size());for(const auto& [name,t]:tensors){(void)name;m.u64(uint32_t(t->role));m.text(t->name);m.text(t->artifact);m.text(t->source_type);m.text(t->runtime_layout);m.text(t->conversion);m.u64(t->offset);m.u64(t->bytes);m.list(t->shape);m.u64(t->codes_bytes);m.u64(t->scales_bytes);m.u64(t->offsets_bytes);m.u64(uint64_t(t->weight_kind));m.u64(uint64_t(t->code_bits));m.u64(uint64_t(t->code_bias));m.u64(uint64_t(t->group_elems));m.u64(uint64_t(t->activation_kind));m.u64(uint64_t(t->native_type));m.u64(t->codebook_iq4nl);m.u64(t->has_offset);m.u64(t->source_scales_fp16);}
        m.u64(uint32_t(g.family));
#define G(field) need(g.field>=0,"negative geometry field");m.u64(uint64_t(g.field))
        G(n_embd);G(n_layers);G(qsa_interval);G(n_vocab);m.number(g.rms_eps);G(ssm_state_size);G(ssm_k_heads);G(ssm_v_heads);G(ssm_d_conv);G(ssm_conv_channels);G(ssm_value_dim);G(n_head);G(n_head_kv);G(head_dim);G(n_rot);G(idx_q_heads);G(idx_key_dim);G(idx_block);G(idx_top_k);G(hc);G(hc_lr);G(n_expert);G(n_expert_used);G(n_ff);G(n_ff_shexp);G(n_mtp_layers);
#undef G
        m.u64(p.enabled);m.list(p.layers);m.list(p.multipliers);m.list(p.vocab);m.list(p.offsets);m.u64(p.ngram_size);m.u64(p.heads_per_ngram);m.u64(p.row_width);m.u64(p.conv_kernel);m.u64(uint64_t(p.null_token));m.u64(uint64_t(p.eos_token));m.u64(uint64_t(p.image_token));m.text(p.format);m.number(p.scale);
        for(auto* s:{&r.engine_version,&r.backend,&r.compiler,&r.toolkit,&r.driver,&r.native_state_layout,&r.kv})m.text(*s);
        m.digest(r.tokenizer_implementation);m.digest(r.template_overrides);m.u64(r.max_context);m.u64(r.kv_resident);m.u64(r.batch_groups);m.u64(uint64_t(r.rope_type));
        for(auto x:{r.rope_base,r.rope_factor,r.rope_scale,r.rope_orig_ctx,r.rope_ext,r.rope_attn,r.rope_beta_fast,r.rope_beta_slow})m.number(x);
        m.u64(switches.size());for(const auto& [k,v]:switches){m.text(k);m.u64(uint64_t(v));}
        m.text("text-only:no-speculation:no-steering:v1");candidate.model_profile=m.h.digest();
        Encode t;t.text("strata-window-topology");t.u64(1);t.text(r.backend);t.text(r.native_state_layout);t.u64(g.n_layers);t.u64(r.max_context);t.u64(r.batch_groups);t.u64(in.stages.size());
        for(const auto& s:in.stages){t.u64(s.id);t.u64(s.layer_begin);t.u64(s.layer_end);t.u64(s.capacity_bytes);t.text(s.device_uuid);t.text(s.architecture);t.text(s.transport);t.list(s.qsa_formats);t.u64(s.ple_owner);}
        candidate.topology=t.h.digest();out=std::move(candidate);error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}
}
} // namespace strata::core
