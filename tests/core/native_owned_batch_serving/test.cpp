#include "strata/core/native_owned_batch_serving.hpp"
#include "strata/core/native_serial_serving.hpp"
#include "strata/core/geometry.hpp"
#include <cassert>
#include <string>
#include <cstdint>
using namespace strata::core;
struct Policy {
    bool serve=true,owned_session_lifetime=true,owned_model_lifetime=true,suffix_cache=true;
    bool suffix_cache_unrepaired_experiment=false,batch_groups_auto=false,batch_mtp=false;
    bool keep_canonical=false,prefill_auto=false,prefill_head_alias=false,vision=false,kv_grow=false;
    bool vram_elastic=false,adapt_async_self=false,spec_split=false,split_skip_if_fits=false;
    uint64_t suffix_cache_host_bytes=6442450944ULL;
    int spec=0,batch=2,batch_groups=1,prefill_chunk=16,short_read=64,spec_corrupt=0;
    int suffix_draft=0,lookup_chain=0,pipeline_windows=0,kv_resident=0,prompt_cache=0;
    int adapt_every=0,adapt_async=0,adapt_warm_every=0,peer_device=-1;
    std::string native_preset="native-shard",mtp,draft,draft_model,spec_oracle,spec_follow;
    std::string kv="int8",layer_split="24",split_device="1";
};
int main(){
    namespace batch=native_owned_batch_serving;
    Policy p;
    assert(batch::eligible(true,p,true));
    assert(!native_serial_serving::eligible(true,p));
    assert(!batch::eligible(false,p,true)&&!batch::eligible(true,p,false));
    auto refused=[&](auto member,auto value){auto changed=p;changed.*member=value;assert(!batch::eligible(true,changed,true));};
    for(auto field:{&Policy::serve,&Policy::owned_session_lifetime,&Policy::owned_model_lifetime,&Policy::suffix_cache})refused(field,false);
    for(auto field:{&Policy::suffix_cache_unrepaired_experiment,&Policy::batch_groups_auto,&Policy::batch_mtp,
        &Policy::keep_canonical,&Policy::prefill_head_alias,&Policy::vision,
        &Policy::kv_grow,&Policy::vram_elastic,&Policy::adapt_async_self,&Policy::spec_split,&Policy::split_skip_if_fits})refused(field,true);
    for(auto field:{&Policy::spec,&Policy::spec_corrupt,&Policy::suffix_draft,&Policy::lookup_chain,
        &Policy::pipeline_windows,&Policy::kv_resident,&Policy::prompt_cache,&Policy::adapt_every,&Policy::adapt_async,&Policy::adapt_warm_every})refused(field,1);
    refused(&Policy::batch,1);refused(&Policy::batch,3);refused(&Policy::batch_groups,2);
    refused(&Policy::prefill_chunk,0);refused(&Policy::prefill_chunk,32769);refused(&Policy::short_read,65);refused(&Policy::peer_device,0);
    for(int chunk:{16,1024,8192,32768}){auto supported=p;supported.prefill_chunk=chunk;assert(batch::eligible(true,supported,true));
        supported.prefill_auto=true;assert(batch::eligible(true,supported,true));}
    refused(&Policy::suffix_cache_host_bytes,uint64_t(0));
    for(auto field:{&Policy::mtp,&Policy::draft,&Policy::draft_model,&Policy::spec_oracle,&Policy::spec_follow})refused(field,std::string("enabled"));
    refused(&Policy::native_preset,std::string());refused(&Policy::kv,std::string("fp16"));
    refused(&Policy::layer_split,std::string("auto"));refused(&Policy::split_device,std::string("0"));
    assert(batch::environment("1","134217728","0","0","0"));
    assert(!batch::environment(nullptr,"134217728","0","0","0"));
    assert(!batch::environment("1","0","0","0","0"));
    assert(!batch::environment("1","134217728",nullptr,"0","0"));
    assert(!batch::environment("1","134217728","0","1","0"));
    assert(!batch::environment("1","134217728","0","0","1"));
    // Actual production predicate: intended fixed-two profile stays reachable.
    assert(batch::eligible(true,p,true) && batch::environment("1","134217728","0","0","0"));
    const auto absent=[](const char*) noexcept -> const char* { return nullptr; };
    assert(!batch::integration_environment_refusal(absent));
    const auto zero=[](const char*) noexcept -> const char* { return "0"; };
    assert(!batch::integration_environment_refusal(zero));
    constexpr const char* unsupported[] = {"STRATA_CKPT_POOL","STRATA_CKPT_POOL_READY","STRATA_SLOT_CKPT_MOVE","STRATA_SLOT_COPY_DIRECT","STRATA_SLOT_COPY_BATCHSYNC","STRATA_BATCH_SELF_COMMIT","STRATA_BATCH_GDN_GROUPS","STRATA_BATCH_ATTN_ROWS","STRATA_QSA_SLOTS","STRATA_HEAD_HOST","STRATA_EMBED_VRAM","STRATA_PARK_ASYNC","STRATA_PARK_VRAM_MIB","STRATA_CKPT_ASYNC","STRATA_EXPERT_NIBBLE","STRATA_BATCH_CONCURRENT_READ","STRATA_COALESCE_PREFILL","STRATA_ADAPT_FETCH","STRATA_DEQUANT_SIDE","STRATA_PLE_COMPACT","STRATA_PLE_COPY_STREAM","STRATA_GDN_TC","STRATA_GDN_CONVL2"};
    for (const char* key : unsupported) {
        const auto enabled=[key](const char* actual) noexcept -> const char* {
            return std::strcmp(key,actual)==0 ? "1" : nullptr;
        };
        const char* why=batch::integration_environment_refusal(enabled);
        assert(why && std::strcmp(why,key)==0);
    }
    // Late production default guards preserve ordinary capability tables and
    // the already accepted SM86/SM60 owned settings; no new KQ injection.
    for (int major : {6,8,12}) for (int minor : {0,1,6}) {
        assert(batch::integration_nv_defaults_allowed(false,major,minor));
        assert(batch::integration_nv_defaults_allowed(true,major,minor)==!(major==12&&minor==1));
    }
    assert(batch::integration_nv_kq_default_allowed(false));
    assert(!batch::integration_nv_kq_default_allowed(true));
    const auto value_for=[](const char* key,const char* value) {
        return [=](const char* name) noexcept -> const char* {
            return std::strcmp(name,key)==0 ? value : nullptr;
        };
    };
    for (const char* value : {"w4a8","fp16","0",""})
        assert(!batch::integration_environment_refusal(value_for("STRATA_PREFILL_NVFP4",value)));
    const char* fp4=batch::integration_environment_refusal(value_for("STRATA_PREFILL_NVFP4","w4a4"));
    assert(fp4 && std::strcmp(fp4,"STRATA_PREFILL_NVFP4")==0);
    for (const char* value : {"1","10","1enabled"}) {
        const char* kq=batch::integration_environment_refusal(value_for("STRATA_MMQ_KQ",value));
        assert(kq && std::strcmp(kq,"STRATA_MMQ_KQ")==0);
    }
    for (const char* value : {"0","2","","false"})
        assert(!batch::integration_environment_refusal(value_for("STRATA_MMQ_KQ",value)));
    for (const char* key : {"STRATA_PF_FUSED","STRATA_HC_UPMIX","STRATA_SPEC_COST"})
        assert(!batch::integration_environment_refusal(value_for(key,"1")));
    assert(batch::eligible(true,p,true) && batch::environment("1","134217728","0","0","0"));
    const ModelGeometry original{};
    assert(batch::integration_geometry_supported(original,8));
    assert(!batch::integration_geometry_supported(original,16));
    { auto changed=original; ++changed.n_embd; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_layers; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.qsa_interval; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_vocab; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.rms_eps; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.ssm_state_size; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.ssm_k_heads; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.ssm_v_heads; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.ssm_d_conv; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.ssm_conv_channels; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.ssm_value_dim; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_head; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_head_kv; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.head_dim; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_rot; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.idx_q_heads; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.idx_key_dim; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.idx_block; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.idx_top_k; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.hc; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.hc_lr; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_expert; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_expert_used; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_ff; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_ff_shexp; assert(!batch::integration_geometry_supported(changed,8)); }
    { auto changed=original; ++changed.n_mtp_layers; assert(!batch::integration_geometry_supported(changed,8)); }
    assert(batch::verifier_capacity(0,true)==2&&batch::execution_rows_per_slot(0,true)==1);
    assert(batch::verifier_capacity(0,false)==0&&batch::execution_rows_per_slot(0,false)==0);
    assert(batch::verifier_capacity(4,true)==4&&batch::execution_rows_per_slot(4,true)==4);
}
