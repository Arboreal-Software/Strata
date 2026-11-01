#include "strata/core/native_owned_batch_serving.hpp"
#include "strata/core/native_serial_serving.hpp"
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
        &Policy::keep_canonical,&Policy::prefill_auto,&Policy::prefill_head_alias,&Policy::vision,
        &Policy::kv_grow,&Policy::vram_elastic,&Policy::adapt_async_self,&Policy::spec_split,&Policy::split_skip_if_fits})refused(field,true);
    for(auto field:{&Policy::spec,&Policy::spec_corrupt,&Policy::suffix_draft,&Policy::lookup_chain,
        &Policy::pipeline_windows,&Policy::kv_resident,&Policy::prompt_cache,&Policy::adapt_every,&Policy::adapt_async,&Policy::adapt_warm_every})refused(field,1);
    refused(&Policy::batch,1);refused(&Policy::batch,3);refused(&Policy::batch_groups,2);
    refused(&Policy::prefill_chunk,32);refused(&Policy::short_read,65);refused(&Policy::peer_device,0);
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
    assert(batch::verifier_capacity(0,true)==2&&batch::execution_rows_per_slot(0,true)==1);
    assert(batch::verifier_capacity(0,false)==0&&batch::execution_rows_per_slot(0,false)==0);
    assert(batch::verifier_capacity(4,true)==4&&batch::execution_rows_per_slot(4,true)==4);
}
