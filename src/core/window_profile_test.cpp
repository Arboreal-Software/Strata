#include "strata/core/window_profile.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
using namespace strata::core;
static StateDigest digest(uint8_t n){StateDigest d{};d[0]=n;return d;}
static WindowProfileInput fixture(){
    WindowProfileInput x;
    for(uint32_t i=0;i<=uint32_t(ProfileRole::engine);++i){if(i==uint32_t(ProfileRole::ple))continue;ProfileArtifact a;a.role=ProfileRole(i);a.name="role"+std::to_string(i);a.bytes=4096;a.content=digest(uint8_t(i+1));a.verification_evidence=digest(99);a.verified_full_content=true;a.immutable_owner=std::make_shared<int>(i);x.artifacts.push_back(a);
        if(i<=uint32_t(ProfileRole::experts))x.tensors.push_back({ProfileRole(i),"weight",a.name,"BF16","dense-fp16","bf16-to-fp16",0,16,{2,4},16,0,0});}
    auto& r=x.runtime;r.engine_version="0.1.50+exact-build";r.backend="cuda";r.compiler="nvcc13";r.toolkit="13.0";r.driver="595.99.02";r.native_state_layout="qwen4exp-native-window-v1";r.tokenizer_implementation=digest(40);r.template_overrides=digest(41);r.kv="int8";r.max_context=262144;r.kv_resident=262144;r.batch_groups=1;r.rope_base=10000000;r.rope_factor=1;r.rope_scale=1;
    for(const auto& key:window_profile_switches())r.switches.emplace_back(key,key=="no_ple");
    x.stages.push_back({0,0,48,24576ull<<20,"GPU-fixture","sm86","single",std::vector<uint32_t>(12,1),false});return x;
}
int main(){
    ProfileLimits limits{32,4096,4096,8,8,256};std::string error;auto in=fixture();WindowProfileIdentity good;
    assert(compose_window_profile(in,limits,good,error));
    auto refuses=[&](const WindowProfileInput& x){auto out=good;assert(!compose_window_profile(x,limits,out,error));assert(out.model_profile==good.model_profile&&out.topology==good.topology&&out.artifact_owners==good.artifact_owners);};
    auto changed=[&](const WindowProfileInput& x,bool topology=false){WindowProfileIdentity out;assert(compose_window_profile(x,limits,out,error));assert(topology?out.topology!=good.topology:out.model_profile!=good.model_profile);};
    auto x=in;std::reverse(x.artifacts.begin(),x.artifacts.end());std::reverse(x.tensors.begin(),x.tensors.end());std::reverse(x.runtime.switches.begin(),x.runtime.switches.end());WindowProfileIdentity reordered;assert(compose_window_profile(x,limits,reordered,error)&&reordered.model_profile==good.model_profile&&reordered.topology==good.topology);
    x=in;x.artifacts[0].content=digest(201);changed(x);x=in;x.artifacts[4].content=digest(202);changed(x);x=in;x.artifacts[5].content=digest(203);changed(x);
    x=in;x.runtime.tokenizer_implementation=digest(51);changed(x);x=in;x.runtime.template_overrides=digest(52);changed(x);x=in;x.runtime.backend="hip";changed(x);changed(x,true);
    x=in;x.runtime.native_state_layout="unknown-layout";refuses(x);x=in;x.stages[0].transport="unknown-transport";refuses(x);
    x=in;x.tensors[0].conversion="alternate-bf16";changed(x);x=in;x.geometry.rms_eps*=2;changed(x);
    x=in;x.stages[0].device_uuid="GPU-other";changed(x,true);x=in;x.stages[0].qsa_formats[0]=17;changed(x,true);
    x=in;x.artifacts.erase(x.artifacts.begin());refuses(x);x=in;x.artifacts.push_back(x.artifacts[0]);refuses(x);x=in;x.tensors.erase(x.tensors.begin());refuses(x);x=in;x.tensors.push_back(x.tensors[0]);refuses(x);
    x=in;x.artifacts[0].verified_full_content=false;refuses(x);x=in;x.artifacts[0].immutable_owner.reset();refuses(x);x=in;x.artifacts[0].verification_evidence={};refuses(x);x=in;x.artifacts[0].content={};refuses(x);
    x=in;x.runtime.switches[0].first="unaudited_numeric_override";refuses(x);x=in;x.runtime.switches.pop_back();refuses(x);x=in;x.runtime.switches[0]=x.runtime.switches[1];refuses(x);
    x=in;x.runtime.rope_base=std::numeric_limits<double>::quiet_NaN();refuses(x);x=in;x.runtime.rope_scale=std::numeric_limits<double>::infinity();refuses(x);
    x=in;x.runtime.rope_ext=-0.0;WindowProfileIdentity zero;assert(compose_window_profile(x,limits,zero,error)&&zero.model_profile==good.model_profile);
    x=in;x.tensors[0].offset=UINT64_MAX;refuses(x);x=in;x.tensors[0].shape={UINT64_MAX,2};refuses(x);x=in;x.stages[0].qsa_formats.pop_back();refuses(x);x=in;x.stages[0].layer_end=47;refuses(x);x=in;x.runtime.vision=true;refuses(x);x=in;x.runtime.mtp=true;refuses(x);x=in;x.runtime.suffix=true;refuses(x);x=in;x.runtime.lookup=true;refuses(x);x=in;x.runtime.steering=true;refuses(x);
    x=in;x.ple.enabled=true;x.ple.layers={1};x.ple.multipliers={3,5,7};x.ple.vocab={11,13};x.ple.offsets={0,11};x.ple.ngram_size=3;x.ple.heads_per_ngram=1;x.ple.row_width=160;x.ple.conv_kernel=4;x.ple.eos_token=248044;x.ple.image_token=248056;x.ple.format="F8_E4M3";x.ple.scale=0.5;
    for(auto& [k,v]:x.runtime.switches)if(k=="no_ple")v=0;
    x.stages[0].ple_owner=true;x.artifacts.push_back({ProfileRole::ple,"ple",4096,digest(77),digest(99),true,std::make_shared<int>(1)});x.tensors.push_back({ProfileRole::ple,"ngram","ple","I8","F8_E4M3","scale",0,160,{1,160},160,0,0});WindowProfileIdentity ple;assert(compose_window_profile(x,limits,ple,error));auto changed_ple=x;changed_ple.ple.multipliers[0]++;WindowProfileIdentity ple2;assert(compose_window_profile(changed_ple,limits,ple2,error)&&ple2.model_profile!=ple.model_profile);x.ple.offsets[1]++;refuses(x);
    auto retained_input=fixture();std::weak_ptr<const void> retained=retained_input.artifacts[0].immutable_owner;WindowProfileIdentity retained_output;assert(compose_window_profile(retained_input,limits,retained_output,error));retained_input={};assert(!retained.expired());retained_output={};assert(retained.expired());
    auto raw=fixture();raw.runtime.frontend=ProfileFrontend::raw_token_ids;raw.runtime.raw_token_abi="model-vocabulary-index-v1";raw.runtime.tokenizer_implementation={};raw.runtime.template_overrides={};
    raw.artifacts.erase(std::remove_if(raw.artifacts.begin(),raw.artifacts.end(),[](const auto& a){return a.role==ProfileRole::tokenizer||a.role==ProfileRole::chat_template;}),raw.artifacts.end());
    WindowProfileIdentity raw_id;assert(compose_window_profile(raw,limits,raw_id,error)&&raw_id.model_profile!=good.model_profile);
    auto bad_raw=raw;bad_raw.runtime.tokenizer_implementation=digest(3);refuses(bad_raw);bad_raw=raw;bad_raw.runtime.raw_token_abi="unknown-token-abi";refuses(bad_raw);bad_raw=raw;bad_raw.artifacts.push_back(fixture().artifacts[4]);refuses(bad_raw);bad_raw=raw;bad_raw.artifacts.erase(bad_raw.artifacts.begin());refuses(bad_raw);
    std::puts("window_profile_test passed");
}
