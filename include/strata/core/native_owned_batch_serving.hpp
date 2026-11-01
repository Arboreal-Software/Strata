#pragma once
#include <cstring>

namespace strata::core::native_owned_batch_serving {
// Configuration eligibility only; this does not issue slot ownership, completion,
// budget, or executor authority. The driver must establish those independently.
inline bool selected(const char* value) noexcept {
    return value && std::strcmp(value,"1")==0;
}
inline bool disabled(const char* value) noexcept {
    return value && std::strcmp(value,"0")==0;
}
inline bool environment(const char* allocation,const char* setup_host_bytes,
                        const char* cpu_share,const char* helper,const char* helper_source) noexcept {
    return selected(allocation) && setup_host_bytes &&
        std::strcmp(setup_host_bytes,"134217728")==0 &&
        disabled(cpu_share) && disabled(helper) && disabled(helper_source);
}
template<class Policy>
bool eligible(bool native_pack,const Policy& p,bool explicit_selector) noexcept {
    return explicit_selector && native_pack && p.serve && p.owned_session_lifetime &&
        p.owned_model_lifetime && p.suffix_cache && !p.suffix_cache_unrepaired_experiment &&
        p.suffix_cache_host_bytes>0 && p.spec==0 && p.batch==2 &&
        p.batch_groups==1 && !p.batch_groups_auto && !p.batch_mtp &&
        !p.native_preset.empty() && !p.keep_canonical && p.prefill_chunk==16 &&
        p.short_read==64 && !p.prefill_auto && !p.prefill_head_alias &&
        p.mtp.empty() && p.draft.empty() && p.draft_model.empty() &&
        p.spec_oracle.empty() && p.spec_follow.empty() && p.spec_corrupt==0 &&
        p.suffix_draft==0 && p.lookup_chain==0 && p.pipeline_windows==0 &&
        !p.vision && !p.kv_grow && !p.vram_elastic && p.kv=="int8" &&
        p.kv_resident==0 && p.prompt_cache==0 && p.adapt_every==0 &&
        p.adapt_async==0 && p.adapt_warm_every==0 && !p.adapt_async_self &&
        !p.spec_split && !p.split_skip_if_fits && p.peer_device<0 &&
        p.layer_split=="24" && p.split_device=="1";
}
inline int verifier_capacity(int requested_spec,bool resolved_owned_batch) noexcept {
    return resolved_owned_batch && requested_spec==0 ? 2 : requested_spec;
}
inline int execution_rows_per_slot(int requested_spec,bool resolved_owned_batch) noexcept {
    return resolved_owned_batch && requested_spec==0 ? 1 : requested_spec;
}
} // namespace strata::core::native_owned_batch_serving
