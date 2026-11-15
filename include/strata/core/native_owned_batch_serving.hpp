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
bool eligible_common(bool native_pack,const Policy& p,bool explicit_selector) noexcept {
    return explicit_selector && native_pack && p.serve && p.owned_session_lifetime &&
        p.owned_model_lifetime && p.suffix_cache && !p.suffix_cache_unrepaired_experiment &&
        p.suffix_cache_host_bytes>0 && p.spec==0 && p.batch==2 &&
        p.batch_groups==1 && !p.batch_groups_auto && !p.batch_mtp &&
        !p.native_preset.empty() && !p.keep_canonical && p.prefill_chunk>0 && p.prefill_chunk<=32768 &&
        p.short_read==64 && !p.prefill_head_alias &&
        p.mtp.empty() && p.draft.empty() && p.draft_model.empty() &&
        p.spec_oracle.empty() && p.spec_follow.empty() && p.spec_corrupt==0 &&
        p.suffix_draft==0 && p.lookup_chain==0 && p.pipeline_windows==0 &&
        !p.kv_grow && !p.vram_elastic && p.kv=="int8" &&
        p.kv_resident==0 && p.prompt_cache==0 && p.adapt_every==0 &&
        p.adapt_async==0 && p.adapt_warm_every==0 && !p.adapt_async_self &&
        !p.spec_split && !p.split_skip_if_fits && p.peer_device<0 &&
        p.layer_split=="24" && p.split_device=="1";
}
// Configuration only. Both routes share EVERY nonvision predicate above.
// Neither branch confers native work, issuer, budget or completion authority.
template<class Policy>
bool eligible(bool native_pack,const Policy& p,bool explicit_selector) noexcept {
    return !p.vision && eligible_common(native_pack,p,explicit_selector);
}
template<class Policy>
bool eligible_image(bool native_pack,const Policy& p,bool explicit_selector) noexcept {
    return p.vision && eligible_common(native_pack,p,explicit_selector);
}
// Additional upstream routes have ordinary-mode support; the paid owned profile
// requires its existing allocation/copy policy until those resources are admitted.
// Returns the actual unsupported variable name; null is the positive path.
// The old SM86/SM60 table remains available. Only the new SM121 table
// injects the unadmitted NVFP4 execute-side probe into the owned route.
// Call with the SAME effective capability normalization as nv_defaults.cpp.
inline bool integration_nv_defaults_allowed(bool owned,int major,int minor) noexcept {
    return !owned || major!=12 || minor!=1;
}
inline bool integration_nv_kq_default_allowed(bool owned) noexcept {
    return !owned;
}
template<class Getenv>
const char* integration_environment_refusal(Getenv getenv_value) noexcept {
    constexpr const char* unpriced[] = {
        "STRATA_CKPT_POOL", "STRATA_CKPT_POOL_READY", "STRATA_SLOT_CKPT_MOVE",
        "STRATA_SLOT_COPY_DIRECT", "STRATA_SLOT_COPY_BATCHSYNC", "STRATA_BATCH_SELF_COMMIT",
        "STRATA_BATCH_GDN_GROUPS", "STRATA_BATCH_ATTN_ROWS", "STRATA_QSA_SLOTS",
        "STRATA_HEAD_HOST", "STRATA_EMBED_VRAM",
        "STRATA_PARK_ASYNC", "STRATA_PARK_VRAM_MIB", "STRATA_CKPT_ASYNC",
        "STRATA_EXPERT_NIBBLE", "STRATA_BATCH_CONCURRENT_READ", "STRATA_COALESCE_PREFILL",
        "STRATA_ADAPT_FETCH", "STRATA_DEQUANT_SIDE", "STRATA_PLE_COMPACT", "STRATA_PLE_COPY_STREAM",
        "STRATA_GDN_TC", "STRATA_GDN_CONVL2"
    };
    for (const char* name : unpriced) {
        const char* value = getenv_value(name);
        if (value && value[0] && std::strcmp(value,"0") != 0) return name;
    }
    // Match the actual consumers: W4A4 selects by exact spelling, while
    // both the linked and DSO KQ consumers activate on a leading '1'.
    const char* fp4=getenv_value("STRATA_PREFILL_NVFP4");
    if(fp4 && std::strcmp(fp4,"w4a4")==0)return "STRATA_PREFILL_NVFP4";
    const char* kq=getenv_value("STRATA_MMQ_KQ");
    if(kq && kq[0]=='1')return "STRATA_MMQ_KQ";
    return nullptr;
}
template<class Geometry>
bool integration_geometry_supported(const Geometry& g, int window) noexcept {
    const Geometry expected{}; // original Flash-Next shape, independent of selected compiled header
    return window == 8 &&
        g.n_embd == expected.n_embd &&
        g.n_layers == expected.n_layers &&
        g.qsa_interval == expected.qsa_interval &&
        g.n_vocab == expected.n_vocab &&
        g.rms_eps == expected.rms_eps &&
        g.ssm_state_size == expected.ssm_state_size &&
        g.ssm_k_heads == expected.ssm_k_heads &&
        g.ssm_v_heads == expected.ssm_v_heads &&
        g.ssm_d_conv == expected.ssm_d_conv &&
        g.ssm_conv_channels == expected.ssm_conv_channels &&
        g.ssm_value_dim == expected.ssm_value_dim &&
        g.n_head == expected.n_head &&
        g.n_head_kv == expected.n_head_kv &&
        g.head_dim == expected.head_dim &&
        g.n_rot == expected.n_rot &&
        g.idx_q_heads == expected.idx_q_heads &&
        g.idx_key_dim == expected.idx_key_dim &&
        g.idx_block == expected.idx_block &&
        g.idx_top_k == expected.idx_top_k &&
        g.hc == expected.hc &&
        g.hc_lr == expected.hc_lr &&
        g.n_expert == expected.n_expert &&
        g.n_expert_used == expected.n_expert_used &&
        g.n_ff == expected.n_ff &&
        g.n_ff_shexp == expected.n_ff_shexp &&
        g.family == expected.family &&
        g.n_mtp_layers == expected.n_mtp_layers;
}
inline int verifier_capacity(int requested_spec,bool resolved_owned_batch) noexcept {
    return resolved_owned_batch && requested_spec==0 ? 2 : requested_spec;
}
inline int execution_rows_per_slot(int requested_spec,bool resolved_owned_batch) noexcept {
    return resolved_owned_batch && requested_spec==0 ? 1 : requested_spec;
}
} // namespace strata::core::native_owned_batch_serving
