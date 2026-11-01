#include "strata/core/verifier_setup_quote.hpp"
#include <algorithm>
#include <initializer_list>
#include <limits>
namespace strata::core {
namespace {
using S=VerifierSetupQuoteStatus;
bool add(uint64_t& total,uint64_t n) noexcept {if(n>UINT64_MAX-total)return false;total+=n;return true;}
bool product(uint64_t& out,std::initializer_list<uint64_t> values) noexcept {
    out=1;for(auto value:values){if(value&&out>UINT64_MAX/value)return false;out*=value;}return true;
}
bool aligned_add(uint64_t& total,uint64_t n) noexcept {
    if(n>UINT64_MAX-255)return false;return add(total,(n+255)&~uint64_t(255));
}
}
VerifierSetupQuoteStatus enumerate_verifier_explicit_setup(const VerifierSetupDimensions& d,
    VerifierSetupNativeRequests native,VerifierSetupProfile profile,VerifierSetupQuoteLimits limits,VerifierSetupLayout& out) noexcept {
    if(!limits.max_device_bytes||!limits.max_host_bytes)return S::invalid_limits;
    if(profile.unsupported_policy_mask||profile.side_mixer_mode>2)return S::unsupported_profile;
    if(d.max_t!=2||d.k!=10||!d.n_layers||d.n_layers>4096||!d.n_qsa_layers||
       d.n_gdn_layers>d.n_layers||d.n_qsa_layers>d.n_layers||d.n_gdn_layers!=d.n_layers-d.n_qsa_layers||
       d.head_dim!=256||d.idx_block!=4||!d.max_cells||d.max_cells%4||d.max_cells>uint64_t(INT32_MAX)-7||
       !d.selection_cap||d.selection_cap>uint64_t(INT32_MAX))return S::invalid_dimensions;
    for(auto value:{d.n_embd,d.hc,d.hc_lr,d.conv_channels,d.value_dim,d.v_heads,d.n_head,d.n_head_kv,
                   d.idx_q_heads,d.idx_key_dim,d.n_ff,d.n_expert,d.n_vocab})
        if(!value||value>uint64_t(INT32_MAX))return S::invalid_dimensions;
    if(d.n_embd%32||d.n_ff%32||!native.hit_scratch_bytes||!native.argmax_scratch_bytes||!native.max_expert_blob_bytes)return S::invalid_dimensions;
    VerifierSetupLayout layout;auto& q=layout.quote;uint64_t n=0,pos_heads=d.n_head;
    if(!add(pos_heads,d.n_head_kv)||!add(pos_heads,d.idx_q_heads)||pos_heads>uint64_t(INT32_MAX))return S::overflow;
    uint64_t head_width=0;if(!product(head_width,{d.n_head,d.head_dim})||head_width>uint64_t(INT32_MAX))return S::invalid_dimensions;
    const uint64_t max_in=std::max({d.n_embd,d.value_dim,head_width});
    if(max_in%32)return S::invalid_dimensions;
    const uint64_t T=2,K=10,N=d.n_embd,HC=d.hc,LR=d.hc_lr,C=d.conv_channels,Z=d.value_dim,HV=d.v_heads;
    const uint64_t NH=d.n_head,HD=d.head_dim,NK=d.n_head_kv,IQ=d.idx_q_heads,ID=d.idx_key_dim;
    const uint64_t nG=d.n_gdn_layers,nQ=d.n_qsa_layers,blocks=d.max_cells/4+2;
    const uint64_t plan_i32=168,HS=9*10240,TS=3*ID;
    auto mapped=[&](VerifierMappedField field,std::initializer_list<uint64_t> factors) noexcept {
        if(layout.mapped_count>=layout.mapped.size()||!product(n,factors)||!add(q.mapped_host_bytes,n))return false;
        layout.mapped[layout.mapped_count++]={field,n};++q.mapped_allocations;return true;
    };
    if(!mapped(VerifierMappedField::tok, {T,4})||!mapped(VerifierMappedField::step, {T,4,4})||!mapped(VerifierMappedField::pos, {T,pos_heads,4})||!mapped(VerifierMappedField::commit, {32})||!mapped(VerifierMappedField::ple, {T,N,4})||
       !mapped(VerifierMappedField::out, {24})||!mapped(VerifierMappedField::nf, {512})||!mapped(VerifierMappedField::x, {T,N,4})||!mapped(VerifierMappedField::ids, {T,K,4})||!mapped(VerifierMappedField::w, {T,K,4}))return S::overflow;
    if(!mapped(VerifierMappedField::seq, {64})||!mapped(VerifierMappedField::flag, {64})||!mapped(VerifierMappedField::flagA, {64})||!mapped(VerifierMappedField::flagB, {64})||!mapped(VerifierMappedField::plan_err, {64}))return S::overflow;
    if(!mapped(VerifierMappedField::ymiss, {T,K,N,4})||!mapped(VerifierMappedField::plan, {1408}))return S::overflow;
    auto take=[&](VerifierArenaField field,std::initializer_list<uint64_t> factors) noexcept {
        const uint64_t offset=q.arena_bytes;
        if(layout.arena_count>=layout.arena.size()||!product(n,factors)||!aligned_add(q.arena_bytes,n))return false;
        layout.arena[layout.arena_count++]={field,n,offset,q.arena_bytes-offset};++q.arena_takes;return true;
    };
    // EXACT same per-take order as current verify.cpp carve, each separately256-aligned.
    if(!take(VerifierArenaField::tok, {T,4})||!take(VerifierArenaField::step, {T,4,4})||!take(VerifierArenaField::pos, {T,pos_heads,4})||!take(VerifierArenaField::commit, {16})||
       !take(VerifierArenaField::ple, {T,N,4})||!take(VerifierArenaField::emb, {T,N,4})||!take(VerifierArenaField::R, {T,HC,N,4})||!take(VerifierArenaField::mixed, {T,N,4})||!take(VerifierArenaField::bo, {T,N,4})||
       !take(VerifierArenaField::inj, {T,HC,4})||!take(VerifierArenaField::inj2, {T,HC,4})||!take(VerifierArenaField::lo, {T,LR,4})||!take(VerifierArenaField::rs, {T,HC,4})||!take(VerifierArenaField::xn, {T,HC,N,4})||
       !take(VerifierArenaField::xq, {T,max_in/32,36})||!take(VerifierArenaField::sh_xq, {T,max_in/32,36})||!take(VerifierArenaField::xil, {max_in/32,4,68})||
       !take(VerifierArenaField::qkv_L, {nG,T,C,4})||!take(VerifierArenaField::h_L, {nG,T,C,4})||!take(VerifierArenaField::gate_L, {nG,T,HV,4})||!take(VerifierArenaField::beta_L, {nG,T,HV,4})||
       !take(VerifierArenaField::z, {T,Z,4})||!take(VerifierArenaField::y, {T,Z,4})||!take(VerifierArenaField::y_dummy, {T,Z,4})||!take(VerifierArenaField::qfull, {T,NH,2,HD,4})||!take(VerifierArenaField::qcur, {T,NH,HD,4})||
       !take(VerifierArenaField::kcur, {T,NK,HD,4})||!take(VerifierArenaField::vcur, {T,NK,HD,4})||!take(VerifierArenaField::idx_raw_L, {nQ,T,ID,4})||!take(VerifierArenaField::qidx, {T,IQ,ID,4})||
       !take(VerifierArenaField::scores, {T,blocks,4})||!take(VerifierArenaField::sel, {T,d.selection_cap,4})||!take(VerifierArenaField::attn, {T,NH,HD,4})||!take(VerifierArenaField::attn32, {T,NH,HD,4}))return S::overflow;
    uint64_t scratch=0;
    if(!product(scratch,{(d.selection_cap+63)/64,NH,258})||!add(scratch,64)||!take(VerifierArenaField::attn_scratch, {T,scratch,4})||
       !take(VerifierArenaField::tail_snap, {nQ,TS,4})||!take(VerifierArenaField::logits, {T,d.n_expert,4})||!take(VerifierArenaField::w, {T,K,4})||!take(VerifierArenaField::ids, {T,K,4})||!take(VerifierArenaField::shared, {T,N,4})||
       !take(VerifierArenaField::parts, {T,K,N,4})||!take(VerifierArenaField::hit_out, {T,K,N,4})||!take(VerifierArenaField::hit_slot, {T,K,4})||!take(VerifierArenaField::hit_dst, {T,K,4})||!take(VerifierArenaField::hit_count, {16})||
       !take(VerifierArenaField::plan, {2,plan_i32+16,4})||!take(VerifierArenaField::staging, {16,native.max_expert_blob_bytes})||
       !take(VerifierArenaField::hit_xq, {T,N/32,34})||!take(VerifierArenaField::hit_xs, {T,N/32,4})||!take(VerifierArenaField::nat_xq, {T,N/32,36}))return S::overflow;
    if(profile.had_gu_any&&!take(VerifierArenaField::nat_xr, {T,N,4}))return S::overflow;
    if(!take(VerifierArenaField::hit_scratch, {native.hit_scratch_bytes})||!take(VerifierArenaField::head_mixed, {T,N,4})||!take(VerifierArenaField::head_inj, {HC,4})||!take(VerifierArenaField::sh_bf16, {T,N,2})||
       !take(VerifierArenaField::sh_gate, {T,d.n_ff,4})||!take(VerifierArenaField::sh_up, {T,d.n_ff,4})||!take(VerifierArenaField::sh_g, {24})||!take(VerifierArenaField::head_logits, {T,d.n_vocab,4})||
       !take(VerifierArenaField::arg_scratch, {native.argmax_scratch_bytes})||!take(VerifierArenaField::one, {16})||!take(VerifierArenaField::hist_snap, {T,HS,4})||!take(VerifierArenaField::ple_key, {T,10240,4})||!take(VerifierArenaField::ple_val, {T,N,4}))return S::overflow;
    q.device_allocations=1;
    if(profile.qfuse_counters){if(!product(q.extra_device_bytes,{N/32,4}))return S::overflow;++q.device_allocations;}
    q.pinned_bytes=q.mapped_host_bytes;
    q.created_streams=profile.external_main_stream?2:3;q.created_events=3;q.first_t1_graph_execs=2;
    if(profile.side_mixer_mode){q.created_streams+=2;q.created_events+=4;}
    uint64_t device=q.arena_bytes;if(!add(device,q.extra_device_bytes))return S::overflow;
    if(device>limits.max_device_bytes||q.mapped_host_bytes>limits.max_host_bytes)return S::bound;
    out=layout;return S::ok;
}
VerifierSetupQuoteStatus quote_verifier_explicit_setup(const VerifierSetupDimensions& dimensions,
    VerifierSetupNativeRequests native,VerifierSetupProfile profile,VerifierSetupQuoteLimits limits,VerifierSetupQuote& out) noexcept {
    VerifierSetupLayout layout;
    const auto status=enumerate_verifier_explicit_setup(dimensions,native,profile,limits,layout);
    if(status==S::ok)out=layout.quote;
    return status;
}
VerifierSetupQuoteStatus quote_verifier_serial_handoffs(uint64_t N,uint64_t HC,uint32_t stages,
    uint64_t host_limit,VerifierSerialHandoffQuote& out) noexcept {
    if(!host_limit)return S::invalid_limits;
    if(!N||N>uint64_t(INT32_MAX)||!HC||HC>uint64_t(INT32_MAX)||!stages||stages>8)return S::invalid_dimensions;
    VerifierSerialHandoffQuote q;uint64_t floats=0;
    if(!product(floats,{HC,N})||!add(floats,N)||!add(floats,HC)||!product(q.bytes_per_edge,{8,floats,4})||
       !product(q.mapped_host_bytes,{stages-1,q.bytes_per_edge}))return S::overflow;
    q.mapped_allocations=stages-1;q.pinned_bytes=q.mapped_host_bytes;
    if(q.mapped_host_bytes>host_limit)return S::bound;
    out=q;return S::ok;
}
} // namespace strata::core
