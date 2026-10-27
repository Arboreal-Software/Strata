#pragma once
#include "strata/core/window_native.hpp"

namespace strata::core::window_test {
inline ModelGeometry geometry() {
    ModelGeometry g;
    g.n_layers = 8; g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    return g;
}
inline SavedConversation saved(int lo, int hi, int format, bool ple) {
    const auto g = geometry();
    SavedConversation s;
    s.geometry = {g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                  g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                  g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    s.layer_lo = lo; s.layer_hi = hi;
    s.live.ids = {1,2,3,4,5,6,7,8,9};
    int q = 0;
    for (int layer = lo; layer < hi; ++layer) {
        if (layer % 4 == 3) ++q;
        else s.live.gdn.insert(s.live.gdn.end(), 128, uint8_t(layer));
    }
    if (!q) q = 1; // support-only allocated QSA
    for (int i = 0; i < q; ++i) {
        ConversationKv kv;
        kv.format = format; kv.cells = 12; kv.heads = 1; kv.head_dim = 64;
        kv.page_size = 4; kv.pooled_rows = 3; kv.idx_dim = 8;
        const bool int8 = format == 1 || format == 3 || format == 17;
        kv.k.resize(format == 2 ? 432 : int8 ? 768 : 1536, 11);
        kv.v.resize(format == 2 || format == 3 ? 432 : int8 ? 768 : 1536, 12);
        kv.k_scale.resize(int8 ? 24 : 0, 13);
        kv.v_scale.resize(int8 && format != 3 ? 24 : 0, 14);
        kv.pooled.resize(96, 15);
        s.kv.push_back(std::move(kv));
    }
    s.live.tails.resize(q * 96, 16); s.live.dead.resize(q * 32, 17); s.live.block_pos.resize(q * 4, 18);
    s.live.ple.resize(ple ? 368640 : 0, 19);
    return s;
}
inline SnapshotDescriptor metadata() {
    SnapshotDescriptor m;
    m.key.engine_generation = 1; m.key.window_generation = 2;
    m.key.session_id = "s"; m.key.thread_id = "t"; m.key.window_id = "w";
    m.key.model_profile[0] = 1; m.key.topology[0] = 2;
    m.cut = {3,10,9,9,7,{4}};
    m.accepted_tokens = {1,2,3,4,5,6,7,8,9,10}; m.pending_tokens = {10};
    m.cut.history = window_history_digest(m.accepted_tokens);
    return m;
}
}
