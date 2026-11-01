#include <cuda_runtime.h>
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/window_native.hpp"
#include "resident_repair_compare.inc"
#include <array>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
namespace strata::kernels {
void kv_stream_reset(const KvStreamMap&,void*){std::abort();}
void kv_ring_restore(const QsaAttnPools&,const KvHostPools&,int,int64_t,int64_t,int64_t,const QsaShapes&,void*){std::abort();}
}
namespace strata::core {strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState&){std::abort();}}
using namespace strata::core;
static void check(bool okay,const char* why){if(!okay){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
struct Pools {
    QsaState st;
    std::array<std::vector<uint8_t>, 8> data;
    Pools(const ModelGeometry& g, int format) {
        st.max_cells = 96; st.n_pages = st.n_slots = 24; st.idx_pooled_rows = 26;
        st.kv_int8 = format == 1; st.kv_q4 = format == 2;
        const size_t per = format == 2 ? strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                      : g.head_dim * (format == 1 ? 1 : 2);
        data[0].resize(96 * g.n_head_kv * per, 0xa5); data[1] = data[0];
        data[2].resize(format == 1 ? 96 * g.n_head_kv * (g.head_dim / 64) * 2 : 0, 0xa5); data[3] = data[2];
        data[4].resize(26 * g.idx_key_dim * 4, 0xa5);
        data[5].resize(3 * g.idx_key_dim * 4, 0xa5);
        data[6].resize(g.idx_key_dim * 4, 0xa5); data[7].resize(4, 0xa5);
        if (format == 2) { st.k_q4 = data[0].data(); st.v_q4 = data[1].data(); }
        else if (format == 1) {
            st.k_q = (int8_t*) data[0].data(); st.v_q = (int8_t*) data[1].data();
            st.k_scale = (uint16_t*) data[2].data(); st.v_scale = (uint16_t*) data[3].data();
        } else { st.k_pool = (uint16_t*) data[0].data(); st.v_pool = (uint16_t*) data[1].data(); }
        st.idx_pooled = (float*) data[4].data(); st.idx_tail = (float*) data[5].data();
        st.idx_dead = (float*) data[6].data(); st.idx_block_pos = (int32_t*) data[7].data();
    }
    ConversationKv image(const ModelGeometry& g, bool index) const {
        ConversationKv k;
        k.format = qsa_kv_format(st); k.cells = 12; k.page_size = 4;
        k.heads = g.n_head_kv; k.head_dim = g.head_dim; k.idx_dim = g.idx_key_dim;
        k.pooled_rows = index ? 3 : 0;
        k.k.resize(data[0].size() / 8, 13); k.v = k.k;
        k.k_scale.resize(data[2].size() / 8, 13); k.v_scale = k.k_scale;
        k.pooled.resize(k.pooled_rows * g.idx_key_dim * 4, 13);
        return k;
    }
};

int main(){
    ModelGeometry g;g.n_layers=8;g.n_expert=8;g.ssm_state_size=2;g.ssm_v_heads=2;g.ssm_conv_channels=8;
    g.n_head_kv=1;g.head_dim=64;g.idx_key_dim=8;
    ConversationStateSizes sizes;std::string error;check(conversation_state_sizes(g,sizes,error),"actual geometry sizes");
    Pools first(g,1),last(g,1);std::array<QsaState,2> layers{first.st,last.st};
    std::vector<uint8_t> gdn(sizes.gdn,27),history(sizes.ple,19);
    SessionState session;session.max_cells=96;session.layer_hi=8;session.gdn_alloc=g.n_gdn_layers();session.qsa_alloc=2;
    session.gdn_state=reinterpret_cast<float*>(gdn.data());session.ple_hist=reinterpret_cast<float*>(history.data());session.qsa_states=layers.data();
    session.ple_prev[0]=session.ple_prev[1]=session.ple_token=-1;
    const std::vector<int32_t> ids;const std::vector<ConversationImageKey> images;const std::vector<ConversationCheckpoint> checkpoints;
    SavedConversation ordinary;const int before_calls=native_empty_test::copies;
    check(!conversation_snapshot_save_private(ordinary,{ids,images,checkpoints,false},session,g,error)&&native_empty_test::copies==before_calls,
        "ordinary nonempty snapshot guard remains unchanged");
    SavedConversation split;split.stage_images.resize(1);
    auto* const child_storage=split.stage_images.data();
    check(conversation_snapshot_empty_save(split,session,g,error)&&split.stage_images.size()==1&&split.stage_images.data()==child_storage,
        "root empty capture preserves preallocated second-stage storage");
    check(conversation_snapshot_empty_save(split.stage_images[0],session,g,error)&&split.stage_images.data()==child_storage,
        "second-stage empty native capture uses original child storage");
    check(conversation_snapshot_empty_validate(split,session,g,nullptr,error)&&
          conversation_snapshot_empty_validate(split.stage_images[0],session,g,nullptr,error),
        "both actual empty stage images validate");
    SavedConversation prior;check(conversation_snapshot_empty_save(prior,session,g,error),"actual diagnostic empty native capture");
    check(prior.live.ids.empty()&&prior.live.gdn==gdn&&prior.live.ple==history&&prior.kv.size()==2,"actual real endpoint tensors captured");
    check(conversation_snapshot_empty_validate(prior,session,g,nullptr,error),"actual diagnostic empty restore validation");
    auto invalid=prior;invalid.live.ids.push_back(2);const int validation_calls=native_empty_test::copies;
    check(conversation_snapshot_empty_restore(invalid,session,g,nullptr,error)==ConversationRestore::invalid&&native_empty_test::copies==validation_calls,
        "nonempty image cannot use empty-only restore authority");
    std::fill(gdn.begin(),gdn.end(),3);std::fill(history.begin(),history.end(),4);
    for(auto* pool:{&first,&last})for(auto& data:pool->data)std::fill(data.begin(),data.end(),9);
    native_empty_test::fail_copy=native_empty_test::copies+1;
    check(conversation_snapshot_empty_restore(prior,session,g,nullptr,error)==ConversationRestore::transfer_failed,"actual first transfer failure cannot succeed");
    native_empty_test::fail_copy=0;
    check(conversation_snapshot_empty_restore(prior,session,g,nullptr,error)==ConversationRestore::restored,"actual empty native restore");
    SavedConversation restored;check(conversation_snapshot_empty_save(restored,session,g,error),"actual empty native recapture");
    check(restored.live.gdn==prior.live.gdn&&restored.live.ple==prior.live.ple&&restored.live.tails==prior.live.tails&&
        restored.live.dead==prior.live.dead&&restored.live.block_pos==prior.live.block_pos,"actual empty GDN PLE QSA endpoints restored byte-for-byte");
    SnapshotDescriptor metadata;metadata.steering_enabled=false;metadata.key.engine_generation=1;metadata.key.window_generation=2;metadata.key.session_id="s";metadata.key.thread_id="t";metadata.key.window_id="w";metadata.key.model_profile[0]=1;metadata.key.topology[0]=1;
    metadata.cut={3,0,0,0,0,window_history_digest(ids)};
    std::vector<NativeWindowStage> stages{{0,8,{1,1},true}};WindowStateLayout layout;
    check(bool(native_window_layout(g,stages,metadata.key,metadata.cut,96,1<<24,layout,false)),"actual zero-cut layout");
    const std::vector<NativePleTokens> ple(1,NativePleTokens{{-1,-1},-1});SnapshotDescriptor left,right;
    const auto left_result=describe_native_window(prior,g,layout,metadata,ple,left);
    if(!left_result)std::fprintf(stderr,"left describe %s\n",left_result.detail.c_str());
    const auto right_result=describe_native_window(restored,g,layout,metadata,ple,right);
    if(!right_result)std::fprintf(stderr,"right describe %s\n",right_result.detail.c_str());
    check(bool(left_result)&&bool(right_result),"actual captures produce genuine empty tensor inventories");
    BundleHandle a,b;check(bool(BundleHandle::freeze(left,left.key,layout,a))&&bool(BundleHandle::freeze(right,right.key,layout,b)),"actual immutable captures");
    strata::program::clean_recompute_qualification::Report report;
    check(strata::program::clean_recompute_qualification::compare_preserved_selection(a,b,layout,report)&&report.completed&&report.bytes_compared>0&&report.tensors_compared>0,
        "actual comparator checks all empty endpoint tensors");
    std::puts("PASS actual diagnostic empty native save/validate/restore+recapture and immutable tensor comparison; host transfer recording only");
}
