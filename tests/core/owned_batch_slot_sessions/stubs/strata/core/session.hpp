#pragma once
#include <cuda_runtime.h>
#include "strata/core/geometry.hpp"
#include <memory>
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <cstddef>
namespace strata::core {
class NativeRetirementContext;
struct QsaElasticPlan {};
struct QsaElasticView {
    bool prepare_retirement(NativeRetirementContext&){return true;}
    bool commit_retirement(NativeRetirementContext&){return true;}
};
// One real ModelGeometry declaration. Explicit bounded legacy geometry is
// selected in test.cpp; session/native storage remains a recording boundary.
struct QsaState {
    bool kv_hybrid=false,kv_rot=false,kv_q4=false,kv_int8=false;int kv_mode=0;
    int64_t max_cells=0,n_pages=0,n_slots=0,idx_pooled_rows=0;
    void *k_pool=nullptr,*v_pool=nullptr,*k_q=nullptr,*v_q=nullptr,*k_q4=nullptr,*v_q4=nullptr,*k_scale=nullptr,*v_scale=nullptr;
    void *idx_pooled=nullptr,*idx_tail=nullptr,*idx_dead=nullptr,*idx_block_pos=nullptr;
    struct Pools {void *k_pool=nullptr,*v_pool=nullptr,*k_q=nullptr,*v_q=nullptr,*k_q4=nullptr,*v_q4=nullptr,*k_scale=nullptr,*v_scale=nullptr;} host;
    // Backing bytes are the recording allocation boundary, never a production
    // ABI/allocator or native size certificate. Allocated only after real lease.
    std::array<std::vector<std::byte>,8> backing;
};
// ACTUAL_QSA_FORMAT inserted into the generated TU from production session.hpp.
int qsa_kv_format(const QsaState&);
struct ReadContext {bool close_for_retirement(){return true;}};
struct SessionState {
    SessionState(){++seam::sessions_alive;}
    ~SessionState(){if(seam::image_mode&&seam::pending())std::abort();delete[] qsa_states;--seam::sessions_alive;++seam::sessions_destroyed;}
    int device=-1;int64_t layer_lo=0,layer_hi=0,max_cells=0,k=0,gdn_ord0=0;
    QsaState* qsa_states=nullptr;int64_t qsa_alloc=1,qsa_ord0=0;
    int64_t gdn_alloc=0;void* gdn_state=nullptr;
    std::vector<std::byte> recorded_gdn,recorded_ple;
    void prepare_recording(const ModelGeometry&);
    void* ple_hist=nullptr;int32_t ple_prev[2]={-1,-1},ple_token=-1;
    struct {std::shared_ptr<ReadContext> read_context;}ple;
};
inline bool session_zero_checked(SessionState& state,const ModelGeometry&,const float*,void*,std::string& error){
    if(++seam::zero_calls==seam::zero_fail){error="recording zero refusal";return false;}
    if(seam::image_mode){
        if(seam::pending())std::abort();
        std::fill(state.recorded_gdn.begin(),state.recorded_gdn.end(),std::byte{});
        std::fill(state.recorded_ple.begin(),state.recorded_ple.end(),std::byte{});
        for(int64_t i=0;i<state.qsa_alloc;++i)for(auto& bytes:state.qsa_states[state.qsa_ord0+i].backing)
            std::fill(bytes.begin(),bytes.end(),std::byte{});
        state.ple_prev[0]=state.ple_prev[1]=-1;state.ple_token=-1;
    }
    seam::events.push_back("zero"+std::to_string(seam::device));return true;
}
inline bool qsa_state_release_host(QsaState&,std::string&){return true;}
inline void session_release(SessionState&){seam::events.push_back("release"+std::to_string(seam::device));}
}
