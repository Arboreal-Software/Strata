#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <cstdint>
#include <string>
namespace strata::core {
class NativeRetirementContext;
struct QsaElasticPlan {};
struct QsaElasticView {
    bool prepare_retirement(NativeRetirementContext&){return true;}
    bool commit_retirement(NativeRetirementContext&){return true;}
};
struct ModelGeometry {int64_t n_layers=4,n_vocab=1000;int64_t n_qsa_layers()const{return 2;}};
struct QsaState {bool kv_hybrid=false,kv_rot=false,kv_q4=false;};
inline int qsa_kv_format(const QsaState&){return 0;}
struct ReadContext {bool close_for_retirement(){return true;}};
struct SessionState {
    SessionState(){++seam::sessions_alive;}
    ~SessionState(){delete[] qsa_states;--seam::sessions_alive;++seam::sessions_destroyed;}
    int device=-1;int64_t layer_lo=0,layer_hi=0,max_cells=0,k=0,gdn_ord0=0;
    QsaState* qsa_states=nullptr;int64_t qsa_alloc=1,qsa_ord0=0;
    void* ple_hist=nullptr;int32_t ple_prev[2]={-1,-1},ple_token=-1;
    struct {std::shared_ptr<ReadContext> read_context;}ple;
};
inline bool session_zero_checked(SessionState&,const ModelGeometry&,const float*,void*,std::string& error){
    if(++seam::zero_calls==seam::zero_fail){error="recording zero refusal";return false;}
    seam::events.push_back("zero"+std::to_string(seam::device));return true;
}
inline bool qsa_state_release_host(QsaState&,std::string&){return true;}
inline void session_release(SessionState&){seam::events.push_back("release"+std::to_string(seam::device));}
}
