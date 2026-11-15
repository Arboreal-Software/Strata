#include "strata/core/session.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace strata::core;
static int calls,fail_at;static bool elastic;
static void check(bool ok,const char* message) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}
}
namespace strata::core {
bool qsa_kv_elastic(){return elastic;}
uint64_t gdn_buffers_bytes(const ModelGeometry&){return 256;}
uint64_t gdn_buffers_init(const ModelGeometry&,void*,GdnBuffers&){return 256;}
uint64_t qsa_state_bytes(const ModelGeometry&,int64_t,bool,int64_t){return 256;}
uint64_t qsa_buffers_bytes(const ModelGeometry&,int64_t){return 256;}
uint64_t qsa_buffers_init(const ModelGeometry&,int64_t,void*,QsaBuffers&){return 256;}
uint64_t moe_buffers_bytes(const ModelGeometry&,int64_t){return 256;}
uint64_t moe_buffers_init(const ModelGeometry&,int64_t,void*,MoEBuffers&){return 256;}
uint64_t block_buffers_bytes(const ModelGeometry&){return 256;}
uint64_t block_buffers_init(const ModelGeometry&,void*,BlockBuffers&){return 256;}
uint64_t qsa_state_init(const ModelGeometry&,int64_t,void*,QsaState&,const QsaState*,int64_t){std::abort();}
uint64_t qsa_state_init_private(const ModelGeometry&,int64_t,void* base,QsaState& state,
                                QsaInitUploads& uploads,const QsaState* shared) {
    ++calls;uploads.started=true;uploads.rope_cos.resize(4,float(calls));state.cos_tab=static_cast<float*>(base);
    if(calls==1)check(!shared,"first allocated ordinal owns its RoPE");
    else check(shared&&shared->cos_tab!=state.cos_tab,"later ordinal borrows earlier initialized state");
    if(calls==fail_at){uploads.error="injected partial initialization";return 0;}
    uploads.drained=true;return 256;
}
}
int main(){
    ModelGeometry g;g.n_layers=16;g.qsa_interval=4;g.ssm_state_size=1;g.ssm_v_heads=1;g.ssm_conv_channels=1;g.ssm_d_conv=2;
    std::vector<unsigned char> arena(16*1024*1024);
    for(int failure=0;failure<=2;++failure){
        SessionState state;SessionInitUploads uploads;calls=0;fail_at=failure;
        const auto bytes=session_init_private(g,16,1,arena.data(),state,uploads,8,16);
        check((bytes!=0)==(failure==0),"private init reports first failing stage");
        check(calls==(failure?failure:2)&&state.qsa_ord0==2&&state.qsa_alloc==2,"split global ordinals initialized");
        check(!uploads.qsa[0].started&&!uploads.qsa[1].started&&uploads.qsa[2].started,"uploads indexed by global ordinal");
        if(failure)check(!uploads.error.empty()&&uploads.qsa[size_t(1+failure)].rope_cos.size()==4,
                         "partial initialization retains failing source storage");
        check(!session_init_private(g,16,1,arena.data(),state,uploads,8,16),"attempt cannot be reused");
        delete[] state.qsa_states;
    }
    SessionState state;SessionInitUploads uploads;elastic=true;calls=0;
    check(!session_init_private(g,16,1,arena.data(),state,uploads,8,16)&&calls==0&&!state.qsa_states,
          "unowned elastic pools refused before mutation");
    std::puts("session_private_init_test passed");
}
