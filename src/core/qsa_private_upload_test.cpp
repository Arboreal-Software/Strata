#include "strata/core/layer.hpp"
#include <cuda_runtime.h>
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace strata::core;
static int copies, fail_copy, registrations, resets;
static bool fail_sync, fail_launch;
static const void* first_source;
static void check(bool value,const char* message) {
    if(!value) { std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1); }
}
extern "C" cudaError_t cudaMemcpy(void* out,const void* in,size_t n,cudaMemcpyKind) {
    ++copies;if(copies==1)first_source=in;
    if(copies==fail_copy)return cudaErrorUnknown;
    std::memcpy(out,in,n);return cudaSuccess;
}
extern "C" cudaError_t cudaDeviceSynchronize() {return fail_sync?cudaErrorUnknown:cudaSuccess;}
extern "C" cudaError_t cudaGetLastError() {return fail_launch?cudaErrorUnknown:cudaSuccess;}
extern "C" const char* cudaGetErrorString(cudaError_t) {return "injected";}
namespace strata::kernels {
const RopeScaling& rope_scaling() {static RopeScaling value;return value;}
void build_rope_table(int n,const RopeScaling&,int cells,float* c,float* s) {
    for(int i=0;i<cells*(n/2);++i){c[i]=float(i);s[i]=-float(i);}
}
void rope_table_set(const float*,const float*,int,const RopeScaling&) {++registrations;}
void kv_stream_reset(const KvStreamMap&,void*) {++resets;}
void kv_ring_table(int32_t*,int64_t,int64_t,void*) {++resets;}
}
int main() {
    float cosine[64],sine[64];int32_t pages[2];
    auto state = [&] {
        copies=registrations=resets=fail_copy=0;fail_sync=fail_launch=false;first_source=nullptr;
        QsaState s;s.max_cells=2;s.n_pages=2;s.n_slots=2;s.cos_tab=cosine;s.sin_tab=sine;s.page_table=pages;return s;
    };
    for(int failure=1;failure<=4;++failure) {
        auto s=state();QsaInitUploads uploads;
        if(failure<=3)fail_copy=failure;else fail_sync=true;
        check(!qsa_upload_private_initial_state(s,64,true,uploads),"copy or sync failure rejected");
        check(uploads.started&&!uploads.drained&&!uploads.error.empty(),"failure is sticky and not drained");
        check(!s.owns_rope&&registrations==0,"failed state never registered");
        check(first_source==uploads.rope_cos.data()&&uploads.rope_cos.size()==64&&uploads.page_table.size()==2,
              "all potentially referenced sources remain owned on failure");
        fail_copy=0;fail_sync=false;
        check(!qsa_upload_private_initial_state(s,64,true,uploads),"later success cannot overwrite failed attempt");
    }
    {
        auto s=state();QsaInitUploads uploads;
        check(qsa_upload_private_initial_state(s,64,true,uploads),"resident uploads complete");
        check(copies==3&&registrations==0&&s.owns_rope&&uploads.drained,"private table ownership never publishes global lookup");
        check(pages[0]==0&&pages[1]==1&&cosine[63]==63&&sine[63]==-63,"uploaded contents preserved");
    }
    for(int mode=1;mode<=2;++mode) {
        auto s=state();s.kv_mode=mode;QsaInitUploads failed;
        fail_launch=true;
        check(!qsa_upload_private_initial_state(s,64,false,failed)&&resets==1,"stream/ring launch error retained");
        s=state();s.kv_mode=mode;QsaInitUploads success;
        check(qsa_upload_private_initial_state(s,64,false,success)&&resets==1&&copies==0&&registrations==0,
              "borrowed RoPE is not re-uploaded or registered");
    }
    auto s=state();s.page_table=nullptr;QsaInitUploads invalid;
    check(!qsa_upload_private_initial_state(s,64,true,invalid)&&copies==0,"invalid layout rejected before copying");
    std::puts("qsa_private_upload_test passed");
}
