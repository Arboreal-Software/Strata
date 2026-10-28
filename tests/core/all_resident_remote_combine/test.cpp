#include "strata/core/remote_expert_opt.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static int additions=0,copies=0;
namespace strata::kernels {
// CPU arithmetic seams. These do not execute CUDA kernels/graphs/mapped IO.
void add_inplace(float* d,const float* s,long long n,void*){++additions;for(long long i=0;i<n;++i)d[i]+=s[i];}
void copy_or_zero_from_mapped(float* d,const float* s,long long n,const uint32_t* skip,uint32_t ring,void*){
 ++copies;for(long long i=0;i<n;++i)d[i]=*skip==ring?0:s[i];
}
}
namespace strata::core {
constexpr int H=4,K=2;
class RemoteExperts {
public:
 float* h_out_;unsigned char owned_[4]={1,1,1,1};
 static void bind(RemoteExpertOpt& o,float* sum,int32_t* mask){o.h_sum_=o.m_sum_=sum;o.h_mask_=o.m_mask_=mask;}
 void accumulate_into(RemoteExpertOpt& o){o.accumulate(*this);}
};
// No production allocations are made by this fixture; caller arrays own storage.
RemoteExpertOpt::~RemoteExpertOpt()=default;
#include "methods.inc"
}
struct Stage {
 bool all_resident_=false,ar_off_=false,device_plan_=false;
 strata::core::RemoteExpertOpt* remote_opt_;
 float* bo_;float* y_dummy_;uint32_t* skip_;
 int tb=0,n=1,grp=0;static constexpr int N=4;uint32_t ring=7;void* cs=nullptr;
#include "ar_on.inc"
 void combine(){
#ifdef BEFORE
#include "before.inc"
#else
#include "fixed.inc"
#endif
 }
};
static void check(bool ok,const char* text){if(!ok){std::fprintf(stderr,"ASSERT: %s\n",text);std::exit(86);}}
int main(){
 float sum[8]={},weights[4]={1,1,1,1},remote_values[8]={3,5,7,9,11,13,15,17},out[8]={},scratch[8]={};
 int32_t mask[4]={};uint32_t skip[1]={0};strata::core::RemoteExpertOpt opt;
 strata::core::RemoteExperts::bind(opt,sum,mask);strata::core::RemoteExperts remote{remote_values};
 Stage stage{false,false,false,&opt,out,scratch,skip};
 // Previous host-served stage leaves a nonzero reduced remote contribution.
 opt.begin(weights,0,1);remote.accumulate_into(opt);opt.end();
 std::fill(out,out+8,10);stage.combine();check(out[0]==13&&out[3]==19,"host-served stage includes remote sum");
 // Next zero-doorbell stage has no begin/accumulate and must never read that sum.
 stage.all_resident_=true;std::fill(out,out+8,20);int before=additions;stage.combine();
 check(out[0]==20&&out[3]==20&&additions==before,"all-resident stage must not add stale remote sum");
 // Device-plan skip is stale too: resident-only graph never updates its ring.
 stage.device_plan_=true;skip[0]=1;std::fill(out,out+8,30);before=copies;stage.combine();
 check(out[0]==30&&copies==before,"all-resident device-plan variant must not consume stale skip or sum");
 // Adaptive fallback turns off all-resident capture, restoring ordinary combine.
 stage.ar_off_=true;opt.begin(weights,0,1);remote.accumulate_into(opt);opt.end();stage.combine();
 check(out[0]==33&&out[3]==39,"adaptive ar-off variant still includes remote contribution");
 // Existing host-path device-plan skip behavior is preserved.
 skip[0]=stage.ring;std::fill(out,out+8,40);stage.combine();check(out[0]==40&&out[3]==40,"host-path primary-only skip still zeros remote contribution");
 // No-helper path leaves the primary result unchanged.
 stage.remote_opt_=nullptr;before=additions;stage.combine();check(additions==before,"no remote optimizer performs no addition");
 std::puts("PASS: actual remote sum producers/combine and all-resident callsite");
}
