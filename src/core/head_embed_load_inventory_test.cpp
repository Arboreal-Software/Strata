#include "strata/core/native_head.hpp"
#include "strata/core/geometry.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <unistd.h>
#include <sys/wait.h>
using namespace strata::core;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static int device=1,allocs=0,copies=0,frees=0,host_frees=0,fail_alloc=0,fail_copy=0,fail_free=0;
static bool fail_get_device=false,fail_set_device=false,fail_restore=false,q6_enabled=false;
static bool fail_host=false,fail_alias=false,fail_sync=false,fail_host_free=false,pack_throw=false;
static std::map<void*,size_t> live,host_live;
static std::map<const void*,void*> packed;
extern "C" cudaError_t cudaMalloc(void** p,size_t n){if(++allocs==fail_alloc){*p=nullptr;return cudaErrorMemoryAllocation;}*p=std::malloc(n);live[*p]=n;return cudaSuccess;}
extern "C" cudaError_t cudaFree(void* p){check(device==1,"device allocation released on owning device");if(++frees==fail_free)return cudaErrorUnknown;check(live.erase(p)==1,"one device allocation released once");std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaMemcpy(void* d,const void* s,size_t n,cudaMemcpyKind){if(++copies==fail_copy)return cudaErrorUnknown;std::memcpy(d,s,n);return cudaSuccess;}
extern "C" cudaError_t cudaHostAlloc(void** p,size_t n,unsigned int){if(fail_host){*p=nullptr;return cudaErrorMemoryAllocation;}*p=std::malloc(n);host_live[*p]=n;return cudaSuccess;}
extern "C" cudaError_t cudaFreeHost(void* p){++host_frees;if(fail_host_free)return cudaErrorUnknown;check(host_live.erase(p)==1,"one pinned allocation released once");std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaHostGetDevicePointer(void** p,void* h,unsigned int){if(fail_alias){*p=nullptr;return cudaErrorUnknown;}*p=h;return cudaSuccess;}
extern "C" cudaError_t cudaGetDevice(int* p){if(fail_get_device)return cudaErrorUnknown;*p=device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){if(fail_set_device||(fail_restore&&d==0))return cudaErrorUnknown;device=d;return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){return fail_sync?cudaErrorUnknown:cudaSuccess;}
extern "C" cudaError_t cudaGetLastError(){return cudaSuccess;}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected";}
namespace strata::kernels {
bool native_mmvq_supported(int t) noexcept{return t==8||t==14;}
size_t native_mmvq_weight_bytes(int t,int a,int b){return t==14?size_t(a)*b/256*210:size_t(a)*b/32*34;}
size_t native_q8_1_bytes(int a,int b){return size_t(a)*b/32*36;}
bool native_q6_k_packed_enabled(){return q6_enabled;}
bool native_q6_k_pack(const void* key,int,int,const char*){void* p=nullptr;if(cudaMalloc(&p,128)!=cudaSuccess)return false;packed[key]=p;return true;}
bool native_q8_0_sm60_pack(const void* key,int,int,const char*){void* p=nullptr;if(cudaMalloc(&p,128)!=cudaSuccess)return false;packed[key]=p;if(pack_throw)throw std::bad_alloc();return true;}
size_t native_head_packed_allocation_bytes(const void* key){return packed.count(key)?128:0;}
bool native_head_packed_release_checked(const void* key){auto it=packed.find(key);if(it==packed.end())return true;if(cudaFree(it->second)!=cudaSuccess)return false;packed.erase(it);return true;}
bool embed_type_supported(int t)noexcept{return t==30;}
size_t iq_row_bytes(int,int64_t n)noexcept{return size_t(n)*2;}
}
template<class T>static void put(std::vector<uint8_t>& b,T v){auto n=b.size();b.resize(n+sizeof v);std::memcpy(b.data()+n,&v,sizeof v);}
static void str(std::vector<uint8_t>& b,const std::string& s){put<uint64_t>(b,s.size());b.insert(b.end(),s.begin(),s.end());}
static std::string fixture(bool empty=false,int head_type=8){
 std::vector<uint8_t>b;put<uint32_t>(b,0x46554747);put<uint32_t>(b,3);put<uint64_t>(b,empty?0:2);put<uint64_t>(b,(empty||head_type==14)?10:7);
 str(b,"general.architecture");put<uint32_t>(b,8);str(b,"qwen4exp");const auto& g=kCompiledGeometry;
 for(const auto& [key,value]:std::vector<std::pair<std::string,int64_t>>{{"block_count",g.n_layers},{"embedding_length",g.n_embd},{"expert_count",g.n_expert},{"expert_used_count",g.n_expert_used},{"attention.head_count",g.n_head},{"attention.head_count_kv",g.n_head_kv}}){str(b,"qwen4exp."+key);put<uint32_t>(b,10);put<uint64_t>(b,value);}
 if(empty||head_type==14){for(const auto& kv:std::vector<std::pair<std::string,uint64_t>>{{"split.count",2},{"split.no",empty?0u:1u},{"split.tensors.count",2}}){str(b,kv.first);put<uint32_t>(b,10);put<uint64_t>(b,kv.second);}}
 if(!empty){str(b,"output.weight");put<uint32_t>(b,2);put<uint64_t>(b,256);put<uint64_t>(b,32);put<uint32_t>(b,head_type);put<uint64_t>(b,0);
 str(b,"token_embd.weight");put<uint32_t>(b,2);put<uint64_t>(b,256);put<uint64_t>(b,32);put<uint32_t>(b,30);put<uint64_t>(b,8704);}
 b.resize((b.size()+31)/32*32+8704+16384,0);
 const auto path="/tmp/strata-head-fixture-"+std::to_string(getpid())+(empty?"-empty":head_type==14?"-q6":"")+".gguf";std::ofstream(path,std::ios::binary).write(reinterpret_cast<const char*>(b.data()),b.size());return path;
}
static void reset(){check(live.empty()&&host_live.empty()&&packed.empty(),"all normal fixture allocations released");allocs=copies=frees=host_frees=fail_alloc=fail_copy=fail_free=0;fail_host=fail_alias=fail_sync=fail_host_free=pack_throw=false;device=1;}
static void reclaim_quarantine(){for(auto [p,n]:live)std::free(p);live.clear();for(auto [p,n]:host_live)std::free(p);host_live.clear();packed.clear();}
int baseline_main(){const auto path=fixture();std::string error;
 for(int scenario=0;scenario<6;++scenario){reset();fail_alloc=scenario==1?1:scenario==2?2:0;fail_copy=scenario==3?1:0;fail_sync=scenario==4;pack_throw=scenario==5;
  {NativeHead h;check(!h.ready_for_adoption(),"unloaded optional head not adoptable");const bool ok=h.load({path},256,32,error);
   check(ok==(scenario==0),"head load success/failure matches injection");check(h.ready_for_adoption()==ok,"head readiness tracks complete load");
   fail_sync=false;
   if(ok){check(h.allocated_bytes()==8704+288+128,"head allocation includes scratch and packed copy");const int before=allocs;check(!h.load({path},256,32,error)&&allocs==before&&h.ready_for_adoption(),"duplicate head load preserves ready model");
    device=0;h.unload();check(device==0&&!h.loaded()&&h.allocated_bytes()==0&&h.weight_bytes()==8704,"unload releases on owning device and preserves informational weight size");device=1;check(h.load({path},256,32,error),"real head reload after unload");}
  }check(live.empty()&&packed.empty(),"failed or successful head teardown releases owned allocations");
 }
 for(int failed_free=1;failed_free<=3;++failed_free){reset();auto h=std::make_unique<NativeHead>();check(h->load({path},256,32,error),"release failure head loads");fail_free=failed_free;
  check(!h->release_checked(error)&&!h->ready_for_adoption(),"head uncertain cleanup fences adoption");
  check(h->allocated_bytes()==uint64_t(failed_free==1?9120:failed_free==2?8992:8704),"partial retirement clears successful allocations while retaining uncertain bytes");int calls=frees;check(!h->release_checked(error)&&frees==calls,"head cleanup failure is sticky");h.reset();check(frees==calls,"destructor does not retry uncertain free");reclaim_quarantine();
 }
 for(int scenario=0;scenario<6;++scenario){reset();fail_host=scenario==1||scenario==3||scenario==4;fail_alias=scenario==2;fail_copy=scenario==3?1:0;fail_alloc=scenario==4?1:0;fail_sync=scenario==5;
  {NativeEmbed e;const bool ok=e.load({path},256,32,error);
#if defined(STRATA_USE_HIP)
   const bool expected=scenario<=2;
#else
   const bool expected=scenario<=1;
#endif
   check(ok==expected&&e.ready_for_adoption()==ok,"embedding readiness follows actual mapping/fallback completion");
   if(ok){check(e.device()==1,"embedding retains allocation device");const bool fallback=scenario==1||scenario==2;check(e.allocated_device_bytes()==(fallback?16384:0)&&e.allocated_host_bytes()==(fallback?0:16384)&&e.allocated_pinned_bytes()==e.allocated_host_bytes(),"embedding host/pinned/device accounting distinguishes fallback");
    check(!e.load({path},256,32,error)&&e.ready_for_adoption(),"duplicate embedding load preserves ready model");set_native_embed(&e);}
   fail_sync=false;device=0;
  }check(device==0&&live.empty()&&host_live.empty()&&!native_embed(),"embedding destructor releases on owner device and clears global alias");
 }
 for(bool fallback:{false,true}){reset();fail_host=fallback;auto e=std::make_unique<NativeEmbed>();check(e->load({path},256,32,error),"embedding retirement fixture loads");fail_free=fallback?1:0;fail_host_free=!fallback;
  check(!e->release_checked(error)&&!e->ready_for_adoption(),"embedding free failure fences adoption");const int calls=frees+host_frees;check(!e->release_checked(error),"embedding failure sticky");e.reset();check(frees+host_frees==calls,"embedding destructor does not retry uncertain release");reclaim_quarantine();
 }
#if defined(STRATA_USE_HIP)
 reset();fail_alias=true;fail_host_free=true;
 {auto e=std::make_unique<NativeEmbed>();check(!e->load({path},256,32,error)&&!e->ready_for_adoption(),"HIP fallback host retirement failure rejects model");
  check(e->allocated_device_bytes()==16384&&e->allocated_host_bytes()==16384&&e->allocated_pinned_bytes()==16384,"uncertain fallback retains and counts both allocations");
  const int before=frees+host_frees;e.reset();check(frees+host_frees==before,"HIP fallback failure does not retry uncertain pinned free");}
 reclaim_quarantine();
#endif
 reset();std::remove(path.c_str());std::puts("native_head_lifetime_test passed");return 0;}

static std::shared_ptr<const ReservationLease> reserve(WindowBudget& b,uint64_t n){return b.reserve({{0},n,0,0,0,0},ReservationPurpose::foreground).lease;}
int main(){
 baseline_main();
 const auto path=fixture();std::string error;NativeInventoryLimits limits{2,1,2,4096};uint64_t bytes=0;
 check(head_embed_inventory_host_bound(limits,bytes,error),"inventory bound");
 for(int mode=0;mode<3;++mode){reset();WindowBudget budget({{0},bytes*2,0,0,0,0},{{0},0,0,0,0,0});
 auto owner=std::make_shared<int>(42);std::weak_ptr<int> weak=owner;
 {
 NativeHead head;
 check(!head.enable_inventory_capture(limits,reserve(budget,bytes-1),owner,error),"undercharge refused");
 check(head.enable_inventory_capture(limits,reserve(budget,bytes),owner,error),"capture admission");owner.reset();
 if(mode==1)fail_alloc=3;
 if(mode==2)fail_copy=1;
 const bool ok=head.load({path},256,32,error);check(ok==(mode!=2),"capture load actual outcome");
 if(ok){const auto* receipt=head.loaded_inventory();check(receipt&&receipt->generation==1&&receipt->tensor.source_type==8&&receipt->tensor.source_bytes==8704&&receipt->tensor.source_shape==std::vector<uint64_t>({256,32}),"actual source metadata");
 check(receipt->sources==std::vector<std::string>({path})&&receipt->tensor.source_offset>0&&receipt->scratch_bytes==288,"actual source extent and scratch");
 check(receipt->packed==(mode?HeadPackedForm::none:HeadPackedForm::q8_sm60_qs_d)&&receipt->helper_bytes==(mode?0:128),"actual pack success not requested flag");
 check(!head.enable_inventory_capture(limits,reserve(budget,bytes),weak.lock(),error)&&head.loaded_inventory()==receipt,"refused enable preserves published view");
 }
 head.unload();check(!head.loaded_inventory()&&!weak.expired(),"unload revokes borrow retains authority until rearm/destruction");
 check(!head.load({path},256,32,error),"captured reload requires fresh admission");
 auto next=std::make_shared<int>(7);check(head.enable_inventory_capture(limits,reserve(budget,bytes),next,error),"rearm");check(weak.expired(),"old input released after retired replacement");
 fail_alloc=fail_copy=0;check(head.load({path},256,32,error),"reload");check(head.loaded_inventory()->generation==uint64_t(ok?2:1),"successful generation only");
 }
 check(budget.snapshot().reserved.host_bytes==0,"normal receipt charge released after resources");
 }
 for(int mode=0;mode<3;++mode){reset();WindowBudget budget({{0},bytes,0,0,0,0},{{0},0,0,0,0,0});auto owner=std::make_shared<int>(1);
 {NativeEmbed embed;check(embed.enable_inventory_capture(limits,reserve(budget,bytes),owner,error),"embedding admission");fail_host=mode==1;fail_alias=mode==2;
 const bool ok=embed.load({path},256,32,error);
#if defined(STRATA_USE_HIP)
 check(ok,"HIP placement outcome");
#else
 check(ok==(mode!=2),"CUDA placement outcome");
#endif
 if(ok){auto* r=embed.loaded_inventory();check(r&&r->role==HeadEmbedRole::embedding&&r->row_bytes==512&&r->tensor.source_bytes==16384&&r->tensor.source_type==30,"actual embedding source");
 check(r->placement==(mode==0?EmbedPlacement::mapped_portable:mode==1?EmbedPlacement::device_pin_fallback:EmbedPlacement::device_alias_fallback),"actual placement receipt");}
 else check(!embed.loaded_inventory(),"failed alias publishes no receipt");}
 check(budget.snapshot().reserved.host_bytes==0,"embedding normal charge release");}


 {reset();const auto empty=fixture(true),q6=fixture(false,14);WindowBudget budget({{0},bytes,0,0,0,0},{{0},0,0,0,0,0});
 auto owner=std::make_shared<int>(1);NativeHead h;q6_enabled=true;
 check(h.enable_inventory_capture(limits,reserve(budget,bytes),owner,error),"second shard admission");
 const bool loaded=h.load({empty,q6},256,32,error);if(!loaded)std::fprintf(stderr,"Q6 fixture: %s\n",error.c_str());check(loaded,"actual second shard Q6");
 strata::GgufFile file(q6);auto* tensor=file.find("output.weight");
 auto* r=h.loaded_inventory();check(r&&r->tensor.source_ordinal==1&&r->tensor.source_type==14&&r->tensor.source_offset==file.data_start()+tensor->offset&&r->packed==HeadPackedForm::q6_planes_256,"exact selected shard span and Q6 result");
 h.unload();q6_enabled=false;std::remove(empty.c_str());std::remove(q6.c_str());}
 for(int bound=0;bound<3;++bound){reset();WindowBudget budget({{0},bytes,0,0,0,0},{{0},0,0,0,0,0});auto owner=std::make_shared<int>(1);NativeHead h;auto small=limits;
 if(bound==0)small.max_sources=1;
 if(bound==1)small.max_string_bytes=1;
 if(bound==2)small.max_rank=1;
 if(bound==2)check(!h.enable_inventory_capture(small,reserve(budget,bytes),owner,error),"rank admission refused");
 else {check(h.enable_inventory_capture(small,reserve(budget,bytes),owner,error),"small envelope enabled");check(!h.load(bound==0?std::vector<std::string>{path,path}:std::vector<std::string>{path},256,32,error)&&allocs==0&&!h.loaded_inventory(),"metadata refusal before GPU allocations");}}

 for(bool embedding:{false,true})for(int fault=0;fault<6;++fault){reset();WindowBudget budget({{0},bytes,0,0,0,0},{{0},0,0,0,0,0});auto owner=std::make_shared<int>(4);
 {NativeHead h;NativeEmbed e;check(embedding?e.enable_inventory_capture(limits,reserve(budget,bytes),owner,error):h.enable_inventory_capture(limits,reserve(budget,bytes),owner,error),"failed-load capture admitted");
 if(embedding){fail_host=fault<=2;fail_alloc=fault==0?1:0;fail_copy=fault==1?1:0;fail_alias=fault==3;fail_sync=fault>=4;}
 else {fail_alloc=fault==0?1:fault==1?2:0;fail_copy=fault==2?1:0;pack_throw=fault==3;fail_sync=fault>=4;}
 const bool ok=embedding?e.load({path},256,32,error):h.load({path},256,32,error);
 bool expected=embedding&&fault==2;
#if defined(STRATA_USE_HIP)
 expected=expected||(embedding&&fault==3);
#endif
 check(ok==expected,"captured load injected outcome");check(bool(embedding?e.loaded_inventory():h.loaded_inventory())==expected,"failure never publishes partial receipt");fail_sync=false;
 }
 check(live.empty()&&host_live.empty()&&budget.snapshot().reserved.host_bytes==0,"failed-load resources and metadata release together");}
 for(bool embedding:{false,true})for(int fault=0;fault<6;++fault){pid_t pid=fork();check(pid>=0,"fork uncertainty");if(pid==0){
 reset();WindowBudget budget({{0},bytes,0,0,0,0},{{0},0,0,0,0,0});auto owner=std::make_shared<int>(3);std::weak_ptr<int> weak=owner;
 auto h=std::make_unique<NativeHead>();auto e=std::make_unique<NativeEmbed>();
 check(embedding?e->enable_inventory_capture(limits,reserve(budget,bytes),owner,error):h->enable_inventory_capture(limits,reserve(budget,bytes),owner,error),"uncertainty admitted");owner.reset();
 check(embedding?e->load({path},256,32,error):h->load({path},256,32,error),"uncertainty loaded");
 if(fault==0){fail_free=1;fail_host_free=true;}if(fault==1)fail_sync=true;if(fault==2)fail_get_device=true;
 if(fault==3){device=0;fail_set_device=true;}
 if(fault==5){device=0;fail_restore=true;}
 if(fault==4){fail_free=1;fail_host_free=true;check(!(embedding?e->release_checked(error):h->release_checked(error)),"explicit failure");}
 const int before=frees+host_frees;h.reset();e.reset();
 if(fault==4)check(before==frees+host_frees,"sticky failure no destructor retry");
 check(!weak.expired()&&budget.snapshot().reserved.host_bytes==bytes,"uncertain destruction retains actual authority and charge");std::_Exit(0);
 }int status=0;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"uncertainty subprocess");}
 std::remove(path.c_str());std::puts("head_embed_load_inventory_test passed");return 0;
}
