#include "strata/core/native_head.hpp"
#include "strata/core/window_model_owner.hpp"
#include "strata/core/window_loader_profile.hpp"
#include "strata/core/verified_input_manifest.hpp"
#include <algorithm>
#include <iterator>
#include <filesystem>
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
static bool skip_head_pack=false;
static int sync_calls=0,fail_sync_at=0;
static WindowBudget* spill_observer=nullptr;
static WindowBudget* spill_retirement=nullptr;
static uint64_t spill_peak_device=0,spill_peak_host=0,spill_peak_pinned=0;
static void observe_spill_peak() {
 if(!spill_observer)return;
 const auto actual=spill_observer->snapshot().reserved;
 check(actual.device_bytes.size()==2&&actual.device_bytes[1]==spill_peak_device&&
       actual.host_bytes==spill_peak_host&&actual.pinned_bytes==spill_peak_pinned,
       "actual native call observes fully paid old-device plus new-host peak");
}
static std::map<void*,size_t> live,host_live;
static std::map<const void*,void*> packed;
extern "C" cudaError_t cudaMalloc(void** p,size_t n){if(++allocs==fail_alloc){*p=nullptr;return cudaErrorMemoryAllocation;}*p=std::malloc(n);live[*p]=n;return cudaSuccess;}
extern "C" cudaError_t cudaFree(void* p){observe_spill_peak();check(device==1,"device allocation released on owning device");if(++frees==fail_free)return cudaErrorUnknown;check(live.erase(p)==1,"one device allocation released once");std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaMemcpy(void* d,const void* s,size_t n,cudaMemcpyKind){if(++copies==fail_copy)return cudaErrorUnknown;std::memcpy(d,s,n);return cudaSuccess;}
extern "C" cudaError_t cudaHostAlloc(void** p,size_t n,unsigned int){observe_spill_peak();if(fail_host){*p=nullptr;return cudaErrorMemoryAllocation;}*p=std::malloc(n);host_live[*p]=n;return cudaSuccess;}
extern "C" cudaError_t cudaFreeHost(void* p){
 if(spill_retirement)check(spill_retirement->snapshot().reserved.host_bytes>=NativeHead::spill_control_bytes()+8704&&
                          spill_retirement->snapshot().reserved.pinned_bytes>=8704,"actual pinned free precedes release of payload and fixed credit");++host_frees;if(fail_host_free)return cudaErrorUnknown;check(host_live.erase(p)==1,"one pinned allocation released once");std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaHostGetDevicePointer(void** p,void* h,unsigned int){if(fail_alias){*p=nullptr;return cudaErrorUnknown;}*p=h;return cudaSuccess;}
extern "C" cudaError_t cudaGetDevice(int* p){if(fail_get_device)return cudaErrorUnknown;*p=device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){if(fail_set_device||(fail_restore&&d==0))return cudaErrorUnknown;device=d;return cudaSuccess;}
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p,int d){check(d==device,"actual spill device properties");*p={};return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){observe_spill_peak();++sync_calls;return (fail_sync||sync_calls==fail_sync_at)?cudaErrorUnknown:cudaSuccess;}
extern "C" cudaError_t cudaGetLastError(){return cudaSuccess;}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected";}
namespace strata::kernels {
bool native_mmvq_supported(int t) noexcept{return t==8||t==14;}
size_t native_mmvq_weight_bytes(int t,int a,int b){return t==14?size_t(a)*b/256*210:size_t(a)*b/32*34;}
size_t native_q8_1_bytes(int a,int b){return size_t(a)*b/32*36;}
bool native_q6_k_packed_enabled(){return q6_enabled;}
bool native_q6_k_pack(const void* key,int,int,const char*){void* p=nullptr;if(cudaMalloc(&p,128)!=cudaSuccess)return false;packed[key]=p;return true;}
bool native_q8_0_sm60_pack(const void* key,int,int,const char*){if(skip_head_pack)return false;void* p=nullptr;if(cudaMalloc(&p,128)!=cudaSuccess)return false;packed[key]=p;if(pack_throw)throw std::bad_alloc();return true;}
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
static void reset(){spill_observer=spill_retirement=nullptr;skip_head_pack=false;sync_calls=fail_sync_at=0;check(live.empty()&&host_live.empty()&&packed.empty(),"all normal fixture allocations released");allocs=copies=frees=host_frees=fail_alloc=fail_copy=fail_free=0;fail_host=fail_alias=fail_sync=fail_host_free=pack_throw=false;device=1;}
static void reclaim_quarantine(){for(auto [p,n]:live)std::free(p);live.clear();for(auto [p,n]:host_live)std::free(p);host_live.clear();packed.clear();}
int baseline_main(){const auto path=fixture();std::string error;
 for(int scenario=0;scenario<6;++scenario){reset();fail_host=scenario==1;fail_alloc=scenario==1?1:scenario==2?2:0;fail_copy=scenario==3?1:0;fail_sync=scenario==4;pack_throw=scenario==5;
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
static std::shared_ptr<const OwnedModelStage> spill_stage(
 WindowBudget& ledger,WindowBudget& metadata,const NativeInventoryLimits& limits,
 uint64_t metadata_bytes,const std::string& head_path,uint64_t fixed,std::string& error) {
 ModelStageResources input;input.device=1;
 input.weights=std::make_unique<WeightTable>();input.native=std::make_unique<NativeDense>();
 input.head=std::make_unique<NativeHead>();input.input_owner=std::make_shared<int>(41);
 check(cudaMalloc(&input.arena,32)==cudaSuccess,"actual spill stage arena");input.arena_bytes=32;
 char name[]="/tmp/strata-head-spill-pack-XXXXXX";check(mkdtemp(name),"actual spill pack directory");
 const std::filesystem::path directory(name);
 {std::ofstream index(directory/"index.txt");index<<"# align 32 pool 32 tensors 1\nfixture.weight 0 2 0 32 0 32 8 1 0 0 0 0 0 0 0 0 0 0\n";}
 {std::ofstream data(directory/"dense.bin",std::ios::binary);float values[8]={1,2,3,4,5,6,7,8};data.write(reinterpret_cast<const char*>(values),sizeof values);}
 check(input.weights->load(name,input.arena,32,error)&&input.weights->load_complete(),"actual canonical model weights loaded");
 std::filesystem::remove_all(directory);
 check(input.head->enable_inventory_capture(limits,reserve(metadata,metadata_bytes),input.input_owner,error),"actual spill inventory issuer");
 check(input.head->load({head_path},256,32,error),"actual spill head loader");
 input.reservation=ledger.reserve({{0,32+input.head->allocated_bytes()},fixed,0,0,0,0},ReservationPurpose::foreground).lease;
 check(bool(input.reservation),"actual existing stage reservation includes explicit fixed control");
 std::shared_ptr<const OwnedModelStage> stage;
 check(adopt_model_stage(input,stage,error)&&!input.head&&!input.arena,"genuine loaded stage adopts exclusively");
 return stage;
}
static void checked_spill_groups(const std::string& path,const NativeInventoryLimits& limits,uint64_t metadata_bytes) {
 std::string error;const uint64_t fixed=NativeHead::spill_control_bytes(),weight=8704;
 check(fixed>WindowBudget::head_spill_control_bytes(),"actual native control is added separately from ledger frame");
 for(int fault=0;fault<12;++fault) {
  reset();skip_head_pack=fault!=11;
  const WindowResources caps{{0,1ull<<20},fixed+weight-(fault==1),weight-(fault==2),0,0,0};
  const WindowResources zero{{0,0},0,0,0,0,0};
  WindowBudget ledger(caps,zero),foreign(caps,zero);
  WindowBudget metadata({{0},metadata_bytes,0,0,0,0},{{0},0,0,0,0,0});
  auto stage=spill_stage(ledger,metadata,limits,metadata_bytes,path,fixed-(fault==10),error);
  const auto* head=stage->head();const auto* original=head->weights();const auto before=ledger.snapshot().reserved;
  const auto* receipt=head->loaded_inventory();check(receipt&&receipt->generation==1,"actual spill source receipt");
  if(fault==3)fail_alias=true;
  if(fault==4)fail_copy=copies+1;
  if(fault==5)device=0;
  if(fault==9)fail_host=true;
  std::shared_ptr<const ReservationLease> shared;
  if(fault==6)shared=stage->reservation();
  if(fault==8)const_cast<HeadEmbedLoadInventory*>(receipt)->generation=UINT64_MAX;
  spill_peak_device=before.device_bytes[1];spill_peak_host=before.host_bytes+weight;spill_peak_pinned=weight;
  spill_observer=&ledger;spill_retirement=&ledger;
  const bool ok=stage->spill_head_startup(fault==7?foreign:ledger,error);
  spill_observer=nullptr;
  check(ok==(fault==0),"actual stage wrapper spill success/refusal matrix");
  if(ok) {
   check(head->on_host()&&head->allocated_bytes()==288&&head->allocated_host_bytes()==weight&&
         head->allocated_pinned_bytes()==weight,"actual mapped head device/host/pinned lower bounds");
   check(ledger.snapshot().reserved==WindowResources{{0,before.device_bytes[1]-weight},fixed+weight,weight,0,0,0},"actual stage commits only freed device weight credit");
   check(head->loaded_inventory()==receipt&&receipt->generation==2&&receipt->placement==EmbedPlacement::mapped_portable&&
         receipt->main_bytes==weight&&receipt->scratch_bytes==288,"actual final inventory, generation and unchanged layout");
   for(size_t i=0;i<weight;++i)check(static_cast<const unsigned char*>(head->weights())[i]==0,"actual mapped bytes unchanged");
   check(!stage->spill_head_startup(ledger,error),"mapped stage cannot repeat native move");
  } else {
   check(!head->on_host()&&head->weights()==original&&head->loaded()&&ledger.snapshot().reserved==before,
         "actual stage refusal/rollback preserves pointers/readiness/complete charge");
   if(fault==8)const_cast<HeadEmbedLoadInventory*>(receipt)->generation=1;
  }
  shared.reset();device=1;stage.reset();spill_retirement=nullptr;
  check(ledger.snapshot().reserved==zero&&metadata.snapshot().reserved.host_bytes==0,"actual stage frees native payload before both credits retire");
 }
 for(int fault=0;fault<4;++fault) {
  const pid_t child=fork();check(child>=0,"fork paid uncertainty");
  if(!child) {
   reset();skip_head_pack=true;
   const WindowResources zero{{0,0},0,0,0,0,0};
   WindowBudget ledger({{0,1ull<<20},fixed+weight,weight,0,0,0},zero);
   WindowBudget metadata({{0},metadata_bytes,0,0,0,0},{{0},0,0,0,0,0});
   auto stage=spill_stage(ledger,metadata,limits,metadata_bytes,path,fixed,error);
   spill_peak_device=stage->reservation()->resources().device_bytes[1];spill_peak_host=fixed+weight;spill_peak_pinned=weight;
   spill_observer=&ledger;spill_retirement=&ledger;
   if(fault==0)fail_sync=true;
   if(fault==1)fail_free=frees+1;
   if(fault==2){fail_alias=true;fail_host_free=true;}
   if(fault==3){fail_copy=copies+1;fail_sync_at=sync_calls+2;}
   (void)stage->spill_head_startup(ledger,error);std::_Exit(9);
  }
  int status=0;check(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==86,
                    "paid uncertainty failstops before native/credit owner unwinding");
 }
 reset();
}
// This group alone constructs genuine verified authority. All older recording
// callbacks and test groups above remain unchanged.
static StateDigest collector_digest(const std::string& bytes) {
 WindowHasher hash;hash.update(bytes.data(),bytes.size());return hash.digest();
}
static std::string collector_hex(const StateDigest& digest) {
 std::string out;constexpr char digits[]="0123456789abcdef";
 for(auto b:digest){out+=digits[b>>4];out+=digits[b&15];}return out;
}
static void collector_write(const std::filesystem::path& path,const std::string& bytes) {
 std::ofstream out(path,std::ios::binary|std::ios::trunc);out.write(bytes.data(),std::streamsize(bytes.size()));
 out.close();check(bool(out),"collector fixture bytes written");
}
static std::string collector_read(const std::filesystem::path& path) {
 std::ifstream in(path,std::ios::binary);check(bool(in),"collector fixture source open");
 std::string bytes((std::istreambuf_iterator<char>(in)),{});
 check(!in.bad()&&bytes.size()<=65536,"collector fixture bounded source read");return bytes;
}
static std::string collector_projection() {
 std::vector<uint8_t> b;put<uint32_t>(b,0x46554747);put<uint32_t>(b,3);put<uint64_t>(b,1);put<uint64_t>(b,7);
 str(b,"general.architecture");put<uint32_t>(b,8);str(b,"qwen4exp");const auto& g=kCompiledGeometry;
 for(const auto& [key,value]:std::vector<std::pair<std::string,int64_t>>{{"block_count",g.n_layers},{"embedding_length",g.n_embd},{"expert_count",g.n_expert},{"expert_used_count",g.n_expert_used},{"attention.head_count",g.n_head},{"attention.head_count_kv",g.n_head_kv}}){str(b,"qwen4exp."+key);put<uint32_t>(b,10);put<uint64_t>(b,value);}
 // A real eligible Q6_K row uses existing MMVQ geometry and copies only; unlike
 // Q8 it does not enter the unqualified optional packed-provider callbacks.
 str(b,"blk.0.attn_q.weight");put<uint32_t>(b,2);put<uint64_t>(b,256);put<uint64_t>(b,1);put<uint32_t>(b,14);put<uint64_t>(b,0);
 b.resize((b.size()+31)/32*32+210,0);
 return std::string(reinterpret_cast<const char*>(b.data()),b.size());
}
static void genuine_collector_group(const std::string& original_head_path) {
 reset();skip_head_pack=true;
 char name[]="/tmp/strata-spill-collector-XXXXXX";check(mkdtemp(name),"collector source directory");
 const std::filesystem::path directory(name);
 const std::string head_bytes=collector_read(original_head_path),projection=collector_projection();
 const auto head_digest=collector_digest(head_bytes),projection_digest=collector_digest(projection);
 const std::string index="# align 32 pool 96 tensors 1\nblk.0.attn_q.weight 0 0 0 68 0 68 256 1 2 0 256 0 0 64 4 0 0 0\n";
 std::string dense(68,'\0');const float scale=1;std::memcpy(dense.data()+64,&scale,sizeof scale);
 const std::vector<std::pair<std::string,std::string>> files{{"index.txt",index},{"dense.bin",dense},{"head.gguf",head_bytes},{"native.gguf",projection}};
 std::string text="STRATA_VERIFIED_INPUT_MANIFEST\t1\n";
 for(const auto& [key,value]:std::vector<std::pair<std::string,uint64_t>>{{"max_files",8},{"max_aliases",8},{"max_path_bytes",256},{"max_depth",8},{"max_metadata_bytes",4096},{"view_host_bytes",8192},{"view_disk_bytes",8192},{"artifact_max_file_bytes",1<<20},{"artifact_block_bytes",64},{"artifact_metadata_host_bytes",4096},{"artifact_disk_overhead_bytes",4096},{"allow_full_copy",1}})
  text+="LIMIT\t"+key+"\t"+std::to_string(value)+"\n";
 for(const auto& [file,bytes]:files) {
  collector_write(directory/file,bytes);const std::string alias=(file.ends_with(".gguf")?"models/":"pack/")+file;
  text+="FILE\t"+file+"\t"+std::to_string(bytes.size())+"\t"+collector_hex(collector_digest(bytes))+"\t"+(directory/file).string()+"\n";
  text+="ALIAS\t"+file+"\t"+alias+"\n";
 }
 text+="GGUF\tmain\t0\tmodels/head.gguf\nGGUF\tdense\t0\tmodels/native.gguf\nPACK\tpack\tmodels/head.gguf\n";
 for(const auto& [role,value]:std::vector<std::pair<std::string,std::string>>{{"pack","pack"},{"native-primary","models/head.gguf"},{"native-group","main"},{"dense-group","dense"},{"head-primary","models/head.gguf"},{"head-group","main"},{"embedding","-"},{"ple","-"}})
  text+="BIND\t"+role+"\t"+value+"\n";
 collector_write(directory/"manifest.tsv",text);
 const WindowResources zero{{0,0},0,0,0,0,0};
 WindowBudget inputs({{0,0},32ull<<20,0,32ull<<20,2,0},zero);
 const uint64_t fixed=NativeHead::spill_control_bytes();check(fixed>0,"collector requires genuine owned feature profile");
 WindowBudget ledger({{0,1ull<<20},fixed+8704,8704,0,0,0},zero);
 std::shared_ptr<const ParsedVerifiedInputs> plan;
 const VerifiedManifestParseLimits parser_limits{8192,64,1024,512};
 check(bool(ParsedVerifiedInputs::read((directory/"manifest.tsv").string(),collector_digest(text),parser_limits,inputs,plan)),"real manifest full SHA parse");
 VerifiedDriverOverrides overrides;overrides.paths={{"pack",{directory.string()}},{"native-primary",{(directory/"head.gguf").string()}},{"native-group",{(directory/"head.gguf").string()}},{"dense-group",{(directory/"native.gguf").string()}},{"head-primary",{(directory/"head.gguf").string()}},{"head-group",{(directory/"head.gguf").string()}}};
 check(bool(plan->validate_overrides(overrides)),"real explicit original-path validation");
 std::shared_ptr<const VerifiedInputBundle> bundle;
 check(bool(plan->prepare_bundle(directory.string(),inputs,overrides,bundle)),"real source-independent verified bundle");
 std::weak_ptr<const ParsedVerifiedInputs> weak_plan=plan;
 std::weak_ptr<const VerifiedDriverBindings> weak_bindings;
 std::weak_ptr<const OwnedModelStage> weak_stage;
 std::weak_ptr<const OwnedModelGang> weak_gang;
 std::shared_ptr<const void> retained_artifact;
 {
  // Natural destruction releases the actual collector reservation after its
  // output buffers. Do not replace this scope with memberwise reset/assignment.
  WindowLoaderProfile profile;
  VerifiedDriverBindings bound;
  check(bool(plan->bind(bundle,overrides,bound)),"real plan binds exact bundle membership");
  auto bindings=std::make_shared<const VerifiedDriverBindings>(std::move(bound));weak_bindings=bindings;
  check(bindings->manifest_digest==collector_digest(text)&&bindings->head.size()==1&&bindings->dense.size()==1&&
        bindings->head[0].size()==head_bytes.size()&&bindings->head[0].content_digest()==head_digest&&
        bindings->dense[0].content_digest()==projection_digest&&bindings->pack_payloads[0].content_digest()==collector_digest(dense),
        "genuine bound file extents and full-content identities");
  check(bindings->head_paths()[0]!=(directory/"head.gguf").string()&&bindings->pack_path()!=directory.string(),"loaders use only retained private aliases");
  // Change an actual tensor byte in the original, not merely its metadata.
  // The loader and eventual mapped head must continue using the frozen bytes.
  size_t source_weight_offset=0;
  {strata::GgufFile original((directory/"head.gguf").string());const auto* tensor=original.find("output.weight");
   check(tensor!=nullptr,"original head payload source span");source_weight_offset=size_t(original.data_start()+tensor->offset);}
  check(source_weight_offset<head_bytes.size(),"original payload mutation is in declared file");
  auto changed_head=head_bytes;changed_head[source_weight_offset]=char(static_cast<unsigned char>(changed_head[source_weight_offset])^0x5a);
  collector_write(directory/"head.gguf",changed_head);
  check(collector_digest(collector_read(directory/"head.gguf"))!=head_digest&&bindings->head[0].content_digest()==head_digest,
        "verified snapshot identity survives original-file replacement");
  ModelStageResources input;input.input_owner=bindings;input.device=1;
  input.weights=std::make_unique<WeightTable>();input.native=std::make_unique<NativeDense>();input.head=std::make_unique<NativeHead>();
  const CanonicalInventoryLimits canonical_limits{1,512};const NativeInventoryLimits native_limits{1,1,2,512};
  uint64_t canonical_host=0,native_host=0,head_host=0;std::string error;
  check(canonical_inventory_host_bound(canonical_limits,canonical_host,error)&&native_inventory_host_bound(native_limits,native_host,error)&&
        head_embed_inventory_host_bound(native_limits,head_host,error),"actual three inventory quotes");
  auto metadata=[&](uint64_t bytes){return inputs.reserve({{0,0},bytes,0,0,0,0},ReservationPurpose::foreground).lease;};
  check(input.weights->enable_inventory_capture(canonical_limits,metadata(canonical_host),bindings,bindings->pack_path(),error),"genuine canonical same-root inventory issuer");
  check(input.native->enable_inventory_capture(native_limits,metadata(native_host),bindings,error)&&
        input.head->enable_inventory_capture(native_limits,metadata(head_host),bindings,error),"genuine native and head same-root inventory issuers");
  check(cudaMalloc(&input.arena,96)==cudaSuccess,"collector actual canonical arena");input.arena_bytes=96;
  check(input.weights->load(bindings->pack_path(),input.arena,input.arena_bytes,error),"real canonical captured loader");
  check(input.native->load(bindings->dense_paths(),*input.weights,error),"real nonempty native captured loader");
  check(input.native->loaded_inventory()&&input.native->loaded_inventory()->records.size()==1&&
        input.native->loaded_inventory()->records[0].source_type==14&&input.native->loaded_inventory()->records[0].source_bytes==210,
        "native receipt is completed by actual supported Q6 projection load");
  check(input.head->load(bindings->head_paths(),256,32,error),"real bound captured head loader");
  const auto* final_head=input.head.get();const auto* head_receipt=final_head->loaded_inventory();
  check(head_receipt&&head_receipt->generation==1&&input.weights->loaded_inventory()&&
        input.weights->loaded_inventory()->retained_input_bindings.get()==bindings.get()&&
        input.native->loaded_inventory()->retained_input_bindings.get()==bindings.get()&&
        head_receipt->retained_input_bindings.get()==bindings.get(),"all completed inventories retain the one actual bindings root");
  input.reservation=ledger.reserve({{0,input.arena_bytes+input.native->allocated_bytes()+input.head->allocated_bytes()},fixed,0,0,0,0},ReservationPurpose::foreground).lease;
  check(bool(input.reservation),"genuine existing model-stage reservation");
  std::shared_ptr<const OwnedModelStage> stage;check(adopt_model_stage(input,stage,error),"actual captured stage adoption");weak_stage=stage;
  ModelGangResources group;group.input_owner=bindings;group.stages.push_back(stage);
  std::shared_ptr<const OwnedModelGang> gang;check(adopt_model_gang(group,gang,error),"actual bound model gang adoption");weak_gang=gang;
  check(stage->spill_head_startup(ledger,error)&&head_receipt->generation==2&&head_receipt->placement==EmbedPlacement::mapped_portable,
        "genuine gang head spill finishes before actual profile collection");
  strata::GgufFile frozen_head(bindings->head_paths()[0]);const auto* frozen_weight=frozen_head.find("output.weight");
  check(frozen_weight&&std::memcmp(final_head->weights(),frozen_head.tensor_data(*frozen_weight),8704)==0&&
        static_cast<const unsigned char*>(final_head->weights())[0]!=static_cast<unsigned char>(changed_head[source_weight_offset]),
        "mapped runtime head consumes genuine frozen payload rather than changed original");
  const LoaderProfileLimits limits{1,8,8,2,512,1ull<<20};
  check(collect_window_loader_profile(gang,bindings,limits,inputs,profile,error)&&bool(profile)&&profile.partial(),"actual postspill collector invoked with genuine authority");
  check(profile.tensors().size()==3&&profile.artifacts().size()==4&&profile.metadata_only_tensor_indices().empty(),"actual canonical native and head profile rows");
  const auto head=std::find_if(profile.tensors().begin(),profile.tensors().end(),[](const auto& t){return t.name=="stage/0/head/output.weight";});
  check(head!=profile.tensors().end()&&head->role==ProfileRole::head&&head->native_type==8&&head->shape==std::vector<uint64_t>({256,32})&&
        head->bytes==8704&&head->offset==head_receipt->tensor.source_offset,"actual head tensor type shape and source extent");
  check(head->runtime_layout.starts_with("head-embed-loader-v1;")&&
        head->runtime_layout.find(";placement="+std::to_string(unsigned(EmbedPlacement::mapped_portable))+";")!=std::string::npos&&
        head->runtime_layout.find(";main_bytes=8704;")!=std::string::npos&&
        head->runtime_layout.find(";scratch_bytes=288;")!=std::string::npos&&head->runtime_layout.find(";helper_bytes=0;")!=std::string::npos,
        "actual returned runtime layout records mapped placement and real byte fields");
  const auto artifact=std::find_if(profile.artifacts().begin(),profile.artifacts().end(),[&](const auto& a){return a.name==head->artifact;});
  check(artifact!=profile.artifacts().end()&&artifact->role==ProfileRole::head&&artifact->verified_full_content&&artifact->immutable_owner&&
        artifact->verification_evidence!=StateDigest{}&&artifact->content==head_digest&&artifact->bytes==head_bytes.size()&&
        head->offset<=artifact->bytes&&head->bytes<=artifact->bytes-head->offset,
        "returned head artifact has genuine full-content verification evidence and retained extent");
  for(const auto& a:profile.artifacts())check(a.verified_full_content&&a.immutable_owner&&a.verification_evidence!=StateDigest{},"every collected source is genuinely retained and verified");
  const auto* prior_tensors=profile.tensors().data();const auto* prior_artifacts=profile.artifacts().data();const auto before=inputs.snapshot().reserved;
  const auto prior_layout=head->runtime_layout;const auto prior_evidence=artifact->verification_evidence;
  retained_artifact=artifact->immutable_owner;
  VerifiedDriverBindings other_bound;check(bool(plan->bind(bundle,overrides,other_bound)),"second genuine bind for wrong-root refusal");
  auto other=std::make_shared<const VerifiedDriverBindings>(std::move(other_bound));
  check(!collect_window_loader_profile(gang,other,limits,inputs,profile,error),"different genuine bindings control root cannot replace actual loader authority");
  auto tight=limits;tight.max_tensors=1;
  check(!collect_window_loader_profile(gang,bindings,tight,inputs,profile,error)&&
        !collect_window_loader_profile(gang,{},limits,inputs,profile,error),"bound and missing-authority collector refusals");
  check(profile.tensors().data()==prior_tensors&&profile.artifacts().data()==prior_artifacts&&profile.tensors().size()==3&&
        profile.artifacts().size()==4&&inputs.snapshot().reserved==before&&head->runtime_layout==prior_layout&&
        head->bytes==8704&&head->offset==head_receipt->tensor.source_offset&&artifact->verified_full_content&&
        artifact->content==head_digest&&artifact->verification_evidence==prior_evidence&&
        artifact->immutable_owner.get()==retained_artifact.get()&&!artifact->immutable_owner.owner_before(retained_artifact)&&
        !retained_artifact.owner_before(artifact->immutable_owner),"collector refusals preserve genuine prior output and reservation");
  other.reset();
  stage.reset();gang.reset();bindings.reset();plan.reset();
  check(!weak_stage.expired()&&!weak_gang.expired()&&!weak_bindings.expired()&&!weak_plan.expired(),"actual profile retains complete model and verified plan root");
  check(bundle->retire_checked()==VerifiedInputRetirement::busy_leases,"actual profile prevents premature private source retirement");
 }
 check(weak_stage.expired()&&weak_gang.expired()&&weak_bindings.expired()&&weak_plan.expired()&&ledger.snapshot().reserved==zero&&
       live.empty()&&host_live.empty()&&packed.empty(),"profile destruction checked-retires actual model and all native charges");
 check(bundle->retire_checked()==VerifiedInputRetirement::busy_leases,"returned immutable artifact independently retains real private bytes");
 retained_artifact.reset();
 check(bundle->retire_checked()==VerifiedInputRetirement::retired&&bundle->retired(),"actual bundle retirement after final source lease");
 bundle.reset();check(inputs.snapshot().reserved==zero,"all genuine parse inventory collector snapshot and view credits released");
 std::filesystem::remove_all(directory);reset();std::puts("genuine_head_spill_collector_test passed");
}

int main(){
 baseline_main();
 const auto path=fixture();std::string error;NativeInventoryLimits limits{2,1,2,4096};uint64_t bytes=0;
 check(head_embed_inventory_host_bound(limits,bytes,error),"inventory bound");
 // First device allocation failure is NOT complete allocation failure: this
 // independent positive executes the genuine mapped-host load fallback.
 reset();fail_alloc=1;skip_head_pack=true;
 {
  strata::GgufFile gguf(path);const auto* tensor=gguf.find("output.weight");
  NativeHead h;
  check(h.load({path},256,32,error)&&h.ready_for_adoption()&&h.loaded()&&h.on_host()&&h.device()==1,
        "ordinary first device failure loads ready mapped head");
  check(allocs==2&&live.size()==1&&host_live.size()==1&&packed.empty(),
        "fallback owns only device scratch and one plain mapped payload");
  check(h.weight_bytes()==8704&&h.allocated_bytes()==288&&h.allocated_host_bytes()==8704&&h.allocated_pinned_bytes()==8704,
        "ordinary load fallback accounts device scratch separately from host and pinned weight");
  check(tensor&&tensor->shape==std::vector<uint64_t>({256,32})&&
        std::memcmp(h.weights(),gguf.tensor_data(*tensor),8704)==0,
        "ordinary load fallback retains exact actual GGUF payload");
  device=0;h.unload();
  check(device==0&&!h.ready_for_adoption()&&!h.loaded()&&h.allocated_bytes()==0&&
        h.allocated_host_bytes()==0&&h.allocated_pinned_bytes()==0&&
        frees==1&&host_frees==1&&live.empty()&&host_live.empty()&&packed.empty(),
        "ordinary load fallback checked teardown uses owner device and restores caller");
 }
 reset();
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
 checked_spill_groups(path,limits,bytes);
 genuine_collector_group(path);
 std::remove(path.c_str());std::puts("head_embed_load_inventory_test passed");return 0;
}
