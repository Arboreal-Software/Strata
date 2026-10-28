// Prepared source fixture, not compiled/run. Tiny generated inputs only.
// Actual immutable bundle, layout and FileExpertSource producers; CUDA operations
// and BF16 native format selection are explicit CPU seams, not device proof.
#include "strata/core/expert_runtime_profile.hpp"
#include "strata/core/window_restore.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_source_inventory.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <unistd.h>
using namespace strata::core;
using namespace strata::core;
using namespace strata::kernels::cpu;
// CPU runtime mocks only; actual source classes, parsers, immutable input groups,
// metadata admission and arena readers/scatter are linked from production.
extern "C" cudaError_t cudaGetDevice(int* d){*d=0;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int){return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){return cudaSuccess;}
extern "C" cudaError_t cudaGetLastError(){return cudaSuccess;}
extern "C" cudaError_t cudaHostRegister(void*,size_t,unsigned){return cudaSuccess;}
extern "C" cudaError_t cudaHostUnregister(void*){return cudaSuccess;}
extern "C" cudaError_t cudaHostGetDevicePointer(void** d,void* h,unsigned){*d=h;return cudaSuccess;}
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p,int){std::memset(p,0,sizeof(*p));return cudaSuccess;}
namespace strata::kernels::cpu {
// Link-only seams for retained router code. This fixture never performs router
// arithmetic; accidentally entering either path must fail without numeric output.
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) {
 std::fputs("UNEXPECTED: expert profile fixture entered AVX2 router arithmetic\n",stderr);
 std::_Exit(91);
}
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
 std::fputs("UNEXPECTED: expert profile fixture entered AVX1 router arithmetic\n",stderr);
 std::_Exit(92);
}
bool native_fmt(int gu,int down,int64_t,int64_t,NativeFmt& f,std::string& error){
 if(gu!=30||down!=30){error="fixture supports BF16 only";return false;}
 f.gu_type=f.d_type=30;f.n_embd=f.n_ff=4;f.gu_row=f.d_row=8;f.up_off=32;f.down_off=64;f.bytes=96;return true;
}
}
static void require(bool ok,const std::string& message){if(!ok){std::fprintf(stderr,"FAIL: %s\n",message.c_str());std::exit(1);}}
static bool same_budget(const WindowBudgetSnapshot& a,const WindowBudgetSnapshot& b){
 return a.capacity==b.capacity&&a.baseline==b.baseline&&a.reserved==b.reserved&&a.overcommitted==b.overcommitted;
}
static StateDigest digest(const std::string& text){WindowHasher h;h.update(text.data(),text.size());return h.digest();}
static std::string gguf(unsigned mask=7,int layer=0,int salt=0,int shard=-1){
 std::string b;auto n=[&](uint64_t x,int bytes){for(int i=0;i<bytes;++i)b+=char(x>>(8*i));};
 auto text=[&](const std::string& t){n(t.size(),8);b+=t;};n(0x46554747,4);n(3,4);n(unsigned(bool(mask&1))+unsigned(bool(mask&2))+unsigned(bool(mask&4)),8);n(shard<0?1:4,8);text("general.architecture");n(8,4);text("qwen4exp");
 if(shard>=0){text("split.count");n(4,4);n(3,4);text("split.no");n(4,4);n(shard,4);text("split.tensors.count");n(10,4);n(6,8);}
 int slot=0;for(int role=0;role<3;++role)if(mask&(1u<<role)){text(std::string("blk.")+std::to_string(layer)+".ffn_"+(role==0?"gate":role==1?"up":"down")+"_exps.weight");n(3,4);n(4,8);n(4,8);n(2,8);n(30,4);n(slot++*64,8);}
 b.resize((b.size()+31)/32*32,0);for(int r=0;r<3;++r)if(mask&(1u<<r))for(int e=0;e<2;++e)b.append(32,char(salt+10*r+e+1));return b;
}
struct Fixture {
 std::string dir;WindowBudget input_budget{{{0},32<<20,0,32<<20,1,0},{{0}}};
 WindowBudget metadata{{{0},16<<20,0,0,0,0},{{0}}};
 std::shared_ptr<const VerifiedInputBundle> bundle;std::shared_ptr<VerifiedDriverBindings> bindings;
 std::shared_ptr<ExpertLayoutCapture> layout;std::vector<std::string> files;
 std::string native_bytes,second_bytes,packed_bytes;uint64_t role_offsets[3]{};bool split=false;
 Fixture(bool packed,bool two_sources=false):split(two_sources){
  require(!packed||!split,"two-source fixture uses native role shards");
  char d[]="/tmp/strata-expert-source-XXXXXX";require(mkdtemp(d),"private fixture dir");dir=d;
  VerifiedInputManifest manifest;auto add=[&](const std::string& name,const std::string& text,const std::string& alias){const auto path=dir+"/"+name;std::ofstream(path,std::ios::binary)<<text;manifest.files.push_back({path,text.size(),digest(text),{alias}});};
  native_bytes=gguf(split?5:7,0,0,split?0:-1);add("model.gguf",native_bytes,"model/model.gguf");strata::GgufFile g(dir+"/model.gguf");
  if(split){second_bytes=gguf(2,0,40,1);add("second.gguf",second_bytes,"model/second.gguf");add("unused.gguf",gguf(7,1,70,2),"model/unused.gguf");}
  for(int r=0;r<3;++r)role_offsets[r]=g.data_start()+uint64_t(r)*64;
  if(split){strata::GgufFile second(dir+"/second.gguf");role_offsets[1]=second.data_start();role_offsets[2]=g.data_start()+64;}
  std::string sidecar="# strata native experts v4 (n_expert 2)\n0 30 30 0 96 ";
  for(int r=0;r<3;++r)sidecar+=std::to_string(role_offsets[r])+" ";
  sidecar+=split?"model.gguf,second.gguf,model.gguf\n":"model.gguf\n";
  add("native_experts.txt",sidecar,"pack/native_experts.txt");
  add("index.txt","# align 16 pool 16 tensors 1\nw 0 2 0 4 0 4 1 1 0 0 0 0 0 4 0 0 0 0\n","pack/index.txt");add("dense.bin","data","pack/dense.bin");
  if(packed){for(int e=0;e<2;++e)for(int r=0;r<3;++r)packed_bytes.append(32,char(r*10+e+1));add("experts.bin",packed_bytes,"pack/experts.bin");}
  manifest.gguf_groups={{"model/model.gguf"}};if(split){manifest.gguf_groups[0].push_back("model/second.gguf");manifest.gguf_groups[0].push_back("model/unused.gguf");}manifest.packs={{"pack","model/model.gguf"}};
  VerifiedInputLimits bounds{16,32,512,4,16384,8192,8192,{1<<20,64,4096,4096,true}};VerifiedArtifactOptions options;options.fault=[](const char* s){return std::string(s)=="clone"?EOPNOTSUPP:0;};
  const auto result=VerifiedInputBundle::prepare(manifest,dir,bounds,input_budget,bundle,options);require(bool(result),result.detail);
  bindings=std::make_shared<VerifiedDriverBindings>();require(bundle->resolve_directory("pack",bindings->pack),"pack lease");require(bundle->resolve("model/model.gguf",bindings->native_primary),"native lease");if(split){VerifiedInputLease unused,second;require(bundle->resolve("model/unused.gguf",unused),"unused native lease");require(bundle->resolve("model/second.gguf",second),"second native lease");bindings->native={unused,second,bindings->native_primary};}
  else bindings->native.push_back(bindings->native_primary);
  require(bundle->resolve("pack/native_experts.txt",bindings->native_expert_layout),"layout lease");bindings->native_expert_layout_absent=false;
  if(packed)require(bundle->resolve("pack/experts.bin",bindings->pack_payloads[2]),"packed lease");
  std::string error;ExpertInventoryLimits l{2,512,1024,4096,2,32};uint64_t charge=0;require(expert_inventory_host_bound(l,charge,error),error);
  auto reservation=metadata.reserve({{0},charge,0,0,0,0},ReservationPurpose::foreground);std::unique_ptr<ExpertLayoutCapture> created;
  require(ExpertLayoutCapture::create(l,reservation.lease,bindings,{bindings->pack_path(),true,false},created,error),error);
  layout=std::move(created);require(expert_layout_load(bindings->pack_path(),1,2,error,layout.get()),error);
 }
 std::shared_ptr<ExpertSourceCapture> capture(){std::string error;ExpertSourceInventoryLimits l{4,2,512,2};uint64_t charge=0;require(expert_source_inventory_host_bound(l,charge,error),error);auto reservation=metadata.reserve({{0},charge,0,0,0,0},ReservationPurpose::foreground);std::shared_ptr<ExpertSourceCapture> p;require(ExpertSourceCapture::create(l,reservation.lease,bindings,layout,p,error),error);return p;}
 ~Fixture(){layout.reset();bindings.reset();require(bundle->retire_checked()==VerifiedInputRetirement::retired,"actual input group retire after source");bundle.reset();require(input_budget.snapshot().reserved.host_bytes==0,"input host release");std::filesystem::remove_all(dir);}
};

struct RetainedRuntime final:NativeRuntimeAttachment,ExpertRuntimeProfileBinding {
 std::shared_ptr<const VerifiedDriverBindings> inputs;
 FileExpertSource source;
 bool expose=true;
 ExpertProfileSourceView expert_profile_source()const noexcept override {
  return {expose?&source:nullptr,nullptr,inputs};
 }
};
static void exact_artifact(const ProfileArtifact& a,const VerifiedInputLease& file,const std::string& bytes,const std::string& name){
 const auto verified=file.profile_artifact(ProfileRole::experts,"expected-verification-only");
 require(a.role==ProfileRole::experts&&a.name==name&&a.bytes==bytes.size(),"exact artifact role/name/extent");
 require(a.content==digest(bytes)&&a.content==file.content_digest(),"exact generated-byte artifact digest");
 require(a.verified_full_content&&a.verification_evidence==verified.verification_evidence&&bool(a.immutable_owner),"verified artifact evidence/retained authority");
}
static void exact_span(const ExpertRuntimeProfileSpan& s,uint64_t role,uint64_t artifact,uint64_t offset,
                       uint64_t stride,uint64_t per,uint64_t dst,uint64_t dst_stride,uint64_t mapped,ExpertSourceTransfer transfer){
 require(s.layer==0&&s.role==role&&s.artifact==artifact&&s.source_offset==offset&&s.source_stride==stride&&
         s.bytes_per_expert==per&&s.experts==2&&s.destination_offset==dst&&s.destination_stride==dst_stride&&
         s.mapped_file_ordinal==mapped&&s.transfer==transfer,"exact expert span fields");
}
static void exact_single(const ExpertRuntimeProfile& profile,const Fixture& f,bool packed){
 require(profile.spans()&&profile.artifacts(),"exact single profile available");
 if(packed){exact_span(profile.spans()->at(0),0,0,0,96,96,0,96,0,ExpertSourceTransfer::mapped_packed);
  exact_artifact(profile.artifacts()->at(0),f.bindings->pack_payloads[2],f.packed_bytes,"expert-source/0");}
 else{for(uint64_t r=0;r<3;++r)exact_span(profile.spans()->at(r),r,0,f.role_offsets[r],32,32,r*32,96,0,ExpertSourceTransfer::mapped_roles);
  exact_artifact(profile.artifacts()->at(0),f.bindings->native_primary,f.native_bytes,"expert-source/0");}
}
static void two_source_remapping(){
 Fixture f(false,true);std::string error;auto runtime=std::make_shared<RetainedRuntime>();runtime->inputs=f.bindings;
 auto capture=f.capture();require(runtime->source.set_strict_retirement(true,{0},error),error);
 require(runtime->source.enable_inventory_capture(capture,error),error);runtime->source.set_gguf(f.bindings->native_primary.loader_path());
 require(runtime->source.open(f.bindings->pack_path(),1,2,error),error);
 {
  const auto* receipt=runtime->source.completed_inventory();
  require(receipt&&receipt->sources.size()==3&&!receipt->sources[0].selected&&receipt->sources[1].selected&&receipt->sources[2].selected,
          "real producer declares unused prefix and two selected sources");
  require(receipt->spans[0].source==2&&receipt->spans[1].source==1&&receipt->spans[2].source==2,"actual source ordinals differ from compact artifact indices");
  ExpertRuntimeProfile profile;require(collect_expert_runtime_profile(runtime,{4,2,512,1<<20},f.metadata,profile,error),error);
  require(profile.current()&&profile.artifacts()->size()==2&&profile.spans()->size()==3,"two selected artifacts only");
  exact_artifact(profile.artifacts()->at(0),f.bindings->native[1],f.second_bytes,"expert-source/1");
  exact_artifact(profile.artifacts()->at(1),f.bindings->native_primary,f.native_bytes,"expert-source/2");
  require(profile.artifacts()->at(0).content!=profile.artifacts()->at(1).content,"distinct sources have discriminating digests");
  exact_span(profile.spans()->at(0),0,1,f.role_offsets[0],32,32,0,96,0,ExpertSourceTransfer::mapped_roles);
  exact_span(profile.spans()->at(1),1,0,f.role_offsets[1],32,32,32,96,1,ExpertSourceTransfer::mapped_roles);
  exact_span(profile.spans()->at(2),2,1,f.role_offsets[2],32,32,64,96,0,ExpertSourceTransfer::mapped_roles);
 }
 runtime.reset();capture.reset();
}
int main(){
 for(bool packed:{true,false}) {
  Fixture f(packed);std::string error;
  auto runtime=std::make_shared<RetainedRuntime>();runtime->inputs=f.bindings;
  auto capture=f.capture();
  require(runtime->source.set_strict_retirement(true,{0},error),error);
  require(runtime->source.enable_inventory_capture(capture,error),error);
  runtime->source.set_gguf(f.bindings->native_primary.loader_path());
  require(runtime->source.open(f.bindings->pack_path(),1,2,error),error);
  ExpertRuntimeProfileLimits limits{4,2,512,1<<20};
  {
   ExpertRuntimeProfile profile;
   require(collect_expert_runtime_profile(runtime,limits,f.metadata,profile,error),error);
   require(profile.partial()&&profile.current(),"partial current source join");
   require(profile.spans()->size()==(packed?1u:3u),"actual packed vs role spans");
   require(profile.artifacts()->size()==1&&profile.layout()==f.layout->completed_inventory(),"actual artifact/layout authority");
   exact_single(profile,f,packed);
   const auto* previous=profile.spans();auto retained=f.metadata.snapshot().reserved.host_bytes;
   auto refused=limits;refused.max_host_bytes=1;
   require(!collect_expert_runtime_profile(runtime,refused,f.metadata,profile,error),"prepaid envelope refusal");
   require(profile.spans()==previous&&f.metadata.snapshot().reserved.host_bytes==retained,"failure preserves output/charge");
   // A1: unchanged VALID administrative limits, but a genuinely exhausted
   // distinct ledger. The precise diagnostic proves reserve(), not envelope
   // validation, refused the attempted replacement of an existing profile.
   WindowBudget exhausted{{{0},1<<20,0,0,0,0},{{0}}};
   auto occupied=exhausted.reserve({{0},1<<20,0,0,0,0},ReservationPurpose::foreground);
   require(bool(occupied),"reserve entire independent metadata ledger");
   const auto original_before=f.metadata.snapshot(),refused_before=exhausted.snapshot();
   const auto inputs_before=f.input_budget.snapshot();
   const auto* previous_artifacts=profile.artifacts();const auto* previous_layout=profile.layout();
   require(!collect_expert_runtime_profile(runtime,limits,exhausted,profile,error),"actual metadata ledger refusal");
   require(error=="expert profile metadata reservation refused","reservation branch, not administrative refusal");
   require(profile.current()&&profile.spans()==previous&&profile.artifacts()==previous_artifacts&&
           profile.layout()==previous_layout,"ledger refusal leaves existing profile identity unchanged");
   require(same_budget(f.metadata.snapshot(),original_before)&&same_budget(exhausted.snapshot(),refused_before),
           "ledger refusal preserves full original and attempted ledgers");
   require(same_budget(f.input_budget.snapshot(),inputs_before),"ledger refusal preserves retained input charges");
   exact_single(profile,f,packed);
   runtime->expose=false;require(!profile.current(),"missing actual source refuses");runtime->expose=true;
   auto genuine=runtime->inputs;
   runtime->inputs=std::shared_ptr<const VerifiedDriverBindings>(genuine.get(),[](const auto*){});
   require(!collect_expert_runtime_profile(runtime,limits,f.metadata,profile,error),"unrelated control block refuses");
   runtime->inputs=genuine;require(profile.current(),"original authority remains");
   std::weak_ptr<RetainedRuntime> weak=runtime;
   runtime.reset();require(!weak.expired()&&profile.current(),"external profile retains actual runtime");
   auto restored=weak.lock();
   std::weak_ptr<const ExpertSourceCapture> old_capture=capture;
   capture.reset();
   restored->source.close();require(!profile.current()&&!profile.spans(),"source retirement invalidates profile");
   auto replacement=f.capture();
   require(restored->source.enable_inventory_capture(replacement,error),error);
   require(restored->source.open(f.bindings->pack_path(),1,2,error),error);
   require(!old_capture.expired(),"profile retains old capture through source replacement");
   require(!profile.current(),"same source/layout generation cannot revive old capture");
   require(collect_expert_runtime_profile(restored,limits,f.metadata,profile,error),error);
   require(profile.current()&&old_capture.expired(),"new receipt replaces old authority without ABA");
  }
  capture.reset();
 }
 two_source_remapping();
 std::puts("PASS expert profile coverage: ledger refusal packed/native; exact packed/native spans; two-source remapping");
}
