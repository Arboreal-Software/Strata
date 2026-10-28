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
static StateDigest digest(const std::string& text){WindowHasher h;h.update(text.data(),text.size());return h.digest();}
static std::string gguf(){
 std::string b;auto n=[&](uint64_t x,int bytes){for(int i=0;i<bytes;++i)b+=char(x>>(8*i));};
 auto text=[&](const std::string& t){n(t.size(),8);b+=t;};n(0x46554747,4);n(3,4);n(3,8);n(1,8);text("general.architecture");n(8,4);text("qwen4exp");
 for(int role=0;role<3;++role){text(std::string("blk.0.ffn_")+(role==0?"gate":role==1?"up":"down")+"_exps.weight");n(3,4);n(4,8);n(4,8);n(2,8);n(30,4);n(role*64,8);}
 b.resize((b.size()+31)/32*32,0);for(int r=0;r<3;++r)for(int e=0;e<2;++e)b.append(32,char(10*r+e+1));return b;
}
struct Fixture {
 std::string dir;WindowBudget input_budget{{{0},32<<20,0,32<<20,1,0},{{0}}};
 WindowBudget metadata{{{0},16<<20,0,0,0,0},{{0}}};
 std::shared_ptr<const VerifiedInputBundle> bundle;std::shared_ptr<VerifiedDriverBindings> bindings;
 std::shared_ptr<ExpertLayoutCapture> layout;std::vector<std::string> files;
 Fixture(bool packed){
  char d[]="/tmp/strata-expert-source-XXXXXX";require(mkdtemp(d),"private fixture dir");dir=d;
  VerifiedInputManifest manifest;auto add=[&](const std::string& name,const std::string& text,const std::string& alias){const auto path=dir+"/"+name;std::ofstream(path,std::ios::binary)<<text;manifest.files.push_back({path,text.size(),digest(text),{alias}});};
  const auto bytes=gguf();add("model.gguf",bytes,"model/model.gguf");strata::GgufFile g(dir+"/model.gguf");
  std::string sidecar="# strata native experts v4 (n_expert 2)\n0 30 30 0 96 ";
  for(int r=0;r<3;++r)sidecar+=std::to_string(g.data_start()+uint64_t(r)*64)+" ";
  sidecar+="model.gguf\n";
  add("native_experts.txt",sidecar,"pack/native_experts.txt");
  add("index.txt","# align 16 pool 16 tensors 1\nw 0 2 0 4 0 4 1 1 0 0 0 0 0 4 0 0 0 0\n","pack/index.txt");add("dense.bin","data","pack/dense.bin");
  if(packed){std::string expert;for(int e=0;e<2;++e)for(int r=0;r<3;++r)expert.append(32,char(r*10+e+1));add("experts.bin",expert,"pack/experts.bin");}
  manifest.gguf_groups={{"model/model.gguf"}};manifest.packs={{"pack","model/model.gguf"}};
  VerifiedInputLimits bounds{16,32,512,4,16384,8192,8192,{1<<20,64,4096,4096,true}};VerifiedArtifactOptions options;options.fault=[](const char* s){return std::string(s)=="clone"?EOPNOTSUPP:0;};
  const auto result=VerifiedInputBundle::prepare(manifest,dir,bounds,input_budget,bundle,options);require(bool(result),result.detail);
  bindings=std::make_shared<VerifiedDriverBindings>();require(bundle->resolve_directory("pack",bindings->pack),"pack lease");require(bundle->resolve("model/model.gguf",bindings->native_primary),"native lease");bindings->native.push_back(bindings->native_primary);
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
   const auto* previous=profile.spans();auto retained=f.metadata.snapshot().reserved.host_bytes;
   auto refused=limits;refused.max_host_bytes=1;
   require(!collect_expert_runtime_profile(runtime,refused,f.metadata,profile,error),"prepaid envelope refusal");
   require(profile.spans()==previous&&f.metadata.snapshot().reserved.host_bytes==retained,"failure preserves output/charge");
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
}
