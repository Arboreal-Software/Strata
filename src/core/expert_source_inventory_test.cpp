#include "strata/core/expert_source.hpp"
#include "strata/core/expert_source_inventory.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <fcntl.h>
#include <cstdarg>
#include <sys/wait.h>
#include <sys/mman.h>
#include "strata/kernels/cpu/hadamard.hpp"
using namespace strata::core;
using namespace strata::kernels::cpu;
static bool fail_read=false,fail_close=false;
static std::string redirect_path,redirect_target;
static int* close_fail_count=nullptr;
static std::weak_ptr<const VerifiedDriverBindings> failure_authority;
static WindowBudget* failure_budget=nullptr;
extern "C" int __real_open(const char*,int,...);
extern "C" int __wrap_open(const char* path,int flags,...){
 mode_t mode=0;if(flags&O_CREAT){va_list ap;va_start(ap,flags);mode=mode_t(va_arg(ap,int));va_end(ap);}
 if(!redirect_path.empty()&&path==redirect_path)return __real_open(redirect_target.c_str(),flags&~O_DIRECT,mode);
 return __real_open(path,flags,mode);
}
extern "C" size_t __real_fread(void*,size_t,size_t,FILE*);
extern "C" size_t __wrap_fread(void* p,size_t a,size_t b,FILE* f){if(fail_read)return 0;return __real_fread(p,a,b,f);}
extern "C" int __real_close(int);
extern "C" int __wrap_close(int fd){
 if(fail_close){if(failure_authority.expired()||!failure_budget||failure_budget->snapshot().reserved.host_bytes==0)std::_Exit(81);
  ++*close_fail_count;const int result=__real_close(fd);(void)result;errno=EIO;return -1;}
 return __real_close(fd);
}
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
int main(){
 for(bool packed:{false,true}){
  Fixture f(packed);std::string error;
  {ExpertSourceInventoryLimits limits{4,2,512,2};uint64_t bytes=0;require(expert_source_inventory_host_bound(limits,bytes,error),error);
   auto small=f.metadata.reserve({{0},bytes-1,0,0,0,0},ReservationPurpose::foreground);std::shared_ptr<ExpertSourceCapture> refused;
   require(!ExpertSourceCapture::create(limits,small.lease,f.bindings,f.layout,refused,error)&&!refused,"underadmission creates no source capture");
   auto prior=f.capture();auto saved=prior;auto admitted=f.metadata.reserve({{0},bytes,0,0,0,0},ReservationPurpose::foreground);
   require(!ExpertSourceCapture::create(limits,admitted.lease,f.bindings,f.layout,prior,error)&&prior==saved,"nonempty factory output unchanged");}
  {FileExpertSource source;require(source.set_strict_retirement(true,{0},error),error);auto capture=f.capture();require(source.enable_inventory_capture(capture,error),error);source.set_gguf(f.bindings->native_primary.loader_path());
   require(source.open(f.bindings->pack_path(),1,2,error),error);auto* receipt=source.completed_inventory();require(receipt&&receipt->from_gguf==!packed&&!receipt->arena,"actual File receipt");
   for(int e=0;e<2;++e){std::vector<uint8_t> out(96);require(source.copy_blob(0,e,out.data()),"actual mapped source copy");for(int r=0;r<3;++r)for(int b=0;b<32;++b)require(out[size_t(r*32+b)]==r*10+e+1,"actual mapped role bytes");}
   require(receipt->spans[0].bytes_per_expert==(packed?96:32),"actual per-expert span");
   FileExpertSource duplicate;require(duplicate.set_strict_retirement(true,{0},error),error);require(!duplicate.enable_inventory_capture(capture,error),"capture cannot be charged to two sources");
   redirect_path=packed?f.bindings->pack_payloads[2].loader_path():f.bindings->native_primary.loader_path();
   redirect_target=f.dir+(packed?"/experts.bin":"/model.gguf");
   require(!source.open_prompt_direct(error)&&error.find("differs from immutable")!=std::string::npos,"actual late direct reopen rejects different fd with identical bytes");redirect_path.clear();redirect_target.clear();
   source.close();require(!source.completed_inventory()&&!capture->completed_inventory(),"close invalidates receipt");
  }
  {FileExpertSource source;require(source.set_strict_retirement(true,{0},error),error);auto capture=f.capture();require(source.enable_inventory_capture(capture,error),error);source.set_gguf(f.bindings->native_primary.loader_path());
   redirect_path=packed?f.bindings->pack_payloads[2].loader_path():f.bindings->native_primary.loader_path();redirect_target=f.dir+(packed?"/experts.bin":"/model.gguf");
   require(!source.open(f.bindings->pack_path(),1,2,error)&&!source.completed_inventory(),"actual map rejects redirected descriptor");redirect_path.clear();redirect_target.clear();source.close();}
  {ArenaExpertSource source;require(source.set_strict_retirement(true,{0},error),error);require(source.enable_inventory_capture(f.capture(),error),error);source.set_gguf(f.bindings->native_primary.loader_path());fail_read=true;
   require(!source.open(f.bindings->pack_path(),1,2,2,error)&&!source.completed_inventory(),"short actual worker read publishes no receipt");fail_read=false;source.close();}
  for(bool padded:{false,true}){ArenaExpertSource source;require(source.set_strict_retirement(true,{0},error),error);auto capture=f.capture();require(source.enable_inventory_capture(capture,error),error);source.set_gguf(f.bindings->native_primary.loader_path());if(padded)source.set_pin_priority({{0,0},{0,1}});
   require(source.open(f.bindings->pack_path(),1,2,2,error),error);auto* receipt=source.completed_inventory();require(receipt&&receipt->arena&&receipt->padded==padded,"actual Arena receipt");
   for(int e=0;e<2;++e){const auto* blob=source.blob_at(0,e);require(blob,"actual arena blob");for(int r=0;r<3;++r)for(int b=0;b<32;++b)require(blob[r*32+b]==r*10+e+1,"actual arena scatter bytes");}
   require(receipt->spans[0].destination_stride==(padded?4096:96),"actual padded destination stride");source.close();require(!source.completed_inventory(),"arena close invalidates receipt");
  }
  close_fail_count=static_cast<int*>(mmap(nullptr,sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));require(close_fail_count!=MAP_FAILED,"shared fault counter");*close_fail_count=0;
  const pid_t child=fork();require(child>=0,"fork strict close");if(child==0){FileExpertSource source;require(source.set_strict_retirement(true,{0},error),error);require(source.enable_inventory_capture(f.capture(),error),error);source.set_gguf(f.bindings->native_primary.loader_path());require(source.open(f.bindings->pack_path(),1,2,error),error);
   failure_authority=f.bindings;failure_budget=&f.metadata;f.bindings.reset();f.layout.reset();fail_close=true;source.close();std::_Exit(82);}
  int status=0;require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==1&&*close_fail_count==1,"uncertain actual close fail-stops once with retained typed authority/charge");munmap(close_fail_count,sizeof(int));close_fail_count=nullptr;
  {FileExpertSource source;require(source.set_strict_retirement(true,{0},error),error);require(source.enable_inventory_capture(f.capture(),error),error);source.set_gguf(f.bindings->native_primary.loader_path());require(source.open(f.bindings->pack_path(),1,2,error),error);hadamard_set({});require(!source.completed_inventory(),"global generation replacement invalidates success receipt");source.close();}
 }
 std::puts("expert_source_inventory_test passed");
}
