#include "window_session_elastic_test_mocks.hpp"
// Prepared genuine source+selection success fixture. No execution authorized.
// Real immutable bundle/source/layout/allocator/initial/selection factories;
// CUDA allocation and session shape/reset are CPU seams, not GPU qualification.
#include "strata/core/expert_host_execution.hpp"
#include "strata/core/window_allocator.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "window_native_test_fixture.hpp"
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

extern "C" cudaError_t cudaMalloc(void** p,size_t n){*p=std::malloc(n);return *p?cudaSuccess:cudaErrorMemoryAllocation;}
extern "C" cudaError_t cudaFree(void* p){std::free(p);return cudaSuccess;}
namespace strata::kernels {std::size_t native_q8_1_bytes(int,int){return 4096;}}
namespace strata::core {
bool session_private_plan(const ModelGeometry&,int64_t,int64_t,int64_t,int64_t,SessionPrivatePlan& p){p={1024,4096,512};return true;}
uint64_t ple_run_scratch_bytes(){return 512;}
uint64_t session_init_private(const ModelGeometry& g,int64_t,int64_t,void* arena,SessionState& s,SessionInitUploads&,int64_t lo,int64_t hi){
 s.layer_lo=lo;s.layer_hi=hi;s.qsa_ord0=lo/4;s.qsa_alloc=1;s.qsa_states=new QsaState[size_t(g.n_qsa_layers())]();s.qsa_states[s.qsa_ord0].kv_int8=true;s.ple_hist=static_cast<float*>(arena);return 1024;
}
bool qsa_state_release_host(QsaState&,std::string&){return true;}
void session_release(SessionState&){}
void session_zero(SessionState& s,const ModelGeometry&,const float*,void*){s.ple_prev[0]=s.ple_prev[1]=-1;s.ple_token=-1;}
bool session_zero_checked(SessionState& s,const ModelGeometry& g,const float* f,void* stream,std::string&){session_zero(s,g,f,stream);return true;}
bool conversation_snapshot_validate(const SavedConversation&,const SessionState&,const ModelGeometry&,const QsaState*,std::string&){assert(false);return false;}
ConversationRestore conversation_snapshot_restore(const SavedConversation&,SessionState&,const ModelGeometry&,const QsaState*,std::string&){assert(false);return ConversationRestore::transfer_failed;}
}
struct RetainedRuntime final:NativeRuntimeAttachment,ExpertHostRuntimeBinding {
 std::shared_ptr<const VerifiedDriverBindings> inputs;
 std::shared_ptr<const ExpertLayoutCapture> layout;
 NativeSessionAllocation sessions;
 FileExpertSource source;
 ExpertPool pool{1,false,true};
 ExpertDispatch dispatch;
 RetainedRuntime(){dispatch.src=&source;dispatch.pool=&pool;dispatch.n_expert=2;}
 bool owns_expert_dispatch(const ExpertDispatch* candidate)const noexcept override {
  return candidate==&dispatch&&dispatch.src==&source&&dispatch.pool==&pool;
 }
};
int main(){
    strata::core::window_test::check_fixed_kv_elastic_mock_contract();
 Fixture f(false);std::string error;
 WindowBudget session_budget({{100000,100000},32<<20,10000,0,0,2},{{0,0},0,0,0,0,0});
 auto runtime=std::make_shared<RetainedRuntime>();runtime->inputs=f.bindings;runtime->layout=f.layout;
 auto capture=f.capture();require(runtime->source.set_strict_retirement(true,{0},error),error);
 require(runtime->source.enable_inventory_capture(capture,error),error);runtime->source.set_gguf(f.bindings->native_primary.loader_path());
 require(runtime->source.open(f.bindings->pack_path(),1,2,error),error);
 auto geometry=window_test::geometry();auto key=window_test::metadata().key;
 std::vector<NativeSessionRequest> requests{{0,0,4,true},{1,4,8,false}};
 require(bool(allocate_native_sessions(geometry,16,1,requests,session_budget,runtime->sessions)),"actual native session allocation");
 std::vector<NativeSessionBinder> binders(2,[](SessionState&,std::string&){return true;});
 PreparedNativeWindow ready;
 require(bool(prepare_owned_native_initial(key,geometry,runtime->sessions,binders,session_budget,1024,1<<20,true,ready,runtime)),"actual initial preparation");
 {
 WindowSelection selected(ready);auto token=selected.token();SelectedWindowWork access;require(selected.admit(token,access),"actual selected admission");
 ExpertHostExecutionContext ctx;ctx.runtime=runtime;ctx.source=capture;ctx.layout=f.layout;ctx.inputs=f.bindings;ctx.dispatch=&runtime->dispatch;ctx.incarnation=1;ctx.layer_end=1;ctx.experts=2;
 ExpertHostExecutionLimits limits{4,16,16,4,1024,2};uint64_t bytes=0;require(expert_host_execution_host_bound(limits,bytes,error),error);
 auto make=[&](const ExpertHostExecutionContext& context,std::unique_ptr<ExpertHostExecutionCapture>& sink){auto lease=f.metadata.reserve({{0},bytes,0,0,0,0},ReservationPurpose::foreground);require(bool(lease),"receipt quota");return ExpertHostExecutionCapture::create(limits,lease.lease,access,context,sink,error);};
 std::unique_ptr<ExpertHostExecutionCapture> sink;
 for(int bad=0;bad<3;++bad){auto wrong=ctx;if(bad==0)wrong.experts=3;if(bad==1)wrong.layer_end=2;if(bad==2)wrong.experts=1;require(!make(wrong,sink)&&!sink,"actual source geometry must match context");}
 // Same real source/pool pointers do not make a foreign receiver runtime-owned.
 {ExpertDispatch foreign;foreign.src=&runtime->source;foreign.pool=&runtime->pool;auto wrong=ctx;wrong.dispatch=&foreign;
  require(!make(wrong,sink)&&!sink,"foreign dispatch with same source/pool refused");}
 // Different real runtime owner cannot authenticate this selected work.
 {auto wrong=ctx;wrong.runtime=std::make_shared<RetainedRuntime>();require(!make(wrong,sink)&&!sink,"different selected runtime rejected");}
 require(make(ctx,sink),error);
 const uint8_t* blob=runtime->source.blob(0,0);require(blob!=nullptr,"actual mapped expert source");
 require(access.borrow([&](const auto&,const auto& owner){
   require(owner.get()==runtime.get(),"genuine selected borrow owner");uint64_t ticket=99;
   require(sink->begin_layer(&runtime->dispatch,&runtime->source,&runtime->pool,0,2,2,true,ticket),"actual CPU layer begin");
   ExpertHostCpuRow row;row.expert=0;row.blob=blob;row.entry=ExpertHostPoolEntry::multi_native_split;row.activation=ExpertHostActivation::native;
   require(sink->cpu_row(ticket,row),"first CPU row");row.row=2;row.token=1;row.job_row=1;require(sink->cpu_row(ticket,row),"same expert distinct token row");
   require(sink->cpu_batch_submitted(ticket,0,1,row.entry),"actual recorded CPU batch");sink->cpu_batch_completed(ticket,0);sink->end_host_dispatch(ticket);
 }),"live selected borrow");
 auto view=sink->view();require(!view.failed&&view.layer_count==1&&view.row_count==2&&view.batch_count==1&&view.batches[0].completed,"CPU-only completed batch receipt");
 require(*view.selection==access.token()&&view.operation==access.lease().operation(),"receipt derives actual selected identity");
 uint64_t untouched=123;require(!sink->begin_layer(&runtime->dispatch,&runtime->source,&runtime->pool,1,1,1,false,untouched)&&untouched==123,"actual geometry rejects layer outside source");
 require(sink->view().failed&&sink->view().batches[0].completed,"partial completed evidence preserved on later failure");
 sink->close_after_producers_joined();sink.reset();
 // Reuse the genuine retained source/selected work for bookkeeping regressions.
 for(int scenario=0;scenario<10;++scenario){
  require(make(ctx,sink),error);uint64_t ticket=0;
  require(sink->begin_layer(&runtime->dispatch,&runtime->source,&runtime->pool,0,2,2,true,ticket),"bookkeeping layer");
  ExpertHostCpuRow row;row.expert=0;row.blob=blob;row.entry=ExpertHostPoolEntry::multi_native_split;row.activation=ExpertHostActivation::native;
  require(sink->cpu_row(ticket,row),"bookkeeping initial row");
  if(scenario==0){
   require(sink->cpu_batch_submitted(ticket,0,1,row.entry),"first batch");sink->cpu_batch_completed(ticket,0);
   row.batch=1;require(!sink->cpu_row(ticket,row),"duplicate row from completed batch refused");
  }else if(scenario==1){row.rank=1;row.row=1;row.job=2;require(!sink->cpu_row(ticket,row),"job gap refused");}
  else if(scenario<9){
   row.rank=1;row.row=1;row.job_row=1;
   switch(scenario){case 2:row.expert=1;break;case 3:row.blob=blob+1;break;case 4:++row.up_offset;break;
    case 5:++row.down_offset;break;case 6:row.zero_copy_roles=!row.zero_copy_roles;break;
    case 7:row.activation=ExpertHostActivation::q8_1;break;case 8:row.entry=ExpertHostPoolEntry::multi_split;break;}
   require(!sink->cpu_row(ticket,row),"same job mismatched execution fields refused");
  }else{
   // Repeated expert in a distinct job is legal; arrival order is retained.
   row.rank=1;row.row=1;row.job=1;require(sink->cpu_row(ticket,row),"distinct job duplicate expert");
   require(sink->cpu_batch_submitted(ticket,0,2,row.entry),"two contiguous jobs");sink->cpu_batch_completed(ticket,0);
   row.batch=1;row.token=1;row.rank=0;row.row=2;row.job=0;row.job_row=0;
   require(sink->cpu_row(ticket,row),"new batch job count reset");
   row.rank=1;row.row=3;row.job_row=1;require(sink->cpu_row(ticket,row),"same job second row");
   require(sink->cpu_batch_submitted(ticket,1,1,row.entry),"second batch");sink->cpu_batch_completed(ticket,1);sink->end_host_dispatch(ticket);
   auto v=sink->view();require(!v.failed&&v.row_count==4&&v.batch_count==2,"multiple batch receipt");
   for(size_t i=0;i<4;++i)require(v.rows[i].row==i,"exact arrival order retained");
   uint64_t next=0;require(sink->begin_layer(&runtime->dispatch,&runtime->source,&runtime->pool,0,1,1,false,next),"next unique layer ticket");
   require(next!=ticket,"no row-seen ticket ABA");row={};row.expert=0;row.blob=blob;row.entry=ExpertHostPoolEntry::single;row.activation=ExpertHostActivation::native;
   require(sink->cpu_row(next,row),"same row index in new layer accepted");
   require(sink->cpu_batch_submitted(next,0,1,row.entry),"next layer batch");sink->cpu_batch_completed(next,0);sink->end_host_dispatch(next);
  }
  sink->close_after_producers_joined();sink.reset();
 }
 auto cut=token.cut;++cut.id;auto work=access.lease();require(selected.quiesce(token),"quiesce real work");require(work.finish(work.operation(),key,{cut,cut},true,true),"finish mock-device real selected gate");
 }
 ready={};capture.reset();runtime.reset();
 // Fixture destructor checks real bundle retirement and input charge release.
}
