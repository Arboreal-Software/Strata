#include "strata/core/window_model_owner.hpp"
#include "strata/core/window_consumer_guard.hpp"
#include "strata/core/window_graph_guard.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <chrono>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static int device=0,fail_device=-1;
static bool fail_embedding=false;
static bool probe_model_lock=false;
static std::vector<int> events;
extern "C" cudaError_t cudaGetDevice(int* p){*p=device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){device=d;return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){
 events.push_back(10+device);
 if(probe_model_lock){
  std::atomic<bool> observed=false;
  std::thread callback([&]{(void)model_stage_quarantine_count();observed.store(true);});
  for(int i=0;i<100&&!observed.load();++i)std::this_thread::sleep_for(std::chrono::milliseconds(1));
  check(observed.load(),"device drain must not hold model registry lock against host callbacks");callback.join();
 }
 return device==fail_device?cudaErrorUnknown:cudaSuccess;
}
extern "C" cudaError_t cudaHostAlloc(void** p,size_t n,unsigned int){*p=std::malloc(n);return *p?cudaSuccess:cudaErrorMemoryAllocation;}
extern "C" cudaError_t cudaFreeHost(void* p){std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaMemcpy(void* d,const void* s,size_t n,cudaMemcpyKind){std::memcpy(d,s,n);return cudaSuccess;}
extern "C" cudaError_t cudaFree(void* p){events.push_back(300+device);std::free(p);return cudaSuccess;}
namespace strata::platform {void advise_willneed(int,uint64_t,uint64_t){}}
namespace strata::core {
NativeDense::~NativeDense()=default;
bool NativeDense::owns_weight_span(const void*,uint64_t)const{return false;}
bool NativeDense::release_checked(std::string&){return true;}
NativeHead::~NativeHead()=default;
bool NativeHead::load(const std::vector<std::string>&,int64_t,int64_t,std::string&){weights_=std::malloc(16);bytes_=16;device_=::device;load_complete_=true;return true;}
uint64_t NativeHead::allocated_bytes()const{return bytes_;}
bool NativeHead::release_checked(std::string&){check(device_==::device,"head device");events.push_back(100+device_);std::free(weights_);weights_=nullptr;bytes_=0;return true;}
NativeEmbed::~NativeEmbed()=default;
bool NativeEmbed::load(const std::vector<std::string>&,int64_t,int64_t,std::string&){host_=std::malloc(16);bytes_=16;device_=::device;load_complete_=true;return true;}
bool NativeEmbed::release_checked(std::string&){check(device_==::device,"embedding allocation device");events.push_back(200+device_);if(fail_embedding)return false;std::free(host_);host_=nullptr;return true;}
}
static std::shared_ptr<const OwnedModelStage> stage(int dev,WindowBudget& budget){
    device=dev;ModelStageResources r;r.device=dev;r.arena=std::malloc(32);r.arena_bytes=32;
    r.weights=std::make_unique<WeightTable>();r.native=std::make_unique<NativeDense>();r.head=std::make_unique<NativeHead>();std::string error;
    check(!r.head->ready_for_adoption(),"empty head not ready");
    char path[]="/tmp/strata-gang-XXXXXX";check(mkdtemp(path),"pack directory");auto dir=std::filesystem::path(path);
    {std::ofstream f(dir/"index.txt");f<<"# align 32 pool 32 tensors 1\nfixture.weight 0 2 0 32 0 32 8 1 0 0 0 0 0 0 0 0 0 0\n";}
    {std::ofstream f(dir/"dense.bin",std::ios::binary);char bytes[32]={};f.write(bytes,32);}
    check(r.weights->load(path,r.arena,32,error),"canonical stage load");std::filesystem::remove_all(dir);
    WindowResources charge{{0,0},0,0,0,0,0};charge.device_bytes[dev]=32;
    r.reservation=budget.reserve(charge,ReservationPurpose::foreground).lease;std::shared_ptr<const OwnedModelStage> out;
    check(!adopt_model_stage(r,out,error),"incomplete head refused");
    check(r.head->load({},0,0,error),"head fixture");
    check(!adopt_model_stage(r,out,error),"head allocation undercharge refused");
    r.reservation.reset();charge.device_bytes[dev]=48;r.reservation=budget.reserve(charge,ReservationPurpose::foreground).lease;
    check(adopt_model_stage(r,out,error),"head stage adopted");return out;
}
static ModelGangResources gang(WindowBudget& budget){
    ModelGangResources r;r.stages.push_back(stage(0,budget));r.stages.push_back(stage(1,budget));
    device=0;r.embedding=std::make_unique<NativeEmbed>();std::string error;check(r.embedding->load({},0,0,error),"embedding fixture");
    r.embedding_reservation=budget.reserve({{0,0},16,16,0,0,0},ReservationPurpose::foreground).lease;return r;
}
int main(){
 const WindowResources cap{{1024,1024},1024,1024,0,0,0},zero{{0,0},0,0,0,0,0};
 {
  WindowBudget budget(cap,zero);auto r=gang(budget);std::shared_ptr<const OwnedModelGang> out;std::string error;
  r.stages.push_back(r.stages.front());check(!adopt_model_gang(r,out,error),"duplicate devices refused");r.stages.pop_back();
  auto reservation=r.embedding_reservation;r.embedding_reservation.reset();check(!adopt_model_gang(r,out,error),"unreserved embedding refused");r.embedding_reservation=reservation;reservation.reset();
  reservation=r.embedding_reservation;r.embedding_reservation=r.stages.front()->reservation();
  check(!adopt_model_gang(r,out,error),"embedding cannot double-spend a stage reservation");r.embedding_reservation=reservation;reservation.reset();
  r.embedding_reservation.reset();r.embedding_reservation=budget.reserve({{0,0},16,0,0,0,0},ReservationPurpose::foreground).lease;
  check(!adopt_model_gang(r,out,error),"pinned subset undercharge refused");r.embedding_reservation.reset();r.embedding_reservation=budget.reserve({{0,0},16,16,0,0,0},ReservationPurpose::foreground).lease;
  check(adopt_model_gang(r,out,error)&&r.stages.empty()&&!r.embedding&&out->healthy(),"gang adoption transfers embedding and retains stages");
  events.clear();probe_model_lock=true;
  check(drain_model_gang_users(*out,error)&&events==std::vector<int>({10,11}),"explicit consumer boundary drains both devices without freeing resources");
  events.clear();{WindowConsumerDrainGuard guard(out);}
  check(events==std::vector<int>({10,11}),"driver scope guard drains before returning to borrowed target destruction");
  events.clear();int graph_releases=0;
  {WindowGraphGuard graph_guard(out,graph_releases,0,+[](int& count,std::string&){check(device==0,"graph release device");events.push_back(400);++count;return true;});
   graph_guard.retire();graph_guard.retire();}
  check(graph_releases==1&&events==std::vector<int>({10,11,400}),"graph guard drains before checked release and retires once");
  events.clear();auto reader=out;out.reset();check(events.empty(),"reader retains every model resource");reader.reset();
  auto embed=std::find(events.begin(),events.end(),200);check(embed!=events.end(),"embedding released");
  check(std::find(events.begin(),embed,10)!=embed&&std::find(events.begin(),embed,11)!=embed,"all stage devices drained before embedding free");
  check(std::find(events.begin(),embed,300)==embed&&std::find(events.begin(),embed,301)==embed,"stage allocations retained until embedding drain");
  check(budget.snapshot().reserved==zero&&device==0,"gang returns all charges and caller device");
  probe_model_lock=false;
 }
 for(int failure=0;failure<2;++failure){auto pid=fork();check(pid>=0,"fork");if(pid==0){
  WindowBudget budget(cap,zero);auto r=gang(budget);std::shared_ptr<const OwnedModelGang> out;std::string error;
  check(adopt_model_gang(r,out,error),"failure gang adopts");events.clear();fail_device=failure==0?1:-1;fail_embedding=failure==1;out.reset();
  check(model_gang_quarantine_count()==1,"uncertain gang retained");
  check(budget.snapshot().reserved.device_bytes==std::vector<uint64_t>({48,48})&&budget.snapshot().reserved.pinned_bytes==16,"failure keeps all gang charges");
  check(std::find(events.begin(),events.end(),300)==events.end()&&std::find(events.begin(),events.end(),301)==events.end(),"failure never releases stage allocation");
  if(failure==0)check(std::find(events.begin(),events.end(),200)==events.end(),"one-device drain failure prevents mapped-host free");
  ModelGangResources empty;check(!adopt_model_gang(empty,out,error),"quarantine fences new gangs");std::exit(0);
 }int status;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"failure case");}
 for(int guard_kind=0;guard_kind<2;++guard_kind){
 int marker[2];check(pipe(marker)==0,"borrowed-target marker pipe");
 auto pid=fork();check(pid>=0,"guard failure isolation");if(pid==0){
  close(marker[0]);WindowBudget budget(cap,zero);auto r=gang(budget);std::shared_ptr<const OwnedModelGang> out;std::string error;
  check(adopt_model_gang(r,out,error),"guard failure gang");
  struct BorrowedTarget {int fd;~BorrowedTarget(){const char byte='D';const auto written=write(fd,&byte,1);(void)written;}};
  {BorrowedTarget target{marker[1]};
   if(guard_kind==0){WindowConsumerDrainGuard guard(out);fail_device=1;}
   else {int graph=0;WindowGraphGuard guard(out,graph,0,+[](int&,std::string&){return false;});}}
  std::_Exit(3);
 }
 close(marker[1]);int status;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==1,"uncertain callback drain stops worker before unwind");
 char byte;check(read(marker[0],&byte,1)==0,"borrowed target never destroyed after failed drain");close(marker[0]);
 }
 std::puts("window_model_gang_test passed");
}
