#pragma once
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

// Fixture-only namespace indirection: the contiguous production extracts are
// unchanged. Only std::thread construction is observed synchronously, while the
// real thread and production JoinedThreadScope still provide joining semantics.
namespace serial_controller_coverage {
struct Witness {int created=0,services=0;bool completed=false,active=true,entered=false;};
inline Witness* witness=nullptr;
namespace controller_std {
using namespace std;
class thread {
 std::thread actual;
public:
 thread()=default;
 template<class F> explicit thread(F&& f) {
  // This assertion occurs BEFORE scheduling the callback. A moved launch cannot
  // evade it merely because the OS delays execution until after ver.run.
  assert(witness->completed&&!witness->active);
  ++witness->created;
  actual=std::thread(std::forward<F>(f));
 }
 thread& operator=(thread&& other) noexcept {actual=std::move(other.actual);return *this;}
 operator std::thread&() noexcept {return actual;}
 bool joinable()const noexcept{return actual.joinable();}
 void join(){actual.join();}
};
}
struct MockVerifier {
 bool fail_run=false;
 template<class Pool,class User>
 bool run(int,int32_t*,int64_t,Pool pool,User user,int32_t* output,std::string& error){
  assert(!witness->completed&&witness->active&&witness->created==0);
  // Explicit caller-driven progress: no independent releaser drains this work.
  for(int i=0;i<3;++i)pool(user);
  witness->active=false;
  if(fail_run){error="injected verifier failure";return false;}
  output[0]=7;witness->completed=true;return true;
 }
 int vocab()const{return 1;}
 bool copy_logits(int,float*)const{return false;}
 const void* final_R_all()const{return nullptr;}
};
struct MockDrive {struct Data {std::vector<int> usage{1};bool failed=false;const char* fail=nullptr;int fail_layer=0,fail_expert=0;} d;};
inline void service(void*){assert(witness->active&&!witness->completed&&witness->created==0);++witness->services;}
inline void drive_pool_multi(MockDrive* p){service(p);}
inline constexpr int cudaMemcpyDeviceToHost=0;
inline int cudaMemcpy(void*,const void*,size_t,int){assert(false&&"disabled diagnostic copy reached");return 1;}
struct MockMtp {static bool top2_env(){return false;}int top2(int)const{return 0;}};
// Qualified static method in serving source is a mock, not the real drafter.
namespace strata {namespace core {using MtpDrafter=MockMtp;using JoinedThreadScope=::strata::core::JoinedThreadScope;}
 namespace platform {using ::strata::platform::name_this_thread;}
 namespace aux_cpus {using ::strata::aux_cpus::pin_current_thread;}}
inline int run(bool serving,int mode){
 MockDrive drive;MockVerifier ver;ver.fail_run=mode==1;drive.d.failed=mode==2;
 Options o;int rounds=0,T=1,T_mtp=1;int64_t p=0;void* ajob=nullptr;
 std::vector<int32_t> window{7},outv{0};std::string err;
 std::function<void()> adapt_fence;if(mode==3)adapt_fence=[]{};
 auto adapt=[] {assert(witness->completed&&!witness->active);witness->entered=true;return true;};
 auto win_pool_fn=&service;void* win_pool_user=&drive;
 bool aflipped=false,first_window=false,merged=false,from_sfx=false;
 auto apply_pending=[](bool){assert(false&&"disabled overlap reached");};
 int chain_n=0,sfx_windows=0,sfx_drafts=0,sfx_ok=0,chain_windows=0,chain_drafts=0,chain_ok=0;
 int dp_tested[8]{},dp_ok[8]{},t2_rej[8]{},t2_hit[8]{};MockMtp mtp;
 int win_cap=1;std::vector<int>window_hist(4);std::FILE* hashes=nullptr;
 struct {int hc=1,n_embd=1;}g;std::vector<float>hash_rows;
 std::vector<int32_t>follow,produced;std::vector<bool>fdiff(1);double ttft_ms=0;auto t_start=Clock::now();
 try {
  if(serving){
#define std controller_std
#include "serve-controller.inc"
#undef std
   if(mode==4)throw std::runtime_error("caller after entry");
  }else{
   // Actual source's OOM retry remains present; these bounded scenarios use T1
   // and a non-OOM failure, so this label must not be revisited.
   retry_round:
#define std controller_std
#include "nonserve-controller.inc"
#undef std
   if(mode==4)throw std::runtime_error("caller after entry");
  }
 }catch(const std::runtime_error&){assert(mode==4);}
 return 0;
}
}
inline void check_controller_entry(){
 using namespace serial_controller_coverage;
 void* shared=mmap(nullptr,sizeof(Witness),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
 assert(shared!=MAP_FAILED);witness=new(shared)Witness;
 for(bool serving:{false,true})for(int mode=0;mode<5;++mode){
  if(mode==3&&!serving)continue; // Non-serving has no adaptation fence.
  *witness=Witness{};
  const auto pid=fork();assert(pid>=0);
  if(!pid)std::_Exit(serial_controller_coverage::run(serving,mode));
  int status=0;assert(waitpid(pid,&status,0)==pid&&WIFEXITED(status));
  assert(WEXITSTATUS(status)==((mode==1||mode==2||mode==3)?1:0));
  assert(witness->services==3&&!witness->active);
  assert(witness->created==((mode==0||mode==4)?1:0));
  assert(witness->entered==(mode==0||mode==4));
  assert(witness->completed==(mode!=1));
 }
 witness->~Witness();assert(munmap(shared,sizeof(Witness))==0);witness=nullptr;
}
