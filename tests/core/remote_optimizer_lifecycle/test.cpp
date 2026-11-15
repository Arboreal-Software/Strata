// Prepared only. Actual lifecycle extraction; CUDA allocation/drain/free are mocks.
#include "remote_expert_opt.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fcntl.h>
#include "cuda_runtime.h"
enum Op { Get, Count, Select, Drain, Stream, HostAlloc, Map, Alloc, Free, HostFree, Nops };
static int device=0, seen[Nops]{}, fail_op=-1, fail_at=0;
static bool forbid_free=false;
static std::vector<int> events;
static bool fault(Op op){
 const int nth=++seen[op];
 if(nth!=fail_at||fail_op!=op)return false;
 std::fprintf(stderr,"FAULT %d %d\n",int(op),nth);
 return true;
}
int cudaGetDevice(int* p){if(fault(Get))return 7;*p=device;return 0;}
int cudaGetDeviceCount(int* p){if(fault(Count))return 7;*p=4;return 0;}
int cudaSetDevice(int d){if(fault(Select))return 7;device=d;events.push_back(10+d);return 0;}
int cudaDeviceSynchronize(){events.push_back(20+device);return fault(Drain)?7:0;}
int cudaStreamSynchronize(void*){events.push_back(30+device);return fault(Stream)?7:0;}
int cudaHostAlloc(void** p,size_t,unsigned){if(fault(HostAlloc))return 7;*p=std::malloc(64);return 0;}
int cudaHostGetDevicePointer(void** p,void* q,int){if(fault(Map))return 7;*p=q;return 0;}
int cudaMalloc(void** p,size_t){if(fault(Alloc))return 7;*p=std::malloc(64);return 0;}
int cudaFree(void* p){if(forbid_free)std::_Exit(73);events.push_back(40+device);if(fault(Free))return 7;std::free(p);return 0;}
int cudaFreeHost(void* p){if(forbid_free)std::_Exit(73);events.push_back(50+device);if(fault(HostFree))return 7;std::free(p);return 0;}
#include "strict_retirement.hpp"
namespace strata::core {
constexpr int MAXT=1,H=1,K=1;
struct RemoteExperts { int device_=1; void* stream_=reinterpret_cast<void*>(1); RemoteExpertOpt* remote_opt_=nullptr; };
struct OnDevice { int old; explicit OnDevice(int d):old(device){cudaSetDevice(d);} ~OnDevice(){cudaSetDevice(old);} };
bool check(int s,std::string& e){if(s)e="allocation failed";return s==0;}
#include "lifecycle.inc"
}
using namespace strata::core;
static void arm(int op=-1,int nth=0){std::fill(std::begin(seen),std::end(seen),0);fail_op=op;fail_at=nth;}
static void reset(){events.clear();device=0;arm();forbid_free=false;}
static void scenario(int op=-1,int nth=0){
 reset(); RemoteExperts a,b;b.device_=2;RemoteExpertOpt opt;std::string e;
 assert(opt.set_strict_retirement(true,{0,1,2},e));
 assert(opt.attach_checked(a,e));assert(opt.attach_checked(b,e));assert(!opt.attach_checked(a,e));
 arm(op,nth);assert(opt.init(e)==(op<0));assert(device==0);
 arm();events.clear();assert(opt.close_after_callers_joined());assert(device==0);
 auto first=std::find_if(events.begin(),events.end(),[](int x){return x>=40;});
 if(op!=HostAlloc){
  assert(first!=events.end());
  for(int d=0;d<3;++d)assert(std::find(events.begin(),first,20+d)!=first);
  for(int d=1;d<3;++d)assert(std::find(events.begin(),first,30+d)!=first);
 }
 assert(!a.remote_opt_&&!b.remote_opt_);auto n=events.size();
 assert(opt.close_after_callers_joined());assert(n==events.size());
 assert(!opt.init(e));assert(!opt.attach_checked(a,e));
}
// Own exactly one unreaped child. Disarm after reap/ECHILD so no stale PID
// can ever be signalled. No process-wide alarm can abandon a live child.
struct Child {
 pid_t pid;
 bool poll(int& status) {
  if(pid<0)return false;
  const pid_t r=waitpid(pid,&status,WNOHANG);
  if(r==pid){pid=-1;return true;}
  if(r<0&&errno==ECHILD){pid=-1;return false;}
  if(r<0&&errno!=EINTR)throw std::runtime_error("waitpid failed");
  return false;
 }
 void stop() noexcept {
  if(pid<0)return;
  int status=0;
  // ECHILD must be handled before signalling an ownership-ambiguous PID.
  const pid_t r=waitpid(pid,&status,WNOHANG);
  if(r==pid||(r<0&&errno==ECHILD)){pid=-1;return;}
  if(kill(pid,SIGKILL)<0&&errno!=ESRCH)std::_Exit(81);
  const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::chrono::steady_clock::now()<end){
   const pid_t got=waitpid(pid,&status,WNOHANG);
   if(got==pid||(got<0&&errno==ECHILD)){pid=-1;return;}
   usleep(1000);
  }
  // A kernel-stalled SIGKILL/reap cannot be certified; outer container cleanup
  // remains required and this fixture must not report success.
  std::_Exit(82);
 }
 ~Child(){stop();}
 int wait(int milliseconds=5000){
  int status=0;
  const auto end=std::chrono::steady_clock::now()+std::chrono::milliseconds(milliseconds);
  while(std::chrono::steady_clock::now()<end){
   if(poll(status))return status;
   if(pid<0)throw std::runtime_error("unexpected ECHILD");
   usleep(1000);
  }
  stop();throw std::runtime_error("child deadline");
 }
};
static const char* expected_operation(bool closing,int op,int nth){
 if(!closing){
  if(op==Get)return "optimizer init get device";
  if(op==Count)return "optimizer init device count";
  if(op==Select)return nth==3?"optimizer init restore device":"optimizer init select helper";
 }else{
  if(op==Get)return "optimizer close get device";
  if(op==Drain)return "optimizer close drain consumer";
  if(op==Stream)return "optimizer close helper stream";
  if(op==Free)return "optimizer close free helper sum";
  if(op==HostFree)return "optimizer close free mapped sum";
  if(op==Select){
   if(nth<=3)return "optimizer close select consumer";
   if(nth<=5)return "optimizer close select stream device";
   if(nth<=7)return "optimizer close select allocation device";
   return nth==8?"optimizer close select mapped device":"optimizer close restore device";
  }
 }
 std::abort();
}
// Bounded stderr capture. The fatal child emits only one marker + production
// diagnostic. Unexpected excessive output may block until the exact-child
// deadline kills/reaps it; it can never satisfy the expected diagnostic oracle.
struct ReadPipe {int fd;~ReadPipe(){if(fd>=0)close(fd);}};
static bool fatal_matches(bool closing,int op,int nth,bool forbidden=false,bool early_unrelated=false){
 int fds[2];assert(pipe(fds)==0);ReadPipe read{fds[0]};
 assert(fcntl(read.fd,F_SETFL,O_NONBLOCK)==0);
 pid_t child=fork();assert(child>=0);
 if(!child){
  close(fds[0]);if(dup2(fds[1],STDERR_FILENO)<0)std::_Exit(74);close(fds[1]);
  reset();RemoteExperts a,b;b.device_=2;RemoteExpertOpt opt;std::string e;
  assert(opt.set_strict_retirement(true,{0,1,2},e));assert(opt.attach_checked(a,e));assert(opt.attach_checked(b,e));
  if(early_unrelated)retirement_failed("unrelated earlier failure");
  if(closing){assert(opt.init(e));arm(op,nth);forbid_free=forbidden;opt.close_after_callers_joined();}
  else{arm(op,nth);opt.init(e);}
  std::_Exit(72);
 }
 close(fds[1]);Child owned{child};const int status=owned.wait();
 char captured[1024];size_t used=0;bool eof=false;
 while(used<sizeof(captured)){
  const auto n=::read(read.fd,captured+used,sizeof(captured)-used);
  if(n>0){used+=size_t(n);continue;}
  if(n==0){eof=true;break;}
  if(errno==EINTR)continue;
  break; // EAGAIN/error is incomplete capture, never PASS.
 }
 char expected[512];const int n=std::snprintf(expected,sizeof(expected),
  "FAULT %d %d\nstrata: strict retirement failed at %s; worker restart required\n",
  op,nth,expected_operation(closing,op,nth));
 return eof&&n>0&&size_t(n)<sizeof(expected)&&WIFEXITED(status)&&WEXITSTATUS(status)==1&&
  used==size_t(n)&&std::memcmp(captured,expected,used)==0;
}
static void fatal(bool closing,int op,int nth,bool forbidden=false){
 assert(fatal_matches(closing,op,nth,forbidden));
}

int main(){
 // Negative oracle control: same exit1 from unrelated earlier fail-stop must fail.
 assert(!fatal_matches(false,Get,1,false,true));
 scenario();scenario(HostAlloc,1);scenario(Map,1);scenario(Alloc,1);scenario(Alloc,2);
 fatal(false,Get,1);fatal(false,Count,1);
 for(int i=1;i<=3;++i)fatal(false,Select,i); // helpers and restoration
 fatal(true,Get,1,true);
 for(int i=1;i<=3;++i)fatal(true,Drain,i,true);
 for(int i=1;i<=2;++i)fatal(true,Stream,i,true);
 // Three consumer selects, two stream selects, two allocation selects,
 // mapped host select, final restoration. All earlier failure paths forbid free.
 for(int i=1;i<=9;++i)fatal(true,Select,i,i<=5);
 fatal(true,Free,1);fatal(true,Free,2);fatal(true,HostFree,1);
 reset();{RemoteExpertOpt opt;assert(!opt.close_after_callers_joined());assert(events.empty());}
 reset();{RemoteExperts a;RemoteExpertOpt opt;std::string e;
 assert(!opt.set_strict_retirement(true,{},e));assert(!opt.set_strict_retirement(true,{0,0},e));
 assert(opt.set_strict_retirement(true,{1},e));assert(opt.attach_checked(a,e));assert(!opt.init(e));
 assert(seen[HostAlloc]==0);assert(opt.close_after_callers_joined());assert(events.empty());}
 // Supervisor hang control: timeout must kill/reap the owned child.
 pid_t hanging=fork();assert(hanging>=0);if(!hanging){for(;;)pause();}
 Child hang{hanging};bool timeout=false;try{hang.wait(30);}catch(const std::runtime_error&){timeout=true;}
 assert(timeout&&hang.pid==-1);int status=0;assert(waitpid(hanging,&status,WNOHANG)==-1&&errno==ECHILD);
 // Already-reaped ownership must disarm without signalling the old PID.
 pid_t done=fork();assert(done>=0);if(!done)std::_Exit(0);
 Child reap{done};assert(WIFEXITED(reap.wait()));Child stale{done};assert(!stale.poll(status));assert(stale.pid==-1);
}
