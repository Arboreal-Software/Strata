#include "submitted_work_drain.hpp"
#include <atomic>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <string>
#include <csignal>
#include <array>
#include <chrono>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
#define STRATA_WINDOW_STATE_AVAILABLE 1
static constexpr int assertion_exit=86;
static void check(bool v,const char* why){if(!v){std::fprintf(stderr,"FAIL: %s\n",why);std::fflush(stderr);std::_Exit(assertion_exit);}}
struct Flag {
 std::atomic<bool> value{false};bool flip_after_next_load=false;
 bool load(){const bool prior=value.load();if(flip_after_next_load){flip_after_next_load=false;value=true;}return prior;}
};
static void run(int mode,bool inject_assertion=false){
 struct Options{bool owned_session_lifetime=true;}o;
 struct Input{Flag input_error;};auto in_shared=std::make_shared<Input>();
 int remaining=2,backend_calls=0,starts=0;
 auto pipe_inflight=[&]{return remaining>0;};
 auto pump_step=[&](bool may_start)->bool {
#ifdef OLD_ADMISSION
#include "old-prefix.inc"
#else
#include "prefix.inc"
#endif
  ++backend_calls;if(may_start){++starts;++remaining;}
  if(mode==2)return false;
  if(!may_start)--remaining;
  return true;
 };
#include "controller.inc"
 if(mode==0){
  // Return false to wrapper's load, then publish true before pump_step entry.
  in_shared->input_error.flip_after_next_load=true;
  check(pump(true),"admitted racing input must not create uncertain failure");
  check(backend_calls==1&&starts==1&&!pipe_completion_uncertain,"one admitted backend step");
  check(!pump(true)&&backend_calls==1&&!pipe_completion_uncertain,"later admission cleanly rejected");
 }else if(mode==1){
  in_shared->input_error.value=true;
  check(!pump(true)&&!pipe_completion_uncertain&&backend_calls==0,"pre-admission rejection has no effects");
 }else if(mode==2){
  const bool sticky=!pump(true)&&pipe_completion_uncertain&&backend_calls==1;
  check(sticky&&!inject_assertion,"real backend failure sticky");
  check(!pump(false)&&backend_calls==1,"uncertain backend never retried");
 }else{
  o.owned_session_lifetime=false;in_shared->input_error.value=true;
  check(!pump(true)&&backend_calls==0,"legacy admission still checked in step");
 }
 {
  strata::core::SubmittedWorkDrainGuard guard(o.owned_session_lifetime,pipe_completion_uncertain,pipe_drain);
 }
 if(o.owned_session_lifetime)check(remaining==0,"all admitted groups drained");
}
static constexpr const char* fatal_teardown=
 "strata: submitted work could not finish before controller teardown; worker restart required\n";
struct ChildResult {
 int status=0;std::string diagnostics;bool timed_out=false,reaped=false,no_child=false,stderr_eof=false,status_lost=false;int kills=0;
 const char* failure=nullptr;
};
enum class ChildBehavior {normal,hang_open,hang_closed,status_loss};
static ChildResult scenario(int mode,bool inject=false,bool noisy=false,
                           ChildBehavior behavior=ChildBehavior::normal){
 // This single-threaded harness owns child reaping. Refuse inherited auto-reap
 // or custom handlers before creating a PID; a concurrent reaper would break
 // the wait-before-kill ownership contract even after a successful preflight.
 struct sigaction disposition{};
 check(sigaction(SIGCHLD,nullptr,&disposition)==0&&disposition.sa_handler==SIG_DFL&&
       !(disposition.sa_flags&SA_NOCLDWAIT),"exclusive child status ownership required");
 int fds[2];check(pipe(fds)==0,"diagnostic pipe");
 const int flags=fcntl(fds[0],F_GETFL,0);
 if(flags<0||fcntl(fds[0],F_SETFL,flags|O_NONBLOCK)<0){close(fds[0]);close(fds[1]);check(false,"nonblocking diagnostic pipe");}
 std::fflush(nullptr);
 const auto pid=fork();
 if(pid<0){close(fds[0]);close(fds[1]);check(false,"fork");}
 if(!pid){
  close(fds[0]);check(dup2(fds[1],STDERR_FILENO)>=0,"diagnostic redirect");close(fds[1]);
  if(behavior==ChildBehavior::status_loss)std::_Exit(0);
  if(behavior!=ChildBehavior::normal){
   if(behavior==ChildBehavior::hang_closed)close(STDERR_FILENO);
   for(;;)pause();
  }
  if(noisy)std::fputs("FAIL: injected teardown diagnostic noise\nAddressSanitizer: injected diagnostic marker only\n",stderr);
  run(mode,inject);std::_Exit(0);
 }
 close(fds[1]);
 // No allocating diagnostics operation occurs while an unreaped child is live.
 // Cleanup targets only our own unreaped PID: never a recycled/reaped PID.
 struct ChildOwner {
  pid_t pid;int fd;ChildBehavior behavior;int status=0,kills=0;
  bool owns_pid=true,reaped=false,status_lost=false,actually_reaped=false;
  pid_t wait(int flags) noexcept {
   const auto result=waitpid(pid,&status,flags);
   if(result==pid){
    actually_reaped=true;
    // Safe syscall seam: consume the real child's status FIRST, then simulate
    // loss. No risky recycled-PID signal is needed to test disarming.
    if(behavior==ChildBehavior::status_loss){errno=ECHILD;return -1;}
   }
   return result;
  }
  void observe(pid_t result,int error) noexcept {
   if(result==pid){reaped=true;owns_pid=false;}
   else if(result<0&&error==ECHILD){status_lost=true;owns_pid=false;}
  }
  void finish() noexcept {
   if(owns_pid){
    pid_t result;do{result=wait(WNOHANG);}while(result<0&&errno==EINTR);
    observe(result,errno);
    if(owns_pid&&result==0){
     // Test seam refuses to signal a PID whose real status was already reaped.
     if(actually_reaped)std::_Exit(assertion_exit);
     ++kills;(void)kill(pid,SIGKILL);
     do{result=wait(0);}while(result<0&&errno==EINTR);
     observe(result,errno);
    }
   }
   if(fd>=0){close(fd);fd=-1;}
  }
  ~ChildOwner(){finish();}
 } owner{pid,fds[0],behavior};
 ChildResult out;std::array<char,65536> diagnostics{};size_t used=0;bool eof=false;
 const auto deadline=std::chrono::steady_clock::now()+
     std::chrono::milliseconds(behavior==ChildBehavior::normal?3000:150);
 while(!eof||!owner.reaped){
  if(std::chrono::steady_clock::now()>=deadline){out.timed_out=true;break;}
  // Bound work per pass so a noisy writer cannot starve the wait/deadline check.
  for(int chunk=0;!eof&&chunk<16;++chunk){
   char buffer[1024];const auto n=read(owner.fd,buffer,sizeof buffer);
   if(n<0){if(errno==EINTR)continue;if(errno==EAGAIN||errno==EWOULDBLOCK)break;out.failure="diagnostic read";break;}
   if(!n){eof=true;break;}
   if(size_t(n)>diagnostics.size()-used){out.failure="diagnostic limit";break;}
   std::memcpy(diagnostics.data()+used,buffer,size_t(n));used+=size_t(n);
  }
  if(out.failure)break;
  if(!owner.reaped){
   const auto waited=owner.wait(WNOHANG);const int wait_error=errno;
   owner.observe(waited,wait_error);
   if(waited<0&&wait_error!=EINTR){out.failure="wait child";break;}
  }
  if(eof&&owner.reaped)break;
  pollfd ready{owner.fd,POLLIN|POLLHUP,0};
  const int polled=poll(eof?nullptr:&ready,eof?0:1,10);
  if(polled<0&&errno!=EINTR){out.failure="diagnostic poll";break;}
 }
 // EOF is not process completion. Timeout/error must kill and reap even when
 // stderr was closed already. SIGKILL cleanup waits for kernel reap before any
 // failure is reported; this is not a promise to interrupt unkillable kernel IO.
 out.stderr_eof=eof;
 owner.finish();out.status=owner.status;out.reaped=owner.reaped;
 out.status_lost=owner.status_lost;out.kills=owner.kills;
 int probe=0;errno=0;out.no_child=waitpid(pid,&probe,WNOHANG)==-1&&errno==ECHILD;
 out.diagnostics.assign(diagnostics.data(),used);
 return out;
}
static bool accepted(const ChildResult& child,int mode){
 if(child.timed_out||child.status_lost||child.failure||!child.reaped||!child.no_child||!WIFEXITED(child.status))return false;
 if(mode==2)return WEXITSTATUS(child.status)==1&&child.diagnostics==fatal_teardown;
 return WEXITSTATUS(child.status)==0&&child.diagnostics.empty();
}
static void status_loss_control(){
 const auto child=scenario(0,false,false,ChildBehavior::status_loss);
 check(child.status_lost&&!child.reaped&&child.no_child&&child.kills==0&&
       child.failure&&!child.timed_out&&!accepted(child,0),
       "lost child status must disarm PID without qualifying completion");
}
static void timeout_negative_controls(){
 for(auto behavior:{ChildBehavior::hang_open,ChildBehavior::hang_closed}){
  const auto child=scenario(0,false,false,behavior);
  check(child.timed_out&&child.reaped&&child.no_child&&!child.failure&&child.kills==1&&
        child.stderr_eof==(behavior==ChildBehavior::hang_closed)&&
        WIFSIGNALED(child.status)&&WTERMSIG(child.status)==SIGKILL&&
        child.diagnostics.empty()&&!accepted(child,0),"timeout control must kill and reap exact child");
 }
}
int main(int argc,char**argv){

 if(argc==2&&std::strcmp(argv[1],"race")==0){run(0);return 0;}
 const bool timeout_controls=argc==2&&std::strcmp(argv[1],"--timeout-controls")==0;
 if(timeout_controls){
  timeout_negative_controls();status_loss_control();
  std::puts("PASS timeout controls: open/closed stderr children killed and reaped; ECHILD proved");
  return 0;
 }
 const bool inject=argc==2&&std::strcmp(argv[1],"--inject-assertion")==0;
 check(argc==1||inject,"known fixture mode");
 for(int mode=0;mode<4;++mode){
  auto result=scenario(mode,inject&&mode==2);
  if(!accepted(result,mode))std::fwrite(result.diagnostics.data(),1,result.diagnostics.size(),stderr);
  check(accepted(result,mode),"scenario result rejected");
 }
 // The public default test itself proves the parent oracle rejects the formerly
 // ambiguous sticky assertion, even if no external negative-control mode is run.
 auto negative=scenario(2,true);
 check(negative.reaped&&negative.no_child&&!negative.timed_out&&!negative.failure&&WIFEXITED(negative.status)&&WEXITSTATUS(negative.status)==assertion_exit&&
       negative.diagnostics=="FAIL: real backend failure sticky\n"&&!accepted(negative,2),
       "assertion injection must not satisfy fatal teardown oracle");
 auto noisy=scenario(2,false,true);
 check(noisy.reaped&&noisy.no_child&&!noisy.timed_out&&!noisy.failure&&WIFEXITED(noisy.status)&&WEXITSTATUS(noisy.status)==1&&!accepted(noisy,2),
       "fatal status with assertion or sanitizer diagnostic noise must be rejected");
 timeout_negative_controls();status_loss_control();
 std::puts("PASS pipeline scenarios and rejected assertion-injection negative control");
}
