#include "submitted_work_drain.hpp"
#include <atomic>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <string>
#include <csignal>
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
struct ChildResult {int status=0;std::string diagnostics;};
static ChildResult scenario(int mode,bool inject=false,bool noisy=false){
 int fds[2];check(pipe(fds)==0,"diagnostic pipe");
 std::fflush(nullptr);
 const auto pid=fork();check(pid>=0,"fork");
 if(!pid){
  close(fds[0]);check(dup2(fds[1],STDERR_FILENO)>=0,"diagnostic redirect");close(fds[1]);
  if(noisy)std::fputs("FAIL: injected teardown diagnostic noise\nAddressSanitizer: injected diagnostic marker only\n",stderr);
  run(mode,inject);std::_Exit(0);
 }
 close(fds[1]);ChildResult out;char buffer[1024];
 for(;;){
  const auto n=read(fds[0],buffer,sizeof buffer);
  if(n<0&&errno==EINTR)continue;
  if(n<0){kill(pid,SIGKILL);close(fds[0]);waitpid(pid,&out.status,0);check(false,"diagnostic read");}
  if(!n)break;
  if(out.diagnostics.size()+size_t(n)>65536){kill(pid,SIGKILL);close(fds[0]);waitpid(pid,&out.status,0);check(false,"diagnostic limit");}
  out.diagnostics.append(buffer,size_t(n));
 }
 close(fds[0]);pid_t waited;
 do{waited=waitpid(pid,&out.status,0);}while(waited<0&&errno==EINTR);
 check(waited==pid,"wait child");return out;
}
static bool accepted(const ChildResult& child,int mode){
 if(!WIFEXITED(child.status))return false;
 if(mode==2)return WEXITSTATUS(child.status)==1&&child.diagnostics==fatal_teardown;
 return WEXITSTATUS(child.status)==0&&child.diagnostics.empty();
}
int main(int argc,char**argv){
 alarm(25);
 if(argc==2&&std::strcmp(argv[1],"race")==0){run(0);return 0;}
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
 check(WIFEXITED(negative.status)&&WEXITSTATUS(negative.status)==assertion_exit&&
       negative.diagnostics=="FAIL: real backend failure sticky\n"&&!accepted(negative,2),
       "assertion injection must not satisfy fatal teardown oracle");
 auto noisy=scenario(2,false,true);
 check(WIFEXITED(noisy.status)&&WEXITSTATUS(noisy.status)==1&&!accepted(noisy,2),
       "fatal status with assertion or sanitizer diagnostic noise must be rejected");
 std::puts("PASS pipeline scenarios and rejected assertion-injection negative control");
}
