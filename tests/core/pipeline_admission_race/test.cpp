#include "submitted_work_drain.hpp"
#include <atomic>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#define STRATA_WINDOW_STATE_AVAILABLE 1
static void check(bool v,const char* why){if(!v){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
struct Flag {
 std::atomic<bool> value{false};bool flip_after_next_load=false;
 bool load(){const bool prior=value.load();if(flip_after_next_load){flip_after_next_load=false;value=true;}return prior;}
};
static void run(int mode){
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
  check(!pump(true)&&pipe_completion_uncertain&&backend_calls==1,"real backend failure sticky");
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
int main(int argc,char**){
 if(argc>1){run(0);return 0;}
 for(int mode=0;mode<4;++mode){const auto pid=fork();check(pid>=0,"fork");if(!pid){run(mode);std::_Exit(0);}int s=0;check(waitpid(pid,&s,0)==pid&&WIFEXITED(s)&&WEXITSTATUS(s)==(mode==2?1:0),"scenario exit");}
}
