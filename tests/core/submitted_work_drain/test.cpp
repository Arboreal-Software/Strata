#include "submitted_work_drain.hpp"
#include <cassert>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#define STRATA_WINDOW_STATE_AVAILABLE 1
struct Shared {int calls=0,remaining=0,targets_destroyed=0,bells_joined=0,may_start_calls=0;};
static void scenario(Shared& s,int mode){
 struct Target {Shared& s;~Target(){++s.targets_destroyed;}} target{s};
 struct Options{bool owned_session_lifetime=true;}o;
 if(mode==5)o.owned_session_lifetime=false;
 struct Input {std::atomic<bool> input_error{false};};
 auto in_shared=std::make_shared<Input>();
 s.remaining=3;
 auto pipe_inflight=[&]{return s.remaining!=0;};
 auto pump_step=[&](bool may_start){
   ++s.calls;if(may_start)++s.may_start_calls;
   assert(!s.targets_destroyed&&s.bells_joined);
   if(mode==2)return false;
   if(mode==3)throw std::runtime_error("pump");
   --s.remaining;return true;
 };
#include "controller.inc"
 {
  strata::core::SubmittedWorkDrainGuard guard(mode!=5,pipe_completion_uncertain,pipe_drain);
  struct Bells {Shared& s;~Bells(){s.bells_joined=1;}} bells{s};
  if(mode==7){in_shared->input_error=true;assert(!pump(true)&&s.calls==0&&!pipe_completion_uncertain);}
  if(mode==1)pipe_completion_uncertain=true;
  if(mode==4){s.bells_joined=1;try{throw std::runtime_error("outer");}catch(...){throw;}}
  if(mode==6){s.bells_joined=1;pipe_completion_uncertain=true;assert(!pump(false)&&s.calls==0);}
 }
}
int main(){
 auto* s=static_cast<Shared*>(mmap(nullptr,sizeof(Shared),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));assert(s!=MAP_FAILED);
 for(int mode=0;mode<8;++mode){
  *s=Shared{};auto child=fork();assert(child>=0);
  if(!child){try{scenario(*s,mode);}catch(const std::runtime_error&){assert(mode==4);}std::_Exit(0);}
  int status=0;assert(waitpid(child,&status,0)==child&&WIFEXITED(status));
  const bool failed=mode==1||mode==2||mode==3||mode==6;
  assert(WEXITSTATUS(status)==(failed?1:0));
  assert(s->may_start_calls==0&&s->bells_joined==1);
  if(failed){assert(s->targets_destroyed==0);assert(s->calls==(mode==2||mode==3?1:0));assert(s->remaining==3);}
  else if(mode==5){assert(s->calls==0&&s->remaining==3&&s->targets_destroyed==1);}
  else {assert(s->calls==3&&s->remaining==0&&s->targets_destroyed==1);}
 }
 assert(munmap(s,sizeof(Shared))==0);
}
