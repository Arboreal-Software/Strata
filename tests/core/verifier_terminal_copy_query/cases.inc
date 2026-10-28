static void retained(const Verifier& v){
 check(v.fl_active_&&v.copy_used_&&v.fl_k_==2&&v.fl_total_==2&&v.windows==0&&
       v.commit_pending_&&!v.released_.load()&&profiles==0&&samples==0&&beats==0&&nonfinite==0&&releases==0,
       "terminal query preserves work flags and counters without releasing GPU waits");
}
int main(int argc,char**argv){
 check(argc==2,"one scenario required");const std::string mode=argv[1];Verifier v;std::string error;
 if(mode=="--pending"){
  copy_status=cudaErrorNotReady;check(!v.done(error)&&error.empty(),"copy pending remains pending");
  check(!v.terminal_copy_query_failed_.load()&&v.copy_used_&&v.fl_active_&&v.windows==0,"pending is not terminal");
  check(event_queries==1&&copy_queries==1,"pending queries each stream once");return 0;
 }
 if(mode=="--success"){
  check(v.done(error)&&error.empty()&&!v.copy_used_,"successful copy completes");int32_t out=-1;
  check(v.pl_finish(&out,error)&&out==7&&!v.fl_active_&&v.windows==1&&profiles==1&&nonfinite==1&&beats==1,
        "success preserves normal profile picks and counters");return 0;
 }
 check(mode=="--copy-error"||mode=="--allocation","known scenario");copy_status=cudaErrorUnknown;
 if(mode=="--allocation"){
  bool threw=false;deny_new=true;
  try{(void)v.done(error);}catch(const std::bad_alloc&){threw=true;}
  deny_new=false;check(threw,"diagnostic allocation failure injected");
 }else{
  // Exact expected-red oracle: the old copy check returns true on CUDA error.
  check(!v.done(error),"copy query error must not report pipeline completion");
 }
 check(v.terminal_copy_query_failed_.load(),"terminal latch survives diagnostic allocation failure");retained(v);
 const int before_events=event_queries,before_copies=copy_queries,before_devices=device_calls;
 copy_status=cudaSuccess;error.clear();
 check(!v.done(error)&&!error.empty(),"repeat done refuses without healing");
 int32_t out=-99,token=7,previous[2]{};int64_t pos=0;
 check(!v.pl_finish(&out,error)&&out==-99,"finish refuses before clearing active or publishing output");
 check(!v.prestage(1,&token,0,previous,error),"prestage refuses");
 check(!v.run_slots(1,&token,&pos,nullptr,nullptr,&out,error),"slot wrapper refuses");
 check(!v.commit_slots(error)&&!v.wait_commit(error),"commit refuses");
 check(v.poll_snapshot_drain(error)==Verifier::SnapshotDrain::unhealthy,"snapshot cannot certify failed copy");
 check(v.check_entry_guards(error),"every actual entry first guard refuses");
 retained(v);
 check(event_queries==before_events&&copy_queries==before_copies&&device_calls==before_devices&&effects==0,
       "terminal entries perform no query callback staging or device effect");
 // Retry diagnostics may also throw, but must not resume/query the failed work.
 std::string().swap(error);bool threw=false;deny_new=true;
 try{(void)v.done(error);}catch(const std::bad_alloc&){threw=true;}
 deny_new=false;check(threw&&v.terminal_copy_query_failed_.load(),"repeat diagnostic failure leaves terminal latch");
 check(event_queries==before_events&&copy_queries==before_copies&&device_calls==before_devices,"throwing refusal has no effects");
}
