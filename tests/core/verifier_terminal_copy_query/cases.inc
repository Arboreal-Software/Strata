static void retained(const Verifier& v){
 check(v.fl_active_&&v.copy_used_&&v.fl_k_==2&&v.fl_total_==2&&v.windows==0&&
       v.commit_pending_&&!v.released_.load()&&profiles==0&&samples==0&&beats==0&&nonfinite==0&&releases==0,
       "terminal query preserves work flags and counters without releasing GPU waits");
}
int main(int argc,char**argv){
 check(argc==2,"one scenario required");const std::string mode=argv[1];Verifier v;std::string error;
 if(mode=="--release-query-error"){
  main_status=cudaErrorUnknown;
  check(!v.release_gpu_waits(10),"main query error cannot certify drain");
  check(v.released_.load()&&releases==1&&fences==1&&drained_traces==0,"host release effects retained on query failure");
  for(auto f:v.flags)check(f==UINT32_MAX,"every mapped wait flag released");
  check(main_queries==1&&copy_queries==0,"stop on main query uncertainty");
  main_status=cudaSuccess;copy_status=cudaErrorUnknown;
  check(!v.release_gpu_waits(10)&&drained_traces==0,"copy query error cannot certify drain");
  check(main_queries==2&&copy_queries==1,"both query branches tested");return 0;
 }
 if(mode=="--release-sticky"){
  copy_status=cudaErrorUnknown;check(!v.done(error),"inject terminal copy error before release");
  check(v.terminal_copy_query_failed_.load(),"copy latch established");
  copy_status=cudaSuccess;
  check(v.release_gpu_waits(10),"actual release method remains available after latch");
  check(v.released_.load()&&releases==1&&fences==1&&drained_traces==1,"actual host release and successful drain trace");
  for(auto f:v.flags)check(f==UINT32_MAX,"all actual mapped wait flags raised");
  check(v.terminal_copy_query_failed_.load()&&v.fl_active_&&v.copy_used_&&v.windows==0,
        "release neither clears latch nor publishes work completion");
  const int cq=copy_queries,eq=event_queries,mq=main_queries,dc=device_calls;
  int32_t out=-99;
  check(!v.done(error)&&!v.pl_finish(&out,error)&&out==-99,"completion and finish remain refused after release");
  check(v.poll_snapshot_drain(error)==Verifier::SnapshotDrain::unhealthy,"snapshot remains unhealthy after release");
  check(copy_queries==cq&&event_queries==eq&&main_queries==mq&&device_calls==dc&&profiles==0&&beats==0&&samples==0,
        "postrelease refusal has no query or publication effects");return 0;
 }
 if(mode=="--pending"){
  copy_status=cudaErrorNotReady;check(!v.done(error)&&error.empty(),"copy pending remains pending");
  check(!v.terminal_copy_query_failed_.load()&&v.copy_used_&&v.fl_active_&&v.windows==0,"pending is not terminal");
  check(event_queries==1&&copy_queries==1,"pending queries each stream once");
  copy_status=cudaSuccess;check(v.done(error)&&!v.copy_used_,"same pending object completes after copy succeeds");
  int32_t out=-1;check(v.pl_finish(&out,error)&&out==7&&v.windows==1&&!v.fl_active_,"same object publishes once");
  const int cq=copy_queries,eq=event_queries;out=-99;
  check(!v.done(error)&&out==-99&&v.windows==1&&profiles==1&&beats==1&&nonfinite==1,
        "finished object cannot announce another completion or duplicate publication");
  check(copy_queries==cq&&event_queries==eq,"finished poll has no query");return 0;
 }
 if(mode=="--copy-unused"){
  v.copy_used_=false;copy_status=cudaErrorUnknown;
  check(v.done(error)&&copy_queries==0&&!v.terminal_copy_query_failed_.load(),"unused copy stream never queried");
  int32_t out=-1;check(v.pl_finish(&out,error)&&out==7&&copy_queries==0&&v.windows==1,
                      "unused copy path finishes without querying unrelated failed stream");return 0;
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
 // Complete production report method: forbid allocation and preserve both
 // device and host statistics after the terminal latch.
 deny_new=true;const auto report=v.route_stats_report();deny_new=false;
 check(report.empty()&&route_copies==0&&route_resets==0&&v.rs_win_[1]==3,
       "route report refuses before allocation copy and device/host reset");
 check(event_queries==before_events&&copy_queries==before_copies&&device_calls==before_devices&&effects==0,
       "terminal entries perform no query callback staging or device effect");
 // Retry diagnostics may also throw, but must not resume/query the failed work.
 std::string().swap(error);bool threw=false;deny_new=true;
 try{(void)v.done(error);}catch(const std::bad_alloc&){threw=true;}
 deny_new=false;check(threw&&v.terminal_copy_query_failed_.load(),"repeat diagnostic failure leaves terminal latch");
 check(event_queries==before_events&&copy_queries==before_copies&&device_calls==before_devices,"throwing refusal has no effects");
}
