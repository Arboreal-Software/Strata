int main(int argc,char**argv){
 std::string err;Verifier v;
 if(argc>1&&std::string(argv[1])=="--copy-error"){
  copy_status=cudaErrorUnknown;int r=v.batch_poll(nullptr,nullptr,err);
  check(r==-1,"copy stream error must not report batch completion");
  check(v.b_running_&&v.batch_copy_failed_&&v.windows==0&&v.out==-99&&profiles==0&&samples==0&&beats==0,"error retains uncertain batch without publication");
  int q=queries,dev=device_calls;copy_status=cudaSuccess;
  check(v.batch_poll(nullptr,nullptr,err)==-1,"later success cannot heal failed copy query");
  check(v.poll_snapshot_drain(err)==Verifier::SnapshotDrain::unhealthy,"failed batch cannot certify snapshot");
  check(!v.batch_launch(0,1,nullptr,nullptr,err),"failed verifier cannot launch another window");
  check(queries==q&&device_calls==dev,"sticky refusal performs no device effects");
 }else if(argc>1&&std::string(argv[1])=="--allocation"){
  copy_status=cudaErrorUnknown;deny_new=true;bool caught=false;
  try{v.batch_poll(nullptr,nullptr,err);}catch(const std::bad_alloc&){caught=true;}deny_new=false;
  check(caught&&v.batch_copy_failed_&&v.b_running_,"latch survives diagnostic allocation failure");
  copy_status=0;int q=queries,dev=device_calls;
  check(v.batch_poll(nullptr,nullptr,err)==-1&&v.poll_snapshot_drain(err)==Verifier::SnapshotDrain::unhealthy,"allocation failure cannot heal terminal uncertainty");
  check(queries==q&&device_calls==dev&&v.windows==0&&v.out==-99,"allocation failure retains output/state");
 }else{
  copy_status=cudaErrorNotReady;check(v.batch_poll(nullptr,nullptr,err)==0,"copy pending stays pending");
  check(v.b_running_&&!v.batch_copy_failed_&&v.windows==0&&v.out==-99&&samples==0,"pending has no publication");
  copy_status=cudaSuccess;check(v.batch_poll(nullptr,nullptr,err)==1,"success completes");
  check(!v.b_running_&&v.windows==1&&v.out==7&&samples==1&&profiles==1&&beats==1,"success publishes once");
  int q=queries;check(v.batch_poll(nullptr,nullptr,err)==1&&queries==q&&v.windows==1,"completed poll stays idempotent");
 }
 std::puts("PASS exact verifier batch copy query fixture");
}
