#include "strata/core/host_graph_launch.hpp"
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
static std::mutex mu;
static std::condition_variable cv;
static bool entered=false,permit=false,launched=false,throw_downstream=false;
static int launch_error=0,copy_error=0,releases=0,primary_syncs=0,copy_syncs=0;
static bool pending_after=false;
static void check(bool ok,const char* message){if(!ok){std::fprintf(stderr,"ASSERT: %s\n",message);std::exit(86);}}
int cudaSetDevice(int){return 0;}
int cudaGraphLaunch(int,int){
 std::unique_lock lock(mu);entered=true;cv.notify_all();cv.wait(lock,[]{return permit;});
 launched=true;cv.notify_all();return launch_error;
}
int cudaStreamQuery(int){std::lock_guard lock(mu);return launched?launch_error:cudaErrorNotReady;}
int cudaStreamSynchronize(int){
 ++primary_syncs;std::unique_lock lock(mu);cv.wait(lock,[]{return launched;});return 0;
}
const char* cudaGetErrorString(int){return "mock launch failure";}
struct OnDevice{explicit OnDevice(int){}};
struct Next {
 bool run(int,const int32_t*,int64_t,void*,void*,int32_t*,std::string&){
  if(throw_downstream)throw std::runtime_error("downstream exception");
  {std::lock_guard lock(mu);permit=true;}cv.notify_all();return true;
 }
};
struct JoinedNext {bool handed=false;bool join(std::string&){return true;}};
struct Fixture {
 Next* next_;JoinedNext nl;bool chain=true,commit_pending_=true,prof_on_=false;int windows=0;
 int T=1,device_=0,cs_=1,G=1;int32_t* tokens=nullptr;int32_t* out=nullptr;
 int64_t pos0=0;void* pool=nullptr;void* next_user_=nullptr;int* h_plan_err_=nullptr;
 bool finish_serial_copies(std::string& error){++copy_syncs;if(copy_error){error="copy failure";return false;}return true;}
 bool handoff_failed(std::string&){return false;}
 bool ar_on(){return false;}
 void collect_profile(){}
 bool run(std::optional<strata::core::HostGraphLaunch>& launch_o,std::string& err){
#ifdef BEFORE
#include "before.inc"
#else
#include "fixed.inc"
#endif
  return false;
 }
};
static void reset(){entered=permit=launched=throw_downstream=false;launch_error=copy_error=releases=primary_syncs=copy_syncs=0;pending_after=false;}
static bool unit(std::string& error,bool threaded=true){
 Next next;Fixture f{&next};
 std::optional<strata::core::HostGraphLaunch> launch;
 if(!threaded)permit=true;
 launch.emplace(1,1,0,threaded,[]{++releases;{std::lock_guard lock(mu);permit=true;}cv.notify_all();});
 if(threaded){std::unique_lock lock(mu);cv.wait(lock,[]{return entered;});}
 check(launch->status()==cudaSuccess,"pending or successful initial launch status");
 const bool ok=f.run(launch,error);pending_after=f.commit_pending_;return ok;
}
int main(){
 reset();launch_error=7;std::string error;
 check(!unit(error),"late asynchronous chain launch error must refuse completion");
 check(error=="verify: launch: mock launch failure"&&pending_after,"launch failure preserves pending state and diagnostic");
 check(primary_syncs==0&&copy_syncs==0&&releases==0,"failed checked finish precedes stream checks and suppresses destructor release");
 reset();error.clear();check(unit(error),"asynchronous chain succeeds");
 check(!pending_after&&primary_syncs==1&&copy_syncs==1&&releases==0,"successful chain finishes launcher before streams and no destructor wait release");
 reset();copy_error=9;check(!unit(error)&&pending_after,"copy failure retains pending state");
 check(primary_syncs==1&&copy_syncs==1&&releases==0,"copy failure follows checked launcher completion");
 reset();throw_downstream=true;bool threw=false;try{unit(error);}catch(const std::runtime_error&){threw=true;}
 check(threw&&releases==1&&launched&&primary_syncs==0&&copy_syncs==0,"downstream exception retains destructor release and joins actual worker");
 reset();check(unit(error,false)&&releases==0,"nonthreaded launch remains successful without release callback");
 std::puts("PASS: actual HostGraphLaunch and chain completion extraction");
}
