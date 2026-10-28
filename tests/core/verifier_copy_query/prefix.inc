#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <new>
#include <string>
#include <vector>
static bool deny_new=false;
void* operator new(std::size_t n){if(deny_new)throw std::bad_alloc();if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void*p)noexcept{std::free(p);}
void operator delete(void*p,std::size_t)noexcept{std::free(p);}
using cudaError_t=int;using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaErrorNotReady=1,cudaErrorUnknown=999;
static void* main_stream=reinterpret_cast<void*>(1);static void* copy_stream=reinterpret_cast<void*>(2);
static int main_status=0,copy_status=0,queries=0,device_calls=0,profiles=0,samples=0,beats=0;
static int cudaStreamQuery(void*s){++queries;return s==copy_stream?copy_status:main_status;}
static int cudaEventQuery(void*){++queries;return 0;}
static int cudaGraphLaunch(void*,void*){++queries;return 0;}
static const char* cudaGetErrorString(int){return "injected copy query failure";}
static void progress_beat(){++beats;}
static std::string released_note(int){return {};}
struct OnDevice {explicit OnDevice(int){++device_calls;}};
struct WindowDeviceScope {explicit WindowDeviceScope(int){++device_calls;}bool ok(){return true;}bool finish(){++device_calls;return true;}};
struct SessionState {int ple_prev[2]{};};
using Clock=std::chrono::steady_clock;
using PoolMultiFn=void(*)();
struct Verifier {
#include "../verifier_terminal_copy_query/latch.inc"
 bool batch_copy_failed_=false;std::atomic<bool>b_running_{true},ext_serve_{false},released_{false};
 std::atomic<int64_t>b_k_{0};int64_t b_steps_=0;int lb_=0,le_=1,device_=0,last_t_=1,last_pos_b_=0;
 Clock::time_point ext_checked_=Clock::now();uint32_t seq=0;uint32_t*h_seq_=&seq;
 void*cs_=main_stream,*copy_=copy_stream,*sh_cs_=nullptr,*side_=nullptr,*side2_=nullptr,*df_side_[2]{};
 bool prof_on_=true,commit_pending_=false,copy_used_=true;void*commit_done_=nullptr;
 std::string snapshot_failure_,commit_failure_;Verifier*next_=nullptr;
 int32_t picked=7,out=-99;int32_t*h_out_=&picked,*b_out_=&out;int windows=0;
 std::function<bool(void*,std::string&)>pre_launch_;std::map<int,void*>exec_bm_,commit_bm_;std::vector<SessionState*>slots_;
 bool ar_on(){return false;}Clock::time_point last_ring(){return Clock::now();}int release_gpu_waits(int){return 0;}
 int batch_serve(PoolMultiFn,void*){return 1;}void collect_profile(){++profiles;}
 bool last_stage(){return true;}bool sample_rows(int,std::string&){++samples;return true;}void note_nonfinite(int,int){}
 bool handoff_failed(std::string&){return false;}bool stage_batch(int*,int,int,const int32_t*,const int64_t*,std::string&){return true;}
 int bkey(int*,int,int){return 0;}bool batch_self_commits(int*,int){return true;}bool ple_stage(){return false;}void ring_now(){}
 enum class SnapshotDrain{pending,drained,unhealthy,unsupported};
 int batch_poll(PoolMultiFn,void*,std::string&);
 bool batch_launch(int,int,const int32_t*,const int64_t*,std::string&);
 SnapshotDrain poll_snapshot_drain(std::string&);
};
static void check(bool yes,const char*what){if(!yes){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}}
