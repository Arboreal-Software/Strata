#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#include <thread>
static bool deny_new=false;
void* operator new(std::size_t n){if(deny_new)throw std::bad_alloc();if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void*p)noexcept{std::free(p);}
void operator delete(void*p,std::size_t)noexcept{std::free(p);}
using cudaError_t=int;using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaErrorNotReady=1,cudaErrorUnknown=999;
static void* main_stream=reinterpret_cast<void*>(1);static void* copy_stream=reinterpret_cast<void*>(2);
static int main_status=0,main_queries=0,drained_traces=0,fences=0;
static void _mm_sfence(){++fences;} // explicit CPU fence seam
static int event_status=0,copy_status=0,event_queries=0,copy_queries=0,device_calls=0,effects=0;
static int route_copies=0,route_resets=0;
constexpr int cudaMemcpyDeviceToHost=2;
static int cudaMemcpy(void*,const void*,size_t,int){++route_copies;return cudaSuccess;}
static int cudaMemset(void*,int,size_t){++route_resets;return cudaSuccess;}
static int profiles=0,samples=0,beats=0,nonfinite=0,releases=0;
static int cudaStreamQuery(void*s){if(s==copy_stream){++copy_queries;return copy_status;}++main_queries;return main_status;}
static int cudaEventQuery(void*){++event_queries;return event_status;}
static int cudaStreamSynchronize(void*){++effects;return cudaSuccess;}
static int cudaEventSynchronize(void*){++effects;return cudaSuccess;}
static const char* cudaGetErrorString(int){return "injected terminal copy query error";}
static double now_ms(){return 0;}
static void progress_beat(){++beats;}
static std::string released_note(bool){return {};}
struct OnDevice {explicit OnDevice(int){++device_calls;}};
struct WindowDeviceScope {explicit WindowDeviceScope(int){++device_calls;}bool ok(){return true;}bool finish(){++device_calls;return true;}};
using Clock=std::chrono::steady_clock;
static double ms_since(Clock::time_point){return 0;}
using PoolMultiFn=void(*)();
struct ModelGeometry {int n_layers=1;};
namespace strata::kernels {
struct SamplerParams {bool greedy=true;float temperature=0;const float* logit_bias=nullptr;uint64_t counter=0;};
template<class...T>void sample_tokens(T...){++samples;}
}
struct Verifier {
#include "latch.inc"
 bool fl_active_=true,copy_used_=true,fl_prof_=true,head_sampling_=false,commit_pending_=true;
 bool batch_copy_failed_=false,pl_prestaged_=false;
 std::atomic<bool>released_{false};
 int fl_k_=2,fl_total_=2,fl_T_=1,device_=0,le_=1,max_t_=1,last_t_=1,windows=0;
 double fl_since_ms_=0,fl_flush_ms_=0,fl_launch_ms_=0,ms_host=0;
 int64_t last_pos0_=0,n_vocab_=1;int hist_len_=0;
 void*cs_=main_stream,*copy_=copy_stream,*sh_cs_=nullptr,*side_=nullptr,*side2_=nullptr,*df_side_[2]{};
 void*ev_done_=reinterpret_cast<void*>(3),*commit_done_=reinterpret_cast<void*>(4);
 std::string snapshot_failure_,commit_failure_;Verifier*next_=nullptr;
 ModelGeometry geometry;ModelGeometry*g_=&geometry;
 strata::kernels::SamplerParams sampling_;
 int32_t picked=7;int32_t*h_out_=&picked,*m_out_=&picked;const int32_t*hist_d_=nullptr;
 float*head_logits_=nullptr;unsigned long long*prof_pin_=nullptr;
 std::vector<uint32_t>pl_ple_rows_{0};std::vector<int>slots_{0};int last_rows_[8]{};
 bool rs_on_=true;unsigned long long route_device[72]{};void*rs_dev_=route_device;int64_t rs_win_[9]{0,3};
 std::string route_stats_report();
 uint32_t flags[4]{};uint32_t*h_flag_=&flags[0],*h_flagA_=&flags[1],*h_flagB_=&flags[2],*hflag_in_h_=&flags[3];
 bool release_gpu_waits(int);
 void accumulate_profile(const unsigned long long*){++profiles;}
 void trace_ev(const char*tag,int64_t,int64_t,int64_t){if(!std::strcmp(tag,"RELEASE"))++releases;if(!std::strcmp(tag,"RELEASE-DRAINED"))++drained_traces;}
 void note_nonfinite(int,const int64_t*){++nonfinite;}
 bool handoff_failed(std::string&){return false;}
 void pl_stage(int,const int32_t*,int64_t,const int32_t*){++effects;}
 bool run_slot_rows(const int*,int,const int32_t*,const int64_t*,PoolMultiFn,void*,int32_t*,std::string&){++effects;return true;}
 bool commit_slot_prefixes(const int*,std::string&){++effects;return true;}
 enum class SnapshotDrain{pending,drained,unhealthy,unsupported};
 bool done(std::string&);bool pl_finish(int32_t*,std::string&);
 bool prestage(int,const int32_t*,int64_t,const int32_t*,std::string&);
 bool run_slots(int,const int32_t*,const int64_t*,PoolMultiFn,void*,int32_t*,std::string&);
 bool commit_slots(std::string&);bool wait_commit(std::string&);
 SnapshotDrain poll_snapshot_drain(std::string&);
#include "entry-guards.inc"
};
static void check(bool yes,const char*what){if(!yes){std::fprintf(stderr,"FAIL: %s\n",what);std::fflush(stderr);std::_Exit(86);}}
