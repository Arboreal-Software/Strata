#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
static void check(bool x,const char* m){if(!x){std::fprintf(stderr,"ASSERT: %s\n",m);std::exit(70);}}
namespace strata::platform {void name_this_thread(const char*){}}
namespace strata::aux_cpus {void pin_current_thread(){}}
namespace strata::kernels::cpu {
bool cpu_avx2_ok(){return false;} bool cpu_avx1_ok(){return false;}
void bf16_rows_dot_multi(const uint16_t*,int,int,const float*,int,float*){std::abort();}
void bf16_rows_dot_multi_avx1(const uint16_t*,int,int,const float*,int,float*){std::abort();}
}
struct ExpertSource {
 std::function<void()> before_read;
 std::atomic<int> warms_count{0};
 bool warms(){return true;} bool wants_scores(){return false;}
 bool read_ahead_open(){if(before_read)before_read();return true;}
 void warm(int64_t,const int64_t*,int64_t){++warms_count;}
 void warm_scored(int64_t,const int64_t*,const float*,int64_t,int){std::abort();}
};
struct ForesightSwap {void predict(ExpertSource*,int64_t,int32_t){}};
[[noreturn]] static void retirement_failed(const char*){std::abort();}
// Test-only CV seam: worker and deadline waiters can be selected independently.
// notify_one legally chooses the deadline waiter, reproducing the lost-worker
// notification deterministically; production now broadcasts to both groups.
struct WakeCv {
 std::condition_variable worker,drain,control;
 std::mutex state;bool worker_waiting=false,pause_next=false,paused=false,release_notify=false;
 int drain_waiters=0;
 template<class L,class P>void wait(L& lock,P pred){
  {std::lock_guard s(state);worker_waiting=true;control.notify_all();}
  worker.wait(lock,pred);
 }
 template<class L,class T,class P>bool wait_until(L& lock,T until,P pred){
  {std::lock_guard s(state);++drain_waiters;control.notify_all();}
  const bool result=drain.wait_until(lock,until,pred);
  {std::lock_guard s(state);--drain_waiters;}
  return result;
 }
 void before_notify(){std::unique_lock s(state);if(pause_next){pause_next=false;paused=true;control.notify_all();control.wait(s,[&]{return release_notify;});}}
 void notify_all(){before_notify();worker.notify_all();drain.notify_all();}
 void notify_one(){before_notify();bool choose_drain;{std::lock_guard s(state);choose_drain=drain_waiters>0;}if(choose_drain)drain.notify_one();else worker.notify_one();}
 void wait_flag(const std::function<bool()>& pred){std::unique_lock s(state);check(control.wait_until(s,ClockForCv::now()+std::chrono::seconds(5),pred),"CV scheduling seam reached");}
 using ClockForCv=std::chrono::steady_clock;
 void arm(){wait_flag([&]{return worker_waiting;});std::lock_guard s(state);pause_next=true;}
 void unpause(){std::lock_guard s(state);release_notify=true;control.notify_all();}
};
// Actual declarations except one explicit CV-type substitution; exact methods
// and worker body execute. Freshness also retains the unmodified production class.
#define private public
#include "class-test.inc"
#undef private
#if defined(LOOKAHEAD_NO_SUBMIT_SEAL)
#include "methods-no-submit-seal.inc"
#elif defined(LOOKAHEAD_OLD_NOTIFY_ONE)
#include "methods-notify-one.inc"
#else
#include "methods.inc"
#endif
#include "worker.inc"
using R=RouterLookahead;using D=R::DrainResult;using Clock=std::chrono::steady_clock;
static_assert(!std::is_copy_constructible_v<R::SubmittedDrain>);
static_assert(!std::is_move_constructible_v<R::SubmittedDrain>);
struct Barrier {
 std::mutex m;std::condition_variable cv;bool entered=false,release=false;
 void block(){std::unique_lock l(m);entered=true;cv.notify_all();cv.wait(l,[&]{return release;});}
 void wait(){std::unique_lock l(m);check(cv.wait_until(l,Clock::now()+std::chrono::seconds(5),[&]{return entered;}),"worker reached source barrier");}
 void open(){std::lock_guard l(m);release=true;cv.notify_all();}
};
static void start(R& r,ExpertSource& s){std::string e;check(r.start(std::vector<std::vector<uint16_t>>(2,std::vector<uint16_t>(16)),8,2,1,&s,e),"start actual worker");}
int main(){int cases=0,idle_seal_checks=0;float x[8]{};int32_t res[4]={-1,-1,-1,-1};
 { // Real worker blocked BEFORE dereferencing borrowed host_res.
  R r;ExpertSource s;Barrier b;s.before_read=[&]{b.block();};start(r,s);r.submit(0,x,1,res);b.wait();
  R::SubmittedDrain held,other;
  check(r.drain_submitted(held)==D::busy,"active raw borrow cannot authorize writer");
  check(r.drain_submitted(held,Clock::time_point::min())==D::busy,"expired deadline retains seal");
  check(r.drain_submitted(other)==D::refused,"second writer refused");
  auto skipped=r.skipped();for(int i=0;i<4;++i)r.submit(0,x,1,res);
  check(r.skipped()==skipped+4,"new units cannot replace pending sealed borrow");
  b.open();check(r.drain_submitted(held,Clock::now()+std::chrono::seconds(5))==D::drained,"retained seal makes bounded progress");
  // Busy is now false. A retained seal must independently reject a new job.
  const auto idle_skipped=r.skipped();const auto idle_warms=s.warms_count.load();
  r.submit(0,x,1,res);
  // Settle even an incorrectly accepted counterfactual job BEFORE changing the
  // borrowed input or asserting. A timeout is cleanup failure, never expected red.
  const bool settled=r.drain_submitted(held,Clock::now()+std::chrono::seconds(5))==D::drained;
  const bool sealed=settled&&r.skipped()==idle_skipped+1&&s.warms_count==idle_warms;
  if(!sealed){held.reset();const bool closed=r.close_for_retirement();
   check(settled&&closed,"idle seal counterfactual cleanup completed");
   check(false,"idle retained guard must reject submission");
  }
  ++idle_seal_checks;
  res[2]=1; // Legal ONLY after actual source callback and raw reads complete.
  check(s.warms_count==1,"first prediction completes once");
  held.reset();r.submit(0,x,1,res);
  check(r.drain_submitted(held,Clock::now()+std::chrono::seconds(5))==D::drained,"reset resumes actual predictions");
  check(s.warms_count==2,"mode not disabled after mutation");++cases;
 }
 {R r;ExpertSource s;start(r,s);R::SubmittedDrain held;check(r.drain_submitted(held)==D::drained,"idle seal");
  check(r.close_for_retirement(),"close with guard alive");held.reset();auto n=r.skipped();r.submit(0,x,1,res);
  check(r.skipped()==n&&r.drain_submitted(held)==D::refused,"guard cannot reopen retired owner");++cases;}
 {R a,b;R::SubmittedDrain held;check(a.drain_submitted(held)==D::drained,"fresh idle guard");check(b.drain_submitted(held)==D::refused,"cross-owner guard rejected");held.reset();check(b.drain_submitted(held)==D::drained,"explicit reset permits reuse");++cases;}
 {R r;ExpertSource s;std::atomic<bool> refused=false;s.before_read=[&]{R::SubmittedDrain own;refused=r.drain_submitted(own)==D::refused;};start(r,s);r.submit(0,x,1,res);R::SubmittedDrain held;
  check(r.drain_submitted(held,Clock::now()+std::chrono::seconds(5))==D::drained,"worker finishes self-refusal");check(refused,"worker callback cannot drain itself");++cases;}
 { // Execute real worker catch on this test thread; production uncaught exception remains fatal.
  R r;ExpertSource s;s.before_read=[] {throw std::runtime_error("source failure");};r.src_=&s;r.routers_.assign(2,std::vector<uint16_t>(16));r.x_.assign(8,0);r.n_embd_=8;r.n_expert_=2;r.k_=1;r.submit(0,x,1,res);
  bool caught=false;std::thread failed([&]{try{r.run();}catch(const std::runtime_error&){caught=true;}});failed.join();check(caught&&r.worker_failed_,"actual worker exception is sticky");
  R::SubmittedDrain held;check(r.drain_submitted(held)==D::refused,"exception never becomes idle proof");++cases;}
 {R r;ExpertSource s;Barrier b;s.before_read=[&]{b.block();};start(r,s);r.submit(0,x,1,res);b.wait();
  R::SubmittedDrain held;check(r.drain_submitted(held)==D::busy,"terminal case sealed busy");
  std::promise<D> result;auto ready=result.get_future();
  std::thread waiter([&]{result.set_value(r.drain_submitted(held,Clock::now()+std::chrono::seconds(5)));});
  r.cv_.wait_flag([&]{return r.cv_.drain_waiters==1;});
  bool closed=false;std::thread closer([&]{closed=r.close_for_retirement();});
  const bool woke=ready.wait_until(Clock::now()+std::chrono::seconds(2))==std::future_status::ready;
  b.open();waiter.join();closer.join(); // always release/reap before asserting
  check(woke&&ready.get()==D::refused&&closed,"terminal retirement wakes blocked drain without claiming idle");
  held.reset();R::SubmittedDrain other;check(r.drain_submitted(other)==D::refused,"terminal remains sealed");++cases;
 }
 { // Pending, not busy: submit has released mu_ but has not notified yet.
  R r;ExpertSource s;start(r,s);r.cv_.arm();
  std::thread submitter([&]{r.submit(0,x,1,res);});
  r.cv_.wait_flag([&]{return r.cv_.paused;});
  R::SubmittedDrain held;D result=D::refused;
  std::thread waiter([&]{result=r.drain_submitted(held,Clock::now()+std::chrono::seconds(2));});
  r.cv_.wait_flag([&]{return r.cv_.drain_waiters==1;});
  r.cv_.unpause();submitter.join();waiter.join();
  // Cleanup must wake/reap even in the intentionally old-notify regression.
  r.cv_.notify_all();const auto cleanup=r.drain_submitted(held,Clock::now()+std::chrono::seconds(5));
  held.reset();const bool closed=r.close_for_retirement();
  check(cleanup==D::drained&&closed,"pending wake cleanup completed");
  check(result==D::drained,"pending submission must wake worker while deadline waiter exists");
  check(s.warms_count==1,"pending prediction ran once");++cases;
 }
 check(cases==7&&idle_seal_checks==1,"all independent scenarios and idle seal assertion executed");std::puts("PASS lookahead submitted drain: 7 scenarios");
}
