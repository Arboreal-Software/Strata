#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <exception>
namespace strata::kernels::cpu {
struct ExpertJob {}; struct ExpertJobMulti {}; struct NativeFmt {};
// Fixture substitutes numerical kernels and worker wake only. Actual guard/stop/wrappers below.
struct ExpertPool {
 const bool checked_retirement_;
 explicit ExpertPool(bool checked=false):checked_retirement_(checked){}
 ~ExpertPool();
 enum class DispatchState { idle, active, sealed, joined, failed };
 std::atomic<DispatchState> dispatch_state_{DispatchState::idle};
 struct DispatchGuard { ExpertPool& pool; bool complete=false; explicit DispatchGuard(ExpertPool&); ~DispatchGuard(); };
 std::atomic<bool> stop_{false}; std::vector<std::thread> threads_;
 std::mutex mu; std::condition_variable cv; bool entered=false, release=false, block=false, fail=false;
 int calls=0; int retained_buffer=42; int publishes=0;
 void publish(){std::lock_guard<std::mutex> l(mu);++publishes;cv.notify_all();}
 bool stop_after_dispatch_callers_joined() noexcept;
 void run(ExpertJob*,int);void run_split(ExpertJob*,int);
 void run_split_multi(ExpertJobMulti*,int);void run_split_multi_native(const NativeFmt&,ExpertJobMulti*,int);
 void run_impl(ExpertJob*,int){
   ++calls; if(fail)throw std::runtime_error("fixture kernel failure");
   if(block){std::unique_lock<std::mutex> l(mu);entered=true;cv.notify_all();cv.wait(l,[&]{return release;});}
 }
 void run_split_impl(ExpertJob*j,int n){run_impl(j,n);}
 void run_split_multi_impl(ExpertJobMulti*,int n){run_split_impl(nullptr,n);}
 void run_split_multi_native_impl(const NativeFmt&,ExpertJobMulti*,int n){run_impl(nullptr,n);}
};
struct Readers { int retired=0; void retire(ExpertPool*){++retired;} } g_pool_readers;
struct NumaSplitPlan {};
NumaSplitPlan g_numa_plan;
ExpertPool* g_numa_owner=nullptr;
#include "actual-destructor.inc"
#include "actual-methods.inc"
}
#include "actual-wrappers.inc"
using namespace strata::kernels::cpu;
template<class F> void refuses(F f){bool caught=false;try{f();}catch(const std::logic_error&){caught=true;}assert(caught);}
int main(){
 ExpertPool p(true); p.block=true;
 // Real worker remains available until accepted stop; no timing-based sleeps.
 p.threads_.emplace_back([&]{std::unique_lock<std::mutex> l(p.mu);p.cv.wait(l,[&]{return p.stop_.load();});});
 std::thread dispatch([&]{p.run(nullptr,1);});
 {std::unique_lock<std::mutex> l(p.mu);p.cv.wait(l,[&]{return p.entered;});}
 assert(!p.stop_after_dispatch_callers_joined());assert(!p.stop_.load());
 refuses([&]{p.run(nullptr,1);});
 assert(p.dispatch_state_.load()==ExpertPool::DispatchState::active);
 {std::lock_guard<std::mutex> l(p.mu);p.release=true;}p.cv.notify_all();dispatch.join();
 p.block=false;
 p.run_split(nullptr,1);p.run_split_multi(nullptr,1);p.run_split_multi_native(NativeFmt{},nullptr,1);
 assert(p.calls==4);assert(p.stop_after_dispatch_callers_joined());
 assert(!p.threads_[0].joinable());assert(p.retained_buffer==42);
 assert(p.stop_after_dispatch_callers_joined());
 refuses([&]{p.run(nullptr,0);});refuses([&]{p.run_split(nullptr,1);});
 refuses([&]{p.run_split_multi(nullptr,1);});refuses([&]{p.run_split_multi_native(NativeFmt{},nullptr,1);});
 ExpertPool failed(true);failed.fail=true;
 try{failed.run(nullptr,1);assert(false);}catch(const std::runtime_error&){}
 assert(failed.dispatch_state_.load()==ExpertPool::DispatchState::failed);
 assert(!failed.stop_after_dispatch_callers_joined());assert(!failed.stop_.load());
 refuses([&]{failed.run(nullptr,1);});
 // Fixture-only reset after observing sticky failure: no production borrower exists.
 failed.dispatch_state_.store(ExpertPool::DispatchState::joined);
 ExpertPool empty(true);assert(empty.stop_after_dispatch_callers_joined());
 {
  ExpertPool legacy; // Default constructor path must remain untracked.
  legacy.dispatch_state_.store(ExpertPool::DispatchState::failed);
  assert(!legacy.stop_after_dispatch_callers_joined());
  assert(!legacy.stop_.load()&&legacy.publishes==0);
  legacy.run(nullptr,0);legacy.run_split(nullptr,1);
  legacy.run_split_multi(nullptr,1);legacy.run_split_multi_native(NativeFmt{},nullptr,1);
  assert(legacy.calls==4);
  assert(legacy.dispatch_state_.load()==ExpertPool::DispatchState::failed);
  legacy.fail=true;bool original=false;
  try{legacy.run(nullptr,1);}catch(const std::runtime_error& e){original=std::string(e.what())=="fixture kernel failure";}
  assert(original);legacy.fail=false;legacy.run(nullptr,1);assert(legacy.calls==6);
  assert(!legacy.stop_after_dispatch_callers_joined());assert(legacy.publishes==0);
 }
 // Actual legacy destructor must signal/join a real waiting worker despite untracked state.
 std::atomic<bool> legacy_worker_done{false};int retired=g_pool_readers.retired;
 {
  ExpertPool legacy;
  legacy.threads_.emplace_back([&]{std::unique_lock<std::mutex> lock(legacy.mu);legacy.cv.wait(lock,[&]{return legacy.stop_.load();});legacy_worker_done=true;});
 }
 assert(legacy_worker_done&&g_pool_readers.retired==retired+1);
}
