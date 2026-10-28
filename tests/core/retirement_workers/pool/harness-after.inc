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
