#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <exception>
namespace strata::kernels::cpu {
struct ExpertJob {}; struct ExpertJobMulti {}; struct NativeFmt {};
// Fixture substitutes numerical kernels and worker wake only. Actual guard/stop/wrappers below.
struct ExpertPool {
 enum class DispatchState { idle, active, sealed, joined, failed };
 std::atomic<DispatchState> dispatch_state_{DispatchState::idle};
 struct DispatchGuard { ExpertPool& pool; bool complete=false; explicit DispatchGuard(ExpertPool&); ~DispatchGuard(); };
 std::atomic<bool> stop_{false}; std::vector<std::thread> threads_;
 std::mutex mu; std::condition_variable cv; bool entered=false, release=false, block=false, fail=false;
 int calls=0; int retained_buffer=42;
 void publish(){std::lock_guard<std::mutex> l(mu);cv.notify_all();}
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
#include "actual-methods.inc"
}
#include "actual-wrappers.inc"
using namespace strata::kernels::cpu;
template<class F> void refuses(F f){bool caught=false;try{f();}catch(const std::logic_error&){caught=true;}assert(caught);}
int main(){
 ExpertPool p; p.block=true;
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
 ExpertPool failed;failed.fail=true;
 try{failed.run(nullptr,1);assert(false);}catch(const std::runtime_error&){}
 assert(failed.dispatch_state_.load()==ExpertPool::DispatchState::failed);
 assert(!failed.stop_after_dispatch_callers_joined());assert(!failed.stop_.load());
 refuses([&]{failed.run(nullptr,1);});
 ExpertPool empty;assert(empty.stop_after_dispatch_callers_joined());
}
