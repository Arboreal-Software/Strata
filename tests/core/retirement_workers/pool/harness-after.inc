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
