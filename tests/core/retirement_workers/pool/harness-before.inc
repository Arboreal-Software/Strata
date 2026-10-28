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
