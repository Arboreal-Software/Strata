#include "joined_thread_scope.hpp"
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <sys/wait.h>
#include <unistd.h>
#define STRATA_WINDOW_STATE_AVAILABLE 1
namespace strata::platform {void name_this_thread(const char*){}}
namespace strata::aux_cpus {void pin_current_thread(){}}
struct Options{bool owned_session_lifetime=true;int adapt_iv(int)const{return 1;}};
struct Drive{struct D{std::vector<int> usage{1};}d;};
using Clock=std::chrono::steady_clock;
struct State{std::mutex mu;std::condition_variable cv;bool entered=false,exit=false,allow=false;std::atomic<bool> finished{false},alive{true};};
static int run(bool serving,int mode){
 State state;Options o;o.owned_session_lifetime=mode!=4;Drive drive;int rounds=0;void* ajob=nullptr;
 std::function<void()> adapt_fence;if(mode==5)adapt_fence=[]{};
 if(mode==6)drive.d.usage.clear();
 auto adapt=[&]{
   {std::unique_lock lock(state.mu);state.entered=true;state.cv.notify_all();state.cv.wait(lock,[&]{return state.allow;});}
   assert(state.alive.load());state.finished=true;
   if(mode==2)throw std::runtime_error("adapt");
   return mode!=3;
 };
 std::thread releaser;
 if(mode!=5&&mode!=6)releaser=std::thread([&]{std::unique_lock lock(state.mu);state.cv.wait(lock,[&]{return state.exit&&state.entered;});state.allow=true;state.cv.notify_all();});
 struct Reap{std::thread& t;~Reap(){if(t.joinable())t.join();}} reap{releaser};
 struct Target{State& s;bool idle;~Target(){if(!idle)assert(s.finished.load());s.alive=false;}}target{state,mode==5||mode==6};
 auto exiting=[&]{std::lock_guard lock(state.mu);state.exit=true;state.cv.notify_all();};
 try {
  if(serving){
#include "serve.inc"
   exiting();
   if(mode==1)throw std::runtime_error("caller");
   if(adapt_thr.joinable())adapt_thr.join();
   assert(adapt_ok==(mode!=2&&mode!=3));
  }else{
#include "nonserve.inc"
   exiting();
   if(mode==1)throw std::runtime_error("caller");
   // Exact actual EOS branch: failure must return1 before the success break.
   const bool eos=true;double total_ms=0;const auto t0=Clock::now();
   for(;;){
#include "eos.inc"
   }
   assert(adapt_ok||!o.owned_session_lifetime);
  }
 }catch(const std::runtime_error&){assert(mode==1);}
 return 0;
}
int main(){
 for(bool serving:{false,true})for(int mode=0;mode<7;++mode){
  if(mode==5&&!serving)continue;
  const auto pid=fork();assert(pid>=0);
  if(!pid)std::_Exit(run(serving,mode));
  int status=0;assert(waitpid(pid,&status,0)==pid&&WIFEXITED(status));
  const int expected=(mode==5||(!serving&&(mode==2||mode==3)))?1:0;
  assert(WEXITSTATUS(status)==expected);
 }
 // Disabled guard must not join; explicit caller closure remains required.
 std::mutex mu;std::condition_variable cv;bool release=false,done=false;
 std::thread worker([&]{std::unique_lock lock(mu);cv.wait(lock,[&]{return release;});done=true;});
 {strata::core::JoinedThreadScope guard(false,worker);}
 {std::lock_guard lock(mu);assert(worker.joinable()&&!done);release=true;}cv.notify_all();worker.join();
}
