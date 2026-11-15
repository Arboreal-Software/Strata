#include "strata/core/diagnostic_readers.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <future>
#include <string>
#include <thread>
using strata::core::DiagnosticReaders;
using Clock=std::chrono::steady_clock;
static double ms_since(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Verifier {
 std::promise<void> entered,proceed;
 std::shared_future<void> released=proceed.get_future();
 std::atomic<bool> cleaned{false};
 int calls=0;
 void access(){assert(!cleaned);++calls;entered.set_value();released.wait();assert(!cleaned);}
 void diag(std::FILE*){access();}
 bool release_gpu_waits(int ms){assert(ms==5000);access();return true;}
};
#include "callbacks.inc"
int main(){
 for(int kind=0;kind<2;++kind){
  Verifier v;
  assert(g_verifier_readers.publish(&v));
  std::thread reader([&]{if(kind)release_live_verifiers(nullptr);else diag_active_verifier(nullptr);});
  v.entered.get_future().wait();
  std::thread retiring([&]{g_verifier_readers.retire(&v);v.cleaned=true;});
  while(g_verifier_readers.acquire(0))std::this_thread::yield();
  assert(!v.cleaned);
  assert(!g_verifier_readers.publish(&v));
  v.proceed.set_value();reader.join();retiring.join();assert(v.cleaned&&v.calls==1);
 }
 Verifier full[17];
 for(int i=0;i<16;++i)assert(g_verifier_readers.publish(&full[i]));
 assert(!g_verifier_readers.publish(&full[16]));
 {auto latest=g_verifier_readers.acquire_latest();assert(latest.get()==&full[15]);}
 for(int i=0;i<16;++i)g_verifier_readers.retire(&full[i]);
 assert(!g_verifier_readers.acquire_latest());
 std::puts("SYCL diagnostic callback lifetime tests passed");
}
