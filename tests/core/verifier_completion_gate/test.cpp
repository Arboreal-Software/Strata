// Actual gate and exact actual factory/hooks; Verifier fields are CPU seams.
#include "strata/core/verifier_completion_gate.hpp"
#include "strata/core/window_budget.hpp"
#include <optional>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <string>
#include <new>
#include <cstdlib>
static bool deny_alloc=false;
void* operator new(std::size_t n){if(deny_alloc)throw std::bad_alloc();if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void*p)noexcept{std::free(p);}void operator delete(void*p,std::size_t)noexcept{std::free(p);}
namespace strata::core {
class Verifier {
public:
 Verifier* next_=nullptr;bool strict_retirement_=true,retirement_init_started_=false;
 std::atomic<bool> released_{false},terminal_copy_query_failed_{false};
 std::shared_ptr<VerifierCompletionGate> completion_gate_;std::size_t completion_stage_=0;
 static bool prepare_serial_completion_gate(Verifier&,std::shared_ptr<const ReservationLease>,std::shared_ptr<VerifierCompletionGate>&,std::string&);
#include "hooks.inc"
};
#include "factory.inc"
}
using namespace strata::core;
static WindowBudget budget({{0},1u<<20},{{0}});
static std::shared_ptr<const ReservationLease> lease(){
 auto r=budget.reserve({{0},VerifierCompletionGate::metadata_bytes()},ReservationPurpose::foreground);
 assert(r&&r.lease);return std::move(r.lease);
}
static std::shared_ptr<VerifierCompletionGate> prepare(Verifier&a){
 std::shared_ptr<VerifierCompletionGate> g;std::string e;assert(Verifier::prepare_serial_completion_gate(a,lease(),g,e));return g;
}
int main(){
 {Verifier a,b;a.next_=&b;auto g=prepare(a);assert(a.completion_gate_==b.completion_gate_);
  VerifierCompletionGate::Token token,other;VerifierCompletionGate::Cut cut;
  deny_alloc=true;assert(g->try_begin(token));assert(!g->try_begin(other));
  assert(g->commit_after_checked_completion(token,cut));assert(cut.committed());
  assert(!g->commit_after_checked_completion(token,cut));deny_alloc=false;
  b.mark_completion_unhealthy();assert(cut.committed()&&!g->healthy_advisory());assert(!g->try_begin(other));}
 {Verifier a;auto g=prepare(a);VerifierCompletionGate::Token token;VerifierCompletionGate::Cut cut;assert(g->try_begin(token));
  std::mutex mu;std::condition_variable cv;bool marked=false;
  std::thread release([&]{a.mark_completion_unhealthy();{std::lock_guard lock(mu);marked=true;}cv.notify_all();});
  {std::unique_lock lock(mu);cv.wait(lock,[&]{return marked;});}
  assert(!g->commit_after_checked_completion(token,cut)&&token.present()&&!cut.committed());release.join();}
 {Verifier a;auto g=prepare(a);VerifierCompletionGate::Token token;VerifierCompletionGate::Cut cut;assert(g->try_begin(token));
  a.latch_copy_query_failure();assert(a.terminal_copy_query_failed_&&!g->commit_after_checked_completion(token,cut));}
 {Verifier a,b;auto g=prepare(a),foreign=prepare(b);VerifierCompletionGate::Token token;VerifierCompletionGate::Cut cut;
  assert(g->try_begin(token));assert(!foreign->commit_after_checked_completion(token,cut));assert(g->commit_after_checked_completion(token,cut));}
 {Verifier a,b;a.next_=&b;b.next_=&a;std::shared_ptr<VerifierCompletionGate> g;std::string e;
  assert(!Verifier::prepare_serial_completion_gate(a,lease(),g,e)&&!g&&!a.completion_gate_&&!b.completion_gate_);
  b.next_=nullptr;b.strict_retirement_=false;assert(!Verifier::prepare_serial_completion_gate(a,lease(),g,e));
  b.strict_retirement_=true;b.released_=true;assert(!Verifier::prepare_serial_completion_gate(a,lease(),g,e));}
 {Verifier a,b;a.next_=&b;b.retirement_init_started_=true;
  std::shared_ptr<VerifierCompletionGate> g;std::string e;
  assert(!Verifier::prepare_serial_completion_gate(a,lease(),g,e)&&!g&&!a.completion_gate_&&!b.completion_gate_);
  b.retirement_init_started_=false;a.retirement_init_started_=true;
  assert(!Verifier::prepare_serial_completion_gate(a,lease(),g,e)&&!g&&!a.completion_gate_&&!b.completion_gate_);}
 {Verifier a;auto g=prepare(a);{VerifierCompletionGate::Token abandoned;assert(g->try_begin(abandoned));}
  VerifierCompletionGate::Token fresh;assert(!g->try_begin(fresh));}
 {Verifier a;std::shared_ptr<VerifierCompletionGate> g;std::string e;auto admitted=lease();deny_alloc=true;
  assert(!Verifier::prepare_serial_completion_gate(a,std::move(admitted),g,e)&&!g&&!a.completion_gate_);deny_alloc=false;}
 {std::array<Verifier,9> stages;for(std::size_t i=0;i+1<stages.size();++i)stages[i].next_=&stages[i+1];
  std::shared_ptr<VerifierCompletionGate> g;std::string e;assert(!Verifier::prepare_serial_completion_gate(stages[0],lease(),g,e));
  for(auto&v:stages)assert(!v.completion_gate_);}
 {WindowBudget retained({{0},1u<<20},{{0}});std::optional<VerifierCompletionGate::Cut> cut;cut.emplace();
  const auto bytes=VerifierCompletionGate::metadata_bytes();
  {Verifier a;auto r=retained.reserve({{0},bytes},ReservationPurpose::foreground);
   std::weak_ptr<const ReservationLease> weak=r.lease;
   std::shared_ptr<VerifierCompletionGate> g;std::string e;
   assert(Verifier::prepare_serial_completion_gate(a,std::move(r.lease),g,e));
   VerifierCompletionGate::Token token;assert(g->try_begin(token));assert(g->commit_after_checked_completion(token,*cut));
   assert(!weak.expired());}
  assert(retained.snapshot().reserved.host_bytes==bytes);cut.reset();assert(retained.snapshot().reserved.host_bytes==0);}
 {Verifier a;std::shared_ptr<VerifierCompletionGate> g;std::string e;
  assert(!Verifier::prepare_serial_completion_gate(a,{},g,e)&&!a.completion_gate_);
  auto small=budget.reserve({{0},1},ReservationPurpose::foreground);
  assert(!Verifier::prepare_serial_completion_gate(a,std::move(small.lease),g,e));
  auto shared=lease();assert(!Verifier::prepare_serial_completion_gate(a,shared,g,e)&&!g&&!a.completion_gate_);}
 assert(budget.snapshot().reserved.host_bytes==0);

}
