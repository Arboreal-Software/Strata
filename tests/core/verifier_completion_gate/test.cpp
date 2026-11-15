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
static long fail_after=-1;
static void* watched_gate=nullptr;
static bool gate_deallocated=false,watch_next_allocation=false;
static void* watched_controlblock=nullptr;
static bool controlblock_deallocated=false;
static unsigned tracked_creation_step=0;
// Enabled ONLY for the successful weak-last-owner case, never fault injection.
static strata::core::WindowBudget* watched_free_budget=nullptr;
static uint64_t expected_reserved_at_free=0;
static std::size_t allocations=0;
void* operator new(std::size_t n){++allocations;if(deny_alloc)throw std::bad_alloc();if(fail_after==0){fail_after=-1;throw std::bad_alloc();}if(fail_after>0)--fail_after;
 if(auto p=std::malloc(n?n:1)){if(tracked_creation_step==1){watched_gate=p;gate_deallocated=false;tracked_creation_step=2;}else if(tracked_creation_step==2){watched_controlblock=p;controlblock_deallocated=false;tracked_creation_step=0;}if(watch_next_allocation){watched_gate=p;gate_deallocated=false;watch_next_allocation=false;}return p;}throw std::bad_alloc();}
static void tracked_free(void* p)noexcept{
 const bool gate=p&&p==watched_gate,block=p&&p==watched_controlblock;
 if(block&&watched_free_budget){
  // snapshot copies vectors: permit this diagnostic ONLY outside all injected
  // allocation faults. Its nested allocations cannot alias this still-live block.
  assert(!deny_alloc&&fail_after<0&&!controlblock_deallocated);
  auto* ledger=watched_free_budget;watched_free_budget=nullptr; // one-shot; no nested hook
  try {assert(ledger->snapshot().reserved.host_bytes==expected_reserved_at_free);}
  catch(...){std::abort();} // no exception may escape replacement delete
 }
 std::free(p);if(gate)gate_deallocated=true;if(block)controlblock_deallocated=true;
}
void operator delete(void*p)noexcept{tracked_free(p);}void operator delete(void*p,std::size_t)noexcept{tracked_free(p);}
namespace strata::core {
class Verifier {
public:
 Verifier* next_=nullptr;bool strict_retirement_=true,retirement_init_started_=false;
 std::atomic<bool> released_{false},terminal_copy_query_failed_{false};
 std::shared_ptr<VerifierCompletionGate> completion_gate_;std::size_t completion_stage_=0;
 static bool prepare_serial_completion_gate(Verifier&,std::shared_ptr<const ReservationLease>,std::shared_ptr<VerifierCompletionGate>&,std::string&,std::shared_ptr<const void> = {});
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
 {Verifier a,b;a.next_=&b;std::shared_ptr<VerifierCompletionGate> gate;std::string error;error.reserve(256);
  auto canonical=lease();const auto bytes=canonical->resources().host_bytes;
  auto first_holder=std::make_shared<std::shared_ptr<const ReservationLease>>(canonical);
  auto second_holder=std::make_shared<std::shared_ptr<const ReservationLease>>(canonical);
  std::shared_ptr<const ReservationLease> first(first_holder,first_holder->get()),second(second_holder,second_holder->get());
  first_holder.reset();second_holder.reset();canonical.reset();
  assert(first.use_count()==1&&second.use_count()==1);
  assert(!first->canonical_owner(first)&&!second->canonical_owner(second));
  const auto before=allocations;
  assert(!Verifier::prepare_serial_completion_gate(a,std::move(first),gate,error));
  assert(!gate&&!a.completion_gate_&&!b.completion_gate_&&allocations==before);
  assert(budget.snapshot().reserved.host_bytes==bytes);
  // snapshot() copies resource vectors and can allocate. Measure each actual
  // admission call independently of this test's intervening observation.
  const auto before_second=allocations;
  assert(!Verifier::prepare_serial_completion_gate(a,std::move(second),gate,error));
  assert(!gate&&!a.completion_gate_&&!b.completion_gate_&&allocations==before_second);
  assert(budget.snapshot().reserved.host_bytes==0);
 }
 assert(budget.snapshot().reserved.host_bytes==0);

 // A real retained credit probe: destruction observes actual metadata charge
 // already released. No claimed admission/completion boolean is consulted.
 struct CreditProbe {
  WindowBudget* ledger;int* destroyed;bool check_release_order=false,check_gate_free=false,check_controlblock_free=false;
  ~CreditProbe(){if(check_controlblock_free)assert(controlblock_deallocated);if(check_gate_free)assert(gate_deallocated);if(check_release_order)assert(ledger->snapshot().reserved.host_bytes==0);++*destroyed;}
 };
 // A live Token alone retains BOTH genuine metadata and the extra owner.
 {WindowBudget retained({{0},1u<<20},{{0}});int destroyed=0;
  std::optional<VerifierCompletionGate::Token> token;token.emplace();
  std::weak_ptr<CreditProbe> weak;
  {Verifier a;auto r=retained.reserve({{0},VerifierCompletionGate::metadata_bytes()},ReservationPurpose::foreground);
   auto credit=std::make_shared<CreditProbe>();credit->ledger=&retained;credit->destroyed=&destroyed;weak=credit;
   std::shared_ptr<VerifierCompletionGate> g;std::string error;
   assert(Verifier::prepare_serial_completion_gate(a,std::move(r.lease),g,error,std::move(credit)));
   auto observed=weak.lock();assert(observed);observed->check_release_order=true;observed.reset();assert(g->try_begin(*token));}
  assert(!weak.expired()&&destroyed==0);token.reset();assert(weak.expired()&&destroyed==1);}
 // A committed Cut has the same retention after all Verifier/gate handles die.
 {WindowBudget retained({{0},1u<<20},{{0}});int destroyed=0;
  std::optional<VerifierCompletionGate::Cut> cut;cut.emplace();std::weak_ptr<CreditProbe> weak;
  {Verifier a;auto r=retained.reserve({{0},VerifierCompletionGate::metadata_bytes()},ReservationPurpose::foreground);
   auto credit=std::make_shared<CreditProbe>();credit->ledger=&retained;credit->destroyed=&destroyed;weak=credit;
   std::shared_ptr<VerifierCompletionGate> g;std::string error;
   assert(Verifier::prepare_serial_completion_gate(a,std::move(r.lease),g,error,std::move(credit)));
   auto observed=weak.lock();assert(observed);observed->check_release_order=true;observed.reset();VerifierCompletionGate::Token token;assert(g->try_begin(token));assert(g->commit_after_checked_completion(token,*cut));}
  assert(!weak.expired()&&destroyed==0);cut.reset();assert(weak.expired()&&destroyed==1);}
 // Reversible refusal and allocator failure release both transferred owners;
 // neither case publishes a partial stage gate. Existing canonical/purpose
 // rejection tests above remain unchanged.
 for(bool fail_allocation:{false,true}) {
  WindowBudget retained({{0},1u<<20},{{0}});int destroyed=0;Verifier a;
  a.retirement_init_started_=!fail_allocation;
  auto r=retained.reserve({{0},VerifierCompletionGate::metadata_bytes()},ReservationPurpose::foreground);
  auto credit=std::make_shared<CreditProbe>();credit->ledger=&retained;credit->destroyed=&destroyed;
  std::weak_ptr<CreditProbe> weak=credit;std::shared_ptr<VerifierCompletionGate> g;std::string error;
  deny_alloc=fail_allocation;
  assert(!Verifier::prepare_serial_completion_gate(a,std::move(r.lease),g,error,std::move(credit)));
  deny_alloc=false;
  assert(!g&&!a.completion_gate_&&weak.expired()&&destroyed==1);
  assert(retained.snapshot().reserved.host_bytes==0);
 }

 // Actual object deallocation must precede both quota releases. Weak Gate
 // owners retain its control block/deleter even though lock() can no longer work.
 {WindowBudget retained({{0},1u<<20},{{0}});int destroyed=0;
  std::weak_ptr<VerifierCompletionGate> weak_gate;std::weak_ptr<CreditProbe> weak_credit;
  const auto bytes=VerifierCompletionGate::metadata_bytes();
  {Verifier a;auto r=retained.reserve({{0},bytes},ReservationPurpose::foreground);
   auto credit=std::make_shared<CreditProbe>();credit->ledger=&retained;credit->destroyed=&destroyed;
   credit->check_release_order=true;credit->check_gate_free=true;credit->check_controlblock_free=true;weak_credit=credit;
   std::shared_ptr<VerifierCompletionGate> g;std::string error;
   tracked_creation_step=1;
   assert(Verifier::prepare_serial_completion_gate(a,std::move(r.lease),g,error,std::move(credit)));
   assert(watched_gate==g.get()&&watched_controlblock&&tracked_creation_step==0);weak_gate=g;
  }
  assert(gate_deallocated&&!controlblock_deallocated&&weak_gate.expired()&&!weak_credit.expired()&&destroyed==0);
  assert(retained.snapshot().reserved.host_bytes==bytes);
  watched_free_budget=&retained;expected_reserved_at_free=bytes;
  weak_gate.reset();
  watched_free_budget=nullptr;expected_reserved_at_free=0;
  assert(controlblock_deallocated&&weak_credit.expired()&&destroyed==1&&retained.snapshot().reserved.host_bytes==0);
  watched_gate=nullptr;watched_controlblock=nullptr;
 }
 // Gate new succeeds, shared_ptr control-block new fails: shared_ptr must
 // delete actual Gate before releasing the retained deleter admissions.
 {WindowBudget retained({{0},1u<<20},{{0}});int destroyed=0;Verifier a;
  auto r=retained.reserve({{0},VerifierCompletionGate::metadata_bytes()},ReservationPurpose::foreground);
  auto credit=std::make_shared<CreditProbe>();credit->ledger=&retained;credit->destroyed=&destroyed;
  credit->check_gate_free=true;credit->check_release_order=true;
  std::shared_ptr<VerifierCompletionGate> g;std::string error;
  watch_next_allocation=true;fail_after=1;
  assert(!Verifier::prepare_serial_completion_gate(a,std::move(r.lease),g,error,std::move(credit)));
  fail_after=-1;watch_next_allocation=false;
  assert(gate_deallocated&&!g&&!a.completion_gate_&&destroyed==1&&retained.snapshot().reserved.host_bytes==0);
  watched_gate=nullptr;
 }

}
