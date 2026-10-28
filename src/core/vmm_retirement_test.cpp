// Prepared, not executed. Link actual vmm.cpp, vmm_managed.cpp, native_retirement.cpp, window_budget.cpp
// and vmm_retirement_test_driver.cpp using CUDA SDK declarations, no CUDA library.
// Backend calls are injected; no real driver/runtime/closure qualification.
#include "strata/core/vmm.hpp"
#include <array>
#include <cassert>
#include <cstdlib>
#include <string>
using namespace strata::core;
extern unsigned fake_queries(),fake_syncs(),fake_releases(),fake_unmaps(),fake_va_frees(),fake_sets();
extern void fake_fail_get(),fake_fail_set_at(int);
extern void fake_fail_sync(int),fake_fail_release_at(int),fake_fail_unmap_at(int),fake_fail_va_free();
struct Owner {
    VmmRange range;
    std::array<DetachedVmmChunk,3> physical;
    std::array<VmmChunkOutcome,3> result;
};
int main(int argc,char** argv){
 const std::string scenario=argc>1?argv[1]:"success";
 const int at=argc>2?std::atoi(argv[2]):1;
 WindowResources zero{{0,0},0,0,0,0,0},cap{{1<<20,1<<20},1<<20,0,0,0,1};
 WindowBudget budget(cap,zero);
 auto owner=std::make_shared<Owner>();std::weak_ptr<Owner> weak=owner;
 uint64_t host=0;assert(VmmRange::managed_host_bound(3,host));
 auto metadata=zero;metadata.host_bytes=host;
 auto lease=budget.reserve(metadata,ReservationPurpose::foreground);assert(lease);
 std::string error;
 assert(owner->range.prepare_managed_metadata(3,4096,std::move(lease.lease),0,error));
 const bool inactive=scenario=="inactive";
 if(!inactive){
  assert(owner->range.activate_managed());
  for(auto& c:owner->physical)assert(DetachedVmmChunk::allocate(budget,0,c,error));
  // Two mapped plus one real detached chunk exercises both commit consumers.
  assert(owner->range.map_managed(0,2,std::span(owner->physical).first(2),std::span(owner->result).first(2)));
 }
 NativeRetirementContext uncovered;const std::array<int,1> other_device{1};
 assert(uncovered.prepare(owner,other_device));const auto before_refusal=fake_queries();
 assert(!owner->range.prepare_retirement(uncovered));
 if(!inactive)assert(!owner->physical[2].prepare_retirement(uncovered));
 assert(fake_queries()==before_refusal);
 NativeRetirementContext context;
 const std::array<int,2> devices{0,1};
 assert(context.prepare(owner,devices));
 NativeRetirementContext other;assert(other.prepare(owner,devices));
 const auto queries=fake_queries(),before=fake_syncs();
 assert(owner->range.prepare_retirement(context));
 for(auto& c:owner->physical)assert(c.prepare_retirement(context));
 assert(fake_queries()==queries&&fake_syncs()==before); // preparation is host-only
 assert(!owner->range.managed_healthy()&&!owner->range.base()&&!owner->range.mapped(0));
 assert(!owner->range.activate_managed()&&!owner->range.reserve(4096));
 assert(!owner->range.map_managed(0,0,{},{}));
 assert(owner->range.commit_retirement(context,owner->result).code==VmmRetirementResult::refused);
 assert(owner->range.commit_retirement(other,owner->result).code==VmmRetirementResult::refused);
 if(scenario=="abandon")return 0; // expected subprocess exit1: sealed resources cannot silently destruct
 if(scenario=="drain")fake_fail_sync(int(before)+at);
 if(scenario=="drain-get")fake_fail_get();
 if(scenario=="drain-set")fake_fail_set_at(int(fake_sets())+at);
 const auto held=budget.snapshot().reserved;
 const bool drained=context.drain_after_producers_closed();
 if(scenario=="drain"||scenario=="drain-get"||scenario=="drain-set"){
  assert(!drained&&!context.drained());const auto calls=fake_syncs();
  assert(!context.drain_after_producers_closed()&&fake_syncs()==calls);
  assert(owner->range.commit_retirement(context,owner->result).code==VmmRetirementResult::refused);
  owner.reset();assert(!weak.expired()&&budget.snapshot().reserved==held&&fake_releases()==0&&fake_unmaps()==0);
  std::_Exit(0); // context intentionally retains unresolved real Owner until worker exit
 }
 assert(drained&&fake_syncs()==before+2);assert(context.drain_after_producers_closed()&&fake_syncs()==before+2);
 if(scenario=="unmap")fake_fail_unmap_at(at);
 if(scenario=="release")fake_fail_release_at(at);
 if(scenario=="address")fake_fail_va_free();
 assert(other.drain_after_producers_closed());
 assert(owner->range.commit_retirement(other,owner->result).code==VmmRetirementResult::refused);
 const auto barrier=fake_syncs();
 const auto result=owner->range.commit_retirement(context,owner->result);
 if(scenario=="unmap"||scenario=="release"||scenario=="address"){
  assert(result.code==VmmRetirementResult::unknown&&fake_syncs()==barrier);
  const auto frees=fake_releases(),unmaps=fake_unmaps(),va=fake_va_frees();
  assert(owner->range.commit_retirement(context,owner->result).code==VmmRetirementResult::unknown);
  assert(fake_releases()==frees&&fake_unmaps()==unmaps&&fake_va_frees()==va);
  assert(!owner->range.mapped(0)&&!owner->range.base());
  if(scenario!="address")assert(result.retired==size_t(at-1));
  for(size_t i=0;i<result.retired;++i)assert(owner->result[i].disposition==VmmDisposition::retired);
  owner.reset();assert(!weak.expired()&&budget.snapshot().reserved.host_bytes>0);
  std::_Exit(0); // raw backend uncertainty not silently destroyed or retried
 }
 assert(result.code==VmmRetirementResult::complete&&result.retired==(inactive?0u:2u));
 for(auto& c:owner->physical){VmmChunkOutcome out;assert(c.commit_retirement(context,out).code==VmmRetirementResult::complete);}
 assert(owner->range.commit_retirement(context,{}).code==VmmRetirementResult::complete);
 assert(fake_syncs()==barrier);owner.reset();assert(!weak.expired());
 // Context retains the actual owner; its already committed destructor has no wait.
 assert(budget.snapshot().reserved==zero);
}
