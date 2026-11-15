// Prepared actual pool/VMM/budget retirement test. CUDA driver is a CPU fake;
// geometry-plan validation is an explicit tiny seam, not numerical qualification.
#include "strata/core/owned_qsa_elastic.hpp"
#include "strata/core/native_retirement.hpp"
#include <cassert>
#include <cstdlib>
#include <new>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
extern unsigned fake_drains(),fake_releases(),fake_queries();
extern void fake_fail_create_after(int),fake_fail_next_zero(),fake_fail_next_drain(),fake_fail_next_release(),fake_fail_next_access();
static bool no_alloc=false;
void* operator new(size_t n){if(no_alloc)throw std::bad_alloc();void* p=std::malloc(n?n:1);if(!p)throw std::bad_alloc();return p;}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete(void* p,size_t)noexcept{std::free(p);}
static QsaElasticPlan tiny_plan(){
 QsaElasticPlan p;p.device=0;p.cells=p.slots=2;p.initial_cells=p.page_size=1;
 p.granularity=4096;p.virtual_bytes=8192;p.initial_chunks=2;p.arrays=2;
 p.offsets={0,4096,0,0};p.per_slot={32,32,0,0};return p;
}
namespace strata::core {
bool qsa_elastic_plan_matches(const ModelGeometry&,const QsaElasticPlan& p){return qsa_elastic_plan_equal(p,tiny_plan());}
}
static void run(int scenario){
 const auto plan=tiny_plan();ModelGeometry g;std::string error;
 WindowResources cap;cap.device_bytes={1<<20};cap.host_bytes=1<<20;cap.shadows=1;
 WindowResources zero;zero.device_bytes={0};WindowBudget budget(cap,zero);
 auto parent_charge=zero;parent_charge.device_bytes[0]=128;parent_charge.shadows=1;
 auto parent=budget.reserve(parent_charge,ReservationPurpose::private_work);assert(parent);
 std::vector<std::shared_ptr<QsaElasticView>> pools;
 assert(prepare_owned_qsa_elastic_pools(g,{plan},budget,parent.lease,pools,error));
 auto pool=pools[0];pools.clear();parent.lease.reset();
 if(scenario==1)fake_fail_create_after(1);
 if(scenario==2)fake_fail_next_zero();
 if(scenario==6)fake_fail_next_access();
 if(scenario!=3){const bool ready=pool->initialize_checked(error);assert(ready==(scenario!=1&&scenario!=2&&scenario!=6));}
 struct Gang {QsaState bound;std::shared_ptr<QsaElasticView> pool;};
 auto gang=std::make_shared<Gang>();gang->pool=pool;
 // Models caller terminal-owner quarantine, NOT a claim that pool raw state
 // pointers alone retain a session. Main allocator supplies the actual deleter.
 static std::shared_ptr<Gang> retained_failed_gang;
 if(pool->healthy())assert(pool->claim_state(&gang->bound));
 const int device=0;
 NativeRetirementContext context,foreign;
 assert(context.prepare(gang,std::span(&device,1)));
 assert(foreign.prepare(gang,std::span(&device,1)));
 const auto queries=fake_queries(),drains=fake_drains(),releases=fake_releases();
 no_alloc=true;const bool prepared=pool->prepare_retirement(context);no_alloc=false;
 assert(fake_queries()==queries&&fake_drains()==drains&&fake_releases()==releases);
 if(scenario==6){assert(!prepared&&!pool->healthy()&&!pool->base());assert(!pool->commit_retirement(context));retained_failed_gang=gang;return;}
 assert(prepared);
 assert(!pool->healthy()&&!pool->base()&&pool->committed_slots()==0&&!pool->claim_state(&gang->bound));
 assert(!pool->prepare_retirement(context));assert(!pool->commit_retirement(context));
 assert(!pool->commit_retirement(foreign));assert(!pool->initialize_checked(error));
 if(scenario==4)fake_fail_next_drain();
 const bool drained=context.drain_after_producers_closed();
 if(scenario==4){
  assert(!drained&&!pool->commit_retirement(context)&&fake_releases()==releases);
  auto weak=std::weak_ptr<QsaElasticView>(pool);pool.reset();
  assert(!weak.expired()); // genuine context owner still pins pool and all resources
  retained_failed_gang=gang;
  return; // unresolved sealed destructor quarantines storage after contexts drop
 }
 assert(drained);assert(foreign.drain_after_producers_closed());assert(!pool->commit_retirement(foreign));const auto after_drain=fake_drains();
 if(scenario==5)fake_fail_next_release();
 no_alloc=true;const bool retired=pool->commit_retirement(context);no_alloc=false;
 assert(fake_drains()==after_drain);
 assert(retired==(scenario!=5));assert(!pool->commit_retirement(context));
 assert(!pool->healthy()&&!pool->base());
 if(retired)assert(budget.snapshot().reserved.device_bytes[0]==0&&budget.snapshot().reserved.shadows==0);
 else {assert(budget.snapshot().reserved.device_bytes[0]>0);retained_failed_gang=gang;} // exact residual owners; no rollback claim
}
int main(){for(int scenario=0;scenario<7;++scenario){
 const auto child=fork();assert(child>=0);if(!child){run(scenario);std::_Exit(0);}
 int status=0;assert(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0);
}}
