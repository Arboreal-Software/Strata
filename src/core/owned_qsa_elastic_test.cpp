// Prepared host-only admission/plan fixture, not compiled or executed.
// Link actual plan, managed batch, budget and owner implementations with no GPU.
#include "strata/core/owned_qsa_elastic.hpp"
#include <cassert>
using namespace strata::core;
int main(){
 ModelGeometry g;
 constexpr uint64_t granularity=65536;
 for(auto format:{QsaElasticFormat::f16,QsaElasticFormat::int8,QsaElasticFormat::q4,QsaElasticFormat::hybrid}){
  QsaElasticPlan plan;
  assert(qsa_elastic_plan(g,16384,format,false,64,0,granularity,plan));
  assert(plan.arrays==(format==QsaElasticFormat::int8?4u:format==QsaElasticFormat::hybrid?3u:2u));
  assert(plan.first_bytes>plan.other_bytes&&plan.virtual_bytes%granularity==0);
  for(size_t i=0;i<plan.arrays;++i)assert(plan.offsets[i]%granularity==0&&plan.per_slot[i]>0);
  auto bad=plan;++bad.offsets[0];assert(!qsa_elastic_plan_matches(g,bad));
  auto unchanged=plan;assert(!qsa_elastic_plan(g,16384,format,false,16385,0,granularity,unchanged));
  assert(qsa_elastic_plan_equal(unchanged,plan));
  assert(!qsa_elastic_plan(g,16384,format,false,64,0,0,unchanged));
  assert(qsa_elastic_plan_equal(unchanged,plan));
  uint64_t owner=0,range=0;assert(OwnedQsaElasticPool::host_bound(plan,owner,range));
  WindowBudget budget({{1ull<<38},1ull<<30,0,0,0,1},{{0},0,0,0,0,0});
  auto parent=budget.reserve({{1048576},4096,0,0,0,1},ReservationPurpose::private_work);assert(parent);
  const auto before=budget.snapshot().reserved;
  std::vector<std::shared_ptr<QsaElasticView>> pools;std::string error;
  assert(prepare_owned_qsa_elastic_pools(g,{plan},budget,parent.lease,pools,error));
  assert(pools.size()==1&&!pools[0]->healthy()&&!pools[0]->base()&&pools[0]->committed_slots()==0);
  assert(qsa_elastic_plan_equal(pools[0]->plan(),plan));
  assert(budget.snapshot().reserved.device_bytes[0]==before.device_bytes[0]+plan.initial_chunks*granularity);
  // Host preparation does not activate. Nonempty output remains untouched.
  auto* identity=pools[0].get();assert(!prepare_owned_qsa_elastic_pools(g,{plan},budget,parent.lease,pools,error)&&pools[0].get()==identity);
  pools.clear();assert(budget.snapshot().reserved==before);
  WindowBudget small({{1048576+plan.initial_chunks*granularity-1},1ull<<30,0,0,0,1},{{0},0,0,0,0,0});
  auto small_parent=small.reserve({{1048576},4096,0,0,0,1},ReservationPurpose::private_work);assert(small_parent);
  const auto small_before=small.snapshot().reserved;
  assert(!prepare_owned_qsa_elastic_pools(g,{plan},small,small_parent.lease,pools,error)&&pools.empty());
  assert(small.snapshot().reserved==small_before);
  assert(!prepare_owned_qsa_elastic_pools(g,{bad},budget,parent.lease,pools,error)&&pools.empty());
 }
}
