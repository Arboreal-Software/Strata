// Prepared extracted HOST admission fixture. No CUDA or production plan sizing.
#include "strata/core/vmm.hpp"
#include "strata/core/qsa_elastic_view.hpp"
#include "strata/core/geometry.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <utility>
static size_t allocations=0;
void* operator new(size_t n){++allocations;if(auto* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}void operator delete(void* p,size_t)noexcept{std::free(p);}
static void check(bool result,const char* what){if(!result){std::fprintf(stderr,"FAIL %s\n",what);std::exit(70);}}
namespace strata::core {
[[noreturn]] static void retirement_failed(const char*){std::abort();}
// Explicit backend/geometry seams; none reimplements the ownership decision.
// Device branches abort if accidentally reached; this fixture never activates.
static constexpr int cudaSuccess=0;
static int cudaDeviceSynchronize(){std::abort();}
struct WindowDeviceScope {
 explicit WindowDeviceScope(int){std::abort();}bool ok(){std::abort();}bool finish(){std::abort();}
};
static const QsaElasticPlan* bound_plan=nullptr;
bool qsa_elastic_plan_matches(const ModelGeometry&,const QsaElasticPlan& p){return &p==bound_plan;}
#include "owned-class.inc"
#if defined(CANONICAL_OLD_RANGE)
#include "range-methods-no-canonical.inc"
#else
#include "range-methods.inc"
#endif
#if defined(CANONICAL_OLD_BATCH)
#include "batch-methods-no-canonical.inc"
#else
#include "batch-methods.inc"
#endif
#if defined(CANONICAL_OLD_POOL)
#include "owned-methods-no-canonical.inc"
#else
#include "owned-methods.inc"
#endif
#include "host-release.inc"
#include "chunk-methods.inc"
#include "owned-cleanup.inc"
// Unreachable virtual/device operations, deliberately no permissive stand-ins.
bool DetachedVmmChunk::retire()noexcept{std::abort();}
bool OwnedQsaElasticPool::initialize_checked(std::string&){std::abort();}
bool OwnedQsaElasticPool::claim_state(QsaState*)noexcept{std::abort();}
bool OwnedQsaElasticPool::prepare_retirement(const NativeRetirementContext&)noexcept{std::abort();}
bool OwnedQsaElasticPool::commit_retirement(const NativeRetirementContext&)noexcept{std::abort();}
const QsaElasticPlan& OwnedQsaElasticPool::plan()const noexcept{return storage_->plan;}
uint8_t* OwnedQsaElasticPool::base()const noexcept{return storage_->range.base();}
int64_t OwnedQsaElasticPool::committed_slots()const noexcept{return 0;}
bool OwnedQsaElasticPool::healthy()const noexcept{return false;}
}
using namespace strata::core;
static WindowResources host(uint64_t n){WindowResources r;r.device_bytes={0};r.host_bytes=n;return r;}
static WindowResources physical(uint64_t n){WindowResources r;r.device_bytes={n};return r;}
static std::shared_ptr<const ReservationLease> mint(WindowBudget& budget,WindowResources charge){
 auto result=budget.reserve(charge,ReservationPurpose::foreground);check(bool(result),"real ledger admission");return std::move(result.lease);
}
static std::shared_ptr<const ReservationLease> rebox(std::shared_ptr<const ReservationLease> genuine){
 auto holder=std::make_shared<std::shared_ptr<const ReservationLease>>(std::move(genuine));
 std::shared_ptr<const ReservationLease> result(holder,holder->get());holder.reset();
 check(result.use_count()==1&&!result->canonical_owner(result),"lifetime-preserving independent owner group");return result;
}
int main(){
 auto capacity=host(8<<20);capacity.device_bytes={8<<20};WindowBudget budget(capacity,host(0));
 constexpr uint64_t gran=65536;std::string error;error.reserve(256);
 uint64_t range_bytes=0,batch_bytes=0;
 check(VmmRange::managed_host_bound(2,range_bytes),"actual range quote");
 check(PreparedVmmAllocationBatch::host_bound(1,batch_bytes),"actual batch quote");
 {
  VmmRange range;auto boxed=rebox(mint(budget,host(range_bytes)));const auto before=allocations;
  check(!range.prepare_managed_metadata(2,gran,std::move(boxed),0,error),"range rejects reboxed metadata");
  check(allocations==before&&range.chunks()==0&&!range.base(),"range refusal before arrays/publication");
  check(budget.snapshot().reserved.host_bytes==0,"range refusal releases moved box");
  auto genuine=mint(budget,host(range_bytes));
  check(range.prepare_managed_metadata(2,gran,std::move(genuine),0,error),"actual canonical range positive control");
  check(range.chunks()==2&&!range.base(),"host-only range retains arrays without activation");
 }
 check(budget.snapshot().reserved.host_bytes==0,"range destructor returns actual charge");
 const VmmBatchDevicePlan batch_plan{0,0,1,gran};
 for(int mode=0;mode<3;++mode){
  auto device=mint(budget,physical(gran)),metadata=mint(budget,host(batch_bytes));
  if(mode==0)device=rebox(std::move(device));if(mode==1)metadata=rebox(std::move(metadata));
  std::unique_ptr<PreparedVmmAllocationBatch> batch;const auto before=allocations;
  const bool accepted=PreparedVmmAllocationBatch::prepare({&batch_plan,1},std::move(device),std::move(metadata),batch,error);
  check(accepted==(mode==2),"batch physical/metadata rebox refusal and canonical positive");
  if(mode!=2)check(!batch&&allocations==before,"batch refuses before allocation/incarnation publication");
  batch.reset();check(budget.snapshot().reserved.host_bytes==0&&budget.snapshot().reserved.device_bytes[0]==0,"batch charge drain");
 }
 ModelGeometry geometry;QsaElasticPlan plan;plan.device=0;plan.granularity=gran;plan.virtual_bytes=2*gran;plan.initial_chunks=1;
 bound_plan=&plan;uint64_t owner_bytes=0;
 check(OwnedQsaElasticPool::host_bound(plan,owner_bytes,range_bytes),"actual owner/range quote using actual storage/class declarations");
 for(int mode=0;mode<3;++mode){
  std::unique_ptr<PreparedVmmAllocationBatch> batch;
  auto device=mint(budget,physical(gran)),metadata=mint(budget,host(batch_bytes));
  check(PreparedVmmAllocationBatch::prepare({&batch_plan,1},std::move(device),std::move(metadata),batch,error),"actual batch for pool");
  auto* original=batch.get();auto owner=mint(budget,host(owner_bytes)),range=mint(budget,host(range_bytes));
  if(mode==0)owner=rebox(std::move(owner));if(mode==1)range=rebox(std::move(range));
  std::shared_ptr<OwnedQsaElasticPool> pool;const auto before=allocations;
  const bool accepted=OwnedQsaElasticPool::prepare(geometry,plan,std::move(owner),std::move(range),batch,pool,error);
  check(accepted==(mode==2),"pool owner/range rebox refusal and canonical positive");
  if(mode!=2)check(!pool&&batch.get()==original&&allocations==before,"pool refusal preserves exact batch before storage allocation");
  else check(pool&&!batch&&!pool->base(),"canonical host owner consumes batch without device activation");
  pool.reset();batch.reset();check(budget.snapshot().reserved.host_bytes==0&&budget.snapshot().reserved.device_bytes[0]==0,"pool actual retained charge drain");
 }
 std::puts("PASS canonical range/batch/owned-pool host admissions (geometry/backend seams)");
}
