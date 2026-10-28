#define main inherited_guard_fixture_main
#include "../lookahead_drain/test.cpp"
#undef main
static thread_local bool fail_allocations=false;
static std::atomic<void*> watched_storage{nullptr};
static std::weak_ptr<const ReservationLease> watched_lease;
static bool storage_freed_charged=false;
void* operator new(std::size_t n){if(fail_allocations)throw std::bad_alloc();if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{if(p==watched_storage.load(std::memory_order_relaxed)&&p){storage_freed_charged=!watched_lease.expired();watched_storage=nullptr;}std::free(p);}
void operator delete[](void* p)noexcept{::operator delete(p);}
void operator delete(void* p,std::size_t)noexcept{::operator delete(p);}
void operator delete[](void* p,std::size_t)noexcept{::operator delete(p);}
static auto routers(){auto rows=std::vector<std::vector<uint16_t>>(2,std::vector<uint16_t>(16));rows[1][0]=0x3f80;return rows;}
static strata::core::WindowResources charge(uint64_t n){strata::core::WindowResources x;x.device_bytes={0};x.host_bytes=n;return x;}
int main(){int cases=0;float x[8]={1};
 {R r;ExpertSource source;Barrier gate;std::vector<std::vector<int64_t>> seen;
  source.before_read=[&]{gate.block();};source.on_warm=[&](int64_t,const int64_t* p,int64_t n){seen.emplace_back();if(n)seen.back().assign(p,p+n);};
  std::string e;check(r.start(routers(),8,2,1,&source,e),"start snapshot worker");
  std::vector<int32_t> original{-1,-1,0,-1};r.submit(0,x,1,original.data());gate.wait();
  // Actual original storage can change/disappear once synchronous submit returns.
  original[2]=-1;original.clear();original.shrink_to_fit();
  int32_t missing[4]={-1,-1,-1,-1};r.submit(0,x,1,missing); // busy must not overwrite A
  gate.open();R::SubmittedDrain seal;
  check(r.drain_submitted(seal,Clock::now()+std::chrono::seconds(5))==D::drained,"snapshot first callback done");
  check(seen.size()==1&&seen[0].empty(),"worker used captured resident table despite freed original and busy submit");
  seal.reset();fail_allocations=true;r.submit(0,x,1,missing);fail_allocations=false;
  check(r.drain_submitted(seal,Clock::now()+std::chrono::seconds(5))==D::drained,"allocation-free accepted submit");
  check(seen.size()==2&&seen.back()==std::vector<int64_t>{0},"next accepted copy uses new missing table");
  int32_t resident[4]={0,0,0,0};seal.reset();r.submit(0,x,1,resident);
  check(r.drain_submitted(seal,Clock::now()+std::chrono::seconds(5))==D::drained,"resident repeat");check(seen.back().empty(),"resident snapshot filters");
  seal.reset();r.submit(0,x,1,nullptr);
  check(r.drain_submitted(seal,Clock::now()+std::chrono::seconds(5))==D::drained,"null repeat");
  check(seen.size()==4&&seen.back()==std::vector<int64_t>{0},"null means no filtering, not stale snapshot");++cases;
 }
 {uint64_t out=777;check(!R::residency_snapshot_bytes(SIZE_MAX,INT64_MAX,out)&&out==777,"overflow unchanged quote");
  check(!R::residency_snapshot_bytes(2,-1,out)&&out==777,"negative geometry");check(!R::residency_snapshot_bytes(2,0,out)&&out==777,"nonempty zero geometry");
  check(R::residency_snapshot_bytes(2,2,out)&&out==16,"exact payload quote");check(R::residency_snapshot_bytes(0,0,out)&&out==0,"legacy empty quote");
  R r;ExpertSource source;std::string e;check(r.start({},8,2,1,&source,e),"legacy empty router start remains valid");check(r.close_for_retirement(),"empty close");++cases;
 }
 {R r;ExpertSource source;std::string e="old error";auto rows=routers();fail_allocations=true;
  const bool ok=r.start(std::move(rows),8,2,1,&source,e);fail_allocations=false;
  check(!ok&&!r.thread_.joinable()&&r.routers_.empty()&&r.residency_snapshot_.empty()&&e.empty(),"persistent failure before thread/state publication");
  check(r.start(routers(),8,2,1,&source,e),"retry after prepublication allocation refusal");++cases;
 }
 {using namespace strata::core;WindowBudget budget(charge(1024),charge(0));std::weak_ptr<const ReservationLease> weak;
  {R r;auto reserved=budget.reserve(charge(16),ReservationPurpose::foreground);check(bool(reserved),"genuine reservation");weak=reserved.lease;
   check(r.configure_residency_snapshot(2,2,reserved.lease)&&!reserved.lease,"consume unique genuine quota");
   ExpertSource source;std::string e;check(!r.start({},8,2,1,&source,e)&&!r.thread_.joinable(),"admitted geometry mismatch before thread");
   check(r.start(routers(),8,2,1,&source,e),"admitted start");watched_storage=r.residency_snapshot_.data();watched_lease=weak;
   check(r.close_for_retirement()&&budget.snapshot().reserved.host_bytes==16,"stop retains array charge");
  }
  check(storage_freed_charged&&weak.expired()&&budget.snapshot().reserved.host_bytes==0,"actual lease survives storage and then returns quota");++cases;
 }
 {using namespace strata::core;WindowBudget budget(charge(1024),charge(0));R r;
  auto short_lease=budget.reserve(charge(15),ReservationPurpose::foreground);auto* identity=short_lease.lease.get();
  check(!r.configure_residency_snapshot(2,2,short_lease.lease)&&short_lease.lease.get()==identity,"short quota output unchanged");
  auto full=budget.reserve(charge(16),ReservationPurpose::foreground);auto copy=full.lease;
  check(!r.configure_residency_snapshot(2,2,full.lease)&&full.lease==copy,"shared quota not reusable");copy.reset();
  check(!r.configure_residency_snapshot(0,2,full.lease)&&bool(full.lease),"owned empty geometry refused");
  check(r.configure_residency_snapshot(2,2,full.lease),"valid quota following refusals");
  auto another=budget.reserve(charge(16),ReservationPurpose::foreground);check(!r.configure_residency_snapshot(2,2,another.lease)&&another.lease,"configured quota never replaced");++cases;
 }
 {using namespace strata::core;auto cap=charge(1024);cap.device_bytes={32};cap.pinned_bytes=32;cap.shadows=2;
  WindowBudget budget(cap,charge(0));R r;
  auto physical=charge(16);physical.device_bytes={1};auto d=budget.reserve(physical,ReservationPurpose::foreground);
  check(bool(d)&&!r.configure_residency_snapshot(2,2,d.lease)&&d.lease,"device charge cannot alias metadata quota");
  auto pinned=charge(16);pinned.pinned_bytes=1;auto p=budget.reserve(pinned,ReservationPurpose::foreground);
  check(bool(p)&&!r.configure_residency_snapshot(2,2,p.lease)&&p.lease,"pinned mixed charge refused");
  auto priv=charge(16);priv.shadows=1;auto q=budget.reserve(priv,ReservationPurpose::private_work);
  check(bool(q)&&!r.configure_residency_snapshot(2,2,q.lease)&&q.lease,"wrong purpose refused with real minted lease");++cases;
 }
 check(cases==6,"all snapshot scenarios executed");std::puts("PASS lookahead snapshot: 6 scenarios");
}
