// Exact actual method bodies/HostCap bookkeeping; ONLY native allocation,
// geometry/config resolution and controller phase fields are CPU seams.
#include "strata/core/window_budget.hpp"
#include "strata/core/verifier_completion_gate.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <optional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
// One-shot allocation fault: diagnostic construction may allocate after failure.
static int new_fail_after=-1;
void* operator new(std::size_t n){if(new_fail_after==0){new_fail_after=-1;throw std::bad_alloc();}if(new_fail_after>0)--new_fail_after;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
namespace strata::core {
class Verifier {
public:
 Verifier* next_=nullptr;bool strict_retirement_=true,retirement_init_started_=false;
 std::atomic<bool> released_{false},terminal_copy_query_failed_{false};
 std::shared_ptr<VerifierCompletionGate> completion_gate_;std::size_t completion_stage_=0;
 static bool prepare_serial_completion_gate(Verifier&,std::shared_ptr<const ReservationLease>,std::shared_ptr<VerifierCompletionGate>&,std::string&,std::shared_ptr<const void> = {});
#include "gate_hooks.inc"
};
#include "gate_factory.inc"
}
namespace strata::program {
using namespace core;
enum class SuffixRawStatus {ok,invalid,budget_refused,poisoned};
struct SuffixRawResult {SuffixRawStatus status;std::string detail;std::shared_ptr<const void> retained_failure;explicit operator bool()const{return status==SuffixRawStatus::ok;}};
using StateDigest=std::array<uint8_t,32>;
struct NativeWindowStage {};
struct WindowKey {uint64_t engine_generation=1,window_generation=1;std::string session_id="s",thread_id="t",window_id="w";StateDigest model_profile{1},topology{1};};
struct AllocationSeam {
 std::shared_ptr<const ReservationLease> parent;bool good=true;
 bool healthy()const{return good;}
 const std::shared_ptr<const ReservationLease>& reservation()const{return parent;}
};
struct GeometrySeam {int64_t n_layers=1;};
struct OccupancySeam {bool active_=true;bool active()const{return active_;}};
class SuffixRawTransaction {
public:
 struct HostCap;struct HostCredit;
#include "host_usage.inc"
 std::shared_ptr<void> pending_;
 enum class Phase {idle,running,drained,capturing,restoring,poisoned};
 std::shared_ptr<HostCredit> setup_credit_;
 GeometrySeam geometry_storage_;GeometrySeam geometry_;const GeometrySeam* const geometry_identity_;
 std::shared_ptr<OccupancySeam> slot_occupancy_;
 std::shared_ptr<void> runtime_=std::make_shared<int>(1);
 std::shared_ptr<HostCredit> slot_host_root_credit_;
 WindowKey identity_;
 AllocationSeam allocation_;WindowBudget& budget_;
 uint64_t max_tokens_=1,max_bytes_=1,host_limit_;Phase phase_=Phase::idle;
 std::thread::id controller_=std::this_thread::get_id();bool initialized_=false;bool steering_enabled_=false,resolve_good_=true;
 std::vector<NativeWindowStage> initial_config_;std::shared_ptr<HostCap> host_cap_;
 SuffixRawTransaction(WindowBudget& b,std::shared_ptr<const ReservationLease> p,uint64_t h,const GeometrySeam* geometry=nullptr):geometry_(geometry?*geometry:geometry_storage_),geometry_identity_(geometry?geometry:&geometry_storage_),allocation_{std::move(p)},budget_(b),host_limit_(h){}
 bool resolve(std::vector<NativeWindowStage>& out,std::string& error){if(!resolve_good_){error="explicit CPU config seam refusal";return false;}out.resize(1);return true;}
 bool identity_matches(const WindowKey&)const;
 SuffixRawResult initialize();
 SuffixRawResult initialize_resident_shared_host(const SuffixRawTransaction&);
 SuffixRawResult initialize_slot_shared_host(const SuffixRawTransaction&);
 SuffixRawResult admit_host(uint64_t,bool,std::shared_ptr<HostCredit>&);
 SuffixRawResult admit_resident_setup(const WindowResources&,std::shared_ptr<const void>&);
 SuffixRawResult admit_completion_gate_metadata(std::shared_ptr<const ReservationLease>&,std::shared_ptr<const void>&);
};
#include "actual.inc"
}
using namespace strata::core;
using namespace strata::program;
static WindowResources caps(uint64_t host=64u<<20){return {{1024},host,host,0,0,16};}
static std::shared_ptr<const ReservationLease> parent(WindowBudget& b){auto r=b.reserve({{0},10,0,0,0,1},ReservationPurpose::private_work);assert(r);return std::move(r.lease);}
static void initialize(SuffixRawTransaction& t){assert(t.initialize());assert(t.initialized_);t.phase_=SuffixRawTransaction::Phase::drained;}
static WindowResources request(uint64_t host=128){return {{8},host,64,0,0,0};}
int main(){
 // Actual typed-slot shared-H bootstrap: physical occupancy/geometry/runtime are
 // recording seams; cap, bootstrap/claim methods, global ledger and leases execute.
 {WindowBudget b(caps(),{{0}});SuffixRawTransaction live(b,parent(b),32u<<20);initialize(live);
  const auto one=live.host_cap_->live.load();auto cap=live.host_cap_;
  SuffixRawTransaction a(b,parent(b),32u<<20,live.geometry_identity_),z(b,parent(b),32u<<20,live.geometry_identity_);
  for(auto* slot:{&a,&z}){slot->runtime_=live.runtime_;slot->slot_occupancy_=std::make_shared<OccupancySeam>();assert(slot->initialize_slot_shared_host(live));slot->phase_=SuffixRawTransaction::Phase::drained;}
  assert(a.host_cap_==cap&&z.host_cap_==cap&&cap->limit==(32u<<20)&&cap->claim_limit==48);
  assert(cap->live==3*one&&cap->claims==3&&a.slot_host_root_credit_==live.setup_credit_&&z.slot_host_root_credit_==live.setup_credit_);
  // Extra slot/private bootstrap cannot invent another32MiB H owner.
  SuffixRawTransaction extra(b,parent(b),32u<<20,live.geometry_identity_);extra.runtime_=live.runtime_;extra.slot_occupancy_=std::make_shared<OccupancySeam>();
  const auto before=b.snapshot().reserved;assert(!extra.initialize_slot_shared_host(live));assert(!extra.host_cap_&&!extra.setup_credit_&&!extra.initialized_&&cap->live==3*one&&cap->claims==3&&b.snapshot().reserved==before);
  // The genuinely shared claim directory admits48 total; the original main-only
  // assertions below still require16 when no slot has actually joined.
  std::vector<std::shared_ptr<const void>> held(45);for(auto& c:held)assert(a.admit_resident_setup(request(),c));
  std::shared_ptr<const void> refused;assert(cap->claims==48&&!z.admit_resident_setup(request(),refused)&&!refused);
  held.clear();assert(cap->claims==3&&cap->live==3*one);
 }
 // Refuse equal-looking foreign geometry, different runtime control block,
 // inactive occupancy and foreign ledger before retaining H/setup aliases.
 for(int bad=0;bad<5;++bad){WindowBudget b(caps(),{{0}}),foreign(caps(),{{0}});SuffixRawTransaction live(b,parent(b),32u<<20);initialize(live);
  SuffixRawTransaction slot(b,parent(b),32u<<20,bad==0?nullptr:live.geometry_identity_);slot.runtime_=live.runtime_;slot.slot_occupancy_=std::make_shared<OccupancySeam>();
  if(bad==1)slot.runtime_=std::shared_ptr<void>(live.runtime_.get(),[](void*){});
  if(bad==2)slot.slot_occupancy_->active_=false;
  if(bad==3)slot.allocation_.parent=parent(foreign);
  if(bad==4)slot.identity_.model_profile[0]=2;
  auto before=b.snapshot().reserved;auto bytes=live.host_cap_->live.load();assert(!slot.initialize_slot_shared_host(live));
  assert(!slot.host_cap_&&!slot.setup_credit_&&!slot.initialized_&&b.snapshot().reserved==before&&live.host_cap_->live==bytes&&live.host_cap_->claims==1);
 }
 // Root bootstrap credit stays alive through a descendant slot even after main
 // adapter destruction, then all real credits/parent leases release on teardown.
 {WindowBudget b(caps(),{{0}});std::optional<SuffixRawTransaction> slot;std::weak_ptr<SuffixRawTransaction::HostCap> cap;GeometrySeam geometry;
  {SuffixRawTransaction live(b,parent(b),32u<<20,&geometry);initialize(live);cap=live.host_cap_;slot.emplace(b,parent(b),32u<<20,&geometry);slot->runtime_=live.runtime_;slot->slot_occupancy_=std::make_shared<OccupancySeam>();assert(slot->initialize_slot_shared_host(live));}
  assert(!cap.expired()&&slot->host_cap_->claims==2&&slot->slot_host_root_credit_);slot.reset();assert(cap.expired()&&b.snapshot().reserved.host_bytes==0);
 }

 // Actual initialize admits its retained private_attachment setup first. Wrong
 // ledger is refused BEFORE initialized_, never forged by this fixture.
 {WindowBudget a(caps(),{{0}}),b(caps(),{{0}});SuffixRawTransaction t(b,parent(a),32u<<20);
  assert(!t.initialize()&&!t.initialized_);assert(t.host_cap_->live==0&&t.host_cap_->claims==0);
  std::shared_ptr<const ReservationLease> m;std::shared_ptr<const void> c;
  assert(!t.admit_completion_gate_metadata(m,c)&&!m&&!c);assert(b.snapshot().reserved.host_bytes==0);}
 {WindowBudget b(caps(),{{0}});SuffixRawTransaction t(b,parent(b),32u<<20);initialize(t);
  const auto base=b.snapshot().reserved;const auto live=t.host_cap_->live.load();const auto slots=t.host_cap_->claims.load();
  std::shared_ptr<const void> credit;assert(t.admit_resident_setup(request(),credit));
  assert(b.snapshot().reserved.host_bytes==base.host_bytes+128&&b.snapshot().reserved.pinned_bytes==64&&b.snapshot().reserved.device_bytes[0]==8);
  assert(t.host_cap_->live==live+128&&t.host_cap_->claims==slots+1);
  auto retained=credit;credit.reset();assert(t.host_cap_->claims==slots+1);retained.reset();
  assert(b.snapshot().reserved==base&&t.host_cap_->live==live&&t.host_cap_->claims==slots);
  // Canonical exclusive foreground metadata and separate aggregate-only credit.
  std::shared_ptr<const ReservationLease> m;assert(t.admit_completion_gate_metadata(m,credit));
  const auto bytes=VerifierCompletionGate::metadata_bytes();
  assert(m->canonical_owner(m)&&m.use_count()==1&&m->purpose()==ReservationPurpose::foreground);
  assert(b.snapshot().reserved.host_bytes==base.host_bytes+bytes&&t.host_cap_->live==live+bytes);
  m.reset();assert(b.snapshot().reserved==base&&t.host_cap_->live==live+bytes);credit.reset();
  assert(t.host_cap_->live==live&&t.host_cap_->claims==slots);
  // Fixed claim count saturated without growing the directory.
  std::vector<std::shared_ptr<const void>> held(15);
  for(auto& c:held)assert(t.admit_resident_setup(request(),c));
  assert(t.host_cap_->claims==16);assert(!t.admit_resident_setup(request(),credit)&&!credit);
  assert(!t.admit_completion_gate_metadata(m,credit)&&!m&&!credit);
  held.clear();assert(b.snapshot().reserved==base&&t.host_cap_->claims==slots);
 }
 // H exhaustion and actual global exhaustion independently roll back claims.
 for(bool global:{false,true}){WindowBudget b(caps(),{{0}});SuffixRawTransaction t(b,parent(b),32u<<20);initialize(t);
  const auto base=b.snapshot().reserved;
  std::shared_ptr<const void> host_blocker;std::shared_ptr<const ReservationLease> blocker;
  if(global){auto r=b.reserve({{0},(64u<<20)-base.host_bytes},ReservationPurpose::foreground);assert(r);blocker=std::move(r.lease);}
  else {assert(t.admit_resident_setup(request(t.host_cap_->limit-t.host_cap_->live.load()-1),host_blocker));}
  const auto live=t.host_cap_->live.load();const auto slots=t.host_cap_->claims.load();
  const auto before=b.snapshot().reserved;std::shared_ptr<const void> c;std::shared_ptr<const ReservationLease> m;
  assert(!t.admit_resident_setup(request(),c)&&!c);
  assert(!t.admit_completion_gate_metadata(m,c)&&!m&&!c);
  assert(b.snapshot().reserved==before&&t.host_cap_->live==live&&t.host_cap_->claims==slots);
 }
 {WindowBudget b(caps(),{{0}});SuffixRawTransaction t(b,parent(b),32u<<20);initialize(t);
  const auto base=b.snapshot().reserved;auto live=t.host_cap_->live.load();auto slots=t.host_cap_->claims.load();
  std::shared_ptr<const void> c;std::shared_ptr<const ReservationLease> m;
  auto bad=request();bad.pinned_bytes=129;assert(!t.admit_resident_setup(bad,c));
  for(int field=0;field<4;++field){bad=request();if(field==0)bad.device_bytes.push_back(0);if(field==1)bad.disk_bytes=1;if(field==2)bad.io_slots=1;if(field==3)bad.shadows=1;assert(!t.admit_resident_setup(bad,c));}
  t.phase_=SuffixRawTransaction::Phase::running;assert(!t.admit_resident_setup(request(),c));assert(!t.admit_completion_gate_metadata(m,c));t.phase_=SuffixRawTransaction::Phase::drained;
  std::thread foreign([&]{assert(!t.admit_resident_setup(request(),c));assert(!t.admit_completion_gate_metadata(m,c));});foreign.join();
  auto marker=std::make_shared<int>(7);c=marker;assert(!t.admit_resident_setup(request(),c)&&c==marker);assert(!t.admit_completion_gate_metadata(m,c)&&c==marker&&!m);c.reset();
  auto existing=b.reserve({{0},1},ReservationPurpose::foreground);assert(existing);m=std::move(existing.lease);auto old=m;
  assert(!t.admit_completion_gate_metadata(m,c)&&m==old&&!c);m.reset();old.reset();
  assert(b.snapshot().reserved==base&&t.host_cap_->live==live&&t.host_cap_->claims==slots);
 }
 // Failure before H admission and after the temporary H claim both unwind.
 for(int fail_at:{0,1})for(bool gate:{false,true}){
  WindowBudget b(caps(),{{0}});SuffixRawTransaction t(b,parent(b),32u<<20);initialize(t);
  const auto before=b.snapshot().reserved;const auto live=t.host_cap_->live.load();const auto slots=t.host_cap_->claims.load();
  std::shared_ptr<const void> c;std::shared_ptr<const ReservationLease> m;auto req=request();
  new_fail_after=fail_at;
  const auto result=gate?t.admit_completion_gate_metadata(m,c):t.admit_resident_setup(req,c);
  assert(new_fail_after==-1);new_fail_after=-1;
  assert(!result&&!m&&!c&&b.snapshot().reserved==before&&t.host_cap_->live==live&&t.host_cap_->claims==slots);
 }
 // A real metadata method result feeds the actual gate factory; historical
 // Cut outlives transaction, native-allocation seam and verifier handles.
 {WindowBudget b(caps(),{{0}});std::optional<VerifierCompletionGate::Cut> cut;cut.emplace();
  std::weak_ptr<SuffixRawTransaction::HostCap> weak;const auto bytes=VerifierCompletionGate::metadata_bytes();
  {SuffixRawTransaction t(b,parent(b),32u<<20);initialize(t);weak=t.host_cap_;
   std::shared_ptr<const ReservationLease> metadata;std::shared_ptr<const void> credit;
   assert(t.admit_completion_gate_metadata(metadata,credit));
   Verifier v;std::shared_ptr<VerifierCompletionGate> gate;std::string error;
   assert(Verifier::prepare_serial_completion_gate(v,std::move(metadata),gate,error,std::move(credit)));
   VerifierCompletionGate::Token token;assert(gate->try_begin(token));assert(gate->commit_after_checked_completion(token,*cut));}
  auto held=weak.lock();assert(held&&held->claims==1&&held->live==bytes);held.reset();
  assert(b.snapshot().reserved.host_bytes==bytes);cut.reset();assert(weak.expired()&&b.snapshot().reserved.host_bytes==0);
 }
 // A resident setup owner itself retains cap/parent after transaction teardown.
 {WindowBudget b(caps(),{{0}});std::shared_ptr<const void> credit;std::weak_ptr<SuffixRawTransaction::HostCap> weak;
  {SuffixRawTransaction t(b,parent(b),32u<<20);initialize(t);weak=t.host_cap_;assert(t.admit_resident_setup(request(),credit));}
  auto held=weak.lock();assert(held&&held->claims==1&&held->live==128);held.reset();
  assert(b.snapshot().reserved.host_bytes==138);credit.reset();assert(weak.expired()&&b.snapshot().reserved.host_bytes==0);
 }
 // Pinned is a subset cap, with no persistent H claim after ledger refusal.
 {auto cap=caps();cap.pinned_bytes=32;WindowBudget b(cap,{{0}});SuffixRawTransaction t(b,parent(b),32u<<20);initialize(t);
  auto before=b.snapshot().reserved;auto live=t.host_cap_->live.load();auto slots=t.host_cap_->claims.load();std::shared_ptr<const void> c;
  assert(!t.admit_resident_setup(request(),c)&&!c);assert(b.snapshot().reserved==before&&t.host_cap_->live==live&&t.host_cap_->claims==slots);
 }

 // Two ACTUAL bootstrap methods share one cap. Each request alone fits the
 // remaining single-owner capacity; the second combined request must refuse.
 {WindowBudget b(caps(),{{0}});SuffixRawTransaction live(b,parent(b),20u<<20);initialize(live);
  const auto one=live.host_cap_->live.load();SuffixRawTransaction resident(b,parent(b),20u<<20);
  assert(resident.initialize_resident_shared_host(live));resident.phase_=SuffixRawTransaction::Phase::drained;
  assert(resident.host_cap_==live.host_cap_&&live.host_cap_->claims==2&&live.host_cap_->live==2*one);
  auto before=b.snapshot().reserved;std::shared_ptr<const void> first,second;
  assert(live.admit_resident_setup(request(2u<<20),first));
  assert(!resident.admit_resident_setup(request(2u<<20),second)&&!second);
  assert(live.host_cap_->live==2*one+(2u<<20)&&live.host_cap_->claims==3);
  first.reset();assert(b.snapshot().reserved==before&&live.host_cap_->live==2*one&&live.host_cap_->claims==2);
 }
 // Distinct budget OBJECT and genuine foreign parent each refuse; no cap alias
 // or initialized flag escapes the failed bootstrap.
 {WindowBudget a(caps(),{{0}}),b(caps(),{{0}});SuffixRawTransaction live(a,parent(a),32u<<20);initialize(live);
  auto before=a.snapshot().reserved;const auto bytes=live.host_cap_->live.load();
  {SuffixRawTransaction wrong_budget(b,parent(b),32u<<20);assert(!wrong_budget.initialize_resident_shared_host(live));assert(!wrong_budget.host_cap_&&!wrong_budget.setup_credit_&&!wrong_budget.initialized_);}
  {SuffixRawTransaction foreign_parent(a,parent(b),32u<<20);assert(!foreign_parent.initialize_resident_shared_host(live));assert(!foreign_parent.host_cap_&&!foreign_parent.setup_credit_&&!foreign_parent.initialized_);}
  assert(a.snapshot().reserved==before&&live.host_cap_->live==bytes&&live.host_cap_->claims==1);
 }
 {WindowBudget b(caps(),{{0}});SuffixRawTransaction live(b,parent(b),32u<<20);initialize(live);SuffixRawTransaction resident(b,parent(b),32u<<20);
  auto before=b.snapshot().reserved;auto bytes=live.host_cap_->live.load();
  std::thread foreign([&]{assert(!resident.initialize_resident_shared_host(live));});foreign.join();
  live.phase_=SuffixRawTransaction::Phase::running;assert(!resident.initialize_resident_shared_host(live));live.phase_=SuffixRawTransaction::Phase::drained;
  resident.phase_=SuffixRawTransaction::Phase::running;assert(!resident.initialize_resident_shared_host(live));resident.phase_=SuffixRawTransaction::Phase::idle;
  assert(!resident.host_cap_&&!resident.setup_credit_&&!resident.initialized_&&b.snapshot().reserved==before&&live.host_cap_->live==bytes&&live.host_cap_->claims==1);
 }
 // Exhaustion before bootstrap and one-shot bad_alloc before/after H claim
 // leave a failed transaction with NO retained alias and no additional charge.
 for(int failure:{0,1,2,3}){WindowBudget b(caps(),{{0}});SuffixRawTransaction live(b,parent(b),32u<<20);initialize(live);
  SuffixRawTransaction resident(b,parent(b),32u<<20);std::shared_ptr<const void> hblock;std::shared_ptr<const ReservationLease> globalblock;
  if(failure==0)assert(live.admit_resident_setup(request(live.host_cap_->limit-live.host_cap_->live.load()-1),hblock));
  if(failure==1){auto rem=(64u<<20)-b.snapshot().reserved.host_bytes;auto r=b.reserve({{0},rem},ReservationPurpose::foreground);assert(r);globalblock=std::move(r.lease);}
  const auto before=b.snapshot().reserved;const auto bytes=live.host_cap_->live.load();const auto claims=live.host_cap_->claims.load();
  if(failure>=2)new_fail_after=failure-2;
  assert(!resident.initialize_resident_shared_host(live));assert(new_fail_after==-1);new_fail_after=-1;
  assert(!resident.host_cap_&&!resident.setup_credit_&&!resident.initialized_&&b.snapshot().reserved==before&&live.host_cap_->live==bytes&&live.host_cap_->claims==claims);
 }
 // Admission succeeded but resolve refused: retained setup charge survives until
 // failed object destruction, matching actual initialize lifetime semantics.
 {WindowBudget b(caps(),{{0}});SuffixRawTransaction live(b,parent(b),32u<<20);initialize(live);auto one=live.host_cap_->live.load();
  {SuffixRawTransaction resident(b,parent(b),32u<<20);resident.resolve_good_=false;
   assert(!resident.initialize_resident_shared_host(live)&&!resident.initialized_&&resident.setup_credit_&&resident.host_cap_==live.host_cap_);
   assert(live.host_cap_->claims==2&&live.host_cap_->live==2*one);}
  assert(live.host_cap_->claims==1&&live.host_cap_->live==one);
 }
 // Successful resident ownership keeps the SAME cap alive after original live
 // transaction destruction; releasing resident drops the final actual charge.
 {WindowBudget b(caps(),{{0}});std::optional<SuffixRawTransaction> resident;std::weak_ptr<SuffixRawTransaction::HostCap> weak;uint64_t one=0;
  {SuffixRawTransaction live(b,parent(b),32u<<20);initialize(live);one=live.host_cap_->live.load();weak=live.host_cap_;
   resident.emplace(b,parent(b),32u<<20);assert(resident->initialize_resident_shared_host(live));}
  auto held=weak.lock();assert(held&&held==resident->host_cap_&&held->claims==1&&held->live==one);held.reset();
  assert(b.snapshot().reserved.host_bytes==one+10);resident.reset();assert(weak.expired()&&b.snapshot().reserved.host_bytes==0);
 }
 // Exact actual observer under real admissions/shared-H, with explicit phase seams.
 {WindowBudget b(caps(),{{0}});SuffixRawTransaction live(b,parent(b),32u<<20);
  using Usage=SuffixRawTransaction::HostUsage;Usage untouched{7,8,9};
  auto same=[](const Usage& a,const Usage& z){return a.limit==z.limit&&a.live==z.live&&a.claims==z.claims;};
  assert(!live.observe_host_usage(untouched)&&same(untouched,Usage{7,8,9}));
  initialize(live);Usage baseline;assert(live.observe_host_usage(baseline));
  const auto ledger=b.snapshot().reserved;
  Usage seen;assert(live.observe_host_usage(seen)&&same(seen,baseline));
  live.phase_=SuffixRawTransaction::Phase::running;assert(!live.observe_host_usage(untouched)&&same(untouched,Usage{7,8,9}));live.phase_=SuffixRawTransaction::Phase::drained;
  live.pending_=std::make_shared<int>(1);assert(!live.observe_host_usage(untouched)&&same(untouched,Usage{7,8,9}));live.pending_.reset();
  live.initialized_=false;assert(!live.observe_host_usage(untouched)&&same(untouched,Usage{7,8,9}));live.initialized_=true;
  bool foreign=true;std::thread t([&]{foreign=live.observe_host_usage(untouched);});t.join();assert(!foreign&&same(untouched,Usage{7,8,9}));
  std::shared_ptr<const void> retained;
  {SuffixRawTransaction child(b,parent(b),32u<<20);assert(child.initialize_resident_shared_host(live));child.phase_=SuffixRawTransaction::Phase::drained;
   Usage child_seen;assert(child.observe_host_usage(child_seen)&&live.observe_host_usage(seen)&&same(child_seen,seen));
   assert(seen.live==2*baseline.live&&seen.claims==baseline.claims+1);
   assert(child.admit_resident_setup(request(128),retained));assert(live.observe_host_usage(seen));
   assert(seen.live==2*baseline.live+128&&seen.claims==baseline.claims+2);
  }
  assert(live.observe_host_usage(seen)&&seen.live==baseline.live+128&&seen.claims==baseline.claims+1);
  auto another=retained;retained.reset();assert(live.observe_host_usage(seen)&&seen.live==baseline.live+128);
  another.reset();assert(live.observe_host_usage(seen)&&same(seen,baseline)&&b.snapshot().reserved==ledger);
 }
 std::puts("actual shared-H observation and inherited admission/lifetime controls PASS");
}
