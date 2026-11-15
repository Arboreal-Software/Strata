// Adapter + actual tracker/inventory test; RemoteOpt callbacks are SCRIPTED CPU
// seams, not execution of actual CUDA adaptation, writer locks or stream drains.
#include "strata/core/remote_policy_publication.hpp"
#include "strata/core/remote_experts.hpp"
#include <sys/stat.h>
#include <new>
#include <type_traits>
#include "genuine_source_fixture.inc"
static bool reject_alloc=false;
void* operator new(size_t n){if(reject_alloc)throw std::bad_alloc();if(void*p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](size_t n){return ::operator new(n);}
void operator delete(void*p)noexcept{std::free(p);}void operator delete[](void*p)noexcept{std::free(p);}
void operator delete(void*p,size_t)noexcept{std::free(p);}void operator delete[](void*p,size_t)noexcept{std::free(p);}
static int scenario=0,submitted=0,source_checks=0;
static ExpertSource* expected_source=nullptr;
static bool reverse_plan=false;
namespace strata::core {
// Fixture-only link seam. Actual ExpertCache's destructor references this even
// for an empty cache. Never simulate release of GPU resources or managed owners:
// only a completely pristine default VmmRange is accepted; any other state aborts.
void VmmRange::release() {
 if(base_ || !h_.empty() || h_.capacity() || metadata_ ||
    !managed_slots_.empty() || managed_slots_.capacity() || retirement_generation_ ||
    retirement_failed_ || retirement_complete_ || managed_ || managed_unknown_ ||
    managed_internal_ || managed_prepared_ || managed_activation_attempted_ ||
    managed_granularity_ || managed_device_!=-1 || strict_retirement_ || retirement_device_!=-1) {
  std::fputs("UNEXPECTED: adapter fixture attempted non-pristine VMM release\n",stderr);
  std::abort();
 }
}
// Empty real helper objects exist for exact runtime pointer ownership checks;
// their kernels/allocator are NOT exercised by this adapter fixture.
RemoteExperts::~RemoteExperts()=default;
RemoteExpertOpt::~RemoteExpertOpt()=default;
struct RemoteExpertOpt::OwnedAdaptWorkspace::Storage {};
RemoteExpertOpt::OwnedAdaptWorkspace::OwnedAdaptWorkspace()=default;
RemoteExpertOpt::OwnedAdaptWorkspace::~OwnedAdaptWorkspace()=default;
RemoteExpertOpt::AdaptResult RemoteExpertOpt::adapt_owned(const std::vector<float>&,
 const std::vector<int32_t>&,const std::vector<std::pair<int32_t,int32_t>>&,
 int,ExpertSource& source,OwnedAdaptWorkspace&,AdaptPublication& sink,std::string&) {
 if(&source!=expected_source)std::abort();
 ++source_checks;
 const int from=reverse_plan?1:0,to=reverse_plan?0:1;
 AdaptPlanEntry plan[]={{0,0,0,from,to,96},{1,0,0,from,to,96}};
 if(scenario==1)plan[1].slot=2; // all-plan refusal before first effect
 if(!sink.prepare_all(plan,2)){sink.abandon_unattempted();return AdaptResult::failed;}
 for(uint32_t p=0;p<2;++p) {
  if(!sink.begin_attempt(p)){sink.fail_unknown(p);sink.abandon_unattempted();return AdaptResult::failed;}
  ++submitted;
  if((scenario==2||scenario==5)&&p==1){sink.fail_unknown(p);sink.abandon_unattempted();return AdaptResult::failed;}
  AdaptPlanEntry supplied=plan[p];if(scenario==3&&p==1)++supplied.bytes;
  if(!sink.publish_after_copies(p,&supplied,1)){sink.fail_unknown(p);sink.abandon_unattempted();return AdaptResult::failed;}
  if(scenario==4&&p==0){sink.fail_unknown(p);sink.abandon_unattempted();return AdaptResult::failed;}
 }
 sink.abandon_unattempted();return AdaptResult::complete;
}
}
struct Runtime final:NativeRuntimeAttachment {
 FileExpertSource source;RemoteExperts helpers[2];RemoteExpertOpt opt;
};
static bool owns(const NativeRuntimeAttachment& base,const ExpertSource& source,const RemoteExpertOpt& opt,
 std::span<const RemotePolicyPeerBinding> bindings)noexcept {
 auto* r=dynamic_cast<const Runtime*>(&base);
 return r&&&source==&r->source&&&opt==&r->opt&&bindings.size()==2&&
 bindings[0].helper==&r->helpers[0]&&bindings[0].tier==512&&bindings[0].device==0&&
 bindings[1].helper==&r->helpers[1]&&bindings[1].tier==514&&bindings[1].device==0;
}
static std::shared_ptr<ExpertPolicyCapture> policy(Fixture& f) {
 std::string e;ExpertProfileLimits limits{4,512};uint64_t bytes=0;require(expert_profile_inventory_host_bound(limits,bytes,e),e);
 const auto& input=f.bindings->expert_profile;struct stat st{};require(fstat(input.readonly_fd(),&st)==0,"real profile stat");
 std::unique_ptr<ExpertProfileCapture> rank;
 require(ExpertProfileCapture::create(limits,f.metadata.reserve({{0},bytes},ReservationPurpose::foreground).lease,f.bindings,
 {input.loader_path(),uint64_t(st.st_dev),uint64_t(st.st_ino),input.size()},rank,e),e);
 std::vector<std::pair<int32_t,int32_t>> order;int64_t slots=0;
 require(read_expert_profile(input.loader_path(),1,2,order,slots,e,rank.get()),e);
 auto ranking=std::shared_ptr<ExpertProfileCapture>(std::move(rank));ExpertPolicyLimits pl{2,2,4,2,2};
 require(expert_policy_inventory_host_bound(pl,bytes,e),e);std::unique_ptr<ExpertPolicyCapture> c;
 require(ExpertPolicyCapture::create(pl,f.metadata.reserve({{0},bytes},ReservationPurpose::foreground).lease,ranking,c,e),e);
 ExpertPolicyTierIdentity ids[]={{512,ExpertPolicyTierKind::remote},{514,ExpertPolicyTierKind::remote}};
 require(c->declare_tiers(ids,2),"declare sparse tiers");
 for(auto id:ids){ExpertPolicyTier t;t.id=id.id;t.kind=id.kind;t.device=0;t.layer_end=1;t.slots=1;t.decision_limit=1;t.slot_guard_bytes=96;
  require(c->configure_tier(t),"configure actual-sized tier");
  require(c->record_placement({id.id,1,96,0,0,0,ExpertPolicyOutcome::copied}),"initial seed placement receipt");
  require(c->complete_tier(id.id,1,ExpertPolicyTermination::ranking_end),"complete seed tier");}
 require(c->publish_initial(),"initial receipt complete");return std::shared_ptr<ExpertPolicyCapture>(std::move(c));
}
int main(int argc,char** argv) {
 static_assert(!std::is_copy_constructible_v<RemotePolicyPublication>&&!std::is_move_constructible_v<RemotePolicyPublication>);
 Fixture f(false);std::string e;auto runtime=std::make_shared<Runtime>();auto source_capture=f.capture();
 require(runtime->source.set_strict_retirement(true,{0},e),e);require(runtime->source.enable_inventory_capture(source_capture,e),e);
 runtime->source.set_gguf(f.bindings->native_primary.loader_path());require(runtime->source.open(f.bindings->pack_path(),1,2,e),e);
 auto capture=policy(f);std::shared_ptr<ExpertSource> source(runtime,&runtime->source);expected_source=source.get();
 RemotePolicyPeerBinding peers[]={{&runtime->helpers[0],512,0},{&runtime->helpers[1],514,0}};
 RemotePolicyPublicationLimits limits{2,1,2,1<<20,{2,2,2,1,1,1<<20}};
 uint64_t need=0;require(RemotePolicyPublication::required_bytes(limits,need),"host bound");
 std::unique_ptr<RemotePolicyPublication> adapter;
 auto baseline=f.metadata.snapshot();
 auto forged=std::shared_ptr<ExpertSource>(&runtime->source,[](ExpertSource*){});
 require(!RemotePolicyPublication::create(capture,runtime,forged,source_capture,runtime->opt,peers,7,limits,f.metadata,owns,adapter,e)&&!adapter,"same address unrelated control block refuses");
 require(same_budget(baseline,f.metadata.snapshot()),"refusal preserves ledger");
 peers[1].tier=513;
 require(!RemotePolicyPublication::create(capture,runtime,source,source_capture,runtime->opt,peers,7,limits,f.metadata,owns,adapter,e),"compact peer is not tier arithmetic");peers[1].tier=514;
 reject_alloc=true;bool created=RemotePolicyPublication::create(capture,runtime,source,source_capture,runtime->opt,peers,7,limits,f.metadata,owns,adapter,e);reject_alloc=false;
 require(!created&&!adapter&&same_budget(baseline,f.metadata.snapshot()),"persistent allocation refusal atomic");
 WindowBudget small({{0},1,0,0,0,0},{{0}});
 require(!RemotePolicyPublication::create(capture,runtime,source,source_capture,runtime->opt,peers,7,limits,small,owns,adapter,e),"real metadata budget refusal");
 require(RemotePolicyPublication::create(capture,runtime,source,source_capture,runtime->opt,peers,7,limits,f.metadata,owns,adapter,e),e);
 auto* original=adapter.get();forged.reset();
 require(!RemotePolicyPublication::create(capture,runtime,{},source_capture,runtime->opt,peers,7,limits,f.metadata,owns,adapter,e)&&adapter.get()==original,"failed replacement leaves exact output");
 capture->begin_dynamic_change();require(!capture->completed_inventory(),"initial receipt remains stale");
 RemoteExpertOpt::OwnedAdaptWorkspace workspace;std::vector<float> usage{0,3};std::vector<int32_t> primary{-1,-1};std::vector<std::pair<int32_t,int32_t>> pending;
 scenario=1;require(adapter->adapt(usage,primary,pending,1,workspace,e)==RemoteExpertOpt::AdaptResult::failed&&submitted==0&&!adapter->view().unknown,"invalid final peer has no effects and no outstanding transaction");
 scenario=argc>1?std::atoi(argv[1]):0;
 if(scenario==6){
  // Same actual source object reopens with a NEW one-shot capture. The adapter
  // retains the old capture; neither object-address nor receipt ABA can heal it.
  runtime->source.close();auto replacement=f.capture();
  require(runtime->source.enable_inventory_capture(replacement,e),e);
  require(runtime->source.open(f.bindings->pack_path(),1,2,e),e);
  const int before=submitted;
  require(adapter->adapt(usage,primary,pending,1,workspace,e)==RemoteExpertOpt::AdaptResult::failed&&submitted==before&&adapter->view().unknown,"real source reopen invalidates original authority");
  adapter.reset();source.reset();source_capture.reset();capture.reset();runtime.reset();
  std::puts("PASS: genuine source reopen refused");return 0;
 }
 reject_alloc=true;
 auto result=adapter->adapt(usage,primary,pending,1,workspace,e);reject_alloc=false;
 if(!scenario){
  require(result==RemoteExpertOpt::AdaptResult::complete&&submitted==2&&source_checks==2,"bounded callbacks do not allocate");
  ExpertPlacementValue h,c;require(adapter->remote_cell(1,0,h,c)&&h==c&&c.expert==1&&c.source_ordinal==0,"sparse tier publishes exact ranking ordinal, not artifact index");
  require(!capture->completed_inventory()&&!adapter->view().active_transactions,"no fabricated initial current receipt");
  reverse_plan=true;require(adapter->adapt(usage,primary,pending,1,workspace,e)==RemoteExpertOpt::AdaptResult::complete,"repeated bounded reuse");
 }else{
  require(result==RemoteExpertOpt::AdaptResult::failed&&adapter->view().unknown&&adapter->view().active_transactions,"partial failure retains unknown transaction");
  const int before=submitted;require(adapter->adapt(usage,primary,pending,1,workspace,e)==RemoteExpertOpt::AdaptResult::failed&&submitted==before,"unknown cannot retry");
  if(scenario==5){
   std::weak_ptr<Runtime> weak=runtime;adapter.reset();source.reset();runtime.reset();
   require(!weak.expired()&&expert_policy_runtime_quarantine_count()>0,"unknown drop retains actual runtime and source");
   // Deliberate process-lifetime quarantine, with no real device effects.
   // Do not run normal fixture retirement against intentionally retained files.
   std::puts("PASS: actual runtime/source quarantined until process exit");std::fflush(stdout);std::_Exit(0);
  }
  require(adapter->resolve_after_teardown(runtime,7),"explicit test-owned no-device teardown resolution");
 }
 adapter.reset();source.reset();source_capture.reset();capture.reset();runtime.reset();
 std::puts("PASS: actual adapter/tracker with genuine source metadata and scripted RemoteOpt callbacks");
}
