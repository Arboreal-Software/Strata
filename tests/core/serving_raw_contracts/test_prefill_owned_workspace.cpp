// The checks include real quote/reservation calls and must execute in Release.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cstddef>
#include "strata/prefill/owned_workspace.hpp"
#include "strata/core/native_owned_batch_serving.hpp"
#include <string>
#include "strata/kernels/mrope.hpp"
#include "strata/core/window_budget.hpp"
#include <cassert>
#include <limits>
#include <memory>
#include <vector>
#include <iostream>
using namespace strata;
static prefill::OwnedWorkspaceShape shape(uint64_t chunk,bool borrowed=false,bool successor=false){
 prefill::OwnedWorkspaceShape s;
 s.chunk=chunk;s.embedding=4096;s.handoff_width=4096;s.routes=10;s.experts=256;s.layers=24;
 s.step_count=6;s.ple_heads=4;s.stager_slots=4;s.max_blob=65536;
 s.object_bytes=4096;s.stager_job_bytes=128;s.atomic_bytes=4;s.upload_slots=4;s.upload_min_bytes=65536;
 s.arena_device_bytes=64ULL<<20;s.borrowed=borrowed;s.successor=successor;return s;
}
static core::WindowResources resources(const prefill::OwnedWorkspaceBytes&q,size_t stage){
 core::WindowResources r;r.device_bytes={0,0};r.device_bytes[stage]=q.device;r.host_bytes=q.host;r.pinned_bytes=q.pinned;return r;
}
static core::WindowResources zero(){core::WindowResources r;r.device_bytes={0,0};return r;}
static void refusal_preserves(){
 auto s=shape(1024);prefill::OwnedWorkspaceBytes out{11,22,33};
 auto check=[&](auto bad){assert(!prefill::owned_workspace_quote(bad,out));assert(out.device==11&&out.host==22&&out.pinned==33);};
 auto bad=s;bad.chunk=0;check(bad);bad=s;bad.upload_slots=0;check(bad);bad=s;bad.upload_min_bytes=0;check(bad);
 bad=s;bad.successor=true;bad.handoff_width=0;check(bad);
 bad=s;bad.chunk=UINT64_MAX;check(bad);bad=s;bad.experts=UINT64_MAX;check(bad);
 bad=s;bad.object_bytes=UINT64_MAX;check(bad);bad=s;bad.max_blob=UINT64_MAX;check(bad);
 bad=s;bad.arena_device_bytes=UINT64_MAX;check(bad);
}
static void minimum_upload_footprint(){
 auto s=shape(1,true);s.embedding=s.handoff_width=s.routes=s.experts=s.layers=s.step_count=s.ple_heads=1;
 prefill::OwnedWorkspaceBytes minimum;
 assert(prefill::owned_workspace_quote(s,minimum));
 prefill::OwnedUploadRingBytes q;
 assert(prefill::owned_upload_ring_quote(1,1,1,1,q)&&q.slot==65536&&q.pinned==4*q.slot);
 assert(prefill::owned_upload_ring_quote(1024,4096,10,256,q)&&q.slot==4ULL*1024*4096&&q.pinned==4*q.slot);
 assert(prefill::owned_upload_ring_quote(1,1,10000,1,q)&&q.slot==4ULL*(3*10000+2*(1+0+2)));
 const auto before=q;
 for(uint64_t invalid:{uint64_t(0),UINT64_MAX}){
   assert(!prefill::owned_upload_ring_quote(invalid,4096,10,256,q));
   assert(q.slot==before.slot&&q.pinned==before.pinned);
 }
 assert(!prefill::owned_upload_ring_quote(1,1,1,UINT64_MAX,q));
 assert(q.slot==before.slot&&q.pinned==before.pinned);
 s.upload_min_bytes=131072;assert(!prefill::owned_workspace_quote(s,minimum));
 s.upload_min_bytes=65536;s.upload_slots=3;assert(!prefill::owned_workspace_quote(s,minimum));
}
static void shapes_and_borrowing(){
 prefill::OwnedWorkspaceBytes small,large,automatic,borrowed,with_handoff;
 assert(prefill::owned_workspace_quote(shape(16),small));
 assert(prefill::owned_workspace_quote(shape(1024),large));
 //8192 represents an ACTUALLY resolved auto shape supplied by producer, not proof of planner resolution.
 assert(prefill::owned_workspace_quote(shape(8192),automatic));
 assert(small.host<large.host&&large.host<automatic.host);
 assert(small.pinned==small.host&&large.pinned==large.host&&automatic.pinned==automatic.host);
 assert(prefill::owned_workspace_quote(shape(1024,true),borrowed));
 assert(borrowed.device==(2ULL<<20));assert(large.device-borrowed.device==(64ULL<<20));
 assert(borrowed.host<large.host); // borrowed arena has no owned-allocation pointer directory
 assert(prefill::owned_workspace_quote(shape(1024,false,true),with_handoff));
 assert(with_handoff.host>large.host&&with_handoff.device==large.device);
}
static void two_stage_ledger(){
 prefill::OwnedWorkspaceBytes q;assert(prefill::owned_workspace_quote(shape(1024,true),q));
 auto a=resources(q,0),b=resources(q,1),cap=zero();cap.device_bytes={q.device,q.device};cap.host_bytes=2*q.host;cap.pinned_bytes=2*q.pinned;
 core::WindowBudget ledger(cap,zero());
 auto first=ledger.reserve(a,core::ReservationPurpose::foreground);assert(first&&first.lease);
 auto second=ledger.reserve(b,core::ReservationPurpose::foreground);assert(second&&second.lease);
 auto snap=ledger.snapshot();assert(snap.reserved.host_bytes==2*q.host);assert(snap.reserved.pinned_bytes==2*q.pinned);
 assert(snap.reserved.device_bytes==cap.device_bytes); // pinned is subset: host is NOT host+pinned
 auto excess=ledger.reserve(a,core::ReservationPurpose::foreground);assert(!excess&&!excess.lease);
 first.lease.reset();assert(ledger.snapshot().reserved.host_bytes==q.host);
 second.lease.reset();assert(ledger.snapshot().reserved==zero());
 cap.host_bytes=2*q.host-1;cap.pinned_bytes=cap.host_bytes;core::WindowBudget short_host(cap,zero());
 auto one=short_host.reserve(a,core::ReservationPurpose::foreground);assert(one);
 auto no=short_host.reserve(b,core::ReservationPurpose::foreground);assert(!no&&no.error==core::ReservationError::host_limit);
 assert(short_host.snapshot().reserved==a);one.lease.reset();assert(short_host.snapshot().reserved==zero());
 cap.host_bytes=2*q.host;cap.pinned_bytes=2*q.pinned-1;core::WindowBudget short_pin(cap,zero());
 one=short_pin.reserve(a,core::ReservationPurpose::foreground);assert(one);
 no=short_pin.reserve(b,core::ReservationPurpose::foreground);assert(!no&&no.error==core::ReservationError::pinned_limit);
 one.lease.reset();assert(short_pin.snapshot().reserved==zero());
}
static void retained_lifetime(){
 prefill::OwnedWorkspaceBytes q;assert(prefill::owned_workspace_quote(shape(16),q));auto r=resources(q,0);
 core::WindowBudget ledger(r,zero());auto result=ledger.reserve(r,core::ReservationPurpose::foreground);assert(result);
 struct Payload{core::WindowBudget* ledger;~Payload(){assert(ledger->snapshot().reserved.host_bytes>0);}};
 struct Owner{std::shared_ptr<const core::ReservationLease> credit;std::unique_ptr<Payload> payload;};
 auto owner=std::make_shared<Owner>();owner->credit=result.lease;owner->payload=std::make_unique<Payload>();owner->payload->ledger=&ledger;
 result.lease.reset();auto retained=owner;owner.reset();assert(ledger.snapshot().reserved==r);
 retained.reset();assert(ledger.snapshot().reserved==zero());
}
static void actual_position_bank_footprint(){
 uint64_t cells=0,bytes=0;assert(kernels::owned_mrope_bank_layout(524288,cells,bytes));
 assert(cells==524352&&bytes==6292224&&3*bytes==18876672);
 const auto old_cells=cells,old_bytes=bytes;
 assert(!kernels::owned_mrope_bank_layout(0,cells,bytes)&&cells==old_cells&&bytes==old_bytes);
 assert(!kernels::owned_mrope_bank_layout(UINT64_MAX,cells,bytes)&&cells==old_cells&&bytes==old_bytes);
 core::WindowResources cap{{3*bytes,3*bytes},bytes,0,0,0,1};
 core::WindowBudget actual(cap,zero());
 auto parent=actual.reserve(core::WindowResources{{0,0},0,0,0,0,1},core::ReservationPurpose::private_work);assert(parent);
 auto quote=cap;quote.shadows=0;
 auto banks=actual.reserve(quote,core::ReservationPurpose::private_attachment,parent.lease);assert(banks);
 assert(actual.snapshot().reserved.device_bytes==cap.device_bytes);
 auto too_many=actual.reserve(quote,core::ReservationPurpose::private_attachment,parent.lease);assert(!too_many);
 banks.lease.reset();parent.lease.reset();assert(actual.snapshot().reserved==zero());
}
static void actual_image_configuration(){
 struct Profile {
  bool serve=true,owned_session_lifetime=true,owned_model_lifetime=true,suffix_cache=true;
  bool suffix_cache_unrepaired_experiment=false,batch_groups_auto=false,batch_mtp=false,keep_canonical=false;
  bool prefill_head_alias=false,vision=false,kv_grow=false,vram_elastic=false,adapt_async_self=false;
  bool spec_split=false,split_skip_if_fits=false;
  int suffix_cache_host_bytes=1,spec=0,batch=2,batch_groups=1,prefill_chunk=16,short_read=64;
  int spec_corrupt=0,suffix_draft=0,lookup_chain=0,pipeline_windows=0,kv_resident=0,prompt_cache=0;
  int adapt_every=0,adapt_async=0,adapt_warm_every=0,peer_device=-1;
  std::string native_preset="strict",mtp,draft,draft_model,spec_oracle,spec_follow,kv="int8",layer_split="24",split_device="1";
 } text;
 const auto accepted=[](const auto& p){return core::native_owned_batch_serving::eligible(true,p,true)||
     core::native_owned_batch_serving::eligible_image(true,p,true);};
 assert(core::native_owned_batch_serving::eligible(true,text,true));
 assert(!core::native_owned_batch_serving::eligible_image(true,text,true));
 auto image=text;image.vision=true;
 assert(!core::native_owned_batch_serving::eligible(true,image,true));
 assert(core::native_owned_batch_serving::eligible_image(true,image,true));
 assert(!core::native_owned_batch_serving::eligible_image(false,image,true));
 assert(!core::native_owned_batch_serving::eligible_image(true,image,false));
 for(bool is_image:{false,true}){auto base=text;base.vision=is_image;
  auto bad=base;bad.spec=1;assert(!accepted(bad));bad=base;bad.batch_groups=2;assert(!accepted(bad));
  bad=base;bad.prefill_chunk=0;assert(!accepted(bad));bad=base;bad.kv_grow=true;assert(!accepted(bad));
  bad=base;bad.prompt_cache=1;assert(!accepted(bad));bad=base;bad.pipeline_windows=1;assert(!accepted(bad));
  bad=base;bad.kv="fp16";assert(!accepted(bad));bad=base;bad.owned_model_lifetime=false;assert(!accepted(bad));
 }
}
int main(){actual_image_configuration();actual_position_bank_footprint();minimum_upload_footprint();refusal_preserves();shapes_and_borrowing();two_stage_ledger();retained_lifetime();
 std::cout<<"PASS owned prefill quote/real ledger/lifetime CPU contracts; no CUDA allocation or planner execution claim\n";}
