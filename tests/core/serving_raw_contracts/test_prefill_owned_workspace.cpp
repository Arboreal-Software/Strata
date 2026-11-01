// The checks include real quote/reservation calls and must execute in Release.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cstddef>
#include "strata/prefill/owned_workspace.hpp"
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
 prefill::OwnedWorkspaceBytes minimum,larger_min;
 assert(prefill::owned_workspace_quote(s,minimum));
 s.upload_min_bytes=131072;assert(prefill::owned_workspace_quote(s,larger_min));
 assert(larger_min.host-minimum.host==4ULL*65536);
 assert(larger_min.pinned-minimum.pinned==4ULL*65536);
 assert(larger_min.device==minimum.device); // ring minimum is host/pinned, never borrowed arena bytes
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
int main(){minimum_upload_footprint();refusal_preserves();shapes_and_borrowing();two_stage_ledger();retained_lifetime();
 std::cout<<"PASS owned prefill quote/real ledger/lifetime CPU contracts; no CUDA allocation or planner execution claim\n";}
