#include "serving_prefix_replay.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
using namespace strata::program;
static void check(bool v,const char* why){if(!v){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
static PrefixReplayPolicy policy(int64_t size,bool automatic=false,int64_t requested=0){
 PrefixReplayPolicy p{};p.configured_max=size;p.allocated_max=size;p.auto_requested_max=requested;p.short_read=64;p.automatic=automatic;return p;
}
static PrefixReplayPlan prepared(const PrefixReplayPolicy& p,uint64_t end=0){
 PrefixReplayPlan x;x.request=2;x.boundary=end?end:std::max<uint64_t>(2,64/uint64_t(p.configured_max)+1)*uint64_t(p.configured_max)+2;
 check(x.bind_policy(p),"valid authentic policy binds");
 std::vector<int32_t> mapping{0,1,2,3};
 check(x.prepare(std::make_shared<int>(1),mapping),"paid plan prepares");
 x.participant_count=2;
 for(size_t i=0;i<2;++i){auto& s=x.participants[i];s.device=int(i);s.begin=int64_t(i);s.end=int64_t(i+1);s.slots=4;s.chunk=p.configured_max;s.allocated_chunk=p.allocated_max;s.ring_slots=2;}
 return x;
}
static void finish(PrefixReplayPlan& p,int64_t configured,uint64_t cycles=0){
 const uint64_t full=p.boundary-2;
 for(uint64_t at=0;at<full;){
  const uint64_t to=std::min<uint64_t>(full,at+uint64_t(configured));
  PrefixReplayEvent first;first.kind=PrefixReplayEvent::Kind::prefill;first.from=at;first.to=to;first.chunk=int64_t(to-at);first.chunked=true;first.lend=true;first.returned_before_lend=cycles!=0;first.loan_cycles=cycles;
  check(p.record(first),"actual full partition records");at=to;
 }
 PrefixReplayEvent tail;tail.kind=PrefixReplayEvent::Kind::window;tail.from=full;tail.to=p.boundary;tail.single=true;
 check(p.record(tail),"actual short tail records");
 std::vector<int32_t> history(size_t(p.boundary),7);check(p.seal(history),"actual partition seals");check(p.matches(history,2,p.boundary),"sealed actual history matches");
}
int main(){
 for(const auto& p:{policy(16),policy(1024),policy(8192,true,8192)}){
  auto x=prepared(p);finish(x,p.configured_max);check(x.policy_matches(p),"same policy matches");
  auto other=p;other.allocated_max+=1;check(!x.policy_matches(other),"foreign resolved policy refused");
  auto digest=x.digest;x.participants[1].ring_slots+=1;std::vector<int32_t> h(size_t(x.boundary),7);check(!x.matches(h,2,x.boundary),"stage ring digest tamper refused");x.participants[1].ring_slots-=1;
  x.participants[1].allocated_chunk+=1;check(!x.matches(h,2,x.boundary),"allocated stage digest tamper refused");x.participants[1].allocated_chunk-=1;check(x.digest==digest,"original digest immutable");
 }
 {PrefixReplayPlan x;x.boundary=130;std::vector<int32_t> m{0};check(!x.prepare(std::make_shared<int>(1),m),"unbound active plan refused");}
 for(int which=0;which<4;++which){auto p=policy(16);if(which==0)p.allocated_max=8;if(which==1)p.short_read=63;if(which==2)p.auto_requested_max=16;if(which==3){p.automatic=true;p.auto_requested_max=8;}PrefixReplayPlan x;check(!x.bind_policy(p),"misresolved policy refused");}
 {auto x=prepared(policy(1024));PrefixReplayEvent e;e.kind=PrefixReplayEvent::Kind::prefill;e.from=0;e.to=130;e.returned_before_lend=true;e.loan_cycles=1;e.lend=false;check(!x.record(e),"return without actual relend refused");}
 {auto x=prepared(policy(1024));finish(x,1024,true);x.events[0].returned_before_lend=false;std::vector<int32_t> h(size_t(x.boundary),7);check(!x.matches(h,2,x.boundary),"return marker digest tamper refused");}
 {auto x=prepared(policy(16));PrefixReplayEvent e;e.kind=PrefixReplayEvent::Kind::prefill;e.from=0;e.to=17;e.chunk=17;e.lend=true;check(!x.record(e),"chunk exceeding actual allocation refused");}
 {auto x=prepared(policy(1024));finish(x,1024,3);check(x.events[0].loan_cycles==3,"actual three-cycle schedule retained");x.events[0].loan_cycles=2;std::vector<int32_t> h(size_t(x.boundary),7);check(!x.matches(h,2,x.boundary),"cycle count digest tamper refused");}
 {auto x=prepared(policy(1024));PrefixReplayEvent e;e.kind=PrefixReplayEvent::Kind::prefill;e.from=0;e.to=128;e.chunk=128;e.lend=true;e.returned_before_lend=false;e.loan_cycles=1;check(!x.record(e),"cycle without return marker refused");}
 {auto x=prepared(policy(1024));PrefixReplayEvent e;e.kind=PrefixReplayEvent::Kind::prefill;e.from=0;e.to=128;e.chunk=128;e.lend=true;e.returned_before_lend=true;e.loan_cycles=0;check(!x.record(e),"return marker without cycle refused");}
 {PrefixReplayPlan x;x.request=1;x.boundary=4;PrefixReplayEvent e;e.from=0;e.to=4;check(x.record(e),"wholly T1 no policy records");std::vector<int32_t> h(4,7);check(x.seal(h),"wholly T1 inactive remains supported");}
 std::puts("PASS actual prefix policy/participant/return transcript contract");
}
