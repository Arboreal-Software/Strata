// Real core registry/graph + program selector, host-only. No model or device API.
#include "serving_thread_bridge.hpp"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cstring>
#include <new>
using namespace strata;
static long fail_after=-1;
void* operator new(std::size_t size){
    if(fail_after==0)throw std::bad_alloc();
    if(fail_after>0)--fail_after;
    if(void* p=std::malloc(size?size:1))return p;throw std::bad_alloc();
}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
static void check(bool ok,const char* message){if(!ok){std::cerr<<message<<'\n';std::abort();}}
struct World {
    core::WindowBudget budget{core::WindowResources{{1},1ULL<<30},core::WindowResources{{0}}};
    program::ServingThreadLimits limits{{-100,100,100,10000,8,4,32},{32,64},1024,8,8<<20};
    std::unique_ptr<program::ServingThreadBridge> bridge;
    uint64_t sequence=0,now=0;
    World(){auto lease=budget.reserve(core::WindowResources{{0},limits.metadata_bytes},core::ReservationPurpose::foreground);
        check(bool(lease),"real metadata reservation");bridge=std::make_unique<program::ServingThreadBridge>(99,limits,std::move(lease.lease));}
    program::ServingThreadReply call(const std::string& suffix,const char* status="ok"){
        auto r=bridge->command("SCHED 1 99 "+std::to_string(++sequence)+" "+suffix,now);
        check(std::string_view(r.status)==status,suffix.c_str());return r;
    }
    uint64_t graph(){check(bridge->refresh(now),"refresh actual core graph");return call("SNAPSHOT t 1").graph_revision;}
    core::OperationKey open(std::string id,uint64_t gen,const char* kind,uint64_t revision=1){
        auto r=call("OPEN "+id+" "+std::to_string(gen)+" "+std::to_string(revision)+" "+kind);
        return {99,r.operation,r.result_revision};
    }
    void edge(core::OperationKey a,core::OperationKey b,bool add=true,const char* status="ok"){
        call("EDGE "+std::to_string(graph())+" "+std::to_string(a.serial)+" "+std::to_string(a.result_revision)+" "+
             std::to_string(b.serial)+" "+std::to_string(b.result_revision)+" "+(add?"1":"0"),status);
    }
    void cancel(core::OperationKey a){call("CANCEL "+std::to_string(graph())+" "+std::to_string(a.serial)+" "+std::to_string(a.result_revision));}
    void bind(size_t slot,const char* request,core::OperationKey fg,core::OperationKey bg,const char* status="ok"){
        call("BIND "+std::to_string(slot)+" "+request+" "+std::to_string(fg.serial)+" "+std::to_string(fg.result_revision)+" "+
             std::to_string(bg.serial)+" "+std::to_string(bg.result_revision),status);
    }
};
static program::ServingRepairSelectionBoundary boundary(){
    program::ServingRepairSelectionBoundary b;b.safe_boundary=true;b.max_candidates=4;b.clock_domain=7;b.now_ns=100;
    b.foreground.context_room=true;
    b.max_consecutive_escalated_units=2;return b;
}
static program::ServingThreadUnit unit(size_t slot,core::OperationKey op,bool repair,uint64_t weight){
    program::ServingThreadUnit u;u.slot=slot;u.request=slot+1;u.occupancy=10+slot;u.operation=op;
    u.unit=repair?core::AdmissionUnit::repair:core::AdmissionUnit::decode;u.demand=u.resource_ready=true;u.class_weight=weight;
    u.repair.age.oldest_unresolved_inherited_age_ns=100;u.repair.observed_at_ns=100;u.repair.clock_domain=7;
    u.repair.record_version=1;u.repair.ancestry_serial=1;return u;
}
int main(int argc,char** argv){
    if(argc==2&&std::string_view(argv[1])=="--os-epoch"){
        const auto epoch=program::serving_thread_epoch();check(epoch!=0,"real nonblocking OS nonce producer");
        std::cout<<epoch<<'\n';return 0;
    }
    check(program::serving_thread_epoch_from_os([](void* p,size_t n){const uint64_t v=12345;std::memcpy(p,&v,n);return n;})==12345,"injected exact OS nonce");
    check(program::serving_thread_epoch_from_os([](void*,size_t){return size_t(0);})==0,"OS entropy error refuses");
    check(program::serving_thread_epoch_from_os([](void*,size_t n){return n-1;})==0,"short OS entropy refuses");
    check(program::serving_thread_epoch_from_os([](void* p,size_t n){std::memset(p,0,n);return n;})==0,"zero OS nonce refuses");
    check(!program::parse_serving_thread_config(nullptr,nullptr).enabled,"default off");
    check(!program::parse_serving_thread_config(program::kServingThreadPolicy,nullptr).valid,"partial config refused");
    const auto cfg=program::parse_serving_thread_config(program::kServingThreadPolicy,"-100,100,100,10000,8,4,32,32,64,1024,8,8388608");
    check(cfg.enabled&&cfg.valid,"explicit policy and limits");
    for(const char* bad:{"-100,100,100,10000,8,4,32,32,64,1024,8,1","-100,100,100,10000,8,4,32,32,64,1024,08,8388608"})
        check(!program::parse_serving_thread_config(program::kServingThreadPolicy,bad).valid,"insufficient/noncanonical limits refuse");
    World w;auto t=w.call("CREATE t 0");auto other=w.call("CREATE other 0");
    check(t.generation==1&&other.generation==2&&t.accepted==1&&t.applied==0,"managed identities and acceptance separate");
    auto fg=w.open("t",1,"foreground"),bg=w.open("t",1,"live_repair");
    auto fg2=w.open("other",2,"foreground"),bg2=w.open("other",2,"live_repair");
    w.bind(0,"000000000000",fg,bg);w.bind(1,"000000000001",fg,bg,"not_owned");
    w.bind(1,"000000000001",fg2,bg,"not_owned");w.bind(1,"000000000001",fg2,bg2);
    check(w.bridge->request_ready(0,1),"bound queued request ready");
    check(w.bridge->attach(0,1,10)&&w.bridge->attach(1,2,11),"actual caller occupancy attached");
    check(!w.bridge->attach(0,1,10),"cannot reattach/steal occupied request");
    auto b=boundary();program::ServingRepairAdmissionPolicy policy;policy.live_repair_age_escalation_ns=50;
    std::array units{unit(0,fg,false,2),unit(1,fg2,false,5),unit(0,bg,true,3),unit(1,bg2,true,7)};
    std::array<size_t,4> counts{};
    for(uint64_t draw=0;draw<17;++draw){auto s=w.bridge->select(policy,b,units,w.now,draw);check(s.ready,"same tier draw");++counts[s.candidate];}
    check(counts==std::array<size_t,4>{2,5,3,7},"same-tier true class-weighted lottery");
    auto winner=w.bridge->select(policy,b,units,w.now,0);
    auto peer=w.bridge->foreground_peer(winner,units[1]);
    check(peer.ready&&peer.operation.key==fg2,"same-tier shared foreground cohort includes actual peer");
    check(w.bridge->acknowledge_unit(winner,units[0],10,40)&&
          w.bridge->acknowledge_unit(peer,units[1],11,41),"ack each actual shared-window member cut");
    w.call("BASE t 1 1 10");auto s=w.bridge->select(policy,b,units,w.now,0);
    check(s.ready&&s.candidate==0&&s.total_weight==5&&s.effective_priority==10,"higher effective tier excludes lower tier");
    check(!w.bridge->foreground_peer(s,units[1]).ready,"lower-tier peer excluded from shared cohort");
    w.call("BASE t 1 2 -10");s=w.bridge->select(policy,b,units,w.now,0);
    check(s.ready&&s.candidate==1&&s.total_weight==12,"base decrease applies future physical selection");
    auto actor=w.call("ACTOR t 1 user-a 3");
    auto boosted=w.call("BOOST t 1 user-a "+std::to_string(actor.incarnation)+" "+std::to_string(actor.lease_version)+" 30 5");
    check(boosted.effective==20&&boosted.boost==30&&boosted.applied<boosted.accepted,"base distinct from activity/effect/application");
    s=w.bridge->select(policy,b,units,w.now,0);check(s.candidate==0,"boost changes selected tier");
    check(!w.bridge->acknowledge_unit(s,units[0],999,1),"wrong actual occupancy cannot acknowledge");
    check(w.bridge->acknowledge_unit(s,units[0],10,42),"selected actual unit acknowledgement");
    auto snap=w.call("SNAPSHOT t 1");check(snap.applied==snap.accepted&&snap.cut==3,"applied only after actual-unit seam");
    w.now=5;check(w.bridge->refresh(w.now),"idle timer refresh without request");
    snap=w.call("SNAPSHOT t 1");check(snap.effective==-10&&snap.accepted>snap.applied,"expiry visible independently of applied cut");
    w.call("BOOST t 1 user-a "+std::to_string(actor.incarnation)+" "+std::to_string(boosted.lease_version)+" 30 5","stale_version");
    auto expired=w.call("ACTOR_SNAPSHOT t 1 user-a "+std::to_string(actor.incarnation));
    check(expired.lease_version==boosted.lease_version+1,"expiry CAS observed through managed actor snapshot");
    auto waiter=w.open("other",2,"foreground",2);w.call("BASE other 2 1 40");w.edge(waiter,bg);
    s=w.bridge->select(policy,b,units,w.now,5);
    check(s.ready&&s.candidate==2&&s.effective_priority==40&&s.total_weight==15,
          "real inherited prerequisite has positive lottery tickets in shared highest tier");
    w.cancel(waiter);s=w.bridge->select(policy,b,units,w.now,0);
    check(s.ready&&s.candidate==1,"waiter cancellation removes inherited priority");
    // Added prerequisites cannot race successful physical BIND. Removals remain
    // valid; new dependencies are configured while operations remain unbound.
    w.edge(fg2,bg,true,"not_ready");
    auto second_waiter=w.open("other",2,"foreground",3);w.edge(second_waiter,bg);
    auto young=units;young[2].repair.age.oldest_unresolved_inherited_age_ns=0;
    s=w.bridge->select(policy,b,young,w.now,0);check(!s.ready||s.candidate!=2,"inherited priority cannot promote young repair past active foreground");
    auto unknown=units;unknown[2].repair.age.oldest_unresolved_inherited_age_ns.reset();
    s=w.bridge->select(policy,b,unknown,w.now,0);check(!s.ready||s.candidate!=2,"unknown age cannot inherit escalation");
    auto capped=b;capped.consecutive_escalated_units=2;
    s=w.bridge->select(policy,capped,units,w.now,0);check(s.ready&&s.candidate==1,"hard consecutive cap stronger than inherited tier");
    w.cancel(bg);check(w.bridge->refresh(w.now)&&w.bridge->cancellation_pending(0)&&
        w.bridge->operation(second_waiter)->state==core::DependencyState::dependency_failed,"cancel propagates to actual graph consumer without discarding native binding");
    check(w.bridge->binding(0)&&w.bridge->binding(0)->occupancy==10,"graph terminal is not physical retirement");
    check(w.bridge->release(0,1,10,true,w.now),"actual caller retirement reclaims bound pair");
    auto close_view=w.call("SNAPSHOT other 2");
    w.call("CLOSE other 2 "+std::to_string(close_view.accepted));
    w.call("CREATE other 0","not_ready");
    check(w.bridge->binding(1)&&w.bridge->cancellation_pending(1),"closed identity retained while actual peer owner remains");
    check(w.bridge->release(1,2,11,true,w.now),"closed owner actually retired before same-id reuse");
    auto recreated=w.call("CREATE other 0");check(recreated.generation>2,"same id safely reminted after actual retirement");
    check(w.call("SNAPSHOT t 1").base==-10,"persistent thread base survives request/operation retirement");
    World blocked;blocked.call("CREATE t 50");blocked.call("CREATE other 0");
    auto consumer=blocked.open("t",1,"foreground"),consumer_bg=blocked.open("t",1,"live_repair");
    auto prerequisite=blocked.open("other",2,"foreground"),prerequisite_bg=blocked.open("other",2,"live_repair");
    blocked.edge(consumer,prerequisite);
    blocked.edge(consumer_bg,prerequisite_bg);
    blocked.bind(0,"000000000000",consumer,consumer_bg,"not_ready");
    check(!blocked.bridge->binding(0)&&!blocked.bridge->request_ready(0,1),"blocked BIND publishes no authority; cannot enter prefill");
    blocked.bind(1,"000000000001",prerequisite,prerequisite_bg);
    check(blocked.bridge->attach(1,2,11)&&blocked.bridge->foreground_finished(1,false,0),"prerequisite actual foreground milestone");
    blocked.bind(0,"000000000000",consumer,consumer_bg,"not_ready");
    check(!blocked.bridge->binding(0),"blocked repair cannot pin a physical slot after foreground readiness");
    check(blocked.bridge->release(1,2,11,false,0),"actual prerequisite repair retirement resolves background result");
    blocked.bind(0,"000000000000",consumer,consumer_bg);
    check(blocked.bridge->request_ready(0,1),"new BIND succeeds after real prerequisite completion");
    program::ServingThreadUnit prefill;prefill.slot=0;prefill.request=1;prefill.operation=consumer;
    prefill.unit=core::AdmissionUnit::prefill;prefill.demand=prefill.resource_ready=true;prefill.class_weight=2;
    auto prefill_selected=blocked.bridge->select(policy,b,std::span(&prefill,1),0,0);
    std::array<int64_t,2> stage_ends{16,16};program::ServingThreadPrefillBoundary proof;
    proof.source_window_generation=9;proof.source_cut=0;proof.source_bootstrap=true;
    proof.from=0;proof.to=16;proof.stage_forwarded=stage_ends;
    proof.pipeline_joined=proof.devices_drained=proof.request_fence_open=true;
    check(blocked.bridge->acknowledge_prefill(prefill_selected,prefill,proof),"genuine zero bootstrap anchor distinct from admitted executor cut");
    stage_ends[1]=15;check(!blocked.bridge->acknowledge_prefill(prefill_selected,prefill,proof),"unmatched stage endpoints cannot acknowledge next chunk");
    stage_ends[1]=16;proof.pipeline_joined=false;
    check(!blocked.bridge->acknowledge_prefill(prefill_selected,prefill,proof),"unjoined prefill callback owners cannot acknowledge");
    World replay_world;auto r=replay_world.call("CREATE t 0");
    auto exact=replay_world.bridge->command("SCHED 1 99 1 CREATE t 0",0);
    check(exact.json()==r.json(),"bounded exact latest replay");
    check(std::string_view(replay_world.bridge->command("SCHED 1 99 1 CREATE t 1",0).status)=="stale_sequence","changed replay refused");
    check(std::string_view(replay_world.bridge->command("SCHED 1 98 2 CREATE t 1",0).status)=="stale_engine","restart epoch fenced");
    replay_world.call("BASE t 1 999 2","stale_version");replay_world.call("SNAPSHOT t 1");
    for(unsigned i=0;i<32;++i){const auto name="closed-"+std::to_string(i);
        auto created=replay_world.call("CREATE "+name+" 0");
        replay_world.call("CLOSE "+name+" "+std::to_string(created.generation)+" "+std::to_string(created.accepted));
    }
    check(replay_world.bridge->refresh(0),"managed closed-thread slots reclaimed across bounded churn");
    // Sweep actual wrapper+core allocations. Failure must not publish a BIND,
    // base change or CREATE/OPEN; successful post-commit replies are fixed data.
    for(const char* verb:{"BASE","CREATE","OPEN","BIND"}){
        bool saw_failure=false,saw_success=false;
        for(long site=0;site<160;++site){World a;a.call("CREATE t 0");
            auto f=a.open("t",1,"foreground"),g=a.open("t",1,"live_repair");
            std::string suffix;
            if(std::string_view(verb)=="BASE")suffix="BASE t 1 1 20";
            if(std::string_view(verb)=="CREATE")suffix="CREATE fresh-allocation-thread 20";
            if(std::string_view(verb)=="OPEN")suffix="OPEN t 1 99 foreground";
            if(std::string_view(verb)=="BIND")suffix="BIND 0 000000000000 "+std::to_string(f.serial)+" 1 "+std::to_string(g.serial)+" 1";
            auto frame="SCHED 1 99 "+std::to_string(++a.sequence)+" "+suffix;
            program::ServingThreadReply result;bool threw=false;
            fail_after=site;try{result=a.bridge->command(frame,0);}catch(const std::bad_alloc&){threw=true;}fail_after=-1;
            check(a.bridge->refresh(0),"post-failure state observable");
            const bool success=!threw&&std::string_view(result.status)=="ok";
            if(success)saw_success=true;else saw_failure=true;
            if(std::string_view(verb)=="BIND")check(bool(a.bridge->binding(0))==success,"allocation refusal cannot hide published BIND");
            else if(std::string_view(verb)=="OPEN")check(bool(a.bridge->operation({99,3,99}))==success,"allocation refusal cannot hide committed operation");
            else {
                // Prepared-frame allocation can fail before sequence consumption.
                // Snapshot at seq+1 first, then the unconsumed seq if needed.
                const auto name=std::string_view(verb)=="CREATE"?"fresh-allocation-thread 2":"t 1";
                auto query=a.bridge->command("SCHED 1 99 "+std::to_string(a.sequence+1)+" SNAPSHOT "+name,0);
                if(std::string_view(query.status)=="stale_sequence")query=a.bridge->command("SCHED 1 99 "+std::to_string(a.sequence)+" SNAPSHOT "+name,0);
                if(std::string_view(verb)=="BASE")check(query.base==(success?20:0),"allocation refusal cannot hide committed base edit");
                else check((std::string_view(query.status)=="ok")==success,"allocation refusal cannot hide committed thread creation");
            }
            if(saw_failure&&saw_success)break;
        }
        check(saw_failure&&saw_success,"allocation sweep reaches both refused and committed paths");
    }
    std::cout<<"serving thread bridge host contracts PASS\n";
}
