#include "strata/core/thread_priority.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <sstream>
#include <new>
#include <type_traits>

// Persistent per-thread failure: once reached, every later allocation fails
// until the fixture disables it. Other concurrency-fixture threads stay normal.
static thread_local long allocation_failure_after=-1;
void* operator new(size_t n){if(allocation_failure_after==0)throw std::bad_alloc();if(allocation_failure_after>0)--allocation_failure_after;if(auto* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,size_t)noexcept{std::free(p);}
void operator delete[](void* p,size_t)noexcept{std::free(p);}
static_assert(std::is_nothrow_move_constructible_v<strata::core::PriorityResult>);

using namespace strata::core;
namespace strata::core {
struct ThreadPriorityTestAccess {
    static void snapshot(std::ostream& o,const ThreadPrioritySnapshot& v){
        o<<v.key.engine_generation<<','<<v.key.generation<<','<<v.key.id.size()<<':'<<v.key.id<<','<<v.base<<','<<v.local_effective<<','<<v.base_version<<','<<v.accepted_revision<<','<<v.applied_revision<<','<<v.applied_cut<<','<<v.closed<<';';
        for(const auto& a:v.activity)actor(o,a);
    }
    static void actor(std::ostream& o,const ActivityPriority& a){o<<a.lease_id.size()<<':'<<a.lease_id<<','<<a.version<<','<<a.expires_at_ms<<','<<a.boost<<','<<a.active<<','<<a.incarnation<<';';}
    static std::string state(const ThreadPriorities& r){std::ostringstream o;o<<r.engine_generation_<<','<<r.now_ms_<<','<<r.next_expiry_ms_<<','<<r.expiry_pending_<<','<<r.retired_generation_floor_<<','<<r.highest_thread_generation_<<','<<r.next_actor_incarnation_<<';';
        for(const auto& [id,t]:r.threads_){o<<id.size()<<':'<<id<<','<<t.registered_base<<','<<t.managed_actors<<';';snapshot(o,t.view);for(const auto& [name,a]:t.leases){o<<name.size()<<':'<<name<<';';actor(o,a);}}return o.str();}
    static std::string response(const PriorityResult& r){std::ostringstream o;o<<int(r.error)<<';';snapshot(o,r.snapshot);return o.str();}
    static void last_actor(ThreadPriorities& r){r.next_actor_incarnation_=UINT64_MAX;}
    static void last_revision(ThreadPriorities& r,const ThreadKey& key){r.find(key)->view.accepted_revision=UINT64_MAX;}
};
}
namespace {
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
PriorityLimits limits() { return {-100,100,50,1000,4,2,64}; }
ThreadKey first{1,1,"first"}, second{1,1,"second"};
}
int main() {
    {
        ThreadPriorities registry(1,limits());
        auto r = registry.register_thread(first,10,0);
        check(bool(r) && r.snapshot.accepted_revision == 1 && r.snapshot.applied_revision == 0,"registration is accepted not applied");
        check(bool(registry.register_thread(second,20,0)),"second isolated user thread registered");
        r = registry.set_activity(first,"user-a",0,30,100,1);
        check(bool(r) && r.snapshot.base == 10 && r.snapshot.local_effective == 40,"activity adds separate temporary boost");
        check(registry.snapshot(second,1).snapshot.local_effective == 20,"activity does not affect another thread");
        check(registry.acknowledge(first,1,1,1).error == PriorityError::invalid_acknowledgement,"outdated executor revision rejected");
        r = registry.acknowledge(first,2,1,1);
        check(bool(r) && r.snapshot.applied_revision == 2,"safe-cut application separately acknowledged");
        r = registry.set_activity(first,"user-b",0,20,200,2);
        check(bool(r) && r.snapshot.local_effective == 40,"concurrent users use max boost not additive stacking");
        r = registry.set_base(first,1,-10,3);
        check(bool(r) && r.snapshot.base == -10 && r.snapshot.local_effective == 20,"base decrease persists under active boost");
        check(r.snapshot.applied_revision == 2 && r.snapshot.accepted_revision == 4,"accepted change does not preempt active work");
        check(registry.acknowledge(first,4,1,3).error == PriorityError::invalid_acknowledgement,
              "new revision cannot reuse earlier application cut");
        check(bool(registry.acknowledge(first,4,2,3)),"new revision requires a later executor boundary");
        check(bool(registry.acknowledge(first,4,2,3)),"same-revision acknowledgement is idempotent");
        check(registry.set_base(first,1,90,3).error == PriorityError::stale_version,"stale UI base update cannot overwrite");
        r = registry.disconnect(first,"user-a",1,4);
        check(bool(r) && r.snapshot.local_effective == 10,"disconnect removes only that user's contribution");
        check(registry.set_activity(first,"user-a",1,50,100,5).error == PriorityError::stale_version,
              "old refresh cannot revive disconnected activity lease");
        check(registry.advance_time(202),"control timer advances expiry without another request");
        r = registry.snapshot(first,202);
        check(r.snapshot.base == -10 && r.snapshot.local_effective == -10,"all expired boosts leave edited base intact");
        check(registry.set_activity(first,"user-b",1,20,100,202).error == PriorityError::stale_version,
              "old refresh cannot revive expired activity lease");
        check(registry.set_activity(first,"user-c",0,10,100,202).error == PriorityError::limit_exceeded,
              "bounded activity tombstones prevent stale-lease ABA");
        check(registry.set_activity(first,"user-a",2,10,1001,202).error == PriorityError::invalid_priority,"TTL is bounded");
        check(registry.snapshot(first,201).error == PriorityError::invalid_time,"clock rollback cannot extend activity");
        r = registry.close(first,r.snapshot.accepted_revision,203);
        check(bool(r) && r.snapshot.closed,"thread closed with expected revision");
        auto newer = first; ++newer.generation;
        check(bool(registry.register_thread(newer,5,203)),"thread ID reused only with newer generation");
        check(registry.set_base(first,2,0,203).error == PriorityError::unknown_generation,"old generation cannot mutate new thread");
        auto wrong_engine = newer; ++wrong_engine.engine_generation;
        check(registry.snapshot(wrong_engine,203).error == PriorityError::unknown_generation,"prior/foreign engine fenced");
    }
    {
        ThreadPriorities registry(1,limits());
        check(bool(registry.register_thread(first,0,0)),"concurrency fixture registered");
        std::atomic<int> accepted{0}, stale{0};
        std::thread a([&]{auto r=registry.set_base(first,1,10,0); if(r)++accepted; else if(r.error==PriorityError::stale_version)++stale;});
        std::thread b([&]{auto r=registry.set_base(first,1,20,0); if(r)++accepted; else if(r.error==PriorityError::stale_version)++stale;});
        a.join(); b.join();
        check(accepted==1 && stale==1,"concurrent base updates have one atomic CAS winner");
        check(registry.snapshot(first,0).snapshot.base_version==2,"base version increments once");
    }
    {
        ThreadPriorities registry(1,limits());
        check(bool(registry.register_thread(first,0,UINT64_MAX-2)),"late-clock fixture");
        check(registry.set_activity(first,"user",0,10,3,UINT64_MAX-2).error==PriorityError::invalid_priority,
              "expiry addition overflow rejected");
        check(bool(registry.set_activity(first,"user",0,10,2,UINT64_MAX-2)),"exact maximum deadline accepted");
        check(registry.snapshot(first,UINT64_MAX-1).snapshot.local_effective==10,"deadline does not expire early");
        check(registry.snapshot(first,UINT64_MAX).snapshot.local_effective==0,"maximum deadline still expires");
    }
    {
        ThreadPriorities registry(1,limits());
        auto first_managed=registry.create_thread("managed",10,0);check(bool(first_managed),"managed generation minted");
        auto key=first_managed.snapshot.key;
        auto r=registry.register_activity(key,"tab",1,0);check(bool(r),"managed actor registered");
        ActivityKey actor{key,"tab",r.snapshot.activity.at(0).incarnation};
        check(actor.incarnation!=0,"actor incarnation observable");
        check(bool(registry.set_activity(actor,1,20,10,0)),"managed activity refresh");
        check(registry.set_activity(key,"tab",2,20,10,0).error==PriorityError::stale_version,"legacy API cannot mutate managed actor");
        check(registry.retire_activity(actor,2,0).error==PriorityError::invalid_priority,"active actor cannot be retired");
        check(bool(registry.disconnect(actor,2,0)),"managed actor disconnected");
        check(bool(registry.retire_activity(actor,3,0)),"inactive actor slot reclaimed");
        check(registry.set_activity(key,"tab",0,20,10,0).error==PriorityError::stale_version,"legacy zero-version resurrection refused");
        r=registry.snapshot(key,0);r=registry.register_activity(key,"tab",r.snapshot.accepted_revision,0);check(bool(r),"same actor name gets new incarnation");
        ActivityKey replacement{key,"tab",r.snapshot.activity.at(0).incarnation};
        check(replacement.incarnation>actor.incarnation,"new actor incarnation monotonic");
        check(registry.set_activity(actor,1,50,10,0).error==PriorityError::unknown_generation&&registry.disconnect(actor,1,0).error==PriorityError::unknown_generation,"old actor cannot alter recreated name");
        check(bool(registry.retire_activity(replacement,1,0)),"unused managed actor retired");
        for(int i=0;i<1000;++i){r=registry.snapshot(key,0);r=registry.register_activity(key,"tab",r.snapshot.accepted_revision,0);check(bool(r),"actor slot lifetime reuse");
            ActivityKey next{key,"tab",r.snapshot.activity.at(0).incarnation};check(bool(registry.retire_activity(next,1,0)),"actor slot retirement");}
        ThreadPriorityTestAccess::last_actor(registry);
        r=registry.snapshot(key,0);r=registry.register_activity(key,"last",r.snapshot.accepted_revision,0);check(bool(r)&&r.snapshot.activity.at(0).incarnation==UINT64_MAX,"last actor incarnation accepted");
        check(registry.register_activity(key,"overflow",r.snapshot.accepted_revision,0).error==PriorityError::limit_exceeded&&registry.lifecycle().next_actor_incarnation==0,"actor exhaustion refuses wrap");
        r=registry.close(key,r.snapshot.accepted_revision,0);check(bool(r)&&bool(registry.retire_closed_thread(key,r.snapshot.accepted_revision,0)),"closed thread record reclaimed");
        check(registry.register_thread(key,10,0).error==PriorityError::unknown_generation,"retired thread create cannot resurrect");
    }
    {
        ThreadPriorities registry(1,limits());
        ThreadKey low{1,2,"low"},high{1,10,"high"},live{1,3,"live"};
        check(bool(registry.register_thread(low,0,0))&&bool(registry.register_thread(high,0,0))&&bool(registry.register_thread(live,0,0)),"out-of-order generations registered");
        auto r=registry.close(high,1,0);check(bool(registry.retire_closed_thread(high,r.snapshot.accepted_revision,0)),"high generation retires first");
        r=registry.close(low,1,0);check(bool(registry.retire_closed_thread(low,r.snapshot.accepted_revision,0)),"lower generation retires later");
        check(registry.lifecycle().retired_generation_floor==10,"retirement floor never regresses");
        check(bool(registry.snapshot(live,0)),"unrelated live older generation survives floor");
        check(registry.register_thread({1,9,"delayed"},0,0).error==PriorityError::unknown_generation,"unrelated delayed unregistered creation refused");
        for(int i=0;i<1000;++i){r=registry.create_thread("churn",0,0);check(bool(r)&&r.snapshot.key.generation>10,"thread generation minted above floor");
            auto k=r.snapshot.key;r=registry.close(k,r.snapshot.accepted_revision,0);check(bool(registry.retire_closed_thread(k,r.snapshot.accepted_revision,0)),"thread capacity reclaimed");}
        ThreadKey last{1,UINT64_MAX,"last"};check(bool(registry.register_thread(last,0,0)),"last external generation admitted");
        check(registry.create_thread("overflow",0,0).error==PriorityError::limit_exceeded&&registry.lifecycle().next_thread_generation==0,"thread mint exhaustion refuses wrap");
    }
    {
        ThreadPriorities registry(1,limits());auto r=registry.register_thread(first,0,0);
        r=registry.set_activity(first,"legacy",0,1,1,0);check(bool(r),"legacy migration fixture");
        r=registry.snapshot(first,1);check(bool(registry.retire_activity({first,"legacy",0},2,1)),"expired legacy tombstone reclaimed");
        check(registry.set_activity(first,"legacy",0,1,1,1).error==PriorityError::stale_version,"legacy migration permanently fences absent version zero");
    }
    {
        // No dependency graph or operation exists: reactor still gets every actor's deadline.
        ThreadPriorities registry(1,limits());
        check(!registry.next_activity_expiry_ms(),"empty registry has no activity timer");
        auto r=registry.create_thread("orphan",5,0);const auto key=r.snapshot.key;
        r=registry.register_activity(key,"actor",r.snapshot.accepted_revision,0);
        const ActivityKey actor{key,"actor",r.snapshot.activity.at(0).incarnation};
        check(!registry.next_activity_expiry_ms(),"registered inactive actor has no timer");
        check(bool(registry.set_activity(actor,1,20,10,0)),"no-operation actor activated");
        check(registry.next_activity_expiry_ms()==10,"no-operation actor deadline visible");
        check(bool(registry.set_activity(actor,2,20,100,1)),"actor refresh extends actual expiry");
        check(registry.next_activity_expiry_ms()==10,"refresh may retain earlier conservative wake");
        check(registry.advance_time(10)&&registry.next_activity_expiry_ms()==101,"early wake recomputes actual refreshed deadline");
        r=registry.snapshot(key,10);
        check(r.snapshot.local_effective==25&&r.snapshot.activity.at(0).active,"early timer does not expire refresh");
        const auto accepted=r.snapshot.accepted_revision;
        check(!registry.advance_time(9)&&registry.next_activity_expiry_ms()==101,"backwards timer rejected without deadline change");
        check(registry.snapshot(key,10).snapshot.accepted_revision==accepted,"deadline accessor does not accept a revision");
        check(bool(registry.disconnect(actor,3,11)),"actor disconnect before deadline");
        check(registry.next_activity_expiry_ms()==101,"disconnect permits conservative early wake");
        check(registry.advance_time(101)&&!registry.next_activity_expiry_ms(),"disconnect wake clears absent expiry");
        check(bool(registry.set_activity(actor,4,20,50,101)),"reactivate after disconnected timer cleared");
        check(bool(registry.set_activity(actor,5,20,5,102)),"refresh can shorten deadline");
        check(registry.next_activity_expiry_ms()==107,"shortened refresh wakes no later than active expiry");
        check(registry.advance_time(107)&&!registry.next_activity_expiry_ms(),"expiry clears final actor timer");
        check(registry.snapshot(key,107).snapshot.local_effective==5,"timer expiry preserves base");
    }
    {
        ThreadPriorities registry(1,limits());
        check(bool(registry.register_thread(first,0,UINT64_MAX-2)),"maximum timer fixture registered");
        check(bool(registry.set_activity(first,"maximum",0,1,2,UINT64_MAX-2)),"maximum timer deadline accepted");
        const auto deadline=registry.next_activity_expiry_ms();
        check(deadline.has_value()&&*deadline==UINT64_MAX,"maximum deadline is distinct from no timer");
        check(registry.advance_time(UINT64_MAX-1)&&registry.next_activity_expiry_ms()==UINT64_MAX,"maximum timer never expires early");
        check(registry.advance_time(UINT64_MAX)&&!registry.next_activity_expiry_ms(),"maximum timer expires without sentinel confusion");
    }
    {
        ThreadPriorities registry(1,limits());
        check(bool(registry.register_thread(first,0,0)),"revision exhaustion timer fixture");
        check(bool(registry.set_activity(first,"early",0,1,10,0))&&
              bool(registry.set_activity(first,"later",0,1,20,0)),"revision exhaustion leases active");
        ThreadPriorityTestAccess::last_revision(registry,first);
        check(registry.advance_time(10),"timer advances even when accepted revision exhausted");
        const auto snapshot=registry.snapshot(first,10).snapshot;
        check(snapshot.closed&&snapshot.accepted_revision==UINT64_MAX&&snapshot.local_effective==0,
              "exhausted expiry closes without wrapping or leaving boost");
        const auto deadline=registry.next_activity_expiry_ms();
        check(!deadline||*deadline<=20,"closed exhausted record can only leave conservative wake");
        check(registry.advance_time(20)&&!registry.next_activity_expiry_ms(),"exhaustion's conservative wake clears");
    }
    {
        // Rebuild an identical registry at every allocation position. Full hidden
        // state comparison includes mint counters, managed mode and timer cache.
        const ThreadKey key{1,1,std::string(40,'t')};
        const std::string actor_name(40,'a'),other_name(40,'z');
        for(int mode=0;mode<15;++mode){
            auto setup=[&](ThreadPriorities& r){
                if(mode<=1)return;
                check(bool(r.register_thread(key,7,0)),"allocation fixture registered");
                check(bool(r.set_activity(key,other_name,0,3,100,0)),"allocation fixture second actor");
                if(mode==5||mode==6)check(bool(r.set_activity(key,actor_name,0,10,100,0)),"legacy actor prepared");
                if(mode>=10&&mode<=12){auto q=r.snapshot(key,0);check(bool(r.register_activity(key,actor_name,q.snapshot.accepted_revision,0)),"managed actor prepared");
                    if(mode!=12){q=r.snapshot(key,0);auto a=q.snapshot.activity.front();check(bool(r.set_activity({key,actor_name,a.incarnation},a.version,10,100,0)),"managed actor activated");}}
                if(mode==2||mode==13){auto q=r.snapshot(key,0);check(bool(r.close(key,q.snapshot.accepted_revision,0)),"closed thread prepared");}
                if(mode==14)ThreadPriorityTestAccess::last_actor(r);
            };
            auto operation=[&](ThreadPriorities& r,uint64_t revision,const ActivityKey& actor,uint64_t version)->PriorityResult{
                switch(mode){
                    case 0:return r.register_thread(key,7,0);
                    case 1:return r.create_thread(key.id,7,0);
                    case 2:return r.register_thread({1,2,key.id},9,0);
                    case 3:return r.set_base(key,1,-4,0);
                    case 4:return r.set_activity(key,actor_name,0,10,100,0);
                    case 5:return r.set_activity(key,actor_name,1,20,200,0);
                    case 6:return r.disconnect(key,actor_name,1,0);
                    case 7:return r.acknowledge(key,revision,1,0);
                    case 8:return r.close(key,revision,0);
                    case 9:case 14:return r.register_activity(key,actor_name,revision,0);
                    case 10:return r.set_activity(actor,version,20,200,0);
                    case 11:return r.disconnect(actor,version,0);
                    case 12:return r.retire_activity(actor,version,0);
                    default:return r.retire_closed_thread(key,revision,0);
                }
            };
            ThreadPriorities reference(1,limits());setup(reference);uint64_t revision=0,version=0;ActivityKey actor{key,actor_name,0};
            if(mode>1){auto q=reference.snapshot(key,0);revision=q.snapshot.accepted_revision;for(const auto& a:q.snapshot.activity)if(a.lease_id==actor_name){actor.incarnation=a.incarnation;version=a.version;}}
            const auto expected=operation(reference,revision,actor,version);check(bool(expected),"allocation reference operation succeeds");
            const auto expected_state=ThreadPriorityTestAccess::state(reference),expected_response=ThreadPriorityTestAccess::response(expected);
            bool reached_success=false;size_t failed=0;
            for(long site=0;site<128;++site){ThreadPriorities registry(1,limits());setup(registry);const auto before=ThreadPriorityTestAccess::state(registry);PriorityResult actual;bool threw=false;
                allocation_failure_after=site;
                try{actual=operation(registry,revision,actor,version);}catch(const std::bad_alloc&){threw=true;}
                catch(...){allocation_failure_after=-1;check(false,"unexpected mutation exception");}
                allocation_failure_after=-1;
                if(threw){++failed;check(ThreadPriorityTestAccess::state(registry)==before,"failed response leaves requested mutation/counters unapplied");}
                else{check(ThreadPriorityTestAccess::state(registry)==expected_state&&ThreadPriorityTestAccess::response(actual)==expected_response,"prebuilt success response equals committed sorted state");
                    if(mode!=13){const auto observed=registry.snapshot(actual.snapshot.key,0);check(ThreadPriorityTestAccess::response(actual)==ThreadPriorityTestAccess::response(observed),"successful projected response equals fresh actual snapshot");}
                    reached_success=true;break;}
            }
            check(reached_success&&failed>0,"persistent sweep reaches every allocating mutation then success");
        }
        // Expiry is independently committed BEFORE requested operation response
        // allocation. It is intentionally not rolled back by that later failure.
        ThreadPriorities registry(1,limits());check(bool(registry.register_thread(key,7,0)),"expiry exception fixture");
        check(bool(registry.set_activity(key,actor_name,0,10,1,0)),"expiry exception actor");
        bool threw=false;allocation_failure_after=0;
        try{registry.set_base(key,1,99,1);}catch(const std::bad_alloc&){threw=true;}
        allocation_failure_after=-1;check(threw,"requested response allocation fails after expiry");
        auto current=registry.snapshot(key,1);
        check(current.snapshot.base==7&&current.snapshot.base_version==1&&current.snapshot.accepted_revision==3&&
              current.snapshot.activity.front().version==2&&!current.snapshot.activity.front().active&&current.snapshot.local_effective==7,
              "expiry remains committed while requested base mutation does not");
        check(!registry.next_activity_expiry_ms()&&registry.snapshot(key,0).error==PriorityError::invalid_time,"expiry time remains monotonic after exception");
    }
    std::puts("thread_priority_test passed");
}
