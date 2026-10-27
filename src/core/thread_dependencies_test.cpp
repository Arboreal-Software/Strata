#include "strata/core/thread_dependencies.hpp"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <atomic>
using namespace strata::core;
namespace strata::core {struct ThreadDependenciesTestAccess {static void last_serial(ThreadDependencies& g){g.next_serial_=UINT64_MAX;}};}
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static uint64_t rev(ThreadDependencies& g){return g.snapshot().revision;}
static DependencyOperation op(ThreadDependencies& g,OperationKey k){for(auto n:g.snapshot().operations)if(n.key==k)return n;std::abort();}
static OperationKey add(ThreadDependencies& g,ThreadKey t,DependencyWorkClass c=DependencyWorkClass::foreground,uint64_t now=0){auto r=g.register_operation(t,1,c,rev(g),now);check(bool(r),"register operation");return r.key;}
static void edge(ThreadDependencies& g,OperationKey a,OperationKey b){check(bool(g.update_dependencies({{a,b,true}},rev(g))),"add edge");}
int main(){
 ThreadKey a{1,1,"a"},b{1,1,"b"},shared{1,1,"shared"};
 PriorityLimits limits{-100,100,50,1000,8,4,64};
 ThreadPriorities p(1,limits);check(bool(p.register_thread(a,10,0))&&bool(p.register_thread(b,20,0))&&bool(p.register_thread(shared,-10,0)),"threads registered");
 ThreadDependencies g(1,{16,24},p);
 auto aa=add(g,a),bb=add(g,b),x=add(g,shared,DependencyWorkClass::saved_maintenance),y=add(g,shared),z=add(g,shared);
 check(bool(g.update_dependencies({{aa,x,true},{bb,x,true},{x,y,true},{aa,z,true},{z,y,true}},rev(g))),"diamond and multiwaiter batch");
 check(op(g,y).effective_priority==20&&op(g,y).priority_source==bb,"transitive highest waiter and provenance");
 check(op(g,x).work_class==DependencyWorkClass::saved_maintenance&&!op(g,aa).dependencies_satisfied,"priority does not promote work class or unblock consumer");
 auto bad=z;++bad.engine_generation;auto transaction_rev=rev(g);
 check(g.update_dependencies({{bb,z,true},{z,bad,true}},transaction_rev).error==DependencyError::unknown_generation&&rev(g)==transaction_rev&&op(g,bb).prerequisites.size()==1,"invalid batch suffix cannot publish valid prefix");
 check(g.finish(aa,DependencyState::complete,rev(g)).error==DependencyError::invalid_argument,"blocked consumer cannot complete early");
 auto old=rev(g);check(g.update_dependencies({{y,aa,true}},old).error==DependencyError::cycle&&rev(g)==old,"cycle refusal atomic");
 check(g.update_dependencies({{bb,z,true}},old-1).error==DependencyError::stale_version,"stale competing graph change refused");
 auto high=p.set_activity(a,"user1",0,40,10,1);check(bool(high),"first user activity");
 check(bool(p.set_activity(a,"user2",0,20,30,1)),"second user same thread");
 check(bool(g.refresh_priorities(1))&&op(g,y).effective_priority==50&&op(g,y).priority_source==aa,"activity inheritance from same-thread users");
 check(op(g,aa).thread.applied_revision==0&&op(g,aa).thread.accepted_revision>1,"accepted remains distinct from applied");
 auto r=p.set_base(a,1,-20,2);check(bool(r),"base reduced while boosted");check(bool(g.refresh_priorities(2))&&op(g,y).effective_priority==20,"decrease recomputed");
 check(bool(g.finish(bb,DependencyState::cancelled,rev(g)))&&op(g,x).state==DependencyState::pending,"one waiter cancellation keeps shared prerequisite alive");
 check(bool(g.refresh_priorities(11))&&op(g,y).effective_priority==0,"expiry timer removes highest boost without UI request");
 check(bool(g.update_dependencies({{aa,x,false},{aa,z,false}},rev(g)))&&op(g,y).effective_priority==-10,"edge removals undo transitive inheritance");
 check(bool(g.finish(y,DependencyState::complete,rev(g)))&&op(g,x).dependencies_satisfied&&op(g,z).dependencies_satisfied,"completion unblocks all direct waiters");
 check(bool(g.retire(y,rev(g))),"terminal node retired");auto replacement=add(g,shared,DependencyWorkClass::foreground,11);
 check(replacement.serial!=y.serial&&g.update_dependencies({{x,y,true}},rev(g)).error==DependencyError::unknown_generation,"retired key cannot target reused capacity");
 auto fake=replacement;++fake.result_revision;check(g.finish(fake,DependencyState::complete,rev(g)).error==DependencyError::unknown_generation,"result milestone exact");
 edge(g,x,replacement);check(bool(g.finish(replacement,DependencyState::failed,rev(g)))&&op(g,x).state==DependencyState::dependency_failed,"dependency failure observable and terminal");
 check(bool(g.retire(replacement,rev(g)))&&op(g,x).state==DependencyState::dependency_failed,"retirement cannot revive failed consumer");
 auto astatus=p.snapshot(a,11);check(bool(p.close(a,astatus.snapshot.accepted_revision,11)),"close thread");
 auto newer=a;++newer.generation;check(bool(p.register_thread(newer,70,11)),"new thread incarnation");
 check(bool(g.refresh_priorities(11))&&op(g,aa).state==DependencyState::cancelled,"old generation operations cancelled on refresh");
 check(g.register_operation(a,1,DependencyWorkClass::foreground,rev(g),11).error==DependencyError::unknown_generation,"stale thread rejected");
 auto fresh=add(g,newer,DependencyWorkClass::live_repair,11);check(op(g,fresh).effective_priority==70,"recreated thread priority");
 auto before=rev(g);check(g.refresh_priorities(10).error==DependencyError::invalid_time&&rev(g)==before,"time rollback atomic");
 // Capacity is reusable for indefinite operation churn; graph retains no retired IDs.
 ThreadDependencies bounded(1,{1,1},p);
 OperationKey previous{};
 for(int i=0;i<1000;++i){auto k=add(bounded,shared,DependencyWorkClass::foreground,11);check(k.serial>previous.serial,"monotonic operation serial");
   check(bounded.register_operation(shared,1,DependencyWorkClass::foreground,rev(bounded),11).error==DependencyError::limit_exceeded,"node cap");
   check(bool(bounded.finish(k,DependencyState::complete,rev(bounded)))&&bool(bounded.retire(k,rev(bounded))),"bounded retire churn");previous=k;}
 ThreadDependencies concurrent(1,{3,3},p);auto c1=add(concurrent,shared,DependencyWorkClass::foreground,11),c2=add(concurrent,shared,DependencyWorkClass::foreground,11);
 const auto version=rev(concurrent);std::atomic<int> accepted=0,stale=0;
 auto change=[&](OperationKey u,OperationKey v){auto result=concurrent.update_dependencies({{u,v,true}},version);if(result)++accepted;else if(result.error==DependencyError::stale_version)++stale;};
 std::thread t1(change,c1,c2),t2(change,c2,c1);t1.join();t2.join();check(accepted==1&&stale==1,"competing cycle additions have one CAS winner");
 ThreadDependencies exhausted(1,{2,2},p);ThreadDependenciesTestAccess::last_serial(exhausted);
 auto last=add(exhausted,shared,DependencyWorkClass::foreground,11);check(last.serial==UINT64_MAX,"last operation serial accepted");
 check(exhausted.register_operation(shared,1,DependencyWorkClass::foreground,rev(exhausted),11).error==DependencyError::limit_exceeded,"operation serial exhaustion never wraps");
 ThreadPriorities restarted_priorities(2,limits);ThreadKey new_engine{2,1,"a"};check(bool(restarted_priorities.register_thread(new_engine,10,0)),"restart thread reconstruction");
 ThreadDependencies restarted(2,{3,1},restarted_priorities);auto r1=add(restarted,new_engine),r2=add(restarted,new_engine),r3=add(restarted,new_engine);
 check(restarted.update_dependencies({{r1,aa,true}},rev(restarted)).error==DependencyError::unknown_generation,"pre-restart operation cannot enter reconstructed graph");
 auto restart_revision=rev(restarted);
 check(restarted.update_dependencies({{r1,r2,true},{r2,r3,true}},restart_revision).error==DependencyError::limit_exceeded&&rev(restarted)==restart_revision,"edge capacity refusal is atomic");
 std::puts("thread_dependencies_test passed");
}
