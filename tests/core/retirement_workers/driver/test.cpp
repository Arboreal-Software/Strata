#include <atomic>
#include <cassert>
#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>
struct Worker {
 int id; std::vector<int>* order; bool ok=true,throws=false;
 std::function<void()> hook;
 bool call(){order->push_back(id);if(hook)hook();if(throws)throw std::runtime_error("worker");return ok;}
 bool close_for_retirement(){return call();}
 bool stop_after_dispatch_callers_joined(){return call();}
 bool stop_background_after_callers_joined(){return call();}
};
struct Ple {std::shared_ptr<Worker> read_context;};
struct Session {Ple ple;};
struct Stage {std::shared_ptr<Session> session;};
struct Sessions {std::vector<Stage> data;const auto& stages()const{return data;}};
struct DriverWindowRuntime {
 std::atomic<unsigned> background_retirement{0};
 std::shared_ptr<Worker> lookahead,foresight,pool,source;
 Sessions sessions;
 bool close_background_after_callers_joined() noexcept;
};
bool DriverWindowRuntime::close_background_after_callers_joined() noexcept {
    unsigned expected=0;
    if(!background_retirement.compare_exchange_strong(expected,1,std::memory_order_acq_rel))
        return expected==2;
    bool closed=false;
    try {
        // No new predictions may reach the swap source or dispatch pool once
        // their workers are stopped. Keep every buffer and callback target alive.
        closed=(!lookahead||lookahead->close_for_retirement())&&
               (!foresight||foresight->close_for_retirement())&&
               (!pool||pool->stop_after_dispatch_callers_joined())&&
               (!source||source->stop_background_after_callers_joined());
        if(closed)for(const auto& stage:sessions.stages()){
            if(stage.session&&stage.session->ple.read_context&&
               !stage.session->ple.read_context->close_for_retirement()){
                closed=false;break;
            }
        }
    } catch(...) {closed=false;}
    background_retirement.store(closed?2u:3u,std::memory_order_release);
    return closed;
}
int main(){
 {DriverWindowRuntime r;assert(r.close_background_after_callers_joined());assert(r.close_background_after_callers_joined());}
 for(int fail=0;fail<=6;++fail){
  DriverWindowRuntime r;std::vector<int> order;std::vector<std::shared_ptr<Worker>> owners;
  for(int i=1;i<=6;++i)owners.push_back(std::make_shared<Worker>(Worker{i,&order,i!=fail,false,{}}));
  r.lookahead=owners[0];r.foresight=owners[1];r.pool=owners[2];r.source=owners[3];
  r.sessions.data={{std::make_shared<Session>(Session{Ple{owners[4]}})},{std::make_shared<Session>(Session{Ple{owners[5]}})}};
  owners[0]->hook=[&]{assert(!r.close_background_after_callers_joined());};
  bool ok=r.close_background_after_callers_joined();assert(ok==(fail==0));
  assert(order.size()==static_cast<size_t>(fail?fail:6));
  for(size_t i=0;i<order.size();++i)assert(order[i]==int(i+1));
  for(auto& p:owners)assert(p.use_count()==2); // No reset/destruction during either outcome.
  auto before=order;assert(r.close_background_after_callers_joined()==ok);assert(order==before);
 }
 {DriverWindowRuntime r;std::vector<int> order;r.lookahead=std::make_shared<Worker>(Worker{1,&order,true,true,{}});r.source=std::make_shared<Worker>(Worker{4,&order,true,false,{}});assert(!r.close_background_after_callers_joined());assert((order==std::vector<int>{1}));assert(r.source&&r.lookahead);assert(!r.close_background_after_callers_joined());}
 {DriverWindowRuntime r;std::vector<int> order;r.sessions.data={{nullptr},{std::make_shared<Session>()},{std::make_shared<Session>(Session{Ple{std::make_shared<Worker>(Worker{6,&order,true,false,{}})}})}};assert(r.close_background_after_callers_joined());assert((order==std::vector<int>{6}));}
}
