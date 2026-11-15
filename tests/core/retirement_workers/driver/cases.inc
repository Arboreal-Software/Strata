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
