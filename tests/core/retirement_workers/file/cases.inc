
int main(){
 {FileExpertSource s;assert(s.stop_background_after_callers_joined());assert(s.stop_background_after_callers_joined());assert(s.background_sealed_);}
 {FileExpertSource s;std::vector<int> order;
 auto check=[&](int n){return [&,n]{std::unique_lock<std::mutex> a(s.ram_mu_,std::try_to_lock),b(s.io_mu_,std::try_to_lock);assert(a.owns_lock()&&b.owns_lock());order.push_back(n);assert(s.async_todo_[0]==1&&s.ram_jobs_[0]==2&&s.io_jobs_[0]==3&&s.io_q_[0]==4&&s.io_fds_[0]==5);};};
 s.async_th_.push_back({true,false,0,{},check(1)});s.ram_thread_={true,false,0,{},check(2)};s.io_workers_.push_back({true,false,0,{},check(3)});
 assert(s.stop_background_after_callers_joined());assert((order==std::vector<int>{1,2,3}));assert(s.ram_quit_&&s.io_quit_);assert(s.async_th_.size()==1&&s.io_workers_.size()==1);assert(s.stop_background_after_callers_joined());assert(order.size()==3);}
 {FileExpertSource s;int joined=0;s.async_th_.push_back({true,false,0,{},[&]{++joined;}});s.io_workers_.push_back({true,false,0,std::this_thread::get_id(),{}});assert(!s.stop_background_after_callers_joined());assert(joined==0&&s.background_sealed_);}
 {FileExpertSource s;int calls=0;s.async_th_.push_back({true,true,0,{},[&]{++calls;}});assert(!s.stop_background_after_callers_joined());assert(s.background_close_failed_);assert(!s.stop_background_after_callers_joined());assert(calls==1&&s.async_todo_[0]==1);}
 {FileExpertSource s;s.async_th_.push_back({true,false,0,{},[&]{assert(!s.stop_background_after_callers_joined());}});assert(s.stop_background_after_callers_joined());}
 {FileExpertSource s;FileBackgroundStopFrame outer(&s);assert(!s.stop_background_after_callers_joined());assert(s.background_sealed_&&!s.background_joined_);}
}
