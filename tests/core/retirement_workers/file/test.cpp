#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
struct MockThread {
 bool live=true, fail=false; int number=0;
 std::thread::id id{};
 std::function<void()> hook;
 bool joinable()const{return live;}
 std::thread::id get_id()const{return id;}
 void join(){if(hook)hook();if(fail)throw std::runtime_error("join");live=false;}
};
struct FileExpertSource {
 std::mutex background_close_mu_,ram_mu_,io_mu_;
 std::atomic<bool> background_sealed_{false},background_close_failed_{false};
 bool background_joined_=false,ram_quit_=false,io_quit_=false;
 std::condition_variable ram_cv_,io_cv_;
 std::vector<MockThread> async_th_,io_workers_;
 MockThread ram_thread_{false};
 // Sentinels model retained storage; actual method has no access/clear of them.
 std::vector<int> async_todo_{1},ram_jobs_{2},io_jobs_{3},io_q_{4},io_fds_{5};
 bool stop_background_after_callers_joined() noexcept;
};
namespace {
struct FileBackgroundStopFrame {
    const void* owner;
    FileBackgroundStopFrame* previous;
    static thread_local FileBackgroundStopFrame* current;
    explicit FileBackgroundStopFrame(const void* p) noexcept:owner(p),previous(current){current=this;}
    ~FileBackgroundStopFrame(){current=previous;}
    static bool active(const void* p) noexcept {
        for(auto* f=current;f;f=f->previous)if(f->owner==p)return true;
        return false;
    }
};
thread_local FileBackgroundStopFrame* FileBackgroundStopFrame::current=nullptr;
} // namespace

bool FileExpertSource::stop_background_after_callers_joined() noexcept {
    background_sealed_.store(true,std::memory_order_release);
    if(FileBackgroundStopFrame::active(this))return false;
    FileBackgroundStopFrame frame(this);
    try {
        std::unique_lock<std::mutex> close(background_close_mu_,std::try_to_lock);
        if(!close.owns_lock()||background_close_failed_.load(std::memory_order_acquire))return false;
        if(background_joined_)return true;
        const auto self=std::this_thread::get_id();
        for(const auto& t:async_th_)if(t.joinable()&&t.get_id()==self)return false;
        for(const auto& t:io_workers_)if(t.joinable()&&t.get_id()==self)return false;
        if(ram_thread_.joinable()&&ram_thread_.get_id()==self)return false;
        // Finish already-issued prefetches before stopping helper IO workers.
        // Do not call prefetch_end/ram_stop/io_stop: they also release storage.
        for(auto& t:async_th_)if(t.joinable())t.join();
        {std::lock_guard<std::mutex> lock(ram_mu_);ram_quit_=true;}
        ram_cv_.notify_all();
        if(ram_thread_.joinable())ram_thread_.join();
        {std::lock_guard<std::mutex> lock(io_mu_);io_quit_=true;}
        io_cv_.notify_all();
        for(auto& t:io_workers_)if(t.joinable())t.join();
        background_joined_=true;
        return true;
    } catch(...) {
        background_close_failed_.store(true,std::memory_order_release);
        return false;
    }
}

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
