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
