#pragma once
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace strata::core {
// Declare after every captured automatic outcome, before launching the thread.
// The caller must establish that joining needs no progress from this thread's
// unwinding caller. In particular, this is not a pipelined doorbell drain.
class JoinedThreadScope {
public:
    JoinedThreadScope(bool enabled, std::thread& worker) noexcept
        : enabled_(enabled), worker_(worker) {}
    JoinedThreadScope(const JoinedThreadScope&)=delete;
    JoinedThreadScope& operator=(const JoinedThreadScope&)=delete;
    ~JoinedThreadScope() noexcept {
        if(!enabled_ || !worker_.joinable())return;
        try {
            if(worker_.get_id()!=std::this_thread::get_id()) {
                worker_.join();
                return;
            }
        } catch(...) {}
        std::fputs("strata: worker join failed before captured state teardown; worker restart required\n",stderr);
        std::fflush(stderr);
        std::_Exit(1);
    }
private:
    bool enabled_;
    std::thread& worker_;
};
} // namespace strata::core
