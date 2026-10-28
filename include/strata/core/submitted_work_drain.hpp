#pragma once
#include <cstdio>
#include <cstdlib>

namespace strata::core {
// Declare after the controller and its callbacks, before any scope that can
// submit work. The callback completes only work already submitted; it must not
// admit a new unit. Stack-owned controllers cannot be quarantined on failure.
template<class Drain>
class SubmittedWorkDrainGuard {
public:
    SubmittedWorkDrainGuard(bool enabled, bool& uncertain, Drain& drain) noexcept
        : enabled_(enabled), uncertain_(uncertain), drain_(drain) {}
    SubmittedWorkDrainGuard(const SubmittedWorkDrainGuard&)=delete;
    SubmittedWorkDrainGuard& operator=(const SubmittedWorkDrainGuard&)=delete;
    ~SubmittedWorkDrainGuard() noexcept {
        if(!enabled_)return;
        if(!uncertain_) {
            try { if(drain_())return; } catch(...) {}
        }
        // Never retry an uncertain poll/launch or destroy its callback targets.
        std::fputs("strata: submitted work could not finish before controller teardown; worker restart required\n",stderr);
        std::fflush(stderr);
        std::_Exit(1);
    }
private:
    bool enabled_;
    bool& uncertain_;
    Drain& drain_;
};
} // namespace strata::core
