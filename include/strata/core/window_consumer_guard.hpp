#pragma once
#include "strata/core/window_model_owner.hpp"
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace strata::core {
// Place after borrowed callback targets and before enqueue-capable workers, so
// workers join first, device callbacks drain next, and targets are destroyed last.
// The caller must seal new submissions. This does not join arbitrary producers.
class WindowConsumerDrainGuard {
public:
    explicit WindowConsumerDrainGuard(std::shared_ptr<const OwnedModelGang> model):model_(std::move(model)){}
    WindowConsumerDrainGuard(const WindowConsumerDrainGuard&)=delete;
    WindowConsumerDrainGuard& operator=(const WindowConsumerDrainGuard&)=delete;
    ~WindowConsumerDrainGuard() noexcept {
        if(!model_)return;
        try {
            std::string error;
            if(drain_model_gang_users(*model_,error))return;
        } catch(...) {}
        // Automatic callback targets cannot be quarantined during unwind. Do
        // not destroy/reuse them after an uncertain device completion result.
        std::fputs("strata: model consumer drain failed during teardown; worker restart required\n",stderr);
        std::_Exit(1);
    }
private:
    std::shared_ptr<const OwnedModelGang> model_;
};
} // namespace strata::core
