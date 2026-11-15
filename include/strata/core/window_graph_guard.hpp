#pragma once
#include "strata/core/window_model_owner.hpp"
#include "strata/core/window_device.hpp"
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace strata::core {
// Driver-only opt-in guard. Producers must have stopped before retirement.
// Explicit retire() is needed before manually freeing graph-referenced buffers.
template<class Graph> class WindowGraphGuard {
public:
    using Retire=bool (*)(Graph&,std::string&);
    WindowGraphGuard(std::shared_ptr<const OwnedModelGang> model,Graph& graph,int device,Retire retire)
        :model_(std::move(model)),graph_(graph),device_(device),retire_(retire){}
    WindowGraphGuard(const WindowGraphGuard&)=delete;
    WindowGraphGuard& operator=(const WindowGraphGuard&)=delete;
    ~WindowGraphGuard() noexcept {retire();}
    void retire() noexcept {
        if(!model_||retired_)return;
        try {
            std::string error;
            if(drain_model_gang_users(*model_,error)) {
                WindowDeviceScope device(device_);
                if(device.ok()&&retire_(graph_,error)&&device.finish()){retired_=true;return;}
            }
        } catch(...) {}
        // Automatic graph/borrower wrappers cannot survive uncertain teardown.
        // Stop before their destruction can free model/session dependencies.
        std::fputs("strata: checked graph retirement failed; worker restart required\n",stderr);
        std::_Exit(1);
    }
private:
    std::shared_ptr<const OwnedModelGang> model_;
    Graph& graph_;
    int device_;
    Retire retire_;
    bool retired_=false;
};
} // namespace strata::core
