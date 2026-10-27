#include "strata/core/window_model_owner.hpp"
#include <cstdio>

// Link the actual model owners, canonical/native loaders and packed registries
// together. The sanitizer fault fixtures replace CUDA/native methods, so they
// cannot detect missing production symbols or a stale shared-header ABI.
// Empty admission is deliberately GPU-free and must not initialize a device.
int main() {
    using namespace strata::core;
    ModelStageResources stage;
    ModelGangResources gang;
    std::shared_ptr<const OwnedModelStage> stage_owner;
    std::shared_ptr<const OwnedModelGang> gang_owner;
    std::string error;
    if(adopt_model_stage(stage,stage_owner,error)||adopt_model_gang(gang,gang_owner,error)||
       model_stage_quarantine_count()||model_gang_quarantine_count())return 1;
    std::puts("window_model_link_test passed");
}
