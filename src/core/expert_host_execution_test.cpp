// Prepared refusal/bounds tests only. Genuine success-authority fixture is pending.
// No private-member injection or fake successful SelectedWindowWork factory.
#include "strata/core/expert_host_execution.hpp"
#include <cassert>
#include <type_traits>
using namespace strata::core;
static_assert(!std::is_copy_constructible_v<ExpertHostExecutionCapture>);
static_assert(!std::is_move_constructible_v<ExpertHostExecutionCapture>);
int main(){
    ExpertHostExecutionLimits limits{2,16,16,8,1024,2};
    std::string error;uint64_t charge=0;
    assert(expert_host_execution_host_bound(limits,charge,error)&&charge>65536);
    for(int i=0;i<6;++i){auto bad=limits;switch(i){case 0:bad.layers=0;break;case 1:bad.rows=UINT64_MAX;break;case 2:bad.batches=UINT64_MAX;break;case 3:bad.rows_per_layer=17;break;case 4:bad.key_bytes=UINT64_MAX;break;case 5:bad.stages=65;break;}
        uint64_t unchanged=123;assert(!expert_host_execution_host_bound(bad,unchanged,error)&&unchanged==123);
    }
    WindowBudget budget({{0},1<<24,0,0,0,0},{{0}});
    SelectedWindowWork missing;
    ExpertHostExecutionContext absent;
    std::unique_ptr<ExpertHostExecutionCapture> out;
    auto lease=budget.reserve({{0},charge,0,0,0,0},ReservationPurpose::foreground);assert(lease);
    assert(!ExpertHostExecutionCapture::create(limits,lease.lease,missing,absent,out,error)&&!out);
    auto insufficient=budget.reserve({{0},charge-1,0,0,0,0},ReservationPurpose::foreground);assert(insufficient);
    assert(!ExpertHostExecutionCapture::create(limits,insufficient.lease,missing,absent,out,error)&&!out);
    // Missing genuine selected admission cannot be papered over with numbers.
    absent.incarnation=1;absent.layer_end=1;absent.experts=2;
    assert(!ExpertHostExecutionCapture::create(limits,lease.lease,missing,absent,out,error)&&!out);
}
