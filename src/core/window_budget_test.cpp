#include "strata/core/window_budget.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>

using namespace strata::core;
namespace {
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
WindowResources zero() { return {{0,0},0,0,0,0,0}; }
WindowResources caps() { return {{100,80},200,20,1000,2,1}; }
}
int main() {
    {
        WindowBudget budget(caps(),zero());
        auto r = budget.reserve({{20,81},10,0,0,0,0},ReservationPurpose::foreground);
        check(r.error == ReservationError::device_limit && r.device == 1,"GPU1-only shortage identified");
        check(budget.snapshot().reserved == zero(),"failed gang reservation holds no GPU0 capacity");
        check(budget.reserve(zero(),ReservationPurpose::private_work).error == ReservationError::invalid_vector,
              "private work cannot omit its shadow charge");
        check(budget.reserve({{0,0},201,0,0,0,0},ReservationPurpose::foreground).error == ReservationError::host_limit,
              "physical host capacity enforced");
        check(budget.reserve({{0,0},30,21,0,0,0},ReservationPurpose::foreground).error == ReservationError::pinned_limit,
              "pinned subset independently capped");
        check(budget.reserve({{0,0},10,11,0,0,0},ReservationPurpose::foreground).error == ReservationError::invalid_vector,
              "pinned bytes cannot exceed total physical host charge");
        check(budget.reserve({{0,0},0,0,1001,0,0},ReservationPurpose::foreground).error == ReservationError::disk_limit,
              "disk quota is reserved without compression assumptions");
        check(budget.reserve({{0,0},0,0,0,3,0},ReservationPurpose::foreground).error == ReservationError::io_limit,
              "bounded IO concurrency");
        auto first = budget.reserve({{10,10},20,5,0,0,1},ReservationPurpose::private_work);
        check(bool(first),"private state admitted");
        check(budget.reserve({{1,1},1,0,0,0,1},ReservationPurpose::private_work).error == ReservationError::shadow_limit,
              "import and repair share one private shadow cap");
        auto async_owner = first.lease;
        first.lease.reset();
        check(budget.snapshot().reserved.shadows == 1,"async owner retains reservation after caller cancellation");
        async_owner.reset();
        check(budget.snapshot().reserved == zero(),"last drained owner releases entire vector");
    }
    {
        WindowBudget budget(caps(),zero()), foreign(caps(),zero());
        const WindowResources work{{10,10},10,0,0,0,1};
        const WindowReturnObligation obligation{"engine1/window1/generation1/cut4",{{10,10},30,0,0,0,0}};
        check(budget.reserve(work,ReservationPurpose::saved_maintenance).error == ReservationError::return_unavailable,
              "maintenance without return reservation refused");
        auto wrong = foreign.reserve({{10,10},30,0,0,0,0},ReservationPurpose::foreground_return,{},&obligation);
        check(budget.reserve(work,ReservationPurpose::saved_maintenance,wrong.lease,&obligation).error == ReservationError::return_unavailable,
              "return reservation from another ledger refused");
        check(budget.reserve(zero(),ReservationPurpose::foreground_return,{},&obligation).error == ReservationError::return_unavailable,
              "zero return reservation cannot authorize real restoration");
        auto empty = obligation; empty.required = zero();
        check(budget.reserve(zero(),ReservationPurpose::foreground_return,{},&empty).error == ReservationError::return_unavailable,
              "vacuous return obligation rejected");
        auto ret = budget.reserve({{20,20},30,0,0,0,0},ReservationPurpose::foreground_return,{},&obligation);
        auto other = obligation; other.identity += "/other";
        check(budget.reserve(work,ReservationPurpose::saved_maintenance,ret.lease,&other).error == ReservationError::return_unavailable,
              "return capacity is tied to the exact window obligation");
        other = obligation; ++other.required.host_bytes;
        check(budget.reserve(work,ReservationPurpose::saved_maintenance,ret.lease,&other).error == ReservationError::return_unavailable,
              "changed return layout requires a new obligation");
        auto maintenance = budget.reserve(work,ReservationPurpose::saved_maintenance,ret.lease,&obligation);
        check(bool(maintenance),"maintenance admitted beside protected return capacity");
        check(budget.reserve(work,ReservationPurpose::saved_maintenance,ret.lease,&obligation).error == ReservationError::return_unavailable,
              "one return lease cannot authorize concurrent maintenance");
        auto child=budget.reserve({{0,0},1,0,0,1,0},ReservationPurpose::private_attachment,maintenance.lease);
        check(bool(child),"maintenance reader child admitted");
        maintenance.lease.reset();
        check(budget.reserve(work,ReservationPurpose::saved_maintenance,ret.lease,&obligation).error == ReservationError::return_unavailable,
              "reader child retains exclusive return obligation after maintenance handle drops");
        child.lease.reset();
        maintenance = budget.reserve(work,ReservationPurpose::saved_maintenance,ret.lease,&obligation);
        check(bool(maintenance),"drained maintenance releases exclusive return claim");
        ret.lease.reset();
        check(budget.snapshot().reserved.host_bytes == 40,"maintenance retains return lease");
        check(budget.reserve({{0,0},161,0,0,0,0},ReservationPurpose::foreground).error == ReservationError::host_limit,
              "protected return bytes cannot be borrowed");
        maintenance.lease.reset();
        check(budget.snapshot().reserved == zero(),"work and retained return release without nested-lock deadlock");
    }
    {
        WindowBudget budget(caps(),zero());
        auto active = budget.reserve({{10,10},50,10,0,0,0},ReservationPurpose::foreground);
        auto baseline = zero(); baseline.host_bytes = 180;
        check(budget.observe_baseline(baseline),"observed expert backing growth recorded");
        check(budget.snapshot().overcommitted,"actual overcommit visible");
        check(budget.reserve(zero(),ReservationPurpose::foreground).error == ReservationError::host_limit,
              "new admission refuses externally overcommitted state");
        check(budget.snapshot().reserved.host_bytes == 50,"observation cannot erase active leases");
        active.lease.reset();
        check(!budget.snapshot().overcommitted,"drained release reconciles overcommit");
    }
    {
        auto capacity = caps(); capacity.host_bytes = UINT64_MAX;
        auto baseline = zero(); baseline.host_bytes = UINT64_MAX - 1;
        WindowBudget budget(capacity,baseline);
        check(budget.reserve({{0,0},2,0,0,0,0},ReservationPurpose::foreground).error == ReservationError::host_limit,
              "addition overflow cannot bypass admission");
    }
    {
        WindowBudget budget(caps(),zero());
        std::atomic<int> in_use{0}, successful{0};
        std::vector<std::thread> threads;
        for (int i=0;i<4;++i) threads.emplace_back([&] {
            for (int j=0;j<1000;++j) {
                auto r = budget.reserve({{10,10},10,0,0,0,1},ReservationPurpose::private_work);
                if (!r) { check(r.error == ReservationError::shadow_limit,"concurrent refusal is shadow cap"); continue; }
                check(in_use.fetch_add(1) == 0,"one private work owner under concurrent requests");
                ++successful;
                check(budget.snapshot().reserved.shadows == 1,"concurrent ledger matches live lease");
                --in_use;
            }
        });
        for (auto& thread : threads) thread.join();
        check(successful > 0 && budget.snapshot().reserved == zero(),"concurrent leases drain without lost accounting");
    }
    {
        WindowBudget budget(caps(),zero()), foreign(caps(),zero());
        const auto attachment=WindowResources{{0,0},20,0,0,1,0};
        auto parent=budget.reserve({{10,10},10,0,0,0,1},ReservationPurpose::private_work);
        check(!budget.reserve(attachment,ReservationPurpose::private_attachment),"attachment requires parent");
        check(!foreign.reserve(attachment,ReservationPurpose::private_attachment,parent.lease),"foreign parent refused");
        auto wrong=budget.reserve(zero(),ReservationPurpose::foreground);
        check(!budget.reserve(attachment,ReservationPurpose::private_attachment,wrong.lease),"foreground cannot authorize private attachment");
        auto invalid=attachment;invalid.shadows=1;
        check(!budget.reserve(invalid,ReservationPurpose::private_attachment,parent.lease),"attachment cannot own another shadow");
        auto child=budget.reserve(attachment,ReservationPurpose::private_attachment,parent.lease);
        check(bool(child),"child admitted against private session");
        check(!budget.reserve(attachment,ReservationPurpose::private_attachment,child.lease),"recursive attachment rejected");
        parent.lease.reset();
        check(budget.snapshot().reserved.shadows==1&&budget.snapshot().reserved.host_bytes==30,"child retains whole parent reservation");
        child.lease.reset();check(budget.snapshot().reserved==zero(),"child releases own and last parent charge");
    }
    std::puts("window_budget_test passed");
}
