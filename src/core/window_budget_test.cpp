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
        auto admitted=budget.reserve({{0,0},32,0,0,0,0},ReservationPurpose::foreground);
        check(bool(admitted),"canonical reservation admitted");
        using Lease=std::shared_ptr<const ReservationLease>;
        auto rebox=[](const Lease& owner) { return Lease(std::make_shared<Lease>(owner),owner.get()); };
        auto first=rebox(admitted.lease), second=rebox(admitted.lease);
        check(first.use_count()==1 && second.use_count()==1,"independent reboxes bypass count-only check");
        check(!first->canonical_owner(first) && !second->canonical_owner(second),"reboxed groups refused");
        check(admitted.lease->canonical_owner(admitted.lease),"original owning group authenticated");
        Lease same_group(admitted.lease,admitted.lease.get());
        check(same_group->canonical_owner(same_group),"same-group alias authenticated, not exclusive");
        check(same_group.use_count()>1,"authentication does not imply exclusivity");
        Lease empty;
        check(!admitted.lease->canonical_owner(empty),"empty ownership refused");
        auto other=budget.reserve({{0,0},1,0,0,0,0},ReservationPurpose::foreground);
        check(bool(other),"independent reservation admitted");
        check(!admitted.lease->canonical_owner(other.lease),"different reservation refused");
        other.lease.reset();same_group.reset();admitted.lease.reset();
        check(budget.snapshot().reserved.host_bytes==32,"reboxes retain genuine charge");
        first.reset();
        check(budget.snapshot().reserved.host_bytes==32,"last alias group still retains charge");
        second.reset();
        check(budget.snapshot().reserved.host_bytes==0,"weak canonical identity does not leak charge");
        auto moved=budget.reserve({{0,0},32,0,0,0,0},ReservationPurpose::foreground);
        std::weak_ptr<const ReservationLease> weak=moved.lease;
        Lease owner=std::move(moved.lease);
        check(owner.use_count()==1 && owner->canonical_owner(owner),"canonical move remains exclusive");
        auto promoted=weak.lock();
        check(promoted && promoted->canonical_owner(promoted) && promoted.use_count()==2,
              "weak promotion retains canonical group but is not exclusive");
        promoted.reset();owner.reset();
        check(weak.expired() && budget.snapshot().reserved.host_bytes==0,"canonical weak owner permits final release");
    }
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
    {
        // The submitted pointer is live during admission but its ownership group
        // retains an unrelated object. Refusal must precede child charge/flags.
        WindowBudget budget(caps(),zero());
        const WindowReturnObligation obligation{"shared-parent",{{0,0},30,0,0,0,0}};
        auto parent=budget.reserve(obligation.required,ReservationPurpose::foreground_return,{},&obligation);
        check(bool(parent),"shared return parent admitted");
        std::shared_ptr<const ReservationLease> unrelated(std::make_shared<int>(7),parent.lease.get());
        const WindowResources work{{0,0},10,0,0,0,1};
        check(budget.reserve(work,ReservationPurpose::saved_maintenance,unrelated,&obligation).error==ReservationError::return_unavailable,
              "live unrelated group cannot retain return parent");
        check(budget.snapshot().reserved.host_bytes==30&&budget.snapshot().reserved.shadows==0,
              "return alias refusal leaves charges unchanged");
        auto shared=parent.lease;
        auto child=budget.reserve(work,ReservationPurpose::saved_maintenance,shared,&obligation);
        check(bool(child),"genuine shared return remains supported and refusal did not mark busy");
        parent.lease.reset();shared.reset();unrelated.reset();
        check(budget.snapshot().reserved.host_bytes==40,"accepted child truly retains return parent");
        child.lease.reset();check(budget.snapshot().reserved==zero(),"shared return releases once after child");
    }
    {
        WindowBudget budget(caps(),zero());
        auto parent=budget.reserve({{0,0},10,0,0,0,1},ReservationPurpose::private_work);
        std::shared_ptr<const ReservationLease> unrelated(std::make_shared<int>(9),parent.lease.get());
        const WindowResources request{{0,0},20,0,0,1,0};
        check(budget.reserve(request,ReservationPurpose::private_attachment,unrelated).error==ReservationError::invalid_vector,
              "live unrelated group cannot retain attachment parent");
        check(budget.snapshot().reserved.host_bytes==10&&budget.snapshot().reserved.io_slots==0,
              "attachment alias refusal does not charge");
        auto shared=parent.lease;
        auto child=budget.reserve(request,ReservationPurpose::private_attachment,shared);
        check(bool(child),"genuine multiply-owned attachment parent accepted");
        parent.lease.reset();shared.reset();unrelated.reset();
        check(budget.snapshot().reserved.host_bytes==30,"attachment retains real parent charge");
        child.lease.reset();check(budget.snapshot().reserved==zero(),"attachment and parent finally released");
    }
    // The actual nonallocating head transition reuses ONE canonical lease.
    for(bool commit : {false,true}) {
        const auto fixed=WindowBudget::head_spill_control_bytes();
        const WindowResources capacity{{0,100},fixed+80,80,0,0,0},baseline{{0,0},0,0,0,0,0};
        WindowBudget budget(capacity,baseline),foreign(capacity,baseline);
        auto r=budget.reserve({{0,60},fixed,0,0,0,0},ReservationPurpose::foreground);
        check(bool(r),"head setup control prepaid in existing foreground lease");
        HeadSpillCharge token;
        const auto before=budget.snapshot().reserved;
        check(!foreign.admit_head_spill(r.lease,1,40,fixed,token),"head peak rejects foreign ledger");
        check(!budget.admit_head_spill(r.lease,-1,40,fixed,token)&&
              !budget.admit_head_spill(r.lease,2,40,fixed,token),"head peak rejects invalid device");
        check(!budget.admit_head_spill(r.lease,1,61,fixed,token)&&
              !budget.admit_head_spill(r.lease,1,40,fixed+1,token),"head peak refuses insufficient allocation/control");
        check(!budget.admit_head_spill(r.lease,1,UINT64_MAX,fixed,token),"head peak refuses overflow-sized request");
        auto alias=r.lease;
        check(!budget.admit_head_spill(r.lease,1,40,fixed,token),"head peak rejects shared genuine owner");
        alias.reset();
        std::shared_ptr<const ReservationLease> rebox(std::make_shared<decltype(r.lease)>(r.lease),r.lease.get());
        check(rebox.use_count()==1&&!budget.admit_head_spill(rebox,1,40,fixed,token),"head peak rejects independent rebox");
        rebox.reset();
        check(budget.snapshot().reserved==before,"all head preadmission refusals preserve ledger");
        check(!budget.settle_head_spill(r.lease,token,true),"head settlement rejects unadmitted token");
        check(budget.admit_head_spill(r.lease,1,40,fixed,token),"head host and pinned peak admitted");
        check(budget.snapshot().reserved==WindowResources{{0,60},fixed+40,40,0,0,0},"old device and new pinned peak simultaneously charged");
        check(!budget.admit_head_spill(r.lease,1,40,fixed,token),"head token cannot admit twice");
        alias=r.lease;
        check(!budget.settle_head_spill(r.lease,token,commit),"head settlement rejects newly shared owner");
        alias.reset();
        check(!foreign.settle_head_spill(r.lease,token,commit),"head settlement rejects foreign ledger");
        check(budget.settle_head_spill(r.lease,token,commit),"head settlement succeeds once on actual owner");
        check(budget.snapshot().reserved==(commit?WindowResources{{0,20},fixed+40,40,0,0,0}:before),"head commit/refund preserves exact unrelated components");
        check(!budget.settle_head_spill(r.lease,token,commit)&&
              !budget.admit_head_spill(r.lease,1,20,fixed,token),"settled head token cannot be reused");
        r.lease.reset();check(budget.snapshot().reserved==baseline,"final original owner returns all head charge");
    }
    {
        const auto fixed=WindowBudget::head_spill_control_bytes();
        WindowBudget budget({{0,100},fixed+39,40,0,0,1},{{0,0},0,0,0,0,0});
        auto owner=budget.reserve({{0,60},fixed,0,0,0,0},ReservationPurpose::foreground);
        HeadSpillCharge refused;
        check(!budget.admit_head_spill(owner.lease,1,40,fixed,refused),"head host shortage fails before native allocation");
        owner.lease.reset();
        auto work=budget.reserve({{0,60},fixed,0,0,0,1},ReservationPurpose::private_work);
        check(!budget.admit_head_spill(work.lease,1,1,fixed,refused),"head mutation rejects private purpose");
    }
    {
        const auto fixed=WindowBudget::head_spill_control_bytes();
        WindowBudget budget({{0,100},fixed+40,39,0,0,0},{{0,0},0,0,0,0,0});
        auto owner=budget.reserve({{0,60},fixed,0,0,0,0},ReservationPurpose::foreground);
        HeadSpillCharge refused;
        check(!budget.admit_head_spill(owner.lease,1,40,fixed,refused),"head pinned shortage independently refused");
    }
    std::puts("window_budget_test passed");
}
