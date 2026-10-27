#include "strata/core/window_lease.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace strata::core;
static void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
int main() {
    WindowKey key{1,1,"session","thread","window",1,1,1,1,{1},{2}};
    StateCut initial{1,3,2,2,1,{1}}, next{2,3,3,3,1,{1}};
    WindowResources capacity{{100},100,0,0,0,1}, baseline{{0},0,0,0,0,0};
    WindowBudget budget(capacity,baseline);
    auto reserve = [&] { return budget.reserve(WindowResources{{40},40,0,0,0,0}, ReservationPurpose::foreground).lease; };
    auto owner = std::make_shared<int>(9);
    std::weak_ptr<const void> lifetime = owner;
    WindowWork work;
    {
        WindowExecution gate(key,initial,2,owner,reserve());
        owner.reset();
        check(gate.admit(key,work), "first writer admitted");
        WindowWork competitor;
        check(!gate.admit(key,competitor), "second writer excluded");
        check(gate.begin_quiesce(key), "quiescence closes new admission");
        check(!gate.admit(key,competitor), "no first unit or setup after quiescence");
        check(!work.finish(work.operation(),key,{next,next},false,true), "device completion cannot replace host join");
        check(!work.finish(work.operation(),key,{next,next},true,false), "host join cannot replace copy drain");
        auto skew = next; ++skew.forwarded; ++skew.accepted;
        check(!work.finish(work.operation(),key,{next,skew},true,true), "stage skew cannot certify snapshot");
        check(gate.view().cut == initial, "failed completion does not advance cut");
        check(!work.finish(work.operation()+1,key,{next,next},true,true), "stale callback rejected");
        check(work.finish(work.operation(),key,{next,next},true,true), "common completed cut accepted");
        check(gate.view().phase == WindowExecutionPhase::quiesced, "drain settles quiescence");
        WindowRead read;
        check(gate.acquire_read(key,read) && read.cut() == next, "capture holds the certified cut");
        check(!gate.resume(key), "capture reader prevents mutation");
        check(gate.acquire_read(key,read), "replacing a read hold does not deadlock");
        auto retained = read; read = {};
        check(!gate.resume(key), "copied reader retains quiescence");
        retained = {};
        auto stale = key; ++stale.selection_epoch;
        check(!gate.resume(stale), "stale selection cannot resume");
        check(gate.resume(key), "selected execution resumes");
        WindowWork later;
        check(gate.admit(key,later), "next operation admitted");
        check(!work.mark_unhealthy(work.operation()), "old callback cannot poison next operation");
        check(gate.cancel(key), "cancel after enqueue enters drain");
        check(gate.view().phase == WindowExecutionPhase::cancelling, "cancel is not completion");
        work = later;
    }
    check(!lifetime.expired() && budget.snapshot().reserved.host_bytes == 40,
          "logical deletion retains buffers and accounting until completion");
    auto wrong_cancel = key; ++wrong_cancel.selection_epoch;
    check(!work.finish(work.operation(),wrong_cancel,{},true,true), "misrouted cancellation completion retains pending work");
    wrong_cancel = key; ++wrong_cancel.append_epoch;
    check(!work.finish(work.operation(),wrong_cancel,{},true,true), "cancel cannot publish an append identity");
    check(work.finish(work.operation(),key,{},true,true), "cancelled partial unit retires without publishing cut");
    work = {};
    check(lifetime.expired() && budget.snapshot().reserved.host_bytes == 0, "last completed use releases owner and reservation");
    {
        auto buffer = std::make_shared<int>(1); lifetime = buffer;
        WindowExecution gate(key,initial,1,buffer,reserve());
        check(gate.admit(key,work), "failure fixture admitted");
        check(work.mark_unhealthy(work.operation()), "device failure is explicit");
        check(!work.finish(work.operation(),key,{next},true,true), "later successful query cannot clear failure");
    }
    check(!lifetime.expired(), "failed device retains source owner");
    check(work.retire_after_teardown(work.operation()), "confirmed runtime teardown retires failed use");
    work = {}; check(lifetime.expired(), "teardown releases owner");
    {
        WindowExecution gate(key,initial,1,std::make_shared<int>(0),reserve());
        std::atomic<int> admitted{0};
        WindowWork contenders[2];
        std::thread a([&]{ if(gate.admit(key,contenders[0])) ++admitted; });
        std::thread b([&]{ if(gate.admit(key,contenders[1])) ++admitted; });
        a.join(); b.join(); check(admitted == 1, "concurrent admission has exactly one writer");
        gate.cancel(key);
        for(auto& c:contenders) if(c) check(c.finish(c.operation(),key,{},true,true), "race winner drains");
    }
    {
        WindowExecution gate(key,initial,1,std::make_shared<int>(0),reserve());
        check(gate.admit(key,work), "append fixture admitted");
        auto appended = next; appended.accepted = 4; appended.history[0] = 2;
        check(!work.finish(work.operation(),key,{appended},true,true), "new history requires append epoch");
        auto next_key = key; ++next_key.append_epoch;
        auto jumped = key; jumped.append_epoch = UINT64_MAX;
        check(!work.finish(work.operation(),jumped,{next},true,true), "unchanged history cannot jump epoch");
        check(!work.finish(work.operation(),jumped,{appended},true,true), "append cannot skip epoch to exhaustion");
        check(work.finish(work.operation(),next_key,{appended},true,true), "append advances history and identity together");
        check(!gate.begin_quiesce(key) && gate.begin_quiesce(next_key), "old append identity cannot acquire cut");
    }
    work = {};
    std::puts("window_lease_test passed");
}
