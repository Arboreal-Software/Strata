// Actual allocator/budget + actual extracted PleReadContext, no real IO or GPU.
// Reuse existing CUDA/session mocks without their rejecting PLE-close definition.
#define STRATA_WINDOW_TEST_ACTUAL_PLE_CONTEXT
#define main allocator_fixture_original_main
#include "window_allocator_test.cpp"
#undef main
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
namespace {
int table_sequence=0,fail_table=-1,reader_collects=0,guard_drops=0;
}
namespace strata::ngram {
struct PleReader {
    struct Ticket {uint32_t id=0;};
    int table=0;
    Ticket issue(const uint32_t*,size_t,uint8_t*){return {1};}
    bool collect(Ticket,std::string& error){++reader_collects;if(table==fail_table){error="injected unknown IO";return false;}return true;}
};
}
namespace strata::kernels {
struct PleTable::Impl {
    std::recursive_mutex mu;
    bool close_failed=false,open=true;uint64_t generation=1;
    PleIo mode=PleIo::Direct;strata::ngram::PleReader reader;
    int rb=PLE_ROW_BYTES;uint64_t bytes_read=0;
    void decode(const uint8_t*,float*){}
};
PleTable::PleTable():impl_(new Impl){impl_->reader.table=table_sequence++;}
PleTable::~PleTable(){delete impl_;}
bool PleTable::is_open()const{return impl_->open;}
void PleTable::close(){++impl_->generation;impl_->open=false;impl_->mode=PleIo::Mmap;}
void PleTable::read_row(uint32_t,float*)const{}
#include "window_allocator_ple_context_test.inc"
}
struct LastGuard {
    std::shared_ptr<const ReservationLease> attachment;
    ~LastGuard(){check(drained_devices==std::set<int>{0,1},"last guard must survive complete gang barrier");++guard_drops;}
};
int main(){
 for(int scenario=0;scenario<3;++scenario)isolated([&]{
    table_sequence=reader_collects=guard_drops=0;fail_table=-1;
    ModelGeometry g;g.n_layers=8;g.qsa_interval=4;
    WindowResources capacity{{100000,100000},100000,10000,0,0,2},zero{{0,0},0,0,0,0,0};
    WindowBudget budget(capacity,zero);NativeSessionAllocation out;
    std::vector<NativeSessionRequest> requests{{0,0,4,true},{1,4,8,false}};
    check(bool(allocate_native_sessions(g,16,1,requests,budget,out)),"combined actual allocator setup");
    std::weak_ptr<LastGuard> guards[2];
    std::weak_ptr<strata::kernels::PleTable> tables[2];
    std::weak_ptr<const ReservationLease> charges[2];
    for(size_t i=0;i<2;++i){
        auto charge=zero;charge.host_bytes=512;
        auto admitted=budget.reserve(charge,ReservationPurpose::private_attachment,out.reservation());
        check(bool(admitted),"genuine same-ledger PLE attachment");
        auto guard=std::make_shared<LastGuard>();guard->attachment=std::move(admitted.lease);
        charges[i]=guard->attachment;guards[i]=guard;
        auto table=std::make_shared<strata::kernels::PleTable>();tables[i]=table;
        auto context=std::make_shared<strata::kernels::PleReadContext>(table,guard);
        uint32_t rows[strata::kernels::PLE_N_HEADS]={};check(context->issue(rows),"actual PLE pending ticket");
        out.stages()[i].session->ple.read_context=std::move(context);
    }
    const auto held=budget.snapshot().reserved;std::string error;
    const auto old_frees=frees,old_host=host_frees;
    drained_devices.clear();check_retirement_order=true;
    {
        NativeRetirementAttempt attempt;check(attempt.prepare(out,{},error),"combined terminal seal");
        if(scenario==1)fail_drain_device=1;
        if(scenario==2)fail_table=1;
        const bool success=attempt.drain_and_commit(error);
        check(success==(scenario==0),"combined terminal result");
        check(guard_drops==0,"contexts retained through terminal attempt");
        for(size_t i=0;i<2;++i)check(!guards[i].expired()&&!tables[i].expired()&&!charges[i].expired(),"all context authorities still retained");
        const auto calls=device_calls,collects=reader_collects;
        check(!attempt.drain_and_commit(error)&&device_calls==calls&&reader_collects==collects,"no repeated IO/device effects");
        out={};
    }
    if(scenario==0){
        check(guard_drops==2&&budget.snapshot().reserved==zero,"successful final owner releases contexts and genuine attachments after barrier");
        for(size_t i=0;i<2;++i)check(guards[i].expired()&&tables[i].expired()&&charges[i].expired(),"success releases last references");
    }else{
        check(guard_drops==0&&budget.snapshot().reserved==held&&native_session_quarantine_count()==1,"failure retains real Storage plus PLE attachments");
        check(frees==old_frees&&host_frees==old_host,"failed precommit keeps every borrowed target");
        for(size_t i=0;i<2;++i)check(!guards[i].expired()&&!tables[i].expired()&&!charges[i].expired(),"quarantine keeps both last-reference guards");
        check(reader_collects==2,"both actual context closes attempted once");
        if(scenario==2)check(drained_devices.empty(),"later close failure precedes all device barriers");
    }
 });
 std::puts("combined allocator/PLE retirement passed");
}
