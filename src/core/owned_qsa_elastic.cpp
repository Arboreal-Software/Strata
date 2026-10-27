#include "strata/core/owned_qsa_elastic.hpp"
#include "strata/core/window_device.hpp"
#include <atomic>
#include <algorithm>
#include <new>
namespace strata::core {
struct OwnedQsaElasticPool::Storage {
    std::shared_ptr<const ReservationLease> owner_metadata;
    std::unique_ptr<PreparedVmmAllocationBatch> batch;
    QsaElasticPlan plan;
    VmmRange range;
    std::vector<DetachedVmmChunk> chunks;
    std::vector<VmmChunkOutcome> outcomes;
    QsaState* bound=nullptr;
    int64_t committed=0;
    bool attempted=false,ready=false,retired=false;
    Storage* next=nullptr;
};
bool OwnedQsaElasticPool::host_bound(const QsaElasticPlan& p,uint64_t& owner,uint64_t& range)noexcept{
    if(!p.granularity||!p.virtual_bytes||p.virtual_bytes%p.granularity||!p.initial_chunks||p.initial_chunks>(1ull<<24))return false;
    const uint64_t slots=p.virtual_bytes/p.granularity;
    uint64_t r=0;if(!VmmRange::managed_host_bound(slots,r))return false;
    uint64_t n=65536+sizeof(Storage)+sizeof(OwnedQsaElasticPool);
    const uint64_t per=sizeof(DetachedVmmChunk)+sizeof(VmmChunkOutcome)+32;
    if(p.initial_chunks>(UINT64_MAX-n)/per)return false;n+=p.initial_chunks*per;
    // managed_host_bound includes both handle and managed-slot arrays.
    if(n>SIZE_MAX)return false;owner=n;range=r;return true;
}
bool OwnedQsaElasticPool::prepare(const ModelGeometry& g,const QsaElasticPlan& plan,
    std::shared_ptr<const ReservationLease> owner,std::shared_ptr<const ReservationLease> range,
    std::unique_ptr<PreparedVmmAllocationBatch>& batch,std::shared_ptr<OwnedQsaElasticPool>& out,std::string& error){
    uint64_t ob=0,rb=0;
    auto metadata=[](const auto& l,uint64_t n){if(!l||l.use_count()!=1||l->purpose()!=ReservationPurpose::foreground||l->resources().host_bytes<n||
        l->resources().pinned_bytes||l->resources().disk_bytes||l->resources().io_slots||l->resources().shadows)return false;
        for(auto d:l->resources().device_bytes)if(d)return false;return true;};
    if(out||!batch||!qsa_elastic_plan_matches(g,plan)||!host_bound(plan,ob,rb)||!metadata(owner,ob)||!metadata(range,rb)){
        error="invalid or underadmitted owned elastic pool";return false;}
    try {
        auto p=std::shared_ptr<OwnedQsaElasticPool>(new OwnedQsaElasticPool);p->storage_=std::make_unique<Storage>();
        auto& s=*p->storage_;s.plan=plan;s.owner_metadata=std::move(owner);
        s.chunks.resize(size_t(plan.initial_chunks));s.outcomes.resize(size_t(plan.initial_chunks));
        if(!s.range.prepare_managed_metadata(plan.virtual_bytes/plan.granularity,plan.granularity,
            std::move(range),plan.device,error))return false;
        s.batch=std::move(batch);out=std::move(p);return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
}
const QsaElasticPlan& OwnedQsaElasticPool::plan()const noexcept{return storage_->plan;}
uint8_t* OwnedQsaElasticPool::base()const noexcept{return storage_->range.base();}
int64_t OwnedQsaElasticPool::committed_slots()const noexcept{return healthy()?storage_->committed:0;}
bool OwnedQsaElasticPool::healthy()const noexcept{return storage_->ready&&!storage_->retired&&storage_->range.managed_healthy();}
bool OwnedQsaElasticPool::claim_state(QsaState* s)noexcept{
    if(!s||!healthy()||storage_->bound)return false;storage_->bound=s;return true;
}
bool OwnedQsaElasticPool::initialize_checked(std::string& error){
    auto& s=*storage_;if(s.attempted||s.retired){error="owned elastic initialization is one-shot";return false;}s.attempted=true;
    WindowDeviceScope device(s.plan.device);
    bool effects_attempted=false;
    auto refuse=[&](const char* why){
        // Quarantine does not establish quiescence: still attempt the drain on
        // every failure after physical/device work begins, even with a sticky
        // pending error. Never retry uncertain cleanup or publish ready here.
        const bool drained=!effects_attempted||cudaDeviceSynchronize()==cudaSuccess;
        const bool restored=device.finish();
        error=!restored?"owned elastic device restoration failed":
            !drained?"owned elastic failure-path drain failed":why;
        return false;
    };
    if(!device.ok())return refuse("owned elastic device selection failed");
    if(cudaGetLastError()!=cudaSuccess)return refuse("preexisting owned elastic device error");
    // Discovery may issue driver queries; every lease already exists at this point.
    if(!vmm_available()||vmm_granularity()!=s.plan.granularity)return refuse("owned elastic VMM domain mismatch");
    effects_attempted=true;
    const auto allocated=s.batch->allocate_device(0,s.chunks);
    if(allocated.code!=VmmBatchResult::Code::complete||allocated.completed!=s.chunks.size())return refuse("owned elastic physical batch incomplete");
    if(!s.range.activate_managed())return refuse("owned elastic address activation failed");
    const uint64_t live=uint64_t((s.plan.initial_cells+s.plan.page_size-1)/s.plan.page_size);
    size_t used=0;
    for(size_t a=0;a<s.plan.arrays;++a){
        const uint64_t bytes=live*s.plan.per_slot[a];
        const uint64_t n=(bytes+s.plan.granularity-1)/s.plan.granularity;
        const int64_t lo=int64_t(s.plan.offsets[a]/s.plan.granularity);
        if(!s.range.map_managed(lo,lo+int64_t(n),std::span<DetachedVmmChunk>(s.chunks).subspan(used,size_t(n)),
            std::span<VmmChunkOutcome>(s.outcomes).subspan(used,size_t(n))))return refuse("owned elastic mapping incomplete");
        // Always clear reused as well as newly created physical chunks.
        if(cudaMemset(s.range.base()+s.plan.offsets[a],0,size_t(n*s.plan.granularity))!=cudaSuccess)return refuse("owned elastic initial zero failed");
        used+=size_t(n);
    }
    const auto pending=cudaGetLastError();
    const auto drained=cudaDeviceSynchronize(); // required even when pending failed
    if(pending!=cudaSuccess||drained!=cudaSuccess){
        // Drain already attempted. Preserve uncertainty without a second wait.
        effects_attempted=false;
        return refuse("owned elastic initial drain failed");
    }
    if(!device.finish()){error="owned elastic initial device restoration failed";return false;}
    s.committed=int64_t(live);s.ready=true;return true;
}
bool OwnedQsaElasticPool::retire_checked()noexcept{
    auto& s=*storage_;if(s.retired)return true;if(s.attempted&&!s.ready)return false;
    if(!s.attempted){s.retired=true;return true;}
    s.ready=false;
    WindowDeviceScope device(s.plan.device);
    if(!device.ok()||cudaDeviceSynchronize()!=cudaSuccess){(void)device.finish();return false;}
    s.range.release();
    if(!device.finish())return false;
    s.retired=true;s.committed=0;return true;
}
OwnedQsaElasticPool::~OwnedQsaElasticPool(){
    if(!storage_)return;
    if(!retire_checked()){
        static std::atomic<Storage*> quarantine{nullptr};auto* p=storage_.release();
        p->next=quarantine.load();while(!quarantine.compare_exchange_weak(p->next,p)){}
    }
}
bool prepare_owned_qsa_elastic_pools(const ModelGeometry& g,const std::vector<QsaElasticPlan>& plans,
    WindowBudget& budget,const std::shared_ptr<const ReservationLease>& parent,
    std::vector<std::shared_ptr<QsaElasticView>>& out,std::string& error) {
    if(!out.empty()||plans.empty()||!parent){error="elastic preparation requires empty output and arena parent";return false;}
    try {
        const size_t devices=budget.snapshot().capacity.device_bytes.size();
        // Reject every bad quote before acquiring any additional lease.
        for(const auto& p:plans)if(!qsa_elastic_plan_matches(g,p)||p.device<0||size_t(p.device)>=devices||
            p.device!=plans[0].device||p.initial_chunks>UINT64_MAX/p.granularity) {
            error="unsupported elastic physical domain/quote";return false;
        }
        std::vector<std::shared_ptr<QsaElasticView>> candidate;candidate.reserve(plans.size());
        for(const auto& p:plans) {
            uint64_t ob=0,rb=0,bb=0;
            if(!OwnedQsaElasticPool::host_bound(p,ob,rb)||!PreparedVmmAllocationBatch::host_bound(1,bb)) {
                error="elastic metadata envelope overflow";return false;
            }
            WindowResources charge;charge.device_bytes.resize(devices);charge.host_bytes=ob;
            auto owner=budget.reserve(charge,ReservationPurpose::foreground);charge.host_bytes=rb;
            auto range=budget.reserve(charge,ReservationPurpose::foreground);charge.host_bytes=bb;
            auto metadata=budget.reserve(charge,ReservationPurpose::foreground);charge.host_bytes=0;
            charge.device_bytes[size_t(p.device)]=p.initial_chunks*p.granularity;
            auto physical=budget.reserve(charge,ReservationPurpose::private_attachment,parent);
            if(!owner||!range||!metadata||!physical){error="elastic preadmission refused";return false;}
            const VmmBatchDevicePlan batch_plan{p.device,size_t(p.device),p.initial_chunks,p.granularity};
            std::unique_ptr<PreparedVmmAllocationBatch> batch;
            if(!PreparedVmmAllocationBatch::prepare(std::span<const VmmBatchDevicePlan>(&batch_plan,1),
                std::move(physical.lease),std::move(metadata.lease),batch,error))return false;
            std::shared_ptr<OwnedQsaElasticPool> pool;
            if(!OwnedQsaElasticPool::prepare(g,p,std::move(owner.lease),std::move(range.lease),batch,pool,error))return false;
            candidate.push_back(std::move(pool));
        }
        out=std::move(candidate);return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
}

}
