struct OwnedQsaElasticPool::Storage {
    std::shared_ptr<const ReservationLease> owner_metadata;
    std::unique_ptr<PreparedVmmAllocationBatch> batch;
    QsaElasticPlan plan;
    VmmRange range;
    std::vector<DetachedVmmChunk> chunks;
    std::vector<VmmChunkOutcome> outcomes;
    std::vector<VmmChunkOutcome> retirement_outcomes;
    uint64_t retirement_generation=0;
    bool sealed=false,retirement_prepared=false,commit_attempted=false;
    QsaState* bound=nullptr;
    int64_t committed=0;
    bool attempted=false,ready=false,retired=false;
    Storage* next=nullptr;
};
bool OwnedQsaElasticPool::host_bound(const QsaElasticPlan& p,uint64_t& owner,uint64_t& range)noexcept{
    if(!p.granularity||!p.virtual_bytes||p.virtual_bytes%p.granularity||!p.initial_chunks||p.initial_chunks>(1ull<<24))return false;
    const uint64_t slots=p.virtual_bytes/p.granularity;
    if(slots>std::vector<VmmChunkOutcome>().max_size())return false;
    uint64_t r=0;if(!VmmRange::managed_host_bound(slots,r))return false;
    uint64_t n=65536+sizeof(Storage)+sizeof(OwnedQsaElasticPool);
    const uint64_t per=sizeof(DetachedVmmChunk)+sizeof(VmmChunkOutcome)+32;
    if(p.initial_chunks>(UINT64_MAX-n)/per)return false;n+=p.initial_chunks*per;
    if(slots>(UINT64_MAX-n)/sizeof(VmmChunkOutcome))return false;
    n+=slots*sizeof(VmmChunkOutcome);
    // managed_host_bound includes both handle and managed-slot arrays;
    // owner envelope additionally retains one range disposition per virtual chunk.
    if(n>SIZE_MAX)return false;owner=n;range=r;return true;
}
bool OwnedQsaElasticPool::prepare(const ModelGeometry& g,const QsaElasticPlan& plan,
    std::shared_ptr<const ReservationLease> owner,std::shared_ptr<const ReservationLease> range,
    std::unique_ptr<PreparedVmmAllocationBatch>& batch,std::shared_ptr<OwnedQsaElasticPool>& out,std::string& error){
    uint64_t ob=0,rb=0;
    auto metadata=[](const auto& l,uint64_t n){if(!l||!l->canonical_owner(l)||l.use_count()!=1||l->purpose()!=ReservationPurpose::foreground||l->resources().host_bytes<n||
        l->resources().pinned_bytes||l->resources().disk_bytes||l->resources().io_slots||l->resources().shadows)return false;
        for(auto d:l->resources().device_bytes)if(d)return false;return true;};
    if(out||!batch||!qsa_elastic_plan_matches(g,plan)||!host_bound(plan,ob,rb)||!metadata(owner,ob)||!metadata(range,rb)){
        error="invalid or underadmitted owned elastic pool";return false;}
    try {
        auto p=std::shared_ptr<OwnedQsaElasticPool>(new OwnedQsaElasticPool);p->storage_=std::make_unique<Storage>();
        auto& s=*p->storage_;s.plan=plan;s.owner_metadata=std::move(owner);
        s.chunks.resize(size_t(plan.initial_chunks));s.outcomes.resize(size_t(plan.initial_chunks));
        s.retirement_outcomes.resize(size_t(plan.virtual_bytes/plan.granularity));
        if(!s.range.prepare_managed_metadata(plan.virtual_bytes/plan.granularity,plan.granularity,
            std::move(range),plan.device,error))return false;
        s.batch=std::move(batch);out=std::move(p);return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
}
