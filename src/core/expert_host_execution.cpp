#include "strata/core/expert_host_execution.hpp"
#include <atomic>
#include <limits>
#include <new>
namespace strata::core {
namespace {
std::atomic<uint64_t> next_ticket{1};
bool same_owner(const auto&a,const auto&b){return a.get()==b.get()&&!a.owner_before(b)&&!b.owner_before(a);}
bool entry_valid(ExpertHostPoolEntry e){return e==ExpertHostPoolEntry::single||e==ExpertHostPoolEntry::single_split||e==ExpertHostPoolEntry::multi_split||e==ExpertHostPoolEntry::multi_native_split;}
}
struct ExpertHostExecutionCapture::Storage {
    std::shared_ptr<const ReservationLease> reservation;
    SelectedWindowWork work;
    ExpertHostExecutionContext context;
    ExpertHostExecutionLimits limits;
    const ExpertSource* source_identity=nullptr;
    const strata::kernels::cpu::ExpertPool* pool_identity=nullptr;
    std::vector<ExpertHostLayerRecord> layers;
    std::vector<ExpertHostCpuRow> rows;
    std::vector<ExpertHostBatchRecord> batches;
    // Unique layer tickets avoid clearing row marks. Job slots below batch_jobs
    // belong to this batch; each new contiguous job initializes both arrays.
    std::vector<uint64_t> row_seen,job_first,job_count;
    uint64_t batch_jobs=0;
    std::atomic_flag entered=ATOMIC_FLAG_INIT;
    std::atomic<bool> failed{false};
    size_t nl=0,nr=0,nb=0;
    uint64_t reserved_rows=0,reserved_batches=0,pending_rows=0;
    bool active=false,submitted=false,closed=false;
    Storage* next=nullptr;
    bool authority()const noexcept {
        const auto* binding=dynamic_cast<const ExpertHostRuntimeBinding*>(context.runtime.get());
        if(!binding||!binding->owns_expert_dispatch(context.dispatch))return false;
        const auto* s=context.source->completed_inventory();
        const auto* l=context.layout->completed_inventory();
        if(!s||!l||l->requested_layers<=0||l->effective_experts<=0||
           s->layers!=uint64_t(l->requested_layers)||s->experts!=uint64_t(l->effective_experts)||
           context.experts!=l->effective_experts||context.layer_begin<0||
           context.layer_end>l->requested_layers||context.layer_end<=context.layer_begin||
           context.dispatch->src!=source_identity||context.dispatch->pool!=pool_identity||
           reservation==s->metadata_reservation||reservation==l->metadata_reservation||!same_owner(s->retained_layout,context.layout)||
           !same_owner(s->retained_input_bindings,context.inputs)||
           !same_owner(l->retained_input_bindings,context.inputs))return false;
        const auto* f=dynamic_cast<const FileExpertSource*>(context.dispatch->src);
        const auto* a=dynamic_cast<const ArenaExpertSource*>(context.dispatch->src);
        return (f&&f->completed_inventory()==s)||(a&&a->completed_inventory()==s);
    }
    bool ticket(uint64_t t)const noexcept{return active&&nl&&layers[nl-1].ticket==t&&!closed&&!failed.load();}
};
struct ExpertHostExecutionCapture::Entry {
    Storage& s;bool held;
    explicit Entry(Storage& v):s(v),held(!s.entered.test_and_set(std::memory_order_acquire)) {if(!held)s.failed.store(true);}
    ~Entry(){if(held)s.entered.clear(std::memory_order_release);}
    bool reject() noexcept{s.failed.store(true);return false;}
};
bool expert_host_execution_host_bound(const ExpertHostExecutionLimits& l,uint64_t& out,std::string& error){
    if(!l.layers||l.layers>65536||!l.rows||l.rows>(1ull<<24)||!l.batches||l.batches>(1ull<<24)||
       !l.rows_per_layer||l.rows_per_layer>l.rows||!l.key_bytes||l.key_bytes>(1ull<<20)||!l.stages||l.stages>64){error="CPU execution receipt limits invalid";return false;}
    uint64_t n=65536;
    auto add=[&](uint64_t count,uint64_t stride){if(count>(UINT64_MAX-n)/stride)return false;n+=count*stride;return n<=SIZE_MAX;};
    if(!add(l.layers,sizeof(ExpertHostLayerRecord)+64)||!add(l.rows,sizeof(ExpertHostCpuRow)+32)||
       !add(l.batches,sizeof(ExpertHostBatchRecord)+32)||!add(l.rows_per_layer,3*sizeof(uint64_t))||
       !add(1,3*(sizeof(std::vector<uint64_t>)+32)+sizeof(uint64_t))||!add(l.key_bytes,4)||!add(l.stages,sizeof(NativeSessionView)+128)){error="CPU execution receipt envelope overflow";return false;}
    out=n;return true;
}
bool ExpertHostExecutionCapture::create(const ExpertHostExecutionLimits& limits,std::shared_ptr<const ReservationLease> lease,
    const SelectedWindowWork& work,const ExpertHostExecutionContext& context,std::unique_ptr<ExpertHostExecutionCapture>& out,std::string& error){
    uint64_t bytes=0;
    if(out||!expert_host_execution_host_bound(limits,bytes,error))return false;
    const auto& key=work.token().key;
    uint64_t keys=0;for(const auto* p:{&key.session_id,&key.thread_id,&key.window_id}){if(p->size()>limits.key_bytes-keys){error="CPU receipt key exceeds envelope";return false;}keys+=p->size();}
    if(!lease||lease->purpose()!=ReservationPurpose::foreground||lease->resources().host_bytes<bytes||
       !context.runtime||!context.source||!context.layout||!context.inputs||!context.dispatch||
       !context.dispatch->src||!context.dispatch->pool||!context.incarnation||context.layer_begin<0||
       context.layer_end<=context.layer_begin||context.experts<=0||!work.lease()||!work.token().publication){error="CPU receipt requires retained selected runtime and source authorities";return false;}
    try {
        bool matched=false;
        // Construct callback before the borrow; no callbacks allocate later.
        std::function<void(const std::vector<NativeSessionView>&,const std::shared_ptr<NativeRuntimeAttachment>&)> check=
            [&](const auto& stages,const auto& runtime){matched=!stages.empty()&&stages.size()<=limits.stages&&same_owner(runtime,context.runtime);};
        if(!work.borrow(check)||!matched){error="CPU receipt selected runtime mismatch";return false;}
        auto capture=std::unique_ptr<ExpertHostExecutionCapture>(new ExpertHostExecutionCapture);
        capture->storage_=std::make_unique<Storage>();auto& s=*capture->storage_;
        s.reservation=std::move(lease);s.work=work;s.context=context;s.limits=limits;
        s.source_identity=context.dispatch->src;s.pool_identity=context.dispatch->pool;
        if(!s.authority()){error="CPU receipt source/layout join refused";return false;}
        s.layers.resize(size_t(limits.layers));s.rows.resize(size_t(limits.rows));s.batches.resize(size_t(limits.batches));
        s.row_seen.resize(size_t(limits.rows_per_layer));s.job_first.resize(size_t(limits.rows_per_layer));s.job_count.resize(size_t(limits.rows_per_layer));
        out=std::move(capture);return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
}
ExpertHostExecutionCapture::~ExpertHostExecutionCapture(){
    if(storage_&&(storage_->active||storage_->failed.load())&&!storage_->closed){
        // This retains data/owners only. Calling the destroyed observer remains
        // invalid; raw callback removal and producer joins are mandatory.
        static std::atomic<Storage*> retained{nullptr};auto* p=storage_.release();
        p->next=retained.load();while(!retained.compare_exchange_weak(p->next,p)){}
    }
}
bool ExpertHostExecutionCapture::begin_layer(const void* d,const void* source,const void* pool,int64_t layer,
    uint64_t tokens,uint64_t width,bool multi,uint64_t& ticket)noexcept{
    auto& s=*storage_;Entry g(s);if(!g.held)return false;
    if(s.failed.load()||s.closed||s.active||s.nl>=s.layers.size()||d!=s.context.dispatch||source!=s.source_identity||
       pool!=s.pool_identity||layer<s.context.layer_begin||layer>=s.context.layer_end||!tokens||!width||
       (!multi&&tokens!=1)||width>s.limits.rows_per_layer/tokens||!s.authority())return g.reject();
    const uint64_t n=tokens*width;
    if(n>s.limits.rows-s.reserved_rows||n>s.limits.batches-s.reserved_batches)return g.reject();
    uint64_t t=next_ticket.load();do{if(t==UINT64_MAX)return g.reject();}while(!next_ticket.compare_exchange_weak(t,t+1));
    auto& l=s.layers[s.nl++];l={t,tokens,width,uint64_t(s.nr),0,uint64_t(s.nb),0,layer,multi,ExpertHostLayerStatus::active};
    s.reserved_rows+=n;s.reserved_batches+=n;s.pending_rows=s.nr;s.active=true;s.submitted=false;s.batch_jobs=0;
    if(s.failed.load())return false;
    ticket=t;return true;
}
bool ExpertHostExecutionCapture::cpu_row(uint64_t ticket,const ExpertHostCpuRow& r)noexcept{
    auto& s=*storage_;Entry g(s);if(!g.held)return false;
    if(!s.ticket(ticket)||s.submitted||s.nr>=s.rows.size())return g.reject();auto& l=s.layers[s.nl-1];
    if(r.token>=l.tokens||r.rank>=l.width||r.row!=r.token*l.width+r.rank||r.batch!=l.batch_count||
       r.expert<0||r.expert>=s.context.experts||!r.blob||!entry_valid(r.entry)||
       (l.multi?(r.entry!=ExpertHostPoolEntry::multi_split&&r.entry!=ExpertHostPoolEntry::multi_native_split):
        (r.entry!=ExpertHostPoolEntry::single&&r.entry!=ExpertHostPoolEntry::single_split))||
       (r.activation!=ExpertHostActivation::q8_1&&r.activation!=ExpertHostActivation::any_q2&&r.activation!=ExpertHostActivation::native))return g.reject();
    if(s.row_seen[size_t(r.row)]==ticket||r.job>s.batch_jobs||r.job>=s.job_first.size())return g.reject();
    const bool new_job=r.job==s.batch_jobs;
    if(new_job){if(r.job_row!=0)return g.reject();}
    else {
        const auto& p=s.rows[size_t(s.job_first[size_t(r.job)])];
        if(p.expert!=r.expert||p.blob!=r.blob||p.up_offset!=r.up_offset||p.down_offset!=r.down_offset||
           p.zero_copy_roles!=r.zero_copy_roles||p.activation!=r.activation||p.entry!=r.entry||
           r.job_row!=s.job_count[size_t(r.job)])return g.reject();
    }
    // Commit prepaid bookkeeping only after every refusal check. No callback
    // allocates; rows retain original arrival order, including duplicate experts.
    if(new_job){s.job_first[size_t(r.job)]=s.nr;s.job_count[size_t(r.job)]=0;++s.batch_jobs;}
    ++s.job_count[size_t(r.job)];s.row_seen[size_t(r.row)]=ticket;
    s.rows[s.nr++]=r;++l.row_count;return !s.failed.load();
}
bool ExpertHostExecutionCapture::cpu_batch_submitted(uint64_t ticket,uint64_t batch,uint64_t jobs,ExpertHostPoolEntry entry)noexcept{
    auto& s=*storage_;Entry g(s);if(!g.held)return false;
    if(!s.ticket(ticket)||s.submitted||s.nb>=s.batches.size()||s.nr==s.pending_rows||!jobs||!entry_valid(entry))return g.reject();
    auto& l=s.layers[s.nl-1];if(batch!=l.batch_count)return g.reject();uint64_t count=0;
    for(size_t i=size_t(s.pending_rows);i<s.nr;++i){const auto&r=s.rows[i];if(r.entry!=entry)return g.reject();if(r.job>=count)count=r.job+1;}
    if(count!=jobs)return g.reject();
    s.batches[s.nb++]={ticket,batch,jobs,s.pending_rows,uint64_t(s.nr)-s.pending_rows,entry,false};s.submitted=true;return !s.failed.load();
}
void ExpertHostExecutionCapture::cpu_batch_completed(uint64_t ticket,uint64_t batch)noexcept{
    auto& s=*storage_;Entry g(s);if(!g.held)return;
    if(!s.ticket(ticket)||!s.submitted||!s.nb||s.batches[s.nb-1].ordinal!=batch){g.reject();return;}
    s.batches[s.nb-1].completed=true;++s.layers[s.nl-1].batch_count;s.submitted=false;s.pending_rows=s.nr;s.batch_jobs=0;
}
void ExpertHostExecutionCapture::end_host_dispatch(uint64_t ticket)noexcept{
    auto& s=*storage_;Entry g(s);if(!g.held)return;
    if(!s.ticket(ticket)||s.submitted||s.nr!=s.pending_rows){g.reject();return;}
    s.layers[s.nl-1].status=ExpertHostLayerStatus::returned;s.active=false;
}
void ExpertHostExecutionCapture::abort_layer(uint64_t ticket)noexcept{
    auto& s=*storage_;Entry g(s);if(!g.held)return;
    if(!s.active||!s.nl||s.layers[s.nl-1].ticket!=ticket){g.reject();return;}
    s.layers[s.nl-1].status=ExpertHostLayerStatus::aborted;s.active=false;g.reject();
}
void ExpertHostExecutionCapture::close_after_producers_joined()noexcept{
    auto& s=*storage_;Entry g(s);if(!g.held)return;
    if(s.active){s.layers[s.nl-1].status=ExpertHostLayerStatus::aborted;s.failed.store(true);s.active=false;}s.closed=true;
}
ExpertHostExecutionView ExpertHostExecutionCapture::view()const noexcept{
    const auto& s=*storage_;
    // Caller must serialize: this is not a concurrent snapshot API.
    return {&s.work.token(),s.work.lease().operation(),s.context.incarnation,s.context.policy_revision,
        s.layers.data(),s.nl,s.rows.data(),s.nr,s.batches.data(),s.nb,s.failed.load(),s.closed};
}
}
