#include "strata/core/window_allocator.hpp"
#include "strata/core/window_device.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <mutex>
#include <new>
#include <set>

namespace strata::core {
namespace {
// Serialize native init/release (stable global configuration is still required).
// Recursive for same-thread partial-allocation cleanup, not callback allocation.
std::recursive_mutex operation_mutex;
// Only health/quarantine metadata; never held across device/IO work or owner destruction.
std::mutex allocator_mutex;
struct Stage {
    std::vector<QsaElasticPlan> elastic_plans;
    std::vector<std::shared_ptr<QsaElasticView>> elastic_pools;
    std::vector<QsaElasticView*> elastic_views;
    int device = -1;
    std::unique_ptr<SessionState> session;
    SessionInitUploads uploads;
    SessionPrivatePlan plan;
    void* arena = nullptr;
    void* ple_embedding = nullptr;
    void* ple_scratch = nullptr;
    void* ple_native_key = nullptr;
    std::vector<float> ple_host;
};
struct Storage {
    std::shared_ptr<const ReservationLease> reservation;
    std::vector<Stage> stages;
    bool uncertain = false;
    Storage* next = nullptr;
};
Storage* quarantined = nullptr;
uint64_t quarantine_count = 0;
bool allocator_failed = false;
template<class... T> bool all_null(T... pointers) { return ((!pointers)&&...); }
bool fresh_session(const SessionState& s) {
    const auto& a=s.gdn;const auto& q=s.qsa_bufs;const auto& m=s.moe;const auto& b=s.block;
    const auto& p=s.ple;const auto& w=p.w;
    // Explicit resource/alias checks: never inspect object padding or take over
    // a caller's device allocations, IO reader, weights, or mutable scratch.
    return all_null(s.gdn_state,s.qsa_states,s.qsa_state_arena,s.qsa_buf_arena,s.moe_arena,s.block_arena,s.R,s.db,s.ple_hist,
        a.x_q8k,a.x_q8_0,a.x_bf16,a.qkv,a.conv_out,a.h,a.alpha,a.beta,a.gate,a.o,a.z,a.y,a.y_q8k,a.y_q8_0,a.state,a.conv_state,
        q.x_q8_0,q.x_q8k,q.x_bf16,q.q_full,q.qcur,q.kcur,q.vcur,q.idx_raw,q.q_idx,q.cell_scores,q.ids,q.k_scratch,q.v_scratch,
        q.attn,q.attn16,q.attn32,q.attn_q8k,q.attn_scratch,
        m.x_bf16,m.x_f16,m.logits,m.ids,m.weights,m.shared,m.sh_scratch,m.x_q8_0,m.x_q8k,
        b.R,b.mixed,b.block_out,b.inject,b.inject2,b.gr_rs,b.head_q8k,b.dump,b.gr.xn,b.gr.xq,b.gr.lq,b.gr.gated,b.gr.lo,
        p.table,p.token,p.prev,p.hist,p.emb_dev,p.emb_host,p.scratch,
        w.key_codes,w.key_scales,w.value_bf16,w.norm_key,w.norm_query,w.norm_conv,w.conv1d_f16,
        w.key_native_data,w.key_native_q8_1,w.key_bf16,w.pre_key,w.pre_value) && !p.read_context &&
        s.max_cells==0&&s.qsa_alloc==0&&s.gdn_alloc==0&&b.gr.bytes==0;
}
bool add(uint64_t& total,uint64_t amount) {
    if(amount>UINT64_MAX-total)return false;
    total+=amount;return true;
}
bool release(Storage& storage) {
    for(auto& stage:storage.stages) {
        if(!stage.session)continue; // host preparation refused before ownership transfer
        if(!stage.arena&&!stage.session->qsa_states&&!stage.ple_embedding&&!stage.ple_scratch&&!stage.ple_native_key)continue;
        WindowDeviceScope device(stage.device);
        if(!device.ok())return false;
        std::string error;
        auto& session=*stage.session;
        // Join the window's pending PLE read before its mutable runtime buffers
        // are released. Shared table/cache storage is retained by the context.
        session.ple.read_context.reset();
        for(int64_t j=0;session.qsa_states&&j<session.qsa_alloc;++j)
            if(!qsa_state_release_host(session.qsa_states[session.qsa_ord0+j],error))return false;
        session_release(session);
        // Real owners remain held until all borrowers have closed. Elastic
        // retirement itself checks device drain before unmapping any KV tensor.
        for(auto& pool:stage.elastic_pools)if(!pool->retire_checked())return false;
        // Callers retain owner handles through all device/host/IO use. A failed
        // release is not evidence that the other allocations are safe to free.
        for(void** pointer:{&stage.ple_native_key,&stage.ple_scratch,&stage.ple_embedding,&stage.arena}) {
            if(*pointer) {
                if(cudaFree(*pointer)!=cudaSuccess)return false;
                *pointer=nullptr;
            }
        }
        if(!device.finish())return false;
        delete[] session.qsa_states;session.qsa_states=nullptr;
    }
    return true;
}
void destroy(Storage* storage) noexcept {
    std::lock_guard operation(operation_mutex);
    bool uncertain;
    { std::lock_guard lock(allocator_mutex);uncertain=storage->uncertain; }
    bool safe=false;
    try { safe=!uncertain&&release(*storage); } catch(...) {}
    if(safe) {delete storage;return;}
    // Intrusive quarantine requires no allocation on an already failing path.
    std::lock_guard lock(allocator_mutex);
    allocator_failed=true;
    storage->next=quarantined;quarantined=storage;++quarantine_count;
}
}
uint64_t native_session_quarantine_count() {
    std::lock_guard lock(allocator_mutex);return quarantine_count;
}
NativeAllocationResult allocate_native_sessions(const ModelGeometry& g,int64_t cells,int64_t k,
    const std::vector<NativeSessionRequest>& requests,WindowBudget& budget,NativeSessionAllocation& out) {
    if(requests.empty()||requests.size()>64)return {NativeAllocationStatus::invalid,"invalid native stage count"};
    try {
        std::vector<std::unique_ptr<SessionState>> fresh;
        fresh.reserve(requests.size());
        for(size_t i=0;i<requests.size();++i)fresh.push_back(std::make_unique<SessionState>());
        return allocate_native_sessions(g,cells,k,requests,fresh,budget,out);
    } catch(const std::bad_alloc&) {
        return {NativeAllocationStatus::budget_refused,"native session host memory unavailable"};
    }
}
NativeAllocationResult allocate_native_sessions(const ModelGeometry& g,int64_t cells,int64_t k,
    const std::vector<NativeSessionRequest>& requests,std::vector<std::unique_ptr<SessionState>>& fresh,
    WindowBudget& budget,NativeSessionAllocation& out) {
    std::lock_guard operation(operation_mutex);
    {
        std::lock_guard lock(allocator_mutex);
        if(allocator_failed)return {NativeAllocationStatus::unhealthy,"native allocator quarantined; worker restart required"};
    }
    if(requests.empty()||requests.size()>64)return {NativeAllocationStatus::invalid,"invalid native stage count"};
    if(fresh.size()!=requests.size())return {NativeAllocationStatus::invalid,"fresh native session count mismatch"};
    for(size_t i=0;i<fresh.size();++i) {
        if(!fresh[i]||!fresh_session(*fresh[i]))return {NativeAllocationStatus::invalid,"native session is not fresh and unbound"};
        for(size_t j=0;j<i;++j)if(fresh[i].get()==fresh[j].get())
            return {NativeAllocationStatus::invalid,"duplicate native session owner"};
    }
    bool attempted=false;
    try {
        auto storage=std::shared_ptr<Storage>(new Storage,destroy);
        storage->stages.resize(requests.size());
        WindowResources resources;
        resources.device_bytes.resize(budget.snapshot().capacity.device_bytes.size());resources.shadows=1;
        int64_t previous=0;std::set<int> devices;
        for(size_t i=0;i<requests.size();++i) {
            const auto& request=requests[i];auto& stage=storage->stages[i];
            if(request.device<0||size_t(request.device)>=resources.device_bytes.size()||!devices.insert(request.device).second||
               request.layer_begin!=previous||request.layer_end<=request.layer_begin||request.layer_end>g.n_layers||
               (request.ple_buffers&&!(request.layer_begin<=1&&request.layer_end>1))||
               (request.ple_native_key&&!request.ple_buffers))
                return {NativeAllocationStatus::invalid,"invalid native stage topology"};
            previous=request.layer_end;stage.device=request.device;
            const QsaElasticPlan* elastic=nullptr;
            if(!request.elastic_plans.empty()) {
                if(requests.size()!=1)return {NativeAllocationStatus::invalid,"owned elastic multi-device domain support pending"};
                if(g.qsa_interval<=0||!request.prepare_elastic)return {NativeAllocationStatus::invalid,"invalid elastic quote/factory"};
                const auto count=std::max<int64_t>(1,request.layer_end/g.qsa_interval-request.layer_begin/g.qsa_interval);
                if(request.elastic_plans.size()!=size_t(count))return {NativeAllocationStatus::invalid,"elastic QSA pool count mismatch"};
                stage.elastic_plans=request.elastic_plans;stage.elastic_views.reserve(stage.elastic_plans.size());
                for(const auto& plan:stage.elastic_plans)if(plan.device!=request.device||plan.cells!=cells||!qsa_elastic_plan_matches(g,plan))
                    return {NativeAllocationStatus::invalid,"elastic quote geometry mismatch"};
                elastic=&stage.elastic_plans[0];
                for(const auto& plan:stage.elastic_plans)if(plan.format!=elastic->format||plan.rotate!=elastic->rotate||plan.granularity!=elastic->granularity)
                    return {NativeAllocationStatus::invalid,"elastic pools disagree on session layout"};
                if(stage.elastic_plans.size()>UINT64_MAX/(sizeof(QsaElasticPlan)*2+64)||
                   !add(resources.host_bytes,uint64_t(stage.elastic_plans.size())*(sizeof(QsaElasticPlan)*2+64)))
                    return {NativeAllocationStatus::invalid,"elastic owner metadata overflow"};
            } else if(request.prepare_elastic)return {NativeAllocationStatus::invalid,"elastic factory without quote"};
            if(!session_private_plan(g,cells,k,request.layer_begin,request.layer_end,stage.plan,elastic))
                return {NativeAllocationStatus::invalid,"unsupported native session allocation profile"};
            if(!add(resources.device_bytes[size_t(request.device)],stage.plan.device_bytes)||
               !add(resources.host_bytes,stage.plan.host_bytes)||!add(resources.pinned_bytes,stage.plan.pinned_bytes))
                return {NativeAllocationStatus::invalid,"native allocation size overflow"};
            if(request.ple_native_key&&!add(resources.device_bytes[size_t(request.device)],
                strata::kernels::native_q8_1_bytes(strata::kernels::NG_N_EMBD)))
                return {NativeAllocationStatus::invalid,"native PLE key allocation size overflow"};
            if(request.ple_buffers) {
                const auto embedding=uint64_t(strata::kernels::NG_N_EMBD)*sizeof(float);
                if(!add(resources.device_bytes[size_t(request.device)],embedding)||
                   !add(resources.device_bytes[size_t(request.device)],ple_run_scratch_bytes())||
                   !add(resources.host_bytes,embedding*2))return {NativeAllocationStatus::invalid,"PLE allocation size overflow"};
            }
        }
        if(previous!=g.n_layers)return {NativeAllocationStatus::invalid,"native stages must cover the complete model"};
        auto reserved=budget.reserve(resources,ReservationPurpose::private_work);
        if(!reserved)return {NativeAllocationStatus::budget_refused,"native gang capacity unavailable"};
        storage->reservation=reserved.lease;
        // Allocate the admitted host embedding before transferring session
        // ownership or issuing device work. Allocation failure can then unwind
        // normally instead of quarantining an already activated gang.
        for(size_t i=0;i<requests.size();++i)
            if(requests[i].ple_buffers)
                storage->stages[i].ple_host.resize(size_t(strata::kernels::NG_N_EMBD));
        // Every physical/metadata admission is host-only, before any CUDA, VA,
        // mapping or session ownership transfer. A failed later stage unwinds all
        // earlier prepared owners and their genuine parent attachments.
        for(size_t i=0;i<requests.size();++i) {
            auto& stage=storage->stages[i];if(stage.elastic_plans.empty())continue;
            std::string error;bool prepared=false;
            try {prepared=requests[i].prepare_elastic(stage.elastic_plans,budget,reserved.lease,stage.elastic_pools,error);}
            catch(...) {return {NativeAllocationStatus::budget_refused,"elastic host preparation threw before device work"};}
            if(!prepared||stage.elastic_pools.size()!=stage.elastic_plans.size())return {NativeAllocationStatus::budget_refused,"elastic host preparation refused: "+error};
            for(size_t j=0;j<stage.elastic_pools.size();++j) {
                const auto& pool=stage.elastic_pools[j];
                if(!pool||pool->healthy()||pool->base()||!qsa_elastic_plan_equal(pool->plan(),stage.elastic_plans[j]))
                    return {NativeAllocationStatus::invalid,"elastic factory changed admitted quote"};
                for(const auto& prior:storage->stages)for(auto* used:prior.elastic_views)if(used==pool.get())
                    return {NativeAllocationStatus::invalid,"elastic pool duplicated across sessions"};
                stage.elastic_views.push_back(pool.get());
            }
        }
        // Build all potentially allocating public handles before issuing CUDA
        // work, so publication allocation cannot strand completed allocations.
        NativeSessionAllocation candidate;candidate.reservation_=reserved.lease;
        const std::weak_ptr<Storage> weak=storage;
        candidate.healthy_=[weak] {
            const auto owner=weak.lock(); // must be destroyed after the state lock
            std::lock_guard lock(allocator_mutex);
            if(!owner||owner->uncertain||allocator_failed)return false;
            for(const auto& stage:owner->stages)for(const auto& pool:stage.elastic_pools)if(!pool->healthy())return false;
            return true;
        };
        candidate.poison_=[weak] {
            const auto owner=weak.lock();
            std::lock_guard lock(allocator_mutex);
            if(owner){owner->uncertain=true;allocator_failed=true;}
        };
        candidate.stages_.reserve(requests.size());
        for(size_t i=0;i<storage->stages.size();++i)
            candidate.stages_.push_back({storage->stages[i].device,fresh[i].get(),storage});
        // No-throw ownership handoff after reservation and public handle prep.
        // Every stage transfers before any device work; partial initialization
        // retains the complete gang, including stages not yet initialized.
        for(size_t i=0;i<storage->stages.size();++i)storage->stages[i].session=std::move(fresh[i]);
        { std::lock_guard lock(allocator_mutex);storage->uncertain=true; }
        attempted=true;
        for(size_t i=0;i<requests.size();++i) {
            const auto& request=requests[i];auto& stage=storage->stages[i];
            WindowDeviceScope device(stage.device);
            if(!device.ok())return {NativeAllocationStatus::unhealthy,"native allocation device selection failed"};
            for(auto& pool:stage.elastic_pools) {
                std::string error;
                if(!pool->initialize_checked(error))return {NativeAllocationStatus::unhealthy,"elastic QSA initialization failed: "+error};
            }
            if(cudaMalloc(&stage.arena,size_t(stage.plan.device_bytes))!=cudaSuccess)
                return {NativeAllocationStatus::unhealthy,"native session arena allocation failed"};
            const auto used=stage.elastic_pools.empty()
                ? session_init_private(g,cells,k,stage.arena,*stage.session,stage.uploads,request.layer_begin,request.layer_end)
                : session_init_private(g,cells,k,stage.arena,*stage.session,stage.uploads,request.layer_begin,request.layer_end,
                    &stage.elastic_pools[0]->plan(),stage.elastic_views.data(),stage.elastic_views.size());
            if(!used||used>stage.plan.device_bytes)return {NativeAllocationStatus::unhealthy,"native session initialization failed: "+stage.uploads.error};
            if(request.ple_buffers) {
                if(cudaMalloc(&stage.ple_embedding,stage.ple_host.size()*sizeof(float))!=cudaSuccess||
                   cudaMalloc(&stage.ple_scratch,size_t(ple_run_scratch_bytes()))!=cudaSuccess)
                    return {NativeAllocationStatus::unhealthy,"native PLE allocation failed"};
                if(request.ple_native_key&&cudaMalloc(&stage.ple_native_key,
                    strata::kernels::native_q8_1_bytes(strata::kernels::NG_N_EMBD))!=cudaSuccess)
                    return {NativeAllocationStatus::unhealthy,"private native PLE key allocation failed"};
                auto& ple=stage.session->ple;
                ple.w.key_native_q8_1=stage.ple_native_key;
                ple.token=&stage.session->ple_token;ple.prev=stage.session->ple_prev;ple.hist=stage.session->ple_hist;
                ple.emb_host=stage.ple_host.data();ple.emb_dev=static_cast<float*>(stage.ple_embedding);
                ple.scratch=static_cast<float*>(stage.ple_scratch);
                // table and weights intentionally remain unbound: PleTable has
                // mutable pending IO and cannot be blindly copied from live state.
            }
            if(cudaDeviceSynchronize()!=cudaSuccess||!device.finish())
                return {NativeAllocationStatus::unhealthy,"native allocation drain/device restoration failed"};
        }
        {
            std::lock_guard lock(allocator_mutex);
            // Poison can arrive while device work drains. Never publish that work.
            if(allocator_failed)return {NativeAllocationStatus::unhealthy,"native allocator poisoned during initialization"};
            storage->uncertain=false;
        }
        out=std::move(candidate);
        return {NativeAllocationStatus::allocated,{}};
    } catch(const std::bad_alloc&) {
        return {attempted?NativeAllocationStatus::unhealthy:NativeAllocationStatus::budget_refused,
                "native allocation host memory unavailable"};
    }
}
} // namespace strata::core
