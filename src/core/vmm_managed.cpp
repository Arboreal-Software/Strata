// Optional window-state allocation entry point. Default VMM has no WindowBudget link dependency.
#include "strata/core/vmm.hpp"
#include <atomic>
#include <new>
#include <utility>
#include <cuda_runtime.h>

namespace strata::core {
namespace {
std::atomic<uint64_t> managed_next{1};
uint64_t mint_chunks(uint64_t count) noexcept {
    auto n = managed_next.load();
    while (n && count && count <= UINT64_MAX-n) {
        if (managed_next.compare_exchange_weak(n,n+count)) return n;
    }
    return 0;
}
}
bool DetachedVmmChunk::allocate(WindowBudget& budget,int device,DetachedVmmChunk& out,std::string& error) {
    if (!out.empty() || out.retirement_generation_ || !allocation_domain(device)) { error="managed allocation domain/output mismatch"; return false; }
    // This first API explicitly requires budget slots indexed by physical device
    // ordinal. Remapped logical budgets must use a future validated mapping API.
    try {
        auto snapshot=budget.snapshot();
        if (device<0 || (size_t)device>=snapshot.capacity.device_bytes.size()) {
            error="physical device absent from budget"; return false;
        }
        WindowResources charge;
        charge.device_bytes.resize(snapshot.capacity.device_bytes.size());
        charge.device_bytes[(size_t)device]=vmm_granularity();
        auto admitted=budget.reserve(charge,ReservationPurpose::foreground);
        if (!admitted) { error="managed physical quota refused"; return false; }
        const auto id=mint_chunks(1);
        if (!id) { error="physical incarnation exhausted"; return false; }
        const auto h=vmm_chunk_new();
        if (!h) { error="physical allocation refused"; return false; }
        out.lease_=std::move(admitted.lease);
        out.device_=device;out.incarnation_=id;out.fresh_=true;out.handle_=h;
        return true;
    } catch(const std::bad_alloc&) { error.clear();return false; }
}
bool PreparedVmmAllocationBatch::host_bound(uint64_t entries,uint64_t& bytes) noexcept {
    if(entries!=1)return false;
    bytes=4096+sizeof(PreparedVmmAllocationBatch);return true;
}
bool PreparedVmmAllocationBatch::prepare(std::span<const VmmBatchDevicePlan> plans,
    std::shared_ptr<const ReservationLease> physical,
    std::shared_ptr<const ReservationLease> metadata,
    std::unique_ptr<PreparedVmmAllocationBatch>& out,std::string& error) {
    try {
        uint64_t host=0;
        if(out||!host_bound(plans.size(),host)||!physical||!metadata||
           physical.use_count()!=1||metadata.use_count()!=1||physical.get()==metadata.get()||
           metadata->purpose()!=ReservationPurpose::foreground||
           (physical->purpose()!=ReservationPurpose::foreground&&physical->purpose()!=ReservationPurpose::private_attachment)) {
            error="prepaid VMM requires exclusive dedicated admissions and one domain";return false;
        }
        const auto& plan=plans[0];const auto& r=physical->resources();const auto& m=metadata->resources();
        if(plan.physical_device<0||!plan.chunks||plan.chunks>SIZE_MAX||!plan.expected_granularity||
           plan.chunks>UINT64_MAX/plan.expected_granularity||plan.budget_index>=r.device_bytes.size()||
           r.host_bytes||r.pinned_bytes||r.disk_bytes||r.io_slots||r.shadows||
           m.host_bytes<host||m.pinned_bytes||m.disk_bytes||m.io_slots||m.shadows) {
            error="prepaid VMM plan or resource envelope invalid";return false;
        }
        for(size_t i=0;i<r.device_bytes.size();++i)
            if(r.device_bytes[i]!=(i==plan.budget_index?plan.chunks*plan.expected_granularity:0)) {
                error="prepaid VMM physical bytes differ from exact plan";return false;
            }
        for(auto n:m.device_bytes)if(n){error="prepaid VMM metadata contains physical charge";return false;}
        auto candidate=std::unique_ptr<PreparedVmmAllocationBatch>(new PreparedVmmAllocationBatch);
        const auto first=mint_chunks(plan.chunks);
        if(!first){error="prepaid VMM allocation incarnations exhausted";return false;}
        candidate->plan_=plan;candidate->first_incarnation_=first;
        candidate->metadata_=std::move(metadata);candidate->physical_=std::move(physical);
        out=std::move(candidate);return true;
    }catch(const std::bad_alloc&){error.clear();return false;}
}
VmmBatchResult PreparedVmmAllocationBatch::allocate_device(size_t index,std::span<DetachedVmmChunk> outputs) noexcept {
    if(index||consumed_||outputs.size()!=plan_.chunks)return {};
    for(const auto& out:outputs)if(!out.empty()||out.retirement_generation_)return {};
    // All independent caller admissions must already be complete. Backend lazy
    // discovery can allocate here, but finishes before the FIRST physical create.
    // Catch only this pre-effect phase; never reinterpret a partial create prefix.
    try {
        if(!DetachedVmmChunk::allocation_domain(plan_.physical_device)||vmm_granularity()!=plan_.expected_granularity)return {};
    }catch(...){return {};}
    consumed_=true;
    size_t done=0;
    for(auto& out:outputs) {
        const auto h=vmm_chunk_new();
        if(!h)return {VmmBatchResult::failed_after_prefix,done};
        out.lease_=physical_;out.device_=plan_.physical_device;
        out.incarnation_=first_incarnation_+done;out.fresh_=true;out.handle_=h;++done;
    }
    return {VmmBatchResult::complete,done};
}
} // namespace strata::core
