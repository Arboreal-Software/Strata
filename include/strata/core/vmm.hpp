#pragma once
// CUDA virtual memory (the elastic K/V, --kv-grow): an address range reserved once, whose physical memory is mapped and
// unmapped in chunks of the device's granularity (2 MiB).  The K/V pools grow with the context by taking chunks the
// expert cache gives up, and neither moves: every captured graph and stored pointer stays valid.  The driver calls
// are reached through the runtime's entry-point query, so nothing links the driver library.  Not under HIP.

#include <cstdint>
#include <vector>
#include <memory>
#include <span>
#include <string>
#include "strata/core/window_budget.hpp"

namespace strata::core {

/// A physical chunk: the driver's allocation handle (CUmemGenericAllocationHandle); 0 = none.
using VmmChunk = unsigned long long;

/// The current device supports it and the entry points resolved.
bool vmm_available();
/// The chunk size (0 when not available).
uint64_t vmm_granularity();
/// A new physical chunk on the current device (0: out of memory), and its release.
VmmChunk vmm_chunk_new();
void vmm_chunk_free(VmmChunk h);
void vmm_chunk_retire(VmmChunk h, bool strict);

// Managed calls require one serialized owner reactor and checked quiescence.
// They do not infer that a GPU consumer has stopped from a host token.
class DetachedVmmChunk {
public:
    DetachedVmmChunk() = default;
    ~DetachedVmmChunk();
    DetachedVmmChunk(const DetachedVmmChunk&) = delete;
    DetachedVmmChunk& operator=(const DetachedVmmChunk&) = delete;
    DetachedVmmChunk(DetachedVmmChunk&&) noexcept;
    DetachedVmmChunk& operator=(DetachedVmmChunk&&) = delete;
    bool empty() const noexcept { return handle_ == 0; }
    uint64_t incarnation() const noexcept { return incarnation_; }
    // Reserves a genuine device charge before creating physical memory. Output
    // must be empty; failed/unknown device release is fail-stop, never retried.
    static bool allocate(WindowBudget&, int device, DetachedVmmChunk&, std::string&);
    bool retire() noexcept;
private:
    friend class VmmRange;
    friend class PreparedVmmAllocationBatch;
    static bool allocation_domain(int device) noexcept;
    VmmChunk handle_ = 0;
    uint64_t incarnation_ = 0;
    int device_ = -1;
    bool fresh_ = true;
    std::shared_ptr<const ReservationLease> lease_;
};
struct VmmBatchDevicePlan {
    int physical_device=-1;
    size_t budget_index=0;
    uint64_t chunks=0,expected_granularity=0;
};
struct VmmBatchResult {
    enum Code { complete, refused_before_effect, failed_after_prefix, unknown };
    Code code=refused_before_effect;
    size_t completed=0;
};
// Host-only preparation. Caller authenticates physical-device/budget-index mapping
// and completes ALL independent admissions before allocate_device. First slice
// accepts one domain; no implicit multi-device support from the singleton backend.
class PreparedVmmAllocationBatch {
public:
    static bool host_bound(uint64_t device_entries,uint64_t& bytes) noexcept;
    static bool prepare(std::span<const VmmBatchDevicePlan>,
        std::shared_ptr<const ReservationLease> physical_exclusive,
        std::shared_ptr<const ReservationLease> metadata_exclusive,
        std::unique_ptr<PreparedVmmAllocationBatch>& empty_out,std::string& error);
    VmmBatchResult allocate_device(size_t index,std::span<DetachedVmmChunk>) noexcept;
    PreparedVmmAllocationBatch(const PreparedVmmAllocationBatch&)=delete;
    PreparedVmmAllocationBatch& operator=(const PreparedVmmAllocationBatch&)=delete;
    PreparedVmmAllocationBatch(PreparedVmmAllocationBatch&&)=delete;
    PreparedVmmAllocationBatch& operator=(PreparedVmmAllocationBatch&&)=delete;
private:
    PreparedVmmAllocationBatch()=default;
    std::shared_ptr<const ReservationLease> metadata_,physical_;
    VmmBatchDevicePlan plan_;
    uint64_t first_incarnation_=0;
    bool consumed_=false;
};
enum class VmmDisposition { not_attempted, access_ready, detached, retired, unknown };
struct VmmChunkOutcome {
    int64_t index = -1;
    uint64_t incarnation = 0;
    bool fresh = false;
    VmmDisposition disposition = VmmDisposition::not_attempted;
};

class VmmRange {
public:
    VmmRange() = default;
    static bool managed_host_bound(uint64_t chunks, uint64_t& bytes) noexcept;
    // Host-only preadmission: both bookkeeping arrays allocated before VA/device effects.
    bool prepare_managed_metadata(uint64_t chunks,uint64_t expected_granularity,
        std::shared_ptr<const ReservationLease>,int device,std::string&);
    // Once-only VA activation; actual domain/granularity validated before reserve.
    bool activate_managed() noexcept;
    // Compatibility API for an already reserved empty range; not host-only.
    // Metadata lease must be dedicated to this range.
    bool prepare_managed(std::shared_ptr<const ReservationLease>, int device, std::string&);
    bool map_managed(int64_t lo, int64_t hi, std::span<DetachedVmmChunk>,
                     std::span<VmmChunkOutcome>) noexcept;
    bool detach_managed(int64_t index, DetachedVmmChunk&, VmmChunkOutcome&) noexcept;
    bool managed_healthy() const noexcept { return managed_ && base_ != 0 && !managed_unknown_; }
    ~VmmRange() { release(); }
    void set_strict_retirement(bool enabled, int device) { if (managed_) return; strict_retirement_ = enabled; retirement_device_ = device; }
    VmmRange(const VmmRange&) = delete;
    VmmRange& operator=(const VmmRange&) = delete;

    /// Reserves `bytes` (rounded up to whole chunks) of addresses, nothing mapped.
    bool reserve(uint64_t bytes);
    /// Unmaps and frees every chunk, then the addresses.
    void release();
    uint8_t* base() const { return (uint8_t*) (uintptr_t) base_; }
    int64_t chunks() const { return (int64_t) h_.size(); }
    bool mapped(int64_t i) const { return i >= 0 && i < chunks() && h_[(std::size_t) i] != 0; }
    int64_t mapped_count() const;
    /// Maps chunks [lo, hi) that are not mapped yet, each to `take()` or, when that returns 0, to a new chunk.
    /// Readable and writable when it returns; false: out of memory (what was mapped stays mapped).
    template <class Take> bool map_range(int64_t lo, int64_t hi, Take take);
    /// Unmaps chunk i and hands back its physical chunk (0 when it was not mapped).
    VmmChunk unmap(int64_t i);

private:
    struct ManagedSlot {
        uint64_t incarnation = 0;
        bool fresh = false;
        std::shared_ptr<const ReservationLease> lease;
    };
    std::shared_ptr<const ReservationLease> metadata_;
    std::vector<ManagedSlot> managed_slots_;
    bool managed_ = false, managed_unknown_ = false;
    bool managed_internal_ = false;
    bool managed_prepared_ = false, managed_activation_attempted_ = false;
    uint64_t managed_granularity_ = 0;
    void release_managed_metadata() noexcept;
    int managed_device_ = -1;
    bool strict_retirement_ = false;
    int retirement_device_ = -1;
    bool map_one(int64_t i, VmmChunk h);
    bool set_access(int64_t lo, int64_t hi);
    /// Readable and writable [lo, hi) (all newly mapped); when that fails they are unmapped and freed again, so no
    /// chunk counts as mapped without access.
    bool commit_run(int64_t lo, int64_t hi);
    unsigned long long base_ = 0;
    std::vector<VmmChunk> h_;   // per chunk: its physical chunk, 0 = unmapped
};

template <class Take> bool VmmRange::map_range(int64_t lo, int64_t hi, Take take) {
    if (managed_) return false;
    lo = lo < 0 ? 0 : lo;
    hi = hi > chunks() ? chunks() : hi;
    int64_t run = -1;   // access is set once per run of newly mapped chunks (it costs as much as the map)
    for (int64_t i = lo; i < hi; ++i) {
        if (mapped(i)) {
            if (run >= 0 && !commit_run(run, i)) return false;
            run = -1;
            continue;
        }
        VmmChunk h = take();
        if (h == 0) h = vmm_chunk_new();
        if (h == 0 || !map_one(i, h)) {
            if (h != 0) vmm_chunk_retire(h, strict_retirement_);
            if (run >= 0) commit_run(run, i);
            return false;
        }
        if (run < 0) run = i;
    }
    return run < 0 || commit_run(run, hi);
}

}  // namespace strata::core
