#include "strata/core/remote_experts.hpp"
#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/core/remote_expert_opt.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;

// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char* failed_step = nullptr;
    cudaError_t* restore_status = nullptr;
    explicit DeviceScope(int device, cudaError_t* restored = nullptr) : restore_status(restored) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) {
            previous = -1;
            failed_step = "cudaGetDevice";
            return;
        }
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            failed_step = "cudaSetDevice";
            return;
        }
        ok = true;
    }
    cudaError_t restore() noexcept {
        const int target=std::exchange(previous,-1);
        const auto result=target>=0?cudaSetDevice(target):cudaSuccess;
        if (restore_status) *restore_status=result;
        return result;
    }
    ~DeviceScope() { if (previous >= 0) (void) restore(); }
    std::string error(int device) const {
        return std::string("CUDA") + std::to_string(device) + " experts: " +
               (failed_step ? failed_step : "device switch") +
               "(" + std::to_string(device) + ") failed: " + cudaGetErrorString(status) +
               " (CUDA error " + std::to_string((int) status) + ")";
    }
};

bool check(cudaError_t result, const char* what, std::string& err, int device) {
    if (result == cudaSuccess) return true;
    err = "CUDA" + std::to_string(device) + " experts: " + what + ": " + cudaGetErrorString(result);
    return false;
}
} // namespace

// BEGIN owned remote routing admission (source-extracted CPU fixtures)
namespace {
std::atomic<uint64_t> remote_quarantines{0};
[[noreturn]] void remote_ownership_fatal() noexcept {
    std::fputs("strata: remote helper ownership or DMA completion uncertain; worker restart required\n", stderr);
    std::fflush(stderr);
    std::_Exit(1);
}
}
uint64_t RemoteExperts::owned_routing_quarantines() noexcept { return remote_quarantines.load(); }
bool RemoteExperts::bound_to_runtime(const std::shared_ptr<NativeRuntimeAttachment>& runtime) const noexcept {
    std::shared_ptr<NativeRuntimeAttachment> observed;
    bool available=false;
    {
        std::lock_guard<std::mutex> lock(ownership_mu_);
        available=owned_bound_&&!owned_unknown_&&!owned_closed_;
        if(available)observed=runtime_owner_.lock();
    }
    // Promotion may be the last reference to another owner. Destroy it only
    // after releasing ownership_mu_, including every mismatch/refusal path.
    return available&&runtime&&observed&&observed.get()==runtime.get()&&
           !observed.owner_before(runtime)&&!runtime.owner_before(observed);
}
void RemoteExperts::unknown_locked(const std::shared_ptr<NativeRuntimeAttachment>& owner) noexcept {
    if (!owned_unknown_) {
        owned_unknown_ = true;
        // This is deliberately a retained active cycle if owner owns this helper.
        // No allocation and no overwritten shared owner/destructor under the lock.
        unknown_owner_ = owner;
        remote_quarantines.fetch_add(1, std::memory_order_relaxed);
    }
}
bool RemoteExperts::set_strict_retirement(bool enabled,const std::vector<int>& devices,std::string& error) {
    // Startup only: no producers or concurrent open/bind are permitted here.
    if (owned_bound_ || remote_open_started_) {
        error="remote routing: strict retirement must precede open"; return false;
    }
    if (!cache_.set_strict_retirement(enabled,devices,error)) return false;
    strict_retirement_configured_=enabled;
    return true;
}
bool RemoteExperts::copy_reader_guard_bytes(uint32_t readers, uint64_t& bytes) noexcept {
    if (!readers || uint64_t(readers) > UINT64_MAX / uint64_t(sizeof(CopyRead))) return false;
    bytes = uint64_t(readers) * uint64_t(sizeof(CopyRead));
    return true;
}

bool RemoteExperts::bind_owned_routing(const std::shared_ptr<NativeRuntimeAttachment>& owner,
                                      uint32_t max_copy_readers, std::string& error) {
    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(ownership_mu_);
        if (!owned_bound_ && strict_retirement_configured_ && owner && max_copy_readers && device_ >= 0 && stream_ && read_stream_) {
            runtime_owner_ = owner;
            max_copy_readers_ = max_copy_readers;
            owned_bound_ = true;
            accepted = true;
        }
    }
    if (!accepted) error = "remote routing: invalid or repeated owner binding";
    return accepted;
}
bool RemoteExperts::owned_routing_healthy() const noexcept {
    std::lock_guard<std::mutex> lock(ownership_mu_);
    return owned_bound_ && !owned_unknown_ && !owned_closed_ && !runtime_owner_.expired();
}
RemoteExperts::RoutingRead::~RoutingRead() noexcept { reset(); }
RemoteExperts::RoutingRead::RoutingRead(RoutingRead&& other) noexcept {
    remote_=std::exchange(other.remote_,nullptr); generation_=std::exchange(other.generation_,0);
    owner_=std::move(other.owner_);
}
RemoteExperts::RoutingRead& RemoteExperts::RoutingRead::operator=(RoutingRead&& other) noexcept {
    if (this != &other) {
        reset(); remote_=std::exchange(other.remote_,nullptr); generation_=std::exchange(other.generation_,0);
        owner_=std::move(other.owner_);
    }
    return *this;
}
void RemoteExperts::RoutingRead::reset() noexcept {
    if (remote_) remote_->end_routing(generation_);
    remote_=nullptr; generation_=0; owner_.reset(); // outside remote mutex/method
}
RemoteExperts::PlacementWrite::~PlacementWrite() noexcept { reset(); }
RemoteExperts::PlacementWrite::PlacementWrite(PlacementWrite&& other) noexcept {
    remote_=std::exchange(other.remote_,nullptr); generation_=std::exchange(other.generation_,0);
    owner_=std::move(other.owner_);
}
RemoteExperts::PlacementWrite& RemoteExperts::PlacementWrite::operator=(PlacementWrite&& other) noexcept {
    if (this != &other) {
        reset(); remote_=std::exchange(other.remote_,nullptr); generation_=std::exchange(other.generation_,0);
        owner_=std::move(other.owner_);
    }
    return *this;
}
void RemoteExperts::PlacementWrite::reset() noexcept {
    if (remote_) remote_->end_writer(generation_,owner_);
    remote_=nullptr; generation_=0; owner_.reset();
}
RemoteExperts::Admission RemoteExperts::try_routing_read(RoutingRead& out) noexcept {
    if (out) return Admission::invalid;
    std::lock_guard<std::mutex> lock(ownership_mu_);
    if (!owned_bound_) return Admission::unbound;
    if (owned_unknown_ || owned_closed_) return Admission::unknown;
    if (routing_generation_ || writer_generation_ || compute_generation_) return Admission::busy;
    if (next_generation_ == UINT64_MAX) return Admission::exhausted;
    auto owner=runtime_owner_.lock();
    if (!owner) return Admission::unknown;
    out.owner_=std::move(owner); out.remote_=this; out.generation_=next_generation_++;
    routing_generation_=out.generation_;
    return Admission::acquired;
}
RemoteExperts::Admission RemoteExperts::try_placement_write(PlacementWrite& out) noexcept {
    if (out) return Admission::invalid;
    std::lock_guard<std::mutex> lock(ownership_mu_);
    if (!owned_bound_) return Admission::unbound;
    if (owned_unknown_ || owned_closed_) return Admission::unknown;
    if (routing_generation_ || compute_generation_ || copy_readers_ || writer_generation_) return Admission::busy;
    if (next_generation_ == UINT64_MAX) return Admission::exhausted;
    auto owner=runtime_owner_.lock();
    if (!owner) return Admission::unknown;
    out.owner_=std::move(owner); out.remote_=this; out.generation_=next_generation_++;
    writer_generation_=out.generation_;
    writer_attempted_=writer_before_drained_=writer_copies_drained_=false;
    return Admission::acquired;
}
void RemoteExperts::end_routing(uint64_t generation) noexcept {
    std::lock_guard<std::mutex> lock(ownership_mu_);
    if (generation && routing_generation_==generation) routing_generation_=0;
}
void RemoteExperts::end_writer(uint64_t generation,const std::shared_ptr<NativeRuntimeAttachment>& owner) noexcept {
    std::lock_guard<std::mutex> lock(ownership_mu_);
    if (!generation || writer_generation_!=generation) return;
    if (writer_attempted_) unknown_locked(owner);
    writer_generation_=0;
}
bool RemoteExperts::PlacementWrite::mark_attempted() noexcept {
    if (!remote_) return false;
    std::lock_guard<std::mutex> lock(remote_->ownership_mu_);
    if (remote_->owned_unknown_ || remote_->writer_generation_!=generation_ ||
        !remote_->writer_before_drained_ || remote_->writer_attempted_) return false;
    remote_->writer_attempted_=true;
    return true;
}
bool RemoteExperts::PlacementWrite::drain_before_overwrite(std::string& error) {
    return remote_ && remote_->writer_drain(generation_,true,owner_,error);
}
bool RemoteExperts::PlacementWrite::drain_copies(std::string& error) {
    return remote_ && remote_->writer_drain(generation_,false,owner_,error);
}
bool RemoteExperts::writer_drain(uint64_t generation,bool before,
                                 const std::shared_ptr<NativeRuntimeAttachment>& owner,std::string& error) {
    {
        std::lock_guard<std::mutex> lock(ownership_mu_);
        if (owned_unknown_ || writer_generation_!=generation ||
            (before ? writer_attempted_ || writer_before_drained_ : !writer_attempted_ || writer_copies_drained_))
            return false;
    }
    // Sole writer/controller owns this operation; no state mutex across CUDA.
    // Both streams matter: the prefill cache reader has a separate stream.
    DeviceScope scope(device_);
    cudaError_t status=scope.status;
    if (scope.ok) {
        status=cudaStreamSynchronize(stream_);
        const auto read_status=cudaStreamSynchronize(read_stream_);
        if (status==cudaSuccess) status=read_status;
    }
    const auto restored=scope.restore();
    if (status==cudaSuccess) status=restored;
    {
        std::lock_guard<std::mutex> lock(ownership_mu_);
        if (status!=cudaSuccess) unknown_locked(owner);
        else if (before) writer_before_drained_=true;
        else writer_copies_drained_=true;
    }
    if (status!=cudaSuccess) {
        error="remote routing: checked stream/device drain failed";
        return false;
    }
    return true;
}
bool RemoteExperts::PlacementWrite::finish_publication() noexcept {
    if (!remote_) return false;
    {
        std::lock_guard<std::mutex> lock(remote_->ownership_mu_);
        if (remote_->owned_unknown_ || remote_->writer_generation_!=generation_ ||
            !remote_->writer_attempted_ || !remote_->writer_copies_drained_) return false;
        remote_->writer_attempted_=false;
        remote_->writer_generation_=0;
    }
    remote_=nullptr; generation_=0; owner_.reset();
    return true;
}
bool RemoteExperts::holds(int64_t layer,int32_t expert) const {
    if (!owned_bound_) return cache_.slot_of(layer,expert)>=0;
    std::lock_guard<std::mutex> lock(ownership_mu_);
    return !owned_unknown_ && !owned_closed_ && !writer_generation_ && cache_.slot_of(layer,expert)>=0;
}
bool RemoteExperts::finish_compute(bool unknown) noexcept {
    std::shared_ptr<NativeRuntimeAttachment> retired;
    {
        std::lock_guard<std::mutex> lock(ownership_mu_);
        if (unknown || owned_unknown_) { unknown_locked(compute_owner_); return false; }
        compute_generation_=0; compute_ready_=compute_finishing_=false;
        retired=std::move(compute_owner_);
    }
    // The controller holds its own runtime reference across every method call.
    return true;
}
RemoteExperts::CopyRead::~CopyRead() noexcept {
    if (remote) {
        std::lock_guard<std::mutex> lock(remote->ownership_mu_);
        --remote->copy_readers_;
    }
    // owner is destroyed after the lock; successful copies retain no cycle.
}
bool RemoteExperts::begin(int64_t layer,const float* x,const int32_t* ids,int64_t n_tok,int64_t k,
                           const int32_t* kind,const int32_t* primary_res,std::string& error) {
    if (!owned_bound_) return begin_impl(layer,x,ids,n_tok,k,kind,primary_res,error);
    // Bound the multiplication in the legacy implementation before entering it.
    if (n_tok<=0 || n_tok>strata::kernels::cpu::MAXT || k!=10) {
        error="remote routing: invalid managed routing dimensions"; return false;
    }
    bool admitted=false;
    {
        std::lock_guard<std::mutex> lock(ownership_mu_);
        if (!owned_unknown_ && !owned_closed_ && routing_generation_ && !writer_generation_ &&
            !compute_generation_ && next_generation_!=UINT64_MAX) {
            compute_owner_=runtime_owner_.lock();
            if (compute_owner_) { compute_generation_=next_generation_++; admitted=true; }
        }
    }
    if (!admitted) { error="remote routing: compute admission refused"; return false; }
    bool attempted=false;
    cudaError_t restored=cudaSuccess;
    try {
        const bool ok=begin_impl(layer,x,ids,n_tok,k,kind,primary_res,error,&attempted,&restored);
        if (!ok || restored!=cudaSuccess) {
            finish_compute(attempted || restored!=cudaSuccess);
            return false;
        }
        // Success retains the exact operation until finish, even with no hits.
        bool healthy;
        { std::lock_guard<std::mutex> lock(ownership_mu_); healthy=!owned_unknown_; compute_ready_=healthy; }
        if (!healthy) { finish_compute(true); return false; }
        return true;
    } catch (...) {
        finish_compute(attempted || restored!=cudaSuccess);
        throw;
    }
}
bool RemoteExperts::finish(float* out,std::string& error) {
    if (!owned_bound_) return finish_impl(out,error);
    bool admitted=false;
    {
        std::lock_guard<std::mutex> lock(ownership_mu_);
        if (!owned_unknown_ && !owned_closed_ && compute_generation_ && compute_ready_ && !compute_finishing_) {
            compute_finishing_=true; admitted=true;
        }
    }
    if (!admitted) { error="remote routing: finish admission refused"; return false; }
    cudaError_t restored=cudaSuccess;
    try {
        const bool ok=finish_impl(out,error,&restored);
        return finish_compute(!ok || restored!=cudaSuccess);
    } catch (...) { finish_compute(true); throw; }
}
bool RemoteExperts::copy_cached_blobs(const int32_t* layers,const int32_t* experts,uint8_t* const* dst,
                                       size_t n,uint8_t* hit,std::string& error) {
    if (!owned_bound_) return copy_cached_blobs_legacy(layers,experts,dst,n,hit,error);
    CopyRead admission;
    {
        std::lock_guard<std::mutex> lock(ownership_mu_);
        if (!owned_unknown_ && !owned_closed_ && !writer_generation_ && copy_readers_<max_copy_readers_) {
            admission.owner=runtime_owner_.lock();
            if (admission.owner) { ++copy_readers_; admission.remote=this; }
        }
    }
    if (!admission.remote) {
        std::fill(hit,hit+n,0);
        error="remote routing: cache read admission refused"; return false;
    }
    // Admission destruction is after read_mu_ release. No GPU work holds ownership_mu_.
    std::lock_guard<std::mutex> reader(read_mu_);
    std::fill(hit,hit+n,0);
    bool healthy;
    { std::lock_guard<std::mutex> lock(ownership_mu_); healthy=!owned_unknown_ && !owned_closed_; }
    if (!healthy) { error="remote routing: queued cache read became unhealthy"; return false; }
    DeviceScope scope(device_);
    bool dirty=false;
    auto drain_destination=[&]() noexcept {
        if (dirty) {
            if (cudaStreamSynchronize(read_stream_)!=cudaSuccess) remote_ownership_fatal();
            dirty=false;
        }
    };
    try {
        if (!scope.ok) {
            if (scope.restore()!=cudaSuccess) remote_ownership_fatal();
            error="remote routing: cache read device selection failed";
            return false;
        }
        bool ok=true;
        const auto& layout=strata::kernels::cpu::expert_layout();
        for (size_t i=0;i<n;++i) {
            if (!dst[i]) continue;
            const auto slot=cache_.slot_of(layers[i],experts[i]);
            if (slot<0) continue;
            dirty=true; // BEFORE even a failing submission; destination cannot escape uncertain
            const auto status=cudaMemcpyAsync(dst[i],cache_.device_slot(slot),size_t(layout.blob_bytes(layers[i])),
                                               cudaMemcpyDeviceToHost,read_stream_);
            if (status!=cudaSuccess) { ok=false; break; }
            hit[i]=1;
        }
        drain_destination();
        if (scope.restore()!=cudaSuccess) {
            { std::lock_guard<std::mutex> lock(ownership_mu_); unknown_locked(admission.owner); }
            error="remote routing: cache read device restoration failed";
            return false;
        }
        if (!ok) error="remote routing: cache read submission failed (destination drained)";
        return ok;
    } catch (...) {
        // Source/layout/diagnostic exceptions never return a still-written caller buffer.
        drain_destination();
        if (scope.restore()!=cudaSuccess) remote_ownership_fatal();
        throw;
    }
}
// END owned remote routing admission

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count) {
        err = "CUDA" + std::to_string(device) + " experts: CUDA device is not visible";
        return false;
    }
    if (device == primary_device()) {   // the primary may be any visible ordinal; the helper cannot be it
        err = "CUDA" + std::to_string(device) + " experts: the helper cannot be the primary device "
                                                 "(STRATA_PRIMARY_DEVICE)";
        return false;
    }
    // The layer waits for this GPU on the CPU pool's critical path: spin instead of sleeping, whose wake-up
    // costs more than a small expert batch takes (measured: ~0.3 ms per round trip on Windows).  Only possible
    // before the device's context exists, so first thing; STRATA_REMOTE_SPIN=0 keeps the driver's default.
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)
    const char* spin = std::getenv("STRATA_REMOTE_SPIN");
    if (!(spin && spin[0] == '0')) cudaInitDevice(device, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
    cudaGetLastError();
#endif
    // HIP has no cudaInitDevice equivalent; retain its default scheduling policy.
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "preflight free memory", err, device)) return false;
    free_gib = (double) free_bytes / 1073741824.0;
    return true;
}

void RemoteExperts::close() {
    if (owned_bound_) {
        std::lock_guard<std::mutex> lock(ownership_mu_);
        if (owned_unknown_ || routing_generation_ || compute_generation_ || writer_generation_ || copy_readers_)
            remote_ownership_fatal();
        owned_closed_=true;
    }
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (strict_retirement_configured_ && !scope.ok) remote_ownership_fatal();
    if (scope.ok) {
        // Pre-open strict configuration covers partial startup BEFORE binding.
        // Missing streams do not imply completed work or require a fake drain.
        if (stream_ && cudaStreamSynchronize(stream_)!=cudaSuccess && strict_retirement_configured_)
            remote_ownership_fatal();
        if (read_stream_ && cudaStreamSynchronize(read_stream_)!=cudaSuccess && strict_retirement_configured_)
            remote_ownership_fatal();
        cache_.close();
        if (d_x_ && cudaFree(d_x_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (d_out_ && cudaFree(d_out_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (d_q8_ && cudaFree(d_q8_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (d_scales_ && cudaFree(d_scales_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (d_xr_ && cudaFree(d_xr_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (d_scratch_ && cudaFree(d_scratch_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (d_meta_ && cudaFree(d_meta_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (h_x_ && cudaFreeHost(h_x_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (h_out_ && cudaFreeHost(h_out_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (h_meta_ && cudaFreeHost(h_meta_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (stream_ && cudaStreamDestroy(stream_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
        if (read_stream_ && cudaStreamDestroy(read_stream_)!=cudaSuccess && strict_retirement_configured_) remote_ownership_fatal();
    }
    if (strict_retirement_configured_ && scope.restore()!=cudaSuccess) remote_ownership_fatal();
    device_ = -1;
    stream_ = nullptr;
    read_stream_ = nullptr;
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_xr_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    d_scratch_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    original_row_.clear();
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked,
                         const ExpertCache& primary, ExpertSource& source,
                         std::vector<uint8_t>& claimed, std::string& err, bool auto_size, ExpertTierOpenObserver* observer) {
    if (owned_bound_) { err="remote routing: bound helper cannot reopen"; return false; }
    remote_open_started_=true;
    ExpertTierOpenTransaction observation(observer);
    if(!observation.valid(ranked.size())) {err="remote open observer storage/callbacks invalid";return false;}
    close();
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts ||
        device == primary_device()) {   // the helper cannot be the primary device (any visible ordinal)
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "free memory", err, device)) { close(); return false; }
    uint64_t needed = 0;
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) std::min<int64_t>(slots, layers * experts));
    std::vector<uint8_t> picked(claimed.size(), 0);
    bool observation_full=false;
    for (size_t rank=0;rank<ranked.size();++rank) {
        const auto& pair=ranked[rank];
        auto* decision=observation.append(rank,pair.first,pair.second);
        if(observer&&!decision){err="remote open observation overflow";close();return false;}
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (primary.slot_of(pair.first, pair.second) < 0 && !claimed[index] && !picked[index]) {
            const uint64_t bytes = lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
            if (auto_size && needed + bytes + (512ull << 20) > free_bytes) {observation_full=true;if(decision)decision->outcome=ExpertTierOpenOutcome::skipped_full;break;}
            if(decision)decision->outcome=ExpertTierOpenOutcome::pending_copy;
            selected.push_back(pair);
            needed += bytes;
            picked[index] = 1;
            if ((int) selected.size() >= slots) {observation_full=rank+1<ranked.size();break;}
        }
    }
    if (selected.empty()) {
        err = "CUDA" + std::to_string(device) + (auto_size ? " experts: no unclaimed expert fits with 512 MiB free"
                                                          : " experts: no unclaimed experts remain");
        close(); return false;
    }
    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(selected.size());
        for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    }
    // Leave room for the CUDA context, staging and later driver allocations, especially under WDDM.
    if (needed + (512ull << 20) > free_bytes) {
        err = "CUDA" + std::to_string(device) + " experts: slots leave less than 512 MiB free; reduce --expert-cache-device" + std::to_string(device);
        close(); return false;
    }
    const bool cache_ok = lay.native ? cache_.open_sized(sizes, layers, experts, err)
                                     : cache_.open((int64_t) selected.size(), layers, experts,
                                                   (int64_t) lay.max_blob, err);
    if (!cache_ok) { err = "CUDA" + std::to_string(device) + " experts: " + err; close(); return false; }
    size_t observation_cursor=0;
    for (const auto& pair : selected) {
        auto* decision=observation.selected(observation_cursor);
        if(observer&&!decision){err="remote open selected observation missing";close();return false;}
        uint64_t completed_bytes=0;
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first),observer?&completed_bytes:nullptr)) {
            err = "CUDA" + std::to_string(device) + " experts: " +
                  (err.empty() ? "cache fill failed" : err);
            close(); return false;
        }
        if(decision){decision->slot=slot;decision->bytes=completed_bytes;decision->outcome=ExpertTierOpenOutcome::copied;}
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                            err, (int64_t) lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close(); return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    const size_t meta_bytes = sizeof(RemoteMeta) + (remote_opt_ ? remote_opt_->metadata_bytes() : 0);
    const bool allocated =
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err, device) &&
        check(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err, device) &&
        check(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err, device) &&
        check(cudaHostAlloc(&h_meta_, meta_bytes, cudaHostAllocPortable), "metadata staging", err, device) &&
        check(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err, device) &&
        check(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err, device) &&
        check(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activation", err, device) &&
        check(cudaMalloc((void**) &d_scales_, (size_t) CAP * (H / 32) * sizeof(float)), "activation scales", err, device) &&
        check(cudaMalloc(&d_scratch_, scratch), "scratch", err, device) &&
        check(cudaMalloc(&d_meta_, meta_bytes), "group metadata", err, device) &&
        (!strata::kernels::cpu::hadamard_spec().any() ||
         check(cudaMalloc((void**) &d_xr_, (size_t) CAP * H * sizeof(float)), "rotated input", err, device));
    if (!allocated) { close(); return false; }
    if (!check(cudaStreamCreateWithFlags(&read_stream_, cudaStreamNonBlocking), "cache read stream", err, device)) {
        close();
        return false;
    }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) FF), (int) FF);
    }
    // Zero-copy: the helper reads its input from, and writes its compact rows into, the pinned host buffers
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
    zero_copy_ = !(zc && zc[0] == '0') &&
                 cudaHostGetDevicePointer((void**) &z_x_, h_x_, 0) == cudaSuccess &&
                 cudaHostGetDevicePointer((void**) &z_out_, h_out_, 0) == cudaSuccess;
    cudaGetLastError();
    auto* meta = (RemoteMeta*) d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    owned_.resize(CAP);
    layers_present_.assign((size_t) layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    if(!observation.publish(device,layers,experts,uint64_t(cache_.slots()),observation_full,uint64_t(cache_.slot_guard()),cache_.per_layer_admission())){err="remote open observation refused";close();return false;}
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    return true;
}

bool RemoteExperts::begin_impl(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                          int64_t k, const int32_t* kind, const int32_t* primary_res,
                          std::string& err, bool* attempted, cudaError_t* restore_status) {
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::cpu::MAXT || k != 10 || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); original_row_.clear();
    if (!layers_present_[(size_t) layer]) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
        }
        group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) dst_.size());
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
            // The grouped kernels read the activation from `tok`, so their
            // output row can instead be packed densely for the USB4 return.
            original_row_.push_back((int32_t) i);
            dst_.push_back((int32_t) dst_.size());
            tok_.push_back((int32_t) (i / k));
        }
    }
    start_.push_back((int32_t) dst_.size());
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    std::memcpy(h_x_, x, (size_t) n_tok * H * sizeof(float));
    auto* meta = (RemoteMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t) group_id_.size();
    const bool reduce = remote_opt_ && remote_opt_->active();
    if (reduce) remote_opt_->prepare(*this, meta + 1);
    if (attempted) *attempted=true; // retained before first device selection/submission
    DeviceScope scope(device_, restore_status);
    if (!scope.ok) { err = scope.error(device_); return false; }
    groups_ = (int32_t) group_id_.size();
    const cudaStream_t s = stream_;
    const bool staged =
        (zero_copy_ || check(cudaMemcpyAsync(d_x_, h_x_, (size_t) n_tok * H * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err, device_)) &&
        check(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(RemoteMeta) + (reduce ? remote_opt_->metadata_bytes() : 0),
                              cudaMemcpyHostToDevice, s), "copy group metadata", err, device_);
    if (!staged) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        const auto& fmt = lay.fmt[(size_t) layer];
        const float* xe = zero_copy_ ? z_x_ : d_x_;
        if (fmt.had_gu) {   // APR: Hadamard-folded gate/up take H (s * x)
            strata::kernels::hadamard_rows(xe, d_xr_, n_tok, H, fmt.had_block, d_sx_, s);
            xe = d_xr_;
        }
        strata::kernels::quantize_q8_1_rows(xe, n_tok, H, d_q8_, s);
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        if (fmt.had_d) { L.had_block = fmt.had_block; L.d_signs = d_sh_; }
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scratch_, zero_copy_ && !reduce ? z_out_ : d_out_, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(zero_copy_ ? z_x_ : d_x_, d_q8_, d_scales_, n_tok * H, s);
        strata::kernels::moe_grouped_s2(d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                        groups_, (int64_t) dst_.size(), d_q8_, d_scales_, d_scratch_, zero_copy_ && !reduce ? z_out_ : d_out_, s);
    }
    const uint64_t compact_bytes = (uint64_t) (reduce ? n_tok : dst_.size()) * H * sizeof(float);
    if (reduce) {
        if (!remote_opt_->reduce(*this, (RemoteMeta*) d_meta_ + 1, err)) return false;
    } else if (!zero_copy_ && !check(cudaMemcpyAsync(h_out_, d_out_, (size_t) compact_bytes, cudaMemcpyDeviceToHost, s),
               "copy results", err, device_)) return false;
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) n * H * sizeof(float);
    return true;
}

bool RemoteExperts::finish_impl(float* out, std::string& err, cudaError_t* restore_status) {
    if (group_id_.empty()) return true;
    DeviceScope scope(device_, restore_status);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const auto w0 = std::chrono::steady_clock::now();
    // Spin on the stream like PeerExperts::finish instead of the blocking sync: the layer waits for this helper
    // on the CPU pool's critical path, and a blocking sync's wake-up costs more than the helper's small batch
    // takes - on HIP especially, where the CUDA branch's cudaInitDevice(ScheduleSpin) does not exist and the
    // device keeps its default policy.  STRATA_REMOTE_SPIN=0 keeps the blocking sync (the same switch the CUDA
    // branch reads).
#if defined(STRATA_USE_HIP)   // HIP only: the CUDA branch's cudaInitDevice(ScheduleSpin) already makes the sync spin
    static const bool spin = [] {
        const char* v = std::getenv("STRATA_REMOTE_SPIN");
        return !(v != nullptr && v[0] == '0');
    }();
    if (spin) {
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!check(e, "finish", err, device_)) return false;
    } else if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) {
        return false;
    }
#else
    if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) return false;
#endif
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    if (remote_opt_ && remote_opt_->active()) { remote_opt_->accumulate(*this); return true; }
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * H, h_out_ + i * H, (size_t) H * sizeof(float));
    return true;
}

bool RemoteExperts::copy_cached_blobs_legacy(const int32_t* layers, const int32_t* experts, uint8_t* const* dst,
                                      size_t n, uint8_t* hit, std::string& err) {
    std::fill(hit, hit + n, 0);
    if (device_ < 0 || read_stream_ == nullptr || n == 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    std::lock_guard<std::mutex> lock(read_mu_);
    bool queued = false;
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        const int32_t slot = cache_.slot_of(layers[i], experts[i]);
        if (slot < 0) continue;
        const size_t bytes = (size_t) lay.blob_bytes(layers[i]);
        if (!check(cudaMemcpyAsync(dst[i], cache_.device_slot(slot), bytes, cudaMemcpyDeviceToHost, read_stream_),
                   "cache read", err, device_)) {
            // Nothing may still be writing a caller's buffer when this returns.  If the drain succeeds, the
            // hits queued before the failure are complete and the caller may keep them; if it does not,
            // forget every hit and let the caller refill all of them from its fallback.
            if (cudaStreamSynchronize(read_stream_) != cudaSuccess) std::fill(hit, hit + n, 0);
            return false;
        }
        hit[i] = 1;
        queued = true;
    }
    if (queued && !check(cudaStreamSynchronize(read_stream_), "cache read wait", err, device_)) {
        std::fill(hit, hit + n, 0);   // a failed wait means no copied buffer may be trusted
        return false;
    }
    return true;
}

bool RemotePrefillSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (dst != nullptr && from_helper(layer, expert)) {
        const int32_t l = (int32_t) layer, e = (int32_t) expert;
        uint8_t* d = dst;
        std::string err;
        for (RemoteExperts* h : helpers_) {
            uint8_t got = 0;
            const bool ok = h->copy_cached_blobs(&l, &e, &d, 1, &got, err);
            if (got) {
                helper_blobs_.fetch_add(1, std::memory_order_relaxed);
                helper_bytes_.fetch_add(strata::kernels::cpu::expert_layout().blob_bytes(layer),
                                        std::memory_order_relaxed);
                return true;
            }
            if (!ok) break;
        }
        warn(err);
    }
    return fallback_.copy_blob(layer, expert, dst);
}

bool RemotePrefillSource::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    if (helpers_.empty() || n == 0) return fallback_.copy_blobs(layers, experts, dst, n);
    thread_local std::vector<int32_t> hl, he;
    thread_local std::vector<uint8_t*> hd;
    thread_local std::vector<size_t> idx;
    thread_local std::vector<uint8_t> got;
    thread_local std::vector<uint8_t> served;
    thread_local std::vector<int32_t> ml, me;
    thread_local std::vector<uint8_t*> md;
    thread_local std::vector<int> which;
    which.assign(n, -1);
    served.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        for (size_t k = 0; k < helpers_.size(); ++k)
            if (helpers_[k]->holds(layers[i], experts[i])) { which[i] = (int) k; break; }
    }
    // One stream batch per helper, so several hits pay one wait; the misses go to the file source in one
    // call, which keeps its own batched-read merging for them.
    uint64_t blobs = 0, bytes = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (size_t k = 0; k < helpers_.size(); ++k) {
        hl.clear(); he.clear(); hd.clear(); idx.clear();
        for (size_t i = 0; i < n; ++i)
            if (which[i] == (int) k) {
                hl.push_back(layers[i]); he.push_back(experts[i]); hd.push_back(dst[i]); idx.push_back(i);
            }
        if (hl.empty()) continue;
        got.assign(hl.size(), 0);
        std::string err;
        if (!helpers_[k]->copy_cached_blobs(hl.data(), he.data(), hd.data(), hl.size(), got.data(), err)) warn(err);
        for (size_t j = 0; j < got.size(); ++j)
            if (got[j]) {
                served[idx[j]] = 1;
                ++blobs;
                bytes += lay.blob_bytes(hl[j]);
            }
    }
    if (blobs != 0) {
        helper_blobs_.fetch_add(blobs, std::memory_order_relaxed);
        helper_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    ml.clear(); me.clear(); md.clear();
    for (size_t i = 0; i < n; ++i)
        if (!served[i]) { ml.push_back(layers[i]); me.push_back(experts[i]); md.push_back(dst[i]); }
    if (ml.empty()) return true;
    return fallback_.copy_blobs(ml.data(), me.data(), md.data(), ml.size());
}

void RemotePrefillSource::warn(const std::string& err) {
    if (err.empty() || warned_.exchange(true)) return;
    std::fprintf(stderr, "prefill: a helper cache read failed (%s); the file source fills that blob instead\n",
                 err.c_str());
}
} // namespace strata::core
