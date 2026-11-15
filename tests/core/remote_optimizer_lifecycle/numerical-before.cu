#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "strata/core/strict_retirement.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstring>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int K = 10, MAXT = strata::kernels::cpu::MAXT, CAP = MAXT * K;
struct ReduceMeta { int32_t row[CAP]; float weight[CAP]; };

bool check(cudaError_t status, std::string& err) {
    if (status == cudaSuccess) return true;
    err = std::string("remote expert decode optimization: ") + cudaGetErrorString(status);
    return false;
}

__global__ void reduce_experts(const float* parts, const ReduceMeta* meta, float* sum) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    float v = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int i = t * K + j, r = meta->row[i];
        if (r >= 0) v += parts[(int64_t) r * H + c] * meta->weight[i];
    }
    sum[(int64_t) t * H + c] = v;
}

// Like the upstream mapped-row copy, also skip rows returned in the remote sum.
__global__ void copy_cpu_rows(float4* dst, const volatile float4* src, const int32_t* mask,
                              const int32_t* hit_rows, const int32_t* count) {
    const int row = blockIdx.x;
    __shared__ int skip;
    if (threadIdx.x == 0) {
        int s = mask[row];
        for (int i = 0; i < *count; ++i) s |= hit_rows[i] == row;
        skip = s;
    }
    __syncthreads();
    const int64_t start = (int64_t) row * (H / 4);
    for (int i = threadIdx.x; i < H / 4; i += blockDim.x)
        dst[start + i] = skip ? make_float4(0, 0, 0, 0) : const_cast<const float4*>(src)[start + i];
}
} // namespace

void RemoteExpertOpt::attach(RemoteExperts& remote) {
    remote.remote_opt_ = this;
    peers_.push_back({&remote});
}

bool RemoteExpertOpt::init(std::string& err) {
    void* mapped = nullptr;
    if (!check(cudaHostAlloc((void**) &h_sum_, MAXT * (H + K) * sizeof(float),
                             cudaHostAllocPortable | cudaHostAllocMapped), err) ||
        !check(cudaHostGetDevicePointer(&mapped, h_sum_, 0), err)) return false;
    m_sum_ = (float*) mapped;
    h_mask_ = reinterpret_cast<int32_t*>(h_sum_ + MAXT * H);
    m_mask_ = reinterpret_cast<int32_t*>(m_sum_ + MAXT * H);
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (!check(cudaMalloc((void**) &p.sum, MAXT * H * sizeof(float)), err)) return false;
    }
    return true;
}

RemoteExpertOpt::~RemoteExpertOpt() {
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (p.remote->stream_) cudaStreamSynchronize(p.remote->stream_);
        if (p.sum) cudaFree(p.sum);
        p.remote->remote_opt_ = nullptr;
    }
    if (h_sum_) cudaFreeHost(h_sum_);
}

size_t RemoteExpertOpt::metadata_bytes() { return sizeof(ReduceMeta); }

void RemoteExpertOpt::begin(const float* weights, int token_begin, int tokens) {
    weights_ = weights;
    token_begin_ = token_begin;
    tokens_ = tokens;
    std::memset(h_sum_ + token_begin * H, 0, tokens * H * sizeof(float));
    std::memset(h_mask_ + token_begin * K, 0, tokens * K * sizeof(int32_t));
}

void RemoteExpertOpt::prepare(const RemoteExperts& remote, void* metadata) const {
    auto* m = (ReduceMeta*) metadata;
    std::fill(m->row, m->row + tokens_ * K, -1);
    for (size_t i = 0; i < remote.original_row_.size(); ++i)
        m->row[remote.original_row_[i]] = (int32_t) i;
    std::memcpy(m->weight, weights_, tokens_ * K * sizeof(float));
}

bool RemoteExpertOpt::reduce(RemoteExperts& remote, const void* metadata, std::string& err) {
    const auto& p = *std::find_if(peers_.begin(), peers_.end(), [&](const Peer& p) { return p.remote == &remote; });
    // finish() synchronizes this stream before the host reads h_out_. When mapped output is available,
    // write the reduced rows there directly, avoiding a separate device-to-host copy for every layer.
    float* sum = remote.zero_copy_ ? remote.z_out_ : p.sum;
    reduce_experts<<<dim3((H + 255) / 256, tokens_), 256, 0, remote.stream_>>>(
        remote.d_out_, (const ReduceMeta*) metadata, sum);
    if (remote.zero_copy_) return check(cudaGetLastError(), err);
    return check(cudaMemcpyAsync(remote.h_out_, p.sum, tokens_ * H * sizeof(float),
                                  cudaMemcpyDeviceToHost, remote.stream_), err);
}

void RemoteExpertOpt::accumulate(const RemoteExperts& remote) {
    for (int64_t i = 0; i < tokens_ * H; ++i) h_sum_[token_begin_ * H + i] += remote.h_out_[i];
    for (int i = 0; i < tokens_ * K; ++i)
        h_mask_[token_begin_ * K + i] |= remote.owned_[(size_t) i] != 0;
}

void RemoteExpertOpt::copy_rows(float* dst, const float* src, int token_begin, int tokens,
                         const int32_t* hit_rows, const int32_t* count, void* stream) const {
    copy_cpu_rows<<<tokens * K, 128, 0, (cudaStream_t) stream>>>(
        (float4*) dst, (const volatile float4*) src, m_mask_ + token_begin * K, hit_rows, count);
}

void RemoteExpertOpt::combine(float* dst, float* scratch, int token_begin, int tokens,
                       const uint32_t* skip, uint32_t ring, void* stream) const {
    const float* sum = m_sum_ + token_begin * H;
    if (skip) {
        // An all-primary device-planned group bypasses the host: ignore its stale sum.
        strata::kernels::copy_or_zero_from_mapped(scratch, sum, tokens * H, skip, ring, stream);
        sum = scratch;
    }
    strata::kernels::add_inplace(dst, sum, tokens * H, stream);
}

bool RemoteExpertOpt::owns(int64_t layer, int32_t expert) const {
    for (const auto& p : peers_)
        if (p.remote->cache_.slot_of(layer, expert) >= 0) return true;
    return false;
}

bool RemoteExpertOpt::adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
                     const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
                     ExpertSource& source) {
    if (max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<uint8_t> resident(primary.size(), 0);
    for (size_t i = 0; i < primary.size(); ++i) resident[i] = primary[i] >= 0;
    for (const auto& [i, slot] : pending) resident[(size_t) i] = 1;
    for (const auto& p : peers_) {
        const auto& r = *p.remote;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l)
            for (int32_t e = 0; e < r.n_expert_; ++e)
                if (r.cache_.slot_of(l, e) >= 0) resident[(size_t) l * r.n_expert_ + e] = 1;
    }
    struct Swap { float gain; int32_t layer, in, out; };
    std::string err;
    for (auto& p : peers_) {
        auto& r = *p.remote;
        std::vector<Swap> swaps;
        std::vector<std::pair<float, int32_t>> candidates, victims;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l) {
            candidates.clear(); victims.clear();
            const size_t base = (size_t) l * r.n_expert_;
            float primary_cold = 1.0e30f;
            for (int32_t e = 0; e < r.n_expert_; ++e) {
                const float u = usage[base + e];
                if (primary[base + e] >= 0) primary_cold = std::min(primary_cold, u);
                if (r.cache_.slot_of(l, e) >= 0) victims.emplace_back(u, e);
                else if (!resident[base + e] && u >= 2.0f) candidates.emplace_back(u, e);
            }
            const size_t n = std::min(candidates.size(), victims.size());
            if (!n) continue;
            std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.first > b.first; });
            std::partial_sort(victims.begin(), victims.begin() + n, victims.end(),
                              [](auto& a, auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < n; ++i) {
                if (victims[i].first > primary_cold) break;
                if (candidates[i].first < victims[i].first + 1.5f) break;
                swaps.push_back({candidates[i].first - victims[i].first, l, candidates[i].second, victims[i].second});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
        if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
        if (swaps.empty()) continue;
        const OnDevice on(r.device_);
        for (const auto& s : swaps) {
            const uint8_t* in_blob = source.blob(s.layer, s.in);
            if (!r.cache_.fill_slot(r.cache_.slot_of(s.layer, s.out), in_blob,
                                    r.stream_, err, (int64_t) lay.blob_bytes(s.layer))) return false;
            source.note_async_read(in_blob, r.stream_);   // #1237
        }
        if (!check(cudaStreamSynchronize(r.stream_), err)) return false;
        for (const auto& s : swaps) {
            r.cache_.replace(s.layer, s.out, s.in);
            const size_t base = (size_t) s.layer * r.n_expert_;
            resident[base + s.out] = 0;
            resident[base + s.in] = 1;
        }
    }
    return true;
}
struct RemoteExpertOpt::OwnedAdaptWorkspace::Storage {
    struct Swap { float gain; int32_t layer, in, out; };
    bool active_ = false;
    size_t rows_ = 0, experts_ = 0, swaps_ = 0;
    std::array<RemoteExperts::PlacementWrite, 3> writers_;
    std::unique_ptr<uint8_t[]> resident_;
    std::unique_ptr<std::pair<float, int32_t>[]> candidates_, victims_;
    std::unique_ptr<Swap[]> swaps_storage_;
    std::unique_ptr<AdaptPlanEntry[]> plans_;
};
RemoteExpertOpt::OwnedAdaptWorkspace::OwnedAdaptWorkspace() = default;
RemoteExpertOpt::OwnedAdaptWorkspace::~OwnedAdaptWorkspace() = default;

bool RemoteExpertOpt::OwnedAdaptWorkspace::required_bytes(size_t rows, size_t experts,
                                                         size_t swaps, uint64_t& bytes) noexcept {
    if (!rows || !experts || experts > rows || !swaps || swaps > rows ||
        rows > size_t(INT32_MAX) || experts > size_t(INT32_MAX)) return false;
    uint64_t total = sizeof(OwnedAdaptWorkspace) + sizeof(Storage) + 4096;
    auto add = [&](uint64_t count, uint64_t width) {
        if (count > (UINT64_MAX - total) / width) return false;
        total += count * width; return true;
    };
    if (!add(rows, sizeof(uint8_t) + sizeof(Storage::Swap)) ||
        !add(experts, 2 * sizeof(std::pair<float, int32_t>)) ||
        !add(swaps, 3 * sizeof(AdaptPlanEntry)) || total > SIZE_MAX) return false;
    bytes = total;
    return true;
}

bool RemoteExpertOpt::OwnedAdaptWorkspace::prepare(size_t rows, size_t experts,
                                                   size_t swaps, std::string& error) {
    uint64_t bytes = 0;
    if (storage_ || !required_bytes(rows, experts, swaps, bytes)) {
        error = "remote adapt: invalid or already prepared workspace"; return false;
    }
    // No allocation in constructor. Caller admits the complete quoted envelope
    // before this host-only preparation; no arbitrary charge marker is accepted.
    auto storage = std::make_unique<Storage>();
    storage->resident_ = std::make_unique<uint8_t[]>(rows);
    storage->candidates_ = std::make_unique<std::pair<float, int32_t>[]>(experts);
    storage->victims_ = std::make_unique<std::pair<float, int32_t>[]>(experts);
    storage->swaps_storage_ = std::make_unique<Storage::Swap[]>(rows);
    storage->plans_ = std::make_unique<AdaptPlanEntry[]>(3 * swaps);
    storage->rows_ = rows; storage->experts_ = experts; storage->swaps_ = swaps;
    storage_ = std::move(storage);
    return true;
}

RemoteExpertOpt::AdaptResult RemoteExpertOpt::adapt_owned(
        const std::vector<float>& usage, const std::vector<int32_t>& primary,
        const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
        ExpertSource& source, OwnedAdaptWorkspace& workspace, AdaptPublication& publication,
        std::string& error) {
    if (!workspace.storage_) {
        error = "remote adapt: workspace is unprepared"; return AdaptResult::failed;
    }
    auto& w = *workspace.storage_;
    if (w.active_ || peers_.size() > 3 ||
        primary.size() > w.rows_ || usage.size() != primary.size() ||
        max_swaps < 0 || size_t(max_swaps) > w.swaps_) {
        error = "remote adapt: unprepared/insufficient workspace or invalid input";
        return AdaptResult::failed;
    }
    // Checked controller contract: configuration/attach cannot race this call.
    // These validations read immutable geometry, never mutable cache residency.
    for (size_t p = 0; p < peers_.size(); ++p) {
        auto* r = peers_[p].remote;
        if (!r || r->n_expert_ <= 0 || uint64_t(r->n_expert_) > w.experts_ ||
            r->layers_present_.size() > size_t(INT32_MAX) ||
            r->layers_present_.size() > primary.size() / size_t(r->n_expert_)) {
            error = "remote adapt: helper geometry exceeds workspace"; return AdaptResult::failed;
        }
        for (size_t q = 0; q < p; ++q) if (peers_[q].remote == r) {
            error = "remote adapt: duplicate helper"; return AdaptResult::failed;
        }
    }
    for (const auto& item : pending) if (item.first < 0 || size_t(item.first) >= primary.size()) {
        error = "remote adapt: pending index out of range"; return AdaptResult::failed;
    }
    w.active_ = true;
    std::array<bool, 3> attempted{}, drained{}, finished{};
    bool prepared_publication = false;
    // Every attempted exit drains BEFORE borrowed source state may unwind.
    // Failure to prove that drain is fatal; quarantine alone cannot own blob().
    auto cleanup = [&]() noexcept {
        for (size_t p = 0; p < peers_.size(); ++p) {
            if (attempted[p] && !drained[p]) {
                try {
                    if (!w.writers_[p].drain_copies(error)) retirement_failed("remote adapt copy drain");
                    drained[p] = true;
                } catch (...) { retirement_failed("remote adapt copy drain exception"); }
            }
            if (attempted[p] && !finished[p]) publication.fail_unknown(uint32_t(p));
        }
        if (prepared_publication) publication.abandon_unattempted();
        for (auto& writer : w.writers_) writer.reset();
        w.active_ = false;
    };
    struct Cleanup { decltype(cleanup)& f; ~Cleanup() noexcept { f(); } } exit{cleanup};
    for (size_t p = 0; p < peers_.size(); ++p) {
        const auto status = peers_[p].remote->try_placement_write(w.writers_[p]);
        if (status == RemoteExperts::Admission::busy) return AdaptResult::deferred;
        if (status != RemoteExperts::Admission::acquired) {
            error = "remote adapt: helper writer unavailable"; return AdaptResult::failed;
        }
    }
    try {
        for (size_t p = 0; p < peers_.size(); ++p)
            if (!w.writers_[p].drain_before_overwrite(error)) return AdaptResult::failed;
        if (max_swaps == 0) return AdaptResult::complete;
        const auto& layout = strata::kernels::cpu::expert_layout();
        for (size_t i = 0; i < primary.size(); ++i) w.resident_[i] = primary[i] >= 0;
        for (const auto& item : pending) w.resident_[size_t(item.first)] = 1;
        for (const auto& peer : peers_) {
            const auto& r = *peer.remote;
            for (int32_t l = 0; l < int32_t(r.layers_present_.size()); ++l)
                for (int32_t e = 0; e < r.n_expert_; ++e)
                    if (r.cache_.slot_of(l,e) >= 0) w.resident_[size_t(l)*r.n_expert_+e] = 1;
        }
        std::array<size_t, 3> starts{}, counts{};
        size_t total = 0;
        for (size_t p = 0; p < peers_.size(); ++p) {
            auto& r = *peers_[p].remote;
            size_t ns = 0;
            for (int32_t l = 0; l < int32_t(r.layers_present_.size()); ++l) {
                size_t nc = 0, nv = 0; float cold = 1.0e30f;
                const size_t base = size_t(l)*r.n_expert_;
                for (int32_t e = 0; e < r.n_expert_; ++e) {
                    const float u = usage[base+e];
                    if (!std::isfinite(u)) { error="remote adapt: nonfinite usage"; return AdaptResult::failed; }
                    if (primary[base+e] >= 0) cold = std::min(cold,u);
                    if (r.cache_.slot_of(l,e) >= 0) w.victims_[nv++] = {u,e};
                    else if (!w.resident_[base+e] && u >= 2.0f) w.candidates_[nc++] = {u,e};
                }
                const size_t n = std::min(nc,nv);
                std::sort(w.candidates_.get(),w.candidates_.get()+nc,[](auto&a,auto&b){return a.first>b.first;});
                std::partial_sort(w.victims_.get(),w.victims_.get()+n,w.victims_.get()+nv,[](auto&a,auto&b){return a.first<b.first;});
                for (size_t i = 0; i < n; ++i) {
                    if (w.victims_[i].first > cold || w.candidates_[i].first < w.victims_[i].first+1.5f) break;
                    w.swaps_storage_[ns++] = {w.candidates_[i].first-w.victims_[i].first,l,w.candidates_[i].second,w.victims_[i].second};
                }
            }
            std::sort(w.swaps_storage_.get(),w.swaps_storage_.get()+ns,[](auto&a,auto&b){return a.gain>b.gain;});
            ns = std::min(ns,size_t(max_swaps)); starts[p] = total; counts[p] = ns;
            for (size_t i = 0; i < ns; ++i) {
                const auto& s = w.swaps_storage_[i];
                const int32_t slot = r.cache_.slot_of(s.layer,s.out);
                const auto bytes = layout.blob_bytes(s.layer);
                if (slot < 0 || bytes <= 0 || uint64_t(bytes)>uint64_t(INT64_MAX) ||
                    uint64_t(bytes)>r.cache_.slot_offset(int64_t(slot)+1)-r.cache_.slot_offset(slot)) {
                    error="remote adapt: invalid actual slot/extent"; return AdaptResult::failed;
                }
                w.plans_[total++] = {uint32_t(p),slot,s.layer,s.out,s.in,uint64_t(bytes)};
                const size_t base = size_t(s.layer)*r.n_expert_;
                w.resident_[base+s.out]=0; w.resident_[base+s.in]=1;
            }
        }
        if (!total) return AdaptResult::complete;
        prepared_publication = true; // partial prepare/throw also requires abandonment
        if (!publication.prepare_all(w.plans_.get(),total)) return AdaptResult::failed;
        for (size_t p = 0; p < peers_.size(); ++p) {
            if (!counts[p]) continue;
            auto& r = *peers_[p].remote;
            if (!w.writers_[p].mark_attempted()) return AdaptResult::failed;
            attempted[p] = true;
            if (!publication.begin_attempt(uint32_t(p))) return AdaptResult::failed;
            int previous = -1;
            if (cudaGetDevice(&previous)!=cudaSuccess) return AdaptResult::failed;
            struct Restore {
                int previous;
                ~Restore() noexcept { if (cudaSetDevice(previous)!=cudaSuccess) retirement_failed("remote adapt restore device"); }
            } restore{previous};
            if (cudaSetDevice(r.device_)!=cudaSuccess) return AdaptResult::failed;
            for (size_t i = starts[p]; i < starts[p]+counts[p]; ++i) {
                const auto& plan = w.plans_[i];
                const uint8_t* blob = source.blob(plan.layer,plan.in);
                uint64_t submitted = 0;
                if (!r.cache_.fill_slot(plan.slot,blob,r.stream_,error,int64_t(plan.bytes),&submitted)) return AdaptResult::failed;
                source.note_async_read(blob,r.stream_);
                if (submitted!=plan.bytes) { error="remote adapt: submitted extent differs from plan"; return AdaptResult::failed; }
            }
            try {
                if (!w.writers_[p].drain_copies(error)) retirement_failed("remote adapt publication drain");
                drained[p] = true;
            } catch (...) { retirement_failed("remote adapt publication drain exception"); }
            for (size_t i = starts[p]; i < starts[p]+counts[p]; ++i) {
                const auto& plan = w.plans_[i]; r.cache_.replace(plan.layer,plan.out,plan.in);
            }
            if (!publication.publish_after_copies(uint32_t(p),w.plans_.get()+starts[p],counts[p])) return AdaptResult::failed;
            if (!w.writers_[p].finish_publication()) return AdaptResult::failed;
            finished[p] = true;
        }
        return AdaptResult::complete;
    } catch (...) {
        // Cleanup drains before returning; no diagnostic allocation on this path.
        return AdaptResult::failed;
    }
}
} // namespace strata::core
