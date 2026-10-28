#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#define private public
#include "foresight_swap.hpp"
namespace strata::platform { inline void name_this_thread(const char*) {} }
namespace strata::aux_cpus { inline void pin_current_thread() {} }
namespace strata::core {
struct ExpertSource {
    std::function<void()> callback;
    int blobs=0;uint8_t data[8]={};
    virtual bool warms(){if(callback)callback();return true;}
    bool pinned(int64_t,int32_t){if(callback)callback();return true;}
    const uint8_t* blob(int64_t,int32_t){++blobs;return data;}
};
inline bool valid_retirement_topology(const std::vector<int>&,std::string&){return true;}
inline int device_calls=0;
using cudaError_t=int;
using cudaEvent_t=void*;
constexpr int cudaSuccess=0,cudaErrorNotReady=1,cudaMemcpyDeviceToHost=2;
int cudaEventQuery(void*){++device_calls;return 0;}
int cudaGetLastError(){++device_calls;return 0;}
int cudaMemcpy(void* out,const void* in,size_t n,int){++device_calls;std::memcpy(out,in,n);return 0;}
struct ExpertResourceDevice{ExpertResourceDevice(bool,int){}};
struct OnDevice{explicit OnDevice(int){}};
[[noreturn]] void retirement_failed(const char*){std::abort();}
class RouterLookahead {
public:
    RouterLookahead() = default;
    ~RouterLookahead();
    // Terminal stop/join only: preserves source/FS targets and all buffers for
    // the caller's subsequent GPU drain. False requires retaining this owner.
    // Concurrent/reentrant/self close refuses; no join under worker mutex.
    bool close_for_retirement() noexcept;
    RouterLookahead(const RouterLookahead&) = delete;
    RouterLookahead& operator=(const RouterLookahead&) = delete;
    /// `routers[l]`: layer l's ffn_gate_inp as BF16 bits, n_expert rows of n_embd.
    bool start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k, ExpertSource* src,
               std::string& err, bool allow_cold_source = false);
    /// One prefetch design (#1348): the predicted experts that no cache holds also go to the Foresight swap space
    /// (ForesightSwap::predict), which copies them to VRAM ahead of the layer.  nullptr detaches it (waits for a
    /// prediction in progress), so the swap space can be destroyed first.
    void set_foresight(ForesightSwap* fs);
    /// Layer `layer`'s MoE input for `n_tok` tokens (host floats): predict and warm layer + 1.  Never waits: a
    /// prediction still running for an earlier layer makes this one skip.
    void submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res);
    /// STRATA_IO_PREFETCH_DEPTH: how many layers ahead of `layer` are predicted and warmed (default 1, the next layer).
    /// A deeper one predicts layer + j from layer's input, which is less exact the further it looks.
    void set_depth(int d) { std::lock_guard<std::mutex> lock(mu_); if(!retirement_started_.load()) depth_ = d < 1 ? 1 : d > 4 ? 4 : d; }
    int depth() const { return depth_; }
    /// STRATA_LOOKAHEAD_STATS=1 (measurement only): remember each prediction (top k, k+4 and k+10 per token) and let the
    /// dispatch score it against the experts the next layer really routes outside the GPU cache.
    void enable_stats(int64_t n_layers);
    bool stats_on() const { return stats_on_; }
    void stats_window_start() { std::lock_guard<std::mutex> lock(mu_); if(!retirement_started_.load()) gen_.fetch_add(1, std::memory_order_relaxed); }
    void stats_score(int64_t layer, const int32_t* ids, int64_t n, const int32_t* host_res, int pcie_num = 0);
    int64_t predicted() const { return predicted_.load(std::memory_order_relaxed); }
    int64_t skipped() const { return skipped_.load(std::memory_order_relaxed); }
    double busy_ms() const { return (double) busy_us_.load(std::memory_order_relaxed) / 1000.0; }

private:
    void run();
    std::mutex retirement_mu_;
    std::atomic<bool> retirement_started_{false};
    bool retirement_done_ = false, retirement_failed_ = false; // retirement_mu_
    std::vector<std::vector<uint16_t>> routers_;
    int64_t n_embd_ = 0, n_expert_ = 0;
    int k_ = 10;
    int depth_ = 1;
    ExpertSource* src_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false, pending_ = false, busy_ = false;
    int64_t layer_ = -1, n_tok_ = 0;
    const int32_t* host_res_ = nullptr;
    uint64_t sub_gen_ = 0;
    ForesightSwap* fs_ = nullptr;                   ///< (under mu_)
    std::vector<float> x_;
    std::atomic<int64_t> predicted_{0}, skipped_{0};
    std::atomic<uint64_t> busy_us_{0};
    // STRATA_LOOKAHEAD_STATS
    bool stats_on_ = false;
    int64_t words_ = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> pmask_;   // [layer][3 sizes][words]
    std::unique_ptr<std::atomic<uint64_t>[]> pstamp_;  // [layer]: gen + 1 of the prediction (0 = none)
    std::unique_ptr<std::atomic<int32_t>[]> pvote_;    // [layer][kVoteN]: experts by votes, expert + 1 (0 = none)
    std::atomic<uint64_t> gen_{0};
    uint64_t st_layers_ = 0, st_nopred_ = 0, st_actual_ = 0, st_hit_[3] = {0, 0, 0}, st_pred_[3] = {0, 0, 0},
             st_predhit_[3] = {0, 0, 0};
    uint64_t st_windows_ = 0;
    uint64_t st_vhit_[3] = {0, 0, 0}, st_vpred_[3] = {0, 0, 0}, st_pch_ = 0, st_pcov_[3] = {0, 0, 0};
    static constexpr int kVoteN = 16;
};
thread_local ForesightSwap::ActiveRequest* ForesightSwap::active_request_ = nullptr;
namespace {
// Detect same-thread lifecycle/source callback reentry BEFORE touching a
// nonrecursive mutex. try_lock on an already-owned std::mutex is not a fence.
struct WorkerLifecycleScope {
    const void* owner;
    WorkerLifecycleScope* previous;
    static thread_local WorkerLifecycleScope* current;
    explicit WorkerLifecycleScope(const void* value) noexcept
        : owner(value), previous(current) { current=this; }
    ~WorkerLifecycleScope() { current=previous; }
    static bool active(const void* value) noexcept {
        for(auto* frame=current;frame;frame=frame->previous)
            if(frame->owner==value)return true;
        return false;
    }
};
thread_local WorkerLifecycleScope* WorkerLifecycleScope::current=nullptr;
} // namespace

bool RouterLookahead::start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k,
                            ExpertSource* src, std::string& err, bool allow_cold_source) {
    if(WorkerLifecycleScope::active(this)){err="worker lifecycle reentry";return false;}
    WorkerLifecycleScope lifecycle_frame(this);
    std::unique_lock<std::mutex> lifecycle(retirement_mu_,std::try_to_lock);
    if(!lifecycle.owns_lock()||retirement_started_.load(std::memory_order_acquire)||thread_.joinable()) {err="lookahead is started or retired";return false;}
    if (src == nullptr || (!allow_cold_source && !src->warms())) { err = "RouterLookahead: the expert source does not warm"; return false; }
    if (n_embd % 8 != 0) { err = "RouterLookahead: n_embd is not a multiple of 8"; return false; }
    for (const auto& r : routers)
        if (r.size() != (size_t) (n_embd * n_expert)) { err = "RouterLookahead: a router of another shape"; return false; }
    if(retirement_started_.load(std::memory_order_acquire)) {err="lookahead is retired";return false;}
    routers_ = std::move(routers);
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    k_ = k < 1 ? 1 : k > (int) n_expert ? (int) n_expert : k;
    src_ = src;
    x_.assign((size_t) (8 * n_embd), 0.f);
    thread_ = std::thread([this] {
        strata::platform::name_this_thread("strata-lookahead");
        strata::aux_cpus::pin_current_thread();
        run();
    });
    return true;
}

bool RouterLookahead::close_for_retirement() noexcept {
    retirement_started_.store(true,std::memory_order_release);
    if(WorkerLifecycleScope::active(this))return false;
    WorkerLifecycleScope lifecycle_frame(this);
    try {
        std::unique_lock<std::mutex> close(retirement_mu_,std::try_to_lock);
        if(!close.owns_lock()||retirement_failed_)return false;
        if(retirement_done_)return true;
        if(thread_.joinable()&&thread_.get_id()==std::this_thread::get_id())return false;
        { std::lock_guard<std::mutex> lock(mu_); quit_=true; }
        cv_.notify_all();
        try { if(thread_.joinable())thread_.join(); }
        catch(...) {retirement_failed_=true;return false;}
        retirement_done_=true;
        return true;
    } catch(...) {return false;}
}

void RouterLookahead::set_foresight(ForesightSwap* fs) {
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if(retirement_started_.load(std::memory_order_acquire))return;
            if (fs != nullptr || !busy_) { fs_ = fs; return; }
        }
        // detaching: a prediction in progress finishes first (the swap space is about to go away)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void RouterLookahead::submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res) {
    if (layer + 1 >= (int64_t) routers_.size() || n_tok <= 0 || x == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if(retirement_started_.load(std::memory_order_acquire))return;
        if (busy_ || pending_) { skipped_.fetch_add(1, std::memory_order_relaxed); return; }
        n_tok_ = std::min<int64_t>(n_tok, 8);
        std::memcpy(x_.data(), x, (size_t) (n_tok_ * n_embd_) * sizeof(float));
        layer_ = layer + 1;
        host_res_ = host_res;
        sub_gen_ = gen_.load(std::memory_order_relaxed);
        pending_ = true;
    }
    cv_.notify_one();
}

void RouterLookahead::enable_stats(int64_t n_layers) {
    std::lock_guard<std::mutex> lock(mu_);
    if(retirement_started_.load(std::memory_order_acquire))return;
    words_ = (n_expert_ + 63) / 64;
    pmask_.reset(new std::atomic<uint64_t>[(size_t) (n_layers * 3 * words_)]);
    pstamp_.reset(new std::atomic<uint64_t>[(size_t) n_layers]);
    for (int64_t i = 0; i < n_layers * 3 * words_; ++i) pmask_[(size_t) i].store(0, std::memory_order_relaxed);
    for (int64_t i = 0; i < n_layers; ++i) pstamp_[(size_t) i].store(0, std::memory_order_relaxed);
    pvote_.reset(new std::atomic<int32_t>[(size_t) (n_layers * kVoteN)]);
    for (int64_t i = 0; i < n_layers * kVoteN; ++i) pvote_[(size_t) i].store(0, std::memory_order_relaxed);
    stats_on_ = true;
}

bool ForesightSwap::set_strict_retirement(bool enabled, const std::vector<int>& devices, std::string& error) {
    if(WorkerLifecycleScope::active(this)){error="worker lifecycle reentry";return false;}
    WorkerLifecycleScope lifecycle_frame(this);
    std::unique_lock<std::mutex> lifecycle(retirement_mu_,std::try_to_lock);
    if(!lifecycle.owns_lock()||retirement_started_.load(std::memory_order_acquire)){error="foresight is retired";return false;}
    if (retirement_init_started_) { error = "strict foresight retirement must be configured before init"; return false; }
#if defined(STRATA_GPU_METAL)
    if (enabled) { error = "strict foresight retirement is unsupported by Metal"; return false; }
#endif
    if (enabled && !valid_retirement_topology(devices, error)) return false;
    retirement_consumers_ = devices;
    strict_retirement_ = enabled;
    return true;
}

bool ForesightSwap::close_for_retirement() noexcept {
    retirement_started_.store(true,std::memory_order_release);
    if(WorkerLifecycleScope::active(this))return false;
    WorkerLifecycleScope lifecycle_frame(this);
    for(auto* frame=active_request_;frame;frame=frame->previous)
        if(frame->owner==this)return false; // source callback still owns smu_
    try {
        std::unique_lock<std::mutex> close(retirement_mu_,std::try_to_lock);
        if(!close.owns_lock()||retirement_failed_)return false;
        if(retirement_done_)return true;
        if(filler_.joinable()&&filler_.get_id()==std::this_thread::get_id())return false;
        {
            std::lock_guard<std::mutex> slots_lock(smu_);
            std::lock_guard<std::mutex> worker_lock(mu_);
            stop_=true;
        }
        cv_.notify_all();
        try {if(filler_.joinable())filler_.join();}
        catch(...) {retirement_failed_=true;return false;}
        retirement_done_=true;
        return true;
    } catch(...) {return false;}
}

bool ForesightSwap::take(int64_t layer, int32_t expert, unsigned long long& ptr) {
    std::lock_guard<std::mutex> smu_lk(smu_);
    if(retirement_started_.load(std::memory_order_acquire))return false;
    const int s = where[(size_t) (layer * n_expert + expert)];
    if (s < 0) return false;
    Slot& sl = slot[(size_t) (layer * slots + s)];
    if (!sl.ready) {
        if (!sl.loading || !sl.issued.load(std::memory_order_acquire)) { ++pending; return false; }
        ExpertResourceDevice selected(strict_retirement_, sl.event_device);
        const cudaError_t q = cudaEventQuery((cudaEvent_t) sl.ev);
        if (q != cudaSuccess) {
            if (strict_retirement_ && q != cudaErrorNotReady) retirement_failed("foresight event readiness");
            if (q != cudaErrorNotReady) cudaGetLastError();
            ++pending;
            return false;
        }
        sl.ready = true;
        sl.loading = false;
        if (verify) {   // test mode: the bytes the GPU will read must be the expert's own
            std::vector<uint8_t> back((size_t) sl.bytes);
            const OnDevice on(strict_retirement_ ? -1 : layer_dev[(size_t) layer]);
            const bool same = cudaMemcpy(back.data(), (const void*) sl.dptr, (size_t) sl.bytes, cudaMemcpyDeviceToHost) ==
                                  cudaSuccess && std::memcmp(back.data(), sl.src, (size_t) sl.bytes) == 0;
            if (same) ++verify_ok;
            else {
                cudaGetLastError();
                ++verify_bad;
                sl.ready = false;                           // never serve it
                where[(size_t) (layer * n_expert + expert)] = -1;
                sl.expert = -1;
                return false;
            }
        }
    }
    sl.last_ref = completed.load(std::memory_order_acquire);
    ptr = sl.dptr;
    ++hits;
    return true;
}

void ForesightSwap::hold(int64_t layer, int32_t expert) {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return;
    std::lock_guard<std::mutex> smu_lk(smu_);
    if(retirement_started_.load(std::memory_order_acquire))return;
    const size_t key = (size_t) (layer * n_expert + expert);
    const int s = where[key];
    where[key] = kHeld;
    if (s < 0) return;
    Slot& sl = slot[(size_t) (layer * slots + s)];
    sl.expert = kDropped;   // `loading` and `last_ref` stay: a copy in flight and a window that read it are waited for
    sl.ready = false;
    ++dropped;
}

void ForesightSwap::unhold(int64_t layer, int32_t expert) {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return;
    std::lock_guard<std::mutex> smu_lk(smu_);
    if(retirement_started_.load(std::memory_order_acquire))return;
    int16_t& w = where[(size_t) (layer * n_expert + expert)];
    if (w == kHeld) w = -1;
}

void ForesightSwap::note_miss(ExpertSource* src, int64_t layer, int32_t expert) {
    std::lock_guard<std::mutex> smu_lk(smu_);
    request_locked(src, layer, expert, false);
}

void ForesightSwap::predict(ExpertSource* src, int64_t layer, int32_t expert) {
    if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return;
    std::lock_guard<std::mutex> smu_lk(smu_);
    request_locked(src, layer, expert, true);
}

void ForesightSwap::request_locked(ExpertSource* src, int64_t layer, int32_t expert, bool predicted_copy) {
    if(retirement_started_.load(std::memory_order_acquire))return;
    WorkerLifecycleScope lifecycle_frame(this);
    ActiveRequest frame{this,active_request_};
    struct Restore { ActiveRequest*& head; ActiveRequest* previous; ~Restore(){head=previous;} } restore{active_request_,active_request_};
    active_request_=&frame;
    if (failed.load(std::memory_order_relaxed)) return;
    const size_t key = (size_t) (layer * n_expert + expert);
    if (where[key] != -1) return;                       // already on its way, or held by an exchange (#2134)
    const uint64_t now = completed.load(std::memory_order_acquire);
    if (admit > 1 && !predicted_copy) {
        if (now > miss_at[key] + 4) miss_cnt[key] = 0;
        miss_at[key] = now;
        if (miss_cnt[key] < 255) ++miss_cnt[key];
        if (miss_cnt[key] < admit) return;
    }
    const int c = layer_card[(size_t) layer];
    if (card_round[(size_t) c] != now) { card_round[(size_t) c] = now; card_used[(size_t) c] = 0; }
    if (card_used[(size_t) c] >= budget) { ++over_budget; return; }
    if (src == nullptr || !src->pinned(layer, expert)) return;   // async copies need the pinned arena
    if(retirement_started_.load(std::memory_order_acquire))return;
    const uint8_t* b = src->blob(layer, expert);
    if (b == nullptr || retirement_started_.load(std::memory_order_acquire)) return;
    int best = -1;
    uint64_t best_ref = ~0ull;
    for (int s = 0; s < slots; ++s) {   // an empty slot, else the one used longest ago that no window can still read
        Slot& sl = slot[(size_t) (layer * slots + s)];
        if (sl.loading) {   // copied in but never asked for since: landed long ago -> an ordinary candidate
            if (sl.last_ref + (uint64_t) depth > now || !sl.issued.load(std::memory_order_acquire)) continue;
            ExpertResourceDevice selected(strict_retirement_, sl.event_device);
            const cudaError_t q = cudaEventQuery((cudaEvent_t) sl.ev);
            if (q != cudaSuccess) {
                if (strict_retirement_ && q != cudaErrorNotReady) retirement_failed("foresight slot reuse readiness");
                if (q != cudaErrorNotReady) cudaGetLastError();
                continue;
            }
            sl.loading = false;
            sl.ready = true;
        }
        if (sl.expert == -1) { best = s; break; }      // never used (a dropped slot, kDropped, waits like a used one)
        if (sl.last_ref + (uint64_t) depth > now) continue;
        if (sl.last_ref < best_ref) { best_ref = sl.last_ref; best = s; }
    }
    if (best < 0) { ++busy; return; }
    Slot& sl = slot[(size_t) (layer * slots + best)];
    if (sl.expert >= 0) where[(size_t) (layer * n_expert + sl.expert)] = -1;
    sl.expert = expert;
    sl.ready = false;
    sl.loading = true;
    sl.issued.store(false, std::memory_order_relaxed);
    sl.src = b;
    sl.last_ref = now;
    where[key] = (int16_t) best;
    ++card_used[(size_t) c];
    ++copies;
    if (predicted_copy) ++predicted;
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(layer * slots + best);
    }
    cv_.notify_one();
}
// Fixture destructors only exercise stop/join, not production GPU frees.
RouterLookahead::~RouterLookahead(){assert(close_for_retirement());}
void RouterLookahead::run() {} // numeric worker loop is not this fixture's target
ForesightSwap::~ForesightSwap(){assert(close_for_retirement());}
} // namespace strata::core
#undef private
using namespace strata::core;
static void setup(ForesightSwap& s){
    s.slots=1;s.n_layers=1;s.n_expert=1;s.slot.reset(new ForesightSwap::Slot[1]);
    s.where={-1};s.miss_cnt={0};s.miss_at={0};s.layer_card={0};s.card_round={~0ull};s.card_used={0};
}
int main(){
    {
        // Actual start holds the lifecycle mutex while invoking virtual warms.
        // Close must reject reentry BEFORE touching that same mutex.
        RouterLookahead r;ExpertSource src;bool refused=false;std::string error;
        src.callback=[&]{refused=!r.close_for_retirement();};
        assert(!r.start({},8,1,1,&src,error));assert(refused);
        assert(!r.thread_.joinable()&&r.routers_.empty()&&r.x_.empty());
        assert(r.close_for_retirement());assert(r.close_for_retirement());
    }
    {
        // Nested starts must detect an owner below the top lifecycle frame.
        RouterLookahead a,b;ExpertSource as,bs;bool refused=false;std::string error;
        as.callback=[&]{assert(b.start({},8,1,1,&bs,error));};
        bs.callback=[&]{refused=!a.close_for_retirement();};
        assert(!a.start({},8,1,1,&as,error));assert(refused&&!a.thread_.joinable());
        assert(a.close_for_retirement());assert(b.close_for_retirement());
    }
    {
        RouterLookahead r;ExpertSource src;bool refused=false;std::string error;
        src.callback=[&]{refused=!r.start({},8,1,1,&src,error);};
        assert(r.start({},8,1,1,&src,error));assert(refused);
        assert(r.close_for_retirement());
    }

    {
        RouterLookahead r;r.routers_.resize(2);r.n_embd_=1;r.x_={19};
        auto* bytes=r.x_.data();float x=7;int32_t res=0;
        assert(r.close_for_retirement());assert(r.close_for_retirement());
        r.submit(0,&x,1,&res);r.set_depth(4);r.stats_window_start();r.enable_stats(2);
        r.set_foresight(reinterpret_cast<ForesightSwap*>(1));
        assert(r.x_.data()==bytes&&r.x_[0]==19&&!r.pending_&&r.depth_==1&&r.gen_==0&&!r.stats_on_&&r.fs_==nullptr);
    }
    {
        // Join must release mu_: the worker needs it to observe quit.
        RouterLookahead r;
        r.thread_=std::thread([&]{std::unique_lock<std::mutex> lock(r.mu_);r.cv_.wait(lock,[&]{return r.quit_;});});
        assert(r.close_for_retirement());assert(!r.thread_.joinable());
    }
    {
        RouterLookahead r;bool refused=false,ready=false,done=false;std::mutex mu;std::condition_variable cv;
        r.thread_=std::thread([&]{
            {std::unique_lock<std::mutex> lock(mu);cv.wait(lock,[&]{return ready;});}
            refused=!r.close_for_retirement();
            {std::lock_guard<std::mutex> lock(mu);done=true;}cv.notify_all();
        });
        {std::lock_guard<std::mutex> lock(mu);ready=true;}cv.notify_all();
        {std::unique_lock<std::mutex> lock(mu);cv.wait(lock,[&]{return done;});}
        r.thread_.join();assert(refused);assert(r.close_for_retirement());
    }
    {
        ForesightSwap s;setup(s);auto* slots=s.slot.get();s.card_mem={reinterpret_cast<void*>(17)};
        assert(s.close_for_retirement());unsigned long long out=91;
        assert(!s.take(0,0,out)&&out==91);s.hold(0,0);s.unhold(0,0);s.note_miss(nullptr,0,0);s.predict(nullptr,0,0);
        assert(s.slot.get()==slots&&s.card_mem[0]==reinterpret_cast<void*>(17)&&s.where[0]==-1&&s.copies==0);
        assert(s.close_for_retirement());
    }
    {
        // Source callback reentry must refuse before trying to relock smu_.
        ForesightSwap s;setup(s);ExpertSource src;bool refused=false;
        src.callback=[&]{refused=!s.close_for_retirement();};s.predict(&src,0,0);
        assert(refused&&src.blobs==0&&s.queue_.empty()&&s.copies==0);
        assert(s.close_for_retirement());
    }
    {
        // Detect this owner even below another active request frame.
        ForesightSwap a,b;setup(a);setup(b);ExpertSource as,bs;bool refused=false;
        as.callback=[&]{b.predict(&bs,0,0);};bs.callback=[&]{refused=!a.close_for_retirement();};
        a.predict(&as,0,0);assert(refused&&as.blobs==0&&a.copies==0);
        assert(a.close_for_retirement());assert(b.close_for_retirement());
    }
    {
        ForesightSwap s;setup(s);std::mutex gate;std::condition_variable cv;bool entered=false,proceed=false,closed=false;
        s.filler_=std::thread([&]{
            {std::unique_lock<std::mutex> lock(gate);entered=true;cv.notify_all();cv.wait(lock,[&]{return proceed;});}
            std::lock_guard<std::mutex> slots(s.smu_);std::lock_guard<std::mutex> worker(s.mu_);assert(s.stop_);
        });
        {std::unique_lock<std::mutex> lock(gate);cv.wait(lock,[&]{return entered;});}
        std::thread closer([&]{closed=s.close_for_retirement();});
        while(!s.retirement_started_.load())std::this_thread::yield();
        // First close owns lifecycle while joining; a contender must not block.
        while(true){std::lock_guard<std::mutex> lock(s.mu_);if(s.stop_)break;}
        assert(!s.close_for_retirement());
        {std::lock_guard<std::mutex> lock(gate);proceed=true;}cv.notify_all();closer.join();assert(closed);
    }
    {
        ForesightSwap s;setup(s);ExpertSource src;bool refused=false;std::string error;
        src.callback=[&]{refused=!s.set_strict_retirement(true,{0},error);};
        s.predict(&src,0,0);assert(refused&&!s.strict_retirement_);
        assert(s.close_for_retirement());
    }
    assert(device_calls==0); // close/closed mutators never synchronize/free GPU storage
}
