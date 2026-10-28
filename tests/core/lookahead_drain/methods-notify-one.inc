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

bool RouterLookahead::residency_snapshot_bytes(size_t layers, int64_t experts, uint64_t& out) noexcept {
    if (experts < 0 || experts > std::numeric_limits<int>::max() ||
        layers > uint64_t(std::numeric_limits<int64_t>::max()) || (layers && !experts)) return false;
    const uint64_t n = uint64_t(experts);
    if (n && uint64_t(layers) > std::numeric_limits<uint64_t>::max() / n) return false;
    const uint64_t cells = uint64_t(layers) * n;
    if (cells > std::numeric_limits<uint64_t>::max() / sizeof(int32_t) ||
        cells > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
        cells > std::vector<int32_t>().max_size()) return false;
    out = cells * sizeof(int32_t);
    return true;
}

bool RouterLookahead::configure_residency_snapshot(size_t layers, int64_t experts,
    std::shared_ptr<const ReservationLease>& lease) noexcept {
    if (WorkerLifecycleScope::active(this)) return false;
    WorkerLifecycleScope frame(this);
    try {
        std::unique_lock<std::mutex> lifecycle(retirement_mu_,std::try_to_lock);
        if (!lifecycle.owns_lock() || retirement_started_.load() || thread_.joinable() || residency_lease_)
            return false;
        uint64_t bytes=0;
        if (!layers || experts<=0 || !residency_snapshot_bytes(layers,experts,bytes) ||
            !lease || lease.use_count()!=1 || lease->purpose()!=ReservationPurpose::foreground) return false;
        const auto& resources=lease->resources();
        if (resources.host_bytes!=bytes || resources.pinned_bytes || resources.disk_bytes ||
            resources.io_slots || resources.shadows ||
            std::any_of(resources.device_bytes.begin(),resources.device_bytes.end(),[](uint64_t n){return n!=0;})) return false;
        admitted_layers_=layers; admitted_experts_=experts;
        residency_lease_=std::move(lease);
        return true;
    } catch (...) { return false; }
}

RouterLookahead::SubmittedDrain::~SubmittedDrain() noexcept { reset(); }
void RouterLookahead::SubmittedDrain::reset() noexcept {
    if (!owner_) return;
    std::lock_guard<std::mutex> lock(owner_->mu_);
    // Owner lifetime and exclusive guard access are caller prerequisites. Clearing
    // the seal never clears terminal state; submit independently checks retirement.
    if (owner_->submitted_drain_ == this) owner_->submitted_drain_ = nullptr;
    owner_ = nullptr;
}

RouterLookahead::DrainResult RouterLookahead::drain_submitted(
    SubmittedDrain& guard, std::optional<std::chrono::steady_clock::time_point> deadline) {
    std::unique_lock<std::mutex> lock(mu_);
    if (worker_id_ == std::this_thread::get_id() ||
        (guard.owner_ && guard.owner_ != this) ||
        (submitted_drain_ && submitted_drain_ != &guard) ||
        retirement_started_.load(std::memory_order_acquire) || quit_ || worker_failed_)
        return DrainResult::refused;
    // Publish seal before observing idle: independent submits cannot slip between
    // observation and writer mutation. Existing pending work is allowed to finish.
    submitted_drain_ = &guard;
    guard.owner_ = this;
    const auto settled = [&] {
        return (!pending_ && !busy_) || quit_ || worker_failed_ ||
               retirement_started_.load(std::memory_order_acquire);
    };
    if (deadline && !settled()) cv_.wait_until(lock, *deadline, settled);
    if (retirement_started_.load(std::memory_order_acquire) || quit_ || worker_failed_)
        return DrainResult::refused;
    return !pending_ && !busy_ ? DrainResult::drained : DrainResult::busy;
}

bool RouterLookahead::close_for_retirement() noexcept {
    { // Serialize with the wait predicate so terminal wakeups cannot be lost.
        std::lock_guard<std::mutex> lock(mu_);
        retirement_started_.store(true,std::memory_order_release);
    }
    cv_.notify_all(); // wake bounded drain waiters even when retirement refuses
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
RouterLookahead::~RouterLookahead() {
    if(!close_for_retirement())retirement_failed("lookahead worker join");
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

bool RouterLookahead::start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k,
                            ExpertSource* src, std::string& err, bool allow_cold_source) {
    if(WorkerLifecycleScope::active(this)){err="worker lifecycle reentry";return false;}
    WorkerLifecycleScope lifecycle_frame(this);
    std::unique_lock<std::mutex> lifecycle(retirement_mu_,std::try_to_lock);
    if(!lifecycle.owns_lock()||retirement_started_.load(std::memory_order_acquire)||thread_.joinable()) {err="lookahead is started or retired";return false;}
    if (src == nullptr || (!allow_cold_source && !src->warms())) { err = "RouterLookahead: the expert source does not warm"; return false; }
    uint64_t snapshot_bytes=0;
    if (n_embd < 0 || n_embd > std::numeric_limits<int>::max() || n_embd % 8 != 0 ||
        (!routers.empty() && !n_embd) ||
        !residency_snapshot_bytes(routers.size(),n_expert,snapshot_bytes) ||
        uint64_t(n_embd) > std::numeric_limits<size_t>::max()/(8*sizeof(float)) ||
        (n_expert && uint64_t(n_embd)>std::numeric_limits<size_t>::max()/uint64_t(n_expert))) {
        err = "RouterLookahead: invalid or overflowing dimensions"; return false;
    }
    if (residency_lease_ && (admitted_layers_!=routers.size() || admitted_experts_!=n_expert)) {
        err = "RouterLookahead: snapshot geometry differs from admitted lease"; return false;
    }
    for (const auto& r : routers)
        if (r.size() != (size_t) (n_embd * n_expert)) { err = "RouterLookahead: a router of another shape"; return false; }
    if(retirement_started_.load(std::memory_order_acquire)) {err="lookahead is retired";return false;}
    std::vector<int32_t> snapshot;
    std::vector<float> activations;
    try {
        snapshot = std::vector<int32_t>(size_t(snapshot_bytes/sizeof(int32_t)));
        if (snapshot.capacity()!=snapshot.size()) { err="RouterLookahead: snapshot capacity exceeds quote";return false; }
        activations = std::vector<float>(size_t(8*n_embd),0.f);
    } catch (const std::bad_alloc&) { err.clear(); return false; }
    routers_ = std::move(routers);
    residency_snapshot_ = std::move(snapshot);
    x_ = std::move(activations);
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    k_ = k < 1 ? 1 : k > (int) n_expert ? (int) n_expert : k;
    src_ = src;
    thread_ = std::thread([this] {
        strata::platform::name_this_thread("strata-lookahead");
        strata::aux_cpus::pin_current_thread();
        run();
    });
    return true;
}

void RouterLookahead::submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res) {
    if (layer < 0 || routers_.size()<2 || uint64_t(layer)>=routers_.size()-1 || n_tok <= 0 || x == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if(retirement_started_.load(std::memory_order_acquire))return;
        if (submitted_drain_ || worker_failed_ || busy_ || pending_) { skipped_.fetch_add(1, std::memory_order_relaxed); return; }
        n_tok_ = std::min<int64_t>(n_tok, 8);
        std::memcpy(x_.data(), x, (size_t) (n_tok_ * n_embd_) * sizeof(float));
        layer_ = layer + 1;
        has_residency_ = host_res != nullptr;
        if (host_res) std::memcpy(residency_snapshot_.data(),host_res,
                                  residency_snapshot_.size()*sizeof(int32_t));
        sub_gen_ = gen_.load(std::memory_order_relaxed);
        pending_ = true;
    }
    // Worker and bounded drain waiters share this CV; waking only a drain
    // waiter could strand pending work while the worker remains asleep.
    cv_.notify_one();
}

