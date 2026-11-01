#include "suffix_raw_runtime.hpp"
#include "strata/core/serving_raw_metadata.hpp"
#include "strata/core/resident_copy_fault.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/raw_int8_restore_clear.hpp"
#include "strata/kernels/qsa.hpp"
#include "../core/conversation_copy.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/window_device.hpp"
#include <algorithm>
#include <exception>
#include <limits>
#include <new>
#include <atomic>
#include "strata/core/native_batch_setup_admission.hpp"

namespace strata::program {
using namespace core;
struct SuffixRawTransaction::HostCap {
    explicit HostCap(uint64_t n):limit(n) {}
    const uint64_t limit;
    unsigned claim_limit=16; // fixed main-only metadata lane; genuine two-slot sharing derives48
    std::atomic<uint64_t> live{0};
    // Fixed admission bound; no growing claim registry or unpriced directory.
    std::atomic<unsigned> claims{0};
};
bool SuffixRawTransaction::observe_host_usage(HostUsage& out) const noexcept {
    if(controller_!=std::this_thread::get_id()||phase_!=Phase::drained||pending_||!initialized_||!host_cap_)return false;
    // All claim producers must already be synchronously joined by this owner.
    // Atomic observations alone are NOT a producer-close or completion proof.
    HostUsage value{host_cap_->limit,host_cap_->live.load(std::memory_order_acquire),
                    host_cap_->claims.load(std::memory_order_acquire)};
    if(value.live>value.limit)return false;
    out=value;return true;
}
struct SuffixRawTransaction::HostCredit {
    std::shared_ptr<HostCap> cap;
    std::shared_ptr<const ReservationLease> global;
    uint64_t bytes=0;
    ~HostCredit() {
        // Relevant payloads must already be destroyed by their owning wrapper.
        global.reset();
        if(cap) {
            if(cap->live.fetch_sub(bytes)<bytes || cap->claims.fetch_sub(1)==0) std::terminate();
        }
    }
};
struct SuffixRawTransaction::Storage {
    // Credit is destroyed AFTER every charged buffer and actual runtime owner.
    std::shared_ptr<HostCredit> credit;
    std::shared_ptr<const void> runtime;
    NativeSessionAllocation allocation;
    BundleHandle source;
    SavedConversation native;
    SnapshotDescriptor descriptor;
    std::vector<NativePleTokens> ple;
};
namespace {
SuffixRawResult invalid(const char* why) { return {SuffixRawStatus::invalid,why,{}}; }
bool add(uint64_t& a,uint64_t b) {
    if(b>UINT64_MAX-a) return false;
    a+=b;return true;
}
bool product(uint64_t a,uint64_t b,uint64_t& result) {
    if(a && b>UINT64_MAX/a) return false;
    result=a*b;return true;
}
bool bounded_metadata(const SnapshotDescriptor& s,uint64_t max_tokens) {
    // The rule itself lives in include/strata/core/serving_raw_metadata.hpp: one definition, driven by the CPU
    // gate (tests/core/serving_raw_contracts). This name and position are what the resident-admission source
    // equality extracts, so only the forwarding stays here.
    return serving_raw::bounded_capture_metadata(s,max_tokens);
}
bool present(const StateDigest& digest) {
    return std::any_of(digest.begin(),digest.end(),[](uint8_t b){return b!=0;});
}
bool capture_metadata(const SnapshotDescriptor& s) {
    return serving_raw::capture_metadata_acceptable(s);   // shared definition, same reason as above
}
}
SuffixRawTransaction::SuffixRawTransaction(const ModelGeometry& g,const WindowKey& identity,NativeSessionAllocation a,
    std::shared_ptr<const void> runtime,std::vector<Verifier*> v,WindowBudget& b,
    uint64_t tokens,uint64_t bytes,uint64_t host_limit,bool steering)
    :geometry_(g),geometry_identity_(&g),identity_(identity),allocation_(std::move(a)),runtime_(std::move(runtime)),
     verifiers_(std::move(v)),budget_(b),max_tokens_(tokens),max_bytes_(bytes),host_limit_(host_limit),steering_enabled_(steering) {}
SuffixRawTransaction::~SuffixRawTransaction() {
    // This helper is exclusively for a disposable process. A driver must exit
    // the process on uncertain work, not unwind and free DMA/callback targets.
    if(phase_==Phase::running || phase_==Phase::capturing ||
       phase_==Phase::restoring || phase_==Phase::poisoned) std::terminate();
}
SuffixRawResult SuffixRawTransaction::poison(const char* why) {
    phase_=Phase::poisoned;allocation_.mark_unhealthy();
    return {SuffixRawStatus::poisoned,why,pending_ ?
        std::static_pointer_cast<const void>(pending_):runtime_};
}
bool SuffixRawTransaction::resolve(std::vector<NativeWindowStage>& config,std::string& error) const {
    const auto& stages=allocation_.stages();
    if(!runtime_ || !allocation_.healthy() || stages.empty() || stages.size()>8 ||
       verifiers_.size()!=stages.size()) {error="missing healthy owned runtime/stage gang";return false;}
    for(size_t i=0;i<stages.size();++i) {
        const auto& stage=stages[i];
        if(!stage.owner || !stage.session || !verifiers_[i] || stage.device<0 ||
           size_t(stage.device)>=allocation_.reservation()->resources().device_bytes.size() ||
           !allocation_.reservation()->resources().device_bytes[size_t(stage.device)] ||
           (slot_occupancy_ ?
             (!slot_occupancy_->active()||slot_bindings_.size()!=stages.size()||
              slot_bindings_[i].token().slot!=slot_occupancy_->slot()||
              !slot_bindings_[i].matches_session(stage.session,stage.device)||
              !verifiers_[i]->owned_batch_slot_matches(slot_bindings_[i])):
             !verifiers_[i]->snapshot_binding_matches(stage.session,stage.device))) {
            error="actual verifier/session/device binding differs";return false;
        }
        for(size_t j=0;j<i;++j) if(stages[j].session==stage.session || verifiers_[j]==verifiers_[i]) {
            error="repeated stage binding";return false;
        }
        const auto& ss=*stage.session;
        if(!ss.qsa_states || ss.qsa_alloc<=0 || ss.qsa_ord0<0 ||
           ss.qsa_alloc>geometry_.n_qsa_layers() ||
           ss.qsa_ord0>geometry_.n_qsa_layers()-ss.qsa_alloc) {
            error="invalid actual QSA allocation";return false;
        }
        NativeWindowStage item{ss.layer_lo,ss.layer_hi,{},ss.ple_hist!=nullptr};
        for(int64_t q=0;q<ss.qsa_alloc;++q) {
            const auto& st=ss.qsa_states[ss.qsa_ord0+q];
            item.qsa_formats.push_back((st.kv_hybrid?3:qsa_kv_format(st))+
                (st.kv_rot&&!st.kv_q4&&!st.kv_hybrid?16:0));
        }
        config.push_back(std::move(item));
    }
    if(initialized_) {
        if(config.size()!=initial_config_.size()) {error="runtime stage configuration changed";return false;}
        for(size_t i=0;i<config.size();++i) {
            const auto& a=config[i];const auto& b=initial_config_[i];
            if(a.layer_begin!=b.layer_begin || a.layer_end!=b.layer_end || a.qsa_formats!=b.qsa_formats || a.ple!=b.ple) {
                error="runtime native/PLE configuration changed";return false;
            }
        }
    }
    return true;
}
bool SuffixRawTransaction::identity_matches(const WindowKey& key) const {
    return key.engine_generation==identity_.engine_generation && key.window_generation==identity_.window_generation &&
        key.session_id==identity_.session_id && key.thread_id==identity_.thread_id && key.window_id==identity_.window_id &&
        key.model_profile==identity_.model_profile && key.topology==identity_.topology;
}
SuffixRawResult SuffixRawTransaction::initialize() {
#if !defined(__linux__) || !defined(__GLIBCXX__) || defined(STRATA_GPU_METAL)
    return invalid("initial standalone RAW helper requires Linux/libstdc++ native sessions");
#else
    if(controller_!=std::this_thread::get_id() || phase_!=Phase::idle || setup_credit_ || !max_tokens_ || !max_bytes_ || !host_limit_ ||
       max_tokens_>SIZE_MAX || max_bytes_>SIZE_MAX || geometry_.n_layers<=0 || geometry_.n_layers>4096) return invalid("invalid setup limits/phase");
    if(!identity_.engine_generation || !identity_.window_generation || identity_.session_id.empty() ||
       identity_.thread_id.empty() || identity_.window_id.empty() || identity_.session_id.size()>4096 ||
       identity_.thread_id.size()>4096 || identity_.window_id.size()>4096 ||
       !present(identity_.model_profile) || !present(identity_.topology))return invalid("missing fixed actual runtime identity");
    const auto& parent=allocation_.reservation();
    if(!parent || !parent->canonical_owner(parent) ||
       (parent->purpose()!=ReservationPurpose::private_work &&
        parent->purpose()!=ReservationPurpose::saved_maintenance))
        return invalid("runtime parent must be genuine private_work/saved_maintenance");
    // Setup baseline admission precedes layout/validation/container metadata.
    // It is intentionally retained for the entire controller lifetime.
    uint64_t tokens=0,layers=0,bytes=8ULL<<20;
    if(!product(max_tokens_,128,tokens) || !product(uint64_t(geometry_.n_layers),65536,layers) ||
       !add(bytes,tokens) || !add(bytes,layers)) return invalid("setup quote overflow");
    try {
        // Bounded bootstrap metadata is included in the setup quote, analogous
        // to WindowBudget's precharge lease/control-block transients.
        if(bytes>host_limit_) return {SuffixRawStatus::budget_refused,"H cannot cover setup metadata",{}};
        host_cap_=std::make_shared<HostCap>(host_limit_);
        auto admitted=admit_host(bytes,true,setup_credit_);if(!admitted)return admitted;
        std::vector<NativeWindowStage> config;std::string error;
        if(!resolve(config,error)) return {SuffixRawStatus::invalid,std::move(error),{}};
        initial_config_=std::move(config);initialized_=true;
        return {SuffixRawStatus::ok,{},{}};
    } catch(const std::bad_alloc&) {return {SuffixRawStatus::budget_refused,"setup allocation refused",{}};}
#endif
}
SuffixRawResult SuffixRawTransaction::initialize_resident_shared_host(const SuffixRawTransaction& live) {
#if !defined(__linux__) || !defined(__GLIBCXX__) || defined(STRATA_GPU_METAL)
    return invalid("resident shared RAW setup requires Linux/libstdc++ native sessions");
#else
    if(this==&live||controller_!=std::this_thread::get_id()||controller_!=live.controller_||
       phase_!=Phase::idle||setup_credit_||initialized_||host_cap_||!live.initialized_||
       live.phase_!=Phase::drained||!live.host_cap_||&budget_!=&live.budget_||
       host_limit_!=live.host_limit_||!max_tokens_||max_tokens_>live.max_tokens_||
       !max_bytes_||max_bytes_>live.max_bytes_||steering_enabled_||live.steering_enabled_||
       !live.allocation_.healthy()||!identity_matches(live.identity_)||
       geometry_.n_layers<=0||geometry_.n_layers>4096)
        return invalid("resident RAW setup requires actual same-ledger drained host-cap owner");
    const auto& parent=allocation_.reservation();
    if(!parent||!parent->canonical_owner(parent)||!allocation_.healthy()||
       (parent->purpose()!=ReservationPurpose::private_work&&parent->purpose()!=ReservationPurpose::saved_maintenance))
        return invalid("resident RAW parent is not genuine private allocation");
    uint64_t tokens=0,layers=0,bytes=8ULL<<20;
    if(!product(max_tokens_,128,tokens)||!product(uint64_t(geometry_.n_layers),65536,layers)||
       !add(bytes,tokens)||!add(bytes,layers))return invalid("resident RAW setup quote overflow");
    try {
        host_cap_=live.host_cap_; // SAME actual H, never a per-runtime second cap
        auto admitted=admit_host(bytes,true,setup_credit_);if(!admitted){host_cap_.reset();return admitted;}
        std::vector<NativeWindowStage> config;std::string error;
        if(!resolve(config,error))return {SuffixRawStatus::invalid,std::move(error),{}};
        initial_config_=std::move(config);initialized_=true;return {SuffixRawStatus::ok,{},{}};
    }catch(const std::bad_alloc&){if(!setup_credit_)host_cap_.reset();return {SuffixRawStatus::budget_refused,"resident RAW shared setup allocation refused",{}};}
#endif
}

SuffixRawResult SuffixRawTransaction::initialize_slot_shared_host(const SuffixRawTransaction& live) {
#if !defined(__linux__) || !defined(__GLIBCXX__) || defined(STRATA_GPU_METAL)
    return invalid("resident shared RAW setup requires Linux/libstdc++ native sessions");
#else
    if(this==&live||controller_!=std::this_thread::get_id()||controller_!=live.controller_||
       phase_!=Phase::idle||setup_credit_||initialized_||host_cap_||!live.initialized_||!live.setup_credit_||
       (live.phase_!=Phase::idle&&live.phase_!=Phase::drained)||!live.host_cap_||&budget_!=&live.budget_||
       host_limit_!=live.host_limit_||!max_tokens_||max_tokens_>live.max_tokens_||
       !max_bytes_||max_bytes_>live.max_bytes_||steering_enabled_||live.steering_enabled_||
       !live.allocation_.healthy()||!slot_occupancy_||!slot_occupancy_->active()||
       geometry_identity_!=live.geometry_identity_||runtime_.get()!=live.runtime_.get()||
       runtime_.owner_before(live.runtime_)||live.runtime_.owner_before(runtime_)||
       identity_.model_profile!=live.identity_.model_profile||
       geometry_.n_layers<=0||geometry_.n_layers>4096)
        return invalid("slot RAW setup requires actual same-runtime typed owner and aggregate H");
    const auto& parent=allocation_.reservation();
    if(!parent||!parent->canonical_owner(parent)||!allocation_.healthy()||
       (parent->purpose()!=ReservationPurpose::private_work&&parent->purpose()!=ReservationPurpose::saved_maintenance))
        return invalid("slot RAW parent is not genuine private allocation");
    uint64_t tokens=0,layers=0,bytes=8ULL<<20;
    if(!product(max_tokens_,128,tokens)||!product(uint64_t(geometry_.n_layers),65536,layers)||
       !add(bytes,tokens)||!add(bytes,layers))return invalid("slot RAW setup quote overflow");
    try {
        // The genuine root bootstrap keeps the expanded fixed claim metadata
        // charged through every descendant, including after root adapter drop.
        if(live.setup_credit_->bytes<48*(sizeof(HostCredit)+256))return invalid("shared H bootstrap metadata bound");
        slot_host_root_credit_=live.setup_credit_;
        host_cap_=live.host_cap_;host_cap_->claim_limit=48; //16*(main+twoactualslotcontexts), bytes H unchanged
        auto admitted=admit_host(bytes,true,setup_credit_);if(!admitted){host_cap_.reset();return admitted;}
        std::vector<NativeWindowStage> config;std::string error;
        if(!resolve(config,error))return {SuffixRawStatus::invalid,std::move(error),{}};
        initial_config_=std::move(config);initialized_=true;return {SuffixRawStatus::ok,{},{}};
    }catch(const std::bad_alloc&){if(!setup_credit_)host_cap_.reset();return {SuffixRawStatus::budget_refused,"slot RAW shared setup allocation refused",{}};}
#endif
}

SuffixRawResult SuffixRawTransaction::begin_driver_work() {
    if(controller_!=std::this_thread::get_id() || !initialized_ || (phase_!=Phase::idle && phase_!=Phase::drained)) return invalid("work requires initialized idle/drained runtime");
    if(!allocation_.healthy()) return poison("owned allocation already unhealthy");
    phase_=Phase::running;return {SuffixRawStatus::ok,{},{}};
}
SuffixRawResult SuffixRawTransaction::poll_all() {
    for(auto* verifier:verifiers_) {
        std::string error;
        const auto result=verifier->poll_snapshot_drain(error);
        if(result!=Verifier::SnapshotDrain::drained)
            return poison(result==Verifier::SnapshotDrain::pending ?
                "actual verifier work still pending at synchronous controller cut":
                "actual verifier drain unhealthy/unsupported");
    }
    return {SuffixRawStatus::ok,{},{}};
}
SuffixRawResult SuffixRawTransaction::complete_driver_work(bool succeeded,const StateCut& cut,
    std::span<const StateCut> stage_cuts) {
    if(controller_!=std::this_thread::get_id() || phase_!=Phase::running) return invalid("completion without a running forward/reset");
    if(!succeeded || stage_cuts.size()!=allocation_.stages().size() ||
       !std::all_of(stage_cuts.begin(),stage_cuts.end(),[&](const auto& c){return c==cut;}) ||
       !cut.id || !present(cut.history) || cut.exact_frontier>cut.forwarded || cut.forwarded>cut.accepted || cut.accepted>max_tokens_)
        return poison("forward/reset did not certify one all-stage cut");
    auto drained=poll_all();if(!drained) return drained;
    cut_=cut;phase_=Phase::drained;return {SuffixRawStatus::ok,{},{}};
}
bool SuffixRawTransaction::quote(const WindowStateLayout& layout,uint64_t copies,uint64_t& bytes) const {
    // Pinned libstdc++ profile, NOT a C++ allocator/RSS guarantee. Empty
    // ConversationBuffer quotes include segment growth and transient directories.
    // Native endpoint vectors can grow across layers: <=2x logical payload in
    // this profile. Capture native(2P)+descriptor(P)+frozen(P) =>4P.
    // Restore materialization can transiently hold old+new endpoint vectors:
    // <=3P during their pinned growth profile. Metadata margin is explicit.
    bytes=1ULL<<20;ConversationBuffer empty;
    uint64_t token_meta=0;
    if(!product(max_tokens_,128,token_meta) || !add(bytes,token_meta)) return false;
    for(const auto& stage:layout.stages) {
        if(!add(bytes,1ULL<<20)) return false;
        for(const auto& tensor:stage.tensors) {
            if(tensor.bytes>SIZE_MAX) return false;
            const auto peak=empty.allocation_peak(tensor.bytes);
            uint64_t payload=0;
            if(peak==SIZE_MAX || !product(peak,copies,payload) ||
               !add(bytes,payload) || !add(bytes,65536)) return false;
        }
    }
    return bytes<=SIZE_MAX;
}
SuffixRawResult SuffixRawTransaction::reserve_payload(uint64_t bytes,std::shared_ptr<Storage>& out) {
    std::shared_ptr<HostCredit> credit;auto admitted=admit_host(bytes,true,credit);if(!admitted)return admitted;
    auto s=std::make_shared<Storage>();s->credit=std::move(credit);
    s->runtime=runtime_;s->allocation=allocation_;out=std::move(s);
    return {SuffixRawStatus::ok,{},{}};
}
SuffixRawResult SuffixRawTransaction::admit_host(uint64_t bytes,bool global_charge,std::shared_ptr<HostCredit>& out) {
    if(!host_cap_ || !bytes || bytes>SIZE_MAX)return invalid("invalid host cap claim");
    auto credit=std::make_shared<HostCredit>();
    // Atomics permit final bundle/claim destruction off the controller thread;
    // admissions themselves remain private and single-controller-thread only.
    unsigned slots=host_cap_->claims.load();
    do {
        if(slots>=host_cap_->claim_limit)return {SuffixRawStatus::budget_refused,"fixed host claim slots exhausted",{}};
    } while(!host_cap_->claims.compare_exchange_weak(slots,slots+1));
    uint64_t live=host_cap_->live.load();
    do {
        if(live>host_cap_->limit || bytes>host_cap_->limit-live) {
            host_cap_->claims.fetch_sub(1);
            return {SuffixRawStatus::budget_refused,"aggregate suffix H exhausted",{}};
        }
    } while(!host_cap_->live.compare_exchange_weak(live,live+bytes));
    credit->cap=host_cap_;credit->bytes=bytes;
    if(global_charge) {
        WindowResources r;r.device_bytes.resize(allocation_.reservation()->resources().device_bytes.size());r.host_bytes=bytes;
        auto reserved=budget_.reserve(r,ReservationPurpose::private_attachment,allocation_.reservation());
        if(!reserved)return {SuffixRawStatus::budget_refused,"actual global host attachment refused (including wrong ledger)",{}};
        credit->global=std::move(reserved.lease);
    }
    out=std::move(credit);return {SuffixRawStatus::ok,{},{}};
}
SuffixRawResult SuffixRawTransaction::admit_controller_host(uint64_t bytes,std::shared_ptr<const void>& out) {
    if(out || controller_!=std::this_thread::get_id() || !initialized_ ||
       (phase_!=Phase::idle && phase_!=Phase::drained))return invalid("controller host admission requires initialized idle/drained phase");
    try {
        std::shared_ptr<HostCredit> credit;auto result=admit_host(bytes,true,credit);
        if(result)out=std::move(credit);
        return result;
    } catch(const std::bad_alloc&) {return {SuffixRawStatus::budget_refused,"controller host admission allocation refused",{}};}
}
SuffixRawResult SuffixRawTransaction::admit_resident_setup(const WindowResources& resources,
    std::shared_ptr<const void>& out) {
    if(out || controller_!=std::this_thread::get_id() || !initialized_ ||
       phase_!=Phase::drained || !allocation_.healthy() || !allocation_.reservation() ||
       resources.device_bytes.size()!=allocation_.reservation()->resources().device_bytes.size() ||
       !resources.host_bytes || resources.pinned_bytes>resources.host_bytes ||
       resources.disk_bytes || resources.io_slots || resources.shadows)
        return invalid("resident setup requires drained owner and bounded setup resources");
    try {
        // Aggregate first, global second; neither allocation payload nor CUDA
        // work exists here. Failure drops the temporary aggregate claim.
        std::shared_ptr<HostCredit> credit;
        auto admitted=admit_host(resources.host_bytes,false,credit);
        if(!admitted)return admitted;
        auto reserved=budget_.reserve(resources,ReservationPurpose::private_attachment,
                                      allocation_.reservation());
        if(!reserved)return {SuffixRawStatus::budget_refused,"resident global setup admission refused",{}};
        credit->global=std::move(reserved.lease);
        out=std::move(credit);
        return {SuffixRawStatus::ok,{},{}};
    } catch(const std::bad_alloc&) {
        return {SuffixRawStatus::budget_refused,"resident setup admission allocation refused",{}};
    }
}
SuffixRawResult SuffixRawTransaction::admit_completion_gate_metadata(
    std::shared_ptr<const ReservationLease>& metadata,
    std::shared_ptr<const void>& aggregate_credit) {
    if(metadata || aggregate_credit || controller_!=std::this_thread::get_id() ||
       !initialized_ || phase_!=Phase::drained || !allocation_.healthy() ||
       !allocation_.reservation())
        return invalid("completion metadata requires actual drained resident controller");
    try {
        const auto bytes=VerifierCompletionGate::metadata_bytes();
        std::shared_ptr<HostCredit> credit;
        auto admitted=admit_host(bytes,false,credit);
        if(!admitted)return admitted;
        WindowResources resources;
        resources.device_bytes.resize(allocation_.reservation()->resources().device_bytes.size());
        resources.host_bytes=bytes;
        auto reserved=budget_.reserve(resources,ReservationPurpose::foreground);
        if(!reserved)return {SuffixRawStatus::budget_refused,"completion metadata global admission refused",{}};
        // Keep the canonical lease exclusive for prepare_serial_completion_gate.
        // The aggregate claim is a separate retained lifetime owner, not a second
        // global charge or an alias that defeats canonical-owner validation.
        metadata=std::move(reserved.lease);aggregate_credit=std::move(credit);
        return {SuffixRawStatus::ok,{},{}};
    } catch(const std::bad_alloc&) {
        return {SuffixRawStatus::budget_refused,"completion metadata allocation refused",{}};
    }
}
SuffixRawResult SuffixRawTransaction::claim_external_splice(uint64_t bytes,std::shared_ptr<const void>& out) {
    if(out || controller_!=std::this_thread::get_id() || !initialized_ || phase_!=Phase::drained)
        return invalid("splice cap admission requires actual drained phase");
    try {
        std::shared_ptr<HostCredit> credit;auto result=admit_host(bytes,false,credit);
        if(result)out=std::move(credit);
        return result;
    } catch(const std::bad_alloc&) {return {SuffixRawStatus::budget_refused,"splice cap admission allocation refused",{}};}
}
SuffixRawResult SuffixRawTransaction::capture(const SnapshotDescriptor& metadata,BundleHandle& out) {
    if(controller_!=std::this_thread::get_id() || phase_!=Phase::drained || !initialized_ || metadata.cut!=cut_ ||
       !identity_matches(metadata.key) || !metadata.cut.forwarded || !metadata.devices.empty() ||
       metadata.steering_enabled!=steering_enabled_ || !bounded_metadata(metadata,max_tokens_) ||
       !capture_metadata(metadata))
        return invalid("capture requires bounded metadata at actual common forwarded cut");
    try {
        std::vector<NativeWindowStage> config;std::string error;
        if(!resolve(config,error)) return {SuffixRawStatus::invalid,std::move(error),{}};
        WindowStateLayout layout;
        auto valid=native_window_layout(geometry_,config,metadata.key,metadata.cut,max_tokens_,max_bytes_,layout,metadata.steering_enabled);
        if(!valid) return {SuffixRawStatus::invalid,valid.detail,{}};
        if(metadata.accepted_tokens.size()!=cut_.accepted || metadata.pending_tokens.size()!=cut_.accepted-cut_.forwarded ||
           window_history_digest(metadata.accepted_tokens)!=cut_.history ||
           !std::equal(metadata.pending_tokens.begin(),metadata.pending_tokens.end(),metadata.accepted_tokens.begin()+cut_.forwarded) ||
           std::any_of(metadata.accepted_tokens.begin(),metadata.accepted_tokens.end(),[](int32_t t){return t<0;}))
            return invalid("capture history/pending differs");
        uint64_t bytes=0;if(!quote(layout,4,bytes)) return invalid("capture peak overflow");
        std::shared_ptr<Storage> storage;auto admitted=reserve_payload(bytes,storage);if(!admitted)return admitted;
        std::vector<int32_t> forwarded(metadata.accepted_tokens.begin(),metadata.accepted_tokens.begin()+cut_.forwarded);
        const std::vector<ConversationImageKey> images;const std::vector<ConversationCheckpoint> checkpoints;
        ConversationView view{forwarded,images,checkpoints,metadata.steering_enabled};
        storage->native.stage_images.resize(config.size()-1);storage->ple.reserve(config.size());
        pending_=storage;phase_=Phase::capturing;
        for(size_t i=0;i<config.size();++i) {
            const auto& stage=allocation_.stages()[i];WindowDeviceScope device(stage.device);
            if(!device.ok()) return poison("capture owning-device selection failed");
            auto& image=i?storage->native.stage_images[i-1]:storage->native;
            if(!conversation_snapshot_save_private(image,view,*stage.session,geometry_,error)) return poison("RAW capture copy/synchronize failed");
            if(!device.finish()) return poison("capture caller-device restoration failed");
            storage->ple.push_back({{stage.session->ple_prev[0],stage.session->ple_prev[1]},stage.session->ple_token});
        }
        valid=describe_native_window(storage->native,geometry_,layout,metadata,storage->ple,storage->descriptor);
        if(!valid) return poison("captured native description failed");
        BundleHandle frozen;
        valid=BundleHandle::freeze(storage->descriptor,metadata.key,layout,frozen,storage->credit);
        if(!valid) return poison("captured immutable freeze failed");
        auto drained=poll_all();if(!drained)return drained;
        out=std::move(frozen);phase_=Phase::drained;pending_.reset();
        return {SuffixRawStatus::ok,{},{}};
    } catch(...) {
        if(phase_==Phase::capturing) return poison("capture exception after transfer attempt");
        return {SuffixRawStatus::budget_refused,"capture preparation allocation refused",{}};
    }
}

// Private disposable diagnostic only; the public nonempty capture guard above stays unchanged.
SuffixRawResult SuffixRawTransaction::capture_empty_oracle(const SnapshotDescriptor& metadata,BundleHandle& out) {
    if(controller_!=std::this_thread::get_id() || phase_!=Phase::drained || !initialized_ || metadata.cut!=cut_ ||
       !identity_matches(metadata.key) || metadata.cut.forwarded || metadata.cut.accepted || !metadata.devices.empty() ||
       metadata.steering_enabled!=steering_enabled_ || !bounded_metadata(metadata,max_tokens_) ||
       !capture_metadata(metadata))
        return invalid("capture requires bounded metadata at actual common forwarded cut");
    try {
        std::vector<NativeWindowStage> config;std::string error;
        if(!resolve(config,error)) return {SuffixRawStatus::invalid,std::move(error),{}};
        WindowStateLayout layout;
        auto valid=native_window_layout(geometry_,config,metadata.key,metadata.cut,max_tokens_,max_bytes_,layout,metadata.steering_enabled);
        if(!valid) return {SuffixRawStatus::invalid,valid.detail,{}};
        if(metadata.accepted_tokens.size()!=cut_.accepted || metadata.pending_tokens.size()!=cut_.accepted-cut_.forwarded ||
           window_history_digest(metadata.accepted_tokens)!=cut_.history ||
           !std::equal(metadata.pending_tokens.begin(),metadata.pending_tokens.end(),metadata.accepted_tokens.begin()+cut_.forwarded) ||
           std::any_of(metadata.accepted_tokens.begin(),metadata.accepted_tokens.end(),[](int32_t t){return t<0;}))
            return invalid("capture history/pending differs");
        uint64_t bytes=0;if(!quote(layout,4,bytes)) return invalid("capture peak overflow");
        std::shared_ptr<Storage> storage;auto admitted=reserve_payload(bytes,storage);if(!admitted)return admitted;
        std::vector<int32_t> forwarded(metadata.accepted_tokens.begin(),metadata.accepted_tokens.begin()+cut_.forwarded);
        const std::vector<ConversationImageKey> images;const std::vector<ConversationCheckpoint> checkpoints;
        ConversationView view{forwarded,images,checkpoints,metadata.steering_enabled};
        storage->native.stage_images.resize(config.size()-1);storage->ple.reserve(config.size());
        pending_=storage;phase_=Phase::capturing;
        for(size_t i=0;i<config.size();++i) {
            const auto& stage=allocation_.stages()[i];WindowDeviceScope device(stage.device);
            if(!device.ok()) return poison("capture owning-device selection failed");
            auto& image=i?storage->native.stage_images[i-1]:storage->native;
            if(!conversation_snapshot_empty_save(image,*stage.session,geometry_,error)) return poison("RAW capture copy/synchronize failed");
            if(!device.finish()) return poison("capture caller-device restoration failed");
            storage->ple.push_back({{stage.session->ple_prev[0],stage.session->ple_prev[1]},stage.session->ple_token});
        }
        valid=describe_native_window(storage->native,geometry_,layout,metadata,storage->ple,storage->descriptor);
        if(!valid) return poison("captured native description failed");
        BundleHandle frozen;
        valid=BundleHandle::freeze(storage->descriptor,metadata.key,layout,frozen,storage->credit);
        if(!valid) return poison("captured immutable freeze failed");
        auto drained=poll_all();if(!drained)return drained;
        out=std::move(frozen);phase_=Phase::drained;pending_.reset();
        return {SuffixRawStatus::ok,{},{}};
    } catch(...) {
        if(phase_==Phase::capturing) return poison("capture exception after transfer attempt");
        return {SuffixRawStatus::budget_refused,"capture preparation allocation refused",{}};
    }
}
SuffixRawResult SuffixRawTransaction::restore(const BundleHandle& source,const WindowKey& expected,core::ResidentCopyFault* fault) {
#if !defined(STRATA_RESIDENT_FAULT_QUALIFICATION) || !defined(__linux__) || defined(STRATA_GPU_METAL) || defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_USE_SYCL) || defined(STRATA_GPU_SYCL)
    if(fault)return invalid("resident copy fault qualification disabled");
#endif
    if(controller_!=std::this_thread::get_id() || phase_!=Phase::drained || !initialized_ || !source ||
       !identity_matches(expected) || source.get()->key!=expected ||
       source.get()->steering_enabled!=steering_enabled_ ||
       !bounded_metadata(*source.get(),max_tokens_)) return invalid("restore requires bounded immutable private image and idle destination");
    try {
        std::vector<NativeWindowStage> config;std::string error;
        if(!resolve(config,error)) return {SuffixRawStatus::invalid,std::move(error),{}};
        WindowStateLayout layout;auto valid=native_window_layout(geometry_,config,expected,source.get()->cut,max_tokens_,max_bytes_,layout,source.get()->steering_enabled);
        if(!valid) return {SuffixRawStatus::invalid,valid.detail,{}};
        valid=validate_window_state(*source.get(),expected,layout);
        if(!valid)return {SuffixRawStatus::invalid,valid.detail,{}};
        uint64_t bytes=0;if(!quote(layout,3,bytes))return invalid("restore peak overflow");
        std::shared_ptr<Storage> storage;auto admitted=reserve_payload(bytes,storage);if(!admitted)return admitted;
        storage->source=source;
        valid=materialize_native_window(*source.get(),expected,geometry_,config,max_tokens_,max_bytes_,source.get()->steering_enabled,storage->native,storage->ple);
        if(!valid)return {SuffixRawStatus::invalid,valid.detail,{}};
        if(storage->ple.size()!=config.size())return invalid("missing actual PLE sidecars");
        for(size_t i=0;i<config.size();++i) {
            const auto& image=i?storage->native.stage_images[i-1]:storage->native;
            if(!conversation_snapshot_validate(image,*allocation_.stages()[i].session,geometry_,static_cast<const QsaState*>(nullptr),error))
                return {SuffixRawStatus::invalid,std::move(error),{}};
        }
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
        if(fault){
            const bool serving=fault->serving_.site()!=core::resident_serving_fault::Site::none;
            if(config.size()!=2||fault->fd_<0||fault->fired_||fault->stage0_completed_||
               (!serving&&fault->publication_!=2)||fault->target_cut_!=source.get()->cut.id||fault->controller_!=controller_)
                return invalid("partial-copy qualification arm differs from actual restore");
            if(serving){const auto& bound=fault->serving_snapshot_;const auto& incoming=*source.get();
                if(!core::resident_serving_fault::copy_scope_valid(fault->serving_,bound)||
                   allocation_.stages()[0].device!=0||allocation_.stages()[1].device!=1||fault->expected_device_!=1||
                   bound.live_cut!=cut_.id||bound.accepted!=cut_.accepted||bound.forwarded!=cut_.forwarded||
                   bound.exact!=cut_.exact_frontier||bound.watermark!=cut_.output_watermark||bound.history!=cut_.history||
                   incoming.cut.forwarded!=bound.forwarded||incoming.cut.accepted!=bound.accepted||
                   incoming.cut.history!=bound.history||incoming.cut.output_watermark!=bound.watermark||
                   (fault->serving_.site()==core::resident_serving_fault::Site::second_partial_copy&&
                       incoming.cut.exact_frontier!=bound.repaired)||
                   (fault->serving_.site()==core::resident_serving_fault::Site::final_exact_copy&&
                       (incoming.cut.exact_frontier!=incoming.cut.forwarded||!incoming.approximation.empty())))
                    return invalid("serving copy purpose differs from actual source/destination cuts");
            }
        }
#endif
        // REPEATED-RESTORE DEFECT: a snapshot serializes a page-rounded prefix, so the destination rows beyond
        // it are omitted by the copy and keep whatever a previous cycle wrote.  Those rows become evidence at
        // the next capture (the kernel's partial page includes them), which is what failed the second cycle's
        // native137 comparison while tokens and logits stayed identical.  Canonicalize the omitted tail here,
        // on the live destination, before the restore copy - never by masking comparator fields or rewriting
        // saved evidence, and never by zeroing serialized bytes, indexer/GDN/PLE state or whole-model state.
        //
        // Every range is planned and checked for ALL stages and layers BEFORE any device mutation, and each
        // range is bound to the extents of the image that stage will actually restore: `image.live.ids.size()`
        // is the same value conversation_snapshot_restore() hands to layout(), so clear-start == copy-end by
        // construction rather than by agreement between two counters.
        const int64_t page=strata::kernels::qsa_real_shapes().page_size;
        std::vector<RawInt8RestoreClearStage> extents(config.size());
        for(size_t i=0;i<config.size();++i){
            const auto& image=i?storage->native.stage_images[i-1]:storage->native;
            const auto& session=*allocation_.stages()[i].session;
            const size_t layers=size_t(std::max<int64_t>(session.qsa_alloc,0));
            if(layers&&!session.qsa_states)return invalid("RAW suffix preflight requires the session's owned QSA states");
            if(image.kv.size()!=layers)return invalid("RAW suffix preflight sees a K/V layer count the restore will not write");
            extents[i].forwarded=uint64_t(image.live.ids.size());
            extents[i].layers.reserve(layers);
            for(size_t j=0;j<layers;++j){
                const auto& kv=image.kv[j];
                extents[i].layers.push_back({kv.format,kv.cells,kv.heads,kv.head_dim,kv.page_size,
                    kv.k.size(),kv.v.size(),kv.k_scale.size(),kv.v_scale.size()});
            }
        }
        std::vector<std::vector<RawInt8RestoreClearPlan>> clear_plans;
        const char* clear_reason=nullptr;
        if(!raw_int8_restore_clear_preflight(extents,[&](size_t i,size_t j)->const QsaState&{
                const auto& session=*allocation_.stages()[i].session;
                return session.qsa_states[session.qsa_ord0+int64_t(j)];},geometry_,page,clear_plans,clear_reason))
            return invalid(clear_reason?clear_reason:"RAW suffix restore refused before any write");
        pending_=storage;phase_=Phase::restoring;
        for(size_t i=0;i<config.size();++i) {
            const auto& stage=allocation_.stages()[i];WindowDeviceScope device(stage.device);
            if(!device.ok())return poison("restore owning-device selection failed");
            const auto& image=i?storage->native.stage_images[i-1]:storage->native;
            // The clear runs on THIS stage's device, in scope, on this thread's private non-blocking copy
            // stream - the same one the restore's copies use.  It is deliberately re-acquired per stage:
            // private_stream() is per-thread PER DEVICE, so a handle taken on stage 0's GPU is invalid here.
            cudaStream_t clear_stream=nullptr;bool clear_stream_ready=false;
            auto clear=[&](void* data,size_t offset,size_t bytes)->bool{
                if(!clear_stream_ready){
                    if(conversation_detail::private_stream(clear_stream)!=cudaSuccess)return false;
                    clear_stream_ready=true;
                }
                return cudaMemsetAsync(static_cast<uint8_t*>(data)+offset,0,bytes,clear_stream)==cudaSuccess;
            };
            for(size_t j=0;j<clear_plans[i].size();++j){
                RawInt8RestoreClearPlan plan;
                const auto& session=*stage.session;
                if(!raw_int8_restore_clear_plan(session.qsa_states[session.qsa_ord0+int64_t(j)],geometry_,
                        extents[i].forwarded,strata::kernels::qsa_real_shapes().page_size,plan)||
                   plan!=clear_plans[i][j])
                    return poison("RAW suffix shape changed after preflight");
                if(!apply_raw_int8_restore_clear_plan(plan,clear))return poison("RAW unserialized INT8 suffix clear failed");
            }
            // Checked completion BEFORE the restore copy or any read: a queued clear that has not landed is not
            // a canonicalized destination.  Failure poisons; the owners in pending_ stay retained.
            if(clear_stream_ready&&cudaStreamSynchronize(clear_stream)!=cudaSuccess)
                return poison("RAW unserialized INT8 suffix clear drain failed");
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
            core::ResidentCopyFaultScope fault_scope(i==1?fault:nullptr);
#endif
            if(conversation_snapshot_restore(image,*stage.session,geometry_,static_cast<const QsaState*>(nullptr),error)!=ConversationRestore::restored)
                return poison("private RAW restore partially failed");
            stage.session->ple_prev[0]=storage->ple[i].previous[0];stage.session->ple_prev[1]=storage->ple[i].previous[1];
            stage.session->ple_token=storage->ple[i].current;
            if(cudaDeviceSynchronize()!=cudaSuccess)return poison("private RAW restore drain failed");
            if(!device.finish())return poison("restore caller-device restoration failed");
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
            if(fault&&i==0)fault->stage0_completed_=true; // actual stage0 restore+sync+device restoration succeeded
            if(fault&&i==1&&!fault->fired_)return poison("armed partial-copy site not reached");
#endif
        }
        auto drained=poll_all();if(!drained)return drained;
        cut_=source.get()->cut;phase_=Phase::drained;pending_.reset();
        return {SuffixRawStatus::ok,{},{}};
    } catch(...) {
        if(phase_==Phase::restoring)return poison("restore exception after transfer attempt");
        return {SuffixRawStatus::budget_refused,"restore preparation allocation refused",{}};
    }
}

// Private disposable native diagnostic; public restore above is unchanged.
SuffixRawResult SuffixRawTransaction::restore_empty_oracle(const BundleHandle& source,const WindowKey& expected,core::ResidentCopyFault* fault) {
#if !defined(STRATA_RESIDENT_FAULT_QUALIFICATION) || !defined(__linux__) || defined(STRATA_GPU_METAL) || defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_USE_SYCL) || defined(STRATA_GPU_SYCL)
    if(fault)return invalid("resident copy fault qualification disabled");
#endif
    if(controller_!=std::this_thread::get_id() || phase_!=Phase::drained || !initialized_ || !source || source.get()->cut.forwarded || source.get()->cut.accepted ||
       !identity_matches(expected) || source.get()->key!=expected ||
       source.get()->steering_enabled!=steering_enabled_ ||
       !bounded_metadata(*source.get(),max_tokens_)) return invalid("restore requires bounded immutable private image and idle destination");
    try {
        std::vector<NativeWindowStage> config;std::string error;
        if(!resolve(config,error)) return {SuffixRawStatus::invalid,std::move(error),{}};
        WindowStateLayout layout;auto valid=native_window_layout(geometry_,config,expected,source.get()->cut,max_tokens_,max_bytes_,layout,source.get()->steering_enabled);
        if(!valid) return {SuffixRawStatus::invalid,valid.detail,{}};
        valid=validate_window_state(*source.get(),expected,layout);
        if(!valid)return {SuffixRawStatus::invalid,valid.detail,{}};
        uint64_t bytes=0;if(!quote(layout,3,bytes))return invalid("restore peak overflow");
        std::shared_ptr<Storage> storage;auto admitted=reserve_payload(bytes,storage);if(!admitted)return admitted;
        storage->source=source;
        valid=materialize_native_window(*source.get(),expected,geometry_,config,max_tokens_,max_bytes_,source.get()->steering_enabled,storage->native,storage->ple);
        if(!valid)return {SuffixRawStatus::invalid,valid.detail,{}};
        if(storage->ple.size()!=config.size())return invalid("missing actual PLE sidecars");
        for(size_t i=0;i<config.size();++i) {
            const auto& image=i?storage->native.stage_images[i-1]:storage->native;
            if(!conversation_snapshot_empty_validate(image,*allocation_.stages()[i].session,geometry_,static_cast<const QsaState*>(nullptr),error))
                return {SuffixRawStatus::invalid,std::move(error),{}};
        }
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
        if(fault){
            const bool serving=fault->serving_.site()!=core::resident_serving_fault::Site::none;
            if(config.size()!=2||fault->fd_<0||fault->fired_||fault->stage0_completed_||
               (!serving&&fault->publication_!=2)||fault->target_cut_!=source.get()->cut.id||fault->controller_!=controller_)
                return invalid("partial-copy qualification arm differs from actual restore");
            if(serving){const auto& bound=fault->serving_snapshot_;const auto& incoming=*source.get();
                if(!core::resident_serving_fault::copy_scope_valid(fault->serving_,bound)||
                   allocation_.stages()[0].device!=0||allocation_.stages()[1].device!=1||fault->expected_device_!=1||
                   bound.live_cut!=cut_.id||bound.accepted!=cut_.accepted||bound.forwarded!=cut_.forwarded||
                   bound.exact!=cut_.exact_frontier||bound.watermark!=cut_.output_watermark||bound.history!=cut_.history||
                   incoming.cut.forwarded!=bound.forwarded||incoming.cut.accepted!=bound.accepted||
                   incoming.cut.history!=bound.history||incoming.cut.output_watermark!=bound.watermark||
                   (fault->serving_.site()==core::resident_serving_fault::Site::second_partial_copy&&
                       incoming.cut.exact_frontier!=bound.repaired)||
                   (fault->serving_.site()==core::resident_serving_fault::Site::final_exact_copy&&
                       (incoming.cut.exact_frontier!=incoming.cut.forwarded||!incoming.approximation.empty())))
                    return invalid("serving copy purpose differs from actual source/destination cuts");
            }
        }
#endif
        // REPEATED-RESTORE DEFECT: a snapshot serializes a page-rounded prefix, so the destination rows beyond
        // it are omitted by the copy and keep whatever a previous cycle wrote.  Those rows become evidence at
        // the next capture (the kernel's partial page includes them), which is what failed the second cycle's
        // native137 comparison while tokens and logits stayed identical.  Canonicalize the omitted tail here,
        // on the live destination, before the restore copy - never by masking comparator fields or rewriting
        // saved evidence, and never by zeroing serialized bytes, indexer/GDN/PLE state or whole-model state.
        //
        // Every range is planned and checked for ALL stages and layers BEFORE any device mutation, and each
        // range is bound to the extents of the image that stage will actually restore: `image.live.ids.size()`
        // is the same value conversation_snapshot_empty_restore() hands to layout(), so clear-start == copy-end by
        // construction rather than by agreement between two counters.
        const int64_t page=strata::kernels::qsa_real_shapes().page_size;
        std::vector<RawInt8RestoreClearStage> extents(config.size());
        for(size_t i=0;i<config.size();++i){
            const auto& image=i?storage->native.stage_images[i-1]:storage->native;
            const auto& session=*allocation_.stages()[i].session;
            const size_t layers=size_t(std::max<int64_t>(session.qsa_alloc,0));
            if(layers&&!session.qsa_states)return invalid("RAW suffix preflight requires the session's owned QSA states");
            if(image.kv.size()!=layers)return invalid("RAW suffix preflight sees a K/V layer count the restore will not write");
            extents[i].forwarded=uint64_t(image.live.ids.size());
            extents[i].layers.reserve(layers);
            for(size_t j=0;j<layers;++j){
                const auto& kv=image.kv[j];
                extents[i].layers.push_back({kv.format,kv.cells,kv.heads,kv.head_dim,kv.page_size,
                    kv.k.size(),kv.v.size(),kv.k_scale.size(),kv.v_scale.size()});
            }
        }
        std::vector<std::vector<RawInt8RestoreClearPlan>> clear_plans;
        const char* clear_reason=nullptr;
        if(!raw_int8_restore_clear_preflight(extents,[&](size_t i,size_t j)->const QsaState&{
                const auto& session=*allocation_.stages()[i].session;
                return session.qsa_states[session.qsa_ord0+int64_t(j)];},geometry_,page,clear_plans,clear_reason))
            return invalid(clear_reason?clear_reason:"RAW suffix restore refused before any write");
        pending_=storage;phase_=Phase::restoring;
        for(size_t i=0;i<config.size();++i) {
            const auto& stage=allocation_.stages()[i];WindowDeviceScope device(stage.device);
            if(!device.ok())return poison("restore owning-device selection failed");
            const auto& image=i?storage->native.stage_images[i-1]:storage->native;
            // The clear runs on THIS stage's device, in scope, on this thread's private non-blocking copy
            // stream - the same one the restore's copies use.  It is deliberately re-acquired per stage:
            // private_stream() is per-thread PER DEVICE, so a handle taken on stage 0's GPU is invalid here.
            cudaStream_t clear_stream=nullptr;bool clear_stream_ready=false;
            auto clear=[&](void* data,size_t offset,size_t bytes)->bool{
                if(!clear_stream_ready){
                    if(conversation_detail::private_stream(clear_stream)!=cudaSuccess)return false;
                    clear_stream_ready=true;
                }
                return cudaMemsetAsync(static_cast<uint8_t*>(data)+offset,0,bytes,clear_stream)==cudaSuccess;
            };
            for(size_t j=0;j<clear_plans[i].size();++j){
                RawInt8RestoreClearPlan plan;
                const auto& session=*stage.session;
                if(!raw_int8_restore_clear_plan(session.qsa_states[session.qsa_ord0+int64_t(j)],geometry_,
                        extents[i].forwarded,strata::kernels::qsa_real_shapes().page_size,plan)||
                   plan!=clear_plans[i][j])
                    return poison("RAW suffix shape changed after preflight");
                if(!apply_raw_int8_restore_clear_plan(plan,clear))return poison("RAW unserialized INT8 suffix clear failed");
            }
            // Checked completion BEFORE the restore copy or any read: a queued clear that has not landed is not
            // a canonicalized destination.  Failure poisons; the owners in pending_ stay retained.
            if(clear_stream_ready&&cudaStreamSynchronize(clear_stream)!=cudaSuccess)
                return poison("RAW unserialized INT8 suffix clear drain failed");
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
            core::ResidentCopyFaultScope fault_scope(i==1?fault:nullptr);
#endif
            if(conversation_snapshot_empty_restore(image,*stage.session,geometry_,static_cast<const QsaState*>(nullptr),error)!=ConversationRestore::restored)
                return poison("private RAW restore partially failed");
            stage.session->ple_prev[0]=storage->ple[i].previous[0];stage.session->ple_prev[1]=storage->ple[i].previous[1];
            stage.session->ple_token=storage->ple[i].current;
            if(cudaDeviceSynchronize()!=cudaSuccess)return poison("private RAW restore drain failed");
            if(!device.finish())return poison("restore caller-device restoration failed");
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
            if(fault&&i==0)fault->stage0_completed_=true; // actual stage0 restore+sync+device restoration succeeded
            if(fault&&i==1&&!fault->fired_)return poison("armed partial-copy site not reached");
#endif
        }
        auto drained=poll_all();if(!drained)return drained;
        cut_=source.get()->cut;phase_=Phase::drained;pending_.reset();
        return {SuffixRawStatus::ok,{},{}};
    } catch(...) {
        if(phase_==Phase::restoring)return poison("restore exception after transfer attempt");
        return {SuffixRawStatus::budget_refused,"restore preparation allocation refused",{}};
    }
}

// ------------------------------------------------------------------ ServingRawSession (P12 serving ownership)
// Binds a private RAW transaction to an OWNED serving session. The allocation reached through
// DriverWindowRuntime::initialize_sessions -> allocate_native_sessions carries ReservationPurpose::private_work
// (src/core/window_allocator.cpp:317), which is exactly what initialize() requires
// (suffix_raw_runtime.cpp:162-166); an unowned serving session would be refused here rather than adapted.
core::WindowKey ServingRawSession::serving_identity(const void* runtime, const NativeSessionAllocation& allocation) {
    core::WindowKey key;
    key.engine_generation = key.window_generation = 1;
    key.session_id = "strata-serve-suffix";
    key.thread_id = "serve";
    key.window_id = "session";
    key.revision = key.append_epoch = key.selection_epoch = key.provenance_epoch = 1;
    // Process-local binding digests over the ACTUAL runtime and stage sessions. They are not a portable content
    // profile and are not a correctness certificate; they bind the operands to this one owned runtime, as the
    // standalone driver's identity() does for its own.
    core::WindowHasher model, topology;
    constexpr char model_domain[] = "serving-owned-session-v1";
    constexpr char topology_domain[] = "serving-stage-binding-v1";
    model.update(model_domain, sizeof(model_domain));
    topology.update(topology_domain, sizeof(topology_domain));
    const auto runtime_ptr = reinterpret_cast<uintptr_t>(runtime);
    model.update(&runtime_ptr, sizeof(runtime_ptr));
    for (const auto& stage : allocation.stages()) {
        if (stage.session == nullptr) continue;
        const auto session = reinterpret_cast<uintptr_t>(stage.session);
        topology.update(&session, sizeof(session));
        topology.update(&stage.device, sizeof(stage.device));
        topology.update(&stage.session->layer_lo, sizeof(stage.session->layer_lo));
        topology.update(&stage.session->layer_hi, sizeof(stage.session->layer_hi));
    }
    key.model_profile = model.digest();
    key.topology = topology.digest();
    return key;
}

std::shared_ptr<ServingRawSession> ServingRawSession::open(const ModelGeometry& geometry,
                                                           NativeSessionAllocation allocation,
                                                           std::shared_ptr<const void> runtime_owner,
                                                           std::vector<Verifier*> verifiers,
                                                           WindowBudget& budget, uint64_t max_tokens,
                                                           uint64_t host_cap_bytes, std::string& error) {
    if (!runtime_owner) { error = "serving RAW owner needs the runtime shared_ptr that actually owns the sessions"; return nullptr; }
    if (!allocation.healthy()) { error = "serving session allocation is not healthy"; return nullptr; }
    if (verifiers.empty() || allocation.stages().empty()) { error = "serving RAW owner has no verifier/stage set"; return nullptr; }
    for (const auto* verifier : verifiers)
        if (verifier == nullptr) { error = "null serving verifier"; return nullptr; }
    if (!max_tokens || max_tokens > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        error = "serving RAW max_tokens is zero or unbounded"; return nullptr;
    }
    if (!host_cap_bytes) { error = "serving RAW host cap must be explicit and positive"; return nullptr; }
    // This is a payload ceiling, not a token-metadata estimate. Recurrent state alone can exceed
    // max_tokens*128 + 16 MiB even for a short request. The actual native layout enumerates every
    // stage/tensor and capture/restore quote their complete overlap against this same aggregate host
    // allowance. A single payload cannot legitimately exceed that allowance; no guessed per-token
    // multiplier should reject a geometry that the real layout and ledger can admit.
    const uint64_t payload = host_cap_bytes;
    try {
        // `new`, not make_shared: the constructor is private and only this class is a friend, so the shared_ptr
        // control block cannot construct it. Ownership is still real: the transaction holds the runtime
        // shared_ptr, the verifier vector, the budget reference and its own setup credit for its whole life.
        const auto identity=serving_identity(runtime_owner.get(),allocation);
        std::shared_ptr<SuffixRawTransaction> tx(new SuffixRawTransaction(
            geometry, identity, allocation, std::move(runtime_owner),
            std::move(verifiers), budget, max_tokens, payload, host_cap_bytes, false));
        const auto initialized = tx->initialize();
        if (!initialized) {
            error = "serving RAW transaction refused at initialize: " + initialized.detail;
            return nullptr;   // nothing was admitted: the caller stays on the exact path
        }
        return std::shared_ptr<ServingRawSession>(new ServingRawSession(std::move(tx)));
    } catch (const std::exception& e) {
        error = std::string("serving RAW owner could not be constructed: ") + e.what();
        return nullptr;
    } catch (...) {
        error = "serving RAW owner could not be constructed";
        return nullptr;
    }
}

std::shared_ptr<ServingRawSession> ServingRawSession::open_owned_slot(const ModelGeometry& geometry,
        NativeSessionAllocation allocation,std::shared_ptr<const void> runtime,
        std::vector<Verifier*> verifiers,std::vector<NativeBatchSlotInspection> inspections,
        WindowBudget& budget,uint64_t max_tokens,ServingRawSession& aggregate_owner,std::string& error) {
    if(!aggregate_owner.tx_||aggregate_owner.slot_occupancy_||!aggregate_owner.healthy()||
       &aggregate_owner.tx_->budget_!=&budget||aggregate_owner.tx_->geometry_identity_!=&geometry||
       runtime.get()!=aggregate_owner.tx_->runtime_.get()||runtime.owner_before(aggregate_owner.tx_->runtime_)||
       aggregate_owner.tx_->runtime_.owner_before(runtime)){
        error="owned slot factory aggregate owner guard: owner="+std::to_string(bool(aggregate_owner.tx_))+
            " slot="+std::to_string(bool(aggregate_owner.slot_occupancy_))+" healthy="+std::to_string(aggregate_owner.healthy())+
            " budget="+std::to_string(aggregate_owner.tx_ && &aggregate_owner.tx_->budget_==&budget)+
            " geometry="+std::to_string(aggregate_owner.tx_ && aggregate_owner.tx_->geometry_identity_==&geometry)+
            " runtime="+std::to_string(aggregate_owner.tx_&&runtime.get()==aggregate_owner.tx_->runtime_.get())+
            " control="+std::to_string(aggregate_owner.tx_&&!runtime.owner_before(aggregate_owner.tx_->runtime_)&&!aggregate_owner.tx_->runtime_.owner_before(runtime));return {};
    }
    const uint64_t host_cap_bytes=aggregate_owner.tx_->host_limit_;
    if(!runtime||!allocation.healthy()||!max_tokens||max_tokens>(SIZE_MAX-65536)/(4*sizeof(int32_t))||
       !host_cap_bytes||verifiers.empty()||verifiers.size()>8||inspections.size()!=verifiers.size()||
       allocation.stages().size()!=verifiers.size()){
        error="owned slot factory capacity/stage bounds: tokens="+std::to_string(max_tokens)+" stages="+
            std::to_string(allocation.stages().size())+" verifiers="+std::to_string(verifiers.size());return {};
    }
    const auto physical=inspections[0].token();
    if(physical.slot>=2||inspections[0].geometry()!=&geometry){error="owned slot factory physical token/geometry mismatch";return {};}
    for(size_t stage=0;stage<verifiers.size();++stage){auto* verifier=verifiers[stage];const auto& native=allocation.stages()[stage];
        if(native.session==nullptr||native.session->max_cells<=0||max_tokens>uint64_t(native.session->max_cells)||
           inspections[stage].geometry()!=&geometry||!verifier||!verifier->owned_batch_setup_||!verifier->owned_batch_setup_->ready()||
           verifier->owned_group_busy_||inspections[stage].token()!=physical||inspections[stage].stage_index()!=stage||
           !inspections[stage].matches_session(native.session,native.device)||
           !verifier->owned_batch_slot_matches(inspections[stage])){
            error="owned slot factory actual binding refused: stage="+std::to_string(stage)+" slot="+std::to_string(physical.slot)+
                " cells="+std::to_string(native.session?native.session->max_cells:0)+" tokens="+std::to_string(max_tokens)+
                " geometry="+std::to_string(inspections[stage].geometry()==&geometry)+
                " ready="+std::to_string(verifier&&verifier->owned_batch_setup_&&verifier->owned_batch_setup_->ready())+
                " busy="+std::to_string(verifier&&verifier->owned_group_busy_)+
                " token="+std::to_string(inspections[stage].token()==physical)+
                " stage_index_match="+std::to_string(inspections[stage].stage_index()==stage)+
                " session="+std::to_string(inspections[stage].matches_session(native.session,native.device))+
                " observed="+std::to_string(verifier&&verifier->owned_batch_slot_matches(inspections[stage]));return {};
        }
        for(size_t slot=0;slot<2;++slot)if(auto existing=verifier->owned_slot_occupancies_[slot].lock()){
            if(existing->active()&&(slot==physical.slot||existing->running_.load(std::memory_order_acquire))){
                error="owned slot factory occupied/running slot="+std::to_string(slot);return {};
            }
        }
    }
    std::shared_ptr<const void> metadata;
    const auto admitted_metadata=aggregate_owner.tx_->admit_controller_host(65536+max_tokens*4*sizeof(int32_t),metadata);
    if(!admitted_metadata){error=admitted_metadata.detail;return {};}
    try{
        // Admit durable input/occupancy metadata before allocating its vector.
        auto retained=[credit=std::move(metadata)](NativeBatchSlotOccupancy* p){delete p;};
        std::shared_ptr<NativeBatchSlotOccupancy> occupancy(new NativeBatchSlotOccupancy,std::move(retained));
        if(!occupancy->incarnation()){error="owned slot factory occupancy incarnation exhausted";return {};}
        occupancy->token_=physical;occupancy->runtime_=runtime;
        auto identity=serving_identity(runtime.get(),allocation);identity.window_generation=occupancy->incarnation();
        const uint64_t payload=host_cap_bytes;
        std::shared_ptr<SuffixRawTransaction> tx(new SuffixRawTransaction(geometry,identity,allocation,
            std::move(runtime),verifiers,budget,max_tokens,payload,host_cap_bytes,false));
        tx->slot_bindings_=std::move(inspections);tx->slot_occupancy_=occupancy;
        const auto initialized=tx->initialize_slot_shared_host(*aggregate_owner.tx_);if(!initialized){error="owned slot factory shared H initialize: "+initialized.detail;return {};}
        auto session=std::shared_ptr<ServingRawSession>(new ServingRawSession(tx));
        session->slot_occupancy_=occupancy;session->slot_current_key_=tx->identity_;session->slot_tokens_.reserve(size_t(max_tokens));
        StateCut empty;empty.id=1;empty.history=window_history_digest(session->slot_tokens_);
        std::vector<StateCut> cuts(verifiers.size(),empty); // before native mutation
        for(auto* verifier:verifiers)verifier->owned_slot_occupancies_[physical.slot]=occupancy;
        const auto begun=tx->begin_driver_work();if(!begun){error=begun.detail;return {};}
        for(const auto& native:allocation.stages()){
            WindowDeviceScope device(native.device);
            if(!device.ok()||!session_zero_checked(*native.session,geometry,nullptr,nullptr,error)||
               cudaDeviceSynchronize()!=cudaSuccess||!device.finish()){
                tx->poison("owned slot reset failed after native mutation");std::terminate();
            }
        }
        if(!tx->complete_driver_work(true,empty,cuts))std::terminate();
        return session;
    }catch(const std::exception& failure){error=failure.what();return {};}
}

bool ServingRawSession::install_owned_slot_image(const BundleHandle& source_image,const WindowKey& expected_source,
        bool edited_prefix,std::string& error) {
    if(alignment_known_||slot_controller_bound_||!slot_occupancy_||!slot_occupancy_->active()||!request_open()||phase()!=Phase::drained||
       !source_image.get()||source_image.get()->key!=expected_source||
       expected_source.model_profile!=tx_->identity_.model_profile||
       source_image.get()->accepted_tokens.size()>tx_->max_tokens_||tx_->cut_.id==UINT64_MAX||
       (!edited_prefix&&(!slot_tokens_.empty()||tx_->cut_.accepted||tx_->cut_.forwarded))||
       (edited_prefix&&(!slot_source_provenance_||
           (slot_current_key_!=tx_->identity_&&(!alignment_donor_||!alignment_edit_origin_||
            alignment_donor_.get()->cut!=tx_->cut_||alignment_donor_.get()->key!=slot_current_key_||
            alignment_known_||tx_->cut_.forwarded%4))))){
        error="owned slot image initial guard: bound="+std::to_string(slot_controller_bound_)+
            " occupancy="+std::to_string(bool(slot_occupancy_)&&slot_occupancy_->active())+
            " open="+std::to_string(request_open())+" phase="+std::to_string(unsigned(phase()))+
            " image="+std::to_string(bool(source_image.get()))+
            " key="+std::to_string(source_image.get()&&source_image.get()->key==expected_source)+
            " model="+std::to_string(expected_source.model_profile==tx_->identity_.model_profile)+
            " source_accepted="+std::to_string(source_image.get()?source_image.get()->accepted_tokens.size():0)+
            " destination_forwarded="+std::to_string(tx_->cut_.forwarded)+" destination_accepted="+std::to_string(tx_->cut_.accepted)+
            " edited="+std::to_string(edited_prefix);return false;
    }
    for(auto* verifier:tx_->verifiers_)for(const auto& weak:verifier->owned_slot_occupancies_)
        if(auto occupancy=weak.lock();occupancy&&(occupancy->running_.load(std::memory_order_acquire)||
           occupancy->submitted_group_.load(std::memory_order_acquire))){
            error="owned slot image blocked by actual running/submitted slot="+std::to_string(occupancy->slot());return false;
        }
    std::vector<NativeWindowStage> config;if(!tx_->resolve(config,error))return false;
    WindowStateLayout layout;
    auto valid=native_window_layout(tx_->geometry_,config,expected_source,source_image.get()->cut,
        tx_->max_tokens_,tx_->max_bytes_,layout,false);
    if(!valid){error=valid.detail;return false;}
    uint64_t bytes=0;if(!tx_->quote(layout,3,bytes)){error="owned slot image transfer quote overflow";return false;}
    std::shared_ptr<const void> credit;
    const auto admitted=tx_->admit_controller_host(bytes,credit);if(!admitted){error=admitted.detail;return false;}
    SavedConversation native;std::vector<NativePleTokens> ple;
    valid=materialize_native_window(*source_image.get(),expected_source,tx_->geometry_,config,
        tx_->max_tokens_,tx_->max_bytes_,false,native,ple);
    if(!valid||ple.size()!=tx_->allocation_.stages().size()){error=valid?"owned slot image PLE stage count differs":valid.detail;return false;}
    for(size_t stage=0;stage<config.size();++stage){const auto& image=stage?native.stage_images[stage-1]:native;
        if(!conversation_snapshot_validate(image,*tx_->allocation_.stages()[stage].session,tx_->geometry_,
            static_cast<const QsaState*>(nullptr),error))return false;
    }
    // A new descriptor will be CAPTURED from the destination. The source bundle
    // keeps its original key; no relabelled image can enter the suffix producer.
    WindowKey destination_key=edited_prefix?edited_key():tx_->identity_;
    StateCut transferred=source_image.get()->cut;transferred.id=tx_->cut_.id+1;
    std::vector<StateCut> cuts(config.size(),transferred);
    if(const auto drained=tx_->poll_all();!drained){error="owned slot image native drain: "+drained.detail;return false;}
    const auto begun=tx_->begin_driver_work();if(!begun){error=begun.detail;return false;}
    lifecycle_.apply(serving_raw::Lifecycle::Event::begin_work);
    slot_occupancy_->running_.store(true,std::memory_order_release);
    // Reset every actual destination stage before any copy. A shorter prefix
    // must not inherit omitted INT8 page/pool/tail rows from the donor import.
    for(const auto& destination:tx_->allocation_.stages()){
        WindowDeviceScope device(destination.device);
        if(!device.ok()||!session_zero_checked(*destination.session,tx_->geometry_,nullptr,nullptr,error)||
           cudaDeviceSynchronize()!=cudaSuccess||!device.finish()){
            tx_->poison("owned slot image reset failed");std::terminate();
        }
    }
    for(size_t stage=0;stage<config.size();++stage){
        const auto& destination=tx_->allocation_.stages()[stage];WindowDeviceScope device(destination.device);
        const auto& image=stage?native.stage_images[stage-1]:native;
        if(!device.ok()||conversation_snapshot_restore(image,*destination.session,tx_->geometry_,
            static_cast<const QsaState*>(nullptr),error)!=ConversationRestore::restored){
            tx_->poison("owned slot image transfer failed after native write attempt");std::terminate();
        }
        destination.session->ple_prev[0]=ple[stage].previous[0];destination.session->ple_prev[1]=ple[stage].previous[1];
        destination.session->ple_token=ple[stage].current;
        if(cudaDeviceSynchronize()!=cudaSuccess||!device.finish()){
            tx_->poison("owned slot image transfer drain failed");std::terminate();
        }
    }
    if(!tx_->complete_driver_work(true,transferred,cuts))std::terminate();
    slot_tokens_.assign(source_image.get()->accepted_tokens.begin(),source_image.get()->accepted_tokens.end());
    slot_current_key_=std::move(destination_key);
    if(edited_prefix)slot_donor_provenance_=slot_source_provenance_;
    slot_source_provenance_=source_image;slot_occupancy_->running_.store(false,std::memory_order_release);
    lifecycle_.set_observed_phase(raw_phase(phase()));return true;
}

bool ServingRawSession::transfer_owned_slot_from(ServingRawSession& source,
        const SnapshotDescriptor& source_metadata,std::string& error) {
    if(&source==this||source.slot_occupancy_||!source.request_open()||source.phase()!=Phase::drained||
       !source.tx_||source.tx_->geometry_identity_!=tx_->geometry_identity_||source.tx_->runtime_!=tx_->runtime_||source.tx_->runtime_.owner_before(tx_->runtime_)||tx_->runtime_.owner_before(source.tx_->runtime_)){
        error="owned slot transfer source guard: self="+std::to_string(&source==this)+
            " source_slot="+std::to_string(bool(source.slot_occupancy_))+" source_open="+std::to_string(source.request_open())+
            " source_phase="+std::to_string(unsigned(source.phase()))+
            " geometry="+std::to_string(source.tx_ && source.tx_->geometry_identity_==tx_->geometry_identity_)+
            " runtime="+std::to_string(source.tx_&&source.tx_->runtime_==tx_->runtime_);return false;
    }
    BundleHandle source_image;const auto captured=source.capture(source_metadata,source_image);
    if(!captured){error=captured.detail;return false;}
    return install_owned_slot_image(source_image,source_metadata.key,false,error);
}

SuffixRawResult ServingRawSession::capture_owned_slot(BundleHandle& out) {
    if(alignment_known_||slot_controller_bound_)return invalid("slot capture requires no pending preparation and its sole metadata authority");
    if(!slot_occupancy_||!slot_occupancy_->active())return invalid("no live owned slot occupancy");
    const auto metadata=endpoint_metadata(slot_current_key_,tx_->cut_,slot_tokens_);
    // Freeze edit-origin metadata before capture/native mutation. This is an actual
    // progressed donor endpoint, never a relabelled original source image.
    std::unique_ptr<WindowKey> edit_origin;
    // Only genuine unedited known-progress may establish an origin. Capturing
    // the subsequently installed edited prefix must NEVER advance that origin.
    const bool alignment_candidate=!alignment_edit_origin_&&!slot_donor_provenance_&&
        alignment_owner_credit_&&slot_source_provenance_&&tx_->identity_matches(slot_current_key_)&&
        slot_current_key_.revision==tx_->identity_.revision&&
        slot_current_key_.selection_epoch==tx_->identity_.selection_epoch&&
        slot_current_key_.provenance_epoch==tx_->identity_.provenance_epoch&&
        slot_current_key_.append_epoch>tx_->identity_.append_epoch;
    if(alignment_candidate) {
        const uint64_t source_f=slot_source_provenance_.get()->cut.forwarded;
        if(source_f>UINT64_MAX-3||source_f%4==0||tx_->cut_.forwarded!=((source_f+3)/4)*4||
           tx_->cut_.accepted!=tx_->cut_.forwarded+1||tx_->cut_.exact_frontier!=tx_->cut_.forwarded||
           tx_->cut_.output_watermark!=slot_source_provenance_.get()->cut.output_watermark||
           slot_current_key_.append_epoch-tx_->identity_.append_epoch!=tx_->cut_.forwarded-source_f)
            return invalid("alignment donor not at authentic completed known-input endpoint");
        try{edit_origin=std::make_unique<WindowKey>(slot_current_key_);}catch(...){return invalid("alignment key allocation refused");}
    }
    auto result=capture(metadata,out);
    if(result&&edit_origin){alignment_edit_origin_=std::move(edit_origin);alignment_donor_=out;}
    return result;
}

const WindowKey* ServingRawSession::suffix_donor_key(const BundleHandle& donor) const noexcept {
    if(!tx_||!donor)return nullptr;
    if(donor.get()->key==tx_->identity_)return &tx_->identity_; // unchanged original serial/aligned domain
    if(!slot_occupancy_||!slot_occupancy_->active()||slot_controller_bound_||alignment_known_||
       !alignment_donor_||!alignment_edit_origin_||donor.get()!=alignment_donor_.get()||
       donor.get()->key!=*alignment_edit_origin_||donor.get()->cut.forwarded%4||
       donor.get()->cut.exact_frontier!=donor.get()->cut.forwarded||
       donor.get()->cut.accepted!=donor.get()->cut.forwarded+1)return nullptr;
    return &alignment_donor_.get()->key;
}

bool ServingRawSession::prepare_owned_alignment_known(
        std::shared_ptr<const std::vector<int64_t>> request,
        std::shared_ptr<const void> request_credit,std::string& error) {
    if(alignment_known_||slot_controller_bound_||!healthy()||!request_open()||phase()!=Phase::drained||
       !slot_occupancy_||!slot_source_provenance_||!request||!request_credit||
       !slot_source_provenance_.get()->cut.forwarded||slot_source_provenance_.get()->cut.forwarded>UINT64_MAX-3||
       cut().forwarded>=((slot_source_provenance_.get()->cut.forwarded+3)/4)*4||
       cut().forwarded<slot_source_provenance_.get()->cut.forwarded||
       cut().accepted!=cut().forwarded+1||cut().exact_frontier!=cut().forwarded||
       slot_tokens_.size()!=cut().accepted||cut().id==UINT64_MAX||
       slot_current_key_.append_epoch==UINT64_MAX||cut().accepted>=max_tokens()||
       cut().accepted>=request->size()||request->at(size_t(cut().forwarded))!=slot_tokens_.back()||
       request->at(size_t(cut().accepted))<0||request->at(size_t(cut().accepted))>=tx_->geometry_.n_vocab){
        error="unaligned admission known preparation requires actual exact pending and owned next input";return false;
    }
    try {
        // SnapshotDescriptor is metadata-only here; no captured device payload is cloned.
        uint64_t bytes=0;
        if(max_tokens()>((UINT64_MAX-(1u<<20))/16)){error="alignment metadata quote overflow";return false;}
        bytes=max_tokens()*16+(1u<<20);
        std::shared_ptr<const void> credit;
        const auto admitted=admit_host(bytes,credit);if(!admitted){error=admitted.detail;return false;}
        auto ticket=std::make_unique<AlignmentKnown>();ticket->credit=std::move(credit);
        ticket->request_credit=std::move(request_credit);ticket->request=std::move(request);
        ticket->before=endpoint_metadata(slot_current_key_,cut(),slot_tokens_);
        auto tokens=slot_tokens_;tokens.push_back(int32_t(ticket->request->at(size_t(cut().accepted))));
        StateCut next=cut();++next.id;++next.forwarded;++next.accepted;next.exact_frontier=next.forwarded;
        next.history=window_history_digest(tokens);
        auto key=slot_current_key_;++key.append_epoch;
        ticket->next=endpoint_metadata(key,next,tokens);
        ticket->stages.assign(tx_->allocation_.stages().size(),next);
        if(!alignment_owner_credit_)alignment_owner_credit_=ticket->credit;
        alignment_known_=std::move(ticket);return true;
    }catch(...){error="alignment known metadata admission allocation failed";return false;}
}
bool ServingRawSession::cancel_owned_alignment_known(std::string& error) {
    if(!alignment_known_)return true;
    if(phase()!=Phase::drained||cut()!=alignment_known_->before.cut||
       slot_occupancy_->submitted_group_.load(std::memory_order_acquire)){
        error="alignment known already submitted; ownership must be retained";return false;
    }
    alignment_known_.reset();return true;
}
bool ServingRawSession::complete_owned_alignment_known(const NativeBatchGroupCompletion& receipt,std::string& error) {
    if(!alignment_known_||slot_controller_bound_)return false;
    auto& ticket=*alignment_known_;CohortReceiptView view;
    if(!inspect_cohort_receipt(receipt,ticket.before,ticket.before.pending_tokens.front(),view,error))return false;
    const auto& next=ticket.next;
    if(next.accepted_tokens.size()>slot_tokens_.capacity()||next.cut.output_watermark!=cut().output_watermark||
       next.pending_tokens.front()!=ticket.request->at(size_t(ticket.before.cut.accepted)))return false;
    bool unused=false;
    if(!receipt.consumed_[view.slot].compare_exchange_strong(unused,true,std::memory_order_acq_rel))return false;
    // All allocations/history preparation precede begin_work. No BT/output publication exists here.
    if(!tx_->complete_driver_work(true,next.cut,ticket.stages))std::terminate();
    slot_tokens_.assign(next.accepted_tokens.begin(),next.accepted_tokens.end());
    slot_current_key_.append_epoch=next.key.append_epoch;
    slot_last_group_=receipt.operation_;slot_occupancy_->running_.store(false,std::memory_order_release);
    slot_occupancy_->submitted_group_.store(0,std::memory_order_release);
    lifecycle_.set_observed_phase(raw_phase(phase()));alignment_known_.reset();return true;
}

bool ServingRawSession::queue_owned_slot_input(int32_t token,std::string& error) {
    if(alignment_known_||!slot_occupancy_||!slot_occupancy_->active()||!request_open()||phase()!=Phase::drained||
       token<0||slot_tokens_.size()!=tx_->cut_.forwarded||slot_tokens_.size()>=tx_->max_tokens_||
       tx_->verifiers_.empty()||int64_t(token)>=tx_->verifiers_.back()->n_vocab_){
        error="owned slot input needs its drained open request without a pending token";return false;
    }
    // Capacity was admitted and reserved before the factory's native reset.
    slot_tokens_.push_back(token);return true;
}

bool ServingRawSession::drain_owned_slot_group(const NativeBatchGroupCompletion& receipt,std::FILE* output,std::string& error) {
    if(alignment_known_||slot_controller_bound_){error="prepared slot requires its sole metadata producer";return false;}
    const bool accept_sample=output!=nullptr; // actual output call below is the authority
    if(!slot_occupancy_||!slot_occupancy_->active()||!request_open()||phase()!=Phase::running||
       !slot_occupancy_->running_.load(std::memory_order_acquire)||!receipt.operation_||
       receipt.operation_<=slot_last_group_)return false;
    const size_t slot=slot_occupancy_->slot();
    if(slot_occupancy_->submitted_group_.load(std::memory_order_acquire)!=receipt.operation_||
       receipt.occupancies_[slot]!=slot_occupancy_||receipt.stages_!=tx_->verifiers_.size()||receipt.accepted_[slot]!=1)return false;
    size_t selected=receipt.rows_;
    for(size_t row=0;row<receipt.rows_;++row)if(receipt.row_slots_[row]==int(slot)){
        if(selected!=receipt.rows_)return false;
        selected=row;
    }
    if(selected==receipt.rows_||slot_tokens_.size()!=tx_->cut_.forwarded+1||
       receipt.positions_[selected]!=int64_t(tx_->cut_.forwarded)||receipt.tokens_[selected]!=slot_tokens_.back()||
       tx_->cut_.id==UINT64_MAX||
       (accept_sample&&(slot_tokens_.size()>=tx_->max_tokens_||tx_->cut_.output_watermark==UINT64_MAX)))return false;
    const bool terminal_output=receipt.nonfinite_[selected]||receipt.outputs_[selected]<0||
        int64_t(receipt.outputs_[selected])>=tx_->verifiers_.back()->n_vocab_;
    if(accept_sample&&terminal_output)return false;
    for(size_t stage=0;stage<receipt.stages_;++stage)
        if(!tx_->verifiers_[stage]->owned_batch_slot_matches(receipt.bindings_[stage*2+slot])||
           !receipt.bindings_[stage*2+slot].matches_session(tx_->allocation_.stages()[stage].session,
                                                          tx_->allocation_.stages()[stage].device))return false;
    StateCut cut=tx_->cut_;++cut.id;++cut.forwarded;
    if(accept_sample)++cut.output_watermark;
    std::vector<StateCut> stages;
    // Hash/container allocation can refuse without consuming the receipt or
    // publishing host metadata. No borrow escapes this serialized reactor call.
    if(accept_sample)slot_tokens_.push_back(receipt.outputs_[selected]);
    try{
        cut.accepted=slot_tokens_.size();cut.history=window_history_digest(slot_tokens_);
        if(cut.exact_frontier==tx_->cut_.forwarded)cut.exact_frontier=cut.forwarded;
        stages.assign(receipt.stages_,cut);
    }catch(...){if(accept_sample)slot_tokens_.pop_back();error="slot completion metadata allocation refused";return false;}
    if(accept_sample)slot_tokens_.pop_back();
    const auto preflight_drained=tx_->poll_all();
    if(!preflight_drained){error=preflight_drained.detail;return false;}
    bool unused=false;
    if(!receipt.consumed_[slot].compare_exchange_strong(unused,true,std::memory_order_acq_rel))return false;
    if(accept_sample){
        if(std::fprintf(output,"BT %zu %d\n",slot,int(receipt.outputs_[selected]))<0||std::fflush(output)!=0){
            tx_->poison("owned slot output publication failed");std::terminate();
        }
        slot_tokens_.push_back(receipt.outputs_[selected]); // pre-reserved capacity
    }
    const auto result=tx_->complete_driver_work(true,cut,stages);
    if(!result){error=result.detail;if(accept_sample)std::terminate();return false;}
    slot_last_group_=receipt.operation_;slot_occupancy_->running_.store(false,std::memory_order_release);
    slot_occupancy_->submitted_group_.store(0,std::memory_order_release);
    lifecycle_.set_observed_phase(raw_phase(phase()));lifecycle_.apply(serving_raw::Lifecycle::Event::drain_work);
    if(terminal_output){
        lifecycle_.apply(serving_raw::Lifecycle::Event::abandon);
        if(!retire_owned_slot(error))std::terminate();
    }
    return true;
}

bool ServingRawSession::inspect_cohort_receipt(const NativeBatchGroupCompletion& receipt,
        const SnapshotDescriptor& base,int32_t input,CohortReceiptView& view,std::string& error) {
    if(view.receipt||!tx_||!slot_occupancy_||!slot_occupancy_->active()||!request_open()||
       phase()!=Phase::running||!slot_occupancy_->running_.load(std::memory_order_acquire)||
       base.cut!=tx_->cut_||base.accepted_tokens!=slot_tokens_||base.pending_tokens.size()!=1||
       slot_tokens_.size()!=base.cut.forwarded+1||slot_tokens_.back()!=input||base.pending_tokens.front()!=input||
       !tx_->identity_matches(base.key)||receipt.operation_<=slot_last_group_||
       slot_occupancy_->submitted_group_.load(std::memory_order_acquire)!=receipt.operation_)return false;
    const size_t slot=slot_occupancy_->slot();
    if(receipt.occupancies_[slot]!=slot_occupancy_||receipt.stages_!=tx_->verifiers_.size()||receipt.accepted_[slot]!=1)return false;
    size_t selected=receipt.rows_;
    for(size_t row=0;row<receipt.rows_;++row)if(receipt.row_slots_[row]==int(slot)){
        if(selected!=receipt.rows_)return false;
        selected=row;
    }
    if(selected==receipt.rows_||receipt.positions_[selected]!=int64_t(base.cut.forwarded)||
       receipt.tokens_[selected]!=input||receipt.nonfinite_[selected]||receipt.outputs_[selected]<0||
       int64_t(receipt.outputs_[selected])>=tx_->verifiers_.back()->n_vocab_||
       receipt.consumed_[slot].load(std::memory_order_acquire))return false;
    for(size_t stage=0;stage<receipt.stages_;++stage)
        if(!tx_->verifiers_[stage]->owned_batch_slot_matches(receipt.bindings_[stage*2+slot])||
           !receipt.bindings_[stage*2+slot].matches_session(tx_->allocation_.stages()[stage].session,
                                                          tx_->allocation_.stages()[stage].device))return false;
    const auto drained=tx_->poll_all();if(!drained){error=drained.detail;return false;}
    view.receipt=&receipt;view.base=&base;view.occupancy=slot_occupancy_;view.row=selected;view.slot=slot;
    view.sample=receipt.outputs_[selected];view.group_operation=receipt.operation_;return true;
}

bool ServingRawSession::commit_cohort_receipt(CohortReceiptView& view,const SnapshotDescriptor& next,
        std::span<const StateCut> stages,std::FILE* output,std::string& error) {
    if(!output||view.consumed||!view.receipt||!view.base||view.occupancy!=slot_occupancy_||
       !slot_occupancy_||!slot_occupancy_->active()||phase()!=Phase::running||!request_open())return false;
    const auto& base=*view.base;const auto& receipt=*view.receipt;const auto& old=base.cut;const auto& cut=next.cut;
    if(old!=tx_->cut_||base.accepted_tokens!=slot_tokens_||view.group_operation!=receipt.operation_||
       slot_occupancy_->submitted_group_.load(std::memory_order_acquire)!=receipt.operation_||
       view.sample!=receipt.outputs_[view.row]||view.slot!=slot_occupancy_->slot()||
       receipt.occupancies_[view.slot]!=slot_occupancy_||old.accepted==UINT64_MAX||old.forwarded==UINT64_MAX||
       old.output_watermark==UINT64_MAX||base.key.append_epoch==UINT64_MAX||cut.id<=old.id||
       cut.forwarded!=old.forwarded+1||cut.accepted!=old.accepted+1||cut.accepted>tx_->max_tokens_||
       cut.output_watermark!=old.output_watermark+1||
       cut.exact_frontier!=(old.exact_frontier==old.forwarded?cut.forwarded:old.exact_frontier)||
       next.accepted_tokens.size()!=slot_tokens_.size()+1||next.pending_tokens.size()!=1||
       next.pending_tokens.front()!=view.sample||next.accepted_tokens.back()!=view.sample||
       !std::equal(slot_tokens_.begin(),slot_tokens_.end(),next.accepted_tokens.begin())||
       next.key.engine_generation!=base.key.engine_generation||next.key.window_generation!=base.key.window_generation||
       next.key.session_id!=base.key.session_id||next.key.thread_id!=base.key.thread_id||next.key.window_id!=base.key.window_id||
       next.key.revision!=base.key.revision||next.key.append_epoch!=base.key.append_epoch+1||
       next.key.selection_epoch!=base.key.selection_epoch||next.key.provenance_epoch!=base.key.provenance_epoch||
       next.key.model_profile!=base.key.model_profile||next.key.topology!=base.key.topology||
       !tx_->identity_matches(next.key)||stages.size()!=tx_->allocation_.stages().size()||
       !std::all_of(stages.begin(),stages.end(),[&](const auto& stage){return stage==cut;})||
       !present(cut.history))return false;
    // Controller is the sole descriptor/hash/ancestry/cut producer. All native
    // and fallible preparation precede this irreversible, real output event.
    bool unused=false;
    if(!receipt.consumed_[view.slot].compare_exchange_strong(unused,true,std::memory_order_acq_rel))return false;
    view.consumed=true;
    if(std::fprintf(output,"BT %zu %d\n",view.slot,int(view.sample))<0||std::fflush(output)!=0){
        tx_->poison("controller cohort output publication failed");std::terminate();
    }
    const auto completed=tx_->complete_driver_work(true,cut,stages);
    if(!completed){error=completed.detail;std::terminate();}
    slot_tokens_.assign(next.accepted_tokens.begin(),next.accepted_tokens.end()); // admitted capacity
    slot_current_key_.revision=next.key.revision;slot_current_key_.append_epoch=next.key.append_epoch;
    slot_current_key_.selection_epoch=next.key.selection_epoch;slot_current_key_.provenance_epoch=next.key.provenance_epoch;
    slot_last_group_=receipt.operation_;slot_occupancy_->running_.store(false,std::memory_order_release);
    slot_occupancy_->submitted_group_.store(0,std::memory_order_release);
    lifecycle_.set_observed_phase(raw_phase(phase()));return true;
}

bool ServingRawSession::commit_cohort_known_receipt(CohortReceiptView& view,const SnapshotDescriptor& next,
        std::span<const StateCut> stages,std::string& error) {
    if(view.consumed||!view.receipt||!view.base||view.occupancy!=slot_occupancy_||!slot_controller_bound_||
       !slot_occupancy_||!slot_occupancy_->active()||phase()!=Phase::running||!request_open())return false;
    const auto& base=*view.base;const auto& receipt=*view.receipt;const auto& old=base.cut;const auto& cut=next.cut;
    if(view.row>=receipt.rows_||view.slot>=2||old!=tx_->cut_||base.accepted_tokens!=slot_tokens_||
       view.group_operation!=receipt.operation_||slot_occupancy_->submitted_group_.load(std::memory_order_acquire)!=receipt.operation_||
       view.sample!=receipt.outputs_[view.row]||view.slot!=slot_occupancy_->slot()||receipt.nonfinite_[view.row]||
       receipt.occupancies_[view.slot]!=slot_occupancy_||old.accepted==UINT64_MAX||old.forwarded==UINT64_MAX||
       base.key.append_epoch==UINT64_MAX||cut.id<=old.id||cut.forwarded!=old.forwarded+1||
       cut.accepted!=old.accepted+1||cut.accepted>tx_->max_tokens_||cut.output_watermark!=old.output_watermark||
       cut.exact_frontier!=(old.exact_frontier==old.forwarded?cut.forwarded:old.exact_frontier)||
       next.accepted_tokens.size()!=slot_tokens_.size()+1||next.accepted_tokens.size()>slot_tokens_.capacity()||
       next.pending_tokens.size()!=1||next.pending_tokens.front()<0||
       int64_t(next.pending_tokens.front())>=tx_->geometry_.n_vocab||
       next.accepted_tokens.back()!=next.pending_tokens.front()||
       !std::equal(slot_tokens_.begin(),slot_tokens_.end(),next.accepted_tokens.begin())||
       next.key.engine_generation!=base.key.engine_generation||next.key.window_generation!=base.key.window_generation||
       next.key.session_id!=base.key.session_id||next.key.thread_id!=base.key.thread_id||next.key.window_id!=base.key.window_id||
       next.key.revision!=base.key.revision||next.key.append_epoch!=base.key.append_epoch+1||
       next.key.selection_epoch!=base.key.selection_epoch||next.key.provenance_epoch!=base.key.provenance_epoch||
       next.key.model_profile!=base.key.model_profile||next.key.topology!=base.key.topology||
       !tx_->identity_matches(next.key)||!tx_->identity_matches(slot_current_key_)||
       stages.size()!=tx_->allocation_.stages().size()||
       !std::all_of(stages.begin(),stages.end(),[&](const auto& stage){return stage==cut;})||!present(cut.history))return false;
    // A private controller supplies its pre-admitted immutable known token and precomputed history. Native
    // completion proves the consumed row only: the model pick is discarded, no output receipt or BT is minted.
    bool unused=false;
    if(!receipt.consumed_[view.slot].compare_exchange_strong(unused,true,std::memory_order_acq_rel))return false;
    view.consumed=true;
    const auto completed=tx_->complete_driver_work(true,cut,stages);
    if(!completed){error=completed.detail;return false;}
    slot_tokens_.assign(next.accepted_tokens.begin(),next.accepted_tokens.end()); // pre-admitted capacity
    // Stable identity strings/profile/topology are unchanged. Adopt controller epochs without an allocating key copy.
    slot_current_key_.revision=next.key.revision;slot_current_key_.append_epoch=next.key.append_epoch;
    slot_current_key_.selection_epoch=next.key.selection_epoch;slot_current_key_.provenance_epoch=next.key.provenance_epoch;
    slot_last_group_=receipt.operation_;slot_occupancy_->running_.store(false,std::memory_order_release);
    slot_occupancy_->submitted_group_.store(0,std::memory_order_release);
    lifecycle_.set_observed_phase(raw_phase(phase()));return true;
}

bool ServingRawSession::retire_owned_slot(std::string& error) {
    if(alignment_known_||!slot_occupancy_||!slot_occupancy_->active()||request_open()||phase()!=Phase::drained||
       slot_occupancy_->submitted_group_.load(std::memory_order_acquire))return false;
    const auto drained=tx_->poll_all();if(!drained){error=drained.detail;return false;}
    slot_occupancy_->active_.store(false,std::memory_order_release);
    for(auto* verifier:tx_->verifiers_)if(verifier->owned_slot_occupancies_[slot_occupancy_->slot()].lock()==slot_occupancy_)
        verifier->owned_slot_occupancies_[slot_occupancy_->slot()].reset();
    return true;
}

ServingRawSession::~ServingRawSession() {
    // The transaction's own destructor is the established failstop-on-uncertain-work boundary
    // (suffix_raw_runtime.cpp:96-98). A serving process must not unwind past it: callers failstop when
    // healthy() is false, so no destructor here ever runs over live consumers.
}

bool ServingRawSession::healthy() const noexcept {
    return tx_ != nullptr && !terminal() && tx_->initialized_ && tx_->allocation_.healthy()&&
        (!slot_occupancy_||slot_occupancy_->active());
}

ServingRawSession::Phase ServingRawSession::phase() const noexcept {
    if (!tx_) return Phase::absent;
    switch (tx_->phase_) {
        case SuffixRawTransaction::Phase::idle: return Phase::idle;
        case SuffixRawTransaction::Phase::running: return Phase::running;
        case SuffixRawTransaction::Phase::drained: return Phase::drained;
        case SuffixRawTransaction::Phase::capturing: return Phase::capturing;
        case SuffixRawTransaction::Phase::restoring: return Phase::restoring;
        case SuffixRawTransaction::Phase::poisoned: return Phase::poisoned;
    }
    return Phase::absent;
}

bool ServingRawSession::cut_acceptable(const StateCut& cut, size_t stage_count, uint64_t max_tokens) noexcept {
    return serving_raw::cut_facts_acceptable(serving_raw::CutFacts{cut.id, cut.accepted, cut.forwarded,
                                                                   cut.exact_frontier, &cut.history},
                                             stage_count, max_tokens);
}

bool ServingRawSession::open_request(std::string& error) {
    if (!tx_) { error = "no serving RAW transaction"; return false; }
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.allow(serving_raw::Lifecycle::Event::open_request)) {
        error = lifecycle_.request_open() ? "a serving request already holds this RAW session"
                                          : "cannot open a request over a poisoned RAW session";
        return false;
    }
    lifecycle_.apply(serving_raw::Lifecycle::Event::open_request);
    return true;
}

bool ServingRawSession::begin_work(std::string& error) {
    if(slot_occupancy_&&(!slot_occupancy_->active()||!tx_||slot_tokens_.size()!=tx_->cut_.forwarded+1)){
        error="owned slot work requires its own actual pending input";return false;
    }
    if (!tx_) { error = "no serving RAW transaction"; return false; }
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.allow(serving_raw::Lifecycle::Event::begin_work)) {
        error = "begin_work requires an open request at an idle or drained cut";
        return false;
    }
    const auto begun = tx_->begin_driver_work();
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!begun) { error = begun.detail; return false; }
    lifecycle_.apply(serving_raw::Lifecycle::Event::begin_work);
    if(slot_occupancy_){
        slot_occupancy_->pending_token_=slot_tokens_.back();slot_occupancy_->pending_position_=int64_t(tx_->cut_.forwarded);
        slot_occupancy_->running_.store(true,std::memory_order_release);
    }
    return true;
}

bool ServingRawSession::drain_work(bool operation_succeeded, const StateCut& cut, std::string& error) {
    if(slot_occupancy_){error="owned slot completion requires its actual group receipt";return false;}
    if (!tx_) { error = "no serving RAW transaction"; return false; }
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.allow(serving_raw::Lifecycle::Event::drain_work)) {
        error = "drain_work requires an open request with the transaction running";
        return false;
    }
    const size_t stages = tx_->allocation_.stages().size();
    // Refuse before touching the transaction: passing an invalid cut would POISON it
    // (suffix_raw_runtime.cpp:233-235), and a poisoned transaction terminates the process on destruction (:96-98).
    if (!operation_succeeded || !cut_acceptable(cut, stages, tx_->max_tokens_)) {
        error = "serving completion cut is not acceptable (id/history/order/stage count)";
        return false;
    }
    // One cut PER STAGE, all equal to the common cut, built from the real allocation: an empty span is
    // unrepresentable. The actual device drain is the transaction's own poll_all(). The request STAYS OPEN:
    // the drained phase this produces is exactly where capture/restore are legal.
    const std::vector<StateCut> stage_cuts(stages, cut);
    const auto completed = tx_->complete_driver_work(true, cut, stage_cuts);
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!completed) { error = completed.detail; return false; }
    lifecycle_.apply(serving_raw::Lifecycle::Event::drain_work);
    return true;
}

bool ServingRawSession::close_request(std::string& error) {
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.allow(serving_raw::Lifecycle::Event::close_request)) {
        error = "cannot close a request while RAW work is running: drain it, or abandon deliberately";
        return false;
    }
    lifecycle_.apply(serving_raw::Lifecycle::Event::close_request);
    return true;
}

void ServingRawSession::abandon() noexcept {
    // The transaction is uncertain (or the fence was never closed): destroying this owner is the failstop,
    // because ~SuffixRawTransaction terminates on running/capturing/restoring/poisoned phases.
    lifecycle_.apply(serving_raw::Lifecycle::Event::abandon);
    // Durability (2026-10-29 review): the poison has to live in the TRANSACTION too, not only in the adapter's
    // state machine. healthy()/phase() read tx_, and every public call re-syncs the observed phase from it, so an
    // adapter-only poison would be overwritten on the next call and the session would look usable again.
    // `poison()` is the RAW contract's own terminal transition: it marks the allocation unhealthy and latches
    // phase_ = poisoned, which is what makes the destructor's terminate real rather than conditional.
    if (tx_) tx_->poison("serving request abandoned over an unclosed RAW fence");
    lifecycle_.set_observed_phase(serving_raw::Phase::poisoned);
}

bool ServingRawSession::terminal() const noexcept {
    // Either the adapter gave up, or the transaction itself reported the terminal phase: both are durable, and
    // `healthy()` below cannot mask them.
    return lifecycle_.terminal() || tx_ == nullptr || tx_->phase_ == SuffixRawTransaction::Phase::poisoned;
}

SuffixRawResult ServingRawSession::capture(const SnapshotDescriptor& descriptor, BundleHandle& out) {
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.allow(serving_raw::Lifecycle::Event::capture))
        return {SuffixRawStatus::invalid,
                "capture requires an open request at a DRAINED all-stage cut (begin_work then drain_work first)",
                {}};
    if(slot_occupancy_&&descriptor.accepted_tokens!=slot_tokens_)
        return {SuffixRawStatus::invalid,"slot capture history differs from its owned accepted sequence",{}};
    return tx_->capture(descriptor, out);
}

SuffixRawResult ServingRawSession::restore(const BundleHandle& bundle, const WindowKey& key,
                                           ResidentCopyFault* fault) {
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.allow(serving_raw::Lifecycle::Event::restore))
        return {SuffixRawStatus::invalid,
                "restore requires an open request at a DRAINED all-stage cut", {}};
    if(slot_occupancy_){
        if(!slot_occupancy_->active()||!bundle.get()||bundle.get()->accepted_tokens.size()>tx_->max_tokens_)
            return {SuffixRawStatus::invalid,"owned slot restore sequence is unavailable or unbounded",{}};
        for(auto* verifier:tx_->verifiers_)for(const auto& weak:verifier->owned_slot_occupancies_)
            if(auto occupancy=weak.lock();occupancy&&occupancy->running_.load(std::memory_order_acquire))
                return {SuffixRawStatus::invalid,"owned slot restore needs a group boundary",{}};
    }
    auto result=tx_->restore(bundle,key,fault);
    if(result&&slot_occupancy_){
        slot_tokens_.assign(bundle.get()->accepted_tokens.begin(),bundle.get()->accepted_tokens.end());
        slot_current_key_.revision=bundle.get()->key.revision;slot_current_key_.append_epoch=bundle.get()->key.append_epoch;
        slot_current_key_.selection_epoch=bundle.get()->key.selection_epoch;slot_current_key_.provenance_epoch=bundle.get()->key.provenance_epoch;
    }
    return result;
}

// ---- bounded suffix REPAIR operands ---------------------------------------------------------------------
// The rules below are the transaction's own; this layer adds nothing except (a) the request-lifecycle gate, so a
// serving request cannot reach a RAW operand outside an open fence, and (b) trusted layout/identity, taken from
// the binding the owner was constructed with rather than from anything the caller can supply.

core::SnapshotDescriptor ServingRawSession::endpoint_metadata(const WindowKey& key, const StateCut& cut,
                                                              const std::vector<int32_t>& tokens) {
    core::SnapshotDescriptor value;
    value.key = key;
    value.cut = cut;
    value.accepted_tokens = tokens;
    if (cut.forwarded <= tokens.size())
        value.pending_tokens.assign(tokens.begin() + static_cast<long long>(cut.forwarded), tokens.end());
    value.steering_enabled = false;
    value.speculation_disabled = true;
    return value;
}

core::WindowKey ServingRawSession::edited_key() const noexcept {
    auto key = alignment_edit_origin_?*alignment_edit_origin_:tx_->identity_;
    ++key.revision;
    ++key.append_epoch;
    ++key.selection_epoch;
    return key;   // donor identity stays distinct: an edited run is never a second name for the same selection
}

SuffixRawResult ServingRawSession::admit_host(uint64_t bytes, std::shared_ptr<const void>& credit) {
    if (!tx_) return {SuffixRawStatus::invalid, "no serving RAW transaction", {}};
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.request_open())
        return {SuffixRawStatus::invalid, "host admission needs an open serving request", {}};
    if (bytes == 0) return {SuffixRawStatus::invalid, "zero-size host admission", {}};
    return tx_->admit_controller_host(bytes, credit);
}

static SuffixRawResult map_suffix_refusal(const core::NativeSuffixResult& result, const char* what) {
    if (result) return {SuffixRawStatus::ok, {}, {}};
    const auto status = result.status == core::NativeSuffixStatus::budget_refused ? SuffixRawStatus::budget_refused
                                                                                 : SuffixRawStatus::invalid;
    return {status,
            std::string(what) + " refused (status=" + std::to_string(unsigned(result.status)) +
            " reason=" + std::to_string(unsigned(result.reason)) + ")",
            {}};
}

SuffixRawResult ServingRawSession::quote_suffix(const BundleHandle& donor, const WindowKey& donor_key,
                                                const BundleHandle& prefix, const WindowKey& prefix_key,
                                                const core::NativeSuffixTarget& target,
                                                const core::NativeSuffixLimits& limits, uint64_t& out_host_bytes) {
    if (!tx_) return {SuffixRawStatus::invalid, "no serving RAW transaction", {}};
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.allow(serving_raw::Lifecycle::Event::restore))
        return {SuffixRawStatus::invalid, "suffix quote requires an open request at a DRAINED all-stage cut", {}};
    if (!donor || !prefix) return {SuffixRawStatus::invalid, "suffix quote needs both operand bundles", {}};
    if (target.cut_id == 0 || target.cut_id == std::numeric_limits<uint64_t>::max() ||
        target.key.revision == std::numeric_limits<uint64_t>::max())
        return {SuffixRawStatus::invalid, "suffix quote target identity is exhausted or zero", {}};
    // The layout the quote may use is the transaction's own initial configuration; a caller cannot present a
    // friendlier one.
    const auto result = core::quote_int8_native_suffix(donor, donor_key, prefix, prefix_key, tx_->geometry_,
                                                       tx_->initial_config_, target, limits, out_host_bytes);
    return map_suffix_refusal(result, "native suffix quote");
}

SuffixRawResult ServingRawSession::claim_splice(uint64_t bytes, std::shared_ptr<const void>& credit) {
    if (!tx_) return {SuffixRawStatus::invalid, "no serving RAW transaction", {}};
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.allow(serving_raw::Lifecycle::Event::restore))
        return {SuffixRawStatus::invalid, "splice claim requires an open request at a DRAINED all-stage cut", {}};
    if (bytes == 0) return {SuffixRawStatus::invalid, "zero-size splice claim", {}};
    return tx_->claim_external_splice(bytes, credit);
}

SuffixRawResult ServingRawSession::prepare_suffix(const BundleHandle& donor, const WindowKey& donor_key,
                                                  const BundleHandle& prefix, const WindowKey& prefix_key,
                                                  const core::NativeSuffixTarget& target,
                                                  const core::NativeSuffixLimits& limits,
                                                  core::PreparedNativeSuffix& out) {
    if (!tx_) return {SuffixRawStatus::invalid, "no serving RAW transaction", {}};
    lifecycle_.set_observed_phase(raw_phase(phase()));
    if (!lifecycle_.allow(serving_raw::Lifecycle::Event::restore))
        return {SuffixRawStatus::invalid, "suffix prepare requires an open request at a DRAINED all-stage cut", {}};
    if (!donor || !prefix)
        return {SuffixRawStatus::invalid, "suffix prepare needs both operand bundles", {}};
    const auto result = core::prepare_int8_native_suffix(donor, donor_key, prefix, prefix_key, tx_->geometry_,
                                                         tx_->initial_config_, target, limits, tx_->budget_, out);
    return map_suffix_refusal(result, "native suffix prepare");
}

} // namespace strata::program
