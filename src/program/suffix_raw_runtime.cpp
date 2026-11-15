#include "suffix_raw_runtime.hpp"
#include "owned_vision_request.hpp"
#include "owned_vision_position_banks.hpp"
#include "owned_image_request_operation.hpp"
#include "serving_thread_bridge.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/core/serving_raw_metadata.hpp"
#include "strata/core/resident_copy_fault.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/raw_int8_restore_clear.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/mrope.hpp"
#include "../core/conversation_copy.hpp"
#include "../core/window_hash_vendor/sha256.h"
#include "strata/core/verify.hpp"
#include "strata/core/window_device.hpp"
#include <algorithm>
#include <exception>
#include <limits>
#include <new>
#include <atomic>
#include <chrono>
#include <type_traits>
#include <bit>
#include <cmath>
#include <cstring>
#include <cerrno>
#if defined(__linux__)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
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
struct SuffixRawTransaction::CaptureStorage {
    // Reuse capture's unused source-owner slot; restore Storage stays unchanged.
    // Scratch payloads die before T, and T dies before this reference to R.
    std::shared_ptr<HostCredit> credit;
    std::shared_ptr<HostCredit> scratch_credit;
    std::shared_ptr<const void> runtime;
    NativeSessionAllocation allocation;
    SavedConversation native;
    SnapshotDescriptor descriptor;
    std::vector<NativePleTokens> ple;
};
struct SuffixRawTransaction::VisionAdmission {
    std::shared_ptr<const HostCredit> family_root;
    std::shared_ptr<const HostCredit> local_setup;
    std::shared_ptr<HostCredit> credit;
};
struct SuffixRawTransaction::PositionAdmission {
    std::shared_ptr<const HostCredit> family_root;
    std::shared_ptr<const HostCredit> local_setup;
    std::shared_ptr<HostCredit> credit;
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
    return !image_exact() && key.engine_generation==identity_.engine_generation && key.window_generation==identity_.window_generation &&
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
SuffixRawResult SuffixRawTransaction::reserve_capture(const WindowStateLayout& layout,
    std::shared_ptr<HostCredit>& retained_keep,std::shared_ptr<CaptureStorage>& out) {
#if !defined(__GLIBCXX__)
    return invalid("capture ownership requires the pinned libstdc++ profile");
#else
    static_assert(sizeof(CaptureStorage)==sizeof(Storage) && alignof(CaptureStorage)==alignof(Storage),
                  "capture owner must reuse the unchanged Storage allocation");
    // GNU make_shared<T> passes allocator<void> through _Sp_alloc_shared_tag.
    // __shared_count allocates ONE value of this control's __allocator_type.
    // Use the actual default shared_ptr policy, not an assumed _S_atomic.
    static_assert(std::is_base_of<std::__shared_ptr<Storage>,std::shared_ptr<Storage>>::value &&
                  std::is_base_of<std::__shared_ptr<CaptureStorage>,std::shared_ptr<CaptureStorage>>::value,
                  "make_shared control policy must match the checked GNU path");
    using RestoreControl=std::_Sp_counted_ptr_inplace<Storage,std::allocator<void>,__gnu_cxx::__default_lock_policy>;
    using CaptureControl=std::_Sp_counted_ptr_inplace<CaptureStorage,std::allocator<void>,__gnu_cxx::__default_lock_policy>;
    using RestoreRequest=std::allocator_traits<RestoreControl::__allocator_type>::value_type;
    using CaptureRequest=std::allocator_traits<CaptureControl::__allocator_type>::value_type;
    static_assert(std::is_same<RestoreControl::__allocator_type,std::allocator<RestoreControl>>::value &&
                  std::is_same<CaptureControl::__allocator_type,std::allocator<CaptureControl>>::value &&
                  std::is_same<RestoreRequest,RestoreControl>::value &&
                  std::is_same<CaptureRequest,CaptureControl>::value &&
                  sizeof(CaptureRequest)==sizeof(RestoreRequest) &&
                  alignof(CaptureRequest)==alignof(RestoreRequest),
                  "capture make_shared request must equal unchanged Storage's request");

    uint64_t peak=0;if(!quote(layout,4,peak))return invalid("capture peak overflow");
    // Keep the original Q4 profile allowance, subtracting only the disjoint
    // native/sidecar and descriptor extents that die before scratch_credit.
    uint64_t logical=0,scratch=0,retained=peak;
    bool split=true;ConversationBuffer empty;
    for(const auto& stage:layout.stages)for(const auto& tensor:stage.tensors) {
        if(tensor.bytes>SIZE_MAX) {split=false;break;}
        const auto capacity=empty.allocation_peak(tensor.bytes);
        if(capacity==SIZE_MAX || capacity<tensor.bytes || !add(logical,tensor.bytes)) {split=false;break;}
    }
    if(!split || !product(logical,2,scratch) || scratch>peak) scratch=0;
    // Admissions are single-controller; off-thread final releases only lower C.
    const auto claims=host_cap_->claims.load();
    if(claims>host_cap_->claim_limit || host_cap_->claim_limit-claims<2) scratch=0;
    retained=peak-scratch;
    uint64_t total=retained;
    if(!add(total,scratch) || total!=peak) {retained=peak;scratch=0;}

    // These locals also cover failed preparation and the owner's control tail.
    std::shared_ptr<HostCredit> credit;
    std::shared_ptr<HostCredit> temporary;
    std::shared_ptr<CaptureStorage> storage;
    auto admitted=admit_host(retained,true,credit);if(!admitted)return admitted;
    if(scratch) {admitted=admit_host(scratch,true,temporary);if(!admitted)return admitted;}
    storage=std::make_shared<CaptureStorage>();
    storage->credit=credit;storage->scratch_credit=temporary;
    storage->runtime=runtime_;storage->allocation=allocation_;
    retained_keep=std::move(credit);out=std::move(storage);
    return {SuffixRawStatus::ok,{},{}};
#endif
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
SuffixRawResult SuffixRawTransaction::admit_position_banks(std::span<const uint64_t> devices,
    uint64_t host_bytes,PositionAdmission& out) {
    PositionAdmission candidate;
    candidate.family_root=setup_credit_;candidate.local_setup=setup_credit_;
    const auto& parent=allocation_.reservation();
    if(out.credit||!candidate.family_root||!candidate.local_setup||!host_cap_||
       candidate.family_root->cap!=host_cap_||candidate.local_setup->cap!=host_cap_||
       candidate.family_root->cap.owner_before(host_cap_)||host_cap_.owner_before(candidate.family_root->cap)||
       candidate.local_setup->cap.owner_before(host_cap_)||host_cap_.owner_before(candidate.local_setup->cap)||
       !candidate.family_root->global||!candidate.family_root->global->canonical_owner(candidate.family_root->global)||
       candidate.family_root->global->purpose()!=ReservationPurpose::private_attachment||
       controller_!=std::this_thread::get_id()||!initialized_||phase_!=Phase::drained||pending_||
       slot_occupancy_||!allocation_.healthy()||!parent||!parent->canonical_owner(parent)||
       parent->purpose()!=ReservationPurpose::private_work||!host_bytes||devices.empty()||devices.size()>64||
       devices.size()!=parent->resources().device_bytes.size())return invalid("position bank private root");
    try {
        // Same ordinary admission bootstrap as admit_host(global=true): retain
        // true prepaid roots before the claim/lease transients. One H claim and
        // ONE global resource vector, not a second device reservation.
        auto admitted=admit_host(host_bytes,false,candidate.credit);if(!admitted)return admitted;
        WindowResources resources;
        resources.device_bytes.assign(devices.begin(),devices.end());resources.host_bytes=host_bytes;
        auto reserved=budget_.reserve(resources,ReservationPurpose::private_attachment,parent);
        if(!reserved)return {SuffixRawStatus::budget_refused,"position bank global vector refused",{}};
        candidate.credit->global=std::move(reserved.lease);
        out.family_root.swap(candidate.family_root);out.local_setup.swap(candidate.local_setup);
        out.credit.swap(candidate.credit);return {SuffixRawStatus::ok,{},{}};
    }catch(const std::bad_alloc&){return {SuffixRawStatus::budget_refused,"position bank admission allocation refused",{}};}
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
        // R must outlive both scratch teardown and the capture owner's control block.
        std::shared_ptr<HostCredit> retained_keep;
        std::shared_ptr<CaptureStorage> storage;auto admitted=reserve_capture(layout,retained_keep,storage);if(!admitted)return admitted;
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
        // R must outlive both scratch teardown and the capture owner's control block.
        std::shared_ptr<HostCredit> retained_keep;
        std::shared_ptr<CaptureStorage> storage;auto admitted=reserve_capture(layout,retained_keep,storage);if(!admitted)return admitted;
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

// The sole production caller proves exclusivity by startup ordering, not a
// pointer observation or a caller-supplied ready bit. All native initialization
// precedes this call; owned graph/request/history producers follow publication.
bool ServingRawSession::open_owned_startup(const ModelGeometry& geometry,
        NativeSessionAllocation allocation,std::shared_ptr<const void> runtime,
        std::vector<Verifier*> verifiers,WindowBudget& budget,uint64_t max_tokens,
        uint64_t host_cap_bytes,std::shared_ptr<ServingRawSession>& out,std::string& error) {
    if(out){error="owned startup cannot replace a published RAW owner";return false;}
#if !defined(__linux__) || !defined(__GLIBCXX__) || defined(STRATA_GPU_METAL)
    error="owned startup requires the inherited native RAW profile";return false;
#else
    // These are EXACTLY the ordinary factory/bootstrap components. No new
    // reset bookkeeping is allocated from an assumed remainder of that quote.
    auto candidate=open(geometry,std::move(allocation),std::move(runtime),std::move(verifiers),
        budget,max_tokens,host_cap_bytes,error);
    if(!candidate)return false;
    auto& tx=*candidate->tx_;
    if(!candidate->healthy()||candidate->request_open()||tx.phase_!=SuffixRawTransaction::Phase::idle||
       tx.cut_.id||tx.pending_||tx.controller_!=std::this_thread::get_id()){
        error="owned startup requires its fresh closed initialized controller";return false;
    }
    for(const auto* verifier:tx.verifiers_){
        if(verifier->g_!=&geometry||verifier->owned_batch_setup_||verifier->owned_group_busy_){
            error="owned startup geometry/fresh graph binding differs";return false;
        }
    }
    const size_t stages=tx.allocation_.stages().size(); // initialize/resolve already binds <=8
    if(!stages||stages>8){error="owned startup stage bound differs from RAW initialization";return false;}
    struct ResetFrame {
        StateCut empty;
        std::array<StateCut,8> cuts;
        std::vector<int32_t> empty_tokens;
    };
    struct ResetFence {
        bool armed=false;
        ~ResetFence(){if(armed)std::terminate();} // before ANY frame/credit/runtime unwind
    };
    using CreditControl=std::_Sp_counted_ptr_inplace<SuffixRawTransaction::HostCredit,
        std::allocator<void>,__gnu_cxx::__default_lock_policy>;
    using CreditRequest=std::allocator_traits<CreditControl::__allocator_type>::value_type;
    using LeaseControl=std::_Sp_counted_ptr<ReservationLease*,__gnu_cxx::__default_lock_policy>;
    static_assert(std::is_same<CreditRequest,CreditControl>::value,
        "startup quote must use the actual GNU make_shared request");
    // window_hash.cpp's private Impl has exactly sha256_t + uint64_t. Charge
    // both possible padding boundaries, its digest copy and its actual4096-byte
    // history buffer. This is the inherited requested-size profile, not RSS.
    uint64_t bytes=sizeof(ResetFrame)+sizeof(ResetFence)+sizeof(std::shared_ptr<const void>)+
        sizeof(CreditRequest)+sizeof(ReservationLease)+sizeof(LeaseControl)+
        2*sizeof(WindowResources)+sizeof(ReservationResult)+3*sizeof(SuffixRawResult)+
        8*sizeof(WindowDeviceScope)+2*sizeof(kernels::QsaShapes)+sizeof(WindowHasher)+
        2*sizeof(sha256_t)+sizeof(uint64_t)+alignof(sha256_t)+alignof(uint64_t)+
        sizeof(StateDigest)+sizeof(std::array<uint8_t,4096>)+sizeof(std::array<uint8_t,8>)+
        sizeof("strata-window-history-v1")+sizeof(unsigned)+sizeof(std::span<const StateCut>)+
        // Fixed added forwarding-frame objects/references, not transferred heap payload.
        sizeof(NativeSessionAllocation)+sizeof(std::shared_ptr<const void>)+
        sizeof(std::vector<Verifier*>)+sizeof(std::shared_ptr<ServingRawSession>)+
        sizeof(const ModelGeometry*)+sizeof(WindowBudget*)+
        sizeof(std::shared_ptr<ServingRawSession>*)+sizeof(std::string*)+
        4*sizeof(size_t)+4*sizeof(uint64_t)+2*sizeof(void*);
    uint64_t directory=0;
    if(!product(uint64_t(tx.allocation_.reservation()->resources().device_bytes.size()),
                2*sizeof(uint64_t),directory)||!add(bytes,directory)){
        error="owned startup reset metadata quote overflow";return false;
    }
    // One private claim under actual H AND actual global parent; no escape or
    // weak export. Inherited admission controls retain their ordinary bootstrap
    // protection; candidate/setup outlive final credit/control deallocation.
    std::shared_ptr<const void> credit;
    const auto admitted=tx.admit_controller_host(bytes,credit);
    if(!admitted){error=admitted.detail;return false;}
    try {
        ResetFrame frame; // all new reset payload/container metadata is now paid
        frame.empty.id=1; // fresh controller was checked cut0, no caller identity
        frame.empty.history=window_history_digest(frame.empty_tokens);
        frame.cuts.fill(frame.empty);
        ResetFence fence; // destroyed first on failure, before frame and credit
        fence.armed=true;
        const auto begun=tx.begin_driver_work();
        if(!begun){
            if(tx.phase_==SuffixRawTransaction::Phase::poisoned)std::terminate();
            fence.armed=false;error=begun.detail;return false;
        }
        // Join ALL stage initialization/copy work before resetting ANY session.
        for(const auto& native:tx.allocation_.stages()){
            WindowDeviceScope device(native.device);
            if(!device.ok()||cudaDeviceSynchronize()!=cudaSuccess||!device.finish())std::terminate();
        }
        if(!tx.poll_all())std::terminate();
        for(const auto& native:tx.allocation_.stages()){
            WindowDeviceScope device(native.device);
            if(!device.ok()||!session_zero_checked(*native.session,geometry,nullptr,nullptr,error)||
               cudaDeviceSynchronize()!=cudaSuccess||!device.finish())std::terminate();
        }
        // Actual reset and checked completion earn ONE common cut. Polling
        // observes all real verifiers; no fake request/open or readiness token.
        if(!tx.complete_driver_work(true,frame.empty,std::span<const StateCut>(frame.cuts.data(),stages)))
            std::terminate();
        if(!candidate->healthy()||candidate->request_open()||candidate->phase()!=Phase::drained)
            std::terminate();
        fence.armed=false;
        out=std::move(candidate); // same owner later used by real TEXT requests
        return true;
    }catch(const std::bad_alloc&){error="owned startup reset metadata allocation refused";return false;}
#endif
}

bool ServingRawSession::prepare_owned_position_banks(
        std::shared_ptr<const NativeBatchSlotBindings> slots,std::span<prefill::Prefill* const> prefills,
        uint64_t fixed_enclosing_bytes,OwnedPositionBankSetup& setup,std::string& error) {
#if !defined(__linux__) || !defined(__GLIBCXX__) || defined(STRATA_GPU_METAL)
    (void)slots;(void)prefills;(void)fixed_enclosing_bytes;(void)setup;
    error="position banks require the inherited native GNU profile";return false;
#else
    using Owner=OwnedVisionPositionBanks;using Binding=OwnedPositionBankBinding;
    if(setup.owner_||setup.keepers_.admitted_credit||!tx_||!healthy()||request_open()||private_window_open()||phase()!=Phase::drained){
        error="position banks require actual closed drained startup";return false;
    }
    auto& tx=*tx_;const size_t stages=tx.allocation_.stages().size();
    const auto& parent=tx.allocation_.reservation();
    if(!slots||slots->geometry()!=tx.geometry_identity_||!stages||stages>8||
       slots->stage_count()!=stages||prefills.size()!=stages||tx.verifiers_.size()!=stages||
       tx.cut_.id!=1||tx.pending_||tx.controller_!=std::this_thread::get_id()||
       !parent||parent->resources().device_bytes.empty()||parent->resources().device_bytes.size()>64||
       !fixed_enclosing_bytes||tx.max_tokens_>uint64_t(INT32_MAX)-64){
        error="position bank startup shape/cut differs";return false;
    }
    for(size_t i=0;i<stages;++i){
        const auto& n=tx.allocation_.stages()[i];auto* v=tx.verifiers_[i];
        if(!v||!prefills[i]||n.device<0||size_t(n.device)>=parent->resources().device_bytes.size()||
           !n.session||n.session->max_cells!=int64_t(tx.max_tokens_)||v->ss_!=n.session||v->g_!=tx.geometry_identity_||
           v->owned_batch_setup_||v->position_banks_||v->owned_group_busy_||v->slot_inventory_ready_||
           v->slot_inventory_generation_==UINT64_MAX||
           !prefills[i]->owned_position_bank_target(tx.geometry_identity_,n.session,n.device)){
            error="position bank stage/consumer binding differs";return false;
        }
        for(size_t slot=0;slot<2;++slot){
            const auto* b=slots->stage(slots->token(slot),i);
            if(!b||b->device!=n.device||b->session==n.session||b->session->max_cells!=n.session->max_cells||
               b->session->layer_lo!=n.session->layer_lo||b->session->layer_hi!=n.session->layer_hi){
                error="position bank physical slot differs";return false;
            }
        }
        for(auto g:v->exec_)if(g){error="position bank install after main capture";return false;}
        for(auto g:v->exec_nr_)if(g){error="position bank install after alternate capture";return false;}
        if(v->commit_exec_||!v->exec_bm_.empty()||!v->commit_bm_.empty()){
            error="position bank install after graph publication";return false;
        }
    }
    struct Frame {
        SuffixRawTransaction::PositionAdmission admission;
        Owner::Keepers keepers;
        std::array<uint64_t,64> devices{};
        std::shared_ptr<Owner> owner;
        std::array<std::shared_ptr<Binding>,8> bindings;
    };
    using OA=Owner::HostAllocator<Owner>;using BA=Owner::HostAllocator<Binding>;
    using Directory=OwnedImageSessionRecords;
    using DA=VisionRequestOwner::HostAllocator<Directory>;
    using DC=std::_Sp_counted_ptr_inplace<Directory,DA,__gnu_cxx::__default_lock_policy>;
    using OC=std::_Sp_counted_ptr_inplace<Owner,OA,__gnu_cxx::__default_lock_policy>;
    using BC=std::_Sp_counted_ptr_inplace<Binding,BA,__gnu_cxx::__default_lock_policy>;
    using CC=std::_Sp_counted_ptr_inplace<SuffixRawTransaction::HostCredit,std::allocator<void>,__gnu_cxx::__default_lock_policy>;
    using LC=std::_Sp_counted_ptr<ReservationLease*,__gnu_cxx::__default_lock_policy>;
    static_assert(std::is_same<typename std::allocator_traits<typename OC::__allocator_type>::value_type,OC>::value);
    static_assert(std::is_same<typename std::allocator_traits<typename BC::__allocator_type>::value_type,BC>::value);
    static_assert(std::is_same<typename std::allocator_traits<typename DC::__allocator_type>::value_type,DC>::value);
    uint64_t cells=tx.max_tokens_,bank_bytes=0,all_banks=0,host=fixed_enclosing_bytes,directory=0;
    if(!kernels::owned_mrope_bank_layout(tx.max_tokens_,cells,bank_bytes)||!product(bank_bytes,3,all_banks)||
       !product(uint64_t(parent->resources().device_bytes.size()),4*sizeof(uint64_t),directory)||
       !add(host,bank_bytes)||!add(host,sizeof(Frame))||!add(host,sizeof(OC))||
       !add(host,stages*sizeof(BC))||!add(host,sizeof(DC)+alignof(DC)-1+4*sizeof(DA)+
           sizeof(VisionRequestOwner::Keepers)+sizeof(std::shared_ptr<Directory>))||
       !add(host,sizeof(CC)+sizeof(ReservationLease)+sizeof(LC))||
       !add(host,directory)||!add(host,stages*sizeof(OwnedNativeSession))||
       !add(host,2*sizeof(std::weak_ptr<const void>))|| // actual allocator callbacks capture one weak each
       !add(host,sizeof(SuffixRawTransaction::PositionAdmission)+
           2*sizeof(WindowResources)+sizeof(ReservationResult)+4*sizeof(SuffixRawResult))||
       !add(host,8*sizeof(Owner::Keepers)+4*sizeof(OA)+4*sizeof(BA)+8*sizeof(WindowDeviceScope))||
       !add(host,sizeof(std::span<Verifier* const>)+
           4*sizeof(std::span<const uint64_t>)+4*sizeof(size_t)+8*sizeof(uint64_t)+8*sizeof(void*))){
        error="position bank checked quote overflow";return false;
    }
    // Only fixed forwarding/enclosing objects exist before admission; their
    // inherited construction interval is explicitly not a whole-entry proof.
    Frame frame;
    for(const auto& n:tx.allocation_.stages())if(!add(frame.devices[size_t(n.device)],all_banks)){
        error="position bank device quote overflow";return false;
    }
    const auto admitted=tx.admit_position_banks(
        std::span<const uint64_t>(frame.devices.data(),parent->resources().device_bytes.size()),host,frame.admission);
    if(!admitted){error=admitted.detail;return false;}
    frame.keepers={frame.admission.family_root,frame.admission.local_setup,frame.admission.credit};
    // Cannot release this admission on a later bad_alloc/native refusal: the
    // pre-existing enlarged consumer allocations have not been freed yet.
    setup.keepers_=frame.keepers;
    try {
        const uint64_t epoch=next_verifier_incarnation();
        if(!epoch){error="position bank install epoch exhausted";return false;}
        frame.owner=std::allocate_shared<Owner>(OA(frame.keepers),frame.keepers,tx.allocation_,slots,size_t(cells),epoch);
        for(size_t i=0;i<stages;++i)frame.bindings[i]=std::allocate_shared<Binding>(BA(frame.keepers),
            frame.owner,i,tx.verifiers_[i]->slot_verifier_incarnation_,tx.verifiers_[i]->slot_inventory_generation_+1);
        auto& owner=*frame.owner;
        owner.staging_=owner.staging_allocator_.allocate(size_t(cells)*3);
        for(size_t c=0;c<size_t(cells);++c)for(size_t axis=0;axis<3;++axis)
            std::construct_at(owner.staging_+3*c+axis,int32_t(c));
        for(size_t i=0;i<stages;++i){
            WindowDeviceScope scope(tx.allocation_.stages()[i].device);
            if(!scope.ok())std::terminate();
            for(size_t bank=0;bank<3;++bank){
                if(cudaMalloc(reinterpret_cast<void**>(&owner.banks_[i][bank]),size_t(bank_bytes))!=cudaSuccess){
                    error="position bank native allocation refused";return false;
                }
                if(cudaMemcpy(owner.banks_[i][bank],owner.staging_,size_t(bank_bytes),cudaMemcpyHostToDevice)!=cudaSuccess)
                    std::terminate();
            }
            if(cudaDeviceSynchronize()!=cudaSuccess||!scope.finish())std::terminate();
        }
        const VisionRequestOwner::Keepers directory_keepers{frame.keepers.family_root,
            frame.keepers.local_setup,frame.keepers.admitted_credit};
        auto directory=std::allocate_shared<Directory>(DA(directory_keepers));
        if(!tx.poll_all())std::terminate();
        // No allocation/failure after publication begins. Same authenticated
        // binding goes to both consumers; immutable epoch/address through capture.
        for(size_t i=0;i<stages;++i){tx.verifiers_[i]->position_banks_=frame.bindings[i];prefills[i]->position_banks_=frame.bindings[i];}
        alignment_owner_credit_=std::move(directory);tx.initialized_=3;
        setup.owner_=std::move(frame.owner);return true;
    }catch(const std::bad_alloc&){error="position bank admitted metadata allocation refused";return false;}
#endif
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
    if((tx_&&tx_->image_exact())||image_operation()){error="generic slot import refuses image context";return false;}
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
    if(tx_&&tx_->image_exact())return invalid("generic slot capture refuses image context");
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
    if(tx_&&tx_->image_exact())return nullptr;
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
    if(tx_&&tx_->image_exact()){error="img context";return false;}
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
    // Destruction is not permission to forget a native image kind on a still
    // owned allocation. The real controller must reset/certify it first.
    if(tx_&&tx_->image_exact())std::terminate();
    if(const auto directory=image_directory())for(const auto& op:directory->prepared_)
        if(op&&op->transport_published_&&!op->producer_closed_)std::terminate();
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
    if(tx_&&tx_->image_exact()&&!image_retire_current(error))return false;
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
    if(tx_&&tx_->image_exact())return invalid("generic suffix operation refuses image context");
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
    if(tx_&&tx_->image_exact())return invalid("image context");
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
    if(tx_&&tx_->image_exact())return invalid("generic suffix operation refuses image context");
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

SuffixRawResult SuffixRawTransaction::admit_vision_host(const ServingRawSession& session,
    uint64_t bytes,VisionAdmission& out) {
    // These are genuine private members, never a caller-supplied parent/credit.
    VisionAdmission candidate;
    candidate.family_root=slot_host_root_credit_?slot_host_root_credit_:setup_credit_;
    candidate.local_setup=setup_credit_;
    const auto same_cap=[&](const auto& credit) {
        return credit&&credit->cap.get()==host_cap_.get()&&
            !credit->cap.owner_before(host_cap_)&&!host_cap_.owner_before(credit->cap)&&
            credit->bytes&&credit->global&&credit->global->canonical_owner(credit->global)&&
            credit->global->purpose()==ReservationPurpose::private_attachment;
    };
    const auto& parent=allocation_.reservation();
    if(controller_!=std::this_thread::get_id()||!initialized_||!host_cap_||
       (phase_!=Phase::idle&&phase_!=Phase::drained)||!allocation_.healthy()||
       session.transaction().get()!=this||!session.request_open()||session.private_window_open()||
       !session.healthy()||!same_cap(candidate.family_root)||!same_cap(candidate.local_setup)||
       (slot_occupancy_&&!slot_host_root_credit_)||!parent||!parent->canonical_owner(parent)||
       (parent->purpose()!=ReservationPurpose::private_work&&parent->purpose()!=ReservationPurpose::saved_maintenance))
        return invalid("vision root");
    // Existing admission checks the actual same ledger and charged parent.
    auto result=admit_host(bytes,true,candidate.credit);
    if(!result)return {result.status,"vision budget",{}};
    // Swapping also preserves keeper-before-credit teardown if an internal
    // caller ever reuses a previously populated private result.
    out.family_root.swap(candidate.family_root);out.local_setup.swap(candidate.local_setup);
    out.credit.swap(candidate.credit);return {SuffixRawStatus::ok,{},{}};
}


VisionRequestOwner::Keepers ServingRawSession::image_reactor_keepers()const {
    // Only the real composition can ask for the actual typed reactor allocator;
    // its full control/map/node quote was included by the bank startup caller.
    if(!tx_||slot_occupancy_||!healthy()||request_open()||phase()!=Phase::drained||
       tx_->controller_!=std::this_thread::get_id()||tx_->verifiers_.empty()||!image_directory())std::terminate();
    const auto binding=tx_->verifiers_[0]->position_banks_;
    if(!binding||!binding->owner_)std::terminate();
    const auto& k=binding->owner_->staging_allocator_.keepers_;
    return {k.family_root,k.local_setup,k.admitted_credit};
}

std::shared_ptr<OwnedImageSessionRecords> ServingRawSession::image_directory()const {
    if(!tx_||slot_occupancy_||(tx_->initialized_!=2&&tx_->initialized_!=3))return {};
    return std::const_pointer_cast<OwnedImageSessionRecords>(
        std::static_pointer_cast<const OwnedImageSessionRecords>(alignment_owner_credit_));
}
std::shared_ptr<OwnedImageRequestOperation> ServingRawSession::image_operation()const {
    if(const auto directory=image_directory())return directory->current_;
    return tx_&&slot_occupancy_&&tx_->image_exact()?std::const_pointer_cast<OwnedImageRequestOperation>(
        std::static_pointer_cast<const OwnedImageRequestOperation>(alignment_owner_credit_)):nullptr;
}
void ServingRawSession::set_image_operation(std::shared_ptr<OwnedImageRequestOperation> operation) {
    if(slot_occupancy_){alignment_owner_credit_=std::move(operation);tx_->initialized_=alignment_owner_credit_?2:1;return;}
    const auto directory=image_directory();if(!directory)std::terminate();
    directory->current_=std::move(operation);tx_->initialized_=directory->current_?2:3;
}
bool ServingRawSession::image_bind_controller(const OwnedAdmissionCancel& admissions,std::mutex& mutex,
    std::shared_ptr<const void> reactor,const ServingThreadBridge* bridge,std::string& error) {
    const auto directory=image_directory();
    if(!directory||directory->reactor_||!reactor||!healthy()||request_open()||phase()!=Phase::drained||
       tx_->controller_!=std::this_thread::get_id()){
        error="img controller";return false;
    }
    // One-time binding by the concrete real reactor. A later preparation cannot
    // nominate a replacement ledger, mutex, manager, or aliased ownership group.
    directory->reactor_=std::move(reactor);directory->admissions_=&admissions;
    directory->mutex_=&mutex;directory->bridge_=bridge;return true;
}
bool ServingRawSession::image_select_preparation(size_t slot,uint64_t request,uint64_t incarnation,std::string& error) {
    const auto directory=image_directory();
    if(!directory||slot>=2||directory->current_||request_open()||!healthy()||phase()!=Phase::drained){
        error="image busy";return false;
    }
    const auto op=directory->prepared_[slot];
    if(!op||op->request_key_!=request||op->admission_incarnation_!=incarnation||
       op->main_install_generation_||op->phase_.load(std::memory_order_acquire)!=OwnedImageRequestOperation::Phase::unsealed){
        error="image stale";return false;
    }
    if(!open_request(error))return false;
    directory->current_=op; // host capture only; no native image kind before reset
    return true;
}

bool ServingRawSession::image_authority_current(bool delivered,std::string& error)const {
    const auto op=image_operation();
    if(!op||!tx_||!healthy()||!request_open()||private_window_open()||
       tx_->controller_!=std::this_thread::get_id()||op->source_session_!=this||
       op->runtime_!=tx_->runtime_.get()||op->source_window_!=key().window_generation||
       !op->admissions_||!op->admission_mutex_||!op->reactor_owner_||
       op->destination_token_.slot>=2){error="image source authority changed";return false;}
    {
        std::lock_guard<std::mutex> lock(*op->admission_mutex_);
        if(!op->admissions_->image_current(int(op->destination_token_.slot),op->request_key_,
            op->admission_incarnation_,delivered)){error="image admission is cancelled or replaced";return false;}
    }
    if(tx_->image_exact()){
        const auto binding=tx_->verifiers_.empty()?nullptr:tx_->verifiers_[0]->position_banks_;
        if(!binding||!binding->owner_||!op->main_install_generation_||
           binding->owner_->content_generations_[0]!=op->main_install_generation_){error="image main bank generation changed";return false;}
        const auto actual=binding->owner_->image_contexts_[0].lock();
        if(actual.get()!=op.get()||actual.owner_before(op)||op.owner_before(actual)){
            error="image main canonical context changed";return false;}
    }
    if(op->bridge_){const auto* binding=op->bridge_->binding(op->destination_token_.slot);
        if(op->bridge_->engine()!=op->engine_||!binding||binding->request!=op->request_key_||
           binding->foreground.serial!=op->foreground_||binding->foreground.result_revision!=op->foreground_revision_||
           !op->bridge_->request_ready(op->destination_token_.slot,op->request_key_)){
            error="image foreground issue changed";return false;
        }
    }
    return true;
}

bool ServingRawSession::image_prepare_transport(ServingRawSession& destination,
    const OwnedAdmissionCancel* admissions,std::mutex* mutex,std::shared_ptr<const void> reactor,
    const ServingThreadBridge* bridge,int slot,uint64_t request,uint64_t incarnation,
    uint64_t file_bytes,uint64_t tokens,uint64_t output,std::string& error) {
#if !defined(__linux__) || !defined(__GLIBCXX__) || __GLIBCXX__!=20240904 || !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE!=13
    error="image transport requires the pinned Linux GNU profile";return false;
#else
    using Op=OwnedImageRequestOperation;using A=Op::Allocator<Op>;
    using C=std::_Sp_counted_ptr_inplace<Op,A,__gnu_cxx::__default_lock_policy>;
    static_assert(std::is_same<typename std::allocator_traits<typename C::__allocator_type>::value_type,C>::value);
    const auto directory=image_directory();
    if(!directory||directory->current_||!tx_||slot_occupancy_||!healthy()||request_open()||phase()!=Phase::drained||
       !admissions||!mutex||!reactor||slot<0||slot>1||!request||!incarnation||
       !tokens||!output||output>max_tokens()||tokens>max_tokens()-output||file_bytes<20||file_bytes%4||
       !destination.slot_occupancy_||!destination.healthy()||destination.alignment_known_||
       destination.slot_controller_bound_||destination.alignment_owner_credit_||destination.phase()!=Phase::drained||
       destination.request_open()||destination.cut().accepted||destination.cut().forwarded||
       destination.slot_occupancy_->slot()!=size_t(slot)||destination.tx_->runtime_!=tx_->runtime_||
       destination.tx_->runtime_.owner_before(tx_->runtime_)||tx_->runtime_.owner_before(destination.tx_->runtime_)){
        error="image prepare needs the actual drained source and vacant destination";return false;
    }
    if(directory->prepared_[size_t(slot)]){error="image duplicate";return false;}
    if(directory->admissions_!=admissions||directory->mutex_!=mutex||directory->bridge_!=bridge||
       directory->reactor_.get()!=reactor.get()||directory->reactor_.owner_before(reactor)||
       reactor.owner_before(directory->reactor_)){error="img controller";return false;}
    {std::lock_guard<std::mutex> lock(*mutex);
        if(!admissions->image_current(slot,request,incarnation,false)){error="image prepare admission stale";return false;}}
    const auto* binding=bridge?bridge->binding(size_t(slot)):nullptr;
    if(bridge&&(!binding||!bridge->request_ready(size_t(slot),request))){error="image prepare is not runnable";return false;}
    uint64_t row=0,bound=0,host=file_bytes;
    if(!product(uint64_t(tx_->geometry_.n_embd),sizeof(float),row)||!add(row,20)||
       !product(tokens,row,bound)||file_bytes>bound){error="image producer extent overflow";return false;}
    // Fixed producer helpers/preflight are paid by the two-slot startup
    // envelope, not admitted late here and not charged again per request.
    const uint64_t metadata=sizeof(C)+alignof(C)-1+sizeof(Op::Keepers)+
        sizeof(SuffixRawTransaction::VisionAdmission)+4*sizeof(A)+8*sizeof(std::shared_ptr<const void>)+
        2*sizeof(std::shared_ptr<Op>)+8*sizeof(uint64_t)+6*sizeof(void*);
    uint64_t token_work=0,command_work=0;
    // Actual first capture and final consume parse the SAME bounded command.
    // Cover both geometrically grown int64 vectors, int32 endpoint history,
    // the retained command and four simultaneous reader/consumer spellings.
    uint64_t native_metadata=0;
    // GNU's NativeSessionAllocation copy copies its stage vector by size;
    // each of the two allocator-produced function targets captures ONE weak
    // storage observation. No source actor/DriverWindowRuntime is retained. Fixed
    // copy/preflight/cut frames are separate from producer/bootstrap objects.
    if(!product(uint64_t(tx_->allocation_.stages().size()),sizeof(OwnedNativeSession),native_metadata)||
       !add(native_metadata,2*sizeof(std::weak_ptr<void>))||
       // image_reset's empty-history vector is one sequential fixed object;
       // its complete requested extent is paid before any image owner exists.
       !add(native_metadata,sizeof(std::vector<int32_t>)+8*sizeof(WindowDeviceScope)+4*sizeof(ConversationStateSizes)+4*sizeof(std::array<const void*,5>)+
           4*sizeof(std::array<uint64_t,5>)+sizeof(std::array<StateCut,8>)+4*sizeof(std::string)+4*128)||
       !product(tokens,8*sizeof(int64_t)+128+2*sizeof(void*),token_work)||
       !product(tx_->max_tokens_,12,command_work)||!add(command_work,65536)||
       !product(command_work,4,command_work)||!add(host,native_metadata)||!add(host,token_work)||
       !add(host,command_work)||!add(host,metadata)||host>SIZE_MAX){error="image quote";return false;}
    if(!open_request(error))return false;
    try {
        SuffixRawTransaction::VisionAdmission admission;
        const auto admitted=tx_->admit_vision_host(*this,host,admission);
        if(!admitted){error=admitted.detail;if(!close_request(error))std::terminate();return false;}
        const Op::Keepers keepers{admission.family_root,admission.local_setup,admission.credit};
        auto op=std::allocate_shared<Op>(A(keepers),keepers);
        op->reactor_owner_=std::move(reactor);op->admissions_=admissions;op->admission_mutex_=mutex;
        op->bridge_=bridge;op->source_session_=this;op->runtime_=tx_->runtime_.get();
        op->source_window_=key().window_generation;op->request_key_=request;op->admission_incarnation_=incarnation;
        op->destination_token_=destination.slot_occupancy_->token_;
        op->destination_occupancy_=destination.slot_occupancy_incarnation();
        op->file_bytes_=file_bytes;op->token_count_=tokens;op->full_output_budget_=output;
        if(binding){op->engine_=bridge->engine();op->foreground_=binding->foreground.serial;
            op->foreground_revision_=binding->foreground.result_revision;}
        directory->prepared_[size_t(slot)]=std::move(op);
        // This was genuine preparation work for the canonical image admission.
        // Release its actual RAW request fence while the producer fills input;
        // the fixed directory retains payment and the SAME unconsumed ticket.
        if(!close_request(error))std::terminate();
        return true;
    }catch(const std::bad_alloc&){error="image transport admitted control allocation refused";
        if(!close_request(error))std::terminate();
        return false;}
#endif
}

bool ServingRawSession::image_capture_transport(int fd,std::span<const int64_t> tokens,std::string& error) {
#if !defined(__linux__)
    (void)fd;(void)tokens;error="sealed image transport requires Linux";return false;
#else
    if(!image_authority_current(true,error)||image_operation()->captured_||fd<0||
       tokens.size()!=image_operation()->token_count_){error="image capture delivery differs";return false;}
    auto& op=*image_operation();
    struct stat status{};
    const int required=F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL;
    const int seals=::fcntl(fd,F_GET_SEALS);
    if(seals<0||(seals&required)!=required||::fstat(fd,&status)!=0||!S_ISREG(status.st_mode)||
       status.st_size<0||uint64_t(status.st_size)!=op.file_bytes_){error="image transport lacks immutable seals/extent";return false;}
    std::shared_ptr<const VisionRequestOwner> owner;
    const auto prepared=prepare_vision_request(fd,tokens,op.full_output_budget_,owner);
    if(!prepared){error=prepared.detail;return false;}
    if(!image_authority_current(true,error))return false;
    op.request_=std::move(owner);op.captured_=true;return true;
#endif
}

bool ServingRawSession::image_close_transport(uint64_t request,uint64_t incarnation,std::string& error) {
    const auto directory=image_directory();std::shared_ptr<OwnedImageRequestOperation> operation;
    if(directory)for(const auto& candidate:directory->prepared_)if(candidate&&candidate->request_key_==request&&
        candidate->admission_incarnation_==incarnation){operation=candidate;break;}
    if(!operation||!operation->transport_published_||operation->producer_closed_){
        error="image close";return false;
    }
    // Real reactor receives this only after actual producer FD close. It may
    // settle a prepared image while a DIFFERENT foreground owns MAIN; it never
    // changes that foreground's fence, STOP flag or current image binding.
    operation->producer_closed_=true;return true;
}

// This visitor never constructs SavedConversation, SnapshotDescriptor,
// SessionKvSource or BundleHandle. It reads the exact authoritative byte domains
// used by conversation_checkpoint_save/conversation_kv_save and adds the actual
// bank bytes. The count pass performs no allocation, device submission or read.
bool ServingRawSession::image_peer_visit(bool read,std::byte* out,size_t capacity,
    size_t& bytes,size_t& parts,std::string& error) {
    bytes=0;parts=0;
    const auto op=image_operation();
    if(!tx_||tx_->controller_!=std::this_thread::get_id()||!tx_->image_exact()||!slot_occupancy_||!request_open()||!healthy()||
       phase()!=Phase::drained||!op||!op->request_||
       slot_occupancy_->image_context_.get()!=op.get()||
       slot_occupancy_->image_context_.owner_before(op)||op.owner_before(slot_occupancy_->image_context_)||
       op->phase_.load(std::memory_order_acquire)!=OwnedImageRequestOperation::Phase::consumed||
       op->runtime_!=tx_->runtime_.get()||op->destination_token_!=slot_occupancy_->token_||
       op->destination_occupancy_!=slot_occupancy_incarnation()||
       slot_tokens_.size()!=cut().accepted||window_history_digest(slot_tokens_)!=cut().history||
       slot_current_key_.engine_generation!=key().engine_generation||slot_current_key_.window_generation!=key().window_generation||
       slot_current_key_.session_id!=key().session_id||slot_current_key_.thread_id!=key().thread_id||
       slot_current_key_.window_id!=key().window_id||slot_current_key_.model_profile!=key().model_profile||
       slot_current_key_.topology!=key().topology||tx_->allocation_.stages().size()>8){error="image peer endpoint changed";return false;}
    if(op->bridge_){const auto* binding=op->bridge_->binding(slot_occupancy_->slot());
        if(!binding||binding->request!=op->request_key_||binding->occupancy!=slot_occupancy_incarnation()){
            error="image peer live binding changed";return false;}}
    const auto banks=tx_->verifiers_.empty()?nullptr:tx_->verifiers_[0]->position_banks_;
    if(!banks||!banks->owner_){error="image peer bank missing";return false;}
    const auto& bank=*banks->owner_;const size_t bank_index=slot_occupancy_->slot()+1;
    const auto installed=bank.image_contexts_[bank_index].lock();
    if(installed.get()!=op.get()||installed.owner_before(op)||op.owner_before(installed)||
       bank.content_generations_[bank_index]!=op->slot_install_generation_||
       bank.stages_!=tx_->allocation_.stages().size()){
        error="image peer bank/context generation changed";return false;}
    const auto append=[&](const void* source,size_t n,bool device,bool tensor=false)->bool {
        if(n>SIZE_MAX-bytes||(read&&(bytes>capacity||n>capacity-bytes))||(!source&&n))return false;
        if(read&&n){
            if(device){if(cudaMemcpy(out+bytes,source,n,cudaMemcpyDefault)!=cudaSuccess)std::terminate();}
            else std::memcpy(out+bytes,source,n);
        }
        bytes+=n;if(tensor)++parts;return true;
    };
    const auto scalar=[&](const auto& value){return append(&value,sizeof(value),false);};
    const auto text=[&](const std::string& value){return scalar(value.size())&&append(value.data(),value.size(),false);};
    const auto& k=slot_current_key_;const auto& c=cut();
    if(!scalar(k.engine_generation)||!scalar(k.window_generation)||!text(k.session_id)||!text(k.thread_id)||
       !text(k.window_id)||!scalar(k.revision)||!scalar(k.append_epoch)||!scalar(k.selection_epoch)||
       !scalar(k.provenance_epoch)||!scalar(k.model_profile)||!scalar(k.topology)||
       !scalar(c.id)||!scalar(c.accepted)||!scalar(c.forwarded)||!scalar(c.exact_frontier)||
       !scalar(c.output_watermark)||!scalar(c.history)||!scalar(op->request_key_)||
       !scalar(op->admission_incarnation_)||!scalar(op->destination_occupancy_)||
       !scalar(op->slot_install_generation_)||!scalar(slot_tokens_.size())||
       slot_tokens_.size()>SIZE_MAX/sizeof(int32_t)||
       !append(slot_tokens_.data(),slot_tokens_.size()*sizeof(int32_t),false))return false;
    for(size_t i=0;i<tx_->allocation_.stages().size();++i){const auto& stage=tx_->allocation_.stages()[i];
        auto& native=*stage.session;ConversationStateSizes sizes;
        if(!conversation_session_sizes(tx_->geometry_,native,sizes,error)||native.qsa_alloc<0||
           native.qsa_ord0<0||native.max_cells<0||cut().forwarded>uint64_t(native.max_cells)||
           !scalar(stage.device)||!scalar(native.layer_lo)||!scalar(native.layer_hi)||
           !scalar(native.qsa_ord0)||!scalar(native.qsa_alloc)||!scalar(native.gdn_alloc))return false;
        // The optional scope exists only for the READ pass: counting must not
        // create a CUDA scope, stream, event or driver allocation.
        const auto read_stage=[&]()->bool {
            if(!append(native.gdn_state,sizes.gdn,true,true)||
               (native.ple_hist&&!append(native.ple_hist,sizes.ple,true,true)))return false;
            if(native.ple_hist&&(!append(native.ple_prev,2*sizeof(int32_t),false,true)||
               !append(&native.ple_token,sizeof(int32_t),false,true)))return false;
            for(int64_t j=0;j<native.qsa_alloc;++j){const auto& q=native.qsa_states[native.qsa_ord0+j];
                std::array<uint64_t,5> extents{};
                if(!conversation_kv_part_sizes(q,tx_->geometry_,int64_t(cut().forwarded),true,extents,error)||
                   !scalar(q.kv_mode)||!scalar(q.kv_q4)||!scalar(q.kv_int8)||!scalar(q.kv_hybrid)||
                   !scalar(q.kv_rot)||!scalar(extents))return false;
                const bool host=q.kv_mode!=0;
                std::array<const void*,5> source{};
                if(q.kv_hybrid)source={host?q.host.k_q:q.k_q,host?q.host.v_q4:q.v_q4,
                    host?q.host.k_scale:q.k_scale,nullptr,q.idx_pooled};
                else if(q.kv_q4)source={host?q.host.k_q4:q.k_q4,host?q.host.v_q4:q.v_q4,nullptr,nullptr,q.idx_pooled};
                else if(q.kv_int8)source={host?q.host.k_q:q.k_q,host?q.host.v_q:q.v_q,
                    host?q.host.k_scale:q.k_scale,host?q.host.v_scale:q.v_scale,q.idx_pooled};
                else source={host?q.host.k_pool:q.k_pool,host?q.host.v_pool:q.v_pool,nullptr,nullptr,q.idx_pooled};
                for(size_t part=0;part<source.size();++part)
                    if(extents[part]>SIZE_MAX||!append(source[part],size_t(extents[part]),true,extents[part]!=0))return false;
                if(!append(q.idx_tail,sizes.tail,true,true)||!append(q.idx_dead,sizes.dead,true,true)||
                   !append(q.idx_block_pos,sizes.block_pos,true,true))return false;
            }
            uint64_t bank_bytes=0;
            return product(bank.cells_,3*sizeof(int32_t),bank_bytes)&&bank_bytes<=SIZE_MAX&&
                scalar(bank.content_generations_[bank_index])&&
                append(bank.banks_[i][bank_index],size_t(bank_bytes),true,true);
        };
        if(read){WindowDeviceScope device(stage.device);
            if(!device.ok()||cudaDeviceSynchronize()!=cudaSuccess)std::terminate();
            if(!read_stage()||cudaDeviceSynchronize()!=cudaSuccess||!device.finish())std::terminate();
        }else if(!read_stage())return false;
    }
    return true;
}

bool ServingRawSession::image_observe_peer(std::shared_ptr<OwnedImagePeerObservation>& out,std::string& error) {
#if !defined(__GLIBCXX__) || __GLIBCXX__!=20240904 || !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE!=13
    (void)out;error="image peer observation requires the pinned GNU profile";return false;
#else
    using W=OwnedImagePeerObservation;using A=W::Allocator<W>;
    using Control=std::_Sp_counted_ptr_inplace<W,A,__gnu_cxx::__default_lock_policy>;
    static_assert(std::is_same<typename std::allocator_traits<typename Control::__allocator_type>::value_type,Control>::value);
    if(out){error="image peer witness output already occupied";return false;}
    size_t count=0,parts=0;if(!image_peer_visit(false,nullptr,0,count,parts,error)||count>SIZE_MAX/2)return false;
    uint64_t quote=2*uint64_t(count);
    // Exact two arenas plus combined typed control. Fixed visitor/forwarding
    // objects are disjoint requested-profile addends, not an RSS/frame proof.
    constexpr uint64_t fixed=sizeof(Control)+alignof(Control)-1+sizeof(SuffixRawTransaction::VisionAdmission)+
        sizeof(W::Keepers)+4*sizeof(A)+4*sizeof(std::shared_ptr<W>)+
        2*sizeof(std::shared_ptr<const OwnedImageRequestOperation>)+2*sizeof(std::array<const void*,5>)+
        2*sizeof(std::array<uint64_t,5>)+2*sizeof(ConversationStateSizes)+2*sizeof(WindowDeviceScope)+
        32*sizeof(uint64_t)+16*sizeof(void*);
    if(!add(quote,fixed))return false;
    SuffixRawTransaction::VisionAdmission admission;
    const auto paid=tx_->admit_vision_host(*this,quote,admission);
    if(!paid){error=paid.detail;return false;}
    const W::Keepers keepers{admission.family_root,admission.local_setup,admission.credit};
    try {
        auto candidate=std::allocate_shared<W>(A(keepers),keepers);
        candidate->context_=image_operation();candidate->session_=this;
        candidate->occupancy_=slot_occupancy_incarnation();candidate->generation_=candidate->context_->slot_install_generation_;
        candidate->bytes_=count;candidate->parts_=parts;candidate->allocation_bytes_=count*2;
        candidate->arena_=candidate->allocator_.allocate(candidate->allocation_bytes_);
        // Strong aggregate is retained BEFORE any actual native read. A failed
        // drain never unwinds the credited arena behind uncertain transfers.
        tx_->pending_=candidate;
        if(!tx_->poll_all())std::terminate();
        size_t actual=0,tensors=0;
        if(!image_peer_visit(true,candidate->arena_,count,actual,tensors,error)||actual!=count||tensors!=parts||
           !tx_->poll_all())std::terminate();
        tx_->pending_.reset();out=std::move(candidate);return true;
    }catch(const std::bad_alloc&){if(tx_->pending_)std::terminate();error="paid image observation allocation refused";return false;}
#endif
}

bool ServingRawSession::image_compare_peer(const std::shared_ptr<OwnedImagePeerObservation>& witness,
    uint64_t& tensors,uint64_t& bytes,std::string& error) {
    tensors=bytes=0;
    if(!tx_||tx_->controller_!=std::this_thread::get_id()||!witness||witness->compared_||witness->session_!=this||tx_->pending_||
       witness->context_.get()!=image_operation().get()||witness->context_.owner_before(image_operation())||
       image_operation().owner_before(witness->context_)||witness->occupancy_!=slot_occupancy_incarnation()||
       witness->generation_!=witness->context_->slot_install_generation_){error="foreign or duplicate image observation";return false;}
    size_t count=0,parts=0;
    if(!image_peer_visit(false,nullptr,0,count,parts,error)||count!=witness->bytes_||parts!=witness->parts_)return false;
    witness->compared_=true; // one comparison attempt; never a reusable certificate
    tx_->pending_=witness;
    if(!tx_->poll_all())std::terminate();
    size_t actual=0,observed=0;
    if(!image_peer_visit(true,witness->arena_+count,count,actual,observed,error)||actual!=count||observed!=parts||
       !tx_->poll_all())std::terminate();
    tx_->pending_.reset();tensors=parts;bytes=count;
    if(std::memcmp(witness->arena_,witness->arena_+count,count)){
        error="image peer native/endpoint/bank byte preservation mismatch";return false;}
    return true;
}

bool ServingRawSession::image_install_bank(size_t bank,std::shared_ptr<const OwnedImageRequestOperation> context,
    std::string& error) {
    if(!tx_||!healthy()||bank>=3||tx_->controller_!=std::this_thread::get_id())return false;
    for(auto* v:tx_->verifiers_)if(v->owned_group_busy_){error="image bank still has a group borrower";return false;}
    if(!tx_->poll_all()){error="image bank borrower drain failed";return false;}
    auto banks=tx_->verifiers_.empty()?nullptr:tx_->verifiers_[0]->position_banks_;
    if(!banks||!banks->owner_){error="image bank has no paid startup owner";return false;}
    auto& owner=const_cast<OwnedVisionPositionBanks&>(*banks->owner_);
    if(owner.content_generations_[bank]==UINT64_MAX||owner.cells_!=max_tokens()+64||!owner.staging_||
       (context&&(!context->request_||context->request_->position_triples().size()!=owner.cells_*3))){
        error="image bank content generation/geometry exhausted";return false;
    }
    // Existing B host staging and the SAME three all-stage device banks are
    // already admitted. This operation neither allocates nor creates events.
    if(context)std::copy(context->request_->position_triples().begin(),context->request_->position_triples().end(),owner.staging_);
    else for(size_t i=0;i<owner.cells_;++i)for(size_t j=0;j<3;++j)owner.staging_[3*i+j]=int32_t(i);
    for(size_t i=0;i<owner.stages_;++i){
        WindowDeviceScope scope(owner.allocation_.stages()[i].device);
        if(!scope.ok()||cudaDeviceSynchronize()!=cudaSuccess||
           cudaMemcpy(owner.banks_[i][bank],owner.staging_,owner.cells_*3*sizeof(int32_t),cudaMemcpyHostToDevice)!=cudaSuccess||
           cudaDeviceSynchronize()!=cudaSuccess||!scope.finish())std::terminate();
    }
    ++owner.content_generations_[bank];owner.image_contexts_[bank]=context;
    if(context){auto& record=const_cast<OwnedImageRequestOperation&>(*context);
        if(bank==0)record.main_install_generation_=owner.content_generations_[bank];
        else record.slot_install_generation_=owner.content_generations_[bank];}
    return true;
}

bool ServingRawSession::image_reset(bool clear_context,std::string& error) {
    if(!tx_||!healthy()||(phase()!=Phase::drained&&phase()!=Phase::running)||
       tx_->cut_.id==UINT64_MAX||!tx_->poll_all())return false;
    const bool running=phase()==Phase::running;
    if(running&&!lifecycle_.allow(serving_raw::Lifecycle::Event::drain_work))return false;
    // Kind remains authoritative for the whole write/drain interval. Neither a
    // later successful sync nor an exception can turn uncertainty into text.
    if(!request_open()||(!running&&!tx_->begin_driver_work()))return false;
    if(!running)lifecycle_.apply(serving_raw::Lifecycle::Event::begin_work);
    if(slot_occupancy_)slot_occupancy_->running_.store(true,std::memory_order_release);
    for(const auto& native:tx_->allocation_.stages()){
        WindowDeviceScope scope(native.device);
        if(!scope.ok()||!session_zero_checked(*native.session,tx_->geometry_,nullptr,nullptr,error)||
           cudaDeviceSynchronize()!=cudaSuccess||!scope.finish())std::terminate();
    }
    const std::vector<int32_t> empty_history; // real history API; no element allocation
    StateCut empty;empty.id=tx_->cut_.id+1;empty.history=window_history_digest(empty_history);
    std::array<StateCut,8> cuts;cuts.fill(empty);
    if(!tx_->complete_driver_work(true,empty,std::span<const StateCut>(cuts.data(),tx_->allocation_.stages().size())))std::terminate();
    if(slot_occupancy_)slot_occupancy_->running_.store(false,std::memory_order_release);
    lifecycle_.set_observed_phase(raw_phase(phase()));lifecycle_.apply(serving_raw::Lifecycle::Event::drain_work);
    const size_t bank=slot_occupancy_?slot_occupancy_->slot()+1:0;
    if(!image_install_bank(bank,clear_context?nullptr:image_operation(),error))std::terminate();
    if(clear_context){
        if(slot_occupancy_)slot_occupancy_->image_context_.reset();
        set_image_operation({});
    }
    return true;
}

bool ServingRawSession::image_cancel_current(std::string& error) {
    using Op=OwnedImageRequestOperation;const auto op=image_operation();
    if(!op||slot_occupancy_||!tx_->image_exact()||!request_open()||!healthy()||
       tx_->controller_!=std::this_thread::get_id()||op->source_session_!=this||
       (phase()!=Phase::running&&phase()!=Phase::drained))return false;
    for(auto* p:op->prefills_)if(p&&!p->owned_image_retirement_ready(op.get()))std::terminate();
    op->source_callers_joined_=true;
    auto state=Op::Phase::unsealed;
    if(!op->phase_.compare_exchange_strong(state,Op::Phase::poisoned,std::memory_order_acq_rel))return false;
    // Real all-stage zero/drain/empty-cut and own-bank reset, not a caller's
    // cancellation flag or fabricated state certificate. Payment stays retained.
    if(!image_reset(true,error))std::terminate();
    for(auto* p:op->prefills_)if(p&&p->image_operation_.get()==op.get()){
        p->image_operation_.reset();p->image_generation_=0;
    }
    const auto directory=image_directory();
    if(directory)for(auto& prepared:directory->prepared_)if(prepared.get()==op.get())prepared.reset();
    return close_request(error);
}

bool ServingRawSession::image_retire_current(std::string& error) {
    using Op=OwnedImageRequestOperation;
    const auto op=image_operation();
    if(!op||!tx_->image_exact()||!request_open()||phase()!=Phase::drained){error="image retire";return false;}
    const auto state=op->phase_.load(std::memory_order_acquire);
    if(state!=Op::Phase::consumed&&state!=Op::Phase::poisoned){error="image unsealed";return false;}
    if(!slot_occupancy_){
        if(!op->source_callers_joined_){error="image callers";return false;}
        for(auto* p:op->prefills_)if(p&&!p->owned_image_retirement_ready(op.get()))std::terminate();
    }
    // Reset/certify only this actor and its bank while authority is still set.
    if(!image_reset(true,error))std::terminate();
    if(!slot_occupancy_){
        for(auto* p:op->prefills_)if(p&&p->image_operation_.get()==op.get()){
            p->image_operation_.reset();p->image_generation_=0;
        }
        const auto directory=image_directory();
        if(directory)for(auto& prepared:directory->prepared_)if(prepared.get()==op.get())prepared.reset();
    }
    return true;
}

bool ServingRawSession::image_bind_prefill(std::span<prefill::Prefill* const> prefills,std::string& error) {
    if(!image_authority_current(true,error)||!image_operation()->captured_||!image_operation()->producer_closed_||
       prefills.size()!=tx_->allocation_.stages().size()||phase()!=Phase::drained)return false;
    const auto op=image_operation();
    for(size_t i=0;i<prefills.size();++i){auto* p=prefills[i];
        // The real ring query validates its current device. Inspect each paid
        // stage on that stage, then restore the caller before publication.
        WindowDeviceScope device(tx_->allocation_.stages()[i].device);
        if(!device.ok()||!p||!p->owned_image_target(tx_->geometry_identity_,tx_->allocation_.stages()[i].session,
               tx_->allocation_.stages()[i].device)||p->position_banks_->stage_!=i||p->position_banks_->owner_->allocation_.stages()[i].session!=tx_->allocation_.stages()[i].session||!device.finish()){
            error="image prefill lacks its actual paid stage binding";return false;
        }
    }
    tx_->initialized_=2; // before first image mutation/exposure
    if(!image_reset(false,error))std::terminate();
    for(size_t i=0;i<prefills.size();++i){auto* p=prefills[i];op->prefills_[i]=p;
        p->image_operation_=op;p->image_generation_=op->main_install_generation_;}
    return true;
}


bool ServingRawSession::image_transfer_to(ServingRawSession& destination,const SnapshotDescriptor& metadata,std::string& error) {
    using Op=OwnedImageRequestOperation;
    if(!image_authority_current(true,error)||!tx_->image_exact()||!image_operation()->request_||
       !image_operation()->producer_closed_||!image_operation()->source_callers_joined_||
       phase()!=Phase::drained||metadata.key!=key()||metadata.cut!=cut()||
       metadata.cut.forwarded!=image_operation()->token_count_||metadata.cut.accepted!=metadata.cut.forwarded+1||
       metadata.cut.exact_frontier!=metadata.cut.forwarded||metadata.accepted_tokens.size()!=metadata.cut.accepted||
       metadata.pending_tokens.size()!=1||metadata.pending_tokens.front()!=metadata.accepted_tokens.back()||
       metadata.pending_tokens.front()<0||int64_t(metadata.pending_tokens.front())>=tx_->geometry_.n_vocab||
       !metadata.devices.empty()||metadata.steering_enabled||!bounded_metadata(metadata,tx_->max_tokens_)||
       window_history_digest(metadata.accepted_tokens)!=cut().history||
       !std::equal(image_operation()->request_->tokens().begin(),image_operation()->request_->tokens().end(),metadata.accepted_tokens.begin())||
       !destination.tx_||!destination.slot_occupancy_||!destination.healthy()||destination.request_open()||
       destination.alignment_known_||destination.slot_controller_bound_||destination.alignment_owner_credit_||
       destination.phase()!=Phase::drained||destination.cut().accepted||destination.cut().forwarded||
       destination.tx_->allocation_.stages().size()!=tx_->allocation_.stages().size()||
       metadata.accepted_tokens.size()>destination.max_tokens()||
       destination.slot_occupancy_incarnation()!=image_operation()->destination_occupancy_||
       destination.slot_occupancy_->token_!=image_operation()->destination_token_||destination.tx_->runtime_!=tx_->runtime_||
       destination.tx_->runtime_.owner_before(tx_->runtime_)||tx_->runtime_.owner_before(destination.tx_->runtime_)){
        error="image transfer lacks its canonical completed source/vacant destination";return false;
    }
    auto op=image_operation();
    if(op->phase_.load(std::memory_order_acquire)!=Op::Phase::unsealed){error="image completion already issued";return false;}
    if(!tx_->poll_all()||!destination.tx_->poll_all()){error="image transfer all-stage fence refused";return false;}
    try {
        op->completion_=metadata;op->pending_token_=metadata.pending_tokens.front();op->source_group_=cut().id;
        op->native_=tx_->allocation_; // real storage/lease, never the source actor/runtime
        const auto same_sizes=[&](const OwnedNativeSession& source,const OwnedNativeSession& target)->bool {
            const auto& from=*source.session;const auto& to=*target.session;
            if(source.device!=target.device||from.layer_lo!=to.layer_lo||from.layer_hi!=to.layer_hi||
               from.qsa_ord0!=to.qsa_ord0||from.qsa_alloc!=to.qsa_alloc||from.gdn_alloc!=to.gdn_alloc||
               (from.ple_hist!=nullptr)!=(to.ple_hist!=nullptr))return false;
            ConversationStateSizes left{},right{};
            if(!conversation_session_sizes(tx_->geometry_,from,left,error)||
               !conversation_session_sizes(tx_->geometry_,to,right,error)||left.gdn!=right.gdn||
               left.ple!=right.ple||left.tail!=right.tail||left.dead!=right.dead||left.block_pos!=right.block_pos)return false;
            for(int64_t j=0;j<from.qsa_alloc;++j){const auto& x=from.qsa_states[from.qsa_ord0+j];const auto& y=to.qsa_states[to.qsa_ord0+j];
                std::array<uint64_t,5> l{},r{};
                if(x.kv_mode!=y.kv_mode||(x.kv_mode!=0&&x.kv_mode!=1)||x.kv_hybrid!=y.kv_hybrid||
                   x.kv_int8!=y.kv_int8||x.kv_q4!=y.kv_q4||x.kv_rot!=y.kv_rot||
                   !conversation_kv_part_sizes(x,tx_->geometry_,int64_t(metadata.cut.forwarded),true,l,error)||
                   !conversation_kv_part_sizes(y,tx_->geometry_,int64_t(metadata.cut.forwarded),true,r,error)||l!=r)return false;
            }
            return true;
        };
        for(size_t i=0;i<op->native_.stages().size();++i){
            const auto& from=*op->native_.stages()[i].session;
            const auto& to=*destination.tx_->allocation_.stages()[i].session;
            if(from.qsa_alloc<0||size_t(from.qsa_alloc)!=tx_->initial_config_[i].qsa_formats.size()||
               to.qsa_alloc<0||size_t(to.qsa_alloc)!=destination.tx_->initial_config_[i].qsa_formats.size())return false;
            for(int64_t j=0;j<from.qsa_alloc;++j){
                const auto format=[](const QsaState& q){return (q.kv_hybrid?3:qsa_kv_format(q))+
                    (q.kv_rot&&!q.kv_q4&&!q.kv_hybrid?16:0);};
                if(format(from.qsa_states[from.qsa_ord0+j])!=tx_->initial_config_[i].qsa_formats[size_t(j)]||
                   format(to.qsa_states[to.qsa_ord0+j])!=destination.tx_->initial_config_[i].qsa_formats[size_t(j)]){
                    error="image native format changed after actual admission";return false;}
            }
            if(!same_sizes(op->native_.stages()[i],destination.tx_->allocation_.stages()[i])){
                error="image actual native pair geometry differs";return false;}
            op->ple_[i]={{from.ple_prev[0],from.ple_prev[1]},from.ple_token};
        }
        if(!op->native_.healthy()||!image_authority_current(true,error)||metadata.cut!=cut())return false;
        auto state=Op::Phase::unsealed;
        if(!op->phase_.compare_exchange_strong(state,Op::Phase::issued,std::memory_order_acq_rel)){
            error="duplicate image source completion";return false;}
        Op::Receipt receipt(op);
        if(!destination.open_request(error)){
            op->phase_.store(Op::Phase::poisoned,std::memory_order_release);return false;}
        if(!image_authority_current(true,error)||!destination.tx_->poll_all()||
           destination.slot_occupancy_incarnation()!=op->destination_occupancy_||
           destination.slot_occupancy_->token_!=op->destination_token_||metadata.cut!=cut()||
           op->source_group_!=cut().id||op->pending_token_!=metadata.pending_tokens.front()){
            op->phase_.store(Op::Phase::poisoned,std::memory_order_release);
            if(!destination.close_request(error))std::terminate();
            return false;
        }
        state=Op::Phase::issued;
        if(receipt.record_.get()!=image_operation().get()||receipt.record_.owner_before(image_operation())||
           image_operation().owner_before(receipt.record_)||
           !op->phase_.compare_exchange_strong(state,Op::Phase::consuming,std::memory_order_acq_rel))std::terminate();
        destination.set_image_operation(receipt.record_);
        destination.slot_occupancy_->image_context_=receipt.record_;
        if(!destination.image_reset(false,error))std::terminate();
        if(!destination.tx_->begin_driver_work())std::terminate();
        destination.lifecycle_.apply(serving_raw::Lifecycle::Event::begin_work);
        destination.slot_occupancy_->running_.store(true,std::memory_order_release);
        tx_->pending_=receipt.record_;destination.tx_->pending_=receipt.record_;
        for(size_t i=0;i<op->native_.stages().size();++i){const auto& stage=op->native_.stages()[i];
            const auto& from=*stage.session;auto& to=*destination.tx_->allocation_.stages()[i].session;
            WindowDeviceScope device(stage.device);
            const auto stream=tx_->verifiers_[i]->copy_; // actual already-paid owned verifier stream
            if(!device.ok()||cudaDeviceSynchronize()!=cudaSuccess)std::terminate();
            const auto copy=[&](void* target,const void* source,size_t bytes){
                if(!bytes)return true;
                return target&&source&&cudaMemcpyAsync(target,source,bytes,cudaMemcpyDefault,stream)==cudaSuccess;
            };
            ConversationStateSizes sizes{};
            if(!conversation_session_sizes(tx_->geometry_,from,sizes,error)||
               !copy(to.gdn_state,from.gdn_state,sizes.gdn)||
               (from.ple_hist&&!copy(to.ple_hist,from.ple_hist,sizes.ple)))std::terminate();
            const auto pools=[](const QsaState& x){
                const bool host=x.kv_mode!=0;
                if(x.kv_hybrid)return std::array<void*,5>{host?x.host.k_q:x.k_q,host?x.host.v_q4:x.v_q4,host?x.host.k_scale:x.k_scale,nullptr,x.idx_pooled};
                if(x.kv_q4)return std::array<void*,5>{host?x.host.k_q4:x.k_q4,host?x.host.v_q4:x.v_q4,nullptr,nullptr,x.idx_pooled};
                if(x.kv_int8)return std::array<void*,5>{host?x.host.k_q:x.k_q,host?x.host.v_q:x.v_q,host?x.host.k_scale:x.k_scale,host?x.host.v_scale:x.v_scale,x.idx_pooled};
                return std::array<void*,5>{host?x.host.k_pool:x.k_pool,host?x.host.v_pool:x.v_pool,nullptr,nullptr,x.idx_pooled};
            };
            for(int64_t j=0;j<from.qsa_alloc;++j){const auto& x=from.qsa_states[from.qsa_ord0+j];const auto& y=to.qsa_states[to.qsa_ord0+j];
                std::array<uint64_t,5> extents{};
                if(!conversation_kv_part_sizes(x,tx_->geometry_,int64_t(metadata.cut.forwarded),true,extents,error))std::terminate();
                const auto source=pools(x),target=pools(y);
                for(size_t part=0;part<source.size();++part)
                    if(extents[part]>SIZE_MAX||!copy(target[part],source[part],size_t(extents[part])))std::terminate();
                if(!copy(y.idx_tail,x.idx_tail,sizes.tail)||!copy(y.idx_dead,x.idx_dead,sizes.dead)||
                   !copy(y.idx_block_pos,x.idx_block_pos,sizes.block_pos))std::terminate();
            }
            // No restored checkpoint rewrites a pooled spare or substitutes a
            // token-derived PLE window. Preserve the complete actual native pair.
            to.ple_prev[0]=op->ple_[i].previous[0];to.ple_prev[1]=op->ple_[i].previous[1];to.ple_token=op->ple_[i].current;
            // Destination reset cleared its resident suffix and streamed map.
            // No actor can read it until these authoritative-pool copies drain;
            // subsequent mode1 resolution refills from that exact copied pool.
            if(cudaStreamSynchronize(stream)!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess||!device.finish())std::terminate();
        }
        if(!tx_->poll_all())std::terminate();
        StateCut transferred=metadata.cut;transferred.id=destination.cut().id+1;
        std::array<StateCut,8> cuts;cuts.fill(transferred);
        if(!destination.tx_->complete_driver_work(true,transferred,
            std::span<const StateCut>(cuts.data(),destination.tx_->allocation_.stages().size())))std::terminate();
        destination.slot_occupancy_->running_.store(false,std::memory_order_release);
        destination.lifecycle_.set_observed_phase(raw_phase(destination.phase()));
        destination.lifecycle_.apply(serving_raw::Lifecycle::Event::drain_work);
        destination.tx_->pending_.reset();tx_->pending_.reset();
        destination.slot_tokens_.assign(metadata.accepted_tokens.begin(),metadata.accepted_tokens.end());
        destination.slot_current_key_=destination.key();
        if(!image_authority_current(true,error)){
            op->phase_.store(Op::Phase::poisoned,std::memory_order_release);
            if(!destination.close_request(error))std::terminate();
            return false;
        }
        state=Op::Phase::consuming;
        if(!op->phase_.compare_exchange_strong(state,Op::Phase::consumed,std::memory_order_acq_rel))std::terminate();
        // All actual source/destination borrowers ended. Drop only the private
        // native view and metadata; operation/control payment remains until its
        // real last-free, and immutable image context stays with the live slot.
        op->native_=NativeSessionAllocation{};op->completion_=SnapshotDescriptor{};
        return true;
    }catch(...){
        if(op->phase_.load(std::memory_order_acquire)==Op::Phase::consuming)std::terminate();
        op->phase_.store(Op::Phase::poisoned,std::memory_order_release);
        error="image private transfer preparation refused";return false;
    }
}

SuffixRawResult ServingRawSession::prepare_vision_request(int embeddings_fd,
    std::span<const int64_t> input,uint64_t full_output_budget,
    std::shared_ptr<const VisionRequestOwner>& out) {
#if !defined(__linux__) || !defined(__GLIBCXX__) || __GLIBCXX__!=20240904 || !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE!=13
    return {SuffixRawStatus::invalid,"vision profile",{}};
#else
    using Owner=VisionRequestOwner;
    using Alloc=Owner::HostAllocator<Owner>;
    using Control=std::_Sp_counted_ptr_inplace<Owner,Alloc,__gnu_cxx::__default_lock_policy>;
    using ControlAlloc=Control::__allocator_type;
    using ControlValue=std::allocator_traits<ControlAlloc>::value_type;
    static_assert(std::is_base_of<std::__shared_ptr<Owner>,std::shared_ptr<Owner>>::value);
    static_assert(std::is_same<ControlAlloc,Owner::HostAllocator<Control>>::value&&
                  std::is_same<ControlValue,Control>::value);
    static_assert(sizeof(Alloc)==sizeof(Owner::Keepers)&&sizeof(ControlAlloc)==sizeof(Owner::Keepers));
    static_assert(std::is_nothrow_copy_constructible<Alloc>::value&&
                  std::is_nothrow_constructible<ControlAlloc,const Alloc&>::value);
    static_assert(sizeof(Owner::Keepers)==3*sizeof(std::shared_ptr<const void>));
    static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);
    if(std::endian::native!=std::endian::little)return {SuffixRawStatus::invalid,"vision profile",{}};
    const auto allowed=[&]() {
        return tx_&&tx_->initialized_&&tx_->controller_==std::this_thread::get_id()&&healthy()&&
            lifecycle_.request_open()&&!lifecycle_.private_window_open()&&
            (tx_->phase_==SuffixRawTransaction::Phase::idle||tx_->phase_==SuffixRawTransaction::Phase::drained);
    };
    if(!allowed()||embeddings_fd<0)return {SuffixRawStatus::invalid,"vision request",{}};
    const uint64_t context=tx_->max_tokens_;
    const auto width=tx_->geometry_.n_embd,vocabulary=tx_->geometry_.n_vocab;
    constexpr int64_t image_pad=248056; // Existing Qwen SVE1/GENI image token.
    if(!input.size()||!full_output_budget||full_output_budget>context||input.size()>context-full_output_budget||
       context>uint64_t(INT32_MAX)-64||width<=0||width>INT32_MAX||vocabulary<=image_pad||vocabulary>INT32_MAX)
        return {SuffixRawStatus::invalid,"vision bounds",{}};
    for(auto token:input)if(token<0||token>=vocabulary)return {SuffixRawStatus::invalid,"vision token",{}};
    if(input.back()==image_pad)return {SuffixRawStatus::invalid,"vision tail",{}};
    struct stat before{},after{};
    if(::fstat(embeddings_fd,&before)!=0||!S_ISREG(before.st_mode)||before.st_size<20)
        return {SuffixRawStatus::invalid,"vision file",{}};
    const uint64_t file_bytes=uint64_t(before.st_size);
    uint64_t per_token=0,file_bound=0,row_bytes=0,row_stride_bytes=0;
    if(!product(uint64_t(width),sizeof(float),row_bytes))return invalid("vision width");
    per_token=row_bytes;row_stride_bytes=row_bytes;
    if(!add(row_stride_bytes,(64-row_stride_bytes%64)%64)||!add(per_token,20)||
       !product(uint64_t(input.size()),per_token,file_bound)||file_bytes>file_bound||file_bytes>SIZE_MAX||file_bytes%4)
        return {SuffixRawStatus::invalid,"vision extent",{}};
    Owner::Layout layout;
    layout.file_bytes=size_t(file_bytes);layout.token_count=input.size();layout.width=size_t(width);
    layout.row_stride=size_t(row_stride_bytes/sizeof(float));
    layout.cells=size_t(context+64);layout.output_budget=full_output_budget;
    uint64_t cursor=0;
    const auto append=[&](uint64_t count,uint64_t size,uint64_t alignment,size_t& offset) {
        uint64_t extent=0;
        if(!product(count,size,extent)||!add(cursor,(alignment-cursor%alignment)%alignment)||cursor>SIZE_MAX)return false;
        offset=size_t(cursor);return add(cursor,extent)&&cursor<=SIZE_MAX;
    };
    // F bounds the number of decoded rows before parsing. Each row is separately
    // 64-byte aligned, including geometries whose width is not a multiple of16.
    const uint64_t max_rows=std::min<uint64_t>(input.size(),file_bytes/row_bytes);
    if(!append(file_bytes,1,1,layout.file)||!append(max_rows,row_stride_bytes,64,layout.decoded)||
       !append(input.size(),sizeof(int64_t),alignof(int64_t),layout.tokens)||
       !append(input.size(),sizeof(const float*),alignof(const float*),layout.pointers)||
       !append(input.size(),sizeof(Owner::Grid),alignof(Owner::Grid),layout.grids)||
       !append(layout.cells,3*sizeof(int32_t),alignof(int32_t),layout.positions)||!add(cursor,63))
        return {SuffixRawStatus::invalid,"vision quote",{}};
    if(cursor>SIZE_MAX)return invalid("vision quote");
    layout.allocation_bytes=size_t(cursor); // includes63 bytes for the aligned base
    // GNU's actual allocate_shared rebound control contains BOTH the payload and
    // its keeper-bearing allocator. Quote it once, including actual alignment.
    // Four transient allocator states bound the normal GNU13 construction path;
    // admission/keeper locals and two stat/header/parser workspaces coexist too.
    uint64_t transient=sizeof(SuffixRawTransaction::VisionAdmission)+sizeof(Owner::Keepers)+
        4*sizeof(Alloc)+2*sizeof(struct stat)+2*sizeof(Owner::Layout)+2*sizeof(Owner::Identity)+
        5*sizeof(int32_t)+sizeof(float)+16*sizeof(uint64_t);
    layout.host_bytes=layout.allocation_bytes;
    if(!add(layout.host_bytes,sizeof(ControlValue))||!add(layout.host_bytes,alignof(ControlValue)-1)||
       !add(layout.host_bytes,transient)||layout.host_bytes>SIZE_MAX)
        return {SuffixRawStatus::invalid,"vision quote",{}};
    try {
        SuffixRawTransaction::VisionAdmission admission;
        auto admitted=tx_->admit_vision_host(*this,layout.host_bytes,admission);if(!admitted)return admitted;
        const Owner::Keepers keepers{admission.family_root,admission.local_setup,admission.credit};
        Owner::Identity identity;
        identity.source_session=this;identity.runtime=tx_->runtime_.get();identity.geometry=tx_->geometry_identity_;
        identity.engine=tx_->identity_.engine_generation;identity.window=tx_->identity_.window_generation;
        identity.occupancy=slot_occupancy_incarnation();identity.model=tx_->identity_.model_profile;
        identity.topology=tx_->identity_.topology;identity.preparation_cut=tx_->cut_;
        const Alloc allocator(keepers);
        auto owner=std::allocate_shared<Owner>(allocator,keepers,layout,identity);
        auto* captured=owner->at<std::byte>(layout.file);
        auto* tokens=owner->at<int64_t>(layout.tokens);
        auto* pointers=owner->at<const float*>(layout.pointers);
        // Freeze and validate our own token copy before the potentially blocking
        // file capture; no external mutable token vector is retained or reread.
        for(size_t i=0;i<input.size();++i){
            std::construct_at(tokens+i,input[i]);std::construct_at(pointers+i,nullptr);
            if(tokens[i]<0||tokens[i]>=vocabulary)return invalid("vision token");
        }
        if(tokens[input.size()-1]==image_pad)return invalid("vision tail");
        size_t read=0;
        while(read<layout.file_bytes) {
            const size_t amount=std::min<size_t>(layout.file_bytes-read,1U<<20);
            const auto got=::pread(embeddings_fd,captured+read,amount,off_t(read));
            if(got<0&&errno==EINTR)continue;
            if(got<=0)return {SuffixRawStatus::invalid,"vision read",{}};
            read+=size_t(got);
        }
        std::byte extra{};ssize_t more;
        do{more=::pread(embeddings_fd,&extra,1,before.st_size);}while(more<0&&errno==EINTR);
        if(more!=0||::fstat(embeddings_fd,&after)!=0||before.st_dev!=after.st_dev||before.st_ino!=after.st_ino||
           before.st_size!=after.st_size||before.st_mtim.tv_sec!=after.st_mtim.tv_sec||
           before.st_mtim.tv_nsec!=after.st_mtim.tv_nsec||before.st_ctim.tv_sec!=after.st_ctim.tv_sec||
           before.st_ctim.tv_nsec!=after.st_ctim.tv_nsec)return {SuffixRawStatus::invalid,"vision changed",{}};
        auto* grids=owner->at<Owner::Grid>(layout.grids);
        auto* decoded=owner->at<float>(layout.decoded);
        auto* positions=owner->at<int32_t>(layout.positions);
        for(size_t i=0;i<layout.cells;++i)for(size_t j=0;j<3;++j)std::construct_at(positions+3*i+j,int32_t(i));
        size_t offset=0,images=0,floats=0,rows=0;
        while(offset<layout.file_bytes) {
            if(layout.file_bytes-offset<20||images>=input.size())return {SuffixRawStatus::invalid,"vision header",{}};
            int32_t h[5];std::memcpy(h,captured+offset,20);offset+=20;
            uint64_t count=0,bytes=0,area=0;
            if(h[0]!=0x31455653||h[1]<=0||h[2]<=0||h[3]<=0||h[4]!=width||
               !product(uint64_t(h[2]),uint64_t(h[3]),area)||area!=uint64_t(h[1])||
               uint64_t(h[1])>input.size()-rows||uint64_t(h[1])>max_rows-rows||!product(uint64_t(h[1]),uint64_t(width),count)||
               !product(count,sizeof(float),bytes)||bytes>layout.file_bytes-offset||count>(file_bytes/4)-floats)
                return {SuffixRawStatus::invalid,"vision header",{}};
            std::construct_at(grids+images,Owner::Grid{0,rows,size_t(h[1]),h[2],h[3]});
            for(size_t j=0;j<size_t(count);++j){
                float value;std::memcpy(&value,captured+offset+j*4,4);
                if(!std::isfinite(value))return {SuffixRawStatus::invalid,"vision finite",{}};
                std::construct_at(decoded+(rows+j/layout.width)*layout.row_stride+j%layout.width,value);
            }
            offset+=size_t(bytes);floats+=size_t(count);rows+=size_t(h[1]);++images;
        }
        if(!images)return {SuffixRawStatus::invalid,"vision empty",{}};
        size_t cell=0,image=0;uint64_t position=0;
        const auto put=[&](size_t c,uint64_t t,uint64_t h,uint64_t w) {
            if(c>=layout.cells||t>INT32_MAX||h>INT32_MAX||w>INT32_MAX)return false;
            positions[3*c]=int32_t(t);positions[3*c+1]=int32_t(h);positions[3*c+2]=int32_t(w);return true;
        };
        while(cell<input.size()) {
            if(tokens[cell]!=image_pad){if(!put(cell,position,position,position))return invalid("vision pos");++cell;++position;continue;}
            if(image>=images)return invalid("vision span");
            auto& grid=grids[image++];grid.token_begin=cell;
            if(grid.rows>input.size()-cell)return invalid("vision span");
            for(size_t j=0;j<grid.rows;++j) {
                if(tokens[cell+j]!=image_pad||!put(cell+j,position,position+j/size_t(grid.nx),position+j%size_t(grid.nx)))
                    return invalid("vision span");
                pointers[cell+j]=decoded+(grid.row_begin+j)*layout.row_stride;
            }
            cell+=grid.rows;position+=uint64_t(std::max(grid.nx,grid.ny));
        }
        if(image!=images)return invalid("vision span");
        for(size_t c=input.size();c<layout.cells;++c)if(!put(c,position+c-input.size(),position+c-input.size(),position+c-input.size()))
            return invalid("vision pos");
        if(!allowed()||slot_occupancy_incarnation()!=identity.occupancy||tx_->cut_!=identity.preparation_cut)
            return invalid("vision stale");
        owner->image_count_=images;owner->decoded_floats_=floats;
        out=std::move(owner);return {SuffixRawStatus::ok,{},{}};
    } catch(const std::bad_alloc&) {return {SuffixRawStatus::budget_refused,"vision alloc",{}};}
      catch(...) {return {SuffixRawStatus::invalid,"vision input",{}};}
#endif
}

} // namespace strata::program
