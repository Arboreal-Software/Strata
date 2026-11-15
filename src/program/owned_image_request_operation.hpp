#pragma once
#include "owned_vision_request.hpp"
#include "owned_admission_cancel.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/window_native.hpp"
#include "strata/core/native_batch_slot_binding.hpp"
#include <atomic>
#include <mutex>

namespace strata::prefill {class Prefill;}
namespace strata::core {class Verifier;}
namespace strata::program {
class ServingThreadBridge;
class OwnedImageRequestDriver;

// One source-session record, never a caller-selectable snapshot certificate.
// The driver is defined only at the real stdin/reactor composition in generate.
// No source actor, DriverWindowRuntime, root, lease, or ordinary credit escapes.
class OwnedImageRequestOperation final {
public:
    OwnedImageRequestOperation(const OwnedImageRequestOperation&)=delete;
    OwnedImageRequestOperation& operator=(const OwnedImageRequestOperation&)=delete;
    ~OwnedImageRequestOperation() noexcept {
        // A missing producer-close proof may still leave the paid transport
        // alive in the other process. Never unwind that payment on an EOF guess.
        if(transport_published_&&!producer_closed_)std::terminate();
        if(phase_.load(std::memory_order_acquire)==Phase::consuming)std::terminate();
    }
private:
    friend class ServingRawSession;
    friend class SuffixRawTransaction;
    friend class OwnedImageRequestDriver;
    friend class OwnedVisionPositionBanks;
    friend class core::Verifier;
    friend class prefill::Prefill;
    template<class>friend class VisionRequestOwner::HostAllocator;
    enum class Phase:uint8_t {unsealed,issued,consuming,consumed,poisoned};
    // Two concrete producer helpers may be active, one per physical admission.
    // Paid once by startup, before those frames/preflight objects can exist;
    // NOT borrowed from per-request F or from an unused payload/bank allowance.
    // Separate from every inherited control term: two structseq allocations
    // (base<=128, <=32 pointer fields, GC/alignment<=64), <=32 numeric
    // children each (<=80), plus six fill_time/arithmetic numeric temporaries.
    // Python checks the representation before stat allocation; never reuse
    // the old envelope as nominal slack. Both producer copies are prepaid.
    static constexpr uint64_t producer_source_version_bytes=
        2*(128+32*sizeof(void*)+64+32*80)+6*80;
    // Three fresh native-int signal sets: object/GC<=256, combined old/new
    // resize tables<=256 entries, <=64 numeric members. No enum wrappers.
    // One synchronous frame, owner buffers/tuples, siginfo, native syscall
    // stack controls, eight scalar temporaries and bounded failure controls.
    static constexpr uint64_t producer_source_readlease_bytes=
        3*(256+256*2*sizeof(void*)+64*80)+(1024+2*64*sizeof(void*))+
        3*80+3*(80+3*80)+(128+7*sizeof(void*)+64+7*80)+
        (2*128+128+128+128+16+2*8+8+1024)+8*80+2*8*128;
    // Cache-free owner encoding: eight-item and call tuples/GC, five bounded
    // scalar intermediates, and typed native conversion-frame word extents.
    // Owner/echo buffers remain in the unchanged readlease account above.
    static constexpr uint64_t producer_source_owner_encoding_bytes=128+80+5*80+(16+8+4+8+8+16)*8;
    static constexpr uint64_t producer_control_bytes=
        12*(1024+64*sizeof(void*))+256*64+8*(4096+128)+2*4096+
        producer_source_version_bytes+producer_source_readlease_bytes+producer_source_owner_encoding_bytes;
    // Actual allocator keepers precede payload AND survive the weak/control free.
    using Keepers=VisionRequestOwner::Keepers;
    template<class T>using Allocator=VisionRequestOwner::HostAllocator<T>;
    Keepers keepers_;
    std::shared_ptr<const void> reactor_owner_;
    std::shared_ptr<const VisionRequestOwner> request_;
    // The paired payload is the real allocation view at its sealed source cut,
    // not a second host snapshot, generic descriptor export or actor back-edge.
    // Its newly copied vector/function controls are priced before construction.
    core::NativeSessionAllocation native_;
    std::array<core::NativePleTokens,8> ple_{};
    core::SnapshotDescriptor completion_;
    std::string command_; // private same-ticket delivery, never a state/authority payload
    std::atomic<Phase> phase_{Phase::unsealed};
    const OwnedAdmissionCancel* admissions_=nullptr;
    std::mutex* admission_mutex_=nullptr;
    const ServingThreadBridge* bridge_=nullptr;
    const void* source_session_=nullptr;
    const void* runtime_=nullptr;
    uint64_t source_window_=0,request_key_=0,admission_incarnation_=0;
    uint64_t engine_=0,foreground_=0,foreground_revision_=0;
    uint64_t source_group_=0,destination_occupancy_=0;
    core::NativeBatchSlotToken destination_token_{};
    uint64_t full_output_budget_=0,file_bytes_=0,token_count_=0,deadline_ms_=0;
    uint64_t main_install_generation_=0,slot_install_generation_=0;
    std::array<prefill::Prefill*,8> prefills_{}; // observed bindings, never actor owners
    bool source_callers_joined_=false;
    int32_t pending_token_=-1;
    bool transport_published_=false,captured_=false,producer_closed_=false;
    explicit OwnedImageRequestOperation(const Keepers& keepers):keepers_(keepers){}

    // Only the source completion issuer can construct a receipt. It owns the
    // SAME record and paired native payload, never an independently reboxed seal.
    class Receipt final {
        friend class ServingRawSession;
        std::shared_ptr<OwnedImageRequestOperation> record_;
        explicit Receipt(std::shared_ptr<OwnedImageRequestOperation> record):record_(std::move(record)){}
    public:
        Receipt(const Receipt&)=delete;
        Receipt& operator=(const Receipt&)=delete;
        Receipt(Receipt&&)=default;
        Receipt& operator=(Receipt&&)=delete;
    };
};
// A single-use observation, deliberately not a native snapshot. It has no
// public payload, descriptor, reader, import/restore method, issuer or receipt.
// Only the actual peer can fill and compare the private byte arena. Both full
// observations are prepaid together; its typed allocator retains roots through
// arena free AND weak/control-block free, including on another thread.
class OwnedImagePeerObservation final {
public:
    OwnedImagePeerObservation(const OwnedImagePeerObservation&)=delete;
    OwnedImagePeerObservation& operator=(const OwnedImagePeerObservation&)=delete;
    ~OwnedImagePeerObservation() noexcept {
        if(arena_)allocator_.deallocate(arena_,allocation_bytes_);
    }
private:
    friend class ServingRawSession;
    friend class OwnedImageRequestDriver;
    template<class>friend class VisionRequestOwner::HostAllocator;
    using Keepers=VisionRequestOwner::Keepers;
    template<class T>using Allocator=VisionRequestOwner::HostAllocator<T>;
    Allocator<std::byte> allocator_;
    std::shared_ptr<const OwnedImageRequestOperation> context_;
    const void* session_=nullptr;
    uint64_t occupancy_=0,generation_=0;
    size_t bytes_=0,parts_=0,allocation_bytes_=0;
    std::byte* arena_=nullptr;
    bool compared_=false;
    explicit OwnedImagePeerObservation(const Keepers& keepers):allocator_(keepers){}
};

// Fixed, source-session-owned preparation slots. Waiting for the producer
// holds resource payment and the original admission ticket, not MAIN's work
// fence. This directory has no actor/runtime/receipt ownership back-edge.
class OwnedImageSessionRecords final {
    friend class ServingRawSession;
    friend class OwnedImageRequestDriver;
    template<class>friend class VisionRequestOwner::HostAllocator;
    std::shared_ptr<const void> reactor_;
    const OwnedAdmissionCancel* admissions_=nullptr;
    std::mutex* mutex_=nullptr;
    const ServingThreadBridge* bridge_=nullptr;
    std::array<std::shared_ptr<OwnedImageRequestOperation>,2> prepared_{};
    std::shared_ptr<OwnedImageRequestOperation> current_;
    OwnedImageSessionRecords()=default;
};
} // namespace strata::program
