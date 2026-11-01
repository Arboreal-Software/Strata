#pragma once

#include "strata/core/serving_raw_contracts.hpp"
#include "strata/core/window_allocator.hpp"
#include "strata/core/window_native.hpp"
#include "strata/core/window_suffix.hpp"
#include <span>
#include "strata/core/native_batch_slot_binding.hpp"
#include "strata/core/native_batch_slot_occupancy.hpp"
#include "strata/core/native_batch_group_completion.hpp"
#include <thread>
#include <cstdio>

namespace strata::core { class Verifier; class ResidentCopyFault; }
namespace strata::program {
// Defined only by the standalone generate entry. This is the trusted controller
// boundary, not a public way to assert that arbitrary host producers joined.
class SuffixExperimentDriver;
namespace resident_repair_detail { class ResidentRepairController; }
// Serving-side owner of a private RAW transaction (P12 serving suffix reuse). Trusted exactly as the standalone
// driver is: it is the only path allowed to bind a transaction to a serving session allocation, and it takes real
// ownership of the runtime, verifiers, budget and host credits. See ServingRawSession below.
class ServingRawSession;
// The serving composition of the qualified import+repair operation (defined where the concrete serving runtime
// lives). It is a trusted friend of the resident repair controller, exactly as the standalone driver is.
class ServingSuffixRepairOperation;
enum class SuffixRawStatus { ok, invalid, budget_refused, poisoned };
struct SuffixRawResult {
    SuffixRawStatus status = SuffixRawStatus::invalid;
    std::string detail;
    // Failure payloads and actual runtime stay owned even if the caller unwinds.
    // A poisoned transaction also failstops on destruction: there is no public
    // "teardown happened" boolean capable of releasing an uncertain transfer.
    std::shared_ptr<const void> retained_failure;
    explicit operator bool() const { return status == SuffixRawStatus::ok; }
};

class SuffixRawTransaction final {
public:
    ~SuffixRawTransaction();
    SuffixRawTransaction(const SuffixRawTransaction&) = delete;
    SuffixRawTransaction& operator=(const SuffixRawTransaction&) = delete;
private:
    friend class SuffixExperimentDriver;
    friend class resident_repair_detail::ResidentRepairController;
    friend class ServingRawSession;
    SuffixRawResult capture_empty_oracle(const core::SnapshotDescriptor&,core::BundleHandle&);
    SuffixRawResult restore_empty_oracle(const core::BundleHandle&,const core::WindowKey&,core::ResidentCopyFault* = nullptr);
    std::vector<core::NativeBatchSlotInspection> slot_bindings_;
    std::shared_ptr<const core::NativeBatchSlotOccupancy> slot_occupancy_;
    struct Storage;
    struct HostCap;
    struct HostCredit;
    std::shared_ptr<const HostCredit> slot_host_root_credit_;
    // Observation only; caller is the same drained concrete driver/controller.
    struct HostUsage { uint64_t limit=0,live=0;unsigned claims=0; };
    bool observe_host_usage(HostUsage&) const noexcept;
    enum class Phase { idle, running, drained, capturing, restoring, poisoned };
    // Allocation is the actual owned allocator result. Parent purpose/group are
    // checked here; reserve(private_attachment,parent) checks the SAME ledger.
    // Runtime owns verifier/dispatch/model/PLE/source/pool callback targets too.
    SuffixRawTransaction(const core::ModelGeometry&, const core::WindowKey& runtime_identity, core::NativeSessionAllocation,
        std::shared_ptr<const void> runtime, std::vector<core::Verifier*>,
        core::WindowBudget&, uint64_t max_tokens, uint64_t max_payload_bytes,
        uint64_t aggregate_host_limit, bool steering_enabled);
    SuffixRawResult initialize();
    // A resident private gang shares the live controller's aggregate host cap;
    // its actual allocation and setup retain their own global reservations.
    SuffixRawResult initialize_resident_shared_host(const SuffixRawTransaction&);
    SuffixRawResult initialize_slot_shared_host(const SuffixRawTransaction&);
    // Driver must call BEFORE any forward OR reset attempt, then join its actual
    // launch/adaptation/foreground/doorbell callers before complete_driver_work.
    // Destruction while running is deliberately fatal, including exceptions.
    // Driver catches failure/exception and failstops BEFORE external input or
    // logits buffers unwind; the destructor is not their lifetime certificate.
    SuffixRawResult begin_driver_work();
    SuffixRawResult complete_driver_work(bool operation_succeeded,
        const core::StateCut&, std::span<const core::StateCut> actual_stage_cuts);
    SuffixRawResult capture(const core::SnapshotDescriptor&, core::BundleHandle&);
    SuffixRawResult restore(const core::BundleHandle&, const core::WindowKey&,core::ResidentCopyFault* fault=nullptr);
    // Main obtains these BEFORE allocating introduced input/controller/logits.
    // Declare the returned owner BEFORE its buffers so they die first. No
    // quota is released merely because a call returned successfully.
    SuffixRawResult admit_controller_host(uint64_t, std::shared_ptr<const void>&);
    // Trusted resident controller reserves explicit setup + separately bounded
    // platform overhead BEFORE constructing private consumers. Host (including
    // pinned subset) shares this transaction's aggregate H; all resources charge
    // its real global ledger once. Credit alone is not execution/completion or
    // proof that an external CUDA runtime obeys the proposed overhead envelope.
    SuffixRawResult admit_resident_setup(const core::WindowResources&,
        std::shared_ptr<const void>&);
    // The gate requires a genuine exclusive foreground metadata lease. Its
    // separate aggregate-H claim must be retained by the gate through all cuts.
    SuffixRawResult admit_completion_gate_metadata(
        std::shared_ptr<const core::ReservationLease>&,
        std::shared_ptr<const void>& aggregate_credit);
    // Producer already reserves its exact quote in the REAL global ledger.
    // This is aggregate-cap-only, never a second global charge. Main retains
    // it with PreparedNativeSuffix through experiment end; no bundle export.
    SuffixRawResult claim_external_splice(uint64_t, std::shared_ptr<const void>&);
    SuffixRawResult poll_all();
    SuffixRawResult poison(const char*);
    bool resolve(std::vector<core::NativeWindowStage>&, std::string&) const;
    bool quote(const core::WindowStateLayout&, uint64_t copies, uint64_t&) const;
    bool identity_matches(const core::WindowKey&) const;
    SuffixRawResult reserve_payload(uint64_t, std::shared_ptr<Storage>&);
    SuffixRawResult admit_host(uint64_t, bool global_charge, std::shared_ptr<HostCredit>&);
    // Keep setup admission until ALL controller members have been destroyed,
    // including the constructor-transferred verifier vector and identity.
    std::shared_ptr<HostCredit> setup_credit_;
    core::ModelGeometry geometry_; // immutable shape snapshot, not pointer provenance
    // Original authoritative geometry object. The retained runtime owns its lifetime;
    // a value-equal transaction copy must never become a second geometry authority.
    const core::ModelGeometry* const geometry_identity_;
    core::WindowKey identity_;
    core::NativeSessionAllocation allocation_;
    std::shared_ptr<const void> runtime_;
    std::vector<core::Verifier*> verifiers_;
    core::WindowBudget& budget_;
    uint64_t max_tokens_, max_bytes_, host_limit_;
    Phase phase_ = Phase::idle;
    core::StateCut cut_{};
    std::shared_ptr<Storage> pending_;
    std::thread::id controller_ = std::this_thread::get_id();
    bool initialized_ = false;
    bool steering_enabled_;
    std::vector<core::NativeWindowStage> initial_config_;
    std::shared_ptr<HostCap> host_cap_;
};

/// A private RAW transaction bound to an OWNED serving session (`--owned-session-lifetime`, whose allocation is
/// reserved with `ReservationPurpose::private_work` in src/core/window_allocator.cpp:317, which is what
/// SuffixRawTransaction::initialize() requires at suffix_raw_runtime.cpp:162-166).
///
/// What this object is for: the serving path must reach the qualified capture/restore (and, next, repair) machinery
/// under the SAME ownership and phase discipline the standalone driver proved - not through a borrowed fake owner
/// and not by relabeling ids. It is the seam the repair operation is built on.
///
/// Lifetime contract, deliberately strict: `SuffixRawTransaction` is written for a disposable process and
/// failstops when destroyed over uncertain work (suffix_raw_runtime.cpp:96-98). A serving process that reaches an
/// uncertain RAW state therefore does NOT unwind and keep serving: it failstops, exactly as it does for a CUDA
/// fault, and the supervisor restores the preserved container. Continuing on possibly corrupt state is forbidden.
class ServingRawSession final {
public:
    /// Opens (and initializes) the transaction. `runtime_owner` must be the shared_ptr that actually owns the
    /// runtime whose `allocation` is being bound, so the transaction keeps it alive; `verifiers` must outlive the
    /// session. Refuses (returns null with a reason) on an unhealthy allocation, missing/empty verifier set, a
    /// zero cap, or any admission failure - nothing is mutated before those checks pass.
    static std::shared_ptr<ServingRawSession> open(const core::ModelGeometry& geometry,
                                                   core::NativeSessionAllocation allocation,
                                                   std::shared_ptr<const void> runtime_owner,
                                                   std::vector<core::Verifier*> verifiers,
                                                   core::WindowBudget& budget, uint64_t max_tokens,
                                                   uint64_t host_cap_bytes, std::string& error);
    // Experimental owned slot factory. It resets the actual complete slot gang
    // under checked drains, then publishes a new conversation occupancy.
    static std::shared_ptr<ServingRawSession> open_owned_slot(const core::ModelGeometry&,
        core::NativeSessionAllocation,std::shared_ptr<const void> runtime,
        std::vector<core::Verifier*>,std::vector<core::NativeBatchSlotInspection>,
        core::WindowBudget&,uint64_t max_tokens,ServingRawSession& aggregate_owner,std::string& error);
    bool install_owned_slot_image(const core::BundleHandle&,const core::WindowKey& expected_source,
        bool edited_prefix,std::string& error);
    bool transfer_owned_slot_from(ServingRawSession& source,const core::SnapshotDescriptor&,std::string& error);
    SuffixRawResult capture_owned_slot(core::BundleHandle&);
    bool queue_owned_slot_input(int32_t token,std::string& error);
    bool drain_owned_slot_group(const core::NativeBatchGroupCompletion&,std::FILE* output,std::string& error);
    bool retire_owned_slot(std::string& error);
    uint64_t slot_occupancy_incarnation()const noexcept{return slot_occupancy_?slot_occupancy_->incarnation():0;}
    ~ServingRawSession();
    ServingRawSession(const ServingRawSession&) = delete;
    ServingRawSession& operator=(const ServingRawSession&) = delete;

    /// Durable terminal state of this adapter: true once the request was abandoned or the transaction reported a
    /// poison. Nothing reopens the fence, and healthy() can never mask it (2026-10-29 review fix).
    bool terminal() const noexcept;
    bool healthy() const noexcept;
    /// OBSERVED, never invented: the transaction's real phase. RAW capture/restore are legal only while it is
    /// drained (suffix_raw_runtime.cpp:361-365, :409-411); begin only from idle/drained (:214-216); completion
    /// only while running (:232). A serving request outlives a single RAW operation, so request lifetime
    /// (`in_flight_`) and the work phase are deliberately separate facts.
    enum class Phase : uint8_t { absent, idle, running, drained, capturing, restoring, poisoned };
    Phase phase() const noexcept;
    /// The lifecycle predicates live in strata/core/serving_raw_contracts.hpp: one definition, used by the server
    /// and by tests/core/serving_raw_contracts, never re-stated in either place.
    static core::serving_raw::Phase raw_phase(Phase p) noexcept {
        switch (p) {
            case Phase::absent: return core::serving_raw::Phase::absent;
            case Phase::idle: return core::serving_raw::Phase::idle;
            case Phase::running: return core::serving_raw::Phase::running;
            case Phase::drained: return core::serving_raw::Phase::drained;
            case Phase::capturing: return core::serving_raw::Phase::capturing;
            case Phase::restoring: return core::serving_raw::Phase::restoring;
            case Phase::poisoned: return core::serving_raw::Phase::poisoned;
        }
        return core::serving_raw::Phase::absent;
    }
    bool capture_allowed() const noexcept { return lifecycle_.allow(core::serving_raw::Lifecycle::Event::capture); }
    bool restore_allowed() const noexcept { return lifecycle_.allow(core::serving_raw::Lifecycle::Event::restore); }

    /// Coherent sequence (request lifetime is NOT the RAW work phase):
    ///   open_request() -> begin_work() -> [actual forward/reset] -> drain_work(cut) ->
    ///   capture()/restore() at the drained cut -> close_request().
    /// Every step is routed through core::serving_raw::Lifecycle, which tests drive directly, so the reachable
    /// order in the server and in the gate are the same code.
    bool open_request(std::string& error);
    bool begin_work(std::string& error);
    /// The caller supplies the ONE common all-stage cut; the per-stage vector the RAW contract requires is built
    /// here from the actual allocation, so an empty or mismatched span is unrepresentable, and an invalid cut is
    /// refused BEFORE the transaction is touched rather than poisoning it afterwards.
    bool drain_work(bool operation_succeeded, const core::StateCut& cut, std::string& error);
    bool close_request(std::string& error);
    /// Deliberate abandonment after a refused/failed RAW step: the transaction is uncertain, so destroying the
    /// owner is the failstop (its destructor terminates on running/poisoned phases).
    void abandon() noexcept;
    bool request_open() const noexcept { return lifecycle_.request_open(); }
    bool in_flight() const noexcept { return lifecycle_.request_open(); }

    /// The pure cut contract (suffix_raw_runtime.cpp:233-235), exposed so the lifecycle gate can test it without
    /// a device: non-zero id, a real history digest, exact_frontier <= forwarded <= accepted <= max_tokens, and
    /// at least one stage to certify.
    static bool cut_acceptable(const core::StateCut& cut, size_t stage_count, uint64_t max_tokens) noexcept;

    /// The qualified RAW operations, reachable only through this owner.
    SuffixRawResult capture(const core::SnapshotDescriptor&, core::BundleHandle&);
    SuffixRawResult restore(const core::BundleHandle&, const core::WindowKey&,
                            core::ResidentCopyFault* fault = nullptr);
    const core::WindowKey& key() const noexcept { return tx_->identity_; }
    /// The transaction this owner holds. The serving repair operation (a trusted friend of the controller) needs
    /// the shared owner itself: the qualified controller retains it across its own receipts, exactly as the
    /// standalone driver does. Handing out a raw reference would let the cap and completion state die early.
    const std::shared_ptr<SuffixRawTransaction>& transaction() const noexcept { return tx_; }
    uint64_t host_limit() const noexcept { return tx_->host_limit_; }

    // ---- bounded suffix REPAIR operands (EXECUTION-PLAN step 3) -------------------------------------------
    // Everything below is the same machinery the qualified standalone driver runs
    // (`src/program/suffix_experiment_driver.hpp`), reached through THIS owner so a serving request cannot
    // construct a borrowed fake owner or publish a completion certificate it did not earn.  Each call refuses
    // unless the request is open at a DRAINED cut; refusals leave operands untouched.

    /// Trusted runtime facts, taken from the transaction's own binding - never from imported metadata.
    const core::ModelGeometry& geometry() const noexcept { return tx_->geometry_; }
    const core::NativeSessionAllocation& native_allocation()const noexcept{return tx_->allocation_;}
    const std::vector<core::NativeWindowStage>& initial_config() const noexcept { return tx_->initial_config_; }
    uint64_t max_tokens() const noexcept { return tx_->max_tokens_; }
    uint64_t max_payload_bytes() const noexcept { return tx_->max_bytes_; }
    core::WindowBudget& budget() const noexcept { return tx_->budget_; }
    const core::StateCut& cut() const noexcept { return tx_->cut_; }

    /// The SAME descriptor shape the qualified driver builds: accepted = the whole token run, pending = the part
    /// past the cut's forwarded position.  History comes from the tokens, so a cut can never be certified against
    /// a digest the caller made up.
    static core::SnapshotDescriptor endpoint_metadata(const core::WindowKey& key, const core::StateCut& cut,
                                                      const std::vector<int32_t>& tokens);

    // Immutable original donor, or the exact bundle captured after authentic
    // unpublished known progress. Returns no caller-selected identity.
    const core::WindowKey* suffix_donor_key(const core::BundleHandle&) const noexcept;

    // Unpublished admission only. Next token is copied from an immutable paid request
    // before dispatch; no caller cut, sampled output or output watermark is accepted.
    bool prepare_owned_alignment_known(std::shared_ptr<const std::vector<int64_t>> request,
        std::shared_ptr<const void> request_credit,std::string& error);
    bool cancel_owned_alignment_known(std::string& error);
    bool complete_owned_alignment_known(const core::NativeBatchGroupCompletion&,std::string& error);

    /// Identity for the EDITED run: same runtime/topology binding, strictly newer epochs, so donor identity and
    /// edited identity can never be confused (or reused as each other's certificate).
    core::WindowKey edited_key() const noexcept;

    /// Host admission for the operation's own buffers/roots/splices.  Charges stay retained by the returned
    /// credit; a credit is not execution or completion.
    SuffixRawResult admit_host(uint64_t bytes, std::shared_ptr<const void>& credit);

    /// The bounded import, in the driver's order: quote (logical bound only), claim (aggregate cap), prepare
    /// (real global reservation + copies).  Quote/prepare take the transaction's OWN geometry and initial stage
    /// config, so a caller cannot substitute a friendlier layout.
    SuffixRawResult quote_suffix(const core::BundleHandle& donor, const core::WindowKey& donor_key,
                                 const core::BundleHandle& prefix, const core::WindowKey& prefix_key,
                                 const core::NativeSuffixTarget& target, const core::NativeSuffixLimits& limits,
                                 uint64_t& out_host_bytes);
    SuffixRawResult claim_splice(uint64_t bytes, std::shared_ptr<const void>& credit);
    SuffixRawResult prepare_suffix(const core::BundleHandle& donor, const core::WindowKey& donor_key,
                                   const core::BundleHandle& prefix, const core::WindowKey& prefix_key,
                                   const core::NativeSuffixTarget& target, const core::NativeSuffixLimits& limits,
                                   core::PreparedNativeSuffix& out);
    /// True only while the request may still be closed safely (no open fence, no poison).
    bool closable() const noexcept { return lifecycle_.allow(core::serving_raw::Lifecycle::Event::close_request); }

    // ---- the retained repair owner's PRIVATE WORK WINDOW (opt-in; the Lifecycle contract extension in
    // include/strata/core/serving_raw_contracts.hpp) --------------------------------------------------------
    // A parked repair owner advances its OWN private RAW transaction while no serving request holds this session.
    // That is legal only in a window the state machine itself grants: the request fence closed, the live fence
    // quiescent (`idle` or `drained` - the same `begin_allowed` cut capture/restore use), nothing terminal, and no
    // window already open. Opening a window does NOT change the live work phase, so nothing here can be mistaken
    // for an open live fence; while a window is open, `open_request`/`begin_work`/`drain_work`/`capture`/`restore`
    // are all refused, which is what keeps a second conversation from starting under a parked owner's private
    // work. With no window opened, every decision this session makes is the reviewed one.
    /// Observation only: may a bounded private unit be offered now? Never mutates anything.
    bool private_window_allowed() const noexcept {
        if (tx_ == nullptr || !healthy() || lifecycle_.private_window_open()) return false;
        // The transaction's OWN phase, not the adapter's last observation: a stale flag must not open a window.
        if (tx_->phase_ != SuffixRawTransaction::Phase::drained &&
            tx_->phase_ != SuffixRawTransaction::Phase::idle) return false;
        return lifecycle_.private_window_allowed();
    }
    bool begin_private_window(std::string& error) {
        lifecycle_.set_observed_phase(raw_phase(phase()));
        if (tx_ == nullptr || !healthy()) { error = "no healthy serving RAW session for a private work window"; return false; }
        if (tx_->phase_ != SuffixRawTransaction::Phase::drained &&
            tx_->phase_ != SuffixRawTransaction::Phase::idle) {
            error = "a private work window needs the live fence at an idle or drained cut";
            return false;
        }
        if (!lifecycle_.allow(core::serving_raw::Lifecycle::Event::begin_private_window)) {
            error = "a private work window needs the request fence closed, and no window already open";
            return false;
        }
        lifecycle_.apply(core::serving_raw::Lifecycle::Event::begin_private_window);
        return true;
    }
    /// The only exit from a window, and always available while one is open (including after a poison), so a
    /// window can never latch the session shut.
    bool end_private_window(std::string& error) {
        if (!lifecycle_.allow(core::serving_raw::Lifecycle::Event::end_private_window)) {
            error = "no private work window is open to end";
            return false;
        }
        lifecycle_.apply(core::serving_raw::Lifecycle::Event::end_private_window);
        return true;
    }
    bool private_window_open() const noexcept { return lifecycle_.private_window_open(); }

private:
    friend class resident_repair_detail::ResidentRepairController;
    struct CohortReceiptView {
        const core::NativeBatchGroupCompletion* receipt=nullptr;
        const core::SnapshotDescriptor* base=nullptr; // synchronous immutable controller borrow
        std::shared_ptr<const core::NativeBatchSlotOccupancy> occupancy;
        size_t row=0,slot=0;
        int32_t sample=-1;
        uint64_t group_operation=0;
        bool consumed=false;
    };
    bool inspect_cohort_receipt(const core::NativeBatchGroupCompletion&,const core::SnapshotDescriptor&,
        int32_t input,CohortReceiptView&,std::string& error);
    bool commit_cohort_receipt(CohortReceiptView&,const core::SnapshotDescriptor&,
        std::span<const core::StateCut>,std::FILE*,std::string& error);
    bool commit_cohort_known_receipt(CohortReceiptView&,const core::SnapshotDescriptor&,
        std::span<const core::StateCut>,std::string& error);
    struct AlignmentKnown {
        std::shared_ptr<const void> credit,request_credit;
        std::shared_ptr<const std::vector<int64_t>> request;
        core::SnapshotDescriptor before,next;
        std::vector<core::StateCut> stages;
    };
    std::unique_ptr<AlignmentKnown> alignment_known_;
    std::shared_ptr<const void> alignment_owner_credit_;
    core::BundleHandle alignment_donor_;
    std::unique_ptr<core::WindowKey> alignment_edit_origin_;
    bool slot_controller_bound_=false;
    explicit ServingRawSession(std::shared_ptr<SuffixRawTransaction> tx) : tx_(std::move(tx)) {}
    static core::WindowKey serving_identity(const void* runtime, const core::NativeSessionAllocation&);
    std::shared_ptr<SuffixRawTransaction> tx_;
    std::shared_ptr<core::NativeBatchSlotOccupancy> slot_occupancy_;
    core::WindowKey slot_current_key_;
    core::BundleHandle slot_source_provenance_,slot_donor_provenance_;
    std::vector<int32_t> slot_tokens_;
    uint64_t slot_last_group_=0;
    core::serving_raw::Lifecycle lifecycle_;   ///< request lifetime + observed RAW work phase
};
} // namespace strata::program
