#pragma once
// Window-keyed RETAINED REPAIR OWNER registry - the ownership half of
// .local/strata-implementation/scheduling/demand-age-scheduler-audit-20261030/BOUNDED-BREAKDOWN.md section 6,
// rewritten for slice 2 (retained repair across a response boundary).
//
// What this file owns:
//  * `RepairOwnerEvidence` - the ONLY way any caller can state a controller's facts. It is an immutable value
//    produced by `ResidentRepairController::owner_evidence()` (resident_repair_controller.inc) through
//    `make_repair_owner_evidence` below, which refuses a self-contradictory snapshot and derives debt/coherence
//    from the cuts instead of accepting them as arguments. A caller cannot supply a subset of the facts, cannot
//    leave them defaulted, and cannot claim `retired` about a controller that still has a private gang.
//  * `ServingRepairOwnerRegistry` - a bounded map from `core::WindowKey` to a LIVE, unretired, debt-retaining
//    repair owner that was moved out of the serving request's `serving_suffix_active` at a certified yield cut,
//    with the owner's real `private_work` charge pinned by the entry and released only by an explicit retire.
//  * the shared gate predicates (`retained_yield_allowed`, `retained_private_unit_allowed`,
//    `retained_park_allowed`, `parked_owner_publishable`, `owner_scope_escape`) that the real controller and the
//    real serving scope call, so the CPU gate drives the same decision code the server runs;
//  * the demand observation + idle-window admission used by the serving loop's bounded private tick, built on the
//    accepted kernel `serving_repair_admission.hpp` (unmodified), plus the serving scope's certified-yield gate
//    (`serving_certified_yield_gate`) and the idle window's wake condition (`serving_input_demand_observed`) -
//    both production predicates the serving loop calls and the CPU cell drives (slice 3, D1 and D2).
//
// NOT INTEGRATED BY DEFAULT. Two selectors, both off unless named:
//  * `STRATA_SERVING_REPAIR_OWNER_SLOTS` - the registry's slot bound; absent/invalid/0 means no registry at all,
//    so the serving release path stays exactly as it is today.
//  * `STRATA_SERVING_REPAIR_IDLE_UNIT_TOKENS` - the per-tick private-unit cap in tokens; absent/invalid/0 means
//    the serving loop never opens a private window. There is NO compiled default for either number, no age
//    threshold, and no escalation here.
//
// WHY THE SHAPES BELOW LOOK THE WAY THEY DO (forced by the source, not chosen):
//  * The payload type is a template parameter. `ServingSuffixRepairOperation::Outcome` is declared inside
//    generate.cpp behind `STRATA_SERVING_SUFFIX_REPAIR_AVAILABLE` (generate.cpp:3151), so no header can name it;
//    generate.cpp instantiates this template with the real Outcome and the requirement is stated once as
//    `RetainedRepairOwner` using the real member names.
//  * The registry never calls into a controller: `ResidentRepairController` lives in the composed
//    `src/program/resident_repair_controller.inc` and owns CUDA-bound types. Instead every registry operation
//    takes an evidence value the controller produced about itself, and re-derives its decision from it.
//
// THE LIFETIME RULE this exists for: `~ResidentRepairController` terminates the process unless `retired_`
// (resident_repair_controller.inc:70-71), and `SfxOpenFenceGuard` (generate.cpp:16783) failstops a request scope
// that unwinds holding a live controller. So `retire_owner`/`retire_all` are the only paths that may destroy a
// parked payload, each goes through the controller's own `retire()` first, and the registry refuses rather than
// evicting, overwriting or dropping an owner in every other case.
#include "serving_repair_admission.hpp"
#include "strata/core/serving_raw_contracts.hpp"
#include "strata/core/window_budget.hpp"
#include "strata/core/window_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::program {

// ---- the two selectors -------------------------------------------------------------------------------
// Both name a number and have no compiled default. An absent, empty, non-numeric or zero value means "selector
// absent". A campaign that enables either must record the number it passed (BOUNDED-BREAKDOWN section 2).
inline constexpr const char* kServingRepairOwnerSlotsEnv = "STRATA_SERVING_REPAIR_OWNER_SLOTS";
inline constexpr const char* kServingRepairIdleUnitEnv = "STRATA_SERVING_REPAIR_IDLE_UNIT_TOKENS";

inline std::size_t serving_repair_selector_number(const char* value) noexcept {
    if (value == nullptr || *value == '\0') return 0;
    std::size_t number = 0;
    for (const char* digit = value;; ++digit) {
        if (*digit == '\0') break;
        if (*digit < '0' || *digit > '9') return 0;  // anything else is "selector absent", never guessed
        if (number > (std::numeric_limits<std::size_t>::max() - std::size_t(*digit - '0')) / 10U) return 0;
        number = number * 10U + std::size_t(*digit - '0');
    }
    return number;
}
inline std::size_t serving_repair_owner_slots() noexcept {
    return serving_repair_selector_number(std::getenv(kServingRepairOwnerSlotsEnv));
}
// The idle private-unit cap, in tokens. 0 means the idle tick never runs. It is a CAP, never a target, a
// threshold or an adopted schedule: BOUNDED-BREAKDOWN section 2 names the per-unit private-work bound (today
// hard-coded 4 tokens at generate.cpp:3483) as a configurable field, not a compiled constant.
inline std::size_t serving_repair_idle_unit_tokens() noexcept {
    return serving_repair_selector_number(std::getenv(kServingRepairIdleUnitEnv));
}

// ---- refusals ---------------------------------------------------------------------------------------
// Every refusal the registry and the shared gates can produce. Admission is a boolean plus this named reason;
// a caller logs the name, nothing here scores or ranks.
enum class RepairOwnerRefusal : uint8_t {
    none,
    refused_selector_absent,                  // capacity 0: the opt-in selector was not present
    refused_full,                             // the slot bound was reached: refuse, never evict
    refused_duplicate_identity,               // an owner is already parked under this exact WindowKey
    refused_duplicate_window,                 // another owner of the same window (epochs aside) is parked
    refused_empty_owner_slot,                 // nothing to park
    refused_unpopulated_owner,                // an empty Outcome carries no owner
    refused_slot_busy,                        // the caller's active slot already owns an outcome
    refused_identity_not_publication,         // requested key differs from the owner's own published key
    refused_evidence_absent,                  // no controller-produced evidence for a live owner
    refused_evidence_invalid,                 // the snapshot contradicts itself: fabricated, never guessed
    refused_busy,                             // busy_
    refused_cancel_requested,                 // cancel_requested_
    refused_retired,                          // retired_: there is no live owner to retain
    refused_not_suspended,                    // the controller was never yielded into the retained state
    refused_already_suspended,                // a second yield/park of an owner that is already suspended
    refused_finished,                         // finish_* already ran: the exact path owns this owner
    refused_phase_not_drained,                // not the drained quiescent cut (serving_raw_contracts.hpp:32-35)
    refused_private_phase_not_drained,        // the private transaction is not quiescent
    refused_allocation_unhealthy,             // live or private allocation health
    refused_service_not_idle,                 // another controller operation holds the service
    refused_wrong_controller_thread,          // the runtime's controller thread is not this thread
    refused_serial_unrecorded,                // published_->serial was never read
    refused_no_debt,                          // zero debt: this is the exact path, not a retained yield
    refused_debt_inconsistent,                // private progress past the published cut
    refused_no_certified_cut,                 // no committed private all-stage witness at the private cut
    refused_certified_cut_not_exact,          // the private cut is not itself exact
    refused_no_retained_charge,               // the private gang's own reservation lease is missing
    refused_charge_purpose_mismatch,          // the retained lease is not the ledger's private_work charge
    refused_published_cut_divergence,         // published_->metadata.cut != live_.cut_ at the park
    refused_no_entry,                         // no owner parked under this WindowKey
    refused_identity_mismatch,                // the live session is not the parked owner's conversation
    refused_state_not_suspended,              // the entry is mid-retirement
    refused_retire_attempt,                   // the controller's own retire() refused
    refused_private_unit,                     // the controller refused (or failed) the bounded private unit
    refused_controller_lost_without_retirement, // no controller and no retirement proof: never drop silently
    refused_retire_left_controller,           // retire() claimed success but the payload still holds it
    refused_retire_requires_attempt,          // retire without an attempt on a payload holding a controller
    refused_close_holds_controller,           // close/destructor with an unretired controller parked
    refused_record_changed,                   // the live controller no longer matches this entry's record
    refused_live_state_moved,                 // a private unit moved the LIVE publication: not permitted
    refused_unit_cap_unset,                   // the idle unit selector named no cap: no work is admitted
    refused_unit_cap_exhausted,               // the private checkpoint already caught the published cut
    refused_no_progress,                      // the unit claimed success but advanced nothing
    refused_unit_too_large,                   // the delivered progress exceeded the selector's cap
    refused_private_window_not_open,          // the session's private work window was not opened for this tick
    refused_request_not_closed,               // a foreground request still holds the session
    refused_stale_publication,                // the live cut no longer stands on the owner's publication
    refused_window_identity_diverged,         // the live session's stable window identity is not the owner's
    refused_quarantined,                      // a failed/throwing unit made this record unverifiable: retire only
    // Slice 5 (A->B->A): the versioned retained RESULT, its owner-record publication and its activation.
    refused_prepared_result_absent,           // no prepared retained result for this owner
    refused_prepared_result_invalid,          // the prepared result contradicts the controller's own cuts
    refused_record_version_stale,             // the owner record moved since the result was prepared
    refused_content_diverged,                 // the accepted sequence (content version) is not the chained one
    refused_residency_diverged,               // the expert-residency generation the chain is not bound to
    refused_capacity_unreserved,              // no up-front capacity/activation reservation for this owner
    refused_activation_resources_invalid,     // the capture/restore/return charges are missing or mis-purposed
    refused_activation_not_drained,           // the live fence is not at a drained all-stage boundary
    refused_activation_fence_not_open,        // no request fence open to capture/restore under
    refused_prior_state_unverified,           // the other conversation's state was not captured/restored+verified
    refused_activation_state_diverged,        // the activated state is not the record it activated
};

inline const char* repair_owner_refusal_name(RepairOwnerRefusal refusal) noexcept {
    switch (refusal) {
        case RepairOwnerRefusal::none: return "none";
        case RepairOwnerRefusal::refused_selector_absent: return "refused_selector_absent";
        case RepairOwnerRefusal::refused_full: return "refused_full";
        case RepairOwnerRefusal::refused_duplicate_identity: return "refused_duplicate_identity";
        case RepairOwnerRefusal::refused_duplicate_window: return "refused_duplicate_window";
        case RepairOwnerRefusal::refused_empty_owner_slot: return "refused_empty_owner_slot";
        case RepairOwnerRefusal::refused_unpopulated_owner: return "refused_unpopulated_owner";
        case RepairOwnerRefusal::refused_slot_busy: return "refused_slot_busy";
        case RepairOwnerRefusal::refused_identity_not_publication: return "refused_identity_not_publication";
        case RepairOwnerRefusal::refused_evidence_absent: return "refused_evidence_absent";
        case RepairOwnerRefusal::refused_evidence_invalid: return "refused_evidence_invalid";
        case RepairOwnerRefusal::refused_busy: return "refused_busy";
        case RepairOwnerRefusal::refused_cancel_requested: return "refused_cancel_requested";
        case RepairOwnerRefusal::refused_retired: return "refused_retired";
        case RepairOwnerRefusal::refused_not_suspended: return "refused_not_suspended";
        case RepairOwnerRefusal::refused_already_suspended: return "refused_already_suspended";
        case RepairOwnerRefusal::refused_finished: return "refused_finished";
        case RepairOwnerRefusal::refused_phase_not_drained: return "refused_phase_not_drained";
        case RepairOwnerRefusal::refused_private_phase_not_drained: return "refused_private_phase_not_drained";
        case RepairOwnerRefusal::refused_allocation_unhealthy: return "refused_allocation_unhealthy";
        case RepairOwnerRefusal::refused_service_not_idle: return "refused_service_not_idle";
        case RepairOwnerRefusal::refused_wrong_controller_thread: return "refused_wrong_controller_thread";
        case RepairOwnerRefusal::refused_serial_unrecorded: return "refused_serial_unrecorded";
        case RepairOwnerRefusal::refused_no_debt: return "refused_no_debt";
        case RepairOwnerRefusal::refused_debt_inconsistent: return "refused_debt_inconsistent";
        case RepairOwnerRefusal::refused_no_certified_cut: return "refused_no_certified_cut";
        case RepairOwnerRefusal::refused_certified_cut_not_exact: return "refused_certified_cut_not_exact";
        case RepairOwnerRefusal::refused_no_retained_charge: return "refused_no_retained_charge";
        case RepairOwnerRefusal::refused_charge_purpose_mismatch: return "refused_charge_purpose_mismatch";
        case RepairOwnerRefusal::refused_published_cut_divergence: return "refused_published_cut_divergence";
        case RepairOwnerRefusal::refused_no_entry: return "refused_no_entry";
        case RepairOwnerRefusal::refused_identity_mismatch: return "refused_identity_mismatch";
        case RepairOwnerRefusal::refused_state_not_suspended: return "refused_state_not_suspended";
        case RepairOwnerRefusal::refused_retire_attempt: return "refused_retire_attempt";
        case RepairOwnerRefusal::refused_private_unit: return "refused_private_unit";
        case RepairOwnerRefusal::refused_controller_lost_without_retirement:
            return "refused_controller_lost_without_retirement";
        case RepairOwnerRefusal::refused_retire_left_controller: return "refused_retire_left_controller";
        case RepairOwnerRefusal::refused_retire_requires_attempt: return "refused_retire_requires_attempt";
        case RepairOwnerRefusal::refused_close_holds_controller: return "refused_close_holds_controller";
        case RepairOwnerRefusal::refused_record_changed: return "refused_record_changed";
        case RepairOwnerRefusal::refused_live_state_moved: return "refused_live_state_moved";
        case RepairOwnerRefusal::refused_unit_cap_unset: return "refused_unit_cap_unset";
        case RepairOwnerRefusal::refused_unit_cap_exhausted: return "refused_unit_cap_exhausted";
        case RepairOwnerRefusal::refused_no_progress: return "refused_no_progress";
        case RepairOwnerRefusal::refused_unit_too_large: return "refused_unit_too_large";
        case RepairOwnerRefusal::refused_private_window_not_open: return "refused_private_window_not_open";
        case RepairOwnerRefusal::refused_request_not_closed: return "refused_request_not_closed";
        case RepairOwnerRefusal::refused_stale_publication: return "refused_stale_publication";
        case RepairOwnerRefusal::refused_window_identity_diverged: return "refused_window_identity_diverged";
        case RepairOwnerRefusal::refused_quarantined: return "refused_quarantined";
        case RepairOwnerRefusal::refused_prepared_result_absent: return "refused_prepared_result_absent";
        case RepairOwnerRefusal::refused_prepared_result_invalid: return "refused_prepared_result_invalid";
        case RepairOwnerRefusal::refused_record_version_stale: return "refused_record_version_stale";
        case RepairOwnerRefusal::refused_content_diverged: return "refused_content_diverged";
        case RepairOwnerRefusal::refused_residency_diverged: return "refused_residency_diverged";
        case RepairOwnerRefusal::refused_capacity_unreserved: return "refused_capacity_unreserved";
        case RepairOwnerRefusal::refused_activation_resources_invalid: return "refused_activation_resources_invalid";
        case RepairOwnerRefusal::refused_activation_not_drained: return "refused_activation_not_drained";
        case RepairOwnerRefusal::refused_activation_fence_not_open: return "refused_activation_fence_not_open";
        case RepairOwnerRefusal::refused_prior_state_unverified: return "refused_prior_state_unverified";
        case RepairOwnerRefusal::refused_activation_state_diverged: return "refused_activation_state_diverged";
    }
    return "unknown"; // unreachable: every enumerator is listed above
}

// The failstop for "a retained owner is about to be lost". It repeats `core::retirement_failed`'s contract
// (include/strata/core/strict_retirement.hpp:8-12: name the operation, then `_Exit(1)` so the supervisor
// restarts the worker) because that helper's header pulls in <cuda_runtime.h> and this registry must stay
// buildable in the CPU gate. Same message shape, same exit status as the controller destructor
// (resident_repair_controller.inc:70-71).
[[noreturn]] inline void owner_registry_failstop(const char* reason) noexcept {
    std::fprintf(stderr, "strata: strict retirement failed at %s; worker restart required\n", reason);
    std::fflush(stderr);
    std::_Exit(1);
}

// ---- the evidence ----------------------------------------------------------------------------------
// The complete snapshot a controller makes about itself. Every field is a fact the controller reads under its
// own invariants; there are no optional facts and no defaults. `make_repair_owner_evidence` is the only maker:
// it validates the snapshot's internal consistency and derives the quantities the gates use, so a caller cannot
// inject a conclusion (debt, coherence, "certified") separately from the cuts that produce it.
struct RepairOwnerFacts {
    core::serving_raw::Phase live_phase = core::serving_raw::Phase::absent;
    core::serving_raw::Phase private_phase = core::serving_raw::Phase::absent;  // absent == no private gang
    bool busy = false, retired = false, cancel_requested = false, finished = false, suspended = false;
    bool live_allocation_healthy = false, private_allocation_healthy = false;
    bool service_idle = false;        ///< runtime_->service_->active_ == nullptr
    bool controller_thread = false;   ///< runtime controller thread AND service controller thread are this thread
    uint64_t published_serial = 0;    ///< 0 == published_ is null
    core::WindowKey published_key{};
    core::StateCut published_cut{}, live_cut{}, private_cut{};
    uint64_t exact_completion_cut = 0;    ///< 0 == no committed private all-stage witness
    bool exact_completion_committed = false;
    // The private gang's ledger charge, type-erased exactly as serving already type-erases an admitted credit
    // (Outcome::operation_credit, generate.cpp:3198).  The CONTROLLER fills both fields from its own
    // NativeSessionAllocation reservation (ResidentRepairController::owner_evidence), including the purpose it
    // reads from the real ReservationLease; a caller never states a purpose.
    std::shared_ptr<const void> private_charge{};
    core::ReservationPurpose private_charge_purpose = core::ReservationPurpose::foreground;
};

// An immutable, self-consistent evidence value. A default-constructed one is INVALID and every gate refuses it,
// so "no facts" can never read as "facts passed".
class RepairOwnerEvidence final {
public:
    RepairOwnerEvidence() = default;   // invalid by construction: `produced_` stays false

    bool produced() const noexcept { return produced_; }
    core::serving_raw::Phase live_phase() const noexcept { return facts_.live_phase; }
    core::serving_raw::Phase private_phase() const noexcept { return facts_.private_phase; }
    bool busy() const noexcept { return facts_.busy; }
    bool retired() const noexcept { return facts_.retired; }
    bool cancel_requested() const noexcept { return facts_.cancel_requested; }
    bool finished() const noexcept { return facts_.finished; }
    bool suspended() const noexcept { return facts_.suspended; }
    bool live_allocation_healthy() const noexcept { return facts_.live_allocation_healthy; }
    bool private_allocation_healthy() const noexcept { return facts_.private_allocation_healthy; }
    bool service_idle() const noexcept { return facts_.service_idle; }
    bool controller_thread() const noexcept { return facts_.controller_thread; }
    uint64_t published_serial() const noexcept { return facts_.published_serial; }
    const core::WindowKey& published_key() const noexcept { return facts_.published_key; }
    const core::StateCut& published_cut() const noexcept { return facts_.published_cut; }
    const core::StateCut& live_cut() const noexcept { return facts_.live_cut; }
    const core::StateCut& private_cut() const noexcept { return facts_.private_cut; }
    uint64_t exact_completion_cut() const noexcept { return facts_.exact_completion_cut; }
    const std::shared_ptr<const void>& private_charge() const noexcept { return facts_.private_charge; }
    core::ReservationPurpose private_charge_purpose() const noexcept { return facts_.private_charge_purpose; }
    // Derived, never supplied: the private checkpoint has not caught the published CURRENT cut.
    uint64_t token_debt() const noexcept {
        return facts_.published_cut.forwarded > facts_.private_cut.forwarded
                   ? facts_.published_cut.forwarded - facts_.private_cut.forwarded
                   : 0;
    }
    // Derived: the controller's publication IS the live transaction's cut (the `check_idle` coherence conjunct,
    // resident_repair_controller.inc:103-105).
    bool live_coherent() const noexcept { return facts_.published_cut == facts_.live_cut; }
    // Derived: the private gang's own cut is what the controller holds a committed all-stage witness for.
    bool certified_at_private_cut() const noexcept {
        return facts_.exact_completion_committed && facts_.exact_completion_cut != 0 &&
               facts_.exact_completion_cut == facts_.private_cut.id &&
               facts_.private_cut.exact_frontier == facts_.private_cut.forwarded;
    }
    const RepairOwnerFacts& raw() const noexcept { return facts_; }

private:
    friend RepairOwnerEvidence make_repair_owner_evidence(const RepairOwnerFacts& facts,
                                                          RepairOwnerRefusal& why) noexcept;
    RepairOwnerFacts facts_{};
    bool produced_ = false;
};

// The one maker, declared before the class so the friendship names it, and defined below.
// The one maker. An inconsistent snapshot is refused (and reports which rule it broke) rather than carried: the
// combinations below are exactly the shapes that end in the terminal controller destructor or in publishing a
// cut the controller never certified.
inline RepairOwnerEvidence make_repair_owner_evidence(const RepairOwnerFacts& facts,
                                                      RepairOwnerRefusal& why) noexcept {
    RepairOwnerEvidence evidence;
    auto refuse = [&why](RepairOwnerRefusal refusal) {
        why = refusal;
        return RepairOwnerEvidence{};
    };
    if (facts.published_serial == 0) return refuse(RepairOwnerRefusal::refused_serial_unrecorded);
    if (facts.live_phase == core::serving_raw::Phase::absent ||
        facts.live_phase == core::serving_raw::Phase::poisoned)
        return refuse(RepairOwnerRefusal::refused_phase_not_drained);
    if (!core::serving_raw::digest_present(facts.published_cut.history) ||
        !core::serving_raw::digest_present(facts.live_cut.history) || facts.published_cut.id == 0 ||
        facts.live_cut.id == 0)
        return refuse(RepairOwnerRefusal::refused_evidence_invalid);
    if (facts.published_cut.accepted < facts.published_cut.forwarded ||
        facts.published_cut.exact_frontier > facts.published_cut.forwarded ||
        facts.live_cut.accepted < facts.live_cut.forwarded ||
        facts.live_cut.exact_frontier > facts.live_cut.forwarded)
        return refuse(RepairOwnerRefusal::refused_evidence_invalid);
    if (facts.private_cut.forwarded > facts.published_cut.forwarded)
        return refuse(RepairOwnerRefusal::refused_debt_inconsistent);
    // `retire()` drops the private transaction (resident_repair_controller.inc:1221-1228); an unretired
    // controller always still has one. A snapshot that claims either half of that pair is fabricated.
    const bool has_private = facts.private_phase != core::serving_raw::Phase::absent;
    if (facts.retired == has_private) return refuse(RepairOwnerRefusal::refused_evidence_invalid);
    if (has_private) {
        if (facts.private_cut.id == 0 || !core::serving_raw::digest_present(facts.private_cut.history))
            return refuse(RepairOwnerRefusal::refused_evidence_invalid);
        if (!facts.private_charge) return refuse(RepairOwnerRefusal::refused_no_retained_charge);
        if (facts.private_charge_purpose != core::ReservationPurpose::private_work)
            return refuse(RepairOwnerRefusal::refused_charge_purpose_mismatch);
    } else if (!facts.retired) {
        return refuse(RepairOwnerRefusal::refused_evidence_invalid);
    }
    evidence.facts_ = facts;
    evidence.produced_ = true;
    why = RepairOwnerRefusal::none;
    return evidence;
}

// The live session's own facts at a cut. Every one is an observation of the serving adapter, never a claim about
// the owner, which is what lets the registry refuse a publication that is stale merely because its OWN recorded
// facts still match.
struct LiveSessionFacts {
    core::WindowKey session_key{};                                    ///< ServingRawSession::key()
    core::StateCut session_cut{};                                     ///< ServingRawSession::cut()
    core::serving_raw::Phase live_phase = core::serving_raw::Phase::absent;  ///< phase()
    bool request_open = false;                                        ///< request_open()
    bool private_window_open = false;                                 ///< the lifecycle's private window
    // The expert-residency inventory generation the live session stands on now (the serving scope's own
    // `a_swapped` / `adapt_nswaps` counter). A parked owner's private chain was built against the inventory that
    // was uploaded when its gang was initialised, so this is the field that decides whether that chain may run
    // again: equality is the exact condition eligibility returns under (slice 5, requirement 5).
    uint64_t residency_generation = 0;
};

// Two WindowKeys name the same window when every field except the mutable epochs agrees. The parked owner's
// publication carries bumped append/selection epochs (controller foreground()/attach_prefix()), while the
// session's own identity (SuffixRawTransaction::identity_) does not; comparing only the stable part is what lets
// the registry recognise "the same conversation" without treating a bumped epoch as a different window.
inline bool same_stable_window(const core::WindowKey& a, const core::WindowKey& b) noexcept {
    return a.engine_generation == b.engine_generation && a.window_generation == b.window_generation &&
           a.session_id == b.session_id && a.thread_id == b.thread_id && a.window_id == b.window_id &&
           a.model_profile == b.model_profile && a.topology == b.topology;
}

// Incoming-conversation identity gate: the request's prompt prefix must carry the parked owner's
// published history. Without this, a nonmatching request (conversation B) can activate a parked
// owner belonging to conversation A because `same_stable_window` is process-constant. The check
// computes the history digest over the first `published_cut.accepted` tokens of the incoming prompt
// and compares against the owner's recorded published history. A mismatch means this request does
// not belong to this owner: refuse activation, leave the owner parked.
inline bool incoming_prompt_identity_matches(const std::vector<int32_t>& incoming_prefix,
                                             const core::StateCut& published_cut) noexcept {
    const uint64_t needed = published_cut.accepted;
    if (needed == 0 || incoming_prefix.size() < needed) return false;
    // digest over exactly the first `needed` tokens; the caller may pass a longer prefix.
    return core::window_history_digest(
        std::vector<int32_t>(incoming_prefix.begin(),
                             incoming_prefix.begin() + static_cast<std::size_t>(needed))
    ) == published_cut.history;
}

// ---- the shared gates (the real controller and the real serving scope call these) -------------------
// Every gate is pure over the evidence: same inputs, same decision, in the server and in the CPU cell.

// A certified yield at a non-zero-debt coherent cut. The controller must be idle on its OWN terms
// (`check_idle`, resident_repair_controller.inc:103-105), must not already be suspended or finished, must have
// real debt, and must hold a committed all-stage witness for its private cut - that witness, not a caller's
// word, is what makes the cut "certified". On success `certified` is the private cut the yield certifies.
inline bool retained_yield_allowed(const RepairOwnerEvidence& evidence, RepairOwnerRefusal& why,
                                   core::StateCut& certified) noexcept {
    if (!evidence.produced()) {
        why = evidence.published_serial() == 0 ? RepairOwnerRefusal::refused_evidence_absent
                                               : RepairOwnerRefusal::refused_evidence_invalid;
        return false;
    }
    if (evidence.retired()) { why = RepairOwnerRefusal::refused_retired; return false; }
    if (evidence.finished()) { why = RepairOwnerRefusal::refused_finished; return false; }
    if (evidence.busy()) { why = RepairOwnerRefusal::refused_busy; return false; }
    if (evidence.cancel_requested()) { why = RepairOwnerRefusal::refused_cancel_requested; return false; }
    if (evidence.suspended()) { why = RepairOwnerRefusal::refused_already_suspended; return false; }
    if (evidence.live_phase() != core::serving_raw::Phase::drained) {
        why = RepairOwnerRefusal::refused_phase_not_drained; return false;
    }
    if (evidence.private_phase() != core::serving_raw::Phase::drained) {
        why = RepairOwnerRefusal::refused_private_phase_not_drained; return false;
    }
    if (!evidence.live_allocation_healthy() || !evidence.private_allocation_healthy()) {
        why = RepairOwnerRefusal::refused_allocation_unhealthy; return false;
    }
    if (!evidence.service_idle()) { why = RepairOwnerRefusal::refused_service_not_idle; return false; }
    if (!evidence.controller_thread()) { why = RepairOwnerRefusal::refused_wrong_controller_thread; return false; }
    if (!evidence.live_coherent()) { why = RepairOwnerRefusal::refused_published_cut_divergence; return false; }
    if (evidence.token_debt() == 0) { why = RepairOwnerRefusal::refused_no_debt; return false; }
    if (!evidence.certified_at_private_cut()) {
        why = evidence.exact_completion_cut() == 0 ? RepairOwnerRefusal::refused_no_certified_cut
                                                   : RepairOwnerRefusal::refused_certified_cut_not_exact;
        return false;
    }
    certified = evidence.private_cut();
    why = RepairOwnerRefusal::none;
    return true;
}

// One bounded private unit for a SUSPENDED owner. This is the gate that replaces live coherence: the parked
// owner's publication is by design no longer the live cut, so the gate insists on the private gang's own
// quiescent state, on the service being idle and on this thread being the controller thread, and on the
// selector-named cap. `target` is the private-forwarded position the unit may reach: never past the published
// cut, never more than `cap_tokens` beyond where the private gang stands.
inline bool retained_private_unit_allowed(const RepairOwnerEvidence& evidence, uint64_t cap_tokens,
                                          uint64_t& target, RepairOwnerRefusal& why) noexcept {
    target = 0;
    if (!evidence.produced()) {
        why = evidence.published_serial() == 0 ? RepairOwnerRefusal::refused_evidence_absent
                                               : RepairOwnerRefusal::refused_evidence_invalid;
        return false;
    }
    if (cap_tokens == 0) { why = RepairOwnerRefusal::refused_unit_cap_unset; return false; }
    if (evidence.retired()) { why = RepairOwnerRefusal::refused_retired; return false; }
    if (evidence.finished()) { why = RepairOwnerRefusal::refused_finished; return false; }
    if (!evidence.suspended()) { why = RepairOwnerRefusal::refused_not_suspended; return false; }
    if (evidence.busy()) { why = RepairOwnerRefusal::refused_busy; return false; }
    if (evidence.cancel_requested()) { why = RepairOwnerRefusal::refused_cancel_requested; return false; }
    if (evidence.private_phase() != core::serving_raw::Phase::drained) {
        why = RepairOwnerRefusal::refused_private_phase_not_drained; return false;
    }
    // The live transaction must still be quiescent and must never be poisoned: the private gang shares the
    // service, its model and its cache with whatever conversation last held the fence.
    if (evidence.live_phase() != core::serving_raw::Phase::drained) {
        why = RepairOwnerRefusal::refused_phase_not_drained; return false;
    }
    if (!evidence.live_allocation_healthy() || !evidence.private_allocation_healthy()) {
        why = RepairOwnerRefusal::refused_allocation_unhealthy; return false;
    }
    if (!evidence.service_idle()) { why = RepairOwnerRefusal::refused_service_not_idle; return false; }
    if (!evidence.controller_thread()) { why = RepairOwnerRefusal::refused_wrong_controller_thread; return false; }
    if (evidence.token_debt() == 0) { why = RepairOwnerRefusal::refused_unit_cap_exhausted; return false; }
    const uint64_t before = evidence.private_cut().forwarded;
    target = std::min(evidence.published_cut().forwarded, before + std::min<uint64_t>(cap_tokens, UINT64_MAX - before));
    if (target <= before) { why = RepairOwnerRefusal::refused_unit_cap_exhausted; return false; }
    why = RepairOwnerRefusal::none;
    return true;
}

// Parking after a certified yield. The evidence must still show a coherent drained cut (the park happens in the
// same step as the yield, before any other conversation moves the live cut), a suspended controller, real debt,
// a certified private cut and the private gang's own `private_work` lease.
inline bool retained_park_allowed(const RepairOwnerEvidence& evidence, const RepairOwnerRefusal& already,
                                  RepairOwnerRefusal& why) noexcept {
    if (already != RepairOwnerRefusal::none) { why = already; return false; }
    if (!evidence.produced()) { why = RepairOwnerRefusal::refused_evidence_invalid; return false; }
    if (!evidence.suspended()) { why = RepairOwnerRefusal::refused_not_suspended; return false; }
    if (evidence.retired()) { why = RepairOwnerRefusal::refused_retired; return false; }
    if (evidence.token_debt() == 0) { why = RepairOwnerRefusal::refused_no_debt; return false; }
    if (!evidence.certified_at_private_cut()) {
        why = RepairOwnerRefusal::refused_no_certified_cut; return false;
    }
    if (!evidence.private_charge()) { why = RepairOwnerRefusal::refused_no_retained_charge; return false; }
    if (evidence.private_charge_purpose() != core::ReservationPurpose::private_work) {
        why = RepairOwnerRefusal::refused_charge_purpose_mismatch; return false;
    }
    if (evidence.live_phase() != core::serving_raw::Phase::drained) {
        why = RepairOwnerRefusal::refused_phase_not_drained; return false;
    }
    if (!evidence.live_coherent()) { why = RepairOwnerRefusal::refused_published_cut_divergence; return false; }
    why = RepairOwnerRefusal::none;
    return true;
}

// Stale-publication gate (BOUNDED-BREAKDOWN section 3). A parked owner may publish into the live session only
// while the LIVE session still stands on exactly the publication the owner was parked under: same stable window
// identity, same live cut as the recorded publication, request fence closed or reopenable at that cut. The
// registry's own recorded facts matching is never enough - it is the session's current key/cut that decides.
inline bool parked_owner_publishable(const RepairOwnerEvidence& evidence, const LiveSessionFacts& session,
                                     const core::WindowKey& recorded_identity, const core::StateCut& recorded_cut,
                                     RepairOwnerRefusal& why) noexcept {
    if (!evidence.produced()) { why = RepairOwnerRefusal::refused_evidence_invalid; return false; }
    if (evidence.retired()) { why = RepairOwnerRefusal::refused_retired; return false; }
    if (!same_stable_window(session.session_key, recorded_identity)) {
        why = RepairOwnerRefusal::refused_window_identity_diverged; return false;
    }
    if (!(evidence.published_key() == recorded_identity)) {
        why = RepairOwnerRefusal::refused_identity_not_publication; return false;
    }
    if (!(evidence.published_cut() == recorded_cut)) {
        why = RepairOwnerRefusal::refused_record_changed; return false;
    }
    // The session moved on (another conversation advanced the live cut): publishing the parked owner's older
    // state would overwrite newer foreground history, so it is refused before any mutation.
    if (!(session.session_cut == recorded_cut)) {
        why = RepairOwnerRefusal::refused_stale_publication; return false;
    }
    if (session.live_phase != core::serving_raw::Phase::drained) {
        why = RepairOwnerRefusal::refused_phase_not_drained; return false;
    }
    why = RepairOwnerRefusal::none;
    return true;
}

// ---- the versioned retained RESULT: publication and activation (slice 5, A->B->A) -----------------
// GAP-A blocker 7 was: a parked owner whose live cut moved on had NO way back, because the only exit from the
// suspended state was `ResidentRepairController::resume_retained`, which refuses unless the controller's published
// cut already equals the live cut and (correctly) performs no reconciliation. That comparison is kept exactly as
// it is. What was missing is the versioned path that makes it true again for the RIGHT conversation:
//   (a) PREPARE  - the controller states a retained RESULT about itself: its certified private cut, the committed
//       all-stage witness that covers it, and the real versioned fields (accepted sequence, pending token, output
//       watermark, content/append/selection/engine/residency versions);
//   (b) PUBLISH-IF-CURRENT - that result is recorded in the OWNER RECORD only (the registry entry plus the
//       controller's own retained-result slot). It never touches the live session, and it is refused unless the
//       record version, identity, serials, cuts, debt and versions all still agree with what the controller says
//       about itself at this instant;
//   (c) ACTIVATE - only at a drained all-stage boundary on the live fence, the owner's published state becomes the
//       live session state, with the live session's PRIOR conversation state captured first and restored+verified
//       if the activation does not land. After a landed activation the controller's own coherence conjunction is
//       genuinely true again, so the untouched `resume_retained` is what clears the suspended state.
// Identity throughout is the stable conversation/window (`core::WindowKey`: revision, append/selection/provenance
// epochs, engine/window generation, model profile, topology), never a request ordinal.

// The version fields an owner record carries. `content` is the digest of the accepted sequence the private chain
// actually replays (`StateCut::history`), so a stale edit shows up as a content-version difference even when the
// stable window identity still matches; `residency_generation` is the expert-residency inventory generation the
// chain was bound to when it was initialised.
struct RetainedVersions {
    uint64_t engine_generation = 0, window_generation = 0, revision = 0;
    uint64_t append_epoch = 0, selection_epoch = 0, provenance_epoch = 0;
    core::StateDigest content{};
    uint64_t residency_generation = 0;
    bool operator==(const RetainedVersions&) const = default;
};
// The one derivation, from the real fields of the identity and the cut the chain stands on. Nothing here is a
// hash the caller invents: `content` is the cut's own history digest.
// The chain version comparison, residency aside: content (the accepted-sequence digest), revision, the append /
// selection / provenance epochs and the engine/window generations.
inline bool same_chain_versions(const RetainedVersions& a, const RetainedVersions& b) noexcept {
    return a.engine_generation == b.engine_generation && a.window_generation == b.window_generation &&
           a.revision == b.revision && a.append_epoch == b.append_epoch &&
           a.selection_epoch == b.selection_epoch && a.provenance_epoch == b.provenance_epoch &&
           a.content == b.content;
}

inline RetainedVersions make_retained_versions(const core::WindowKey& identity, const core::StateCut& chain_cut,
                                               uint64_t residency_generation) noexcept {
    return RetainedVersions{identity.engine_generation, identity.window_generation, identity.revision,
                            identity.append_epoch, identity.selection_epoch, identity.provenance_epoch,
                            chain_cut.history, residency_generation};
}

// The facts the controller reads about itself to prepare a retained result. There is no caller-facing way to state
// any of them: `ResidentRepairController::prepare_retained_result` fills this from its own private transaction,
// its committed completion witness, its publication and its identity, and `make_prepared_retained_result` below
// refuses a self-contradictory set.
struct PreparedRetainedResultFacts {
    core::WindowKey identity{};
    uint64_t published_serial = 0, record_version = 0;
    core::StateCut published_cut{}, certified_cut{};
    uint64_t accepted_sequence = 0;      ///< the length of the accepted sequence the chain replays
    int32_t pending_token = -1;          ///< the one token the activated state is about to consume
    uint64_t output_watermark = 0;
    bool witness_committed = false;      ///< the real all-stage completion witness for `certified_cut`
    bool holds_frozen_image = false;     ///< the all-stage image of `certified_cut` exists and is retained
};

// An immutable, self-consistent retained result. A default-constructed one is INVALID, so "no result" can never
// read as "a result that passed".
class PreparedRetainedResult final {
public:
    PreparedRetainedResult() = default;
    bool produced() const noexcept { return produced_; }
    const core::WindowKey& identity() const noexcept { return facts_.identity; }
    uint64_t published_serial() const noexcept { return facts_.published_serial; }
    uint64_t record_version() const noexcept { return facts_.record_version; }
    const core::StateCut& published_cut() const noexcept { return facts_.published_cut; }
    const core::StateCut& certified_cut() const noexcept { return facts_.certified_cut; }
    uint64_t accepted_sequence() const noexcept { return facts_.accepted_sequence; }
    int32_t pending_token() const noexcept { return facts_.pending_token; }
    uint64_t output_watermark() const noexcept { return facts_.output_watermark; }
    // The chain's version stamp as the CONTROLLER can state it: the identity's epochs/rev/generations plus the
    // content digest of the certified cut. The expert-residency generation is a record-side fact (the payload
    // carries the baseline its gang was initialised against), so it is compared separately, never claimed here.
    RetainedVersions versions() const noexcept {
        return make_retained_versions(facts_.identity, facts_.certified_cut, 0);
    }
    uint64_t token_debt() const noexcept {
        return facts_.published_cut.forwarded > facts_.certified_cut.forwarded
                   ? facts_.published_cut.forwarded - facts_.certified_cut.forwarded
                   : 0;
    }

private:
    friend PreparedRetainedResult make_prepared_retained_result(const PreparedRetainedResultFacts& facts,
                                                                RepairOwnerRefusal& why) noexcept;
    PreparedRetainedResultFacts facts_{};
    bool produced_ = false;
};

// The one maker. Every rule below is a shape that would otherwise let an owner activate a state its controller
// never certified: an uncommitted witness, a cut that is not exact at its own frontier, a chain that caught (or
// passed) the publication, a watermark the private chain does not carry, a pending token that does not exist.
inline PreparedRetainedResult make_prepared_retained_result(const PreparedRetainedResultFacts& facts,
                                                            RepairOwnerRefusal& why) noexcept {
    auto refuse = [&why](RepairOwnerRefusal refusal) { why = refusal; return PreparedRetainedResult{}; };
    if (facts.identity.window_id.empty() || facts.published_serial == 0 || facts.record_version == 0)
        return refuse(RepairOwnerRefusal::refused_prepared_result_invalid);
    if (facts.published_cut.id == 0 || !core::serving_raw::digest_present(facts.published_cut.history) ||
        facts.certified_cut.id == 0 || !core::serving_raw::digest_present(facts.certified_cut.history))
        return refuse(RepairOwnerRefusal::refused_prepared_result_invalid);
    if (facts.published_cut.accepted != facts.published_cut.forwarded + 1 ||
        facts.published_cut.exact_frontier > facts.published_cut.forwarded)
        return refuse(RepairOwnerRefusal::refused_prepared_result_invalid);
    // The certified cut is the private chain's own all-stage cut: exact at its own frontier, and carrying the
    // publication's output watermark (advance_private_to copies it, it never invents one). Exact private replay
    // certifies only consumed rows (accepted == forwarded), including at zero debt. The publication separately
    // owns the genuine pending token. A checked pending-history sealing may also carry accepted == forwarded+1.
    if (facts.certified_cut.exact_frontier != facts.certified_cut.forwarded)
        return refuse(RepairOwnerRefusal::refused_certified_cut_not_exact);
    if (facts.certified_cut.accepted != facts.certified_cut.forwarded &&
        facts.certified_cut.accepted != facts.certified_cut.forwarded + 1)
        return refuse(RepairOwnerRefusal::refused_prepared_result_invalid);
    if (facts.certified_cut.output_watermark != facts.output_watermark ||
        facts.published_cut.output_watermark != facts.output_watermark)
        return refuse(RepairOwnerRefusal::refused_prepared_result_invalid);
    if (facts.certified_cut.forwarded > facts.published_cut.forwarded)
        return refuse(RepairOwnerRefusal::refused_debt_inconsistent);
    // Zero forwarded-debt is NOT refused here: a chain that has caught the published cut is exactly the state an
    // activation can land (the yield and park gates still require real debt, so a result like this can only come
    // from an owner that was advanced to the endpoint while it was parked).
    if (facts.accepted_sequence != facts.certified_cut.accepted || facts.pending_token < 0)
        return refuse(RepairOwnerRefusal::refused_prepared_result_invalid);
    if (!facts.witness_committed) return refuse(RepairOwnerRefusal::refused_no_certified_cut);
    if (!facts.holds_frozen_image) return refuse(RepairOwnerRefusal::refused_no_retained_charge);
    PreparedRetainedResult result;
    result.facts_ = facts;
    result.produced_ = true;
    why = RepairOwnerRefusal::none;
    return result;
}

// The owner record's versioned facts: what the registry holds, and what a prepared result must match.
struct RetainedRecordFacts {
    core::WindowKey identity{};
    uint64_t published_serial = 0, record_version = 0, token_debt = 0;
    core::StateCut published_cut{}, certified_cut{};
    RetainedVersions versions{};
    bool capacity_reserved = false;      ///< an up-front capacity/activation reservation backs this entry
    bool resources_pinned = false;       ///< the capture/restore/return charges the activation will need are held
};

template <typename Owner>
concept RetainedRepairOwner = requires(const Owner& owner) {
    { owner.populated() } -> std::convertible_to<bool>;
    { owner.retired } -> std::convertible_to<bool>;
    { owner.yielded } -> std::convertible_to<bool>;
    { owner.controller != nullptr } -> std::convertible_to<bool>;
    { owner.published.key } -> std::convertible_to<const core::WindowKey&>;
    { owner.certified_cut } -> std::convertible_to<const core::StateCut&>;
    { owner.retained_debt } -> std::convertible_to<uint64_t>;
    { owner.retained_residency_swaps } -> std::convertible_to<uint64_t>;
    { owner.owner_evidence_snapshot() } -> std::convertible_to<RepairOwnerEvidence>;
};


template <typename Owner> requires RetainedRepairOwner<Owner>
class ServingRepairOwnerRegistry;

// Only the registry may issue this immutable copy, after it has checked the entry's actual reservations,
// pinned resources and current controller evidence. It does not change the controller's unreserved mirror.
class RetainedActivationAuthority final {
public:
    RetainedActivationAuthority() = default;
    bool issued() const noexcept { return issued_; }
    const RetainedRecordFacts& record() const noexcept { return record_; }
private:
    template <typename Owner> requires RetainedRepairOwner<Owner>
    friend class ServingRepairOwnerRegistry;
    explicit RetainedActivationAuthority(RetainedRecordFacts record) : record_(std::move(record)), issued_(true) {}
    RetainedRecordFacts record_{};
    bool issued_ = false;
};

// The concrete resources a later activation needs, admitted BEFORE the yield decision and pinned by the entry:
// the frozen private image the entry already holds, the capture of the live session's prior conversation state,
// and the restore the activation itself performs. The purposes are read back from the charges (the same shape as
// `OwnerReservationHandle`); `ReservationPurpose` already names `private_attachment` and `foreground_return`, so
// no new enumerator is introduced here.
struct RetainedActivationResources {
    std::shared_ptr<const void> image_charge, prior_capture_charge, restore_charge;
    core::ReservationPurpose image_purpose = core::ReservationPurpose::foreground;
    core::ReservationPurpose prior_capture_purpose = core::ReservationPurpose::foreground;
    core::ReservationPurpose restore_purpose = core::ReservationPurpose::foreground;
    uint64_t image_bytes = 0, prior_capture_bytes = 0, restore_bytes = 0;
    bool pinned() const noexcept {
        return bool(image_charge) && bool(prior_capture_charge) && bool(restore_charge);
    }
    bool purposes_correct() const noexcept {
        return image_purpose == core::ReservationPurpose::private_attachment &&
               prior_capture_purpose == core::ReservationPurpose::foreground_return &&
               restore_purpose == core::ReservationPurpose::private_attachment;
    }
    bool bytes_bounded() const noexcept {
        return image_bytes != 0 && prior_capture_bytes != 0 && restore_bytes != 0 &&
               image_bytes <= UINT64_MAX - prior_capture_bytes &&
               image_bytes + prior_capture_bytes <= UINT64_MAX - restore_bytes;
    }
};

// The live-side facts of one activation attempt. `live_all_stage_drained` is the drained all-stage boundary the
// activation is allowed at; the prior-state fields are what the activation reports back about the OTHER
// conversation, and the registry verifies them against the facts it was handed before the attempt ran.
struct RetainedActivationFacts {
    RetainedActivationAuthority authority{}; ///< issued only after the registry activation gate passes
    uint64_t record_version = 0;           ///< the OWNER RECORD's authoritative version, from the registry
    bool live_all_stage_drained = false;   ///< the live fence stands at a committed, drained, all-stage cut
    bool request_fence_open = false;       ///< the returning conversation holds the fence (capture/restore need it)
    // Derived by the registry from the captured state it was handed; a caller cannot assert them. The restore and
    // its verification are what the activation ATTEMPT reports (`ConversationStateFacts` out), compared by
    // `activate_parked_owner` against the facts captured before the attempt ran.
    bool prior_state_captured = false;
    // Slice 6 (reviewed contract extension, not a relaxation): the A->B->A case in which the serving scope has
    // ALREADY restored the owner's publication image onto the live transaction, so the prior conversation's state
    // is a RETAINED image captured at the session's earlier cut rather than at the cut the session stands on now.
    // The registry sets this only when the session now stands exactly on this entry's recorded publication; it can
    // never be asserted by a caller.
    bool prior_state_retained_image = false;
    core::StateCut prior_session_cut{};    ///< the session cut the captured prior state was taken at
};

// The captured prior conversation state, in the fields the exact comparator itself insists on verifying after a
// reset/replay: cut id, forwarded, accepted, exact_frontier, output_watermark, history digest, and the session's
// own cut. Nothing approximate is carried here, and nothing here claims exactness for the foreground suffix.
struct ConversationStateFacts {
    core::WindowKey key{};
    core::StateCut cut{};
    core::StateCut session_cut{};    ///< the adapter's observed session cut (ServingRawSession::cut())
    bool operator==(const ConversationStateFacts&) const = default;
    // The discriminating comparison: every field above, one by one, so a report that lost a watermark or a digest
    // byte cannot pass as "restored".
    static bool identical(const ConversationStateFacts& captured, const ConversationStateFacts& restored) noexcept {
        return restored.key == captured.key && restored.session_cut == captured.session_cut &&
               restored.cut.id == captured.cut.id &&
               restored.cut.forwarded == captured.cut.forwarded &&
               restored.cut.accepted == captured.cut.accepted &&
               restored.cut.exact_frontier == captured.cut.exact_frontier &&
               restored.cut.output_watermark == captured.cut.output_watermark &&
               restored.cut.history == captured.cut.history;
    }
};

// (b) Publish-if-current. Pure over the controller's own current evidence, the entry's record and the prepared
// result. Every refusal names its rule and leaves the owner parked and resumable: publication never mutates the
// live session, never retires, and never poisons.
inline bool retained_result_publishable(const PreparedRetainedResult& result, const RepairOwnerEvidence& current,
                                        const RetainedRecordFacts& record, RepairOwnerRefusal& why) noexcept {
    if (!result.produced()) { why = RepairOwnerRefusal::refused_prepared_result_absent; return false; }
    if (!current.produced()) { why = RepairOwnerRefusal::refused_evidence_absent; return false; }
    if (!current.suspended()) { why = RepairOwnerRefusal::refused_not_suspended; return false; }
    if (current.retired()) { why = RepairOwnerRefusal::refused_retired; return false; }
    if (current.finished()) { why = RepairOwnerRefusal::refused_finished; return false; }
    if (current.busy()) { why = RepairOwnerRefusal::refused_busy; return false; }
    // A cancelled owner never gains a published result: its continuation is not the one that was repaired.
    if (current.cancel_requested()) { why = RepairOwnerRefusal::refused_cancel_requested; return false; }
    if (current.live_phase() != core::serving_raw::Phase::drained) {
        why = RepairOwnerRefusal::refused_phase_not_drained; return false;
    }
    if (current.private_phase() != core::serving_raw::Phase::drained) {
        why = RepairOwnerRefusal::refused_private_phase_not_drained; return false;
    }
    if (!current.live_allocation_healthy() || !current.private_allocation_healthy()) {
        why = RepairOwnerRefusal::refused_allocation_unhealthy; return false;
    }
    // The versioned current check, in order: the record the result was prepared against, the identity, the
    // publication serial, the cuts, the debt, then the real content/append/selection/engine/residency versions.
    if (result.record_version() != record.record_version) {
        why = RepairOwnerRefusal::refused_record_version_stale; return false;
    }
    if (!(result.identity() == record.identity) || !(current.published_key() == record.identity)) {
        why = RepairOwnerRefusal::refused_identity_mismatch; return false;
    }
    if (result.published_serial() != record.published_serial ||
        result.published_serial() != current.published_serial()) {
        why = RepairOwnerRefusal::refused_record_changed; return false;
    }
    if (!(result.published_cut() == record.published_cut) || !(result.published_cut() == current.published_cut())) {
        why = RepairOwnerRefusal::refused_record_changed; return false;
    }
    // The certified cut must be the private chain the controller stands on RIGHT NOW, not the one the result was
    // prepared against before an intervening unit or edit.
    if (!(result.certified_cut() == current.private_cut()) || !(result.certified_cut() == record.certified_cut)) {
        why = RepairOwnerRefusal::refused_no_certified_cut; return false;
    }
    if (result.token_debt() != record.token_debt || result.token_debt() != current.token_debt()) {
        why = RepairOwnerRefusal::refused_debt_inconsistent; return false;
    }
    if (result.accepted_sequence() != current.private_cut().accepted ||
        result.output_watermark() != current.published_cut().output_watermark) {
        why = RepairOwnerRefusal::refused_prepared_result_invalid; return false;
    }
    if (!same_chain_versions(result.versions(), record.versions)) {
        // A stale edit: the accepted sequence (content digest) or an epoch moved under the owner, so the result no
        // longer describes the conversation the chain replays.
        why = RepairOwnerRefusal::refused_content_diverged; return false;
    }
    if (!current.certified_at_private_cut()) {
        why = RepairOwnerRefusal::refused_certified_cut_not_exact; return false;
    }
    if (!current.private_charge()) { why = RepairOwnerRefusal::refused_no_retained_charge; return false; }
    why = RepairOwnerRefusal::none;
    return true;
}

// (c) Activation. Stricter than publication: it is the one path allowed to make the owner's published state the
// live session state, so the live fence must be quiescent at a drained all-stage boundary, no request fence or
// private work window may be open, the live session must be this owner's stable window at its recorded
// publication, the residency generation must still be the one the chain is bound to, and the entry must carry the
// reservation and the capture/restore/return charges admitted before the yield.
inline bool retained_activation_allowed(const PreparedRetainedResult& result, const RepairOwnerEvidence& current,
                                        const RetainedRecordFacts& record, const LiveSessionFacts& session,
                                        const RetainedActivationFacts& activation, RepairOwnerRefusal& why) noexcept {
    if (!retained_result_publishable(result, current, record, why)) return false;   // everything (b) requires
    if (!record.capacity_reserved || !record.resources_pinned) {
        // A park that never admitted its later activation is a defect, so activation refuses by name and the
        // owner stays parked: the serving scope must retire it deliberately.
        why = RepairOwnerRefusal::refused_capacity_unreserved; return false;
    }
    // The boundary an activation is allowed at: the live transaction at a DRAINED, all-stage quiescent cut, with
    // the returning conversation's request fence open (RAW capture/restore are legal only there,
    // include/strata/core/serving_raw_contracts.hpp `capture`/`restore`), and no bounded private work window
    // standing in the way. Anything else refuses by name and mutates nothing.
    if (session.live_phase != core::serving_raw::Phase::drained || !activation.live_all_stage_drained) {
        why = RepairOwnerRefusal::refused_activation_not_drained; return false;
    }
    if (!activation.request_fence_open || !session.request_open) {
        why = RepairOwnerRefusal::refused_activation_fence_not_open; return false;
    }
    if (session.private_window_open) {
        why = RepairOwnerRefusal::refused_private_window_not_open; return false;   // a bounded unit is not a boundary
    }
    if (!same_stable_window(session.session_key, record.identity)) {
        why = RepairOwnerRefusal::refused_window_identity_diverged; return false;
    }
    // What the activation replaces must be accounted for. Either the live session still stands on exactly this
    // owner's recorded publication (A returns to the state it left), or the state that IS standing there has been
    // captured and the capture matches the live session cut - which is the A->B->A case: B's turns are retained by
    // the entry's `foreground_return` charge and can be brought back, so replacing them is reversible. An
    // uncaptured or mismatched replacement is a stale publication, refused before any mutation.
    const bool installed_cut_is_recorded_publication = session.session_cut == record.published_cut;
    const bool stands_on_owner_publication =
        installed_cut_is_recorded_publication && session.session_cut == current.published_cut();
    const bool replaces_a_captured_state =
        activation.prior_state_captured && activation.prior_session_cut == session.session_cut;
    // The restored case (slice 6): the scope brought the live transaction onto the owner's RECORDED publication -
    // every field of it, so an installed image that relabeled D (a different exact frontier, history digest,
    // accepted count or watermark is not the recorded publication and is refused here) - and the state being
    // displaced is a retained capture taken at an earlier session cut, which the entry's `foreground_return`
    // charge pays for and the attempt must restore and verify if the activation cannot land.
    const bool replaces_a_retained_image =
        activation.prior_state_retained_image && activation.prior_state_captured &&
        installed_cut_is_recorded_publication && activation.prior_session_cut != core::StateCut{};
    if (!stands_on_owner_publication && !replaces_a_captured_state && !replaces_a_retained_image) {
        why = RepairOwnerRefusal::refused_stale_publication; return false;
    }
    if (session.residency_generation != record.versions.residency_generation) {
        why = RepairOwnerRefusal::refused_residency_diverged; return false;
    }
    // The state being displaced has to be accounted for either way: captured at the cut the session stands on now
    // (nothing was restored), or captured earlier and RETAINED as an image while the scope restored this owner's
    // publication onto the session (the A->B->A case, slice 6). The second is only ever set by the registry, and
    // only when the session stands exactly on the entry's recorded publication; the attempt still has to restore
    // that retained image and verify it field by field, which is what `activate_parked_owner` checks afterwards.
    if (!activation.prior_state_captured && !activation.prior_state_retained_image) {
        why = RepairOwnerRefusal::refused_prior_state_unverified; return false;
    }
    // The initial supported profile, stated as a gate and not as a comment: activation lands the certified state
    // at the position the live conversation already stands on. It never moves the live position, because the
    // request's own history, its accepted sequence and its output watermark are all keyed to that position. A
    // chain that has not caught the published cut therefore cannot be activated: the serving scope keeps the owner
    // parked (or retires it), rather than rewinding the conversation under the request.
    if (result.certified_cut().forwarded != record.published_cut.forwarded ||
        record.certified_cut.forwarded != record.published_cut.forwarded) {
        why = RepairOwnerRefusal::refused_activation_state_diverged; return false;
    }
    // Slice 6 (reviewed contract change to the ACTIVATION boundary; the yield and park gates still demand real
    // debt and a certified R): what an activation installs is the owner's own publication cut D, which suffix
    // reuse leaves APPROXIMATE by construction. Exactness is therefore required of the certified private
    // checkpoint R - the committed all-stage witness the prepared result already speaks for - and never of D.
    // Requiring `D.exact_frontier == D.forwarded` here would make activation unreachable for the feature; what is
    // required instead, above, is that the installed cut is the recorded publication FIELD BY FIELD, which is what
    // refuses an installed image that claims an exactness the record does not carry or that dropped approximation
    // regions on the way in.
    if (result.certified_cut().exact_frontier != result.certified_cut().forwarded) {
        why = RepairOwnerRefusal::refused_certified_cut_not_exact; return false;
    }
    // And the activated state must carry THIS conversation's own pending token and output watermark: those are the
    // two fields the request continues from, so a result whose pending token or watermark differs is refused here
    // rather than installed.
    if (result.pending_token() < 0 || result.output_watermark() != record.published_cut.output_watermark) {
        why = RepairOwnerRefusal::refused_activation_state_diverged; return false;
    }
    why = RepairOwnerRefusal::none;
    return true;
}

// ---- the request-scope guard's decision table (generate.cpp:16783 SfxOpenFenceGuard) ----------------
// The guard's reviewed rule was: a request scope may not unwind holding a live controller. Slice 2 extends it:
// a request scope may also not unwind having LOST an owner it retained - the owner must be either retired or
// registered somewhere that can retire it. These are the only facts the guard can observe, and the table below
// is the whole decision, so the CPU cell exercises the same rule the server applies.
struct OwnerScopeFacts {
    bool active_holds_owner = false;          // *serving_suffix_active != nullptr
    bool active_holds_controller = false;     // (*serving_suffix_active)->controller != nullptr
    bool retention_recorded = false;          // this request certified a retained yield and named its identity
    core::WindowKey recorded_identity{};      // the identity the request parked the owner under
    bool registry_holds_record = false;       // the registry has an entry under that identity
};
enum class OwnerScopeEscape : uint8_t {
    inert,                       // nothing retained, nothing lost: today's guard does nothing
    failstop_live_unregistered,  // today's rule, unchanged: a live controller escaped the scope
    failstop_retained_lost,      // new rule: a retained yield is neither retired nor registered
};
inline const char* owner_scope_escape_name(OwnerScopeEscape escape) noexcept {
    switch (escape) {
        case OwnerScopeEscape::inert: return "inert";
        case OwnerScopeEscape::failstop_live_unregistered: return "failstop_live_unregistered";
        case OwnerScopeEscape::failstop_retained_lost: return "failstop_retained_lost";
    }
    return "unknown";
}
inline OwnerScopeEscape owner_scope_escape(const OwnerScopeFacts& facts) noexcept {
    if (facts.active_holds_owner && facts.active_holds_controller)
        return OwnerScopeEscape::failstop_live_unregistered;
    if (facts.retention_recorded && !facts.active_holds_owner && !facts.registry_holds_record)
        return OwnerScopeEscape::failstop_retained_lost;
    return OwnerScopeEscape::inert;
}

// ---- the idle-window demand observation ---------------------------------------------------------
// The three demand sources BOUNDED-BREAKDOWN section 2 names, mapped onto what the serving engine can actually
// observe, in one production function the serving loop calls and the CPU cell drives:
//   * the current request's own remaining generation - no serving request is open at the idle tick, and the
//     retained-owner flag says whether a request in flight would be generating through this owner;
//   * queued stdin lines (StdinShared::lines, generate.cpp:14427);
//   * admitted work in other lanes - the serving engine has no WindowAdmission reactor, so its other-lane state
//     IS the pipelined/batched prefill work in flight, plus an open RAW fence.
struct ServingIdleDemand {
    bool current_request_generating = false;
    uint64_t queued_input_lines = 0;
    bool other_lane_work = false;       // pipeline groups in flight, a batch step pending, admitted work
    bool raw_fence_running = false;     // an open serving RAW fence (serving_raw_fence_running)
    bool request_open = false;          // the session still has a request open
    bool context_room = false;          // published forwarded < max_context
    bool stop_or_eos_seen = false;
    bool owner_retired = false;
};
inline ServingForegroundState make_serving_foreground_state(const ServingIdleDemand& demand) noexcept {
    ServingForegroundState state;
    state.current_request_generating = demand.current_request_generating || demand.request_open ||
                                       demand.raw_fence_running;
    state.queued_input_lines = demand.queued_input_lines;
    state.context_room = demand.context_room;
    state.stop_or_eos_seen = demand.stop_or_eos_seen;
    state.repair_owner_retired = demand.owner_retired;
    return state;
}
// The idle tick's admission: one bounded private unit for a parked owner is admissible only when the kernel says
// `admitted_idle`, i.e. nothing is waiting and nothing else is eligible. No age escalation is ever considered
// here: the policy object carries no bound, so the escalation branch cannot open (that is the reviewed default,
// not a value chosen here), and no threshold appears in this file. `token_debt` is the entry's recorded debt -
// the kernel only echoes it, and `demand` is derived from it rather than asserted by a caller.
inline RepairAdmissionDecision serving_repair_idle_admission(const ServingIdleDemand& demand,
                                                             uint64_t token_debt,
                                                             bool resource_ready) noexcept {
    const ServingRepairAdmissionPolicy policy = ServingRepairAdmissionPolicy::review_required();
    const ServingForegroundState foreground = make_serving_foreground_state(demand);
    const RepairCandidateState candidate{core::DependencyWorkClass::live_repair, core::AdmissionUnit::repair,
                                         token_debt > 0, resource_ready, false, false};
    const RepairAgeInputs age{{}, 0, 0, 0, 0, token_debt};
    RepairAdmissionDecision decision =
        evaluate_serving_repair_admission(policy, foreground, candidate, age, demand.other_lane_work);
    // The kernel consults `other_nonmaintenance_demand` for the saved-maintenance class; for the live-repair
    // class the idle tick is stricter still: work already admitted in another lane (pipelined/batched prefill
    // groups in flight) is demand too, so an otherwise-admitted unit is refused with that named reason.  This is
    // the third demand source BOUNDED-BREAKDOWN section 2 requires; it never opens a gate, only closes one.
    if (demand.other_lane_work && decision.reason == RepairAdmissionReason::admitted_idle)
        decision.reason = RepairAdmissionReason::refused_other_eligible_work;
    return decision;
}

// ---- the serving scope's certified-yield gate (slice 3, D1) --------------------------------------
// The production gate sits in the serving request scope, where the only available facts are that request's own
// counters.  The predicate lives here so the CPU cell drives the SAME rule the server runs: generate.cpp builds
// these facts from its locals (`serving_repair_owners`, `cancelled`, `stop_req`, `produced_n`, `max_new`,
// `finish`, `use_mtp`, `o.suffix_draft`, `o.lookup_chain`, the qualification selector, `exact_ready`) and calls
// `serving_certified_yield_gate`.
//
// WHY THIS CONJUNCTION REPLACES THE SLICE-2 ONE (`SLICE2-DEFECTS-CONFIRMED.md` D1): the retained yield is an
// ALTERNATIVE to the forced drain and must never truncate requested output, so it belongs at the RESPONSE
// COMPLETION boundary.  Slice 2 also required `produced_n < max_new` and live context room, which contradicts
// every way the token loop can exit, so the yield was unreachable and a one-token response always drained.
//
// Nothing here ends generation early - the caller has already left the token loop - and nothing here states a
// controller-side condition: the drained coherent cut, the real debt and the committed all-stage witness for the
// private cut belong to `retained_yield_allowed` alone, over the evidence the controller produces about itself.
// `refused_qualification_arm` is now the D3 rule, not the D3 gap: in the arm that runs the strict frontier
// oracle, a yield is admissible ONLY when the dedicated retained-yield oracle comparator is admitted, i.e. only
// when the certified private cut is about to be compared against an independent clean recomputation and emitted
// as `yield_frontier`/`yield_continuation` (see `harness/ORACLE-DESIGN.md` and `qualify_serving_yield_frontier`).
// An uncertified yield is still refused in that arm; whether the emitted records actually certify the cut is a
// separate decision (`serving_yield_record_decision`). A refusal there is NOT recovered from: the comparison has
// already mutated the live session, so the established failstop/poison semantics apply, exactly as a failed or
// interrupted strict oracle ends the request today.
struct ServingYieldGateFacts {
    bool registry_present = false;           // serving_repair_owners != nullptr (STRATA_SERVING_REPAIR_OWNER_SLOTS)
    bool cancelled = false;                  // the request was cancelled
    bool stop_requested = false;             // stop_req.load(): a committed STOP or an input failure
    uint64_t produced_tokens = 0;            // produced_n at the token loop's exit
    uint64_t max_new_tokens = 0;             // max_new for this request
    bool ended_at_length_limit = false;      // finish == "length"
    bool ended_at_stop_token = false;        // finish == "stop" (EOS or a stop token)
    bool mtp_chain_active = false;           // use_mtp: a drafter chain owns the verifier path
    bool suffix_draft = false;               // o.suffix_draft > 0
    bool lookup_chain = false;               // o.lookup_chain > 0
    bool qualification_arm = false;          // SuffixExperimentDriver::serving_qualification_enabled()
    bool debt_open = false;                  // !exact_ready(operation): private cut behind the published cut
    // Slice 4 (D3): the retained-yield oracle comparator (yield_frontier/yield_continuation) is admissible in
    // this arm - pinned producer identity, admitted scratch, genuine request fence. It is an input to the gate
    // only so the strict-frontier arm cannot take an UNCERTIFIED yield; the verdict of the comparison itself is
    // decided afterwards by `serving_yield_record_decision`, which is what lets the yield stand.
    bool yield_oracle_admitted = false;
    // Slice 5 (requirement 4): the registry slot AND the concrete capture/restore/return charges a later
    // activation will need were admitted up front (`reserve_retained_capacity`). Without it, a yield would park an
    // owner whose activation can never be admitted, so the yield itself is refused and today's drain path runs.
    bool activation_capacity_reserved = false;
};
enum class ServingYieldRefusal : uint8_t {
    none,
    refused_selector_absent,        // no registry: today's drain-and-retire path, byte for byte
    refused_cancelled,              // an interrupted response is not a completion
    refused_stop_requested,         // ditto
    refused_response_not_complete,  // the loop exited somewhere else (e.g. the context limit): today's path
    refused_request_shape,          // MTP / suffix draft / lookup chain: the retained request shape is incoherent
    refused_qualification_arm,      // the oracle does not yet qualify a debt-retaining cut (D3)
    refused_activation_capacity,    // no up-front slot + capture/restore/return admission (slice 5)
    refused_no_debt,                // zero debt: that is the exact finish path, not a yield
};
inline const char* serving_yield_refusal_name(ServingYieldRefusal refusal) noexcept {
    switch (refusal) {
        case ServingYieldRefusal::none: return "admitted";
        case ServingYieldRefusal::refused_selector_absent: return "refused_selector_absent";
        case ServingYieldRefusal::refused_cancelled: return "refused_cancelled";
        case ServingYieldRefusal::refused_stop_requested: return "refused_stop_requested";
        case ServingYieldRefusal::refused_response_not_complete: return "refused_response_not_complete";
        case ServingYieldRefusal::refused_request_shape: return "refused_request_shape";
        case ServingYieldRefusal::refused_qualification_arm: return "refused_qualification_arm";
        case ServingYieldRefusal::refused_activation_capacity: return "refused_activation_capacity";
        case ServingYieldRefusal::refused_no_debt: return "refused_no_debt";
    }
    return "unknown";
}
// True (none) only for a response that finished normally - it produced everything it asked for, or it ended at a
// stop token - on a request shape the retained path can hold, with debt open.  Order is deliberate: the
// opt-in check first, so with no selector nothing else is even consulted.
inline ServingYieldRefusal serving_certified_yield_gate(const ServingYieldGateFacts& facts) noexcept {
    if (!facts.registry_present) return ServingYieldRefusal::refused_selector_absent;
    if (facts.cancelled) return ServingYieldRefusal::refused_cancelled;
    if (facts.stop_requested) return ServingYieldRefusal::refused_stop_requested;
    const bool response_complete =
        (facts.ended_at_length_limit && facts.produced_tokens >= facts.max_new_tokens) ||
        facts.ended_at_stop_token;
    if (!response_complete) return ServingYieldRefusal::refused_response_not_complete;
    if (facts.mtp_chain_active || facts.suffix_draft || facts.lookup_chain)
        return ServingYieldRefusal::refused_request_shape;
    if (facts.qualification_arm && !facts.yield_oracle_admitted)
        return ServingYieldRefusal::refused_qualification_arm;
    if (!facts.activation_capacity_reserved) return ServingYieldRefusal::refused_activation_capacity;
    if (!facts.debt_open) return ServingYieldRefusal::refused_no_debt;
    return ServingYieldRefusal::none;
}

// ---- the serving scope's yield-record decision (slice 4, D3) ---------------------------------------
// The pure predicate that decides whether a yield which has ALREADY been taken may stand as a CERTIFIED retained
// yield, over the facts of the records the comparator emitted for it. It is deliberately a second decision from
// the pre-yield gate: the emission order the harness contract requires is `retained_yield` progress ->
// `yield_frontier` -> `yield_continuation` -> one restoration record (ORACLE-DESIGN.md §2), so the yield must be
// taken and reported before its certificate exists. This function is what the serving scope calls on the
// comparator's own report, and what the CPU cell drives; the same rule, one definition.
//
// The facts are read from the emitted records, never supplied by a caller that wants the answer: a missing or
// half-emitted record, a cancelled comparison, a non-exact certified cut, a continuation that is not exactly one
// ordinary-loop window above it, an unpinned producer identity, or a debt that is not strictly positive each
// refuse with their own name. A refusal means the certified yield did not happen: it is reached only once the
// comparison already ran, so the serving scope takes the established failstop/poison path rather than recovering.
struct ServingYieldRecordFacts {
    bool pair_emitted = false;              // BOTH yield_frontier and yield_continuation records were written
    bool restoration_emitted = false;       // exactly one restoration record followed the pair
    bool restored = false;                  // that record reports the state restored (restored=1)
    bool cancelled = false;                 // cancelled=1: the comparison was interrupted
    bool completed = false;                 // completed=1 on both phase records
    bool payload_equal = false;             // equal=1: the repaired image matches the clean recomputation
    uint64_t mismatches = 0;                // mismatches=0 on both phase records
    bool frontier_exact = false;            // yield_frontier.exact_frontier == yield_frontier.forwarded
    bool continuation_exact = false;        // yield_continuation.exact_frontier == .forwarded
    bool identity_pinned = false;           // binary/source are the host-measured pinned hashes
    bool ordinal_bound = false;             // every record carries this request's own engine ordinal
    uint64_t published_cut = 0;             // D: the CURRENT cut the response ended on
    uint64_t certified_cut = 0;             // R: the private cut the controller certified and yielded
    uint64_t frontier_forwarded = 0;        // the pair's own cut fields, as emitted
    uint64_t continuation_forwarded = 0;
    uint64_t frontier_accepted = 0;
};
inline uint64_t serving_yield_record_debt(const ServingYieldRecordFacts& facts) noexcept {
    return facts.published_cut > facts.certified_cut ? facts.published_cut - facts.certified_cut : 0;
}
enum class ServingYieldRecordRefusal : uint8_t {
    none,
    refused_pair_absent,            // a yield with no certified-frontier/continuation pair: D3 again
    refused_restoration_absent,     // a pair with no restoration: the comparison never completed coherently
    refused_cancelled,              // an interrupted comparison cannot certify anything
    refused_not_completed,          // completed=0: the clean recomputation did not run to its end
    refused_payload_mismatch,       // equal=0 or mismatches!=0: the state is not the recomputed one
    refused_cut_not_exact,          // the certified cut is not exact at its own frontier (capture_exact's rule)
    refused_no_debt,                // certified cut >= published cut: the debt is hidden, not retained
    refused_pair_geometry,          // the continuation is not that certified state advanced by one window
    refused_identity_unpinned,      // the producer identity is not the pinned, host-measured pair
    refused_ordinal_unbound,        // the records are not attributed to this request
};
inline const char* serving_yield_record_refusal_name(ServingYieldRecordRefusal refusal) noexcept {
    switch (refusal) {
        case ServingYieldRecordRefusal::none: return "certified";
        case ServingYieldRecordRefusal::refused_pair_absent: return "refused_pair_absent";
        case ServingYieldRecordRefusal::refused_restoration_absent: return "refused_restoration_absent";
        case ServingYieldRecordRefusal::refused_cancelled: return "refused_cancelled";
        case ServingYieldRecordRefusal::refused_not_completed: return "refused_not_completed";
        case ServingYieldRecordRefusal::refused_payload_mismatch: return "refused_payload_mismatch";
        case ServingYieldRecordRefusal::refused_cut_not_exact: return "refused_cut_not_exact";
        case ServingYieldRecordRefusal::refused_no_debt: return "refused_no_debt";
        case ServingYieldRecordRefusal::refused_pair_geometry: return "refused_pair_geometry";
        case ServingYieldRecordRefusal::refused_identity_unpinned: return "refused_identity_unpinned";
        case ServingYieldRecordRefusal::refused_ordinal_unbound: return "refused_ordinal_unbound";
    }
    return "unknown";
}
// `none` only for a complete, ordered, mismatch-free pair that certifies a strictly-behind private cut of the
// published cut. The default-constructed value ("the comparator reported nothing") is refused.
inline ServingYieldRecordRefusal serving_yield_record_decision(const ServingYieldRecordFacts& facts) noexcept {
    if (!facts.pair_emitted) return ServingYieldRecordRefusal::refused_pair_absent;
    if (!facts.restoration_emitted) return ServingYieldRecordRefusal::refused_restoration_absent;
    if (facts.cancelled) return ServingYieldRecordRefusal::refused_cancelled;
    if (!facts.restored) return ServingYieldRecordRefusal::refused_restoration_absent;
    if (!facts.completed) return ServingYieldRecordRefusal::refused_not_completed;
    if (!facts.payload_equal || facts.mismatches != 0) return ServingYieldRecordRefusal::refused_payload_mismatch;
    if (!facts.frontier_exact || !facts.continuation_exact)
        return ServingYieldRecordRefusal::refused_cut_not_exact;
    if (serving_yield_record_debt(facts) == 0) return ServingYieldRecordRefusal::refused_no_debt;
    if (facts.frontier_forwarded != facts.certified_cut || facts.frontier_accepted != facts.certified_cut ||
        facts.continuation_forwarded != facts.certified_cut + 1)
        return ServingYieldRecordRefusal::refused_pair_geometry;
    if (!facts.identity_pinned) return ServingYieldRecordRefusal::refused_identity_unpinned;
    if (!facts.ordinal_bound) return ServingYieldRecordRefusal::refused_ordinal_unbound;
    return ServingYieldRecordRefusal::none;
}

// The fixed oracle vocabulary for the retained yield. One place spells it, so the emitter, the gate and the
// harness contract (harness/retained_yield_oracle.py YIELD_PROGRESS_PHASE / YIELD_PHASE_NAMES) cannot drift.
inline constexpr const char* kServingYieldProgressPhase = "retained_yield";
inline constexpr const char* kServingYieldFrontierPhase = "yield_frontier";
inline constexpr const char* kServingYieldContinuationPhase = "yield_continuation";

// ---- the idle window's own wake condition (slice 3, D2) -------------------------------------------
// One observation of the stdin queue, taken by the caller under the queue's OWN mutex (StdinShared::mu,
// generate.cpp:14425-14427) - the same mutex the detached reader notifies under when it appends a line
// (generate.cpp:14469) and the same one the ordinary input read waits on (generate.cpp:14671).  The serving idle
// loop asks this before EVERY private unit, so it needs no cadence constant, no sleep and no spin: when the
// answer says "demand", the loop exits into the ordinary blocking read, which is the wake-up.
// A committed STOP counts as demand even though the reader consumes it without queueing a line
// (generate.cpp:14650), which is why this is not a queue-depth check alone.
struct ServingIdleQueueSignal {
    uint64_t queued_input_lines = 0;   // StdinShared::lines.size()
    bool input_error = false;          // StdinShared::input_error
    bool end_of_input = false;         // StdinShared::eof
    bool stop_requested = false;       // StdinShared::stop_req
};
inline bool serving_input_demand_observed(const ServingIdleQueueSignal& signal) noexcept {
    return signal.queued_input_lines > 0 || signal.input_error || signal.end_of_input || signal.stop_requested;
}

// ---- the registry ------------------------------------------------------------------------------
// The requirement on a parkable owner, spelled with the real `ServingSuffixRepairOperation::Outcome` members
// (generate.cpp:3192-3230) so the serving instantiation needs no adapter: `controller` (the shared_ptr whose
// unretired drop is terminal), `retired`, `published` (whose `key` is the conversation identity), `populated()`,
// and the retention bookkeeping slice 2 adds (`yielded`, `certified_cut`, `retained_debt`), plus
// `owner_evidence_snapshot()` - the payload's own call into `ResidentRepairController::owner_evidence()`, which
// is the only route by which controller facts reach this header.

enum class RepairOwnerState : uint8_t { suspended, retiring };

// The up-front capacity admission `reserve_retained_capacity` issues and `park_retained_owner` consumes. It is
// issued BEFORE the yield/DONE decision, so a yield is never taken for a park that could not later be admitted,
// and it is the only way a slot can be claimed. An unissued or spent ticket is refused, so a caller cannot park
// around the reservation (requirement 4: reserve the slot and the activation's capture/restore/return charges up
// front, or refuse the yield up front).
class RetainedCapacityTicket final {
public:
    RetainedCapacityTicket() = default;
    // The registry-issued form: a nonce this registry recognises, bound to the window it reserved.
    RetainedCapacityTicket(uint64_t nonce, core::WindowKey identity)
        : nonce_(nonce), identity_(std::move(identity)) {}
    bool issued() const noexcept { return nonce_ != 0; }
    uint64_t nonce() const noexcept { return nonce_; }
    const core::WindowKey& identity() const noexcept { return identity_; }

private:
    uint64_t nonce_ = 0;
    core::WindowKey identity_{};
};

// One charge the entry pins. The purpose is READ FROM the lease (`ReservationLease::purpose()`), never supplied
// by a caller, and the lease itself comes from the controller's own private allocation, so the entry - not an
// anonymous scope - is what keeps the parked owner's `private_work` reservation alive.
struct OwnerReservationHandle {
    core::ReservationPurpose purpose;
    std::shared_ptr<const void> credit_or_lease;
    bool operator==(const OwnerReservationHandle&) const = default;
};

// Inputs to `park_retained_owner`. `owner` must come from `ResidentRepairController::owner_evidence()`; the
// registry refuses anything else. `session` must come from the live `ServingRawSession`. There is deliberately
// NO age, no clock reading and no reservation vector in here: age is folded in later by `observe_age` from a
// caller that read the ancestry, and the charge is part of the evidence.
struct ParkOwnerFacts {
    RepairOwnerEvidence owner{};
    LiveSessionFacts session{};
    std::optional<uint64_t> observed_at_ns;
    // The reservation admitted before the yield. A park without one is refused: an owner that can be parked but
    // can never be activated is a defect, not a retention.
    RetainedCapacityTicket capacity{};
    RetainedActivationResources resources{};
};

struct SuspendedOwnerFacts {
    core::WindowKey identity{};
    RepairOwnerState state = RepairOwnerState::suspended;
    bool quarantined = false;
    uint64_t published_serial = 0;
    core::StateCut published_cut{}, live_cut_at_park{}, certified_cut{};
    uint64_t token_debt = 0;
    core::serving_raw::Phase suspended_phase = core::serving_raw::Phase::absent;
    bool holds_live_controller = false;
    std::vector<core::ReservationPurpose> reservation_purposes;
    std::optional<uint64_t> observed_at_ns;
    // Slice 5: the versioned record, and what has been admitted/published onto it.
    uint64_t record_version = 0;
    RetainedVersions versions{};
    bool capacity_reserved = false;
    bool activation_resources_pinned = false;
    bool result_published = false;
    PreparedRetainedResult published_result{};
    bool operator==(const SuspendedOwnerFacts&) const = default;
};

template <typename Owner>
    requires RetainedRepairOwner<Owner>
class ServingRepairOwnerRegistry final {
public:
    explicit ServingRepairOwnerRegistry(std::size_t max_suspended_owners) noexcept
        : max_suspended_owners_(max_suspended_owners) {}
    ServingRepairOwnerRegistry(const ServingRepairOwnerRegistry&) = delete;
    ServingRepairOwnerRegistry& operator=(const ServingRepairOwnerRegistry&) = delete;
    ServingRepairOwnerRegistry(ServingRepairOwnerRegistry&&) = delete;
    ServingRepairOwnerRegistry& operator=(ServingRepairOwnerRegistry&&) = delete;
    ~ServingRepairOwnerRegistry() {
        std::string error;
        if (!close(error))
            owner_registry_failstop("retained repair owner registry closed with an unretired controller parked");
    }

    std::size_t capacity() const noexcept { return max_suspended_owners_; }
    std::size_t size() const noexcept { return entries_.size(); }
    std::size_t owners_released() const noexcept { return released_owners_; }
    uint64_t units_advanced() const noexcept { return units_advanced_; }
    uint64_t tokens_repaired() const noexcept { return tokens_repaired_; }
    uint64_t activations_completed() const noexcept { return activations_completed_; }
    uint64_t results_published() const noexcept { return results_published_; }
    std::size_t outstanding_reservations() const noexcept { return tickets_.size(); }
    bool enabled() const noexcept { return max_suspended_owners_ > 0; }
    RepairOwnerRefusal last_refusal() const noexcept { return last_refusal_; }
    bool contains(const core::WindowKey& window) const noexcept { return find(window) != nullptr; }
    // The parked identities, in entry order. The idle tick walks them; it cannot reach a payload through this,
    // only through `advance_parked_owner`/`retire_owner`, which re-validate the entry each time.
    std::vector<core::WindowKey> parked_identities() const {
        std::vector<core::WindowKey> identities;
        identities.reserve(entries_.size());
        for (const Entry& entry : entries_) identities.push_back(entry.identity);
        return identities;
    }

    // Claim a slot and the activation's concrete resources BEFORE the yield decision (requirement 4). Bounds:
    // at most one outstanding ticket per free slot, tickets plus entries never exceed `capacity()`, and each
    // ticket pins exactly three charges (image, prior-state capture, restore) - never a growing set. A refusal
    // issues nothing, so the serving scope takes today's drain path and the response is unaffected.
    bool reserve_retained_capacity(const core::WindowKey& window, const RetainedActivationResources& resources,
                                   RetainedCapacityTicket& out, std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        out = RetainedCapacityTicket{};
        if (max_suspended_owners_ == 0)
            return fail(error, RepairOwnerRefusal::refused_selector_absent,
                        "the retained owner selector was absent, so no capacity is reserved");
        if (!resources.pinned() || !resources.purposes_correct() || !resources.bytes_bounded())
            return fail(error, RepairOwnerRefusal::refused_activation_resources_invalid,
                        "the activation needs its image, prior-state-capture and restore charges pinned, each with "
                        "the purpose its lease carries (private_attachment / foreground_return) and a real quote");
        if (find(window) != nullptr)
            return fail(error, RepairOwnerRefusal::refused_duplicate_identity,
                        "an owner is already parked under this WindowKey");
        for (const Entry& entry : entries_)
            if (same_stable_window(entry.identity, window))
                return fail(error, RepairOwnerRefusal::refused_duplicate_window,
                            "another live owner of this window is already parked");
        if (entries_.size() + tickets_.size() >= max_suspended_owners_)
            return fail(error, RepairOwnerRefusal::refused_full,
                        "no slot remains for a retained owner AND its activation resources: refusing the yield "
                        "before it is taken, rather than parking an owner that can never be activated");
        const uint64_t nonce = next_nonce_++;
        if (nonce == 0) return fail(error, RepairOwnerRefusal::refused_full, "ticket identity exhausted");
        tickets_.push_back(nonce);
        out = RetainedCapacityTicket(nonce, window);
        return true;
    }
    // Release an outstanding reservation when the serving scope does NOT park (a refused controller-side yield,
    // or an oracle that took the failstop path): the slot becomes available again for a later response.
    bool release_retained_capacity(const RetainedCapacityTicket& ticket, std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        if (!ticket.issued()) return fail(error, RepairOwnerRefusal::refused_capacity_unreserved, "no ticket");
        const auto found = std::find(tickets_.begin(), tickets_.end(), ticket.nonce());
        if (found == tickets_.end())
            return fail(error, RepairOwnerRefusal::refused_capacity_unreserved,
                        "the ticket was already consumed or never issued by this registry");
        tickets_.erase(found);
        return true;
    }

    // Move a LIVE, unretired, debt-retaining owner - just certified at a non-zero-debt cut by
    // `ResidentRepairController::yield_retained` - out of the serving request's slot and into the registry.
    // On success `active` is empty, which is what makes `SfxOpenFenceGuard` inert for this request; the entry
    // now owns the payload and pins its charge. Every refusal leaves `active` exactly as it was, so the caller
    // can retire deliberately (it must not simply drop a live controller).
    bool park_retained_owner(const core::WindowKey& window, std::unique_ptr<Owner>& active,
                             const ParkOwnerFacts& facts, std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        if (max_suspended_owners_ == 0)
            return fail(error, RepairOwnerRefusal::refused_selector_absent,
                        "the retained owner selector was absent, so nothing is retained");
        if (!active)
            return fail(error, RepairOwnerRefusal::refused_empty_owner_slot, "no owner in the active slot");
        if (!active->populated())
            return fail(error, RepairOwnerRefusal::refused_unpopulated_owner,
                        "an empty outcome carries no repair owner");
        if (active->controller == nullptr)
            return fail(error, RepairOwnerRefusal::refused_retired,
                        "only a live, unretired owner can be retained: a retired one has nothing to resume");
        if (active->retired)
            return fail(error, RepairOwnerRefusal::refused_retired,
                        "the payload reports retirement while still holding its controller");
        if (!active->yielded)
            return fail(error, RepairOwnerRefusal::refused_not_suspended,
                        "the owner was never certified at a retained yield cut");
        if (!(window == active->published.key))
            return fail(error, RepairOwnerRefusal::refused_identity_not_publication,
                        "the requested window is not the window the owner published");
        if (find(window) != nullptr)
            return fail(error, RepairOwnerRefusal::refused_duplicate_identity,
                        "an owner is already parked under this WindowKey");
        for (const Entry& entry : entries_)
            if (same_stable_window(entry.identity, window))
                return fail(error, RepairOwnerRefusal::refused_duplicate_window,
                            "another live owner of this window is already parked: a second private gang for one "
                            "conversation is not admitted");
        if (entries_.size() >= max_suspended_owners_)
            return fail(error, RepairOwnerRefusal::refused_full, "the registry is at its configured slot bound");
        // Requirement 4, enforced at the only entry point: a park without an up-front reservation is refused, so
        // there is no state in this tree where an owner is retained but its activation can never be admitted.
        if (!facts.capacity.issued() || !(facts.capacity.identity() == window))
            return fail(error, RepairOwnerRefusal::refused_capacity_unreserved,
                        "the serving scope reserved no capacity for this window before the yield");
        const auto ticket = std::find(tickets_.begin(), tickets_.end(), facts.capacity.nonce());
        if (ticket == tickets_.end())
            return fail(error, RepairOwnerRefusal::refused_capacity_unreserved,
                        "the capacity ticket is not one this registry issued and has not consumed");
        if (!facts.resources.pinned() || !facts.resources.purposes_correct() || !facts.resources.bytes_bounded())
            return fail(error, RepairOwnerRefusal::refused_activation_resources_invalid,
                        "the parked entry must pin the capture/restore/return charges its activation will need");
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        if (!retained_park_allowed(facts.owner, RepairOwnerRefusal::none, why))
            return fail(error, why, "the controller's own facts refuse a retained park");
        if (!(facts.owner.published_key() == window))
            return fail(error, RepairOwnerRefusal::refused_identity_not_publication,
                        "the controller no longer publishes the identity being parked under");
        if (!(facts.owner.published_cut() == active->published.cut))
            return fail(error, RepairOwnerRefusal::refused_record_changed,
                        "the controller's publication is not the payload's publication");
        if (!(facts.owner.private_cut() == active->certified_cut))
            return fail(error, RepairOwnerRefusal::refused_no_certified_cut,
                        "the payload's certified cut is not the controller's private cut");
        if (facts.session.request_open)
            return fail(error, RepairOwnerRefusal::refused_request_not_closed,
                        "the serving request fence is still open: the owner was not released");
        if (facts.session.live_phase != core::serving_raw::Phase::drained)
            return fail(error, RepairOwnerRefusal::refused_phase_not_drained,
                        "the session is not at the drained cut the park requires");
        if (!same_stable_window(facts.session.session_key, window)) {
            std::fprintf(stderr, "park_window_diag: session[gen=%llu/%llu sid=%s tid=%s wid=%s rev=%llu"
                         " prof=%02x%02x%02x topo=%02x%02x%02x] owner[gen=%llu/%llu sid=%s tid=%s wid=%s"
                         " rev=%llu prof=%02x%02x%02x topo=%02x%02x%02x]\n",
                         (unsigned long long)facts.session.session_key.engine_generation,
                         (unsigned long long)facts.session.session_key.window_generation,
                         facts.session.session_key.session_id.c_str(),
                         facts.session.session_key.thread_id.c_str(),
                         facts.session.session_key.window_id.c_str(),
                         (unsigned long long)facts.session.session_key.revision,
                         facts.session.session_key.model_profile[0],
                         facts.session.session_key.model_profile[1],
                         facts.session.session_key.model_profile[2],
                         facts.session.session_key.topology[0],
                         facts.session.session_key.topology[1],
                         facts.session.session_key.topology[2],
                         (unsigned long long)window.engine_generation,
                         (unsigned long long)window.window_generation,
                         window.session_id.c_str(), window.thread_id.c_str(),
                         window.window_id.c_str(), (unsigned long long)window.revision,
                         window.model_profile[0], window.model_profile[1], window.model_profile[2],
                         window.topology[0], window.topology[1], window.topology[2]);
            return fail(error, RepairOwnerRefusal::refused_window_identity_diverged,
                        "the live session is a different window than the owner published into");
        }
        Entry entry;
        entry.identity = window;
        entry.state = RepairOwnerState::suspended;
        entry.suspended_phase = facts.session.live_phase;
        entry.published_serial = facts.owner.published_serial();
        entry.published_cut = facts.owner.published_cut();
        entry.live_cut_at_park = facts.owner.live_cut();
        entry.certified_cut = facts.owner.private_cut();
        entry.token_debt = facts.owner.token_debt();
        entry.observed_at_ns = facts.observed_at_ns;
        entry.charge = OwnerReservationHandle{facts.owner.private_charge_purpose(), facts.owner.private_charge()};
        // The versioned record. `residency_generation` is the PAYLOAD's own recorded baseline (the expert-residency
        // inventory its private chain was initialised against), never a caller's claim; the live side is compared
        // against it before any further private work and before an activation.
        entry.residency_generation = active->retained_residency_swaps;
        entry.versions = make_retained_versions(window, entry.certified_cut, entry.residency_generation);
        entry.record_version = 1;
        entry.capacity_nonce = facts.capacity.nonce();
        entry.resources = facts.resources;
        entry.owner = std::move(active);
        tickets_.erase(ticket);
        entries_.push_back(std::move(entry));
        return true;
    }

    // One bounded private unit for a parked owner, through the payload's own controller path.
    // `unit(payload, before_evidence, after_evidence, advanced_tokens, error)` reads the controller's evidence
    // before the work and runs the real `ResidentRepairController::retained_private_unit`; both evidence values
    // come from the controller itself. The registry validates the entry against the BEFORE evidence (so a record
    // that no longer matches the live controller is refused rather than trusted), validates that the session
    // really is in a private work window with no request open, and validates the delivered progress afterwards
    // (monotonic, within the cap, never moving the live publication). Nothing in here can publish: publication
    // stays behind the controller's coherence gates, and `publish_gate` is what refuses it once the live cut
    // moved on.
    template <typename Unit>
    bool advance_parked_owner(const core::WindowKey& window, const LiveSessionFacts& session,
                              uint64_t cap_tokens, Unit&& unit, RepairOwnerEvidence& after_out,
                              uint64_t& advanced_out, std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        advanced_out = 0;
        after_out = RepairOwnerEvidence{};
        Entry* entry = find(window);
        if (entry == nullptr)
            return fail(error, RepairOwnerRefusal::refused_no_entry, "no owner is parked for this window");
        if (entry->state != RepairOwnerState::suspended)
            return fail(error, RepairOwnerRefusal::refused_state_not_suspended, "the entry is mid-retirement");
        if (entry->quarantined)
            return fail(error, RepairOwnerRefusal::refused_quarantined,
                        "an earlier unit left this record unverifiable: retire it, never advance it");
        if (entry->owner == nullptr || entry->owner->controller == nullptr)
            return fail(error, RepairOwnerRefusal::refused_retired, "the parked owner has no live controller");
        if (session.request_open)
            return fail(error, RepairOwnerRefusal::refused_request_not_closed,
                        "a foreground request holds the session: this is not an idle window");
        if (!session.private_window_open)
            return fail(error, RepairOwnerRefusal::refused_private_window_not_open,
                        "the serving lifecycle refused the private work window for this tick");
        if (!same_stable_window(session.session_key, entry->identity))
            return fail(error, RepairOwnerRefusal::refused_window_identity_diverged,
                        "the live session is not this owner's window");
        // The declared initial supported residency profile (slice 5, requirement 5): the parked owner's private
        // chain computes against the expert-residency inventory uploaded when its gang was initialised, so a unit
        // is admitted only while the live inventory generation still equals that baseline. The exact condition
        // under which the owner becomes eligible again is therefore `session.residency_generation ==
        // record.residency_generation`, i.e. no residency swap has been published since the park. A swap is
        // refused by name, the owner stays parked and resumable, and no chain is rebound in this pass.
        if (session.residency_generation != entry->residency_generation)
            return fail(error, RepairOwnerRefusal::refused_residency_diverged,
                        "expert residency changed since this owner was parked: no rebind is implemented, so the "
                        "chain is refused rather than run against a stale inventory");
        if (!(session.live_phase == entry->suspended_phase))
            return fail(error, RepairOwnerRefusal::refused_phase_not_drained,
                        "the session phase is no longer the phase this owner was parked at");
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        // The BEFORE evidence is the controller speaking about itself: the entry's record is what gets checked
        // against it, so a record that no longer matches cannot vouch for the work it is about to authorise.
        const RepairOwnerEvidence before = unit_evidence(*entry->owner);
        if (!before.produced())
            return fail(error, RepairOwnerRefusal::refused_evidence_absent,
                        "the parked owner produced no self-consistent evidence about itself");
        if (!entry_record_matches(*entry, before, why))
            return fail(error, why, "the parked record no longer matches the live controller");
        uint64_t target = 0;
        if (!retained_private_unit_allowed(before, cap_tokens, target, why))
            return fail(error, why, "the controller's own facts refuse a bounded private unit");
        RepairOwnerEvidence after{};
        uint64_t advanced = 0;
        std::string unit_error;
        bool performed = false;
        try {
            performed = unit(*entry->owner, advanced, after, unit_error);
        } catch (...) {
            entry->quarantined = true;   // the unit may have run: the record is no longer trustworthy
            throw;
        }
        if (!performed) {
            // A refused unit must not have moved anything; if it did, the record is unverifiable from here on.
            const RepairOwnerEvidence observed = unit_evidence(*entry->owner);
            if (observed.produced() && observed.private_cut() != before.private_cut()) entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_private_unit, unit_error);
        }
        if (!after.produced()) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_evidence_invalid,
                        "the unit returned no self-consistent controller evidence");
        }
        if (after.retired() || after.finished()) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_retired,
                        "a bounded private unit may not retire or finish the owner");
        }
        if (!(after.published_cut() == before.published_cut()) ||
            after.published_serial() != before.published_serial()) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_live_state_moved,
                        "a private unit moved the live publication: retention never republishes CURRENT");
        }
        if (after.private_cut().forwarded < before.private_cut().forwarded) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_no_progress, "private progress moved backwards");
        }
        const uint64_t gained = after.private_cut().forwarded - before.private_cut().forwarded;
        if (gained == 0 || advanced == 0) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_no_progress, "the unit advanced nothing");
        }
        if (gained > cap_tokens || advanced > cap_tokens || advanced != gained) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_unit_too_large,
                        "the unit exceeded the selector's cap or its own reported progress");
        }
        if (after.private_cut().forwarded > after.published_cut().forwarded) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_debt_inconsistent,
                        "private progress passed the published cut");
        }
        if (!after.certified_at_private_cut()) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_no_certified_cut,
                        "the unit left no committed witness for its own cut");
        }
        entry->certified_cut = after.private_cut();
        entry->token_debt = after.token_debt();
        entry->published_serial = after.published_serial();
        // The chain moved, so any retained result published for the previous certified cut is no longer current:
        // the versioned record is bumped and the result dropped. That is what makes a later publish-if-current
        // prepared against the old version refuse with `refused_record_version_stale`.
        entry->versions = make_retained_versions(entry->identity, entry->certified_cut,
                                                 entry->residency_generation);
        entry->published_result = {};
        ++entry->record_version;
        ++units_advanced_;
        tokens_repaired_ += gained;
        after_out = after;
        advanced_out = gained;
        return true;
    }

    // (b) Publish-if-current into THIS owner's record only. The prepared result must come from the payload's own
    // controller; the registry re-reads the controller's evidence and runs the shared gate, so a caller cannot
    // publish a result the controller does not currently stand on. A refusal changes nothing: the entry keeps its
    // payload, its charge and its reservation, stays resumable, and names the rule that refused. Nothing here
    // touches the live session.
    bool publish_retained_if_current(const core::WindowKey& window, const PreparedRetainedResult& result,
                                     PreparedRetainedResult& published_out, std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        published_out = {};
        Entry* entry = find(window);
        if (entry == nullptr)
            return fail(error, RepairOwnerRefusal::refused_no_entry, "no owner is parked for this window");
        if (entry->state != RepairOwnerState::suspended)
            return fail(error, RepairOwnerRefusal::refused_state_not_suspended, "the entry is mid-retirement");
        if (entry->quarantined)
            return fail(error, RepairOwnerRefusal::refused_quarantined,
                        "an earlier unit left this record unverifiable: retire it, never publish onto it");
        if (entry->owner == nullptr || entry->owner->controller == nullptr)
            return fail(error, RepairOwnerRefusal::refused_retired, "the parked owner has no live controller");
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        const RepairOwnerEvidence current = unit_evidence(*entry->owner);
        if (!current.produced())
            return fail(error, RepairOwnerRefusal::refused_evidence_absent,
                        "the parked owner produced no self-consistent evidence about itself");
        if (!entry_record_matches(*entry, current, why))
            return fail(error, why, "the parked record no longer matches the live controller");
        if (!retained_result_publishable(result, current, record_of(*entry), why))
            return fail(error, why, "the retained result is not current for this owner record");
        // The record version tracks the CHAIN (park, then one bump per bounded unit): a published result is
        // stamped with the version it was prepared against, and any unit that moves the chain drops it. Publishing
        // the same current result again is therefore idempotent, while a result prepared before a unit can never
        // be published onto the moved record.
        entry->published_result = result;
        ++results_published_;
        published_out = result;
        return true;
    }

    // (b') Prepare-on-behalf: the registry asks the PARKED PAYLOAD to state its own retained result, and publishes
    // what the payload produces onto the record. This is the only route from a parked owner to an activatable
    // record: `park_retained_owner` MOVED the payload out of the request slot, so the serving scope holds neither
    // the controller nor the frozen image and cannot hand `publish_retained_if_current` a result without
    // fabricating controller-only facts (the committed all-stage witness and the held image) - which is what
    // `refused_prepared_result_absent` refuses, before any attempt runs. `prepare(payload, record_version,
    // result_out, error)` is the payload's own path into `ResidentRepairController::prepare_retained_result`: the
    // registry supplies the payload AND the record version, so the version authority stays here and every fact
    // stays the controller's. The gate is exactly the one the caller-supplied form runs, with the controller's
    // evidence re-read AFTER the payload spoke, so a record that drifted while preparing is refused. Every refusal
    // changes nothing: the entry keeps its payload, charge and reservation, stays parked and resumable.
    template <typename Prepare>
    bool publish_retained_result_from_owner(const core::WindowKey& window, Prepare&& prepare,
                                            PreparedRetainedResult& published_out, std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        published_out = {};
        Entry* entry = find(window);
        if (entry == nullptr)
            return fail(error, RepairOwnerRefusal::refused_no_entry, "no owner is parked for this window");
        if (entry->state != RepairOwnerState::suspended)
            return fail(error, RepairOwnerRefusal::refused_state_not_suspended, "the entry is mid-retirement");
        if (entry->quarantined)
            return fail(error, RepairOwnerRefusal::refused_quarantined,
                        "an earlier unit left this record unverifiable: retire it, never publish onto it");
        if (entry->owner == nullptr || entry->owner->controller == nullptr)
            return fail(error, RepairOwnerRefusal::refused_retired, "the parked owner has no live controller");
        // The payload speaks first, into a result the caller cannot pre-seed: `prepared` starts empty here and the
        // only version it can carry is the one this method passed.
        PreparedRetainedResult prepared{};
        std::string prepare_error;
        try {
            if (!prepare(*entry->owner, entry->record_version, prepared, prepare_error)) {
                // The controller refused to state a result about itself. Nothing was published and nothing moved.
                return fail(error, RepairOwnerRefusal::refused_prepared_result_invalid,
                            prepare_error.empty() ? "the parked owner refused to state a retained result"
                                                  : prepare_error);
            }
        } catch (...) {
            entry->quarantined = true;   // the payload may have been touched: the record is not trustworthy
            throw;
        }
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        const RepairOwnerEvidence current = unit_evidence(*entry->owner);
        if (!current.produced())
            return fail(error, RepairOwnerRefusal::refused_evidence_absent,
                        "the parked owner produced no self-consistent evidence about itself");
        if (!entry_record_matches(*entry, current, why))
            return fail(error, why, "the parked record no longer matches the live controller");
        if (!retained_result_publishable(prepared, current, record_of(*entry), why))
            return fail(error, why, "the result the owner stated is not current for this owner record");
        entry->published_result = prepared;
        ++results_published_;
        published_out = prepared;
        return true;
    }

    // (c) Activate a parked owner: make its published state the live session state, at a drained all-stage
    // boundary only. `attempt(payload, activation, after, prior_restored, error)` runs the real
    // `ResidentRepairController::activate_retained` on the serving path - capture the live session's prior
    // conversation state, restore the owner's certified image, verify, and on any refusal restore the prior state
    // and report it. Admissibility is decided BEFORE the attempt, so nothing mutates on a refusal; the registry
    // then verifies what the attempt reports. On success the owner leaves the registry: its payload and every
    // pinned charge go back to the serving scope, which now owns a live unretired controller again and must retire
    // it deliberately. On a refusal or a failed verification the entry stays parked and keeps its charge, and can
    // be advanced or retired - never dropped, never silently drained.
    template <typename Activation>
    bool activate_parked_owner(const core::WindowKey& window, const LiveSessionFacts& session,
                               const ConversationStateFacts& prior_state, Activation&& attempt,
                               std::unique_ptr<Owner>& active_out, ConversationStateFacts& prior_restored_out,
                               RepairOwnerEvidence& after_out, std::vector<OwnerReservationHandle>& charges_out,
                               std::string& error) {
        // `active_out` receives the payload back: an activated owner is a LIVE controller again, so the request
        // scope that owns it must be the one that retires it (or failstops if it unwinds holding it).
        last_refusal_ = RepairOwnerRefusal::none;
        after_out = {};
        prior_restored_out = {};
        charges_out.clear();
        active_out.reset();
        Entry* entry = find(window);
        if (entry == nullptr)
            return fail(error, RepairOwnerRefusal::refused_no_entry, "no owner is parked for this window");
        if (entry->state != RepairOwnerState::suspended)
            return fail(error, RepairOwnerRefusal::refused_state_not_suspended, "the entry is mid-retirement");
        if (entry->quarantined)
            return fail(error, RepairOwnerRefusal::refused_quarantined,
                        "an earlier unit left this record unverifiable: retire it, never activate it");
        if (entry->owner == nullptr || entry->owner->controller == nullptr)
            return fail(error, RepairOwnerRefusal::refused_retired, "the parked owner has no live controller");
        if (!entry->published_result.produced())
            return fail(error, RepairOwnerRefusal::refused_prepared_result_absent,
                        "no retained result has been published onto this owner record: nothing to activate");
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        const RepairOwnerEvidence before = unit_evidence(*entry->owner);
        if (!before.produced())
            return fail(error, RepairOwnerRefusal::refused_evidence_absent,
                        "the parked owner produced no self-consistent evidence about itself");
        if (!entry_record_matches(*entry, before, why))
            return fail(error, why, "the parked record no longer matches the live controller");
        RetainedActivationFacts activation;
        activation.record_version = entry->record_version;
        activation.live_all_stage_drained = session.live_phase == core::serving_raw::Phase::drained;
        activation.request_fence_open = session.request_open;
        activation.prior_session_cut = prior_state.session_cut;
        // The prior conversation state must exist and be coherent before an activation can be admitted: an
        // activation that cannot say what it is replacing is refused before anything mutates.
        const bool prior_is_coherent =
            prior_state.cut.id != 0 && core::serving_raw::digest_present(prior_state.cut.history) &&
            same_stable_window(prior_state.key, session.session_key);
        activation.prior_state_captured = prior_is_coherent && prior_state.session_cut == session.session_cut;
        // The restored case (slice 6): the scope already put this owner's recorded publication onto the live
        // transaction, so a coherent capture taken at an EARLIER session cut is a retained image rather than the
        // state now standing there. Only the registry can say so, and only from the entry's own recorded
        // publication: a session that does not stand exactly on it is not a restored session.
        activation.prior_state_retained_image =
            prior_is_coherent && prior_state.session_cut != session.session_cut &&
            session.session_cut == entry->published_cut && session.session_cut == before.published_cut();
        if (!retained_activation_allowed(entry->published_result, before, record_of(*entry), session, activation,
                                         why))
            return fail(error, why, "the parked owner may not be activated at this boundary");
        activation.authority = RetainedActivationAuthority(record_of(*entry));
        RepairOwnerEvidence after{};
        ConversationStateFacts restored{};
        std::string attempt_error;
        bool performed = false;
        try {
            performed = attempt(*entry->owner, activation, after, restored, attempt_error);
        } catch (...) {
            entry->quarantined = true;   // the restore may have run: the record is no longer trustworthy
            throw;
        }
        const bool restored_verifies = ConversationStateFacts::identical(prior_state, restored);
        if (!performed) {
            // The activation did not land: the other conversation's state must be exactly what it was, and the
            // owner must still be parked. A restore that does not verify is the named refusal, and the entry is
            // quarantined because the live state is no longer provable - it is never dropped or drained silently.
            if (!restored_verifies) {
                entry->quarantined = true;
                return fail(error, RepairOwnerRefusal::refused_prior_state_unverified,
                            attempt_error.empty() ? "the refused activation did not restore the prior conversation state exactly"
                                                 : "the refused activation did not restore the prior conversation state exactly: " + attempt_error);
            }
            if (after.produced() && after.suspended() && after.private_cut() == entry->certified_cut) {
                // Nothing moved: the owner stays parked and resumable, and may be advanced or retired later.
                prior_restored_out = restored;
                return fail(error, RepairOwnerRefusal::refused_private_unit,
                            attempt_error.empty() ? "the controller refused the activation" : attempt_error);
            }
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_activation_state_diverged,
                        attempt_error.empty() ? "a refused activation left the owner in a state the record cannot vouch for"
                                              : "a refused activation left the owner in a state the record cannot vouch for: " + attempt_error);
        }
        // Landed: verify the activated state field by field against the record it claims to be.
        if (!after.produced() || after.suspended() || after.retired() || after.finished() ||
            after.cancel_requested()) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_activation_state_diverged,
                        "the activation reported success on an owner that is not live, unretired and uncancelled");
        }
        // What must be true after a landed activation: the live session stands on the RESULT this record
        // publishes, the owner's own coherence conjunction holds (which is what lets the untouched
        // `resume_retained` clear the suspended state), and the debt is closed because the activated state IS the
        // certified one. Anything else is a divergence the record cannot vouch for.
        if (!(after.published_key() == entry->identity) ||
            !(after.published_cut() == entry->published_cut) ||
            after.published_serial() != entry->published_serial ||
            after.private_cut().forwarded != entry->published_cut.forwarded ||
            after.published_cut().output_watermark != entry->published_result.output_watermark() ||
            !(after.live_cut() == after.published_cut()) || after.token_debt() != 0 || !after.live_coherent()) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_activation_state_diverged,
                        "the activated state is not the certified result this owner record publishes");
        }
        if (!same_stable_window(restored.key, prior_state.key)) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_prior_state_unverified,
                        "the activation reported a prior state for another window");
        }
        prior_restored_out = restored;
        after_out = after;
        active_out = std::move(entry->owner);
        if (active_out == nullptr) {
            entry->quarantined = true;
            return fail(error, RepairOwnerRefusal::refused_activation_state_diverged,
                        "the activated payload is not there to hand back: the owner would be lost");
        }
        charges_out.push_back(entry->charge);
        if (entry->resources.pinned()) {
            charges_out.push_back(OwnerReservationHandle{entry->resources.image_purpose,
                                                         entry->resources.image_charge});
            charges_out.push_back(OwnerReservationHandle{entry->resources.prior_capture_purpose,
                                                         entry->resources.prior_capture_charge});
            charges_out.push_back(OwnerReservationHandle{entry->resources.restore_purpose,
                                                         entry->resources.restore_charge});
        }
        // The owner is back in the serving scope: the registry keeps nothing, so it cannot release it twice, and
        // the entry's pins (private_work lease + image/capture/restore charges) move with the payload.
        const std::size_t index = static_cast<std::size_t>(entry - entries_.data());
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
        ++activations_completed_;
        return true;
    }

    // Fold one caller-observed age into the parked owner's retained record. Age is never computed here and an
    // unknown observation never clears a known one (RetainedRepairAge, serving_repair_admission.hpp).
    bool observe_age(const core::WindowKey& window, const RepairAgeInputs& age) noexcept {
        Entry* entry = find(window);
        if (entry == nullptr) return false;
        entry->age.observe(age);
        return true;
    }

    bool entry_facts(const core::WindowKey& window, SuspendedOwnerFacts& out) const noexcept {
        const Entry* entry = find(window);
        if (entry == nullptr) return false;
        out.identity = entry->identity;
        out.state = entry->state;
        out.quarantined = entry->quarantined;
        out.published_serial = entry->published_serial;
        out.published_cut = entry->published_cut;
        out.live_cut_at_park = entry->live_cut_at_park;
        out.certified_cut = entry->certified_cut;
        out.token_debt = entry->token_debt;
        out.suspended_phase = entry->suspended_phase;
        out.holds_live_controller = entry->owner && entry->owner->controller != nullptr;
        out.reservation_purposes.clear();
        if (entry->charge.credit_or_lease) out.reservation_purposes.push_back(entry->charge.purpose);
        out.observed_at_ns = entry->observed_at_ns;
        out.record_version = entry->record_version;
        out.versions = entry->versions;
        out.capacity_reserved = entry->capacity_nonce != 0;
        out.activation_resources_pinned = entry->resources.pinned();
        out.result_published = entry->published_result.produced();
        out.published_result = entry->published_result;
        return true;
    }
    // Read-only trusted projection; no unit admission or mutation is implied.
    template <typename Inspect>
    bool inspect_parked_owner(const core::WindowKey& window, Inspect&& inspect) const {
        const Entry* entry=find(window);
        return entry && entry->owner && !entry->quarantined &&
            inspect(static_cast<const Owner&>(*entry->owner));
    }
    const RetainedRepairAge* age_of(const core::WindowKey& window) const noexcept {
        const Entry* entry = find(window);
        return entry == nullptr ? nullptr : &entry->age;
    }

    // May this parked owner publish into the live session? False with a named reason once the session moved on,
    // regardless of whether the entry's own recorded facts still agree with each other.
    bool publish_gate(const core::WindowKey& window, const LiveSessionFacts& session,
                      const RepairOwnerEvidence& evidence, RepairOwnerRefusal& why, std::string& error) const {
        const Entry* entry = find(window);
        if (entry == nullptr) {
            why = RepairOwnerRefusal::refused_no_entry;
            error = std::string(repair_owner_refusal_name(why));
            return false;
        }
        const bool ok = parked_owner_publishable(evidence, session, entry->identity, entry->published_cut, why);
        error = std::string(repair_owner_refusal_name(why));
        return ok;
    }

    // The only release path. `attempt(payload, error)` performs the existing
    // `ResidentRepairController::retire` (resident_repair_controller.inc:1221) and the payload-side bookkeeping
    // `finish()` does (generate.cpp:3556-3557: `out.retired=c.retire(error)`, then `out.controller.reset()`).
    // The registry then re-verifies the controller is provably gone, and only then destroys the payload - which
    // is what releases the pinned `private_charge` and the payload's own credits.
    // The overload with NO retirement attempt.  It exists to be REFUSED for an entry that still holds a
    // controller: a caller cannot ask the registry to release a retained owner without the controller's own
    // retire() path, which is the rule the terminal controller destructor enforces in the engine.
    bool retire_owner(const core::WindowKey& window, const char* reason, std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        Entry* entry = find(window);
        if (entry == nullptr)
            return fail(error, RepairOwnerRefusal::refused_no_entry,
                        reason == nullptr ? "no owner is parked for this window"
                                          : std::string("no owner is parked for this window: ") + reason);
        return retire_entry(*entry, reason, NoRetireAttempt{}, error);
    }
    template <typename RetireAttempt>
    bool retire_owner(const core::WindowKey& window, const char* reason, RetireAttempt&& attempt,
                      std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        Entry* entry = find(window);
        if (entry == nullptr)
            return fail(error, RepairOwnerRefusal::refused_no_entry,
                        reason == nullptr ? "no owner is parked for this window"
                                          : std::string("no owner is parked for this window: ") + reason);
        return retire_entry(*entry, reason, std::forward<RetireAttempt>(attempt), error);
    }
    // Shutdown / slot-exhaustion / eviction path: retire every parked owner, in order, and refuse the whole
    // operation (leaving everything parked) if any one of them cannot be retired.
    template <typename RetireAttempt>
    bool retire_all(const char* reason, RetireAttempt&& attempt, std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        // Always retire the front entry: `retire_entry` erases on success, so holding an iterator (or a
        // range-for cursor) across the call would be invalidated by the erase.
        while (!entries_.empty())
            if (!retire_entry(entries_.front(), reason, attempt, error)) return false;
        return true;
    }

    // Deliberate teardown. Refuses (leaving every entry intact) while any controller is still parked; an
    // unretired payload is never dropped. The destructor turns a refusal into the named failstop.
    bool close(std::string& error) {
        last_refusal_ = RepairOwnerRefusal::none;
        for (const Entry& entry : entries_)
            if (entry.owner && entry.owner->controller != nullptr)
                return fail(error, RepairOwnerRefusal::refused_close_holds_controller,
                            "retire_owner/retire_all is the only release path for a retained controller");
        entries_.clear();
        return true;
    }

private:
    struct Entry {
        core::WindowKey identity{};
        RepairOwnerState state = RepairOwnerState::suspended;
        // Set when a unit ran (or threw) and the delivered facts could not be verified against the record. The
        // entry keeps its payload and its charge, refuses further advancement, and can only be retired.
        bool quarantined = false;
        std::unique_ptr<Owner> owner;
        RetainedRepairAge age;
        OwnerReservationHandle charge{core::ReservationPurpose::private_work, {}};
        uint64_t published_serial = 0;
        core::StateCut published_cut{}, live_cut_at_park{}, certified_cut{};
        uint64_t token_debt = 0;
        core::serving_raw::Phase suspended_phase = core::serving_raw::Phase::absent;
        std::optional<uint64_t> observed_at_ns;
        // The versioned retained-result record (slice 5). `record_version` bumps on every change to the record
        // (park, bounded unit, publication), which is what a prepared result is checked against; `resources` and
        // `capacity_nonce` are the up-front admission that makes a later activation reachable.
        uint64_t record_version = 0;
        uint64_t residency_generation = 0;
        RetainedVersions versions{};
        uint64_t capacity_nonce = 0;
        RetainedActivationResources resources{};
        PreparedRetainedResult published_result{};
    };

    // The entry's versioned facts, as the shared gates read them. Built here so no caller can state a record.
    static RetainedRecordFacts record_of(const Entry& entry) noexcept {
        RetainedRecordFacts record;
        record.identity = entry.identity;
        record.published_serial = entry.published_serial;
        record.record_version = entry.record_version;
        record.token_debt = entry.token_debt;
        record.published_cut = entry.published_cut;
        record.certified_cut = entry.certified_cut;
        record.versions = entry.versions;
        record.capacity_reserved = entry.capacity_nonce != 0;
        record.resources_pinned = entry.resources.pinned() && entry.resources.purposes_correct() &&
                                  entry.resources.bytes_bounded();
        return record;
    }

    // The payload is the only route to the controller's facts (see `RetainedRepairOwner`); the registry never
    // states any of them itself.
    static RepairOwnerEvidence unit_evidence(const Owner& owner) noexcept {
        return owner.owner_evidence_snapshot();
    }

    static bool entry_record_matches(const Entry& entry, const RepairOwnerEvidence& evidence,
                                     RepairOwnerRefusal& why) noexcept {
        if (!evidence.produced()) { why = RepairOwnerRefusal::refused_evidence_invalid; return false; }
        if (evidence.published_serial() != entry.published_serial ||
            !(evidence.published_cut() == entry.published_cut) ||
            !(evidence.private_cut() == entry.certified_cut) ||
            evidence.token_debt() != entry.token_debt) {
            why = RepairOwnerRefusal::refused_record_changed;
            return false;
        }
        if (!(evidence.published_key() == entry.identity)) {
            why = RepairOwnerRefusal::refused_identity_not_publication;
            return false;
        }
        if (!evidence.private_charge() ||
            !entry.charge.credit_or_lease ||
            evidence.private_charge().get() != entry.charge.credit_or_lease.get()) {
            why = RepairOwnerRefusal::refused_no_retained_charge;
            return false;
        }
        return true;
    }

    template <typename RetireAttempt>
    bool retire_entry(Entry& entry, const char* reason, RetireAttempt&& attempt, std::string& error) {
        if (entry.state != RepairOwnerState::suspended)
            return fail(error, RepairOwnerRefusal::refused_state_not_suspended,
                        "the entry is already mid-retirement");
        if (entry.owner && entry.owner->controller != nullptr) {
            if constexpr (std::is_same_v<std::remove_cvref_t<RetireAttempt>, NoRetireAttempt>) {
                return fail(error, RepairOwnerRefusal::refused_retire_requires_attempt,
                            "the parked owner still holds a controller: retirement must go through the "
                            "controller's own retire()");
            } else {
                entry.state = RepairOwnerState::retiring;
                std::string retire_error;
                bool retired_now = false;
                try {
                    retired_now = attempt(*entry.owner, retire_error);
                } catch (...) {
                    entry.state = RepairOwnerState::suspended;  // a throwing attempt may not half-release
                    throw;
                }
                if (!retired_now) {
                    entry.state = RepairOwnerState::suspended;
                    return fail(error, RepairOwnerRefusal::refused_retire_attempt,
                                retire_error.empty() ? (reason == nullptr ? "retirement refused" : reason)
                                                     : retire_error);
                }
                if (entry.owner->controller != nullptr || !entry.owner->retired) {
                    entry.state = RepairOwnerState::suspended;
                    return fail(error, RepairOwnerRefusal::refused_retire_left_controller,
                                "retirement reported success but the payload still holds its controller");
                }
            }
        } else if (entry.owner && !entry.owner->retired) {
            return fail(error, RepairOwnerRefusal::refused_controller_lost_without_retirement,
                        "the parked payload lost its controller without a retirement witness");
        }
        // Destroying the payload releases its credits after its payloads (Outcome declares the credits first so
        // they die last, generate.cpp:3195-3197); the controller pointer is already null, so the terminal
        // controller destructor is never reached. The entry's pinned charge dies with the entry.
        const std::size_t index = static_cast<std::size_t>(&entry - entries_.data());
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
        ++released_owners_;
        return true;
    }

    Entry* find(const core::WindowKey& window) noexcept {
        for (Entry& entry : entries_) if (entry.identity == window) return &entry;
        return nullptr;
    }
    const Entry* find(const core::WindowKey& window) const noexcept {
        for (const Entry& entry : entries_) if (entry.identity == window) return &entry;
        return nullptr;
    }

    bool fail(std::string& error, RepairOwnerRefusal why, const std::string& detail) noexcept {
        last_refusal_ = why;
        error = std::string(repair_owner_refusal_name(why));
        if (!detail.empty()) error += ": " + detail;
        return false;
    }

    struct NoRetireAttempt {
        bool operator()(Owner&, std::string&) const noexcept { return false; }
    };

    std::size_t max_suspended_owners_ = 0;
    std::vector<Entry> entries_;
    // Outstanding up-front capacity reservations (slice 5): bounded with the entries, because a reservation is
    // only issued while `entries_.size() + tickets_.size()` is below the slot bound, and a park consumes one.
    std::vector<uint64_t> tickets_;
    uint64_t next_nonce_ = 1;
    std::size_t released_owners_ = 0;
    uint64_t units_advanced_ = 0, tokens_repaired_ = 0, activations_completed_ = 0, results_published_ = 0;
    RepairOwnerRefusal last_refusal_ = RepairOwnerRefusal::none;
};

} // namespace strata::program
