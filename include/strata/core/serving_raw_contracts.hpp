#pragma once
// The SERVING RAW adapter's lifecycle contract, in one place: which phase allows which operation, what an
// acceptable completion cut is, and when destroying the owner is a failstop rather than a cleanup.
//
// These rules mirror what `SuffixRawTransaction` itself enforces
// (`src/program/suffix_raw_runtime.cpp`: begin :214-216, completion :232-238, capture :361-365, restore
// :409-411, destructor :96-98). They are defined here ONCE and used by `ServingRawSession`, so the CPU gate
// (`tests/core/serving_raw_contracts`) exercises the same code the server runs instead of a parallel
// test-only copy or a source-text regex.
//
// Deliberate consequences, stated so nobody "fixes" them by weakening the RAW checks:
//  - capture/restore are reachable only at the DRAINED cut, i.e. only after a real all-stage drain;
//  - an invalid cut is refused BEFORE the transaction is asked to certify it (a certification attempt on a
//    fabricated cut would poison the transaction);
//  - destroying an owner whose transaction is running/capturing/restoring/poisoned terminates the process: a
//    serving process may NOT continue after an unclosed fence by clearing token or checkpoint metadata.
#include "strata/core/window_hash.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace strata::core::serving_raw {

enum class Phase : uint8_t { absent, idle, running, drained, capturing, restoring, poisoned };

inline bool digest_present(const StateDigest& digest) noexcept {
    return std::any_of(digest.begin(), digest.end(), [](uint8_t byte) { return byte != 0; });
}

inline bool begin_allowed(Phase phase) noexcept { return phase == Phase::idle || phase == Phase::drained; }
inline bool complete_allowed(Phase phase) noexcept { return phase == Phase::running; }
inline bool capture_allowed(Phase phase) noexcept { return phase == Phase::drained; }
inline bool restore_allowed(Phase phase) noexcept { return phase == Phase::drained; }

inline bool terminates_on_destruction(Phase phase) noexcept {
    return phase == Phase::running || phase == Phase::capturing || phase == Phase::restoring ||
           phase == Phase::poisoned;
}

/// The caller-supplied facts of the common all-stage cut. `history` must point at the digest of the ACTUAL
/// accepted token history; a zeroed digest means "nothing was really certified" and is refused.
struct CutFacts {
    uint64_t id = 0, accepted = 0, forwarded = 0, exact_frontier = 0;
    const StateDigest* history = nullptr;
};

inline bool cut_facts_acceptable(const CutFacts& cut, size_t stage_count, uint64_t max_tokens) noexcept {
    return stage_count > 0 && cut.id != 0 && cut.history != nullptr && digest_present(*cut.history) &&
           cut.exact_frontier <= cut.forwarded && cut.forwarded <= cut.accepted && cut.accepted <= max_tokens;
}

/// Request-parameterized repair plan: WHERE the donor's suffix state gets re-attached under the changed run,
/// and what the operation may claim afterwards.  Every number comes from the request; none of the qualified
/// experiment's fixture values (32/36/40/128/129/132/137/138) appear here or may be assumed by callers.
/// The bounds mirror the controller's own acceptance rules (`src/program/resident_repair_controller.inc`:
/// `repair_to` refuses a target outside [private cut forwarded, published cut forwarded]; an attachment requires
/// exact_frontier P < repair R < current C, all four-aligned, with C-R >= 4) without copying its scenario counts.
/// Shared capacity of the qualified repair. The controller retains two attachment slots, but an attachment after
/// the first requires an intervening `foreground()` - which on the controller is a GENERATION step (it samples,
/// appends the picked token to the accepted history and raises the output watermark), not a re-read of a known
/// request token. A serving request may not substitute generated tokens for its own prompt, so the production
/// serving mode admits exactly ONE attachment and no foreground steps at all.
inline constexpr size_t kServingRepairMaxFrontiers = 1;
/// Shared minimum distance between a frontier and the current endpoint (the qualified attachment bound).
inline constexpr uint64_t kServingRepairMinGap = 4;

/// Trace policy at the controller's final exact CURRENT publication. These are counter/history requirements,
/// never a completion witness: the controller additionally requires its real private all-stage committed cut.
/// The experiment deliberately tests two attachments with later generation; serving consumes no generated token.
enum class RepairFinishMode : uint8_t { two_attachment_generation, serving_no_generation, request_continuation };
struct RepairFinishTrace {
    unsigned attachments = 0;
    uint64_t foreground = 0, foreground_at_attachment = 0, receipts = 0;
    uint64_t initial_accepted = 0, accepted = 0;
    uint64_t initial_watermark = 0, watermark = 0;
};
inline bool repair_finish_trace_acceptable(RepairFinishMode mode, const RepairFinishTrace& trace) noexcept {
    switch (mode) {
        case RepairFinishMode::two_attachment_generation:
            return trace.attachments >= 2 && trace.foreground > trace.foreground_at_attachment;
        case RepairFinishMode::serving_no_generation:
            return trace.attachments == kServingRepairMaxFrontiers && trace.foreground == 0 &&
                   trace.foreground_at_attachment == 0 && trace.receipts == 0 &&
                   trace.accepted == trace.initial_accepted && trace.watermark == trace.initial_watermark;
        case RepairFinishMode::request_continuation:
            return trace.attachments >= 1 && trace.attachments <= 2 && trace.receipts <= trace.foreground &&
                   trace.foreground <= UINT64_MAX - trace.initial_accepted &&
                   trace.accepted == trace.initial_accepted + trace.foreground &&
                   trace.receipts <= UINT64_MAX - trace.initial_watermark &&
                   trace.watermark == trace.initial_watermark + trace.receipts;
    }
    return false;
}

/// The controller starts selection serial 1; each partial attachment increments it before acknowledgement.
inline bool repair_attachment_publication_acceptable(uint64_t publication, size_t attachment_index) noexcept {
    return attachment_index <= UINT64_MAX - 2 && publication == uint64_t(attachment_index) + 2;
}

struct RepairPlanFacts {
    uint64_t edit_from = 0, edit_to = 0;      ///< the changed run [edit_from, edit_to), |B| == |B'|
    uint64_t boundary = 0;                    ///< first reused suffix position == the exact-forwarded prefix end
    uint64_t donor_forwarded = 0;             ///< F: the captured donor endpoint (state covers [0, F])
    uint64_t accepted = 0;                    ///< tokens A+B'+C the imported state stands on (no generation)
    std::span<const uint64_t> repair_at;      ///< exactly one frontier, boundary < r < donor_forwarded
    unsigned foreground_between = 0;          ///< MUST be 0 in serving: foreground() generates, it does not re-read
    uint64_t page = 0;                        ///< real page size from the geometry (four-token aligned)
};

enum class RepairPlanRefusal : uint8_t {
    none, no_page, unaligned_page, empty_plan, edit_out_of_order, edit_not_before_boundary,
    boundary_unaligned, boundary_at_or_past_donor, donor_unaligned, suffix_too_short,
    accepted_history, repair_unaligned, repair_not_past_boundary, repair_at_or_past_donor,
    repair_out_of_order, repair_too_close_to_donor, repair_past_accepted, unexpected_foreground,
    frontiers_too_close, foreground_reaches_current, too_many_repairs
};

inline const char* repair_plan_reason(RepairPlanRefusal why) noexcept {
    switch (why) {
        case RepairPlanRefusal::none: return "admitted";
        case RepairPlanRefusal::no_page: return "page size unknown";
        case RepairPlanRefusal::unaligned_page: return "page size is not four-token aligned";
        case RepairPlanRefusal::empty_plan: return "no repair frontier was requested";
        case RepairPlanRefusal::edit_out_of_order: return "changed run is empty or reversed";
        case RepairPlanRefusal::edit_not_before_boundary: return "changed run does not end at or before the reuse boundary";
        case RepairPlanRefusal::boundary_unaligned: return "reuse boundary is not page aligned";
        case RepairPlanRefusal::boundary_at_or_past_donor: return "reuse boundary is at or past the donor endpoint";
        case RepairPlanRefusal::donor_unaligned: return "donor endpoint is not page aligned";
        case RepairPlanRefusal::suffix_too_short: return "reused suffix is shorter than the qualified minimum (4)";
        case RepairPlanRefusal::accepted_history: return "the accepted history must be the donor's forwarded tokens plus its one pending token (N == forwarded + 1)";
        case RepairPlanRefusal::repair_unaligned: return "a repair frontier is not page aligned";
        case RepairPlanRefusal::repair_not_past_boundary: return "a repair frontier is not past the reuse boundary";
        case RepairPlanRefusal::repair_at_or_past_donor: return "a repair frontier is at or past the donor endpoint";
        case RepairPlanRefusal::repair_out_of_order: return "repair frontiers are not strictly increasing";
        case RepairPlanRefusal::repair_too_close_to_donor: return "a repair frontier is within 4 tokens of the donor endpoint";
        case RepairPlanRefusal::repair_past_accepted: return "a repair frontier exceeds the accepted history";
        case RepairPlanRefusal::unexpected_foreground: return "serving repair admits no foreground step: it would generate tokens, not re-read the request's own";
        case RepairPlanRefusal::frontiers_too_close: return "a frontier is closer than the request's own fresh forwards";
        case RepairPlanRefusal::foreground_reaches_current: return "fresh forwards would reach the current endpoint";
        case RepairPlanRefusal::too_many_repairs: return "more repair frontiers than the plan can separate";
    }
    return "unknown";
}

/// True when the plan is admissible, else reports the first reason.  Pure arithmetic over request facts: it does
/// not touch a device, a transaction or a bundle, which is what makes it a CPU gate.
inline bool repair_plan_acceptable(const RepairPlanFacts& plan, RepairPlanRefusal& why) noexcept {
    auto aligned = [&plan](uint64_t position) { return plan.page != 0 && position % plan.page == 0; };
    why = RepairPlanRefusal::none;
    if (plan.page == 0) { why = RepairPlanRefusal::no_page; return false; }
    if (plan.page % 4) { why = RepairPlanRefusal::unaligned_page; return false; }
    if (plan.repair_at.empty()) { why = RepairPlanRefusal::empty_plan; return false; }
    if (plan.repair_at.size() > kServingRepairMaxFrontiers) { why = RepairPlanRefusal::too_many_repairs; return false; }
    if (plan.foreground_between != 0) { why = RepairPlanRefusal::unexpected_foreground; return false; }
    // The geometry of the reuse first: an unaligned boundary or donor endpoint can never be a certified cut, and
    // the reused region must be wide enough to hold the qualified attachment bound current-frontier >= 4.
    if (!aligned(plan.boundary)) { why = RepairPlanRefusal::boundary_unaligned; return false; }
    if (plan.boundary >= plan.donor_forwarded) { why = RepairPlanRefusal::boundary_at_or_past_donor; return false; }
    if (!aligned(plan.donor_forwarded)) { why = RepairPlanRefusal::donor_unaligned; return false; }
    if (plan.donor_forwarded < plan.boundary + 4) { why = RepairPlanRefusal::suffix_too_short; return false; }
    // The native suffix contract is exact about this: the donor snapshot carries accepted N with forwarded
    // N-1, i.e. the history ends at the ONE pending token the device state was about to consume.  A donor cut
    // built by decrementing the consumed count (certifying a state one token behind the real one) is refused, and
    // so is a history that reaches past that pending token.
    if (plan.accepted != plan.donor_forwarded + 1) { why = RepairPlanRefusal::accepted_history; return false; }
    // Then the changed run: non-empty, and it must not extend past the reuse boundary (past that point the
    // suffix is the request's own unchanged continuation of the donor).
    if (plan.edit_from >= plan.edit_to) { why = RepairPlanRefusal::edit_out_of_order; return false; }
    if (plan.edit_to > plan.boundary) { why = RepairPlanRefusal::edit_not_before_boundary; return false; }
    uint64_t previous = plan.boundary;
    for (const uint64_t position : plan.repair_at) {
        if (!aligned(position)) { why = RepairPlanRefusal::repair_unaligned; return false; }
        if (position <= plan.boundary) { why = RepairPlanRefusal::repair_not_past_boundary; return false; }
        if (position <= previous) { why = RepairPlanRefusal::repair_out_of_order; return false; }
        if (position >= plan.donor_forwarded) { why = RepairPlanRefusal::repair_at_or_past_donor; return false; }
        // The qualified attachment bound is current-forwarded minus the frontier being repaired >= 4.
        if (plan.donor_forwarded - position < kServingRepairMinGap) { why = RepairPlanRefusal::repair_too_close_to_donor; return false; }
        if (position > plan.accepted) { why = RepairPlanRefusal::repair_past_accepted; return false; }
        previous = position;
    }
    // The single production attachment must leave the qualified gap to the current endpoint; the request's own
    // prompt tokens past that endpoint are read normally afterwards (nothing is generated on its behalf).
    if (plan.donor_forwarded - plan.repair_at.back() < kServingRepairMinGap)
        { why = RepairPlanRefusal::foreground_reaches_current; return false; }
    return true;
}

/// The adapter's TRANSITION sequence, modelled explicitly so it can be exercised without a device: request
/// lifetime and RAW work phase are different things, and a public sequence must be able to reach capture/restore.
/// The adapter routes every public call through this state machine, so the test drives the same decisions the
/// server makes rather than a second copy of the rules.
///
/// SEMANTIC CONTRACT EXTENSION (2026-10-31, retained-repair slice 2) - the PRIVATE WORK WINDOW.
/// `begin_private_window` / `end_private_window` name one narrower permission: a parked repair owner may run its
/// OWN private RAW transaction while the serving request fence is closed. The window is legal ONLY where the
/// live fence is provably quiescent - no request open, nothing terminal, the observed live phase `idle` or
/// `drained` (the same `begin_allowed` cut that capture/restore use) - and no window is already open. It does NOT
/// change `phase_`: a private unit never touches the live transaction, so a private window must never be readable
/// as an open live fence and `terminates_on_destruction` keeps its exact meaning.
/// While a window is open every live operation is refused - `open_request`, `begin_work`, `drain_work`, `capture`,
/// `restore` - so no foreground request can start, and no live RAW operand can run, under a parked owner's private
/// work. `end_private_window` is the only exit and stays available while the window is open (including after a
/// poison), so a window can never latch the session shut. With no window opened, every decision in this machine
/// is identical to the reviewed behaviour.
class Lifecycle {
public:
    enum class Event : uint8_t {
        open_request, begin_work, drain_work, capture, restore, close_request, abandon,
        begin_private_window, end_private_window,
    };

    Phase phase() const noexcept { return phase_; }
    bool request_open() const noexcept { return request_open_; }
    /// True while a parked owner's private RAW work legitimately owns the service (contract above).
    bool private_window_open() const noexcept { return private_window_; }
    /// The observation the serving loop makes before it offers a parked owner one bounded private unit.
    bool private_window_allowed() const noexcept { return allow(Event::begin_private_window); }

    bool allow(Event event) const noexcept {
        // 2026-10-29 review: an abandoned or transaction-reported poison must be DURABLE. Before this, `abandon`
        // poisoned only the bookkeeping, and the next `set_observed_phase` (every public call re-syncs from the
        // transaction) silently resurrected the fence, so `healthy()`/`phase()` could go back to idle/drained.
        // The one exception is deliberate: `end_private_window` must always be able to close an open window.
        if (terminal_ && event != Event::end_private_window) return false;
        switch (event) {
            case Event::open_request:  return !request_open_ && !private_window_ && phase_ != Phase::poisoned;
            case Event::begin_work:    return request_open_ && !private_window_ && begin_allowed(phase_);
            case Event::drain_work:    return request_open_ && !private_window_ && complete_allowed(phase_);
            case Event::capture:
            case Event::restore:       return request_open_ && !private_window_ && phase_ == Phase::drained;
            // closing a request while RAW work is still running would destroy an open fence (terminate); the
            // caller must either drain the work or abandon deliberately.  A POISONED fence cannot be closed into
            // service either: the only legal move left is destroying the owner, which is the failstop.
            case Event::close_request: return request_open_ && phase_ != Phase::running && phase_ != Phase::poisoned;
            case Event::abandon:       return request_open_;
            // The private window needs the closed, quiescent fence, never a running/capturing/restoring one.
            case Event::begin_private_window:
                return !request_open_ && !private_window_ && begin_allowed(phase_);
            case Event::end_private_window: return private_window_;
        }
        return false;
    }

    /// True once this fence is terminal (abandoned, or the transaction reported a poison). Nothing reopens it.
    bool terminal() const noexcept { return terminal_; }

    /// Applies the transition and reports whether it happened. Illegal events change nothing.
    bool apply(Event event) noexcept {
        if (!allow(event)) return false;
        switch (event) {
            case Event::open_request: request_open_ = true; break;
            case Event::begin_work: phase_ = Phase::running; break;
            case Event::drain_work: phase_ = Phase::drained; break;
            case Event::capture:
            case Event::restore: break;   // the transaction owns the sub-phase during the call itself
            case Event::close_request: request_open_ = false; break;
            case Event::abandon: request_open_ = false; phase_ = Phase::poisoned; terminal_ = true; break;
            // Orthogonal to the live work phase on purpose: it neither opens nor closes a live fence, and it never
            // touches `phase_`, which stays the transaction's own observed phase.
            case Event::begin_private_window: private_window_ = true; break;
            case Event::end_private_window: private_window_ = false; break;
        }
        return true;
    }

    /// The adapter re-syncs from the transaction. A poison (observed or deliberately applied) is never undone by
    /// that re-sync: an abandoned fence cannot be brought back to idle/drained/closed by reading the phase again.
    void set_observed_phase(Phase p) noexcept {
        if (terminal_) { phase_ = Phase::poisoned; return; }
        if (p == Phase::poisoned) { terminal_ = true; phase_ = p; return; }
        phase_ = p;
    }

private:
    bool request_open_ = false;
    bool terminal_ = false;
    bool private_window_ = false;
    Phase phase_ = Phase::absent;
};

}   // namespace strata::core::serving_raw
