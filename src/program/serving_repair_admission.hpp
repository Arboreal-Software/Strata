#pragma once
// Decision kernel for demand-gated serving repair admission. Pure logic: it reads
// caller-supplied foreground and age state and returns one hard eligibility reason.
// It owns no clock, no stdin queue, no repair controller, no reactor object and no
// engine call. NOT INTEGRATED: nothing in the serving path calls this kernel yet, so
// no behavior is enabled by default and nothing here is qualified or deployed.
//
// Owner rule this encodes (.local/handoffs/owner-repair-scheduling-correction-20261030.md):
// repair may run when no decode/generation is actively waiting; live repair may
// outrank waiting generation ONLY under age escalation; saved/disk maintenance is
// eligible only when there is no other eligible work at all AND foreground return is
// protected. Every one of those is a boolean gate here — a gate can never be opened
// by a small weight, an extreme debt value or an extreme position value.
#include "strata/core/thread_dependencies.hpp"
#include "strata/core/window_admission.hpp"
#include "strata/core/window_repair_ancestry.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace strata::program {

// The only foreground-demand sources this engine can actually observe
// (src/program/generate.cpp:14413-14429 StdinShared, :19491-19555 token loop + drain).
struct ServingForegroundState final {
    // The current request is still generating: !cancelled && !stop_req &&
    // produced_n<max_new (generate.cpp:19491; the same condition
    // live_handoff_requested reuses at :19533). While this holds, this
    // conversation's own remaining tokens are waiting decode work.
    bool current_request_generating=false;
    // Lines already typed into the stdin pipe under StdinShared::mu
    // (std::deque<std::string> lines, generate.cpp:14427). Non-empty means the
    // next request is queued before this turn released the single slot.
    uint64_t queued_input_lines=0;
    // published cut.forwarded<max_context (generate.cpp:19491, :19534): the only place a
    // live handoff stays coherent, so the only place escalation may cut in.
    bool context_room=false;
    // A stop sequence or EOS ended the token loop (finish=="stop",
    // generate.cpp:19515-19517). Generation is not continuing from here.
    bool stop_or_eos_seen=false;
    // The repair owner retired: finish() set out.retired, so repair_unit has no
    // active owner ("retained repair has no active owner", generate.cpp:3479).
    bool repair_owner_retired=false;
    bool next_line_queued() const noexcept { return queued_input_lines>0; }
    // "decode/generation actively waiting" for this engine is exactly these two
    // sources (audit section 4b). No other signal may be treated as demand here.
    bool decode_waiting() const noexcept { return current_request_generating||next_line_queued(); }
    bool operator==(const ServingForegroundState&) const = default;
};

// Hard eligibility outcomes. The reason is the log record; admission is a boolean,
// never a score. Names follow the reactor's vocabulary.
enum class RepairAdmissionReason {
    admitted_idle,                        // no decode/generation waiting, gates clear
    admitted_age_escalated,               // live repair outranks waiting generation by the caller's reviewed age bound
    refused_foreground_demand,            // decode/generation actively waiting and no reviewed escalation applied
    refused_other_eligible_work,          // saved maintenance vetoed by any other eligible work
    refused_not_idle_authorized,          // saved maintenance without an authorized idle window
    refused_foreground_return_unprotected,// saved maintenance without protected foreground return
    refused_context_or_lifecycle,         // retired owner, or escalation at an incoherent handoff point
    refused_not_resource_ready,           // AdmissionCandidate::resource_ready is false
    refused_not_demanded,                 // AdmissionCandidate::demand is false
    refused_age_unknown,                  // unknown inherited age fails toward NOT escalating
    refused_age_policy_unset,             // caller supplied no reviewed age bound: escalation disabled
    refused_age_below_bound,              // known age has not reached the caller's bound
    refused_class_unit_mismatch,          // work_class and AdmissionUnit disagree, e.g. foreground+repair
    refused_unsupported_unit,             // unit is neither AdmissionUnit::repair nor ::save
};

// Caller-supplied age bound for live-repair escalation. NO number is adopted here:
// any threshold needs measured evidence plus root review before a caller may pass
// it (audit section 3 items 2 and 5; P13/P14 "Owner addition: repair urgency":
// "Thresholds require evidence, not guessed constants"). Unset means escalation is
// disabled, so the default state of this type is the reviewed safe state, not a
// policy. A caller that sets a value is asserting it was priced and reviewed.
struct ServingRepairAdmissionPolicy final {
    std::optional<uint64_t> live_repair_age_escalation_ns;
    // The only construction a reviewer should see before evidence exists.
    static ServingRepairAdmissionPolicy review_required() noexcept { return {}; }
    bool escalation_bound_unset() const noexcept { return !live_repair_age_escalation_ns.has_value(); }
    bool operator==(const ServingRepairAdmissionPolicy&) const = default;
};

// The repair/save unit under consideration, as a narrow view of the reactor's own
// fields: AdmissionCandidate::demand/resource_ready/idle_authorized/
// foreground_return_protected (include/strata/core/window_admission.hpp:23) plus
// DependencyOperation::work_class. The kernel never mutates them.
struct RepairCandidateState final {
    core::DependencyWorkClass work_class=core::DependencyWorkClass::live_repair;
    core::AdmissionUnit unit=core::AdmissionUnit::repair;
    bool demand=false, resource_ready=false, idle_authorized=false, foreground_return_protected=false;
    bool operator==(const RepairCandidateState&) const = default;
};

// Age inputs are passed in, never computed by this kernel, and stay three
// separate quantities (P13/P14): wall-clock inherited age, earliest affected token
// position, and token debt.
struct RepairAgeInputs final {
    // Mirrors strata::core::RepairAgeSummary::oldest_monotonic_age_ns
    // (include/strata/core/window_repair_ancestry.hpp:76-83): present ONLY when
    // every root birth shares the caller's nonzero clock domain and none lies in
    // its future. nullopt therefore already means unknown anchor, foreign domain
    // or future birth, and must fail toward not escalating. age==0 stays
    // distinguishable from unknown.
    std::optional<uint64_t> oldest_unresolved_inherited_age_ns;
    uint64_t oldest_unresolved_unix_ms=0;
    std::size_t roots=0,unknown_monotonic_anchors=0;
    // Positional debt, reported separately; never folded into age.
    uint64_t earliest_affected_token_position=0;
    // Token debt: published cut.forwarded minus private_forwarded.
    uint64_t token_debt=0;
    bool operator==(const RepairAgeInputs&) const = default;
};

// Projection of the real ancestry age summary. The caller must run
// strata::core::repair_ancestry_age() itself and supply position/debt: this kernel
// reads no clock, no ancestry table and no controller.
inline RepairAgeInputs make_repair_age_inputs(const core::RepairAgeSummary& summary,
                                              uint64_t earliest_affected_token_position,
                                              uint64_t token_debt) noexcept {
    return {summary.oldest_monotonic_age_ns,summary.oldest_unresolved_unix_ms,
            summary.roots,summary.unknown_monotonic_anchors,
            earliest_affected_token_position,token_debt};
}

// Mirrors the exact_ready relation the serving path already uses
// (ServingSuffixRepairOperation::exact_ready, generate.cpp:3506:
// private_forwarded()==published_->metadata.cut.forwarded). Debt 0 means the
// private checkpoint has caught the published CURRENT cut. Caller-side arithmetic
// only; the decision path never calls it.
inline uint64_t repair_token_debt(uint64_t published_cut_forwarded,uint64_t private_forwarded) noexcept {
    return published_cut_forwarded>private_forwarded?published_cut_forwarded-private_forwarded:0;
}

// Caller-owned retention for one retained repair owner: records the oldest
// inherited age observed so later edits, partial attachment or a newer affected
// position cannot reset urgency (P13: "Preserve that age across later
// edits/ancestry"). It fabricates nothing: a nullopt observation never clears the
// retention, and a younger age never replaces an older one. Position and debt of
// the newest observation are kept in separate fields.
class RetainedRepairAge final {
public:
    void observe(const RepairAgeInputs& age) noexcept {
        latest_position_=age.earliest_affected_token_position;
        latest_debt_=age.token_debt;
        if(!age.oldest_unresolved_inherited_age_ns) return; // unknown never clears
        const uint64_t observed=*age.oldest_unresolved_inherited_age_ns;
        if(oldest_age_&&observed<=*oldest_age_) return;     // never reset to younger
        oldest_age_=observed;
        oldest_unix_ms_=age.oldest_unresolved_unix_ms;
        oldest_roots_=age.roots;
        oldest_unknown_anchors_=age.unknown_monotonic_anchors;
        oldest_position_=age.earliest_affected_token_position;
        oldest_debt_=age.token_debt;
    }
    std::optional<uint64_t> oldest_age_ns() const noexcept { return oldest_age_; }
    uint64_t oldest_position() const noexcept { return oldest_position_; }
    uint64_t oldest_debt() const noexcept { return oldest_debt_; }
    uint64_t latest_position() const noexcept { return latest_position_; }
    uint64_t latest_debt() const noexcept { return latest_debt_; }
    bool empty() const noexcept { return !oldest_age_.has_value(); }
    // What a caller feeds to the kernel: the preserved oldest age, with the live
    // position and debt of the newest observation kept in their own fields.
    RepairAgeInputs inputs() const noexcept {
        return {oldest_age_,oldest_unix_ms_,oldest_roots_,oldest_unknown_anchors_,
                latest_position_,latest_debt_};
    }
private:
    std::optional<uint64_t> oldest_age_;
    uint64_t oldest_unix_ms_=0,oldest_position_=0,oldest_debt_=0,latest_position_=0,latest_debt_=0;
    std::size_t oldest_roots_=0,oldest_unknown_anchors_=0;
};

// The decision. `reason` is authoritative; `admitted()` is derived from it so the
// two can never disagree. The age/position/debt echoes are carried in separate
// fields for logging, so no log can merge them into one urgency number.
struct RepairAdmissionDecision final {
    RepairAdmissionReason reason=RepairAdmissionReason::refused_not_demanded;
    core::DependencyWorkClass work_class=core::DependencyWorkClass::live_repair;
    core::AdmissionUnit unit=core::AdmissionUnit::repair;
    std::optional<uint64_t> oldest_unresolved_inherited_age_ns;
    uint64_t earliest_affected_token_position=0,token_debt=0;
    bool escalation_bound_unset=false;  // policy carried no reviewed bound
    bool escalation_coherent=false;     // handoff point could carry an escalated unit
    // Which escalation gate did not pass, when an escalation was the only way out
    // and was considered. Kept apart from `reason` so a log can name both the hard
    // gate that blocked and the reason the age gate did not release it. Left unset
    // when escalation was never applicable, which is always the case for saved
    // maintenance.
    std::optional<RepairAdmissionReason> escalation_refusal;
    bool admitted() const noexcept {
        return reason==RepairAdmissionReason::admitted_idle||
               reason==RepairAdmissionReason::admitted_age_escalated;
    }
    bool operator==(const RepairAdmissionDecision&) const = default;
};

inline const char* repair_admission_reason_name(RepairAdmissionReason reason) noexcept {
    switch(reason){
        case RepairAdmissionReason::admitted_idle: return "admitted_idle";
        case RepairAdmissionReason::admitted_age_escalated: return "admitted_age_escalated";
        case RepairAdmissionReason::refused_foreground_demand: return "refused_foreground_demand";
        case RepairAdmissionReason::refused_other_eligible_work: return "refused_other_eligible_work";
        case RepairAdmissionReason::refused_not_idle_authorized: return "refused_not_idle_authorized";
        case RepairAdmissionReason::refused_foreground_return_unprotected: return "refused_foreground_return_unprotected";
        case RepairAdmissionReason::refused_context_or_lifecycle: return "refused_context_or_lifecycle";
        case RepairAdmissionReason::refused_not_resource_ready: return "refused_not_resource_ready";
        case RepairAdmissionReason::refused_not_demanded: return "refused_not_demanded";
        case RepairAdmissionReason::refused_age_unknown: return "refused_age_unknown";
        case RepairAdmissionReason::refused_age_policy_unset: return "refused_age_policy_unset";
        case RepairAdmissionReason::refused_age_below_bound: return "refused_age_below_bound";
        case RepairAdmissionReason::refused_class_unit_mismatch: return "refused_class_unit_mismatch";
        case RepairAdmissionReason::refused_unsupported_unit: return "refused_unsupported_unit";
    }
    return "unknown"; // unreachable: every enumerator is listed above
}

inline const char* repair_admission_escalation_refusal_name(
        const std::optional<RepairAdmissionReason>& refusal) noexcept {
    return refusal.has_value()?repair_admission_reason_name(*refusal):"not_considered";
}

// Exploration seam, explicitly NOT an adopted formula. Stochastic/importance
// weighting was exploration only (sources/fig-24.md, "Work-class scheduling
// exploration"): weights alone cannot express "never run this while that other
// work is available". evaluate_serving_repair_admission below has no parameter of
// this type and no code path that consults it, so no weight can open a hard gate.
// Adoption would require a measured formula plus root review.
struct SelectionWeightingProbe final {
    bool enabled=false; // disabled by construction
    bool operator==(const SelectionWeightingProbe&) const = default;
};
enum class SelectionWeightingStatus { not_adopted };
inline SelectionWeightingStatus selection_weighting_probe(const SelectionWeightingProbe&) noexcept {
    return SelectionWeightingStatus::not_adopted;
}
inline const char* selection_weighting_status_name(SelectionWeightingStatus status) noexcept {
    return status==SelectionWeightingStatus::not_adopted?"not adopted":"unknown";
}

// One decision, side-effect free and allocation-free (fixed-size value types,
// noexcept, no exceptions for any input combination), so it is usable in front of
// every bounded unit in a reactor loop. Fixed precedence, kept stable so logs from
// different builds can be compared:
//   1 the candidate is a repair/save unit with a coherent work_class pairing,
//   2 the repair owner is alive (lifecycle),
//   3 the candidate's own gates (demand, resource readiness),
//   4 the class gate: saved maintenance is idle-only, always,
//   5 live repair: admit when nothing is waiting; otherwise the age-escalation
//     gate only, which additionally requires a coherent handoff point.
// `other_nonmaintenance_demand` mirrors WindowAdmission::admit's
// other_nonmaintenance_demand argument: demand the caller knows about but did not
// submit as a candidate (window_admission.cpp:128,145-146).
inline RepairAdmissionDecision evaluate_serving_repair_admission(
        const ServingRepairAdmissionPolicy& policy,
        const ServingForegroundState& foreground,
        const RepairCandidateState& candidate,
        const RepairAgeInputs& age,
        bool other_nonmaintenance_demand) noexcept {
    RepairAdmissionDecision decision;
    decision.work_class=candidate.work_class;
    decision.unit=candidate.unit;
    decision.oldest_unresolved_inherited_age_ns=age.oldest_unresolved_inherited_age_ns;
    decision.earliest_affected_token_position=age.earliest_affected_token_position;
    decision.token_debt=age.token_debt;
    decision.escalation_bound_unset=policy.escalation_bound_unset();

    const bool save_unit=candidate.unit==core::AdmissionUnit::save;
    if(!save_unit&&candidate.unit!=core::AdmissionUnit::repair){
        decision.reason=RepairAdmissionReason::refused_unsupported_unit;
        return decision;
    }
    // The idle-only gate belongs to the class. Accepting only the two coherent
    // pairings stops a mislabeled candidate (live_repair + save, or a foreground
    // unit tagged repair) from slipping past it.
    const bool saved_class=candidate.work_class==core::DependencyWorkClass::saved_maintenance;
    const bool live_class=candidate.work_class==core::DependencyWorkClass::live_repair;
    if(saved_class!=save_unit||(!saved_class&&!live_class)){
        decision.reason=RepairAdmissionReason::refused_class_unit_mismatch;
        return decision;
    }
    if(foreground.repair_owner_retired){
        decision.reason=RepairAdmissionReason::refused_context_or_lifecycle;
        return decision;
    }
    if(!candidate.demand){
        decision.reason=RepairAdmissionReason::refused_not_demanded;
        return decision;
    }
    if(!candidate.resource_ready){
        decision.reason=RepairAdmissionReason::refused_not_resource_ready;
        return decision;
    }

    if(saved_class){
        // Saved/disk maintenance: eligible only with no other eligible work at all
        // AND protected foreground return. Age escalation never applies to this
        // class; the owner's age rule speaks only to live repair.
        if(foreground.decode_waiting()){
            decision.reason=RepairAdmissionReason::refused_foreground_demand;
            return decision;
        }
        if(other_nonmaintenance_demand){
            decision.reason=RepairAdmissionReason::refused_other_eligible_work;
            return decision;
        }
        if(!candidate.idle_authorized){
            decision.reason=RepairAdmissionReason::refused_not_idle_authorized;
            return decision;
        }
        if(!candidate.foreground_return_protected){
            decision.reason=RepairAdmissionReason::refused_foreground_return_unprotected;
            return decision;
        }
        decision.reason=RepairAdmissionReason::admitted_idle;
        return decision;
    }

    if(!foreground.decode_waiting()){
        decision.reason=RepairAdmissionReason::admitted_idle;
        return decision;
    }
    // Live repair facing waiting generation: only the age gate can cut in, so the
    // headline reason stays the hard demand gate and the escalation sub-reason names
    // which age condition did not hold.
    if(!policy.live_repair_age_escalation_ns.has_value()){
        decision.escalation_refusal=RepairAdmissionReason::refused_age_policy_unset;
        decision.reason=RepairAdmissionReason::refused_foreground_demand;
        return decision;
    }
    if(!age.oldest_unresolved_inherited_age_ns.has_value()){
        decision.escalation_refusal=RepairAdmissionReason::refused_age_unknown;
        decision.reason=RepairAdmissionReason::refused_foreground_demand;
        return decision;
    }
    if(*age.oldest_unresolved_inherited_age_ns<*policy.live_repair_age_escalation_ns){
        decision.escalation_refusal=RepairAdmissionReason::refused_age_below_bound;
        decision.reason=RepairAdmissionReason::refused_foreground_demand;
        return decision;
    }
    // Escalation is only coherent where the existing live handoff is coherent:
    // generation could still continue and there is room below the context limit
    // (generate.cpp:19533-19536). Otherwise the age gate is not the reason.
    decision.escalation_coherent=foreground.context_room&&!foreground.stop_or_eos_seen;
    if(!decision.escalation_coherent){
        decision.escalation_refusal=RepairAdmissionReason::refused_context_or_lifecycle;
        decision.reason=RepairAdmissionReason::refused_context_or_lifecycle;
        return decision;
    }
    decision.reason=RepairAdmissionReason::admitted_age_escalated;
    return decision;
}
} // namespace strata::program
