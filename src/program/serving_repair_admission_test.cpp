// CPU-only decision-kernel tests for serving_repair_admission.hpp. No GPU, no CUDA,
// no engine call, no reactor object: every input is a plain value the serving loop
// could observe. All ages, positions and debts below are test fixtures chosen to
// discriminate the gates; none of them is an adopted threshold or policy number.
#include "serving_repair_admission.hpp"
#include "serving_repair_scheduler.hpp"
#include "serving_repair_age.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
using namespace strata::program;
namespace core = strata::core;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}

// The decision path must stay reactor-friendly: fixed-size value types, no
// allocation, no exceptions, and no numeric weight parameter anywhere.
static_assert(std::is_trivially_copyable_v<RepairAdmissionDecision>);
static_assert(std::is_trivially_copyable_v<ServingForegroundState>);
static_assert(std::is_trivially_copyable_v<RepairCandidateState>);
static_assert(std::is_trivially_copyable_v<RepairAgeInputs>);
static_assert(std::is_trivially_copyable_v<ServingRepairAdmissionPolicy>);
static_assert(std::is_trivially_copyable_v<SelectionWeightingProbe>);
static_assert(std::is_nothrow_destructible_v<RetainedRepairAge>);
static_assert(std::is_nothrow_copy_constructible_v<RepairAdmissionDecision>);
static_assert(noexcept(evaluate_serving_repair_admission(
    std::declval<const ServingRepairAdmissionPolicy&>(),std::declval<const ServingForegroundState&>(),
    std::declval<const RepairCandidateState&>(),std::declval<const RepairAgeInputs&>(),true)));

namespace {
constexpr uint64_t ns_per_second=1000000000ULL;
std::optional<uint64_t> known_age(uint64_t seconds){return uint64_t(seconds)*ns_per_second;}
// The reviewed default: no bound, so escalation is off.
ServingRepairAdmissionPolicy reviewed_policy(uint64_t bound_seconds){
    ServingRepairAdmissionPolicy policy;
    policy.live_repair_age_escalation_ns=known_age(bound_seconds);
    return policy;
}
RepairCandidateState live_repair(){
    RepairCandidateState candidate;
    candidate.work_class=core::DependencyWorkClass::live_repair;
    candidate.unit=core::AdmissionUnit::repair;
    candidate.demand=true;candidate.resource_ready=true;
    return candidate;
}
RepairCandidateState saved_save(){
    RepairCandidateState candidate=live_repair();
    candidate.work_class=core::DependencyWorkClass::saved_maintenance;
    candidate.unit=core::AdmissionUnit::save;
    candidate.idle_authorized=true;candidate.foreground_return_protected=true;
    return candidate;
}
// Nothing generating, nothing queued, owner still alive, room below context.
ServingForegroundState quiet(){
    ServingForegroundState state;state.context_room=true;return state;
}
// The current request's own remaining tokens are waiting decode work.
ServingForegroundState generating(){
    ServingForegroundState state=quiet();state.current_request_generating=true;return state;
}
RepairAgeInputs debt_only(uint64_t debt){
    RepairAgeInputs age;age.token_debt=debt;return age;
}

// Queue empty and the current request finished-but-not-retired: this is the window
// today's forced drain at generate.cpp:19539-19555 occupies without ever rechecking
// demand. A young live repair unit must be admitted here.
void admits_live_repair_when_queue_empty_and_request_finished_but_unretired(){
    ServingForegroundState finished=quiet();finished.stop_or_eos_seen=true;
    const auto by_eos=evaluate_serving_repair_admission(ServingRepairAdmissionPolicy::review_required(),
        finished,live_repair(),debt_only(8),false);
    check(by_eos.reason==RepairAdmissionReason::admitted_idle&&by_eos.admitted(),
          "live repair is admitted after EOS with the owner still unretired and the queue empty");
    const auto by_length=evaluate_serving_repair_admission(ServingRepairAdmissionPolicy::review_required(),
        quiet(),live_repair(),debt_only(8),false);
    check(by_length.reason==RepairAdmissionReason::admitted_idle,
          "live repair is admitted after the length finish with nothing queued");
    check(!quiet().decode_waiting()&&quiet().current_request_generating==false,
          "the quiet observation really carries no waiting decode");
}

// The request's own generation is still running: strict reading of the owner rule
// means its remaining tokens are waiting decode work, so young repair may not be
// admitted between those tokens. Catches the current generate.cpp:19519 behaviour of
// calling repair_unit unconditionally after every token.
void refuses_live_repair_while_its_own_request_is_generating(){
    const auto decision=evaluate_serving_repair_admission(ServingRepairAdmissionPolicy::review_required(),
        generating(),live_repair(),debt_only(8),false);
    check(decision.reason==RepairAdmissionReason::refused_foreground_demand&&!decision.admitted(),
          "live repair is refused while its own request is still generating");
    // An unrelated boolean must not open the demand gate.
    RepairCandidateState authorised=live_repair();authorised.idle_authorized=true;
    check(evaluate_serving_repair_admission(ServingRepairAdmissionPolicy::review_required(),
              generating(),authorised,debt_only(8),false).reason==
          RepairAdmissionReason::refused_foreground_demand,
          "idle_authorized on a live repair unit cannot substitute for an idle window");
}

// The Python server already typed the next command into the bounded stdin pipe
// (StdinShared::lines non-empty): the next request must not queue behind a drain.
void refuses_live_repair_while_a_next_line_is_queued(){
    ServingForegroundState queued=quiet();queued.queued_input_lines=1;
    check(queued.decode_waiting()&&queued.next_line_queued(),"a queued line is foreground demand");
    const auto decision=evaluate_serving_repair_admission(ServingRepairAdmissionPolicy::review_required(),
        queued,live_repair(),debt_only(8),false);
    check(decision.reason==RepairAdmissionReason::refused_foreground_demand&&!decision.admitted(),
          "live repair is refused while a next stdin line is queued");
    ServingForegroundState bounded_queue=quiet();bounded_queue.queued_input_lines=8;
    check(evaluate_serving_repair_admission(ServingRepairAdmissionPolicy::review_required(),
              bounded_queue,live_repair(),debt_only(8),false).reason==
          RepairAdmissionReason::refused_foreground_demand,
          "any non-zero queue depth refuses, not only a depth of one");
}

// Saved/disk maintenance is idle-only: eligible only when there is no other
// eligible work at all. Catches a save starting while a live repair (or any other
// non-maintenance candidate) could run instead.
void saved_maintenance_refused_when_any_other_eligible_work_exists(){
    const auto by_observed_demand=evaluate_serving_repair_admission(
        ServingRepairAdmissionPolicy::review_required(),generating(),saved_save(),debt_only(64),false);
    check(by_observed_demand.reason==RepairAdmissionReason::refused_foreground_demand,
          "saved maintenance is refused while generation is waiting");
    const auto by_caller_demand=evaluate_serving_repair_admission(
        ServingRepairAdmissionPolicy::review_required(),quiet(),saved_save(),debt_only(64),true);
    check(by_caller_demand.reason==RepairAdmissionReason::refused_other_eligible_work&&
          !by_caller_demand.admitted(),
          "saved maintenance is refused when the caller reports other eligible non-maintenance work");
    const auto control=evaluate_serving_repair_admission(
        ServingRepairAdmissionPolicy::review_required(),quiet(),saved_save(),debt_only(64),false);
    check(control.reason==RepairAdmissionReason::admitted_idle,
          "the same saved unit is admitted once no other eligible work exists");
}

// P14 requires protected foreground return: no return capacity means the save must
// not start. Catches a save that strands the next foreground request.
void saved_maintenance_refused_without_foreground_return_protection(){
    RepairCandidateState unprotected=saved_save();unprotected.foreground_return_protected=false;
    const auto decision=evaluate_serving_repair_admission(
        ServingRepairAdmissionPolicy::review_required(),quiet(),unprotected,debt_only(64),false);
    check(decision.reason==RepairAdmissionReason::refused_foreground_return_unprotected&&
          !decision.admitted(),
          "saved maintenance is refused without protected foreground return");
    RepairCandidateState unauthorised=saved_save();unauthorised.idle_authorized=false;
    check(evaluate_serving_repair_admission(ServingRepairAdmissionPolicy::review_required(),
              quiet(),unauthorised,debt_only(64),false).reason==
          RepairAdmissionReason::refused_not_idle_authorized,
          "saved maintenance is refused without an authorized idle window");
}

// The owner's age rule, with a caller-supplied bound: an aged live repair may
// outrank waiting generation. Also proves the escalation reason is distinct from the
// idle reason, so logs can tell the two admissions apart.
void age_escalation_admits_live_repair_over_waiting_generation(){
    const auto decision=evaluate_serving_repair_admission(reviewed_policy(60),generating(),live_repair(),
        RepairAgeInputs{.oldest_unresolved_inherited_age_ns=known_age(90),.token_debt=16},false);
    check(decision.reason==RepairAdmissionReason::admitted_age_escalated&&decision.admitted()&&
          decision.escalation_coherent,
          "age escalation admits live repair despite waiting generation when age is known and the policy permits it");
    check(decision.work_class==core::DependencyWorkClass::live_repair&&
          decision.unit==core::AdmissionUnit::repair,
          "the decision echoes the class and unit it decided on");
    const auto young=evaluate_serving_repair_admission(reviewed_policy(60),generating(),live_repair(),
        RepairAgeInputs{.oldest_unresolved_inherited_age_ns=known_age(30),.token_debt=16},false);
    check(young.reason==RepairAdmissionReason::refused_foreground_demand&&
          young.escalation_refusal==RepairAdmissionReason::refused_age_below_bound&&!young.admitted(),
          "age below the caller's bound does not escalate and stays blocked by the demand gate");
    const auto exact=evaluate_serving_repair_admission(reviewed_policy(60),generating(),live_repair(),
        RepairAgeInputs{.oldest_unresolved_inherited_age_ns=uint64_t(0),.token_debt=0},false);
    check(exact.escalation_refusal==RepairAdmissionReason::refused_age_below_bound,
          "age 0 stays distinguishable from unknown and does not escalate");
    check(!decision.escalation_refusal.has_value(),"a successful escalation reports no refusal");
}

// Unknown age must fail toward NOT escalating (audit section 3 item 4). Catches a
// kernel that read a missing anchor as "young age 0" or as "old enough".
void age_escalation_refuses_unknown_age(){
    const auto decision=evaluate_serving_repair_admission(reviewed_policy(60),generating(),live_repair(),
        debt_only(16),false);
    check(!decision.oldest_unresolved_inherited_age_ns.has_value()&&
          decision.reason==RepairAdmissionReason::refused_foreground_demand&&
          decision.escalation_refusal==RepairAdmissionReason::refused_age_unknown&&!decision.admitted(),
          "unknown inherited age never escalates");
    // Same through the real core type: RepairAgeSummary leaves oldest_monotonic_age_ns
    // unset for unknown anchors, foreign clock domains and future births.
    core::RepairAgeSummary summary;
    summary.oldest_unresolved_unix_ms=1780000000000ULL;summary.roots=3;summary.unknown_monotonic_anchors=2;
    const auto projected=make_repair_age_inputs(summary,4096,16);
    check(!projected.oldest_unresolved_inherited_age_ns.has_value()&&
          projected.roots==3&&projected.unknown_monotonic_anchors==2&&
          projected.oldest_unresolved_unix_ms==1780000000000ULL,
          "the projection keeps the summary's unknown-anchor meaning and its separate fields");
    check(evaluate_serving_repair_admission(reviewed_policy(60),generating(),live_repair(),projected,false)
              .escalation_refusal==RepairAdmissionReason::refused_age_unknown,
          "a summary with unknown anchors still cannot escalate");
    core::RepairAgeSummary same_clock;
    same_clock.oldest_monotonic_age_ns=known_age(90);same_clock.roots=1;
    const auto usable=make_repair_age_inputs(same_clock,4096,16);
    check(usable.oldest_unresolved_inherited_age_ns==known_age(90)&&
          evaluate_serving_repair_admission(reviewed_policy(60),generating(),live_repair(),usable,false)
              .reason==RepairAdmissionReason::admitted_age_escalated,
          "a summary that does carry a monotonic age is usable unchanged");
}

// No threshold is adopted: with the caller's policy left unset, escalation is
// disabled even for an arbitrarily old age.
void age_escalation_refuses_when_caller_left_policy_unset(){
    const auto decision=evaluate_serving_repair_admission(ServingRepairAdmissionPolicy{},generating(),
        live_repair(),RepairAgeInputs{.oldest_unresolved_inherited_age_ns=known_age(100000)},false);
    check(decision.escalation_bound_unset&&
          decision.escalation_refusal==RepairAdmissionReason::refused_age_policy_unset&&
          decision.reason==RepairAdmissionReason::refused_foreground_demand&&!decision.admitted(),
          "an unset policy disables escalation instead of defaulting to a number");
    check(ServingRepairAdmissionPolicy::review_required().escalation_bound_unset()&&
          ServingRepairAdmissionPolicy{}.live_repair_age_escalation_ns==std::nullopt,
          "the policy type has no meaningful default bound");
}

// P13: "Preserve that age across later edits/ancestry." Catches a scheduler that
// resets urgency because the affected position moved forward, or because a later
// ancestry snapshot could not resolve an anchor.
void oldest_inherited_age_survives_a_newer_position_with_an_older_age(){
    RetainedRepairAge retained;
    retained.observe(RepairAgeInputs{.oldest_unresolved_inherited_age_ns=known_age(90),
                                     .earliest_affected_token_position=500,.token_debt=12});
    check(retained.oldest_age_ns()==known_age(90)&&retained.latest_position()==500&&retained.latest_debt()==12,
          "the first observation is retained");
    retained.observe(RepairAgeInputs{.oldest_unresolved_inherited_age_ns=known_age(120),
                                     .earliest_affected_token_position=900,.token_debt=4});
    check(retained.oldest_age_ns()==known_age(120)&&retained.oldest_position()==900,
          "a newer position with an older inherited age keeps the older age");
    retained.observe(debt_only(0));
    check(retained.oldest_age_ns()==known_age(120)&&retained.latest_debt()==0,
          "a later observation with an unknown anchor never clears the retained age");
    retained.observe(RepairAgeInputs{.oldest_unresolved_inherited_age_ns=known_age(30),
                                     .earliest_affected_token_position=950,.token_debt=2});
    check(retained.oldest_age_ns()==known_age(120)&&retained.latest_position()==950,
          "a younger age never replaces the retained oldest age");
    const auto escalated=evaluate_serving_repair_admission(reviewed_policy(60),generating(),live_repair(),
        retained.inputs(),false);
    check(escalated.reason==RepairAdmissionReason::admitted_age_escalated&&
          escalated.oldest_unresolved_inherited_age_ns==known_age(120)&&
          escalated.earliest_affected_token_position==950&&escalated.token_debt==2,
          "the retained oldest age, not the newest position, is what gates escalation");
    RetainedRepairAge untouched;
    check(untouched.empty()&&!untouched.inputs().oldest_unresolved_inherited_age_ns.has_value(),
          "retention fabricates no age before anything was observed");
}

// P13/P14: wall-clock age, earliest affected token position and token debt are three
// separate quantities; positional debt must not be conflated with age.
void token_debt_and_position_are_reported_separately_from_age(){
    const RepairAgeInputs inputs{.oldest_unresolved_inherited_age_ns=known_age(90),
                                 .oldest_unresolved_unix_ms=1780000000000ULL,
                                 .roots=2,.earliest_affected_token_position=4096,.token_debt=37};
    const auto decision=evaluate_serving_repair_admission(reviewed_policy(60),generating(),live_repair(),
        inputs,false);
    check(decision.token_debt==37&&decision.earliest_affected_token_position==4096&&
          decision.oldest_unresolved_inherited_age_ns==known_age(90)&&
          decision.oldest_unresolved_inherited_age_ns!=decision.token_debt,
          "age, position and debt are carried as three separate fields");
    check(repair_token_debt(100,64)==36&&repair_token_debt(64,64)==0&&repair_token_debt(10,20)==0,
          "debt mirrors exact_ready's published-minus-private relation and never wraps");
    // exact_ready (generate.cpp:3506) is debt == 0.
    check(repair_token_debt(uint64_t(4)<<40,uint64_t(4)<<40)==0,"zero debt at context scale");
}

// The owner's age rule speaks only to live repair; saved maintenance stays
// idle-only even for an arbitrarily old slot. Catches "old therefore allowed"
// leaking into the disk path.
void saved_maintenance_never_escalates(){
    const auto decision=evaluate_serving_repair_admission(reviewed_policy(1),generating(),saved_save(),
        RepairAgeInputs{.oldest_unresolved_inherited_age_ns=std::numeric_limits<uint64_t>::max()},false);
    check(decision.reason==RepairAdmissionReason::refused_foreground_demand&&!decision.admitted(),
          "no amount of age admits saved maintenance against waiting generation");
    const auto against_other_work=evaluate_serving_repair_admission(reviewed_policy(1),quiet(),saved_save(),
        RepairAgeInputs{.oldest_unresolved_inherited_age_ns=std::numeric_limits<uint64_t>::max()},true);
    check(against_other_work.reason==RepairAdmissionReason::refused_other_eligible_work,
          "no amount of age admits saved maintenance against other eligible work");
    check(!decision.escalation_refusal.has_value()&&!against_other_work.escalation_refusal.has_value(),
          "escalation is never even considered for the saved class");
}

// Hard gates stay hard: extreme age, debt and position values cannot substitute for
// lifecycle, resource readiness, demand or the idle gate.
void hard_gates_are_not_expressible_as_weights(){
    const RepairAgeInputs extreme{.oldest_unresolved_inherited_age_ns=std::numeric_limits<uint64_t>::max(),
                                  .earliest_affected_token_position=std::numeric_limits<uint64_t>::max(),
                                  .token_debt=std::numeric_limits<uint64_t>::max()};
    const auto policy=reviewed_policy(1);
    ServingForegroundState retired=quiet();retired.repair_owner_retired=true;
    check(evaluate_serving_repair_admission(policy,retired,live_repair(),extreme,false)
              .reason==RepairAdmissionReason::refused_context_or_lifecycle,
          "a retired repair owner is refused whatever the age or debt is");
    RepairCandidateState unready=live_repair();unready.resource_ready=false;
    check(evaluate_serving_repair_admission(policy,quiet(),unready,extreme,false)
              .reason==RepairAdmissionReason::refused_not_resource_ready,
          "resource readiness is required even for the oldest possible age");
    RepairCandidateState undemanded=live_repair();undemanded.demand=false;
    check(evaluate_serving_repair_admission(policy,quiet(),undemanded,extreme,false)
              .reason==RepairAdmissionReason::refused_not_demanded,
          "an undemanded unit is refused even at maximum age");
    check(evaluate_serving_repair_admission(policy,generating(),live_repair(),debt_only(
              std::numeric_limits<uint64_t>::max()),false)
              .escalation_refusal==RepairAdmissionReason::refused_age_unknown,
          "huge token debt cannot stand in for a known inherited age");
    // The age gate is a bound comparison, not a magnitude that can be outshouted.
    const auto unbound=evaluate_serving_repair_admission(
        reviewed_policy(std::numeric_limits<uint64_t>::max()/ns_per_second),generating(),live_repair(),
        RepairAgeInputs{.oldest_unresolved_inherited_age_ns=known_age(90)},false);
    check(unbound.reason==RepairAdmissionReason::refused_foreground_demand&&
          unbound.escalation_refusal==RepairAdmissionReason::refused_age_below_bound,
          "a huge caller bound does not escalate a 90 second slot");
}

// generate.cpp:3479 refuses with "retained repair has no active owner" once
// finish() retired the owner; the kernel must not admit anything for it.
void retired_owner_refuses_before_the_other_gates(){
    ServingForegroundState retired=quiet();retired.repair_owner_retired=true;
    check(evaluate_serving_repair_admission(ServingRepairAdmissionPolicy::review_required(),retired,
              live_repair(),debt_only(8),false).reason==RepairAdmissionReason::refused_context_or_lifecycle,
          "a retired owner's live repair unit is refused in an idle window");
    check(evaluate_serving_repair_admission(ServingRepairAdmissionPolicy::review_required(),retired,
              saved_save(),debt_only(8),false).reason==RepairAdmissionReason::refused_context_or_lifecycle,
          "a retired owner's saved unit is refused in an idle window");
}

// A mislabeled candidate must not slip past the class gate, and this kernel decides
// only repair/save units.
void mislabeled_or_unrelated_units_are_refused(){
    const auto policy=ServingRepairAdmissionPolicy::review_required();
    RepairCandidateState live_save=live_repair();live_save.unit=core::AdmissionUnit::save;
    check(evaluate_serving_repair_admission(policy,quiet(),live_save,debt_only(4),false)
              .reason==RepairAdmissionReason::refused_class_unit_mismatch,
          "a live_repair candidate carrying a save unit cannot use the live path");
    RepairCandidateState maintenance_repair=saved_save();maintenance_repair.unit=core::AdmissionUnit::repair;
    check(evaluate_serving_repair_admission(policy,generating(),maintenance_repair,debt_only(4),false)
              .reason==RepairAdmissionReason::refused_class_unit_mismatch,
          "a saved_maintenance candidate relabelled as a repair unit cannot bypass the idle gate");
    RepairCandidateState foreground_tagged=live_repair();
    foreground_tagged.work_class=core::DependencyWorkClass::foreground;
    check(evaluate_serving_repair_admission(policy,quiet(),foreground_tagged,debt_only(4),false)
              .reason==RepairAdmissionReason::refused_class_unit_mismatch,
          "a foreground candidate is not a repair admission decision");
    RepairCandidateState decode_unit=live_repair();decode_unit.unit=core::AdmissionUnit::decode;
    check(evaluate_serving_repair_admission(policy,quiet(),decode_unit,debt_only(4),false)
              .reason==RepairAdmissionReason::refused_unsupported_unit,
          "the kernel refuses units it is not a repair/save decision for");
}

// Escalation may only cut in where the existing live handoff is coherent
// (generate.cpp:19533-19536): room below the context limit and continuing
// generation. Catches an escalation at a point that cannot hand the slot back.
void escalation_refuses_at_an_incoherent_handoff_point(){
    const RepairAgeInputs old{.oldest_unresolved_inherited_age_ns=known_age(90)};
    ServingForegroundState full=generating();full.context_room=false;
    const auto at_context_limit=evaluate_serving_repair_admission(reviewed_policy(60),full,live_repair(),old,false);
    check(at_context_limit.reason==RepairAdmissionReason::refused_context_or_lifecycle&&
          at_context_limit.escalation_refusal==RepairAdmissionReason::refused_context_or_lifecycle&&
          !at_context_limit.escalation_coherent&&!at_context_limit.admitted(),
          "no context room means the escalation cannot hand the slot back");
    ServingForegroundState stopped=generating();stopped.stop_or_eos_seen=true;
    check(evaluate_serving_repair_admission(reviewed_policy(60),stopped,live_repair(),old,false)
              .reason==RepairAdmissionReason::refused_context_or_lifecycle,
          "a contradictory stop/EOS witness blocks escalation instead of being ignored");
    // Idle admission is unaffected by those two fields: a finished-but-unretired
    // request is exactly where repair should run.
    check(evaluate_serving_repair_admission(reviewed_policy(60),quiet(),live_repair(),old,false)
              .reason==RepairAdmissionReason::admitted_idle,
          "the idle gate does not require the escalation handoff condition");
}

// Reactor use requires a stable, repeatable answer with no allocation and no
// exception for expected inputs, plus defaults that admit nothing.
void decision_is_deterministic_and_starts_disabled(){
    check(RepairCandidateState{}.demand==false&&RepairCandidateState{}.resource_ready==false,
          "a default candidate carries no demand or resources");
    check(RepairAdmissionDecision{}.admitted()==false,
          "a default decision admits nothing");
    const auto policy=reviewed_policy(60);
    for(bool other:{false,true})for(bool ready:{false,true})for(uint64_t lines:{0ULL,4ULL}){
        ServingForegroundState state=quiet();state.queued_input_lines=lines;
        RepairCandidateState candidate=live_repair();candidate.resource_ready=ready;
        const RepairAgeInputs age{.oldest_unresolved_inherited_age_ns=known_age(90),.token_debt=9};
        const auto first=evaluate_serving_repair_admission(policy,state,candidate,age,other);
        const auto second=evaluate_serving_repair_admission(policy,state,candidate,age,other);
        check(first==second,"identical inputs give an identical decision");
        check(std::string(repair_admission_reason_name(first.reason))==
              std::string(repair_admission_reason_name(second.reason)),"reason name is stable");
        check(first.work_class==candidate.work_class&&first.unit==candidate.unit,
              "the decision never rewrites the candidate identity");
        check(std::string(repair_admission_escalation_refusal_name(first.escalation_refusal))==
              std::string(repair_admission_escalation_refusal_name(second.escalation_refusal)),
              "the escalation sub-reason is stable and named");
    }
    for(const auto reason:{RepairAdmissionReason::admitted_idle,RepairAdmissionReason::admitted_age_escalated,
                           RepairAdmissionReason::refused_foreground_demand,
                           RepairAdmissionReason::refused_other_eligible_work,
                           RepairAdmissionReason::refused_not_idle_authorized,
                           RepairAdmissionReason::refused_foreground_return_unprotected,
                           RepairAdmissionReason::refused_context_or_lifecycle,
                           RepairAdmissionReason::refused_not_resource_ready,
                           RepairAdmissionReason::refused_not_demanded,
                           RepairAdmissionReason::refused_age_unknown,
                           RepairAdmissionReason::refused_age_policy_unset,
                           RepairAdmissionReason::refused_age_below_bound,
                           RepairAdmissionReason::refused_class_unit_mismatch,
                           RepairAdmissionReason::refused_unsupported_unit}){
        const char* name=repair_admission_reason_name(reason);
        check(name&&*name,"every reason has an explicit log name");
    }
    check(std::string(repair_admission_reason_name(RepairAdmissionReason::admitted_age_escalated))==
          "admitted_age_escalated"&&
          std::string(repair_admission_reason_name(RepairAdmissionReason::refused_age_unknown))==
          "refused_age_unknown","reason names match the enumerator spelling");
}

// Importance/stochastic weighting was exploration, not an adopted formula. The seam
// exists but is inert: flipping its flag changes no decision, and the kernel takes
// no weight parameter at all.
void weighting_probe_is_disabled_and_never_consulted(){
    SelectionWeightingProbe off;
    SelectionWeightingProbe on;on.enabled=true;
    check(!SelectionWeightingProbe{}.enabled,"the weighting probe is disabled by default");
    check(selection_weighting_probe(off)==SelectionWeightingStatus::not_adopted&&
          selection_weighting_probe(on)==SelectionWeightingStatus::not_adopted,
          "the weighting probe reports not adopted whether or not its flag is set");
    check(std::string(selection_weighting_status_name(selection_weighting_probe(on)))=="not adopted",
          "the weighting probe name is reported honestly");
    const auto policy=reviewed_policy(60);
    const auto without=evaluate_serving_repair_admission(policy,generating(),live_repair(),
        RepairAgeInputs{.oldest_unresolved_inherited_age_ns=known_age(90)},false);
    const auto with=evaluate_serving_repair_admission(policy,generating(),live_repair(),
        RepairAgeInputs{.oldest_unresolved_inherited_age_ns=known_age(90)},false);
    check(without==with,"no weighting state can change the hard decision");
}
void scheduler_configuration_and_ticket_stream_are_explicit(){
    auto absent=parse_serving_repair_scheduler_config({nullptr,nullptr,nullptr,nullptr,nullptr});
    check(absent.valid&&!absent.enabled&&!absent.policy.live_repair_age_escalation_ns,
          "absent policy adopts no age threshold or stochastic formula");
    auto configured=parse_serving_repair_scheduler_config({"100","3","2","1","0"});
    check(configured.valid&&configured.enabled&&configured.policy.live_repair_age_escalation_ns==100&&
          configured.live_weight==3&&configured.foreground_weight==2&&configured.max_consecutive==1&&configured.seed==0,
          "complete experimental policy preserves explicit inputs, including seed zero");
    const std::array<const char*,5> invalid[]={
        {"100",nullptr,"2","1","0"},{"100","3","2","0","0"},{"-1","3","2","1","0"},
        {"100","3x","2","1","0"},{"100","1000001","2","1","0"},
        {"18446744073709551616","3","2","1","0"},{"","3","2","1","0"},
        {" 100","3","2","1","0"},{"+100","3","2","1","0"}};
    for(const auto& values:invalid)check(!parse_serving_repair_scheduler_config(values).valid,
                                      "partial, malformed, overflow and unbounded policy inputs refuse");
    uint64_t first=3,second=3;
    check(serving_repair_uniform_ticket(first,101)==50,"specified seeded stream reproduces the known first ticket");
    second=first;
    for(unsigned i=0;i<100;++i){
        auto a=serving_repair_uniform_ticket(first,17),b=serving_repair_uniform_ticket(second,17);
        check(a==b&&a<17,"same explicit seed/state reproduces bounded unbiased tickets");
    }
}
void weighted_selection_preserves_hard_gates_and_bounds(){
    ServingRepairAdmissionPolicy policy;policy.live_repair_age_escalation_ns=60;
    ServingRepairSelectionBoundary b;
    b.safe_boundary=true;b.now_ns=1000;b.clock_domain=7;b.max_candidates=3;
    b.foreground.context_room=true;b.foreground.queued_input_lines=1;
    b.foreground_weight=2;b.max_consecutive_escalated_units=1;
    ServingRepairSelectionCandidate young;
    young.work=live_repair();young.age.oldest_unresolved_inherited_age_ns=30;young.age.token_debt=9;
    young.weight=100;young.observed_at_ns=b.now_ns;young.clock_domain=7;young.record_version=2;young.ancestry_serial=3;
    auto old=young;old.age.oldest_unresolved_inherited_age_ns=90;old.weight=3;
    auto saved=old;saved.work.work_class=core::DependencyWorkClass::saved_maintenance;
    saved.work.unit=core::AdmissionUnit::save;saved.work.idle_authorized=true;
    saved.work.foreground_return_protected=true;saved.weight=UINT64_MAX;
    std::array<ServingRepairSelectionCandidate,3> candidates{young,old,saved};
    for(uint64_t ticket=0;ticket<5;++ticket){
        auto result=select_serving_repair_unit(policy,b,candidates,ticket);
        check(result.status==ServingRepairSelectionStatus::ready&&result.total_weight==5,
              "weights count only foreground and eligible aged live repair");
        check(result.foreground==(ticket<2),"explicit uniform tickets implement supplied 2:3 weights");
        if(ticket>=2)check(result.candidate==1&&result.record_version==2&&result.ancestry_serial==3&&
            result.admission.reason==RepairAdmissionReason::admitted_age_escalated,
            "selection carries the actual eligible owner versions and escalation reason");
    }
    check(select_serving_repair_unit(policy,b,candidates,5).status==ServingRepairSelectionStatus::invalid_ticket,
          "out-of-range random ticket refuses");
    b.consecutive_escalated_units=1;
    check(select_serving_repair_unit(policy,b,candidates,0).total_weight==2,
          "bounded escalated share returns control to waiting foreground");
    b.consecutive_escalated_units=0;
    candidates[1].observed_at_ns--;
    check(select_serving_repair_unit(policy,b,candidates,0).total_weight==2,"cached age cannot escalate");
    candidates[1]=old;candidates[1].clock_domain=8;
    check(select_serving_repair_unit(policy,b,candidates,0).total_weight==2,"foreign clock cannot escalate");
    candidates[1]=old;candidates[1].age.oldest_unresolved_inherited_age_ns.reset();
    check(select_serving_repair_unit(policy,b,candidates,0).total_weight==2,"unknown ancestry cannot escalate");
    candidates[1]=old;candidates[1].work.resource_ready=false;
    check(select_serving_repair_unit(policy,b,candidates,0).total_weight==2,"large weight cannot open resource gate");
    candidates[1]=old;b.other_lane_work=true;
    check(select_serving_repair_unit(policy,b,candidates,0).status==ServingRepairSelectionStatus::invalid_boundary,
          "age cannot preempt another lane");
    b.other_lane_work=false;b.terminal_input=true;
    check(select_serving_repair_unit(policy,b,candidates,0).status==ServingRepairSelectionStatus::invalid_boundary,
          "terminal input vetoes all maintenance");
    b.terminal_input=false;b.foreground.queued_input_lines=0;
    auto idle=select_serving_repair_unit(policy,b,candidates,100);
    check(idle.status==ServingRepairSelectionStatus::ready&&idle.candidate==1&&idle.total_weight==103,
          "idle live owners use explicit weights and veto saved maintenance");
    candidates[0].work.demand=false;candidates[1].work.demand=false;
    check(select_serving_repair_unit(policy,b,candidates,0).candidate==2,"saved maintenance can run alone at idle");
    candidates[2].work.foreground_return_protected=false;
    check(select_serving_repair_unit(policy,b,candidates,0).status==ServingRepairSelectionStatus::no_work,
          "saved work needs protected return even when alone");
    candidates={young,old,saved};candidates[0].weight=UINT64_MAX;
    check(select_serving_repair_unit(policy,b,candidates,0).status==ServingRepairSelectionStatus::weight_overflow,
          "weight sum overflow refuses before selection");
    candidates[0].weight=0;
    check(select_serving_repair_unit(policy,b,candidates,0).status==ServingRepairSelectionStatus::invalid_weight,
          "eligible zero weight refuses instead of silently starving an owner");
    b.max_candidates=2;
    check(select_serving_repair_unit(policy,b,candidates,0).status==ServingRepairSelectionStatus::capacity,
          "candidate workspace bound is explicit");
}
void authentic_age_projection_uses_roots_and_preserves_unknown(){
    core::RepairCausalRoot root;
    root.identity[0]=1;root.original_history[0]=2;root.engine_generation=1;root.window_generation=1;
    root.runtime_incarnation=7;root.original_cut=1;root.original_begin=4;root.original_count=12;
    root.first_unresolved_unix_ms=100;
    check(bind_initial_serving_repair_birth(root,{7,10},7),"causal import birth binds to its runtime");
    check(!bind_initial_serving_repair_birth(root,{7,20},7)&&root.monotonic_birth.ticks_ns==10,
          "reconstruction cannot replace an authentic old birth with a newer birth");
    auto foreign=root;foreign.monotonic_birth={};
    check(!bind_initial_serving_repair_birth(foreign,{8,10},7)&&foreign.monotonic_birth.domain==0,
          "foreign runtime event refuses without changing root");
    check(!bind_initial_serving_repair_birth(foreign,{0,10},7),"unknown domain cannot carry fabricated ticks");
    check(bind_initial_serving_repair_birth(foreign,{},7)&&foreign.monotonic_birth.domain==0,
          "restored unknown input stays unknown without stamping current time");
    const core::RepairAncestryLimits limits{64,core::RepairAncestry::retained_bytes()};
    core::RepairAncestry ancestry;
    check(core::make_repair_ancestry({&root,1},limits,ancestry)==core::RepairAncestryStatus::ok,
          "production ancestry accepts actual root fields");
    core::RepairSpatialCoverage coverage;
    check(core::make_repair_spatial_coverage(ancestry,4,16,coverage)==core::RepairAncestryStatus::ok,
          "production coverage binds ancestry");
    const auto first=observe_serving_repair_age(coverage,{7,100},16,8,3);
    const auto later=observe_serving_repair_age(coverage,{7,140},16,12,3);
    check(first.valid&&first.age.oldest_unresolved_inherited_age_ns==90&&first.age.token_debt==8&&
          later.age.oldest_unresolved_inherited_age_ns==130&&later.age.token_debt==4,
          "real clock passage ages roots independently of shrinking token debt");
    core::RepairSpatialCoverage partial;
    check(core::advance_repair_spatial_begin(coverage,12,partial)==core::RepairAncestryStatus::ok&&
          observe_serving_repair_age(partial,{7,140},16,12,4).age.oldest_unresolved_inherited_age_ns==130,
          "partial attachment preserves causal age while changing spatial position");
    check(!observe_serving_repair_age(coverage,{8,140},16,8,3).age.oldest_unresolved_inherited_age_ns&&
          !observe_serving_repair_age(coverage,{7,5},16,8,3).age.oldest_unresolved_inherited_age_ns,
          "foreign clock and future birth cannot escalate");
    auto unknown=root;unknown.identity[0]=3;unknown.monotonic_birth={};
    core::RepairAncestry unknown_ancestry,combined;
    check(core::make_repair_ancestry({&unknown,1},limits,unknown_ancestry)==core::RepairAncestryStatus::ok&&
          core::union_repair_ancestry(ancestry,unknown_ancestry,limits,combined)==core::RepairAncestryStatus::ok,
          "real union retains a new unknown ancestor alongside known old roots");
    core::RepairSpatialCoverage inherited;
    check(core::make_repair_spatial_coverage(combined,12,16,inherited)==core::RepairAncestryStatus::ok,
          "inherited coverage retains both roots");
    auto observation=observe_serving_repair_age(inherited,{7,140},16,12,4);
    check(observation.valid&&!observation.age.oldest_unresolved_inherited_age_ns&&
          observation.age.unknown_monotonic_anchors==1&&observation.age.roots==2,
          "known old age cannot hide an unknown unresolved inherited root");
    check(!observe_serving_repair_age(coverage,{7,140},16,17,3).valid,
          "invalid controller cut relation refuses observation");
    check(serving_repair_clock_now(0).domain==0&&serving_repair_clock_now(7).domain==7,
          "production steady clock uses explicit runtime domain and unknown stays unknown");
}
} // namespace

int main(){
    admits_live_repair_when_queue_empty_and_request_finished_but_unretired();
    refuses_live_repair_while_its_own_request_is_generating();
    refuses_live_repair_while_a_next_line_is_queued();
    saved_maintenance_refused_when_any_other_eligible_work_exists();
    saved_maintenance_refused_without_foreground_return_protection();
    age_escalation_admits_live_repair_over_waiting_generation();
    age_escalation_refuses_unknown_age();
    age_escalation_refuses_when_caller_left_policy_unset();
    oldest_inherited_age_survives_a_newer_position_with_an_older_age();
    token_debt_and_position_are_reported_separately_from_age();
    saved_maintenance_never_escalates();
    hard_gates_are_not_expressible_as_weights();
    retired_owner_refuses_before_the_other_gates();
    mislabeled_or_unrelated_units_are_refused();
    escalation_refuses_at_an_incoherent_handoff_point();
    decision_is_deterministic_and_starts_disabled();
    weighting_probe_is_disabled_and_never_consulted();
    scheduler_configuration_and_ticket_stream_are_explicit();
    weighted_selection_preserves_hard_gates_and_bounds();
    authentic_age_projection_uses_roots_and_preserves_unknown();
    std::puts("serving_repair_admission_test passed");
    return 0;
}
