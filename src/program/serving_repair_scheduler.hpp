#pragma once
// Cooperative selection only: no clocks, controller calls, leases or random generator.
// The caller supplies priced weights and a uniform ticket in [0,total_weight).
// Hard eligibility is evaluated before weights. No threshold or weight formula is adopted here.
#include "serving_repair_admission.hpp"
#include <limits>
#include <array>
#include <charconv>
#include <cstdlib>
#include <string_view>
#include <span>

namespace strata::program {
// Experimental policy v1 supplies weights directly; it does not infer an importance formula.
// All five fields are mandatory together. Absent configuration means idle entry-order
// progress only and no active-generation repair while foreground remains.
struct ServingRepairSchedulerConfig {
    bool enabled=false, valid=true;
    ServingRepairAdmissionPolicy policy;
    uint64_t live_weight=0, foreground_weight=0, max_consecutive=0, seed=0;
};
inline constexpr std::array<const char*,5> kServingRepairSchedulerEnv{
    "STRATA_SERVING_REPAIR_AGE_NS", "STRATA_SERVING_REPAIR_LIVE_WEIGHT",
    "STRATA_SERVING_REPAIR_FOREGROUND_WEIGHT", "STRATA_SERVING_REPAIR_MAX_ESCALATED_UNITS",
    "STRATA_SERVING_REPAIR_RANDOM_SEED"};
inline ServingRepairSchedulerConfig parse_serving_repair_scheduler_config(
        const std::array<const char*,5>& values) noexcept {
    ServingRepairSchedulerConfig config;
    size_t present=0;for(auto value:values)if(value)++present;
    if(!present)return config;
    if(present!=values.size()){config.valid=false;return config;}
    std::array<uint64_t,5> parsed{};
    for(size_t i=0;i<values.size();++i){
        const std::string_view value(values[i]);
        const auto result=std::from_chars(value.data(),value.data()+value.size(),parsed[i]);
        if(value.empty() || result.ec!=std::errc{} || result.ptr!=value.data()+value.size() ||
           (i!=4 && !parsed[i])){config.valid=false;return config;}
    }
    // Bounded integer ticket arithmetic and bounded configured consecutive work.
    if(parsed[1]>1000000 || parsed[2]>1000000 || parsed[3]>1000000){config.valid=false;return config;}
    config.enabled=true;config.policy.live_repair_age_escalation_ns=parsed[0];
    config.live_weight=parsed[1];config.foreground_weight=parsed[2];
    config.max_consecutive=parsed[3];config.seed=parsed[4];return config;
}
inline ServingRepairSchedulerConfig serving_repair_scheduler_config() noexcept {
    std::array<const char*,5> values{};
    for(size_t i=0;i<values.size();++i)values[i]=std::getenv(kServingRepairSchedulerEnv[i]);
    return parse_serving_repair_scheduler_config(values);
}
// Specified SplitMix64 stream plus rejection sampling: seed and draws reproduce
// across standard libraries; modulo reduction alone would bias importance weights.
inline uint64_t serving_repair_uniform_ticket(uint64_t& state,uint64_t total) noexcept {
    if(!total)return 0;
    const uint64_t threshold=(uint64_t(0)-total)%total;
    for(;;){
        uint64_t word=(state+=0x9e3779b97f4a7c15ULL);
        word=(word^(word>>30))*0xbf58476d1ce4e5b9ULL;
        word=(word^(word>>27))*0x94d049bb133111ebULL;word^=word>>31;
        if(word>=threshold)return word%total;
    }
}
struct ServingRepairSelectionCandidate {
    RepairCandidateState work;
    RepairAgeInputs age;
    uint64_t weight=0;
    // Trusted caller's refreshed observation, bound to the selected owner/publication.
    uint64_t observed_at_ns=0, clock_domain=0, record_version=0, ancestry_serial=0;
};
struct ServingRepairSelectionBoundary {
    ServingForegroundState foreground;
    bool safe_boundary=false, terminal_input=false, other_lane_work=false;
    uint64_t now_ns=0, clock_domain=0;
    size_t max_candidates=0;
    uint64_t foreground_weight=0;
    uint64_t consecutive_escalated_units=0, max_consecutive_escalated_units=0;
};
enum class ServingRepairSelectionStatus { ready, no_work, invalid_boundary, capacity, invalid_weight,
    weight_overflow, invalid_ticket };
struct ServingRepairSelection {
    ServingRepairSelectionStatus status=ServingRepairSelectionStatus::no_work;
    bool foreground=false;
    size_t candidate=0;
    uint64_t total_weight=0, record_version=0, ancestry_serial=0, observed_at_ns=0;
    RepairAdmissionDecision admission;
};

// A cached duration is not a clock. Only an observation refreshed for THIS boundary,
// in this process clock domain, can escalate. Unknown roots in the real producer must
// remain unknown even if an older retained observation was known.
inline RepairAdmissionDecision serving_repair_selection_admission(
        const ServingRepairAdmissionPolicy& policy,const ServingRepairSelectionBoundary& boundary,
        const ServingRepairSelectionCandidate& candidate,bool other_nonmaintenance) noexcept {
    RepairAgeInputs age=candidate.age;
    if(!boundary.clock_domain || candidate.clock_domain!=boundary.clock_domain ||
       candidate.observed_at_ns!=boundary.now_ns || !candidate.record_version || !candidate.ancestry_serial)
        age.oldest_unresolved_inherited_age_ns.reset();
    auto decision=evaluate_serving_repair_admission(policy,boundary.foreground,candidate.work,age,
                                                   other_nonmaintenance);
    if(decision.reason==RepairAdmissionReason::admitted_age_escalated &&
       (!boundary.max_consecutive_escalated_units ||
        boundary.consecutive_escalated_units>=boundary.max_consecutive_escalated_units)) {
        decision.reason=RepairAdmissionReason::refused_foreground_demand;
    }
    return decision;
}

// Repeat with the same immutable boundary/candidates and a new uniform ticket after
// observing total_weight. The execution seam must revalidate the returned versions.
inline ServingRepairSelection select_serving_repair_unit(
        const ServingRepairAdmissionPolicy& policy,const ServingRepairSelectionBoundary& boundary,
        std::span<const ServingRepairSelectionCandidate> candidates,uint64_t ticket) noexcept {
    ServingRepairSelection result;
    if(!boundary.safe_boundary || boundary.terminal_input || boundary.other_lane_work) {
        result.status=ServingRepairSelectionStatus::invalid_boundary; return result;
    }
    if(!boundary.max_candidates || candidates.size()>boundary.max_candidates) {
        result.status=ServingRepairSelectionStatus::capacity; return result;
    }
    bool live_eligible=false;
    for(const auto& candidate:candidates)
        if(candidate.work.work_class==core::DependencyWorkClass::live_repair &&
           serving_repair_selection_admission(policy,boundary,candidate,false).admitted())live_eligible=true;
    const bool foreground=boundary.foreground.decode_waiting();
    uint64_t total=0;
    auto add=[&](uint64_t weight) {
        if(!weight) {result.status=ServingRepairSelectionStatus::invalid_weight;return false;}
        if(weight>std::numeric_limits<uint64_t>::max()-total) {
            result.status=ServingRepairSelectionStatus::weight_overflow;return false;
        }
        total+=weight;return true;
    };
    if(foreground && !add(boundary.foreground_weight))return result;
    for(const auto& candidate:candidates)
        if(serving_repair_selection_admission(policy,boundary,candidate,foreground||live_eligible).admitted() &&
           !add(candidate.weight))return result;
    result.total_weight=total;
    if(!total)return result;
    if(ticket>=total) {result.status=ServingRepairSelectionStatus::invalid_ticket;return result;}
    result.status=ServingRepairSelectionStatus::ready;
    if(foreground) {
        if(ticket<boundary.foreground_weight) {result.foreground=true;return result;}
        ticket-=boundary.foreground_weight;
    }
    for(size_t i=0;i<candidates.size();++i) {
        const auto& candidate=candidates[i];
        const auto admission=serving_repair_selection_admission(policy,boundary,candidate,foreground||live_eligible);
        if(!admission.admitted())continue;
        if(ticket<candidate.weight) {
            result.candidate=i;result.admission=admission;
            result.record_version=candidate.record_version;result.ancestry_serial=candidate.ancestry_serial;
            result.observed_at_ns=candidate.observed_at_ns;return result;
        }
        ticket-=candidate.weight;
    }
    result.status=ServingRepairSelectionStatus::invalid_ticket;return result;
}
} // namespace strata::program
