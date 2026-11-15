// CPU lifecycle gate for the SERVING RAW adapter contract (include/strata/core/serving_raw_contracts.hpp).
// It exercises the same predicates ServingRawSession uses - not a copy - and pins the three properties the
// 2026-10-29 source review found broken:
//   1. capture/restore are legal ONLY at the drained cut, never while driver work is running (so an "in flight"
//      request flag can never be mistaken for permission to touch device state);
//   2. completion is legal only while running, and an incomplete/fabricated cut is refused BEFORE the
//      transaction is asked to certify it;
//   3. destroying an owner whose transaction is running/capturing/restoring/poisoned terminates the process -
//      clearing token or checkpoint metadata is not recovery.
#include "strata/core/serving_raw_contracts.hpp"
#include "strata/core/serving_raw_metadata.hpp"
#include "strata/core/native_serial_serving.hpp"

#include <array>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::core;
using namespace strata::core::serving_raw;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("FAIL: %s\n", what.c_str()); ++failures; }
}
StateDigest zero_digest() { return {}; }
StateDigest real_digest() { StateDigest d{}; d[7] = 0x5a; return d; }
}   // namespace

int main() {
    // Actual native serial policy used by the IQ loader: backing capacity must satisfy Verifier::init without
    // changing requested spec0, turning on any drafter, or requiring suffix/owned flags on the cold reference.
    {
        struct NativePolicy {
            bool serve=true,keep_canonical=false,batch_mtp=false,suffix_cache=false,owned_session_lifetime=false;
            int spec=0,prefill_chunk=16,suffix_draft=0,lookup_chain=0,pipeline_windows=0,batch=0;
            std::string native_preset="unit-native",mtp,draft,draft_model,spec_oracle,spec_follow;
        };
        namespace serial=strata::core::native_serial_serving;
        NativePolicy cold;
        check(serial::eligible(true,cold),"native feature-free unowned cold reference resolves serial spec0");
        auto repaired=cold;repaired.suffix_cache=true;repaired.owned_session_lifetime=true;
        check(serial::eligible(true,repaired),"native repaired owned serving resolves the same serial profile");
        check(serial::verifier_capacity(cold.spec,true)==2&&serial::execution_window(cold.spec,true)==1&&cold.spec==0,
              "native serial uses capacity2 and actual T1 while preserving requested spec0");
        using Change=void(*)(NativePolicy&);
        const Change writers[]={
            [](auto& p){p.mtp="mtp";},[](auto& p){p.draft="draft";},[](auto& p){p.draft_model="draft-model";},
            [](auto& p){p.spec_oracle="oracle";},[](auto& p){p.spec_follow="follow";},
            [](auto& p){p.suffix_draft=1;},[](auto& p){p.lookup_chain=1;},
            [](auto& p){p.pipeline_windows=1;},[](auto& p){p.batch=1;},[](auto& p){p.batch_mtp=true;},
            [](auto& p){p.serve=false;},[](auto& p){p.prefill_chunk=0;},
            [](auto& p){p.native_preset.clear();},[](auto& p){p.keep_canonical=true;},
        };
        for(auto change:writers){auto unsupported=cold;change(unsupported);
            check(!serial::eligible(true,unsupported),"real draft/batch writer or unsupported loader cannot acquire serial marker");}
        check(!serial::eligible(false,cold),"canonical-pack behavior remains unchanged");
        for(int requested:{1,2,8}){
            auto existing=cold;existing.spec=requested;
            check(!serial::eligible(true,existing)&&serial::verifier_capacity(requested,false)==requested&&
                  serial::execution_window(requested,false)==requested,
                  "existing explicit speculative capacities and invalid spec1 are unchanged");
        }
        auto experiment=cold;experiment.serve=false;experiment.spec=2;
        check(!serial::eligible(true,experiment)&&serial::verifier_capacity(experiment.spec,false)==2,
              "existing standalone experiment keeps its independently qualified storage2 path");
    }
    const Phase all[] = {Phase::absent, Phase::idle, Phase::running, Phase::drained, Phase::capturing,
                         Phase::restoring, Phase::poisoned};

    // 1. request operations and RAW phases are distinct permissions
    for (Phase p : all) check(begin_allowed(p) == (p == Phase::idle || p == Phase::drained),
                              "begin only from idle/drained");
    for (Phase p : all) check(complete_allowed(p) == (p == Phase::running), "completion only while running");
    for (Phase p : all) {
        check(capture_allowed(p) == (p == Phase::drained), "capture only at a drained all-stage cut");
        check(restore_allowed(p) == (p == Phase::drained), "restore only at a drained cut");
    }
    check(!capture_allowed(Phase::running), "a running request alone never grants capture (the reviewed defect)");
    check(!restore_allowed(Phase::running), "a running request alone never grants restore");
    check(capture_allowed(Phase::drained), "capture IS reachable: after a real drain");

    // 2. destruction over an unclosed or poisoned fence is a failstop, not a cleanup
    for (Phase p : all)
        check(terminates_on_destruction(p) == (p == Phase::running || p == Phase::capturing ||
                                               p == Phase::restoring || p == Phase::poisoned),
                              "terminate on destroying an open/poisoned transaction");

    // 3. the completion cut contract
    const StateDigest history = real_digest();
    const CutFacts good{7, 216, 216, 0, &history};
    check(cut_facts_acceptable(good, 2, 512), "a real all-stage cut at 216 accepted/forwarded is acceptable");
    check(cut_facts_acceptable(CutFacts{7, 216, 216, 216, &history}, 2, 512),
          "exact_frontier == forwarded is acceptable (a certified frontier)");
    check(!cut_facts_acceptable(CutFacts{0, 216, 216, 0, &history}, 2, 512), "cut id 0 is refused");
    check(!cut_facts_acceptable(CutFacts{7, 216, 216, 0, nullptr}, 2, 512), "a missing history digest is refused");
    const StateDigest empty = zero_digest();
    check(!cut_facts_acceptable(CutFacts{7, 216, 216, 0, &empty}, 2, 512),
          "a zeroed digest (nothing actually certified) is refused, not fabricated");
    check(!cut_facts_acceptable(CutFacts{7, 200, 216, 0, &history}, 2, 512), "forwarded > accepted is refused");
    check(!cut_facts_acceptable(CutFacts{7, 216, 216, 217, &history}, 2, 512),
          "exact_frontier > forwarded is refused (no certificate past the work)");
    check(!cut_facts_acceptable(CutFacts{7, 513, 513, 0, &history}, 2, 512), "accepted > max_tokens is refused");
    check(!cut_facts_acceptable(good, 0, 512),
          "zero stage count is refused: an all-stage cut must cover every stage (empty spans are not a cut)");
    check(digest_present(history) && !digest_present(empty), "digest presence is byte-real, not pointer-truth");

    // 5. ACTUAL adapter transition sequences (the reviewed defect: predicates passed while no public sequence
    //    could reach capture/restore). Lifecycle is the very object ServingRawSession routes every call through.
    {
        Lifecycle seq;
        check(seq.phase() == Phase::absent && !seq.request_open(), "a fresh adapter session is idle-closed");
        check(!seq.allow(Lifecycle::Event::begin_work), "begin_work before open_request is refused");
        check(!seq.allow(Lifecycle::Event::capture), "capture before any request is refused");
        check(seq.apply(Lifecycle::Event::open_request), "open_request");
        seq.set_observed_phase(Phase::idle);            // what a freshly initialized transaction reports
        check(!seq.allow(Lifecycle::Event::capture), "capture before any work phase is refused");
        check(seq.apply(Lifecycle::Event::begin_work) && seq.phase() == Phase::running,
              "begin_work enters the running work phase");
        check(!seq.allow(Lifecycle::Event::capture),
              "capture is refused while work runs - the exact sequence the review found unreachable before");
        check(!seq.allow(Lifecycle::Event::close_request),
              "closing a request with work still running is refused (destroying it would terminate)");
        check(seq.apply(Lifecycle::Event::drain_work) && seq.phase() == Phase::drained && seq.request_open(),
              "drain_work leaves the REQUEST open at a drained cut");
        check(seq.allow(Lifecycle::Event::capture) && seq.allow(Lifecycle::Event::restore),
              "capture/restore ARE reachable: after a real drain, inside the open request");
        check(seq.apply(Lifecycle::Event::capture) && seq.apply(Lifecycle::Event::restore),
              "capture/restore transitions apply at the drained cut");
        check(seq.allow(Lifecycle::Event::begin_work), "a second work phase may begin from drained");
        check(seq.apply(Lifecycle::Event::close_request) && !seq.request_open(),
              "close_request ends the request while the session stays drained for the next one");
        check(!seq.allow(Lifecycle::Event::capture), "capture after close is refused");

        Lifecycle bad;
        bad.apply(Lifecycle::Event::open_request);
        bad.set_observed_phase(Phase::idle);
        bad.apply(Lifecycle::Event::begin_work);
        bad.apply(Lifecycle::Event::abandon);
        check(bad.phase() == Phase::poisoned && !bad.request_open(), "abandon marks the session poisoned");
        check(!bad.allow(Lifecycle::Event::open_request), "no new request over a poisoned session");
        check(!bad.allow(Lifecycle::Event::capture) && !bad.allow(Lifecycle::Event::begin_work),
              "a poisoned session admits no RAW operation");
    }

    // 6. Request-parameterized REPAIR PLAN admission (EXECUTION-PLAN step 3).  The serving mode admits exactly ONE
    //    attachment and NO foreground step (the controller's foreground() generates tokens, it does not re-read the
    //    request's own), so every number here is a request fact and each refusal class is exercised.
    {
        const std::vector<uint64_t> one{36};
        RepairPlanRefusal why = RepairPlanRefusal::none;
        check(repair_plan_acceptable(RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(one), 0, 4}, why),
              "one page-aligned frontier inside the donor run admits");
        check(why == RepairPlanRefusal::none, "an admitted plan reports no reason");
        check(kServingRepairMaxFrontiers == 1 && kServingRepairMinGap == 4,
              "the shared constants the controller and this gate both use are the qualified ones");

        auto refuses = [&](RepairPlanFacts f, RepairPlanRefusal expected, const char* what) {
            RepairPlanRefusal reason = RepairPlanRefusal::none;
            check(!repair_plan_acceptable(f, reason) && reason == expected, what);
            check(std::string_view(repair_plan_reason(reason)) != std::string_view("admitted"),
                  "a refusal always names a reason");
        };
        const std::vector<uint64_t> none_positions;
        const std::vector<uint64_t> two{36, 40};
        refuses(RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(none_positions), 0, 4},
                RepairPlanRefusal::empty_plan, "a repair with no frontier admits nothing");
        refuses(RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(two), 0, 4},
                RepairPlanRefusal::too_many_repairs, "a second attachment needs a foreground step, so it is refused");
        refuses(RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(one), 4, 4},
                RepairPlanRefusal::unexpected_foreground, "a foreground step would generate tokens, not re-read the request's");
        refuses(RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(one), 0, 0},
                RepairPlanRefusal::no_page, "an unknown page size is refused, not guessed");
        refuses(RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(one), 0, 3},
                RepairPlanRefusal::unaligned_page, "a page that is not four-token aligned is refused");
        const std::vector<uint64_t> off_page{35};
        refuses(RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(off_page), 0, 4},
                RepairPlanRefusal::repair_unaligned, "a frontier off the page boundary is refused");
        const std::vector<uint64_t> at_boundary{32};
        refuses(RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(at_boundary), 0, 4},
                RepairPlanRefusal::repair_not_past_boundary, "a frontier at the reuse boundary repairs nothing");
        const std::vector<uint64_t> at_donor{128};
        refuses(RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(at_donor), 0, 4},
                RepairPlanRefusal::repair_at_or_past_donor, "a frontier at the donor endpoint is not a repair");
        {   // a four-token reused suffix has no interior frontier at all: the qualified gap leaves no room
            const std::vector<uint64_t> only{36};
            refuses(RepairPlanFacts{24, 32, 32, 36, 37, std::span<const uint64_t>(only), 0, 4},
                    RepairPlanRefusal::repair_at_or_past_donor, "a four-token reused suffix admits no frontier");
            RepairPlanRefusal gap = RepairPlanRefusal::none;
            check(repair_plan_acceptable(RepairPlanFacts{24, 32, 32, 40, 41, std::span<const uint64_t>(only), 0, 4}, gap),
                  "a frontier four tokens before the donor endpoint is the qualified minimum and admits");
            refuses(RepairPlanFacts{24, 32, 32, 34, 35, std::span<const uint64_t>(only), 0, 4},
                    RepairPlanRefusal::donor_unaligned, "a donor endpoint off the page is refused");
        }
        refuses(RepairPlanFacts{31, 32, 31, 128, 129, std::span<const uint64_t>(one), 0, 4},
                RepairPlanRefusal::boundary_unaligned, "a reuse boundary off the page cannot be a certified cut");
        refuses(RepairPlanFacts{24, 32, 128, 128, 129, std::span<const uint64_t>(one), 0, 4},
                RepairPlanRefusal::boundary_at_or_past_donor, "a reuse boundary at the donor endpoint reuses nothing");
        refuses(RepairPlanFacts{24, 32, 32, 130, 129, std::span<const uint64_t>(one), 0, 4},
                RepairPlanRefusal::donor_unaligned, "a donor endpoint off the page cannot be a certified cut");
        refuses(RepairPlanFacts{24, 32, 32, 128, 128, std::span<const uint64_t>(one), 0, 4},
                RepairPlanRefusal::accepted_history,
                "the accepted history must carry the donor's pending token (N == forwarded + 1)");
        refuses(RepairPlanFacts{40, 32, 32, 128, 129, std::span<const uint64_t>(one), 0, 4},
                RepairPlanRefusal::edit_out_of_order, "an empty or reversed changed run is refused");
        refuses(RepairPlanFacts{24, 60, 32, 128, 129, std::span<const uint64_t>(one), 0, 4},
                RepairPlanRefusal::edit_not_before_boundary, "a changed run crossing the reuse boundary is refused");
        refuses(RepairPlanFacts{24, 32, 32, 128, 300, std::span<const uint64_t>(one), 0, 4},
                RepairPlanRefusal::accepted_history, "a truncated history must end at the donor's pending token");
    }

    // 7. Adapter FAILURE PATHS: an inadmissible plan or cut must be refused BEFORE any fence event, and a refused
    //    fence event must leave the session exactly where it was (never a partial mutation, never a poison).
    {
        const std::vector<uint64_t> bogus{35};
        RepairPlanRefusal why = RepairPlanRefusal::none;
        Lifecycle fence;
        fence.apply(Lifecycle::Event::open_request);
        fence.set_observed_phase(Phase::drained);      // a session already at a certified drained cut
        const bool bad_plan = !repair_plan_acceptable(
            RepairPlanFacts{24, 32, 32, 128, 129, std::span<const uint64_t>(bogus), 0, 4}, why);
        check(bad_plan, "the serving path refuses a bad plan before touching the fence");
        check(fence.phase() == Phase::drained && fence.request_open(),
              "a refused plan mutates no phase and closes no request (the fence is untouched)");
        check(!fence.apply(Lifecycle::Event::drain_work),
              "drain_work while already drained is refused without poisoning the session");
        check(fence.phase() == Phase::drained, "a refused drain leaves the drained cut intact");
        check(fence.apply(Lifecycle::Event::begin_work), "the fence may still be re-opened for a legitimate attempt");
        check(fence.phase() == Phase::running, "begin_work entered running");
        check(!fence.apply(Lifecycle::Event::capture) && !fence.apply(Lifecycle::Event::restore),
              "capture/restore refuse mid-run, so no operand can slip past an unclosed fence");
        fence.set_observed_phase(Phase::poisoned);      // what an uncertain completion reports
        check(!fence.allow(Lifecycle::Event::capture) && !fence.allow(Lifecycle::Event::restore) &&
              !fence.allow(Lifecycle::Event::begin_work) && !fence.allow(Lifecycle::Event::close_request),
              "a poisoned fence admits nothing, not even close: destroying it IS the failstop");
        check(terminates_on_destruction(Phase::poisoned) && terminates_on_destruction(Phase::running) &&
              !terminates_on_destruction(Phase::drained) && !terminates_on_destruction(Phase::idle),
              "destruction is fatal exactly over uncertain work, never over a drained certified cut");
    }

    // 8. Donor admission as the request path computes it (LABELLED: admission arithmetic only, no adapter, no
    //    device, no proof that a real request sequence ran - that needs the engine scenario).  The donor cut must
    //    carry the session's genuine pending token: accepted N with forwarded N-1.  A session whose live history
    //    was replaced (parked restore, session file, slot hand-off, relabel, cancelled read) has no certified
    //    pending token, so its history is `forwarded` long and the plan refuses it until a completion re-
    //    establishes the token.
    {
        const std::vector<uint64_t> frontier{36};
        auto plan = [&](uint64_t consumed, bool pending_certified) {
            return RepairPlanFacts{24, 32, 32, consumed,
                                   pending_certified ? consumed + 1 : consumed,
                                   std::span<const uint64_t>(frontier), 0, 4};
        };
        RepairPlanRefusal why = RepairPlanRefusal::none;
        check(repair_plan_acceptable(plan(128, true), why),
              "a donor with a certified pending token admits: accepted N, forwarded N-1");
        why = RepairPlanRefusal::none;
        check(!repair_plan_acceptable(plan(128, false), why) && why == RepairPlanRefusal::accepted_history,
              "a donor with no certified pending token (history == forwarded) is refused, not guessed");
        why = RepairPlanRefusal::none;
        check(repair_plan_acceptable(plan(128, true), why),
              "after the pending token is re-established the same shape admits again");

        // Consecutive repaired requests on ONE fence, driven through the real transition machine the adapter uses.
        Lifecycle seq;
        seq.apply(Lifecycle::Event::open_request);
        seq.set_observed_phase(Phase::idle);
        check(seq.apply(Lifecycle::Event::begin_work) && seq.apply(Lifecycle::Event::drain_work),
              "first request: donor phase, then the fenced prefix phase");
        check(seq.apply(Lifecycle::Event::capture) && seq.apply(Lifecycle::Event::close_request),
              "first request ends at a drained cut with the request CLOSED (else the next one cannot open)");
        check(seq.allow(Lifecycle::Event::open_request) && seq.apply(Lifecycle::Event::open_request),
              "the second request may open the same fence with no lifecycle reset");
        seq.set_observed_phase(Phase::drained);
        check(seq.apply(Lifecycle::Event::begin_work) && seq.apply(Lifecycle::Event::drain_work) &&
              seq.apply(Lifecycle::Event::capture) && seq.apply(Lifecycle::Event::close_request),
              "the second request runs the whole sequence again");
        Lifecycle cancelled;
        cancelled.apply(Lifecycle::Event::open_request);
        cancelled.set_observed_phase(Phase::idle);
        cancelled.apply(Lifecycle::Event::begin_work);
        check(!cancelled.allow(Lifecycle::Event::close_request) && !cancelled.allow(Lifecycle::Event::capture),
              "a read cancelled inside the running fence admits neither close nor capture: drain or failstop");
    }

    // The controller uses this exact policy before its independent real completion/cut/CAS checks. Serving's
    // one attachment must reach final publication without weakening the experiment's two-attachment trace.
    {
        const RepairFinishTrace serving{1, 0, 0, 0, 129, 129, 7, 7};
        check(repair_finish_trace_acceptable(RepairFinishMode::serving_no_generation, serving),
              "one serving attachment with unchanged history and inherited watermark can finish");
        check(!repair_finish_trace_acceptable(RepairFinishMode::two_attachment_generation, serving),
              "the experiment still refuses a one-attachment serving trace");
        auto bad = serving;
        bad.attachments = 0;
        check(!repair_finish_trace_acceptable(RepairFinishMode::serving_no_generation, bad),
              "serving cannot finish before its actual attachment");
        bad = serving; bad.foreground = 1;
        check(!repair_finish_trace_acceptable(RepairFinishMode::serving_no_generation, bad),
              "serving cannot finish after sampling a foreground token");
        bad = serving; bad.receipts = 1;
        check(!repair_finish_trace_acceptable(RepairFinishMode::serving_no_generation, bad),
              "serving cannot finish with an output receipt");
        bad = serving; ++bad.accepted;
        check(!repair_finish_trace_acceptable(RepairFinishMode::serving_no_generation, bad),
              "serving cannot finish after silently extending accepted history");
        bad = serving; ++bad.watermark;
        check(!repair_finish_trace_acceptable(RepairFinishMode::serving_no_generation, bad),
              "serving cannot finish after changing the inherited output watermark");
        RepairFinishTrace scenario{2, 8, 4, 8, 129, 137, 0, 8};
        check(repair_finish_trace_acceptable(RepairFinishMode::two_attachment_generation, scenario),
              "the original experiment's two attachments and later foreground still admit");
        scenario.foreground = scenario.foreground_at_attachment;
        check(!repair_finish_trace_acceptable(RepairFinishMode::two_attachment_generation, scenario),
              "the experiment still refuses catchup without foreground after the second attachment");
        check(repair_attachment_publication_acceptable(2, 0) &&
              repair_attachment_publication_acceptable(3, 1),
              "attachment receipts bind incremented selection serials starting at two");
        check(!repair_attachment_publication_acceptable(1, 0) &&
              !repair_attachment_publication_acceptable(3, 0),
              "a stale or skipped first publication cannot pass serving receipt validation");
        RepairFinishTrace production{2, 12, 4, 8, 129, 141, 7, 15};
        check(repair_finish_trace_acceptable(RepairFinishMode::request_continuation, production),
              "known request forwards and immutable emitted outputs share the final accepted sequence");
        auto invalid=production; ++invalid.watermark;
        check(!repair_finish_trace_acceptable(RepairFinishMode::request_continuation, invalid),
              "known prompt forwarding cannot add output effects");
        invalid=production; --invalid.accepted;
        check(!repair_finish_trace_acceptable(RepairFinishMode::request_continuation, invalid),
              "final history cannot omit an already forwarded prompt or generated token");
        invalid=production; invalid.receipts=13;
        check(!repair_finish_trace_acceptable(RepairFinishMode::request_continuation, invalid),
              "a receipt cannot exist without its real foreground step");
        check(repair_finish_trace_acceptable(RepairFinishMode::request_continuation,
                                            RepairFinishTrace{1,1,0,1,129,130,0,1}),
              "early EOS finishes exact with its one real output, without inventing extra output to attach again");
        check(repair_finish_trace_acceptable(RepairFinishMode::request_continuation,
                                            RepairFinishTrace{1,0,0,0,129,129,0,0}),
              "cancellation at the initial attachment can still settle exact and retire");
        invalid=production; invalid.initial_accepted=UINT64_MAX;
        check(!repair_finish_trace_acceptable(RepairFinishMode::request_continuation, invalid),
              "accepted-history overflow is refused before final publication");
    }

    // ---- slice 7: an ORDINARY request's own certificate, through the predicates the transaction runs -----
    // This is the real capture-metadata predicate and the real tracked-state derivation, not a simulation: the
    // serving path calls the same functions (include/strata/core/serving_raw_metadata.hpp).
    {
        // A request that started from a zeroed session, forwarded 232 rows and was about to consume one more.
        std::vector<int32_t> consumed;
        consumed.reserve(232);
        for (int32_t token = 0; token < 232; ++token) consumed.push_back(token);
        const int32_t pending = 99991;
        const uint64_t capacity = 4096;

        TrackedState certified;
        TrackedStateRefusal why = TrackedStateRefusal::none;
        const auto ok_facts = [&] {
            return TrackedStateFacts{7, std::span<const int32_t>(consumed), pending, true, 0,
                                     ApproximationProvenance{}};
        }();
        why = track_state_certificate(ok_facts, capacity, certified);
        check(why == TrackedStateRefusal::none, "a clean-reset ordinary completion certifies its own cut");
        check(certified.cut.id == 7 && certified.cut.forwarded == consumed.size() &&
              certified.cut.accepted == consumed.size() + 1 &&
              certified.cut.exact_frontier == certified.cut.forwarded,
              "the certified cut names the rows actually forwarded plus the one pending token");
        check(certified.cut.history == window_history_digest(certified.accepted) &&
              certified.accepted.size() == consumed.size() + 1 &&
              certified.accepted.back() == pending,
              "the history digest is derived from the accepted run, which ends at the real pending token");
        check(certified.approximation.empty(), "a clean reset declares no approximation gap");
        const auto exact_descriptor = tracked_metadata(WindowKey{}, certified);
        check(bounded_capture_metadata(exact_descriptor, capacity),
              "the tracked descriptor is inside the transaction's extent bound");
        check(capture_metadata_acceptable(exact_descriptor),
              "the transaction's own capture gate accepts the ordinary completion's descriptor");
        check(certified.valid(), "a certificate with a cut identity and an accepted run is valid");

        // The same state, mis-stated with a caller-supplied digest: the derived digest must be the one that pairs.
        StateCut stale_pair = certified.cut;
        stale_pair.history = real_digest();
        check(!accepted_run_certifies_cut(stale_pair, certified.cut, certified.accepted),
              "an accepted run cannot be paired with a digest it does not produce");

        // What the rejected approach did: keep an earlier conversation's cut and history across an ordinary
        // request that ran from them.  The live cut is the ordinary request's own, so the stale pair refuses.
        TrackedState earlier;
        std::vector<int32_t> earlier_run;
        for (int32_t token = 0; token < 293; ++token) earlier_run.push_back(token + 1000);
        earlier_run.push_back(4242);
        check(track_state_certificate(TrackedStateFacts{5, std::span<const int32_t>(earlier_run), -1, false, 0,
                        ApproximationProvenance{1700000000000ull, window_history_digest(earlier_run)}},
                        capacity, earlier) == TrackedStateRefusal::none,
              "the earlier conversation state certifies with its own honest approximation provenance");
        check(!accepted_run_certifies_cut(earlier.cut, certified.cut, earlier.accepted),
              "an earlier conversation's accepted run and cut cannot certify the state an ordinary request left");
        check(!accepted_run_certifies_cut(earlier.cut, certified.cut, certified.accepted),
              "the ordinary request's accepted run cannot be paired with another conversation's cut");
        check(!accepted_run_certifies_cut(certified.cut, certified.cut, {}),
              "no accepted run at all certifies nothing");

        // The deterministic descriptor fault behind the rejected campaign's capture refusal: a cut whose
        // approximation gap is left undeclared.  Both the import marking (frontier 0) and an intermediate
        // frontier fail the same gate, and neither may be patched by pushing exact_frontier to forwarded.
        for (const uint64_t frontier : {0ull, 44ull}) {
            std::vector<int32_t> gap_run;
            for (int32_t token = 0; token < 293; ++token) gap_run.push_back(token);
            SnapshotDescriptor gap;
            gap.cut = StateCut{9, gap_run.size(), gap_run.size(), frontier, 0,
                               window_history_digest(gap_run)};
            gap.accepted_tokens = gap_run;
            gap.speculation_disabled = true;
            check(!capture_metadata_acceptable(gap),
                  "an approximate cut with undeclared approximation coverage is refused by the capture gate");
            SnapshotDescriptor relabelled = gap;
            relabelled.cut.exact_frontier = relabelled.cut.forwarded;
            check(capture_metadata_acceptable(relabelled),
                  "the gate would accept a relabelled cut, which is why exactness comes only from provenance");
            TrackedState relabel_refused;
            check(track_state_certificate(TrackedStateFacts{9, std::span<const int32_t>(gap_run), -1, false,
                            frontier, ApproximationProvenance{}}, capacity, relabel_refused) ==
                      TrackedStateRefusal::approximation_unanchored,
                  "an inexact state with no anchor is refused instead of being relabelled exact");
        }

        // The honest version of that same state: the gap named, anchored at a real observation, with the digest
        // of the state whose suffix was reused.
        std::vector<int32_t> gap_run;
        for (int32_t token = 0; token < 293; ++token) gap_run.push_back(token);
        TrackedState honest;
        check(track_state_certificate(TrackedStateFacts{11, std::span<const int32_t>(gap_run), -1, false, 44,
                        ApproximationProvenance{1700000000000ull, real_digest()}}, capacity, honest) ==
                  TrackedStateRefusal::none,
              "an inexact state with a real anchor and ancestor certifies");
        check(honest.approximation.size() == 1 && honest.approximation.front().begin == 44 &&
              honest.approximation.front().end == honest.cut.forwarded &&
              honest.approximation.front().first_unresolved_unix_ms == 1700000000000ull,
              "the region covers exactly [exact_frontier, forwarded) and keeps the inherited anchor");
        check(capture_metadata_acceptable(tracked_metadata(WindowKey{}, honest)),
              "the capture gate accepts the honestly described approximate state");
        check(track_state_certificate(TrackedStateFacts{12, std::span<const int32_t>(gap_run), -1, false, 44,
                        ApproximationProvenance{0, real_digest()}}, capacity, honest) ==
                  TrackedStateRefusal::approximation_unanchored,
              "a gap with no age anchor is refused");
        check(track_state_certificate(TrackedStateFacts{12, std::span<const int32_t>(gap_run), -1, false, 44,
                        ApproximationProvenance{1700000000000ull, {}}}, capacity, honest) ==
                  TrackedStateRefusal::approximation_ancestor_missing,
              "a gap that names no ancestor state is refused");
        check(track_state_certificate(TrackedStateFacts{12, std::span<const int32_t>(gap_run), -1, false, 293,
                        ApproximationProvenance{1700000000000ull, real_digest()}}, capacity, honest) ==
                  TrackedStateRefusal::unresolvable_approximation,
              "an inexact state cannot be reported with a frontier already at the current position");
        check(track_state_certificate(TrackedStateFacts{0, std::span<const int32_t>(consumed), pending, true, 0,
                        ApproximationProvenance{}}, capacity, honest) ==
                  TrackedStateRefusal::no_cut_id, "a completion with no cut identity certifies nothing");
        check(track_state_certificate(TrackedStateFacts{13, std::span<const int32_t>{}, pending, true, 0,
                        ApproximationProvenance{}}, capacity, honest) ==
                  TrackedStateRefusal::empty_history, "an empty consumed run certifies nothing");
        {
            std::vector<int32_t> poisoned_run = consumed;
            poisoned_run.back() = -1;
            check(track_state_certificate(TrackedStateFacts{13, std::span<const int32_t>(poisoned_run), pending,
                            true, 0, ApproximationProvenance{}}, capacity, honest) ==
                      TrackedStateRefusal::negative_token,
                  "a consumed run carrying a negative token is refused");
        }
        check(track_state_certificate(TrackedStateFacts{13, std::span<const int32_t>(consumed), pending, true, 0,
                        ApproximationProvenance{}}, consumed.size(), honest) ==
                  TrackedStateRefusal::history_past_capacity,
              "the accepted run past the session capacity is refused");
        check(track_state_certificate(TrackedStateFacts{14, std::span<const int32_t>(consumed), -1, true, 0,
                        ApproximationProvenance{}}, capacity, honest) == TrackedStateRefusal::none &&
              honest.cut.accepted == honest.cut.forwarded && honest.cut.forwarded == consumed.size(),
              "a completion with no certified pending token stays one row behind its own history");

        // ---- cancellation: the two clean answers, and nothing in between -------------------------------
        // (a) a request cancelled at a whole-window boundary still names its own run: the read finished whole and
        // the last commit joined, so the consumed run and its pending token describe the device.
        TrackedState cancelled_whole_row;
        check(track_state_certificate(TrackedStateFacts{21, std::span<const int32_t>(consumed), pending, true, 0,
                        ApproximationProvenance{}}, capacity, cancelled_whole_row) ==
                  TrackedStateRefusal::none,
              "a cancel at a committed window boundary certifies the rows it actually forwarded");
        check(cancelled_whole_row.cut.forwarded == consumed.size() && accepted_run_certifies_cut(
                  cancelled_whole_row.cut, cancelled_whole_row.cut, cancelled_whole_row.accepted),
              "that certificate pairs its accepted run with its own cut");
        // (b) a session deliberately returned to zero: the truthful certificate is that it stands on NO run, and
        // the activation guard refuses it.  An empty run is never a way to certify an unknown history.
        TrackedState zeroed;
        check(track_state_certificate(TrackedStateFacts{22, std::span<const int32_t>{}, -1, true, 0,
                        ApproximationProvenance{}, true}, capacity, zeroed) == TrackedStateRefusal::none &&
              zeroed.cut.forwarded == 0 && zeroed.cut.accepted == 0 && zeroed.accepted.empty() &&
              zeroed.cut.history == window_history_digest(std::vector<int32_t>{}),
              "a reset session certifies the empty run it really stands on, with the empty run's own digest");
        check(!accepted_run_certifies_cut(zeroed.cut, zeroed.cut, zeroed.accepted) && !zeroed.valid(),
              "an empty certificate names no accepted run, so no retained activation may capture over it");
        check(track_state_certificate(TrackedStateFacts{22, std::span<const int32_t>{}, -1, false, 0,
                        ApproximationProvenance{1700000000000ull, real_digest()}, true}, capacity, zeroed) ==
                  TrackedStateRefusal::empty_needs_exactness,
              "only an exactly reset session may certify emptiness: an inherited gap cannot be dropped");
        check(track_state_certificate(TrackedStateFacts{23, std::span<const int32_t>{}, -1, true, 0,
                        ApproximationProvenance{}}, capacity, zeroed) == TrackedStateRefusal::empty_history,
              "an empty run without the deliberate-reset fact is not a certificate at all");

        // The lifecycle both of these depend on, through the adapter's own state machine.
        Lifecycle cancelled_row;
        cancelled_row.apply(Lifecycle::Event::open_request);
        cancelled_row.set_observed_phase(Phase::drained);
        cancelled_row.apply(Lifecycle::Event::begin_work);
        cancelled_row.set_observed_phase(Phase::running);
        cancelled_row.apply(Lifecycle::Event::drain_work);
        cancelled_row.set_observed_phase(Phase::drained);
        check(cancelled_row.allow(Lifecycle::Event::close_request) &&
              cancelled_row.apply(Lifecycle::Event::close_request),
              "a cancel that forwarded whole rows drains, then closes its own fence");
        Lifecycle gave_way;
        gave_way.apply(Lifecycle::Event::open_request);
        gave_way.set_observed_phase(Phase::drained);
        gave_way.apply(Lifecycle::Event::begin_work);
        gave_way.set_observed_phase(Phase::running);
        check(!gave_way.allow(Lifecycle::Event::close_request) && !gave_way.allow(Lifecycle::Event::capture) &&
              !gave_way.allow(Lifecycle::Event::restore),
              "a read that gave way may not close, capture or restore over its own running work phase");
        // The clean answer: put the session back on a describable position, then drain THAT and close.  A drain is
        // only ever asked for after a real completion, so the machine admits it exactly once per work phase.
        check(gave_way.allow(Lifecycle::Event::drain_work), "the reset position is a drain the machine admits");
        gave_way.apply(Lifecycle::Event::drain_work);
        gave_way.set_observed_phase(Phase::drained);
        check(gave_way.allow(Lifecycle::Event::capture),
              "the drained reset state is capturable, but its certificate names no accepted run to capture with");
        check(gave_way.apply(Lifecycle::Event::close_request),
              "and the request fence closes on that drained position, so serving continues");
        // And the failure mode that must stay terminal: a refused drain leaves the work phase running, where the
        // only legal move is abandonment, and abandonment is durable across the adapter's phase re-sync.
        Lifecycle poisoned;
        poisoned.apply(Lifecycle::Event::open_request);
        poisoned.set_observed_phase(Phase::drained);
        poisoned.apply(Lifecycle::Event::begin_work);
        poisoned.set_observed_phase(Phase::running);
        poisoned.set_observed_phase(Phase::running);   // the drain was refused: nothing changed
        check(!poisoned.allow(Lifecycle::Event::close_request) && poisoned.allow(Lifecycle::Event::abandon),
              "a refused drain leaves abandonment as the only legal move");
        poisoned.apply(Lifecycle::Event::abandon);
        poisoned.set_observed_phase(Phase::idle);      // the adapter re-syncs the phase on every public call
        check(poisoned.terminal() && poisoned.phase() == Phase::poisoned &&
              !poisoned.allow(Lifecycle::Event::open_request) && !poisoned.allow(Lifecycle::Event::capture),
              "abandonment is durable: re-reading the phase cannot bring the fence back");
        check(terminates_on_destruction(poisoned.phase()),
              "and destroying that owner is the failstop, never a cache reset");

        // The lifecycle the ORDINARY fence must actually run: the fence must be opened and begun before the
        // request's first reset, the capture must wait for the real drain, and the request may not be closed over
        // a running work phase.  Same state machine the adapter routes every public call through.
        Lifecycle ordinary;
        check(ordinary.allow(Lifecycle::Event::open_request), "an idle session admits a request fence");
        ordinary.apply(Lifecycle::Event::open_request);
        ordinary.set_observed_phase(Phase::idle);
        check(ordinary.allow(Lifecycle::Event::begin_work), "an open request admits a work phase before mutation");
        check(!ordinary.allow(Lifecycle::Event::capture), "nothing may be captured before the work even begins");
        ordinary.apply(Lifecycle::Event::begin_work);
        ordinary.set_observed_phase(Phase::running);
        check(!ordinary.allow(Lifecycle::Event::close_request),
              "the request fence cannot be closed over the running work an ordinary request opened");
        check(!ordinary.allow(Lifecycle::Event::capture) && !ordinary.allow(Lifecycle::Event::restore),
              "no capture or restore is legal while the request's own work is running");
        check(ordinary.allow(Lifecycle::Event::drain_work) &&
              cut_facts_acceptable(CutFacts{certified.cut.id, certified.cut.accepted, certified.cut.forwarded,
                                            certified.cut.exact_frontier, &certified.cut.history},
                                   2, capacity),
              "the certified completion is the cut the drain admits");
        ordinary.apply(Lifecycle::Event::drain_work);
        ordinary.set_observed_phase(Phase::drained);
        check(ordinary.allow(Lifecycle::Event::capture) && ordinary.allow(Lifecycle::Event::restore),
              "after the real all-stage drain the tracked state may be captured for a rollback image");
        check(ordinary.apply(Lifecycle::Event::close_request), "the ordinary request closes on a drained phase");
        check(!ordinary.request_open() && ordinary.allow(Lifecycle::Event::open_request),
              "the closed fence leaves the session able to admit the returning conversation's activation");
        // And the failure this ordering prevents: a request that never drained keeps the observed phase at the
        // work it left running, where capture is refused and destroying the owner is the failstop.
        Lifecycle abandoned;
        abandoned.apply(Lifecycle::Event::open_request);
        abandoned.apply(Lifecycle::Event::begin_work);
        abandoned.set_observed_phase(Phase::running);
        check(!abandoned.allow(Lifecycle::Event::capture),
              "an undrained ordinary work phase can never be read as a capturable state");
        check(terminates_on_destruction(Phase::running),
              "unwinding that phase is the contract's failstop, not a cache reset");
    }

    if (failures) { std::printf("serving RAW lifecycle contract gate FAILED: %d checks\n", failures); return 1; }
    std::printf("serving RAW lifecycle contract gate PASS (phase/cut/destruction rules; no device touched)\n");
    return 0;
}
