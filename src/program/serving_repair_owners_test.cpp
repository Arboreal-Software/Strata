// CPU-only tests for the retained repair owner machinery in serving_repair_owners.hpp (slice 2: retained
// repair across a response boundary).  No GPU, no CUDA, no engine call, no download.
//
// WHAT IS REAL CODE HERE (no stand-in for any decision):
//  * `core::serving_raw::Lifecycle` (include/strata/core/serving_raw_contracts.hpp) including the NEW private
//    work window events - the same state machine ServingRawSession routes every public call through;
//  * the registry's charge pinning: `OwnerReservationHandle` holds the same type-erased `shared_ptr<const void>`
//    shape serving uses for an admitted credit, and `FixtureCharge` below plays the role of the ledger lease so
//    the cell can prove the ENTRY pins it and releases it exactly once.  (The real `core::WindowBudget`/
//    `ReservationLease` symbols live in src/core/window_budget.cpp and cannot be linked into this single-TU cell;
//    the ledger itself has its own registered tests.)
//  * `make_repair_owner_evidence` and every shared gate (`retained_yield_allowed`,
//    `retained_private_unit_allowed`, `retained_park_allowed`, `parked_owner_publishable`, `owner_scope_escape`) -
//    the exact functions `ResidentRepairController::yield_retained` / `retained_private_unit` and the serving
//    scope call.  `FixtureController::evidence()` below builds `RepairOwnerFacts` and routes them through that
//    same production maker: no test states a controller fact directly to the registry;
//  * slice 3's two serving-scope predicates, `serving_certified_yield_gate` (the D1 response-completion gate
//    generate.cpp builds its facts for) and `serving_input_demand_observed` (the D2 idle window's wake condition
//    over the stdin queue) - the same functions the server calls, not a restatement of them;
//  * `ServingRepairOwnerRegistry` itself, the accepted kernel's demand/admission logic
//    (`evaluate_serving_repair_admission`, serving_repair_admission.hpp, unmodified), `RetainedRepairAge`, and
//    `core::WindowKey`/`core::StateCut` built from their own definitions.
//
// WHAT IS SIMULATED, AND THEREFORE NOT PROVEN BY THIS CELL: the private all-stage DEVICE advance.
// `FixtureController` mirrors the controller's state fields and moves `private_cut` the way the real device loop
// would; it has no device.  The mapping from the REAL controller's members into `RepairOwnerFacts`
// (`ResidentRepairController::owner_evidence()`, resident_repair_controller.inc) and the real retirement of a GPU
// gang cannot run here - they need the frozen CUDA build.  Nor is the real serving token loop reached: `src/
// program/generate.cpp` cannot be compiled on this host (`cuda_runtime.h` is absent), so the D1 cell drives the
// production gate with the facts the token loop leaves behind, and the D2 cell drives the production wake
// condition and the per-unit admission the serving thread performs before every unit - not the thread itself.
// Those cells are listed as still-needed in SLICE2-WORKER-RESULT.md and SLICE3-WORKER-RESULT.md.
//
// Every number below is an explicit test fixture (slot bounds, serials, cuts, caps, ages) chosen to discriminate
// one gate.  None of them is an adopted threshold, slot count, unit bound or policy value.
#include "serving_repair_owners.hpp"

#include "strata/core/serving_raw_contracts.hpp"
#include "strata/core/window_budget.hpp"   // ReservationPurpose only; the ledger itself is not linkable here
#include "strata/core/window_state.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strata::program;
namespace core = strata::core;

static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); std::exit(1); }
}

namespace {

constexpr std::size_t kOneSlot = 1, kTwoSlots = 2;
constexpr uint64_t kFixtureSerial = 3;                     // a fixture published serial
constexpr uint64_t kFixtureCap = 4;                        // a fixture idle unit cap, in tokens
constexpr uint64_t kFixtureAgeNs = 7ULL * 1000000000ULL;   // a fixture observed age, not a threshold
constexpr uint64_t kPublishedForwarded = 40;               // the fixture conversation's endpoint
constexpr uint64_t kPrivateForwarded = 28;                 // the fixture private checkpoint (debt 12)

core::StateDigest fixture_digest(uint8_t seed) {
    core::StateDigest digest{};
    digest[0] = seed == 0 ? 0x5a : seed;
    return digest;
}

core::StateCut fixture_cut(uint64_t id, uint64_t forwarded, uint64_t exact, uint8_t seed) {
    core::StateCut cut;
    cut.id = id;
    cut.accepted = forwarded + 1;
    cut.forwarded = forwarded;
    cut.exact_frontier = exact;
    cut.output_watermark = 1;
    cut.history = fixture_digest(seed);
    return cut;
}

// A private-CHAIN cut. The controller's private transaction completes with accepted == forwarded ==
// exact_frontier at every all-stage private unit (advance_private_to, resident_repair_controller.inc); a
// PUBLICATION instead carries its one pending token (accepted == forwarded + 1). Keeping the two shapes apart is
// what lets the prepared-result maker refuse a result that is not really a chain cut.
core::StateCut fixture_chain_cut(uint64_t id, uint64_t forwarded, uint8_t seed) {
    core::StateCut cut = fixture_cut(id, forwarded, forwarded, seed);
    cut.accepted = forwarded;
    return cut;
}

core::WindowKey fixture_key(uint64_t append_epoch = 5, uint64_t selection_epoch = 2,
                            const char* window_id = "window-a") {
    core::WindowKey key;
    key.engine_generation = 1;
    key.window_generation = 4;
    key.session_id = "session-a";
    key.thread_id = "thread-a";
    key.window_id = window_id;
    key.revision = 9;
    key.append_epoch = append_epoch;
    key.selection_epoch = selection_epoch;
    key.provenance_epoch = 3;
    key.model_profile = fixture_digest(0x11);
    key.topology = fixture_digest(0x22);
    return key;
}

// Stand-in for the ledger charge the controller pins.  The real controller fills `private_charge` and
// `private_charge_purpose` from its private gang's `NativeSessionAllocation::reservation()` - a genuine
// `ReservationLease` that `allocate_native_sessions` took with `ReservationPurpose::private_work`
// (src/core/window_allocator.cpp:317) - and reads the purpose off the lease itself.  This object plays that role
// (handle + purpose carried by the charge, never by a caller) so the cell can prove the registry entry pins the
// charge and releases it exactly once.
struct FixtureCharge {
    core::ReservationPurpose purpose = core::ReservationPurpose::private_work;
    inline static int live = 0;
    FixtureCharge() { ++live; }
    ~FixtureCharge() { --live; }
};

// Stand-ins for the three ledger charges an activation needs (slice 5): the frozen private image, the capture of
// the live session's prior conversation state, and the activation's restore. Counted separately from
// `FixtureCharge` (the private gang's own lease) so every existing leak assertion keeps its exact meaning.
struct FixtureResourceCharge {
    core::ReservationPurpose purpose = core::ReservationPurpose::private_attachment;
    inline static int live = 0;
    FixtureResourceCharge() { ++live; }
    ~FixtureResourceCharge() { --live; }
};

// The three charges a park must pin, with their purposes READ BACK from the charge (the same discipline as the
// gang lease), and fixture quotes. A campaign names the real numbers; these are test fixtures, not adopted caps.
RetainedActivationResources fixture_resources() {
    RetainedActivationResources resources;
    auto image = std::make_shared<FixtureResourceCharge>();
    image->purpose = core::ReservationPurpose::private_attachment;
    auto capture = std::make_shared<FixtureResourceCharge>();
    capture->purpose = core::ReservationPurpose::foreground_return;
    auto restore = std::make_shared<FixtureResourceCharge>();
    restore->purpose = core::ReservationPurpose::private_attachment;
    resources.image_charge = image;
    resources.image_purpose = image->purpose;
    resources.prior_capture_charge = capture;
    resources.prior_capture_purpose = capture->purpose;
    resources.restore_charge = restore;
    resources.restore_purpose = restore->purpose;
    resources.image_bytes = 4096;
    resources.prior_capture_bytes = 4096;
    resources.restore_bytes = 4096;
    return resources;
}

// ---- the controller fixture ----------------------------------------------------------------------
// Only the device work is simulated.  Every transition below is gated by the SAME production predicate the real
// controller calls, and its evidence comes from the SAME production maker, so a gate cannot be bypassed by the
// fixture any more than it can be bypassed in the server.
class FixtureController final {
public:
    // Mirrors the controller state fields the evidence reads (resident_repair_controller.inc subset).
    uint64_t published_serial = kFixtureSerial;
    core::WindowKey published_key = fixture_key();
    core::StateCut published_cut = fixture_cut(11, kPublishedForwarded, 24, 0x31);
    core::StateCut live_cut = published_cut;
    core::StateCut private_cut = fixture_chain_cut(12, kPrivateForwarded, 0x32);
    core::serving_raw::Phase live_phase = core::serving_raw::Phase::drained;
    core::serving_raw::Phase private_phase = core::serving_raw::Phase::drained;
    bool busy = false, retired = false, cancel_requested = false, finished = false, suspended = false;
    bool live_healthy = true, private_healthy = true, service_idle = true, controller_thread = true;
    uint64_t exact_completion_cut = private_cut.id;
    bool exact_completion_committed = true;
    std::shared_ptr<const FixtureCharge> charge;
    uint64_t next_cut_id = 100;
    uint64_t overshoot = 0;   // non-zero only so a test can prove the registry REFUSES an overrunning unit
    bool move_live = false;   // ditto, for a unit that republishes CURRENT

    static int unit_calls;
    static int drops;
    static int unretired_drops;

    // The fixture's analogue of `ResidentRepairController::owner_evidence()`: facts about itself, through the
    // production maker (which refuses a self-contradictory snapshot).
    RepairOwnerEvidence evidence() const {
        RepairOwnerFacts facts;
        facts.live_phase = live_phase;
        facts.private_phase = private_phase;
        facts.busy = busy; facts.retired = retired; facts.cancel_requested = cancel_requested;
        facts.finished = finished; facts.suspended = suspended;
        facts.live_allocation_healthy = live_healthy;
        facts.private_allocation_healthy = private_healthy && !retired;
        facts.service_idle = service_idle;
        facts.controller_thread = controller_thread;
        facts.published_serial = published_serial;
        facts.published_key = published_key;
        facts.published_cut = published_cut;
        facts.live_cut = live_cut;
        facts.private_cut = retired ? core::StateCut{} : private_cut;
        facts.exact_completion_cut = retired ? 0 : exact_completion_cut;
        facts.exact_completion_committed = !retired && exact_completion_committed;
        facts.private_charge = retired ? std::shared_ptr<const void>{} : std::shared_ptr<const void>(charge);
        facts.private_charge_purpose = retired ? core::ReservationPurpose::foreground : charge->purpose;
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        return make_repair_owner_evidence(facts, why);
    }

    // The analogue of `ResidentRepairController::yield_retained`: same gate, same one-way state change; the
    // only simulated part is that no device capture happens.
    bool yield_retained(core::StateCut& certified, std::string& error) {
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        core::StateCut gate{};
        if (!retained_yield_allowed(evidence(), why, gate)) {
            error = repair_owner_refusal_name(why);
            return false;
        }
        certified = gate;
        suspended = true;
        return true;
    }

    // The analogue of `ResidentRepairController::retained_private_unit`: same gate, then the private cut moves
    // the way the real all-stage loop moves it (new cut id, new committed witness).
    bool retained_private_unit(uint64_t cap, uint64_t& advanced, RepairOwnerEvidence& after, std::string& error) {
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        uint64_t target = 0;
        if (!retained_private_unit_allowed(evidence(), cap, target, why)) {
            error = repair_owner_refusal_name(why);
            return false;
        }
        ++unit_calls;
        const uint64_t before = private_cut.forwarded;
        const uint64_t reached = target + overshoot;
        private_cut = fixture_chain_cut(++next_cut_id, reached, uint8_t(0x40 + reached));
        // Exact private work certifies only consumed rows, including the unit that closes forwarded debt.
        // The genuine pending token remains in the separate publication, as actual advance_private_to produces.

        exact_completion_cut = private_cut.id;
        if (move_live) ++published_serial;
        advanced = reached - before;
        after = evidence();
        return true;
    }

    uint64_t residency_generation = 0;   // the expert-residency inventory this chain is bound to
    bool activation_denied = false;      // non-zero only to prove a refused activation restores the prior state

    // (a) The analogue of `ResidentRepairController::prepare_retained_result`: the controller states its OWN
    // certified private cut, the committed all-stage witness that covers it, and the real versioned fields
    // (accepted sequence, pending token, output watermark, content/append/selection/engine versions), routed
    // through the production maker. `record_version` is the owner record it prepares against.
    PreparedRetainedResult prepare_retained_result(uint64_t record_version, std::string& error) const {
        PreparedRetainedResultFacts facts;
        facts.identity = published_key;
        facts.published_serial = published_serial;
        facts.record_version = record_version;
        facts.published_cut = published_cut;
        facts.certified_cut = private_cut;
        facts.accepted_sequence = private_cut.accepted;
        facts.pending_token = int32_t(published_cut.forwarded + 1);   // publication fixture token, not private accepted history
        facts.output_watermark = published_cut.output_watermark;
        facts.witness_committed = exact_completion_committed && exact_completion_cut == private_cut.id;
        facts.holds_frozen_image = charge != nullptr;
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        PreparedRetainedResult result = make_prepared_retained_result(facts, why);
        if (!result.produced()) error = repair_owner_refusal_name(why);
        return result;
    }

    // (c) The analogue of `ResidentRepairController::activate_retained`: capture the live session's PRIOR
    // conversation state, restore this owner's certified image so the live cut becomes the owner's publication,
    // and only then let the untouched `resume_retained` coherence conjunction clear the suspended state. On a
    // refusal the prior state is restored and reported back field by field, and the owner stays suspended, parked
    // and resumable.
    bool activate_retained(const ConversationStateFacts& prior, ConversationStateFacts& restored_out,
                           RepairOwnerEvidence& after, std::string& error) {
        if (activation_denied) {
            live_cut = prior.cut;        // the capture/restore primitive puts the session back exactly
            restored_out = prior;
            after = evidence();
            error = "the controller refused the activation: no supported boundary";
            return false;
        }
        // The serving scope has already brought the live transaction onto THIS owner's own publication cut - the
        // same precondition the real `activate_retained` insists on (resident_repair_controller.inc: the live cut
        // must equal the publication before anything mutates). Activation does not republish: it makes the
        // published == live conjunction genuinely true again, leaves the publication and its serial exactly as the
        // owner record carries them, and the caught-up private chain closes the debt at that position.
        live_cut = published_cut;
        suspended = false;
        restored_out = prior;            // B's state is retained by the entry's foreground_return charge
        after = evidence();
        return true;
    }

    // The analogue of `ResidentRepairController::retire` (resident_repair_controller.inc:1221-1228): it drops the
    // private gang and its ledger charge, as `retire_initialized` frees the native sessions.
    bool retire(std::string& error) {
        if (retired) return true;
        if (private_phase != core::serving_raw::Phase::drained ||
            live_phase != core::serving_raw::Phase::drained) {
            error = "resident RAW work alive at retirement";
            return false;
        }
        retired = true;
        suspended = false;
        private_phase = core::serving_raw::Phase::absent;
        charge.reset();
        return true;
    }

    ~FixtureController() {
        ++drops;
        if (!retired) ++unretired_drops;
    }
};
int FixtureController::unit_calls = 0;
int FixtureController::drops = 0;
int FixtureController::unretired_drops = 0;

// Shape mirror of the registry-visible subset of `ServingSuffixRepairOperation::Outcome` (generate.cpp:3192-3230)
// plus slice 2's retention fields, spelled exactly as the real members so `RetainedRepairOwner` pins the real
// names.  Non-copyable and non-movable like the real Outcome: the registry owns it through unique_ptr, so a
// payload can never have two owners.
struct FixtureOutcome {
    FixtureOutcome() = default;
    FixtureOutcome(const FixtureOutcome&) = delete;
    FixtureOutcome(FixtureOutcome&&) = delete;
    FixtureOutcome& operator=(const FixtureOutcome&) = delete;
    FixtureOutcome& operator=(FixtureOutcome&&) = delete;
    ~FixtureOutcome() {
        ++dropped;
        if (controller && !controller->retired) ++dropped_holding_live_controller;
    }
    bool populated() const noexcept {
        return bool(operation_credit) || bool(splice_credit) || bool(controller) || next_cut != 0 || retired ||
               yielded || retained_debt != 0 || certified_cut.id != 0;
    }
    // The concept requirement: the payload's own route to its controller's evidence.  The registry has no other.
    RepairOwnerEvidence owner_evidence_snapshot() const {
        return controller ? controller->evidence() : RepairOwnerEvidence{};
    }
    std::shared_ptr<const void> operation_credit, splice_credit;
    core::SnapshotDescriptor published{};
    std::shared_ptr<const void> retained_exact;
    core::StateCut certified_cut{};
    uint64_t retained_debt = 0, retained_residency_swaps = 0;
    std::shared_ptr<FixtureController> controller;
    uint64_t next_cut = 0;
    bool retired = false, yielded = false;
    inline static int dropped = 0;
    inline static int dropped_holding_live_controller = 0;
};

static_assert(RetainedRepairOwner<FixtureOutcome>, "the fixture must satisfy the real owner requirement");
using FixtureRegistry = ServingRepairOwnerRegistry<FixtureOutcome>;

// The retire attempt the registry requires: the real controller path's shape, i.e. finish()'s bookkeeping
// (generate.cpp:3556-3557) - retire first, then drop the controller pointer.
struct FixtureRetire {
    bool operator()(FixtureOutcome& out, std::string& error) const {
        if (!out.controller) { error = "no controller to retire"; return false; }
        if (out.retired) { error = "the parked owner already reported retirement"; return false; }
        out.retired = out.controller->retire(error);
        if (out.retired) out.controller.reset();
        return out.retired;
    }
};

struct FixtureOwner {
    std::unique_ptr<FixtureOutcome> owner;
    std::shared_ptr<FixtureController> controller;   // the test's handle, for deliberate post-park corruption
};

// A coherent, unretired, debt-carrying owner in exactly the state a certified yield produces.
FixtureOwner make_yielded_owner(core::WindowKey key = fixture_key(),
                                uint64_t published_forwarded = kPublishedForwarded,
                                uint64_t private_forwarded = kPrivateForwarded) {
    FixtureOwner made;
    made.controller = std::make_shared<FixtureController>();
    made.controller->charge = std::make_shared<FixtureCharge>();
    check(FixtureCharge::live > 0, "the fixture private gang holds a charge");
    made.controller->published_key = key;
    made.controller->published_cut = fixture_cut(11, published_forwarded, published_forwarded - 16, 0x31);
    made.controller->live_cut = made.controller->published_cut;
    made.controller->private_cut = fixture_chain_cut(12, private_forwarded, 0x32);
    made.controller->exact_completion_cut = made.controller->private_cut.id;
    made.owner = std::make_unique<FixtureOutcome>();
    made.owner->operation_credit = std::make_shared<int>(1);
    made.owner->controller = made.controller;
    made.owner->published.key = key;
    made.owner->published.cut = made.controller->published_cut;
    made.owner->next_cut = 13;
    std::string error;
    core::StateCut certified{};
    check(made.controller->yield_retained(certified, error), "a coherent debt owner can certify a yield");
    made.owner->yielded = true;
    made.owner->certified_cut = certified;
    made.owner->retained_debt = repair_token_debt(made.controller->published_cut.forwarded, certified.forwarded);
    return made;
}

// Retire an owner that was never parked, so every cell leaves no live controller and no ledger charge behind.
void retire_release(FixtureOwner& left) {
    std::string error;
    check(FixtureRetire{}(*left.owner, error), "un-parked fixture owners are retired deliberately");
    left.owner.reset();
    left.controller.reset();
}

// The serving scope's up-front admission, then the park facts. The ticket is issued for the window the OWNER
// publishes (the parked identity), which is what `park_retained_owner` requires; the session key stays a separate
// fact so a cell can still prove a foreign live session is refused.
ParkOwnerFacts park_facts(FixtureRegistry& registry, const FixtureController& controller,
                          const core::WindowKey& session_key,
                          core::serving_raw::Phase live_phase = core::serving_raw::Phase::drained,
                          bool request_open = false) {
    ParkOwnerFacts facts;
    facts.owner = controller.evidence();
    facts.session.session_key = session_key;
    facts.session.session_cut = controller.live_cut;
    facts.session.live_phase = live_phase;
    facts.session.request_open = request_open;
    facts.session.private_window_open = false;
    facts.resources = fixture_resources();
    std::string reserve_error;
    registry.reserve_retained_capacity(controller.published_key, facts.resources, facts.capacity, reserve_error);
    if (!facts.capacity.issued()) facts.resources = RetainedActivationResources{};
    return facts;
}

LiveSessionFacts idle_session(const FixtureController& controller, bool window_open = true,
                              const core::WindowKey& session_key = fixture_key(5, 2)) {
    LiveSessionFacts session;
    session.session_key = session_key;
    session.session_cut = controller.live_cut;
    session.live_phase = core::serving_raw::Phase::drained;
    session.request_open = false;
    session.private_window_open = window_open;
    return session;
}

bool run_unit(FixtureRegistry& registry, const core::WindowKey& identity, const LiveSessionFacts& session,
              uint64_t cap, RepairOwnerEvidence& after, uint64_t& advanced, std::string& error) {
    return registry.advance_parked_owner(
        identity, session, cap,
        [&](FixtureOutcome& payload, uint64_t& tokens, RepairOwnerEvidence& after_evidence,
            std::string& unit_error) {
            return payload.controller->retained_private_unit(cap, tokens, after_evidence, unit_error);
        }, after, advanced, error);
}

// ---- 1. the lifecycle state machine's new private work window (blocker 1 / requirement 5) ----------
void test_private_window_needs_a_closed_quiescent_fence() {
    core::serving_raw::Lifecycle seq;
    using Event = core::serving_raw::Lifecycle::Event;
    check(!seq.allow(Event::begin_private_window),
          "no private window while the fence is absent (nothing has ever been opened)");
    check(seq.apply(Event::open_request), "open_request");
    seq.set_observed_phase(core::serving_raw::Phase::idle);   // what a freshly initialized transaction reports
    check(!seq.allow(Event::begin_private_window), "no private window while a request holds the fence");
    check(seq.apply(Event::begin_work) && seq.phase() == core::serving_raw::Phase::running, "begin_work");
    check(!seq.allow(Event::begin_private_window), "no private window over a running live fence");
    check(seq.apply(Event::drain_work) && seq.phase() == core::serving_raw::Phase::drained, "drain_work");
    check(!seq.allow(Event::begin_private_window), "no private window while the request is still open");
    check(seq.apply(Event::close_request) && !seq.request_open(), "close_request");
    check(seq.private_window_allowed(), "the closed fence at the drained cut is the idle window");
    check(seq.apply(Event::begin_private_window) && seq.private_window_open(), "begin_private_window");
    // While the window is open every live operation is refused: a second conversation cannot start, and no live
    // RAW operand can run under a parked owner's private work.
    check(!seq.allow(Event::open_request), "no new request inside a private work window");
    check(!seq.allow(Event::begin_work) && !seq.allow(Event::drain_work), "no live work phase inside a window");
    check(!seq.allow(Event::capture) && !seq.allow(Event::restore), "no live capture/restore inside a window");
    check(!seq.allow(Event::close_request) && !seq.allow(Event::abandon), "no request event inside a window");
    check(!seq.allow(Event::begin_private_window), "a window may not be opened twice");
    check(seq.phase() == core::serving_raw::Phase::drained,
          "a private window never changes the live work phase: it is not an open live fence");
    check(seq.apply(Event::end_private_window) && !seq.private_window_open(), "end_private_window");
    check(seq.allow(Event::open_request), "the fence is serviceable again once the window closes");
}

void test_private_window_refusals_are_stable() {
    using Event = core::serving_raw::Lifecycle::Event;
    core::serving_raw::Lifecycle poisoned;
    poisoned.apply(Event::open_request);
    poisoned.set_observed_phase(core::serving_raw::Phase::idle);
    poisoned.apply(Event::begin_work);
    poisoned.apply(Event::abandon);
    check(poisoned.terminal(), "abandon is terminal");
    check(!poisoned.allow(Event::begin_private_window), "no private window over a poisoned fence");
    core::serving_raw::Lifecycle running;
    running.apply(Event::open_request);
    running.set_observed_phase(core::serving_raw::Phase::idle);
    running.apply(Event::begin_work);
    check(!running.allow(Event::begin_private_window), "no private window over a running fence");
    // A poison observed while a window is open must not latch the session shut: the window stays closable.
    core::serving_raw::Lifecycle windowed;
    windowed.apply(Event::open_request);
    windowed.set_observed_phase(core::serving_raw::Phase::idle);
    windowed.apply(Event::begin_work);
    windowed.apply(Event::drain_work);
    windowed.apply(Event::close_request);
    check(windowed.apply(Event::begin_private_window), "window opened at the drained cut");
    windowed.set_observed_phase(core::serving_raw::Phase::poisoned);
    check(windowed.terminal() && windowed.private_window_open(), "the poison is durable, the window stays open");
    check(!windowed.allow(Event::begin_private_window), "no further window over a poisoned fence");
    check(windowed.apply(Event::end_private_window), "the window is still closable after a poison");
    // A closed fence at the idle phase (never worked) is quiescent too: the contract admits it explicitly.
    core::serving_raw::Lifecycle idle;
    idle.apply(Event::open_request);
    idle.apply(Event::close_request);
    check(!idle.allow(Event::begin_private_window),
          "an absent (never initialized) observed phase is not a quiescent cut: no window");
    idle.set_observed_phase(core::serving_raw::Phase::idle);
    check(idle.apply(Event::begin_private_window), "a closed fence at the idle phase admits a private window");
}

// ---- 2. the evidence maker: a caller cannot invent controller facts -------------------------------
RepairOwnerFacts sane_facts() {
    RepairOwnerFacts facts;
    facts.live_phase = core::serving_raw::Phase::drained;
    facts.private_phase = core::serving_raw::Phase::drained;
    facts.published_serial = kFixtureSerial;
    facts.published_key = fixture_key();
    facts.published_cut = fixture_cut(11, kPublishedForwarded, 24, 0x31);
    facts.live_cut = facts.published_cut;
    facts.private_cut = fixture_chain_cut(12, kPrivateForwarded, 0x32);
    facts.exact_completion_cut = facts.private_cut.id;
    facts.exact_completion_committed = true;
    facts.controller_thread = true;
    facts.live_allocation_healthy = true;
    facts.private_allocation_healthy = true;
    facts.service_idle = true;
    return facts;
}

void test_default_evidence_is_refused_not_assumed() {
    RepairOwnerEvidence none;
    check(!none.produced(), "a default evidence value carries no facts");
    check(!none.certified_at_private_cut() && none.token_debt() == 0 && none.published_serial() == 0 &&
              !none.retired() && !none.suspended(),
          "an unproduced evidence states no serial, no certification and no retained state: every gate that "
          "consults it refuses first, on produced()");
    RepairOwnerRefusal why = RepairOwnerRefusal::none;
    core::StateCut certified{};
    check(!retained_yield_allowed(none, why, certified) && why == RepairOwnerRefusal::refused_evidence_absent,
          "no facts reads as no facts, never as facts passed");
    uint64_t target = 0;
    check(!retained_private_unit_allowed(none, kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_evidence_absent,
          "and the private-unit gate refuses it too");
    RepairOwnerFacts facts = sane_facts();
    facts.published_serial = 0;
    check(!make_repair_owner_evidence(facts, why).produced() &&
              why == RepairOwnerRefusal::refused_serial_unrecorded,
          "a controller that never published cannot vouch for itself");
    facts = sane_facts();
    facts.private_charge = nullptr;   // the private gang has no ledger charge
    facts.private_charge_purpose = core::ReservationPurpose::private_work;
    check(!make_repair_owner_evidence(facts, why).produced() &&
              why == RepairOwnerRefusal::refused_no_retained_charge,
          "an unretired owner with no private_work reservation is refused");
    auto wrong = std::make_shared<FixtureCharge>();
    wrong->purpose = core::ReservationPurpose::saved_maintenance;
    facts = sane_facts();
    facts.private_charge = wrong;
    facts.private_charge_purpose = wrong->purpose;
    check(!make_repair_owner_evidence(facts, why).produced() &&
              why == RepairOwnerRefusal::refused_charge_purpose_mismatch,
          "the retained charge must be the ledger's own private_work reservation, not just any lease");
    facts = sane_facts();
    facts.published_cut = core::StateCut{};   // zeroed digest: "nothing was really certified"
    check(!make_repair_owner_evidence(facts, why).produced() &&
              why == RepairOwnerRefusal::refused_evidence_invalid,
          "a cut with no history digest is refused as evidence");
    facts = sane_facts();
    facts.private_cut = core::StateCut{};
    check(!make_repair_owner_evidence(facts, why).produced() &&
              why == RepairOwnerRefusal::refused_evidence_invalid,
          "an unretired owner cannot claim a private transaction with no cut behind it");
    wrong.reset();
}

void test_contradictory_snapshots_cannot_be_invented() {
    RepairOwnerRefusal why = RepairOwnerRefusal::none;
    auto lease = std::make_shared<FixtureCharge>();
    // `retire()` drops the private transaction; claiming retired while a private gang still exists is the shape
    // that ends in the terminal controller destructor (resident_repair_controller.inc:70-71).
    RepairOwnerFacts retired_with_private = sane_facts();
    retired_with_private.retired = true;
    retired_with_private.private_charge = lease;
    retired_with_private.private_charge_purpose = lease->purpose;
    check(!make_repair_owner_evidence(retired_with_private, why).produced() &&
              why == RepairOwnerRefusal::refused_evidence_invalid,
          "retired + a live private gang is contradictory and refused");
    RepairOwnerFacts unretired_without_private = sane_facts();
    unretired_without_private.private_phase = core::serving_raw::Phase::absent;
    unretired_without_private.private_charge = lease;
    unretired_without_private.private_charge_purpose = lease->purpose;
    check(!make_repair_owner_evidence(unretired_without_private, why).produced() &&
              why == RepairOwnerRefusal::refused_evidence_invalid,
          "unretired + no private transaction is contradictory and refused");
    RepairOwnerFacts private_ahead = sane_facts();
    private_ahead.private_cut = fixture_cut(12, kPublishedForwarded + 8, kPublishedForwarded + 8, 0x33);
    private_ahead.exact_completion_cut = private_ahead.private_cut.id;
    private_ahead.private_charge = lease;
    private_ahead.private_charge_purpose = lease->purpose;
    check(!make_repair_owner_evidence(private_ahead, why).produced() &&
              why == RepairOwnerRefusal::refused_debt_inconsistent,
          "negative debt cannot be stated");
    RepairOwnerFacts poisoned = sane_facts();
    poisoned.live_phase = core::serving_raw::Phase::poisoned;
    poisoned.private_charge = lease;
    poisoned.private_charge_purpose = lease->purpose;
    check(!make_repair_owner_evidence(poisoned, why).produced() &&
              why == RepairOwnerRefusal::refused_phase_not_drained,
          "a poisoned live fence can never be evidence of a parkable owner");
    // What a real retirement DOES produce: no private gang, no charge, `retired` true.
    RepairOwnerFacts retired = sane_facts();
    retired.retired = true;
    retired.private_phase = core::serving_raw::Phase::absent;
    retired.private_cut = {};
    retired.private_charge = nullptr;
    retired.exact_completion_cut = 0;
    retired.exact_completion_committed = false;
    const RepairOwnerEvidence retired_evidence = make_repair_owner_evidence(retired, why);
    check(retired_evidence.produced() && retired_evidence.retired(), "a genuinely retired snapshot is evidence");
    core::StateCut certified{};
    check(!retained_yield_allowed(retired_evidence, why, certified) &&
              why == RepairOwnerRefusal::refused_retired, "a retired owner can never be yielded again");
    // A retired owner that still carries the suspended flag (the shape a hidden retirement inside a retained
    // request would leave) must be refused by the park gate, not parked as if it were live.
    RepairOwnerFacts retired_suspended = retired;
    retired_suspended.suspended = true;
    check(!retained_park_allowed(make_repair_owner_evidence(retired_suspended, why), RepairOwnerRefusal::none,
                                 why) &&
              why == RepairOwnerRefusal::refused_retired, "and a retired owner is never parked as a live one");
    lease.reset();
}

void test_evidence_derives_debt_and_coherence() {
    auto lease = std::make_shared<FixtureCharge>();
    RepairOwnerRefusal why = RepairOwnerRefusal::none;
    RepairOwnerFacts facts = sane_facts();
    facts.private_charge = lease;
    facts.private_charge_purpose = lease->purpose;
    const RepairOwnerEvidence evidence = make_repair_owner_evidence(facts, why);
    check(evidence.produced(), "sane facts produce evidence");
    check(evidence.token_debt() == kPublishedForwarded - kPrivateForwarded,
          "debt is derived from the cuts, never supplied by a caller");
    check(evidence.live_coherent(), "the published cut equals the live cut at the park");
    check(evidence.certified_at_private_cut(), "the private cut carries the committed witness");
    RepairOwnerFacts moved = facts;
    moved.live_cut = fixture_cut(99, kPublishedForwarded + 4, kPublishedForwarded + 4, 0x77);
    const RepairOwnerEvidence moved_evidence = make_repair_owner_evidence(moved, why);
    check(moved_evidence.produced() && !moved_evidence.live_coherent(),
          "a diverged live cut is representable but not coherent");
    core::StateCut certified{};
    check(!retained_yield_allowed(moved_evidence, why, certified) &&
              why == RepairOwnerRefusal::refused_published_cut_divergence,
          "publication divergence refuses the certified yield");
    RepairOwnerFacts uncommitted = facts;
    uncommitted.exact_completion_committed = false;
    const RepairOwnerEvidence uncommitted_evidence = make_repair_owner_evidence(uncommitted, why);
    check(uncommitted_evidence.produced() && !uncommitted_evidence.certified_at_private_cut(),
          "an uncommitted witness is representable and never counts as certification");
    check(!retained_yield_allowed(uncommitted_evidence, why, certified) &&
              why == RepairOwnerRefusal::refused_certified_cut_not_exact,
          "an uncommitted witness does not certify the cut it names");
    RepairOwnerFacts no_witness = facts;
    no_witness.exact_completion_cut = 0;
    no_witness.exact_completion_committed = false;
    check(!retained_yield_allowed(make_repair_owner_evidence(no_witness, why), why, certified) &&
              why == RepairOwnerRefusal::refused_no_certified_cut,
          "and no all-stage witness at all is refused by name");
    RepairOwnerFacts inexact = facts;
    inexact.private_cut = fixture_cut(12, kPrivateForwarded, kPrivateForwarded - 4, 0x32);
    const RepairOwnerEvidence inexact_evidence = make_repair_owner_evidence(inexact, why);
    check(inexact_evidence.produced() && !inexact_evidence.certified_at_private_cut(),
          "an inexact private cut is representable");
    check(!retained_yield_allowed(inexact_evidence, why, certified) &&
              why == RepairOwnerRefusal::refused_certified_cut_not_exact,
          "a private cut that is not itself exact cannot be certified");
    RepairOwnerFacts zero_debt = facts;
    zero_debt.private_cut = fixture_cut(12, kPublishedForwarded, kPublishedForwarded, 0x32);
    zero_debt.exact_completion_cut = zero_debt.private_cut.id;
    check(!retained_yield_allowed(make_repair_owner_evidence(zero_debt, why), why, certified) &&
              why == RepairOwnerRefusal::refused_no_debt,
          "zero debt is the exact path: the drain/finish path owns it, never a retained yield");
    lease.reset();
}

void test_yield_gate_refuses_every_unsafe_owner() {
    auto lease = std::make_shared<FixtureCharge>();
    auto attempt = [&](auto mutate) {
        RepairOwnerFacts facts = sane_facts();
        facts.private_charge = lease;
        facts.private_charge_purpose = lease->purpose;
        mutate(facts);
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        const RepairOwnerEvidence evidence = make_repair_owner_evidence(facts, why);
        core::StateCut certified{};
        bool accepted = evidence.produced();
        if (accepted) accepted = retained_yield_allowed(evidence, why, certified);
        return std::make_pair(accepted, why);
    };
    check(attempt([](RepairOwnerFacts&) {}).first, "the coherent debt owner is accepted");
    check(attempt([](RepairOwnerFacts& f) { f.busy = true; }).second == RepairOwnerRefusal::refused_busy,
          "a busy controller cannot yield");
    check(attempt([](RepairOwnerFacts& f) { f.cancel_requested = true; }).second ==
              RepairOwnerRefusal::refused_cancel_requested, "a pending cancellation cannot yield");
    check(attempt([](RepairOwnerFacts& f) { f.finished = true; }).second == RepairOwnerRefusal::refused_finished,
          "an owner that already ran finish_* cannot yield");
    check(attempt([](RepairOwnerFacts& f) { f.suspended = true; }).second ==
              RepairOwnerRefusal::refused_already_suspended, "a suspended owner cannot yield twice");
    check(attempt([](RepairOwnerFacts& f) { f.live_phase = core::serving_raw::Phase::running; }).second ==
              RepairOwnerRefusal::refused_phase_not_drained, "no yield inside a live work phase");
    check(attempt([](RepairOwnerFacts& f) { f.private_phase = core::serving_raw::Phase::running; }).second ==
              RepairOwnerRefusal::refused_private_phase_not_drained, "no yield with the private gang working");
    check(attempt([](RepairOwnerFacts& f) { f.live_allocation_healthy = false; }).second ==
              RepairOwnerRefusal::refused_allocation_unhealthy, "an unhealthy live allocation cannot yield");
    check(attempt([](RepairOwnerFacts& f) { f.private_allocation_healthy = false; }).second ==
              RepairOwnerRefusal::refused_allocation_unhealthy, "an unhealthy private allocation cannot yield");
    check(attempt([](RepairOwnerFacts& f) { f.service_idle = false; }).second ==
              RepairOwnerRefusal::refused_service_not_idle, "another controller operation holds the service");
    check(attempt([](RepairOwnerFacts& f) { f.controller_thread = false; }).second ==
              RepairOwnerRefusal::refused_wrong_controller_thread, "only the controller thread may yield");
    // The same conjunctions drive the PARK gate: only a suspended, certified, debt-carrying, coherent owner
    // passes it.  Cancellation is refused one step earlier, by the yield gate above, and the park happens in the
    // same step as the yield.
    RepairOwnerRefusal why = RepairOwnerRefusal::none;
    RepairOwnerFacts unyielded = sane_facts();
    unyielded.private_charge = lease;
    unyielded.private_charge_purpose = lease->purpose;
    check(!retained_park_allowed(make_repair_owner_evidence(unyielded, why), RepairOwnerRefusal::none, why) &&
              why == RepairOwnerRefusal::refused_not_suspended,
          "an owner that was never yielded cannot be parked");
    RepairOwnerFacts uncoherent = sane_facts();
    uncoherent.suspended = true;
    uncoherent.live_cut = fixture_cut(99, kPublishedForwarded + 4, kPublishedForwarded + 4, 0x77);
    uncoherent.private_charge = lease;
    uncoherent.private_charge_purpose = lease->purpose;
    check(!retained_park_allowed(make_repair_owner_evidence(uncoherent, why), RepairOwnerRefusal::none, why) &&
              why == RepairOwnerRefusal::refused_published_cut_divergence,
          "a park at an incoherent cut is refused");
    lease.reset();
}

void test_private_unit_gate_requires_the_suspended_state_and_a_cap() {
    auto lease = std::make_shared<FixtureCharge>();
    RepairOwnerFacts facts = sane_facts();
    facts.private_charge = lease;
    facts.private_charge_purpose = lease->purpose;
    RepairOwnerRefusal why = RepairOwnerRefusal::none;
    const RepairOwnerEvidence unyielded = make_repair_owner_evidence(facts, why);
    uint64_t target = 0;
    check(!retained_private_unit_allowed(unyielded, kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_not_suspended,
          "a live-coherent owner may not run retained private work: it was never yielded");
    facts.suspended = true;
    const RepairOwnerEvidence suspended = make_repair_owner_evidence(facts, why);
    check(retained_private_unit_allowed(suspended, kFixtureCap, target, why) &&
              target == kPrivateForwarded + kFixtureCap,
          "a suspended owner advances its private gang by at most the selector's cap");
    check(retained_private_unit_allowed(suspended, kPublishedForwarded * 2, target, why) &&
              target == kPublishedForwarded, "the cap never carries private work past the published cut");
    check(!retained_private_unit_allowed(suspended, 0, target, why) &&
              why == RepairOwnerRefusal::refused_unit_cap_unset,
          "no selector cap means no private unit: there is no compiled default here");
    RepairOwnerFacts caught = facts;
    caught.private_cut = fixture_cut(12, kPublishedForwarded, kPublishedForwarded, 0x32);
    caught.exact_completion_cut = caught.private_cut.id;
    check(!retained_private_unit_allowed(make_repair_owner_evidence(caught, why), kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_unit_cap_exhausted,
          "an owner with no debt left has nothing to advance");
    RepairOwnerFacts busy = facts;
    busy.busy = true;
    check(!retained_private_unit_allowed(make_repair_owner_evidence(busy, why), kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_busy, "a busy owner is refused");
    RepairOwnerFacts cancelled = facts;
    cancelled.cancel_requested = true;
    check(!retained_private_unit_allowed(make_repair_owner_evidence(cancelled, why), kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_cancel_requested, "a cancel-requested owner is refused");
    RepairOwnerFacts service_busy = facts;
    service_busy.service_idle = false;
    check(!retained_private_unit_allowed(make_repair_owner_evidence(service_busy, why), kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_service_not_idle,
          "the private unit needs the service idle, exactly as run_private_one does");
    RepairOwnerFacts wrong_thread = facts;
    wrong_thread.controller_thread = false;
    check(!retained_private_unit_allowed(make_repair_owner_evidence(wrong_thread, why), kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_wrong_controller_thread,
          "only the runtime's controller thread may run the unit");
    RepairOwnerFacts live_running = facts;
    live_running.live_phase = core::serving_raw::Phase::running;
    check(!retained_private_unit_allowed(make_repair_owner_evidence(live_running, why), kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_phase_not_drained,
          "a parked unit may not run while a foreground live fence is running");
    RepairOwnerFacts private_unhealthy = facts;
    private_unhealthy.private_allocation_healthy = false;
    check(!retained_private_unit_allowed(make_repair_owner_evidence(private_unhealthy, why), kFixtureCap, target,
                                         why) &&
              why == RepairOwnerRefusal::refused_allocation_unhealthy, "an unhealthy private gang is refused");
    lease.reset();
}

// ---- 3. parking a LIVE unretired owner with debt outstanding ---------------------------------------
void test_a_live_yielded_owner_is_parked_and_its_charge_is_pinned() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    check(FixtureCharge::live == 1, "the private gang's charge is held before the park");
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "the parked owner is the live, unretired, debt-carrying one");
    check(made.owner == nullptr, "the registry moved the payload out of the request slot (guard stays inert)");
    check(registry.size() == 1 && registry.contains(identity), "the entry is keyed by the published identity");
    SuspendedOwnerFacts facts;
    check(registry.entry_facts(identity, facts), "the entry is readable");
    check(facts.holds_live_controller && !facts.quarantined, "a LIVE unretired owner is what is parked");
    check(facts.token_debt == kPublishedForwarded - kPrivateForwarded,
          "the entry records the outstanding debt, not a closed one");
    check(facts.certified_cut.forwarded == kPrivateForwarded &&
              facts.certified_cut.exact_frontier == kPrivateForwarded,
          "the entry records the certified private cut");
    check(facts.live_cut_at_park == facts.published_cut, "and the live cut as it stood at the park");
    check(facts.reservation_purposes.size() == 1 &&
              facts.reservation_purposes.front() == core::ReservationPurpose::private_work,
          "the entry pins the ledger's own private_work charge (no new enumerator)");
    // The test's own controller handle is a second shared_ptr; the ENTRY pin is what holds the lease, so the
    // charge stays while the entry lives.
    check(FixtureCharge::live == 1, "the parked entry pins the charge");
    std::string retire_error;
    check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error),
          "retirement goes through the controller's own retire()");
    check(registry.size() == 0 && registry.owners_released() == 1, "the entry is gone, released once");
    check(FixtureCharge::live == 0, "the parked owner's charge is fully released: no leak");
    made.controller.reset();
}

void test_a_retired_payload_is_refused() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    check(made.controller->retire(error), "retire the fixture controller");
    made.owner->retired = true;
    made.owner->controller.reset();
    check(!registry.park_retained_owner(identity, made.owner, ParkOwnerFacts{}, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_retired,
          "a retired payload is refused: the registry holds live owners only");
    check(bool(made.owner), "and the caller keeps the payload it must release today");
    made.owner.reset();   // safe: the controller is already retired
    made.controller.reset();
    check(FixtureCharge::live == 0, "retirement released the charge");
}

void test_park_refusals() {
    std::string error;
    {   // No selector: no registry capacity, nothing retained.
        FixtureRegistry closed(0);
        FixtureOwner made = make_yielded_owner();
        const core::WindowKey identity = made.owner->published.key;
        check(!closed.park_retained_owner(identity, made.owner, park_facts(closed, *made.controller, identity), error) &&
                  closed.last_refusal() == RepairOwnerRefusal::refused_selector_absent,
              "with the selector absent nothing is retained and the caller keeps today's release");
        check(bool(made.owner), "and the caller still owns the payload");
        retire_release(made);
    }
    {   // An empty slot is not an owner.
        FixtureRegistry registry(kOneSlot);
        std::unique_ptr<FixtureOutcome> empty_slot;
        check(!registry.park_retained_owner(fixture_key(), empty_slot, ParkOwnerFacts{}, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_empty_owner_slot,
              "an empty slot is not an owner");
        auto unpopulated = std::make_unique<FixtureOutcome>();
        check(!registry.park_retained_owner(fixture_key(), unpopulated, ParkOwnerFacts{}, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_unpopulated_owner,
              "an empty outcome is never reported as a retained owner");
    }
    {   // Only an owner certified at a retained yield may be parked.
        FixtureRegistry registry(kOneSlot);
        FixtureOwner made = make_yielded_owner();
        made.owner->yielded = false;
        check(!registry.park_retained_owner(made.owner->published.key, made.owner,
                                            park_facts(registry, *made.controller, made.owner->published.key), error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_not_suspended,
              "an owner never yielded is refused");
        retire_release(made);
    }
    {   // The live session must be the window the owner published into.
        FixtureRegistry registry(kOneSlot);
        FixtureOwner made = make_yielded_owner();
        auto foreign = park_facts(registry, *made.controller, fixture_key(5, 2, "window-b"));
        check(!registry.park_retained_owner(made.owner->published.key, made.owner, foreign, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_window_identity_diverged,
              "a foreign session window is refused");
        // Parking under a key the owner never published is refused too.
        auto not_published = park_facts(registry, *made.controller, fixture_key(5, 2, "window-c"));
        check(!registry.park_retained_owner(fixture_key(5, 2, "window-c"), made.owner, not_published, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_identity_not_publication,
              "an owner cannot be parked under a key it never published");
        retire_release(made);
    }
    {   // The request fence must be released, at the drained cut, before an owner is retained.  Two slots: an
        // unconsumed up-front reservation stays outstanding until the serving scope releases it, so each of the
        // two refusals below needs its own slot.
        FixtureRegistry registry(kTwoSlots);
        FixtureOwner made = make_yielded_owner();
        const core::WindowKey identity = made.owner->published.key;
        auto open_request = park_facts(registry, *made.controller, identity, core::serving_raw::Phase::drained, true);
        check(!registry.park_retained_owner(identity, made.owner, open_request, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_request_not_closed,
              "the request fence must be released before an owner can be retained");
        auto running = park_facts(registry, *made.controller, identity, core::serving_raw::Phase::running, false);
        check(!registry.park_retained_owner(identity, made.owner, running, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_phase_not_drained,
              "a park needs the drained cut, never a live work phase");
        retire_release(made);
    }
    {   // Evidence that no longer matches the payload is refused, not trusted.
        FixtureRegistry registry(kTwoSlots);
        FixtureOwner made = make_yielded_owner();
        const core::WindowKey identity = made.owner->published.key;
        auto stale = park_facts(registry, *made.controller, identity);
        stale.owner = RepairOwnerEvidence{};   // no evidence at all
        check(!registry.park_retained_owner(identity, made.owner, stale, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_evidence_invalid,
              "a park with no controller evidence is refused");
        made.owner->published.cut = fixture_cut(88, kPublishedForwarded - 4, kPublishedForwarded - 4, 0x61);
        check(!registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_record_changed,
              "a payload whose publication differs from its controller's is refused, not parked on a guess");
        retire_release(made);
    }
    {   // Duplicate identity, then a second live owner of the same window.
        FixtureRegistry registry(kOneSlot);
        FixtureOwner first = make_yielded_owner();
        const core::WindowKey first_identity = first.owner->published.key;
        check(registry.park_retained_owner(first_identity, first.owner,
                                           park_facts(registry, *first.controller, first_identity), error),
              "first park");
        FixtureOwner again = make_yielded_owner();
        check(!registry.park_retained_owner(again.owner->published.key, again.owner,
                                            park_facts(registry, *again.controller, again.owner->published.key), error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_duplicate_identity,
              "a duplicate identity is refused, never replaced");
        check(bool(again.owner), "and the caller keeps the second owner to retire");
        retire_release(again);
        FixtureOwner bumped = make_yielded_owner(fixture_key(6, 3));   // same window, newer epochs
        check(!registry.park_retained_owner(bumped.owner->published.key, bumped.owner,
                                            park_facts(registry, *bumped.controller, bumped.owner->published.key), error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_duplicate_window,
              "a second live owner of the SAME window (epochs aside) is refused: the slot bound does not "
              "authorise two private gangs for one conversation");
        retire_release(bumped);
        std::string drain_error;
        check(registry.retire_all("test", FixtureRetire{}, drain_error) && registry.size() == 0,
              "the first registry is drained explicitly");
        check(FixtureCharge::live == 0, "draining releases the charge");
        // Slot exhaustion: refuse, never evict.
        FixtureRegistry bounded(kOneSlot);
        FixtureOwner kept = make_yielded_owner();
        const core::WindowKey kept_identity = kept.owner->published.key;
        check(bounded.park_retained_owner(kept_identity, kept.owner, park_facts(bounded, *kept.controller, kept_identity),
                                          error),
              "park into the bounded registry");
        FixtureOwner other = make_yielded_owner(fixture_key(5, 2, "window-d"));
        check(!bounded.park_retained_owner(other.owner->published.key, other.owner,
                                           park_facts(bounded, *other.controller, other.owner->published.key), error) &&
                  bounded.last_refusal() == RepairOwnerRefusal::refused_full,
              "at the slot bound the registry refuses rather than evicting a live controller");
        check(bounded.size() == 1 && bounded.contains(kept_identity), "the parked owner was not disturbed");
        retire_release(other);
        std::string retire_error;
        check(bounded.retire_all("test", FixtureRetire{}, retire_error) && bounded.size() == 0,
              "the bounded registry can be drained explicitly");
        check(FixtureCharge::live == 0, "and draining it releases every charge");
    }
}

// ---- 4. the bounded private unit, driven through the real registry --------------------------------
void test_one_bounded_unit_advances_a_parked_owner() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    const core::WindowKey session_key = fixture_key(5, 2);   // the session's own identity (epochs aside)
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, session_key), error),
          "park");
    const int units_before = FixtureController::unit_calls;
    RepairOwnerEvidence after;
    uint64_t advanced = 0;
    check(run_unit(registry, identity, idle_session(*made.controller), kFixtureCap, after, advanced, error),
          "one bounded unit runs through the owner's own private path");
    check(advanced == kFixtureCap && FixtureController::unit_calls == units_before + 1,
          "exactly one unit, exactly the selector's cap in tokens");
    check(after.produced() && after.private_cut().forwarded == kPrivateForwarded + kFixtureCap,
          "the private checkpoint moved by the cap and no further");
    check(registry.units_advanced() == 1 && registry.tokens_repaired() == kFixtureCap, "the tick is counted");
    SuspendedOwnerFacts record;
    check(registry.entry_facts(identity, record) &&
              record.certified_cut.forwarded == kPrivateForwarded + kFixtureCap &&
              record.token_debt == kPublishedForwarded - (kPrivateForwarded + kFixtureCap),
          "the entry's certified cut and debt are refreshed from the real facts");
    check(made.controller->published_serial == kFixtureSerial &&
              made.controller->published_cut.forwarded == kPublishedForwarded,
          "the unit never touched the live publication");
    check(record.state == RepairOwnerState::suspended && !record.quarantined,
          "the owner is still suspended and usable after the unit");
    after = RepairOwnerEvidence{};   // the delivered evidence also pins the charge: drop it before the check
    std::string retire_error;
    check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error), "retire");
    check(FixtureCharge::live == 0, "charge released");
    made.controller.reset();
}

void test_advance_refuses_outside_a_legal_idle_window() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "park");
    const int units_before = FixtureController::unit_calls;
    RepairOwnerEvidence after;
    uint64_t advanced = 0;
    check(!run_unit(registry, identity, idle_session(*made.controller, false), kFixtureCap, after, advanced,
                    error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_private_window_not_open,
          "no unit without the lifecycle's private work window: the state machine is what admits it");
    LiveSessionFacts request_open = idle_session(*made.controller);
    request_open.request_open = true;
    check(!run_unit(registry, identity, request_open, kFixtureCap, after, advanced, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_request_not_closed,
          "no unit while a foreground request holds the session");
    LiveSessionFacts foreign = idle_session(*made.controller, true, fixture_key(5, 2, "window-b"));
    check(!run_unit(registry, identity, foreign, kFixtureCap, after, advanced, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_window_identity_diverged,
          "no unit for a session that is not this owner's window");
    LiveSessionFacts working = idle_session(*made.controller);
    working.live_phase = core::serving_raw::Phase::running;
    check(!run_unit(registry, identity, working, kFixtureCap, after, advanced, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_phase_not_drained,
          "no unit while the live fence is working");
    check(!registry.advance_parked_owner(
              fixture_key(5, 2, "window-never"), idle_session(*made.controller), kFixtureCap,
              [](FixtureOutcome&, uint64_t&, RepairOwnerEvidence&, std::string&) { return true; },
              after, advanced, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_no_entry,
          "no entry, no unit");
    check(!run_unit(registry, identity, idle_session(*made.controller), 0, after, advanced, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_unit_cap_unset,
          "the idle unit selector names the cap: with none, no work is admitted");
    check(FixtureController::unit_calls == units_before, "and not one unit ran for any refused window");
    std::string retire_error;
    check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error), "retire");
    check(FixtureCharge::live == 0, "charge released");
    made.controller.reset();
}

void test_advance_refuses_a_record_that_no_longer_matches() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "park");
    // The conversation was published underneath the parked owner: its serial moved.
    made.controller->published_serial = kFixtureSerial + 1;
    const int units_before = FixtureController::unit_calls;
    RepairOwnerEvidence after;
    uint64_t advanced = 0;
    check(!run_unit(registry, identity, idle_session(*made.controller), kFixtureCap, after, advanced, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_record_changed,
          "the record cannot vouch for work the controller no longer matches");
    check(FixtureController::unit_calls == units_before, "and the unit never ran");
    SuspendedOwnerFacts record;
    check(registry.entry_facts(identity, record) && !record.quarantined,
          "a refused advance leaves the owner parked and retireable");
    made.controller->published_serial = kFixtureSerial;   // restore, so the retire path is the normal one
    std::string retire_error;
    check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error), "retire");
    check(FixtureCharge::live == 0, "charge released");
    made.controller.reset();
}

void test_advance_refuses_a_unit_that_breaks_the_contract() {
    {   // A unit that republished CURRENT.
        FixtureRegistry registry(kTwoSlots);
        FixtureOwner made = make_yielded_owner();
        const core::WindowKey identity = made.owner->published.key;
        std::string error;
        check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
              "park");
        made.controller->move_live = true;
        RepairOwnerEvidence after;
        uint64_t advanced = 0;
        check(!run_unit(registry, identity, idle_session(*made.controller), kFixtureCap, after, advanced, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_live_state_moved,
              "a private unit that moves the live publication is refused, not adopted");
        after = RepairOwnerEvidence{};
        SuspendedOwnerFacts record;
        check(registry.entry_facts(identity, record) && record.quarantined,
              "the entry is quarantined: its record can no longer be trusted");
        check(!run_unit(registry, identity, idle_session(*made.controller), kFixtureCap, after, advanced, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_quarantined,
              "a quarantined owner is never advanced again");
        std::string retire_error;
        check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error) &&
                  FixtureCharge::live == 0,
              "a quarantined owner is still retired deliberately, releasing its charge");
        made.controller.reset();
    }
    {   // A unit that ran past the selector's cap.
        FixtureRegistry registry(kTwoSlots);
        FixtureOwner made = make_yielded_owner();
        const core::WindowKey identity = made.owner->published.key;
        std::string error;
        check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
              "park");
        made.controller->overshoot = 3;
        RepairOwnerEvidence after;
        uint64_t advanced = 0;
        check(!run_unit(registry, identity, idle_session(*made.controller), kFixtureCap, after, advanced, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_unit_too_large,
              "the selector's cap is enforced against the controller's own facts, not the unit's report");
        std::string retire_error;
        check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error),
              "an overrunning unit quarantines the record; the owner is still retireable");
        after = RepairOwnerEvidence{};
        check(FixtureCharge::live == 0, "charge released");
        made.controller.reset();
    }
}

// ---- 5. stale publication and the resume gates: recorded facts matching is NOT enough ---------------
void test_publication_is_refused_once_the_live_session_moved_on() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "park");
    FixtureController& controller = *made.controller;
    LiveSessionFacts session = idle_session(controller);
    RepairOwnerRefusal why = RepairOwnerRefusal::none;
    check(registry.publish_gate(identity, session, controller.evidence(), why, error) &&
              why == RepairOwnerRefusal::none,
          "with the session still standing on the owner's publication, publication is admissible");
    // A second conversation advances the live session.  Every fact the entry recorded still matches the
    // controller, so a registry that trusted its own record would publish a stale state.
    const core::StateCut newer = fixture_cut(77, kPublishedForwarded + 8, kPublishedForwarded + 8, 0x91);
    session.session_cut = newer;
    check(!registry.publish_gate(identity, session, controller.evidence(), why, error) &&
              why == RepairOwnerRefusal::refused_stale_publication,
          "stale publication is refused against the SESSION's current cut, not the recorded one");
    controller.live_cut = newer;   // the live transaction really did move
    check(!registry.publish_gate(identity, session, controller.evidence(), why, error) &&
              why == RepairOwnerRefusal::refused_stale_publication,
          "and it stays refused when the controller's own live cut diverged too");
    check(!controller.evidence().live_coherent(),
          "the controller's own coherence conjunction now fails, so no live-requiring entry point accepts it");
    check(controller.suspended, "the owner stays suspended: nothing reconciles it by force (GAP-A blocker 7)");
    // Its private work remains legal, and still cannot publish.
    RepairOwnerEvidence after;
    uint64_t advanced = 0;
    check(run_unit(registry, identity, idle_session(controller), kFixtureCap, after, advanced, error),
          "retention keeps advancing its own private copy while the live session is someone else's");
    check(!registry.publish_gate(identity, session, controller.evidence(), why, error) &&
              why == RepairOwnerRefusal::refused_stale_publication,
          "and still cannot publish after advancing");
    LiveSessionFacts foreign = idle_session(controller, true, fixture_key(5, 2, "window-z"));
    check(!registry.publish_gate(identity, foreign, controller.evidence(), why, error) &&
              why == RepairOwnerRefusal::refused_window_identity_diverged,
          "a foreign session identity never publishes this owner's state");
    // A non-drained session refuses publication as well, once the session is back on the owner's cut: the phase
    // gate is independent of the staleness gate.
    controller.live_cut = controller.published_cut;
    LiveSessionFacts running = idle_session(controller);
    running.live_phase = core::serving_raw::Phase::running;
    check(!registry.publish_gate(identity, running, controller.evidence(), why, error) &&
              why == RepairOwnerRefusal::refused_phase_not_drained,
          "no publication while the live fence is working");
    after = RepairOwnerEvidence{};
    check(same_stable_window(fixture_key(5, 2), fixture_key(9, 9)), "bumped epochs are the same window");
    check(!same_stable_window(fixture_key(5, 2, "window-a"), fixture_key(5, 2, "window-b")),
          "a different window_id is a different window");
    // incoming_prompt_identity_matches: the F2 gate
    {
        std::vector<int32_t> owner_tokens = {100, 200, 300, 400, 500};
        strata::core::StateCut cut;
        cut.accepted = owner_tokens.size();
        cut.history = strata::core::window_history_digest(owner_tokens);
        // matching prefix (the owner's own conversation returning)
        std::vector<int32_t> a_return = {100, 200, 300, 400, 500, 600, 700};
        check(incoming_prompt_identity_matches(a_return, cut),
              "incoming prompt extending the owner's published history matches");
        // exact match (no extra tokens)
        check(incoming_prompt_identity_matches(owner_tokens, cut),
              "exact published history matches");
        // non-matching (different conversation B)
        std::vector<int32_t> b_prompt = {999, 888, 777, 666, 555, 444};
        check(!incoming_prompt_identity_matches(b_prompt, cut),
              "different conversation does not match");
        // too short
        std::vector<int32_t> too_short = {100, 200};
        check(!incoming_prompt_identity_matches(too_short, cut),
              "prompt shorter than published history refuses");
        // zero accepted
        strata::core::StateCut empty_cut;
        empty_cut.accepted = 0;
        check(!incoming_prompt_identity_matches(owner_tokens, empty_cut),
              "zero accepted refuses");
    }
    // Reservation admission: the new correctly-dimensioned charge pattern
    {
        // Simulate a 2-device budget with 32 GiB host capacity and shadow slots
        strata::core::WindowResources capacity;
        capacity.host_bytes = capacity.pinned_bytes = 32ULL << 30;
        capacity.device_bytes = {0, 0};  // 2 devices
        capacity.shadows = 4;  // allow private_work reservations
        strata::core::WindowResources baseline;
        baseline.host_bytes = baseline.pinned_bytes = 1ULL << 30;
        baseline.device_bytes = {0, 0};
        baseline.shadows = 0;
        strata::core::WindowBudget budget(capacity, baseline);
        // A correctly dimensioned charge should admit
        strata::core::WindowResources charge;
        charge.host_bytes = 100ULL << 20;  // 100 MiB capture
        charge.device_bytes = {0, 0};
        auto r1 = budget.reserve(charge, strata::core::ReservationPurpose::private_attachment);
        check(!r1, "private_attachment without parent refuses");  // requires parent (gang_lease)
        // With a proper parent: the controller's gang lease is private_work; parent must be that.
        auto parent_r = budget.reserve(
            []{ strata::core::WindowResources r; r.host_bytes=r.pinned_bytes=1ULL<<30; r.device_bytes={0,0}; r.shadows=1; return r; }(),
            strata::core::ReservationPurpose::private_work);
        check(bool(parent_r), "private_work parent admits");
        if (parent_r) {
            auto r2 = budget.reserve(charge, strata::core::ReservationPurpose::private_attachment,
                                     parent_r.lease);
            check(bool(r2), "private_attachment with private_work parent and device-dimensioned charge admits");
        }
        // foreground_return with obligation admits
        strata::core::WindowReturnObligation obl;
        obl.identity = "test-return";
        obl.required = charge;
        auto r3 = budget.reserve(charge, strata::core::ReservationPurpose::foreground_return, {}, &obl);
        check(bool(r3), "foreground_return with obligation and null parent admits");
        // foreground_return WITHOUT obligation refuses
        auto r4 = budget.reserve(charge, strata::core::ReservationPurpose::foreground_return);
        check(!r4, "foreground_return without obligation refuses");
        // Empty device_bytes vector (the old bug) refuses
        strata::core::WindowResources bad_charge;
        bad_charge.host_bytes = 100ULL << 20;
        bad_charge.device_bytes = {};  // empty — invalid
        auto r5 = budget.reserve(bad_charge, strata::core::ReservationPurpose::foreground);
        check(!r5 && r5.error == strata::core::ReservationError::invalid_vector,
              "empty device_bytes refused with invalid_vector");
    }
    std::string retire_error;
    check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error), "retire");
    check(FixtureCharge::live == 0, "charge released");
}

void test_resume_refusals_on_a_parked_owner() {
    // "Resume" in this slice is the parked owner being taken back for work or for publication; every route is
    // gated on the same conjunctions: identity, cancellation, retirement, phase and cut coherence.
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "park");
    RepairOwnerRefusal why = RepairOwnerRefusal::none;
    made.controller->cancel_requested = true;
    uint64_t target = 0;
    check(!retained_private_unit_allowed(made.controller->evidence(), kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_cancel_requested,
          "and it cannot be resumed for work either");
    made.controller->cancel_requested = false;
    made.controller->live_phase = core::serving_raw::Phase::running;
    check(!retained_private_unit_allowed(made.controller->evidence(), kFixtureCap, target, why) &&
              why == RepairOwnerRefusal::refused_phase_not_drained,
          "a non-drained live fence refuses the resume");
    made.controller->live_phase = core::serving_raw::Phase::drained;
    check(!parked_owner_publishable(made.controller->evidence(),
                                    idle_session(*made.controller, true, fixture_key(5, 2, "window-q")), identity,
                                    made.controller->published_cut, why) &&
              why == RepairOwnerRefusal::refused_window_identity_diverged,
          "identity mismatch refuses the resume");
    std::string retire_error;
    check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error), "retire cleanly");
    check(FixtureCharge::live == 0, "charge released");
    {
        // Retired behind the registry's back: the record stops matching the controller, so the unit is refused
        // and the owner is never publishable.  This is the shape a silent drop would otherwise hide.
        FixtureRegistry second(kOneSlot);
        FixtureOwner gone = make_yielded_owner();
        const core::WindowKey gone_identity = gone.owner->published.key;
        check(second.park_retained_owner(gone_identity, gone.owner, park_facts(second, *gone.controller, gone_identity),
                                         error),
              "park the second owner");
        gone.controller->retired = true;
        gone.controller->private_phase = core::serving_raw::Phase::absent;
        gone.controller->charge.reset();
        RepairOwnerEvidence after;
        uint64_t advanced = 0;
        check(!run_unit(second, gone_identity, idle_session(*gone.controller), kFixtureCap, after, advanced,
                        error) &&
                  second.last_refusal() == RepairOwnerRefusal::refused_record_changed,
              "a parked owner whose controller retired behind the registry is refused, not advanced");
        after = RepairOwnerEvidence{};
        check(!parked_owner_publishable(gone.controller->evidence(), idle_session(*gone.controller), gone_identity,
                                        gone.controller->published_cut, why) &&
                  why == RepairOwnerRefusal::refused_retired, "and a retired owner is never publishable");
        // Retirement is still available and still goes through the controller's own retire(): the payload is
        // released only once the registry can verify the controller is gone.  An already-retired controller is
        // idempotent there (resident_repair_controller.inc:1222), so the entry can be released cleanly.
        check(second.retire_owner(gone_identity, "test", FixtureRetire{}, retire_error) &&
                  second.size() == 0 && second.owners_released() == 1,
              "a retired-behind-the-back owner is still released exactly once, through the controller path");
        check(FixtureCharge::live == 0, "and its charge is gone");
        gone.controller.reset();
        check(!second.retire_owner(gone_identity, "test", FixtureRetire{}, retire_error) &&
                  second.last_refusal() == RepairOwnerRefusal::refused_no_entry,
              "a released entry is never released twice");
    }
}

// ---- 6. retirement, shutdown and the no-silent-drop invariants -------------------------------------
void test_close_and_shutdown_refuse_then_retire() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner first = make_yielded_owner();
    const core::WindowKey first_identity = first.owner->published.key;
    std::string error;
    check(registry.park_retained_owner(first_identity, first.owner, park_facts(registry, *first.controller, first_identity),
                                       error),
          "park the first");
    check(!registry.close(error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_close_holds_controller,
          "close refuses while a live controller is parked");
    check(!registry.retire_owner(first_identity, "test", error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_retire_requires_attempt,
          "retiring without the controller path is refused: a silent drop is not offered");
    check(!registry.retire_owner(first_identity, "test",
                                 [](FixtureOutcome&, std::string& e) { e = "device said no"; return false; }, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_retire_attempt,
          "a refused controller retirement leaves the owner parked, and reports the controller's reason");
    // An attempt that claims success while still holding the controller is caught, not trusted.
    check(!registry.retire_owner(first_identity, "test",
                                 [](FixtureOutcome&, std::string&) { return true; }, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_retire_left_controller,
          "an attempt that claims success while still holding the controller is refused, not trusted");
    SuspendedOwnerFacts record;
    check(registry.entry_facts(first_identity, record) && record.state == RepairOwnerState::suspended &&
              record.holds_live_controller,
          "a refused retirement leaves the owner intact, not half-released");
    // A throwing attempt must not half-release either.
    bool propagated = false;
    try {
        registry.retire_owner(first_identity, "test",
                              [](FixtureOutcome&, std::string&) -> bool { throw std::runtime_error("boom"); },
                              error);
    } catch (const std::runtime_error&) {
        propagated = true;
    }
    check(propagated, "a throwing retirement attempt propagates");
    check(registry.entry_facts(first_identity, record) && record.state == RepairOwnerState::suspended &&
              record.holds_live_controller,
          "and leaves the entry suspended, not retiring");
    // The first registry is drained explicitly before the cell moves on.
    std::string retire_error;
    check(registry.retire_all("test", FixtureRetire{}, retire_error) && registry.size() == 0,
          "the first registry is drained explicitly");
    check(FixtureCharge::live == 0, "and its charge is released");
    first.controller.reset();
    // Shutdown: retire everything, in order, and release every charge exactly once.
    FixtureRegistry shutdown(kTwoSlots);
    FixtureOwner first_owner = make_yielded_owner();
    FixtureOwner second = make_yielded_owner(fixture_key(5, 2, "window-b"));
    check(shutdown.park_retained_owner(first_owner.owner->published.key, first_owner.owner,
                                       park_facts(shutdown, *first_owner.controller, first_owner.owner->published.key),
                                       error),
          "park first into the shutdown registry");
    check(shutdown.park_retained_owner(second.owner->published.key, second.owner,
                                       park_facts(shutdown, *second.controller, second.owner->published.key), error),
          "park second");
    check(FixtureCharge::live == 2, "both charges are held while parked");
    check(shutdown.retire_all("serving shutdown", FixtureRetire{}, retire_error),
          "retire_all is the shutdown / eviction path");
    check(shutdown.size() == 0 && shutdown.owners_released() == 2, "every owner released exactly once");
    check(FixtureCharge::live == 0, "and the ledger shows no leak");
    check(shutdown.close(error), "close after retirement is trivially admissible");
    check(!shutdown.retire_owner(first_identity, "test", FixtureRetire{}, error) &&
              shutdown.last_refusal() == RepairOwnerRefusal::refused_no_entry,
          "retiring an already-released owner is refused rather than double-released");
    first_owner.controller.reset();
    second.controller.reset();
}

// ---- 7. the request-scope guard's extended rule -----------------------------------------------------
void test_scope_guard_fires_when_an_owner_is_lost_and_stays_inert_when_registered() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    OwnerScopeFacts facts;
    facts.active_holds_owner = made.owner != nullptr;
    facts.active_holds_controller = made.owner != nullptr && made.owner->controller != nullptr;
    check(owner_scope_escape(facts) == OwnerScopeEscape::failstop_live_unregistered,
          "today's rule, unchanged: a live controller may not escape the request scope");
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "park it");
    facts = OwnerScopeFacts{};
    facts.retention_recorded = true;
    facts.recorded_identity = identity;
    facts.registry_holds_record = registry.contains(identity);
    check(owner_scope_escape(facts) == OwnerScopeEscape::inert,
          "the guard stays inert when the owner is legitimately registered");
    std::string retire_error;
    check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error), "retire it");
    facts.registry_holds_record = registry.contains(identity);
    check(owner_scope_escape(facts) == OwnerScopeEscape::failstop_retained_lost,
          "a recorded retention whose entry is gone and whose slot is empty is a LOST owner: failstop");
    OwnerScopeFacts nothing_retained;
    check(owner_scope_escape(nothing_retained) == OwnerScopeEscape::inert,
          "nothing retained, nothing lost: the reviewed guard does nothing");
    made.controller.reset();
}

// ---- 8. the demand gate for the idle tick ----------------------------------------------------------
ServingIdleDemand quiet_demand(uint64_t queued = 0) {
    ServingIdleDemand demand;
    demand.queued_input_lines = queued;
    return demand;
}

void test_idle_admission_is_hard_gated_on_demand_not_on_any_number() {
    const uint64_t debt = kPublishedForwarded - kPrivateForwarded;
    check(serving_repair_idle_admission(quiet_demand(), debt, true).reason ==
              RepairAdmissionReason::admitted_idle,
          "an idle window with debt and resources admits one bounded unit");
    check(serving_repair_idle_admission(quiet_demand(1), debt, true).reason ==
              RepairAdmissionReason::refused_foreground_demand,
          "a queued stdin line is demand: no unit");
    ServingIdleDemand generating = quiet_demand();
    generating.current_request_generating = true;
    check(serving_repair_idle_admission(generating, debt, true).reason ==
              RepairAdmissionReason::refused_foreground_demand,
          "the current request still generating is demand: no unit");
    ServingIdleDemand fenced = quiet_demand();
    fenced.raw_fence_running = true;
    check(serving_repair_idle_admission(fenced, debt, true).reason ==
              RepairAdmissionReason::refused_foreground_demand,
          "an open RAW fence is demand: no unit");
    ServingIdleDemand open_request = quiet_demand();
    open_request.request_open = true;
    check(serving_repair_idle_admission(open_request, debt, true).reason ==
              RepairAdmissionReason::refused_foreground_demand,
          "an open serving request is demand: no unit");
    ServingIdleDemand lane = quiet_demand();
    lane.other_lane_work = true;
    check(serving_repair_idle_admission(lane, debt, true).reason ==
              RepairAdmissionReason::refused_other_eligible_work,
          "admitted work in another lane blocks it: no unit");
    check(serving_repair_idle_admission(quiet_demand(), 0, true).reason ==
              RepairAdmissionReason::refused_not_demanded,
          "zero debt is not a unit: the exact path owns that owner");
    check(serving_repair_idle_admission(quiet_demand(), debt, false).reason ==
              RepairAdmissionReason::refused_not_resource_ready,
          "no session window, or a quarantined record, means not resource-ready");
    ServingIdleDemand busy = quiet_demand(1);
    const RepairAdmissionDecision escalated = serving_repair_idle_admission(busy, debt, true);
    check(!escalated.admitted() &&
              escalated.escalation_refusal == RepairAdmissionReason::refused_age_policy_unset &&
              escalated.escalation_bound_unset,
          "the idle tick never escalates by age: no bound is supplied and no threshold exists here");
    const ServingForegroundState foreground = make_serving_foreground_state(quiet_demand(3));
    check(foreground.queued_input_lines == 3 && foreground.decode_waiting(),
          "the demand mapping feeds the kernel's own state, unchanged");
}

// ---- 10. D1 (slice 3): the certified yield is reachable at RESPONSE COMPLETION --------------------
// `SLICE2-DEFECTS-CONFIRMED.md` D1 enumerated every exit of the serving token loop against the slice-2 gate and
// showed the conjunction was unsatisfiable, so a one-token served response always took the forced drain.  This
// cell drives the SAME production predicate `generate.cpp` now calls (`serving_certified_yield_gate`) with the
// facts that gate is built from, restates the removed slice-2 conjunction field for field for comparison, and
// then runs a real yield and a real park through the shared controller gate and the registry - so what is shown
// to be reachable is the yield itself, not only a boolean.
// The removed slice-2 conjunction, restated here from the source it came from (generate.cpp, slice 2): the
// response could still continue (`produced < max_new`, live room) AND finished at "length".  It exists only to
// show which rows the new gate changes; nothing in the engine calls it.
bool slice2_yield_conjunction(const ServingYieldGateFacts& facts, bool live_room) {
    return facts.registry_present && !facts.cancelled && !facts.stop_requested &&
           facts.produced_tokens < facts.max_new_tokens && facts.ended_at_length_limit && live_room &&
           !facts.mtp_chain_active && !facts.suffix_draft && !facts.lookup_chain &&
           !facts.qualification_arm && facts.debt_open;
}

ServingYieldGateFacts completed_response(uint64_t produced, uint64_t max_new, const char* finish) {
    ServingYieldGateFacts facts;
    facts.registry_present = true;
    facts.produced_tokens = produced;
    facts.max_new_tokens = max_new;
    facts.ended_at_length_limit = std::string(finish) == "length";
    facts.ended_at_stop_token = std::string(finish) == "stop";
    facts.debt_open = true;
    facts.activation_capacity_reserved = true;   // the up-front admission slice 5 requires (see its own cell)
    return facts;
}

void test_d1_yield_is_reachable_at_response_completion() {
    // The rows the D1 table enumerates, one per way the serving token loop can exit.  `live_room` is the slice-2
    // context-room conjunct, which the production gate no longer consults.
    struct Row {
        const char* exit_path;
        ServingYieldGateFacts facts;
        bool live_room;
        ServingYieldRefusal expected;
    };
    const ServingYieldGateFacts one_token = completed_response(1, 1, "length");
    const ServingYieldGateFacts full_length = completed_response(8, 8, "length");
    const ServingYieldGateFacts at_eos = completed_response(3, 8, "stop");
    ServingYieldGateFacts zero_debt = completed_response(8, 8, "length");
    zero_debt.debt_open = false;                      // the exact-ready break: that is the finish path
    ServingYieldGateFacts zero_debt_mid = completed_response(3, 8, "length");
    zero_debt_mid.debt_open = false;                  // and the same break before the response completed
    ServingYieldGateFacts cancelled = completed_response(8, 8, "length");
    cancelled.cancelled = true;
    ServingYieldGateFacts stopped = completed_response(8, 8, "length");
    stopped.stop_requested = true;
    ServingYieldGateFacts mtp = completed_response(8, 8, "length");
    mtp.mtp_chain_active = true;
    ServingYieldGateFacts suffix = completed_response(8, 8, "length");
    suffix.suffix_draft = true;
    ServingYieldGateFacts lookup = completed_response(8, 8, "length");
    lookup.lookup_chain = true;
    ServingYieldGateFacts qualified = completed_response(8, 8, "length");
    qualified.qualification_arm = true;
    // The D3 relaxation, exactly: only the `qualification_arm` conjunct changes, and only on the strength of the
    // yield oracle being admitted. Every other conjunct still refuses with the same name.
    ServingYieldGateFacts qualified_with_oracle = qualified;
    qualified_with_oracle.yield_oracle_admitted = true;
    ServingYieldGateFacts qualified_oracle_incomplete = completed_response(3, 8, "length");
    qualified_oracle_incomplete.qualification_arm = qualified_oracle_incomplete.yield_oracle_admitted = true;
    ServingYieldGateFacts qualified_oracle_zero_debt = qualified_with_oracle;
    qualified_oracle_zero_debt.debt_open = false;
    ServingYieldGateFacts qualified_oracle_no_registry = qualified_with_oracle;
    qualified_oracle_no_registry.registry_present = false;
    ServingYieldGateFacts no_registry = completed_response(8, 8, "length");
    no_registry.registry_present = false;
    const Row rows[] = {
        // A one-token served response: exits at max_new with finish "length".  Slice 2 refused it; this is the
        // acceptance behaviour - it can now yield with debt instead of draining.
        {"one-token length completion", one_token, true, ServingYieldRefusal::none},
        {"full length completion", full_length, true, ServingYieldRefusal::none},
        {"EOS / stop token", at_eos, true, ServingYieldRefusal::none},
        {"exact-ready break (zero debt)", zero_debt, true, ServingYieldRefusal::refused_no_debt},
        {"exact-ready break mid-generation", zero_debt_mid, true,
         ServingYieldRefusal::refused_response_not_complete},
        // The context limit: generation stopped for a reason that is NOT a completed response, so the yield is
        // refused and today's drain path runs - the yield must never be a way to end generation early.
        {"context limit reached", completed_response(3, 8, "length"), false,
         ServingYieldRefusal::refused_response_not_complete},
        {"cancelled", cancelled, true, ServingYieldRefusal::refused_cancelled},
        {"STOP requested", stopped, true, ServingYieldRefusal::refused_stop_requested},
        {"MTP chain active", mtp, true, ServingYieldRefusal::refused_request_shape},
        {"suffix draft", suffix, true, ServingYieldRefusal::refused_request_shape},
        {"lookup chain", lookup, true, ServingYieldRefusal::refused_request_shape},
        // D3 (slice 4): the qualification arm is no longer excluded outright. It is refused while the
        // retained-yield oracle is not admitted, i.e. while this yield would be uncertified, and admitted once
        // that oracle is admitted; the oracle's own verdict is then decided by
        // `serving_yield_record_decision` below, not by this gate.
        {"qualification arm, no yield oracle", qualified, true,
         ServingYieldRefusal::refused_qualification_arm},
        {"qualification arm, yield oracle admitted", qualified_with_oracle, true,
         ServingYieldRefusal::none},
        {"qualification arm, oracle admitted, response incomplete", qualified_oracle_incomplete, true,
         ServingYieldRefusal::refused_response_not_complete},
        {"qualification arm, oracle admitted, zero debt", qualified_oracle_zero_debt, true,
         ServingYieldRefusal::refused_no_debt},
        {"qualification arm, oracle admitted, no registry", qualified_oracle_no_registry, true,
         ServingYieldRefusal::refused_selector_absent},
        // Both selectors off: the default path, identical to today.
        {"no owner slots", no_registry, true, ServingYieldRefusal::refused_selector_absent},
    };
    for (const Row& row : rows) {
        check(serving_certified_yield_gate(row.facts) == row.expected, row.exit_path);
        const bool slice2_admits = slice2_yield_conjunction(row.facts, row.live_room);
        const bool production_admits = serving_certified_yield_gate(row.facts) == ServingYieldRefusal::none;
        check(!slice2_admits, "slice 2 could not reach any exit in this table: the confirmed defect");
        check(!slice2_admits || production_admits, "the production gate admits every row slice 2 admitted");
    }
    // The D1 defect exactly as the table states it: every completion row below was refused by the slice-2
    // conjunction, and the rows the new gate admits are precisely the completion rows.
    check(!slice2_yield_conjunction(one_token, true),
          "slice 2 refused a one-token completed response: the confirmed defect");
    check(!slice2_yield_conjunction(full_length, true),
          "slice 2 refused a full-length completion (produced_n < max_new cannot hold)");
    check(!slice2_yield_conjunction(at_eos, true),
          "slice 2 refused an EOS completion (strcmp(finish,\"length\") cannot hold)");
    check(!slice2_yield_conjunction(completed_response(3, 8, "length"), false),
          "slice 2 refused the context-limit exit through live room as well");
    check(serving_certified_yield_gate(one_token) == ServingYieldRefusal::none &&
              serving_certified_yield_gate(at_eos) == ServingYieldRefusal::none,
          "the production gate admits both completion exits");
}

// Slice 4 (D3): the record-shape decision the serving scope calls after the retained-yield oracle emitted its
// pair. This is the SAME production predicate generate.cpp drives at the yield site
// (`serving_yield_record_decision`); the cell only supplies record facts, so each row proves one conjunct bites.
void test_d3_yield_record_decision_certifies_only_a_complete_clean_pair() {
    // A well-formed capture: published cut D=42, certified private cut R=26, debt 16, the pair at R and R+1.
    auto clean = [] {
        ServingYieldRecordFacts facts;
        facts.pair_emitted = facts.restoration_emitted = facts.restored = true;
        facts.completed = facts.payload_equal = true;
        facts.mismatches = 0;
        facts.frontier_exact = facts.continuation_exact = true;
        facts.identity_pinned = facts.ordinal_bound = true;
        facts.published_cut = 42;
        facts.certified_cut = 26;
        facts.frontier_forwarded = 26;
        facts.frontier_accepted = 26;
        facts.continuation_forwarded = 27;
        return facts;
    };
    check(serving_yield_record_decision(clean()) == ServingYieldRecordRefusal::none,
          "a complete, clean, debt-carrying pair certifies");
    // Real observed shape from abab7/8: R44 F293 frontier_acc44 frontier_fwd44 continuation_fwd45
    auto real_observed = [] {
        ServingYieldRecordFacts facts{};
        facts.pair_emitted = facts.restoration_emitted = facts.restored = true;
        facts.completed = facts.payload_equal = true; facts.mismatches = 0;
        facts.frontier_exact = facts.continuation_exact = true;
        facts.identity_pinned = facts.ordinal_bound = true;
        facts.published_cut = 293; facts.certified_cut = 44;
        facts.frontier_forwarded = 44; facts.frontier_accepted = 44;
        facts.continuation_forwarded = 45;
        return facts;
    };
    check(serving_yield_record_decision(real_observed()) == ServingYieldRecordRefusal::none,
          "observed R44/F293 closed-prefix pair certifies");
    ServingYieldRecordFacts real_old_shape = real_observed();
    real_old_shape.frontier_accepted = 45;  // old live-shape: accepted=certified+1
    check(serving_yield_record_decision(real_old_shape) == ServingYieldRecordRefusal::refused_pair_geometry,
          "R44 with old accepted=R+1 shape refuses");
    check(serving_yield_record_debt(clean()) == 16, "the reported debt is published-cut minus private-cut");
    // A default-constructed report ("the comparator said nothing") is refused, never assumed.
    check(serving_yield_record_decision(ServingYieldRecordFacts{}) ==
              ServingYieldRecordRefusal::refused_pair_absent,
          "no records: no certificate");
    struct Mutation {
        const char* name;
        ServingYieldRecordFacts facts;
        ServingYieldRecordRefusal expected;
    };
    ServingYieldRecordFacts no_pair = clean();
    no_pair.pair_emitted = false;
    ServingYieldRecordFacts no_restoration = clean();
    no_restoration.restoration_emitted = false;
    ServingYieldRecordFacts unrestored = clean();
    unrestored.restored = false;
    ServingYieldRecordFacts cancelled = clean();
    cancelled.cancelled = true;
    ServingYieldRecordFacts incomplete = clean();
    incomplete.completed = false;
    ServingYieldRecordFacts unequal = clean();
    unequal.payload_equal = false;
    ServingYieldRecordFacts mismatching = clean();
    mismatching.mismatches = 3;
    ServingYieldRecordFacts frontier_inexact = clean();
    frontier_inexact.frontier_exact = false;
    ServingYieldRecordFacts continuation_inexact = clean();
    continuation_inexact.continuation_exact = false;
    ServingYieldRecordFacts zero_debt = clean();
    zero_debt.certified_cut = zero_debt.published_cut;
    zero_debt.frontier_forwarded = zero_debt.frontier_accepted;
    zero_debt.continuation_forwarded = zero_debt.certified_cut + 1;
    ServingYieldRecordFacts certified_above = clean();
    certified_above.certified_cut = 43;                       // R above D: impossible, and unverifiable
    certified_above.frontier_forwarded = 43;
    certified_above.frontier_accepted = 43;
    certified_above.continuation_forwarded = 44;
    ServingYieldRecordFacts wrong_frontier = clean();
    wrong_frontier.frontier_forwarded = 25;                   // the pair is not at the cut the yield reported
    ServingYieldRecordFacts wrong_accepted = clean();
    wrong_accepted.frontier_accepted = 27;
    ServingYieldRecordFacts no_window = clean();
    no_window.continuation_forwarded = 26;                    // a "continuation" that advanced nothing
    ServingYieldRecordFacts two_windows = clean();
    two_windows.continuation_forwarded = 28;
    ServingYieldRecordFacts unpinned = clean();
    unpinned.identity_pinned = false;
    ServingYieldRecordFacts unbound = clean();
    unbound.ordinal_bound = false;
    const Mutation mutations[] = {
        {"pair missing", no_pair, ServingYieldRecordRefusal::refused_pair_absent},
        {"restoration missing", no_restoration, ServingYieldRecordRefusal::refused_restoration_absent},
        {"restoration reports no restore", unrestored, ServingYieldRecordRefusal::refused_restoration_absent},
        {"comparison cancelled", cancelled, ServingYieldRecordRefusal::refused_cancelled},
        {"comparison incomplete", incomplete, ServingYieldRecordRefusal::refused_not_completed},
        {"payload digests differ", unequal, ServingYieldRecordRefusal::refused_payload_mismatch},
        {"mismatching tensors", mismatching, ServingYieldRecordRefusal::refused_payload_mismatch},
        {"certified frontier not exact", frontier_inexact, ServingYieldRecordRefusal::refused_cut_not_exact},
        {"continuation not exact at its own window", continuation_inexact,
         ServingYieldRecordRefusal::refused_cut_not_exact},
        // The two ways the debt disappears: R reported equal to D (the drain path under new names), and R above D.
        {"zero debt", zero_debt, ServingYieldRecordRefusal::refused_no_debt},
        {"private cut above the published cut", certified_above, ServingYieldRecordRefusal::refused_no_debt},
        // A pair that does not describe the cut the yield reported is not that yield's certificate.
        {"frontier is not the certified cut", wrong_frontier, ServingYieldRecordRefusal::refused_pair_geometry},
        {"accepted is not the certified cut", wrong_accepted, ServingYieldRecordRefusal::refused_pair_geometry},
        {"continuation advanced no window", no_window, ServingYieldRecordRefusal::refused_pair_geometry},
        {"continuation advanced two windows", two_windows,
         ServingYieldRecordRefusal::refused_pair_geometry},
        {"producer identity not pinned", unpinned, ServingYieldRecordRefusal::refused_identity_unpinned},
        {"records not bound to this request's ordinal", unbound,
         ServingYieldRecordRefusal::refused_ordinal_unbound},
    };
    for (const Mutation& row : mutations) {
        check(serving_yield_record_decision(row.facts) == row.expected, row.name);
        check(std::string_view(serving_yield_record_refusal_name(row.expected)) != "unknown",
              "every refusal is named");
    }
    // A refusal never means "hidden": the debt the caller must still report is derived from the two cuts.
    check(serving_yield_record_debt(zero_debt) == 0, "a zero-debt yield reports a zero debt, visibly");
    check(serving_yield_record_debt(certified_above) == 0,
          "a private cut above the published cut is not a debt, it is a contradiction");
    // The fixed vocabulary the harness contract names, spelled once.
    check(std::string_view(kServingYieldProgressPhase) == "retained_yield",
          "the progress phase matches harness YIELD_PROGRESS_PHASE");
    check(std::string_view(kServingYieldFrontierPhase) == "yield_frontier" &&
              std::string_view(kServingYieldContinuationPhase) == "yield_continuation",
          "the oracle pair names match harness YIELD_PHASE_NAMES");
}

// A live, unyielded, debt-carrying owner in the state the token loop leaves behind at a completed response.
struct PreYieldOwner {
    std::shared_ptr<FixtureController> controller;
    std::unique_ptr<FixtureOutcome> owner;
};
PreYieldOwner make_completed_owner() {
    PreYieldOwner made;
    made.controller = std::make_shared<FixtureController>();
    made.controller->charge = std::make_shared<FixtureCharge>();
    made.controller->published_cut = fixture_cut(11, kPublishedForwarded, kPublishedForwarded - 16, 0x31);
    made.controller->live_cut = made.controller->published_cut;
    made.controller->private_cut = fixture_cut(12, kPrivateForwarded, kPrivateForwarded, 0x32);
    made.controller->exact_completion_cut = made.controller->private_cut.id;
    made.owner = std::make_unique<FixtureOutcome>();
    made.owner->operation_credit = std::make_shared<int>(1);
    made.owner->controller = made.controller;
    made.owner->published.key = fixture_key();
    made.owner->published.cut = made.controller->published_cut;
    made.owner->next_cut = 13;
    return made;
}

void test_d1_admitted_response_still_needs_the_controller_s_own_certified_cut() {
    FixtureRegistry registry(kOneSlot);
    const core::WindowKey session_key = fixture_key(5, 2);
    // The coherent case: the serving gate admits the completed response, the controller's own gate certifies its
    // private cut, and the real registry parks it.  That is the whole yield, end to end, through the same
    // production predicates the serving scope calls.
    PreYieldOwner coherent = make_completed_owner();
    check(serving_certified_yield_gate(completed_response(1, 1, "length")) == ServingYieldRefusal::none,
          "the serving gate admits this response");
    std::string error;
    core::StateCut certified{};
    check(coherent.controller->yield_retained(certified, error),
          "the controller certifies its own private cut at the admitted completion");
    check(certified.forwarded == kPrivateForwarded && certified.id == 12,
          "the certified cut is the private checkpoint, not the published one");
    coherent.owner->yielded = true;
    coherent.owner->certified_cut = certified;
    coherent.owner->retained_debt = repair_token_debt(kPublishedForwarded, certified.forwarded);
    check(coherent.owner->retained_debt == kPublishedForwarded - kPrivateForwarded,
          "the retained debt is reported, never hidden");
    const core::WindowKey identity = coherent.owner->published.key;
    check(registry.park_retained_owner(identity, coherent.owner, park_facts(registry, *coherent.controller, session_key),
                                       error) &&
              registry.size() == 1,
          "and the debt-retaining owner is parked instead of drained");
    std::string retire_error;
    check(registry.retire_all("test", FixtureRetire{}, retire_error) && FixtureCharge::live == 0,
          "the parked owner releases its charge through the registry only");
    coherent.controller.reset();

    // Incoherent cuts stay refused by the controller's own evidence even though the serving gate admits the
    // response - the serving gate states no controller fact and cannot substitute for the witness.
    struct Incoherent {
        const char* why;
        RepairOwnerRefusal expected;
    } incoherent[] = {
        {"the live cut moved under the publication", RepairOwnerRefusal::refused_published_cut_divergence},
        {"no committed all-stage witness", RepairOwnerRefusal::refused_no_certified_cut},
        {"a witness for a different cut", RepairOwnerRefusal::refused_certified_cut_not_exact},
        {"the live transaction is not drained", RepairOwnerRefusal::refused_phase_not_drained},
        {"the private transaction is not drained", RepairOwnerRefusal::refused_private_phase_not_drained},
        {"the owner already has zero debt", RepairOwnerRefusal::refused_no_debt},
        {"a second yield of the same owner", RepairOwnerRefusal::refused_already_suspended},
        {"a cancellation reached the controller", RepairOwnerRefusal::refused_cancel_requested},
        {"another operation holds the service", RepairOwnerRefusal::refused_service_not_idle},
        {"this is not the controller thread", RepairOwnerRefusal::refused_wrong_controller_thread},
    };
    for (const Incoherent& row : incoherent) {
        PreYieldOwner owner = make_completed_owner();
        if (std::string(row.why) == "the live cut moved under the publication")
            owner.controller->live_cut = fixture_cut(99, kPublishedForwarded + 4, kPublishedForwarded + 4, 0x33);
        else if (std::string(row.why) == "no committed all-stage witness")
            owner.controller->exact_completion_cut = 0;
        else if (std::string(row.why) == "a witness for a different cut")
            owner.controller->exact_completion_cut = owner.controller->private_cut.id + 1;
        else if (std::string(row.why) == "the live transaction is not drained")
            owner.controller->live_phase = core::serving_raw::Phase::running;
        else if (std::string(row.why) == "the private transaction is not drained")
            owner.controller->private_phase = core::serving_raw::Phase::running;
        else if (std::string(row.why) == "the owner already has zero debt")
            owner.controller->private_cut = fixture_cut(12, kPublishedForwarded, kPublishedForwarded, 0x34);
        else if (std::string(row.why) == "a cancellation reached the controller")
            owner.controller->cancel_requested = true;
        else if (std::string(row.why) == "another operation holds the service")
            owner.controller->service_idle = false;
        else if (std::string(row.why) == "this is not the controller thread")
            owner.controller->controller_thread = false;
        else if (std::string(row.why) == "a second yield of the same owner") {
            core::StateCut first{};
            check(owner.controller->yield_retained(first, error), "the first yield succeeds");
        }
        core::StateCut certified_now{};
        check(!owner.controller->yield_retained(certified_now, error) &&
                  certified_now.id == 0,
              row.why);
        RepairOwnerRefusal why = RepairOwnerRefusal::none;
        core::StateCut gate{};
        check(!retained_yield_allowed(owner.controller->evidence(), why, gate) && why == row.expected,
              "and the refusal is the named one");
        // Every cell retires deliberately before its handle goes away: dropping an unretired controller is the
        // terminal rule this machinery exists to respect (resident_repair_controller.inc:70-71).
        owner.controller->live_phase = core::serving_raw::Phase::drained;
        owner.controller->private_phase = core::serving_raw::Phase::drained;
        std::string retire_now;
        check(owner.controller->retire(retire_now), "the fixture owner is retired deliberately");
        owner.controller.reset();
    }
    check(FixtureController::unretired_drops == 0, "no incoherent-cut cell dropped a live controller");
    check(FixtureCharge::live == 0, "and released every charge it took");
}

// ---- 11. D2 (slice 3): the idle period rechecks demand before EVERY unit --------------------------
void test_d2_idle_period_rechecks_demand_before_every_unit() {
    const uint64_t initial_debt = kPublishedForwarded - kPrivateForwarded;   // 12 fixture tokens
    const uint64_t units_to_close = initial_debt / kFixtureCap;              // 3 units at the fixture cap of 4
    check(initial_debt % kFixtureCap == 0, "the fixture debt divides by the fixture cap");

    // (a) A quiet idle period: units keep being offered, one per pass, until the owner's OWN debt is closed -
    //     no compiled per-period count and no spin: a pass that admits nothing ends the period.
    FixtureRegistry registry(kOneSlot);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, fixture_key(5, 2)), error),
          "park");
    const int unit_calls_before = FixtureController::unit_calls;
    uint64_t passes = 0, units_taken = 0;
    while (true) {
        ++passes;
        const ServingIdleQueueSignal quiet;   // nothing queued, no STOP, no EOF, no input error
        check(!serving_input_demand_observed(quiet), "an idle period has no demand by definition");
        SuspendedOwnerFacts record;
        check(registry.entry_facts(identity, record), "the record is readable");
        if (!serving_repair_idle_admission(quiet_demand(), record.token_debt, true).admitted()) break;
        RepairOwnerEvidence after;
        uint64_t advanced = 0;
        if (!run_unit(registry, identity, idle_session(*made.controller), kFixtureCap, after, advanced, error))
            break;
        units_taken += 1;
        check(advanced == kFixtureCap, "one unit, capped by the selector's number");
    }
    check(units_taken == units_to_close && passes == units_to_close + 1,
          "the period advanced exactly the units the debt allowed and then stopped without any counter");
    check(FixtureController::unit_calls == unit_calls_before + int(units_to_close),
          "the controller ran exactly that many private units");
    SuspendedOwnerFacts drained_record;
    check(registry.entry_facts(identity, drained_record) && drained_record.token_debt == 0 &&
              drained_record.certified_cut.forwarded == kPublishedForwarded,
          "the debt is closed at the published cut, never past it");

    // (b) Input arrives mid-period: the recheck before the NEXT unit sees it, so the period ends having taken
    //     one unit where the debt allowed four more.  A per-pass tick (slice 2) would have taken them all.
    const int unit_calls_at_break = FixtureController::unit_calls;
    uint64_t units_after_input = 0;
    while (true) {
        const ServingIdleQueueSignal signal = [&] {
            ServingIdleQueueSignal s;
            s.queued_input_lines = units_after_input >= 1 ? 1 : 0;   // the line lands after the first unit
            return s;
        }();
        if (serving_input_demand_observed(signal)) break;   // the wake condition, not a queue-depth guess
        ServingIdleDemand demand = quiet_demand(signal.queued_input_lines);
        demand.stop_or_eos_seen = signal.stop_requested;
        SuspendedOwnerFacts record;
        check(registry.entry_facts(identity, record), "the record is readable");
        if (!serving_repair_idle_admission(demand, record.token_debt, true).admitted()) break;
        RepairOwnerEvidence after;
        uint64_t advanced = 0;
        if (!run_unit(registry, identity, idle_session(*made.controller), kFixtureCap, after, advanced, error))
            break;
        units_after_input += 1;
    }
    check(units_after_input == 0 && FixtureController::unit_calls == unit_calls_at_break,
          "a queued line stops the period before any further unit is attempted");

    // (c) The wake condition's own discriminating rows.  A committed STOP never reaches `lines`
    // (generate.cpp:14650), so a queue-depth check alone would keep working through a cancellation.
    ServingIdleQueueSignal stopped;
    stopped.stop_requested = true;
    check(serving_input_demand_observed(stopped), "a committed STOP wakes the idle window with an empty queue");
    ServingIdleQueueSignal eof;
    eof.end_of_input = true;
    check(serving_input_demand_observed(eof), "EOF wakes it");
    ServingIdleQueueSignal failed;
    failed.input_error = true;
    check(serving_input_demand_observed(failed), "an input failure wakes it");
    ServingIdleQueueSignal queued;
    queued.queued_input_lines = 1;
    check(serving_input_demand_observed(queued), "a queued line wakes it");
    check(!serving_input_demand_observed(ServingIdleQueueSignal{}), "and nothing at all does not");
    // The kernel's demand gate is still the hard gate behind it: the same signal mapped into the demand
    // observation refuses the unit for the named reason.
    check(serving_repair_idle_admission(quiet_demand(1), initial_debt, true).reason ==
              RepairAdmissionReason::refused_foreground_demand,
          "the per-unit admission re-derives the refusal from the fresh observation");

    std::string retire_error;
    check(registry.retire_all("test", FixtureRetire{}, retire_error) && registry.size() == 0 &&
              FixtureCharge::live == 0,
          "the parked owner is retired deliberately, releasing its charge");
    made.controller.reset();
}

// ---- 9. selectors, records, and the destructor failstop -------------------------------------------
void test_selectors_have_no_compiled_default() {
    ::unsetenv(kServingRepairOwnerSlotsEnv);
    ::unsetenv(kServingRepairIdleUnitEnv);
    check(serving_repair_owner_slots() == 0 && serving_repair_idle_unit_tokens() == 0,
          "absent selectors mean off, never a default number");
    ::setenv(kServingRepairOwnerSlotsEnv, "0", 1);
    ::setenv(kServingRepairIdleUnitEnv, "0", 1);
    check(serving_repair_owner_slots() == 0 && serving_repair_idle_unit_tokens() == 0, "zero means off");
    ::setenv(kServingRepairOwnerSlotsEnv, "abc", 1);
    ::setenv(kServingRepairIdleUnitEnv, "4x", 1);
    check(serving_repair_owner_slots() == 0 && serving_repair_idle_unit_tokens() == 0,
          "a non-numeric selector is off, never guessed");
    ::setenv(kServingRepairOwnerSlotsEnv, "3", 1);
    ::setenv(kServingRepairIdleUnitEnv, "12", 1);
    check(serving_repair_owner_slots() == 3 && serving_repair_idle_unit_tokens() == 12,
          "the selector numbers are taken literally");
    ::setenv(kServingRepairOwnerSlotsEnv, "-1", 1);
    check(serving_repair_owner_slots() == 0, "a negative selector is off");
    ::unsetenv(kServingRepairOwnerSlotsEnv);
    ::unsetenv(kServingRepairIdleUnitEnv);
}

void test_age_and_identity_records() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    ParkOwnerFacts facts = park_facts(registry, *made.controller, identity);
    facts.observed_at_ns = 12345;
    check(registry.park_retained_owner(identity, made.owner, facts, error), "park with an observation record");
    check(!registry.observe_age(fixture_key(1, 1, "window-none"), RepairAgeInputs{{}, 0, 0, 0, 0, 1}),
          "an unknown window cannot be aged");
    check(registry.observe_age(identity, RepairAgeInputs{kFixtureAgeNs, 100, 2, 0, 17, 12}), "age recorded");
    check(registry.age_of(identity)->oldest_age_ns() == kFixtureAgeNs,
          "the oldest inherited age survives the park");
    check(registry.observe_age(identity, RepairAgeInputs{{}, 0, 0, 0, 18, 11}), "an unknown observation folds in");
    check(registry.age_of(identity)->oldest_age_ns() == kFixtureAgeNs,
          "and never clears the known age (no invented age)");
    check(registry.age_of(identity)->latest_debt() == 11 && registry.age_of(identity)->oldest_debt() == 12,
          "position and debt stay in their own fields");
    SuspendedOwnerFacts record;
    check(registry.entry_facts(identity, record) && record.observed_at_ns == 12345, "observation recorded");
    const std::vector<core::WindowKey> identities = registry.parked_identities();
    check(identities.size() == 1 && identities.front() == identity, "the idle walk sees the parked identity");
    facts = ParkOwnerFacts{};   // the park facts' evidence copy also pins the charge: drop it before the check
    std::string retire_error;
    check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error), "retire");
    check(FixtureCharge::live == 0, "charge released");
    made.controller.reset();
}

void test_two_owners_park_and_survive_each_other() {
    FixtureRegistry registry(kTwoSlots);
    std::string error;
    FixtureOwner first = make_yielded_owner(fixture_key(5, 2, "window-a"));
    const core::WindowKey first_identity = first.owner->published.key;
    check(registry.park_retained_owner(first_identity, first.owner, park_facts(registry, *first.controller, first_identity),
                                       error),
          "park first");
    FixtureOwner second = make_yielded_owner(fixture_key(5, 2, "window-b"));
    const core::WindowKey second_identity = second.owner->published.key;
    check(registry.park_retained_owner(second_identity, second.owner,
                                       park_facts(registry, *second.controller, second_identity), error),
          "park a second window's owner while the first stays parked");
    check(registry.size() == 2, "both owners are retained");
    check(!registry.park_retained_owner(fixture_key(5, 2, "window-c"), second.owner, ParkOwnerFacts{}, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_empty_owner_slot,
          "a moved-from slot is refused: the payload already belongs to the registry");
    const std::vector<core::WindowKey> identities = registry.parked_identities();
    check(identities.size() == 2 && identities[0] == first_identity && identities[1] == second_identity,
          "the walk is in entry order");
    std::string retire_error;
    check(registry.retire_all("test", FixtureRetire{}, retire_error) && registry.size() == 0, "drain both");
    check(FixtureCharge::live == 0, "and release both charges");
    first.controller.reset();
    second.controller.reset();
}

// ---- 8. slice 5: the versioned retained RESULT, its owner-record publication, its activation ---------
// These cells drive the production functions the serving path calls: `make_prepared_retained_result` (the maker
// the controller's `prepare_retained_result` routes through), `retained_result_publishable` and
// `retained_activation_allowed` (the gates the controller's `publish_retained_if_current` /
// `activate_retained` call), and the registry's `reserve_retained_capacity` / `park_retained_owner` /
// `publish_retained_if_current` / `activate_parked_owner`. What is still fixture-only is the controller body
// itself and the device capture/restore: `resident_repair_controller.inc` and `generate.cpp` do not compile on
// this host (cuda_runtime.h is absent), so the CUDA build and the runtime A/B/A capture remain lead steps.

// The live session as the returning conversation holds it: an open request fence at a drained all-stage cut.
LiveSessionFacts activation_session(const FixtureController& controller, uint64_t residency_generation = 0,
                                    core::serving_raw::Phase live_phase = core::serving_raw::Phase::drained,
                                    const core::WindowKey& session_key = fixture_key(5, 2)) {
    LiveSessionFacts session;
    session.session_key = session_key;
    session.session_cut = controller.live_cut;
    session.live_phase = live_phase;
    session.request_open = true;
    session.private_window_open = false;
    session.residency_generation = residency_generation;
    return session;
}

// The other conversation's state, exactly as a capture records it: the live session's own key and cut.
ConversationStateFacts captured_prior_state(const FixtureController& controller) {
    ConversationStateFacts prior;
    prior.key = controller.published_key;
    prior.cut = controller.live_cut;
    prior.session_cut = controller.live_cut;
    return prior;
}

// Park an owner, advance it to a chosen private position, prepare its result and publish it onto the record.
// Every step goes through the production registry function, so a cell cannot smuggle a published result in.
PreparedRetainedResult publish_result_of(FixtureRegistry& registry, FixtureController& controller,
                                         const core::WindowKey& identity, std::string& error) {
    SuspendedOwnerFacts record;
    check(registry.entry_facts(identity, record), "the parked owner has a record");
    const PreparedRetainedResult result = controller.prepare_retained_result(record.record_version, error);
    check(result.produced(), "the controller can state a retained result about itself");
    PreparedRetainedResult published;
    check(registry.publish_retained_if_current(identity, result, published, error),
          "publish-if-current onto the owner record");
    return published;
}

void test_prepared_retained_result_is_refused_when_it_is_not_a_certified_chain_cut() {
    // The maker is the only route to a result, and it refuses every shape that would let an owner activate a
    // state its controller never certified. Built from a real controller's own fields, then broken one way each.
    FixtureOwner made = make_yielded_owner();
    std::string error;
    PreparedRetainedResultFacts facts;
    {
        SuspendedOwnerFacts unused;   // the fixture's own fields are the source, as on the controller
        (void)unused;
        const FixtureController& c = *made.controller;
        facts.identity = c.published_key;
        facts.published_serial = c.published_serial;
        facts.record_version = 1;
        facts.published_cut = c.published_cut;
        facts.certified_cut = c.private_cut;
        facts.accepted_sequence = c.private_cut.accepted;
        facts.pending_token = int32_t(c.published_cut.forwarded + 1);
        facts.output_watermark = c.published_cut.output_watermark;
        facts.witness_committed = true;
        facts.holds_frozen_image = true;
    }
    RepairOwnerRefusal why = RepairOwnerRefusal::none;
    check(make_prepared_retained_result(facts, why).produced(), "a certified chain cut is accepted");
    const PreparedRetainedResultFacts clean = facts;   // each row breaks exactly one rule of this shape
    auto row = [&clean, &why](const PreparedRetainedResultFacts& changed, RepairOwnerRefusal expected,
                              const char* what) {
        const PreparedRetainedResult result = make_prepared_retained_result(changed, why);
        check(!result.produced() && why == expected, what);
        check(make_prepared_retained_result(clean, why).produced(), "the unbroken shape still produces: one row, one rule");
    };
    PreparedRetainedResultFacts base = clean;
    base.witness_committed = false;
    row(base, RepairOwnerRefusal::refused_no_certified_cut, "no committed all-stage witness: no result");
    base = clean;
    base.holds_frozen_image = false;
    row(base, RepairOwnerRefusal::refused_no_retained_charge,
        "no retained image of the certified cut: no result");
    base = clean;
    base.certified_cut.exact_frontier = base.certified_cut.forwarded - 4;
    row(base, RepairOwnerRefusal::refused_certified_cut_not_exact, "a chain cut that is not exact is refused");
    base = clean;
    base.certified_cut.accepted = base.certified_cut.forwarded + 2;
    row(base, RepairOwnerRefusal::refused_prepared_result_invalid,
        "a chain cut carrying more than one pending token is not a shape the chain can be in");
    // A checked pending-history sealing may retain the genuine pending token. This is distinct from ordinary
    // private replay, whose consumed-only certificate remains truthful even when its forwarded debt reaches0.
    PreparedRetainedResultFacts caught = clean;
    caught.certified_cut.forwarded = caught.published_cut.forwarded;   // chain caught the publication
    caught.certified_cut.exact_frontier = caught.certified_cut.forwarded;
    caught.certified_cut.accepted = caught.certified_cut.forwarded + 1;
    caught.accepted_sequence = caught.certified_cut.accepted;
    check(make_prepared_retained_result(caught, why).produced(),
          "a chain cut that has caught the publication (one pending token) is a valid prepared result");
    base = clean;
    base.certified_cut.forwarded = base.published_cut.forwarded;
    base.certified_cut.accepted = base.certified_cut.forwarded;
    base.certified_cut.exact_frontier = base.certified_cut.forwarded;
    base.accepted_sequence = base.certified_cut.accepted;
    check(make_prepared_retained_result(base,why).produced(),
          "zero-debt consumed-only private checkpoint is valid; genuine pending token stays in publication");
    base = clean;
    base.certified_cut.forwarded = base.published_cut.forwarded + 1;
    base.certified_cut.accepted = base.certified_cut.forwarded;
    base.certified_cut.exact_frontier = base.certified_cut.forwarded;
    base.accepted_sequence = base.certified_cut.accepted;
    row(base, RepairOwnerRefusal::refused_debt_inconsistent, "private progress past the published cut is refused");
    base = clean;
    base.output_watermark = base.published_cut.output_watermark + 1;
    row(base, RepairOwnerRefusal::refused_prepared_result_invalid,
        "a watermark the chain does not carry (advance_private_to copies it) is refused");
    base = clean;
    base.pending_token = -1;
    row(base, RepairOwnerRefusal::refused_prepared_result_invalid, "no pending token, no activatable state");
    base = clean;
    base.accepted_sequence = base.certified_cut.accepted + 1;
    row(base, RepairOwnerRefusal::refused_prepared_result_invalid,
        "an accepted sequence the certified cut does not back is refused");
    base = clean;
    base.record_version = 0;
    row(base, RepairOwnerRefusal::refused_prepared_result_invalid,
        "a result prepared against no record version is refused");
    base = clean;
    base.published_cut.history = core::StateDigest{};
    row(base, RepairOwnerRefusal::refused_prepared_result_invalid, "a publication with no history digest");
    retire_release(made);
}

void test_publish_if_current_is_accepted_for_the_current_record_and_refused_otherwise() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity),
                                       error),
          "park with its up-front capacity reservation");
    FixtureController& controller = *made.controller;
    const core::StateCut live_before = controller.live_cut;
    const core::StateCut published_before = controller.published_cut;

    SuspendedOwnerFacts record;
    check(registry.entry_facts(identity, record) && record.record_version == 1 && !record.result_published &&
              record.capacity_reserved && record.activation_resources_pinned,
          "the record starts at version 1, with the reservation and the activation charges pinned");
    const PreparedRetainedResult result = publish_result_of(registry, controller, identity, error);
    check(registry.entry_facts(identity, record) && record.result_published && record.record_version == 1,
          "the published result is on the OWNER RECORD; the version tracks the chain, not the publication");
    check(controller.live_cut == live_before && controller.published_cut == published_before &&
              registry.size() == 1 && controller.suspended,
          "publication touched the record only: the live session and the owner's publication are untouched");
    check(record.published_result.produced() && record.published_result.certified_cut() == controller.private_cut,
          "the record carries the certified cut the controller stands on");

    // A result prepared before the chain moves cannot be published once the record version has moved.
    const PreparedRetainedResult stale = controller.prepare_retained_result(1, error);
    PreparedRetainedResult ignored;
    RepairOwnerEvidence after;
    uint64_t advanced = 0;
    LiveSessionFacts idle = idle_session(controller);
    idle.residency_generation = controller.residency_generation;
    check(run_unit(registry, identity, idle, kFixtureCap, after, advanced, error) && advanced > 0,
          "the parked owner is still advanceable after a publication");
    check(registry.entry_facts(identity, record) && !record.result_published && record.record_version == 2,
          "advancing the chain drops the older published result and moves the version");
    check(!registry.publish_retained_if_current(identity, stale, ignored, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_record_version_stale,
          "a stale record version refuses publication; nothing was published");
    check(registry.entry_facts(identity, record) && record.record_version == 2 && !record.result_published,
          "and the record is exactly as it was");
    check(!registry.publish_retained_if_current(identity, stale, ignored, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_record_version_stale,
          "the pre-unit result stays refused");
    check(!registry.publish_retained_if_current(identity, PreparedRetainedResult{}, ignored, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_prepared_result_absent,
          "no result at all is refused, never assumed");

    // The controller-side refusals, through the registry (which re-reads the controller's own evidence).
    const PreparedRetainedResult current = controller.prepare_retained_result(2, error);
    controller.cancel_requested = true;
    check(!registry.publish_retained_if_current(identity, current, ignored, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_cancel_requested,
          "a cancelled owner gains no published result");
    controller.cancel_requested = false;
    check(registry.publish_retained_if_current(identity, current, ignored, error),
          "and the same owner is publishable again once the cancellation is not standing: a refusal never poisoned "
          "or dropped it");
    controller.private_phase = core::serving_raw::Phase::running;
    check(!registry.publish_retained_if_current(identity, current, ignored, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_private_phase_not_drained,
          "a private transaction that is not quiescent refuses publication");
    controller.private_phase = core::serving_raw::Phase::drained;
    controller.live_phase = core::serving_raw::Phase::capturing;
    check(!registry.publish_retained_if_current(identity, current, ignored, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_phase_not_drained,
          "a live fence mid-capture refuses publication");
    controller.live_phase = core::serving_raw::Phase::drained;
    controller.private_healthy = false;
    check(!registry.publish_retained_if_current(identity, current, ignored, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_allocation_unhealthy,
          "an unhealthy allocation refuses publication");
    controller.private_healthy = true;
    check(registry.entry_facts(identity, record) && record.state == RepairOwnerState::suspended &&
              !record.quarantined && record.holds_live_controller,
          "every refusal above left the owner parked, verifiable and resumable");

    // The gate's own identity and content rules, on the production predicate the controller calls: a record for
    // another window, and an edit that changed the accepted sequence under the owner.
    RepairOwnerRefusal why = RepairOwnerRefusal::none;
    RetainedRecordFacts gate_record;
    gate_record.identity = identity;
    gate_record.published_serial = record.published_serial;
    gate_record.record_version = record.record_version;
    gate_record.token_debt = record.token_debt;
    gate_record.published_cut = record.published_cut;
    gate_record.certified_cut = record.certified_cut;
    gate_record.versions = record.versions;
    gate_record.capacity_reserved = true;
    gate_record.resources_pinned = true;
    const PreparedRetainedResult fresh = controller.prepare_retained_result(record.record_version, error);
    check(retained_result_publishable(fresh, controller.evidence(), gate_record, why) &&
              why == RepairOwnerRefusal::none,
          "the production gate accepts the current version");
    RetainedRecordFacts foreign = gate_record;
    foreign.identity = fixture_key(5, 2, "window-z");
    check(!retained_result_publishable(fresh, controller.evidence(), foreign, why) &&
              why == RepairOwnerRefusal::refused_identity_mismatch,
          "identity mismatch refuses publication");
    RetainedRecordFacts edited = gate_record;
    edited.versions.content = fixture_digest(0x77);   // the accepted sequence changed under the owner
    check(!retained_result_publishable(fresh, controller.evidence(), edited, why) &&
              why == RepairOwnerRefusal::refused_content_diverged,
          "a stale edit (content version moved) refuses publication");
    RetainedRecordFacts selected = gate_record;
    ++selected.versions.selection_epoch;
    check(!retained_result_publishable(fresh, controller.evidence(), selected, why) &&
              why == RepairOwnerRefusal::refused_content_diverged,
          "a selection-epoch change refuses publication too");

    // The evidence copies taken above hold a reference to the gang lease; dropping them is what releases the
    // entry's last pin, so the leak check below means "no charge object survives", not "no shared_ptr does".
    after = RepairOwnerEvidence{};
    std::string retire_error;
    check(registry.retire_owner(identity, "test", FixtureRetire{}, retire_error) && registry.size() == 0,
          "the owner is released through the controller's own retire");
    check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0,
          "the gang lease and all three activation charges are released exactly once");
    made.controller.reset();
}

void test_activation_lands_only_at_a_drained_boundary_and_restores_the_other_conversation() {
    FixtureRegistry registry(kTwoSlots);
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    std::string error;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity),
                                       error),
          "park");
    FixtureController& controller = *made.controller;
    const PreparedRetainedResult published = publish_result_of(registry, controller, identity, error);
    check(published.produced(), "the record carries a published result before any activation is attempted");

    const ConversationStateFacts prior = captured_prior_state(controller);
    ConversationStateFacts restored;
    RepairOwnerEvidence after;
    std::vector<OwnerReservationHandle> charges;
    std::unique_ptr<FixtureOutcome> returned;
    auto attempt = [](FixtureOutcome& payload, const RetainedActivationFacts&, RepairOwnerEvidence& after_evidence,
                      ConversationStateFacts& restored_out, std::string& unit_error) {
        return payload.controller->activate_retained(captured_prior_state(*payload.controller), restored_out,
                                                     after_evidence, unit_error);
    };
    // (1) Not at a drained boundary: the live fence is working, so nothing is admitted and nothing mutates.
    check(!registry.activate_parked_owner(identity, activation_session(controller, 0,
                                                                       core::serving_raw::Phase::running),
                                          prior, attempt, returned, restored, after, charges, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_activation_not_drained,
          "activation is refused when the live session is not at a drained all-stage boundary");
    // (2) No request fence open to capture/restore under.
    LiveSessionFacts closed_fence = activation_session(controller);
    closed_fence.request_open = false;
    check(!registry.activate_parked_owner(identity, closed_fence, prior, attempt, returned, restored, after, charges,
                                          error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_activation_fence_not_open,
          "an activation needs the returning conversation's open fence at the drained cut");
    // (3) A bounded private work window is not an activation boundary.
    LiveSessionFacts windowed = activation_session(controller);
    windowed.private_window_open = true;
    check(!registry.activate_parked_owner(identity, windowed, prior, attempt, returned, restored, after, charges, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_private_window_not_open,
          "no activation inside a private work window");
    // (4) The residency profile: the chain is bound to one expert-residency inventory generation.
    check(!registry.activate_parked_owner(identity, activation_session(controller, 9), prior, attempt, returned, restored,
                                          after, charges, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_residency_diverged,
          "a residency change refuses the activation by name (no rebind is implemented in this pass)");
    // (5) A replacement that is not accounted for: the session moved on and nothing was captured there.
    LiveSessionFacts moved = activation_session(controller);
    moved.session_cut = fixture_cut(91, kPublishedForwarded + 6, kPublishedForwarded + 6, 0x93);
    controller.live_cut = moved.session_cut;
    const ConversationStateFacts uncaptured = prior;   // captured at the OLD session cut
    check(!registry.activate_parked_owner(identity, moved, uncaptured, attempt, returned, restored, after, charges, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_stale_publication,
          "replacing an uncaptured live state is refused as a stale publication");
    controller.live_cut = controller.published_cut;
    // (6) No result published: there is nothing to activate.
    SuspendedOwnerFacts record;
    check(registry.entry_facts(identity, record) && record.result_published, "the record has the result");
    // (6b) The registry derives the activation facts itself: a caller cannot describe a boundary it does not have.
    // (7) An open debt is refused at the boundary gate: activation lands the certified state at the position the
    // live conversation already stands on, so a chain that has not caught the published cut cannot be activated and
    // the controller's own activation never runs (`after` stays the invalid default the registry cleared).
    check(!registry.activate_parked_owner(identity, activation_session(controller), prior, attempt, returned, restored,
                                          after, charges, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_activation_state_diverged && !after.produced(),
          "a mid-debt chain is refused at the boundary, before any mutation, rather than rewinding the request");
    check(controller.suspended && controller.published_serial == kFixtureSerial && controller.private_cut.id == 12,
          "the refused boundary moved nothing: the owner is still parked on the chain it certified");
    // A stays resumable across every refusal above: the bounded idle units close its debt. The unit that reaches
    // the published cut lands the chain in the one shape an activation can land (zero debt, pending token).
    uint64_t advanced = 0;
    RepairOwnerEvidence evidence_after;
    for (uint64_t units = 1; controller.private_cut.forwarded < controller.published_cut.forwarded; ++units) {
        LiveSessionFacts idle = idle_session(controller);
        idle.residency_generation = controller.residency_generation;
        check(units <= 8 && run_unit(registry, identity, idle, kFixtureCap, evidence_after, advanced, error) &&
                  advanced > 0 &&
                  controller.private_cut.forwarded <= controller.published_cut.forwarded,
              "and it can still be advanced by bounded idle units afterwards, up to the published cut");
    }
    check(registry.entry_facts(identity, record) && record.token_debt == 0 && record.certified_cut.id != 12,
          "the debt is closed at the published cut and the record certifies the chain that closed it");
    after = RepairOwnerEvidence{};
    charges.clear();
    // (7b) The controller's own refusal, reached at a boundary the gate admits: the prior state of the OTHER
    // conversation comes back field by field, and A stays parked, charged and resumable.
    controller.activation_denied = true;
    publish_result_of(registry, controller, identity, error);
    const ConversationStateFacts other = captured_prior_state(controller);
    check(!registry.activate_parked_owner(identity, activation_session(controller), other, attempt, returned, restored, after,
                                          charges, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_private_unit &&
              error.find("the controller refused the activation") != std::string::npos,
          "the controller's own refusal is reported, not hidden: the named refusal carries the controller's words, "
          "which is only reachable once the attempt actually ran");
    check(ConversationStateFacts::identical(other, restored),
          "the other conversation's state is restored exactly: cut id, forwarded, accepted, exact_frontier, "
          "watermark, history digest and session cut");
    check(registry.contains(identity) && registry.entry_facts(identity, record) &&
              record.state == RepairOwnerState::suspended && !record.quarantined && record.holds_live_controller,
          "A stays parked, resumable and charged after the refusal");
    controller.activation_denied = false;
    // (8) The admitted activation: A's published state becomes the live session state. The unit above moved the
    // chain, so the result that activation publishes is the one prepared against the moved record.
    const PreparedRetainedResult landed_result = publish_result_of(registry, controller, identity, error);
    const ConversationStateFacts before_activation = captured_prior_state(controller);
    check(registry.activate_parked_owner(identity, activation_session(controller), before_activation, attempt,
                                         returned, restored, after, charges, error),
          "activation is admitted at the drained all-stage boundary");
    check(registry.size() == 0 && registry.activations_completed() == 1,
          "the owner leaves the registry: the serving scope owns a live controller again and must retire it");
    check(after.produced() && !after.suspended() && after.live_coherent() && after.token_debt() == 0 &&
              after.published_serial() == kFixtureSerial &&
              after.published_cut() == landed_result.published_cut() &&
              after.published_cut() == after.live_cut() &&
              after.private_cut().forwarded == landed_result.certified_cut().forwarded &&
              landed_result.certified_cut() != published.certified_cut(),
          "the activated state IS the certified result at R: live == published (an activation does not rewrite the "
          "publication or its serial), the caught-up chain stands at that same position, debt closed");
    check(ConversationStateFacts::identical(before_activation, restored),
          "the prior conversation state the activation replaced is captured and reported field by field");
    check(charges.size() == 4 && charges[0].purpose == core::ReservationPurpose::private_work,
          "the gang lease and the three activation charges move to the caller: nothing is released under a live "
          "owner");
    check(FixtureCharge::live == 1 && FixtureResourceCharge::live == 3, "and every charge is still alive");
    // The serving scope retires it deliberately (the same rule the request scope guard enforces).
    std::string retire_error;
    check(bool(returned), "the activated payload came back to the serving scope: it is a live controller again");
    check(FixtureRetire{}(*returned, retire_error) && returned->controller == nullptr,
          "the reactivated owner retires through the controller's own retire()");
    returned.reset();
    // Every evidence copy taken above pins the gang lease; dropping them is what releases the last reference.
    after = RepairOwnerEvidence{};
    evidence_after = RepairOwnerEvidence{};
    charges.clear();
    check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0,
          "every charge is released exactly once; nothing leaked");
    made.controller.reset();
}

void test_capacity_is_admitted_before_the_yield_and_a_yield_without_it_is_refused() {
    std::string error;
    {   // The yield gate itself refuses without the up-front admission (requirement 4).
        ServingYieldGateFacts facts = completed_response(8, 8, "length");
        facts.activation_capacity_reserved = false;
        check(serving_certified_yield_gate(facts) == ServingYieldRefusal::refused_activation_capacity,
              "no up-front slot + capture/restore/return admission: the yield is refused, the drain runs");
        facts.activation_capacity_reserved = true;
        check(serving_certified_yield_gate(facts) == ServingYieldRefusal::none,
              "with the reservation admitted, the same completed response may yield");
    }
    {   // The registry refuses an unusable reservation before anything is parked.
        FixtureRegistry registry(kOneSlot);
        RetainedCapacityTicket ticket;
        RetainedActivationResources wrong = fixture_resources();
        wrong.prior_capture_purpose = core::ReservationPurpose::private_work;   // not the lease's real purpose
        check(!registry.reserve_retained_capacity(fixture_key(), wrong, ticket, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_activation_resources_invalid,
              "a mis-purposed capture/restore/return charge is refused: no new ReservationPurpose is invented");
        wrong = fixture_resources();
        wrong.prior_capture_charge.reset();
        check(!registry.reserve_retained_capacity(fixture_key(), wrong, ticket, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_activation_resources_invalid,
              "a missing return resource is refused up front, not discovered at activation time");
        check(!ticket.issued(), "and no ticket was issued for a refused reservation");
        wrong = RetainedActivationResources{};   // the caller's own copies go; the registry pinned nothing
        check(FixtureResourceCharge::live == 0, "a refused reservation pins nothing");
        check(registry.reserve_retained_capacity(fixture_key(), fixture_resources(), ticket, error) &&
                  ticket.issued() && registry.outstanding_reservations() == 1,
              "the reservation is admitted before the yield");
        // A ticket bound to another window authorises no park here, and an unissued/spent ticket none either.
        RetainedCapacityTicket other;
        check(registry.reserve_retained_capacity(fixture_key(5, 2, "window-e"), fixture_resources(), other, error)
                  ? true
                  : registry.last_refusal() == RepairOwnerRefusal::refused_full,
              "a second reservation is bounded by the single slot");
        // A second reservation cannot over-subscribe the single slot: this is the bounded-capacity rule.
        RetainedCapacityTicket second;
        check(!registry.reserve_retained_capacity(fixture_key(5, 2, "window-d"), fixture_resources(), second,
                                                  error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_full,
              "reservations plus entries never exceed the slot bound: no unbounded growth");
        // Parking without the ticket is refused, so an owner can never be retained without its activation.
        FixtureOwner made = make_yielded_owner();
        const core::WindowKey identity = made.owner->published.key;
        ParkOwnerFacts unreserved = park_facts(registry, *made.controller, identity);
        unreserved.capacity = RetainedCapacityTicket{};
        check(!registry.park_retained_owner(identity, made.owner, unreserved, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_capacity_unreserved,
              "a park with no capacity ticket is refused: a park whose activation could never be admitted is a "
              "defect, so it is not representable");
        check(bool(made.owner), "and the caller keeps the owner (it is retired deliberately, never dropped)");
        // Releasing an outstanding reservation returns the slot for a later response, which is what a refused
        // controller-side yield must do before the drain path runs.
        check(registry.release_retained_capacity(ticket, error) && registry.outstanding_reservations() == 0,
              "an unused reservation is released");
        unreserved = ParkOwnerFacts{};   // the caller's copies of the refused park's pins go away with it
        RetainedCapacityTicket spare;
        check(registry.reserve_retained_capacity(fixture_key(5, 2, "window-f"), fixture_resources(), spare, error),
              "the released slot is available for a later reservation");
        check(registry.outstanding_reservations() == 1, "and the bound counts only what is outstanding");
        check(registry.release_retained_capacity(spare, error) && registry.outstanding_reservations() == 0,
              "it is released again");
        spare = RetainedCapacityTicket{};
        retire_release(made);
        check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0,
              "a refused park pins nothing: no gang lease and no activation charge survives it");
    }
    {   // A ticket is bound to the window it reserved: park A with B's ticket and the park is refused.
        FixtureRegistry registry(kTwoSlots);
        FixtureOwner made = make_yielded_owner();
        const core::WindowKey identity = made.owner->published.key;
        RetainedCapacityTicket for_other;
        check(registry.reserve_retained_capacity(fixture_key(5, 2, "window-e"), fixture_resources(), for_other,
                                                 error),
              "reserve for a window that will not be parked");
        ParkOwnerFacts facts = park_facts(registry, *made.controller, identity);
        facts.capacity = for_other;
        check(!registry.park_retained_owner(identity, made.owner, facts, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_capacity_unreserved,
              "a ticket issued for another window cannot park this owner");
        check(bool(made.owner), "and the owner is still the caller's to retire");
        facts = ParkOwnerFacts{};   // the caller's copies of the pins it never got to publish
        retire_release(made);
        check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0,
              "a park refused for its ticket pins nothing: the registry took no lease and no activation charge, and "
              "the caller still has the owner to retire");
    }
    {   // A parked owner without any activation admission cannot be activated.
        FixtureRegistry registry(kOneSlot);
        FixtureOwner made = make_yielded_owner();
        const core::WindowKey identity = made.owner->published.key;
        check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity),
                                           error),
              "park with its reservation");
        FixtureController& controller = *made.controller;
        publish_result_of(registry, controller, identity, error);
        ConversationStateFacts restored;
        RepairOwnerEvidence after;
        std::vector<OwnerReservationHandle> charges;
        std::unique_ptr<FixtureOutcome> returned;
        check(!registry.activate_parked_owner(fixture_key(5, 2, "window-none"), activation_session(controller),
                                              captured_prior_state(controller),
                                              [](FixtureOutcome&, const RetainedActivationFacts&,
                                                 RepairOwnerEvidence&, ConversationStateFacts&, std::string&) {
                                                  return true;
                                              },
                                              returned, restored, after, charges, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_no_entry,
              "no owner, no activation");
        check(!registry.activate_parked_owner(identity, idle_session(controller),
                                              captured_prior_state(controller),
                                              [](FixtureOutcome&, const RetainedActivationFacts&,
                                                 RepairOwnerEvidence&, ConversationStateFacts&, std::string&) {
                                                  return true;
                                              },
                                              returned, restored, after, charges, error) &&
                  registry.last_refusal() == RepairOwnerRefusal::refused_activation_fence_not_open,
              "an attempt cannot smuggle a boundary: the registry derives the activation facts itself");
        std::string retire_error;
        check(registry.retire_all("test", FixtureRetire{}, retire_error) && registry.size() == 0, "drain");
        check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0, "and release everything");
        made.controller.reset();
    }
}

void test_a_parked_owner_survives_a_second_conversation_and_is_activated_afterwards() {
    // The A->B->A seam at the registry level: A is parked, B's turns advance the live session (and a second
    // parked owner of another window advances too), and A stays eligible - advanceable, publishable, and finally
    // activatable at a drained boundary, with B's captured state accounted for.
    FixtureRegistry registry(kTwoSlots);
    std::string error;
    FixtureOwner owner_a = make_yielded_owner(fixture_key(5, 2, "window-a"));
    const core::WindowKey identity_a = owner_a.owner->published.key;
    check(registry.park_retained_owner(identity_a, owner_a.owner,
                                       park_facts(registry, *owner_a.controller, identity_a), error),
          "park A");
    FixtureOwner owner_b = make_yielded_owner(fixture_key(5, 2, "window-b"));
    const core::WindowKey identity_b = owner_b.owner->published.key;
    check(registry.park_retained_owner(identity_b, owner_b.owner,
                                       park_facts(registry, *owner_b.controller, identity_b), error),
          "park B's owner too (two conversations, two slots)");
    check(registry.size() == 2 && FixtureResourceCharge::live == 6, "both owners are retained and charged");

    FixtureController& a = *owner_a.controller;
    // B's turns: a bounded unit for B's owner, and the live session moves on B's conversation.
    RepairOwnerEvidence after_b;
    uint64_t advanced_b = 0;
    LiveSessionFacts idle_b = idle_session(*owner_b.controller, true, identity_b);
    idle_b.residency_generation = owner_b.controller->residency_generation;
    check(run_unit(registry, identity_b, idle_b, kFixtureCap, after_b, advanced_b, error) && advanced_b > 0,
          "B's owner advances");
    const core::StateCut b_live = fixture_cut(80, kPublishedForwarded + 4, kPublishedForwarded + 4, 0x81);
    a.live_cut = b_live;   // the shared live session now stands on a later state than A's publication

    // A is still eligible: same stable window, same residency generation, its own chain untouched.
    RepairOwnerEvidence after_a;
    uint64_t advanced_a = 0;
    LiveSessionFacts idle_a = idle_session(a, true, identity_a);
    idle_a.residency_generation = a.residency_generation;
    check(run_unit(registry, identity_a, idle_a, kFixtureCap, after_a, advanced_a, error) && advanced_a > 0,
          "A remains resumable after B's turns: the exact eligibility condition is the stable window identity "
          "plus an unchanged expert-residency generation");
    SuspendedOwnerFacts record_a;
    check(registry.entry_facts(identity_a, record_a) && record_a.token_debt > 0,
          "A's debt is still open and reported");
    // A residency change, by contrast, is refused by name and strands nothing that was not already stranded: the
    // owner stays parked, charged and retireable.
    LiveSessionFacts swapped = idle_session(a, true, identity_a);
    swapped.residency_generation = a.residency_generation + 5;   // an expert-residency swap was published
    check(!run_unit(registry, identity_a, swapped, kFixtureCap, after_a, advanced_a, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_residency_diverged,
          "the idle unit is refused when the live residency generation is not the chain's baseline");
    check(registry.contains(identity_a) && registry.entry_facts(identity_a, record_a) && !record_a.quarantined,
          "and the refused owner stays parked and resumable (a rebind of the private chain is not implemented)");
    // The residency refusal left A advanceable, and an activation can only land a chain that has caught the
    // published cut, so bounded units close the rest of A's debt before the activation is attempted.
    for (uint64_t units = 1; a.private_cut.forwarded < a.published_cut.forwarded; ++units) {
        LiveSessionFacts catch_up = idle_session(a, true, identity_a);
        catch_up.residency_generation = a.residency_generation;
        check(units <= 8 && run_unit(registry, identity_a, catch_up, kFixtureCap, after_a, advanced_a, error) &&
                  advanced_a > 0 && a.private_cut.forwarded <= a.published_cut.forwarded,
              "the residency refusal left A advanceable: bounded idle units close its debt");
    }

    // Publish A's repaired result onto its record, then activate it at the boundary B's conversation leaves.
    const PreparedRetainedResult published = publish_result_of(registry, a, identity_a, error);
    ConversationStateFacts restored;
    RepairOwnerEvidence after;
    std::vector<OwnerReservationHandle> charges;
    std::unique_ptr<FixtureOutcome> returned;
    auto attempt = [](FixtureOutcome& payload, const RetainedActivationFacts&, RepairOwnerEvidence& after_evidence,
                      ConversationStateFacts& restored_out, std::string& unit_error) {
        return payload.controller->activate_retained(captured_prior_state(*payload.controller), restored_out,
                                                     after_evidence, unit_error);
    };
    const ConversationStateFacts stale_capture = captured_prior_state(a);   // captured at B's live cut
    LiveSessionFacts boundary = activation_session(a, a.residency_generation,
                                                   core::serving_raw::Phase::drained, identity_a);
    boundary.session_cut = a.live_cut;
    a.published_key = identity_a;
    const ConversationStateFacts prior_b = stale_capture;
    check(registry.activate_parked_owner(identity_a, boundary, prior_b, attempt, returned, restored, after,
                                         charges, error),
          "A's activation at a drained all-stage boundary replaces B's captured state, which the entry's "
          "foreground_return charge retains");
    check(registry.size() == 1 && registry.contains(identity_b) && registry.activations_completed() == 1,
          "A leaves the registry; B's owner is untouched");
    check(!after.suspended() && after.live_coherent() && after.token_debt() == 0 &&
              after.published_cut() == published.published_cut() &&
              after.private_cut().forwarded == published.certified_cut().forwarded,
          "the live session now stands on A's publication, at the position A's certified result reached");
    check(FixtureResourceCharge::live == 6,
          "B's owner keeps its three charges and A's three move to the serving scope: nothing was freed under a "
          "live owner");
    after = RepairOwnerEvidence{};
    after_a = RepairOwnerEvidence{};
    after_b = RepairOwnerEvidence{};
    charges.clear();
    std::string retire_error;
    check(registry.retire_all("test", FixtureRetire{}, retire_error) && registry.size() == 0, "drain B's owner");
    check(bool(returned) && FixtureRetire{}(*returned, retire_error),
          "and retire A deliberately through its controller: the serving scope owns it after activation");
    returned.reset();
    check(!bool(owner_a.owner), "the registry moved A's payload out when it was parked");
    check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0, "every charge is released exactly once");
    owner_a.controller.reset();
    owner_b.controller.reset();
}

void test_the_registry_prepares_and_publishes_the_parked_owner_s_own_result() {
    // Slice 6 blocker 1: after a park there is no caller that can hand the registry a result - `park_retained_owner`
    // moved the payload out of the request slot, so the scope holds neither the controller nor its frozen image.
    // `publish_retained_result_from_owner` asks the PAYLOAD to state its own result, with the record version the
    // registry owns, and runs the same publish gate over what the controller said. The fixture keeps its own
    // controller handle for convenience; production reaches the payload only through the entry, and that is the
    // route this cell drives.
    FixtureRegistry registry(kOneSlot);
    std::string error;
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "park the owner");
    check(!bool(made.owner), "the registry owns the payload: the request scope has nothing to prepare from");
    PreparedRetainedResult smuggled{}, published{};
    check(!registry.publish_retained_if_current(identity, smuggled, published, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_prepared_result_absent,
          "the caller-supplied form still refuses a result the scope cannot hold");
    std::unique_ptr<FixtureOutcome> returned;
    ConversationStateFacts restored;
    RepairOwnerEvidence after;
    std::vector<OwnerReservationHandle> charges;
    check(!registry.activate_parked_owner(identity, activation_session(*made.controller, 0,
                                                                      core::serving_raw::Phase::drained, identity),
                                          captured_prior_state(*made.controller),
                                          [](FixtureOutcome&, const RetainedActivationFacts&,
                                             RepairOwnerEvidence&, ConversationStateFacts&, std::string&) {
                                              return true;
                                          },
                                          returned, restored, after, charges, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_prepared_result_absent,
          "activation refuses for want of a published result, before the attempt ever sees the payload");
    // A prepare the payload itself refuses publishes nothing and moves nothing.
    check(!registry.publish_retained_result_from_owner(
              identity,
              [](FixtureOutcome&, uint64_t, PreparedRetainedResult& result, std::string& why) {
                  result = {}; why = "the controller holds no witness for its own cut"; return false;
              },
              published, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_prepared_result_invalid,
          "a refused prepare-on-behalf is named, not guessed");
    SuspendedOwnerFacts record;
    check(registry.entry_facts(identity, record) && !record.result_published &&
              !record.published_result.produced() && registry.results_published() == 0,
          "and the record is exactly as it was: no result, no counter");
    // The payload's own controller states the result; the version it is stamped with is the registry's.
    check(registry.publish_retained_result_from_owner(
              identity,
              [](FixtureOutcome& payload, uint64_t record_version, PreparedRetainedResult& result,
                 std::string& why) {
                  result = payload.controller->prepare_retained_result(record_version, why);
                  return result.produced();
              },
              published, error),
          "the registry publishes the result the parked payload stated about itself");
    check(published.produced() && registry.results_published() == 1 &&
              published.record_version() == record.record_version && published.identity() == identity,
          "the published result is the controller's own, versioned by the registry's record");
    check(registry.entry_facts(identity, record) && record.result_published &&
              record.published_result.produced(),
          "the record now carries a result, which is what activation requires");
    // The route is not a bypass: a payload that states a result for a version the registry did not name is
    // refused by the same gate the caller-supplied form runs.
    PreparedRetainedResult stale;
    check(!registry.publish_retained_result_from_owner(
              identity,
              [](FixtureOutcome& payload, uint64_t record_version, PreparedRetainedResult& result,
                 std::string& why) {
                  result = payload.controller->prepare_retained_result(record_version + 1, why);
                  return true;   // the payload vouches for it; the registry does not have to
              },
              stale, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_record_version_stale,
          "a result prepared against a version the registry never named is refused");
    // Activation now reaches the gate that matters: the chain is still mid-debt, so it stays parked and charged.
    check(!registry.activate_parked_owner(identity, activation_session(*made.controller, 0,
                                                                      core::serving_raw::Phase::drained, identity),
                                          captured_prior_state(*made.controller),
                                          [](FixtureOutcome&, const RetainedActivationFacts&,
                                             RepairOwnerEvidence&, ConversationStateFacts&, std::string&) {
                                              return true;
                                          },
                                          returned, restored, after, charges, error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_activation_state_diverged,
          "the published result clears the absence gate; the open debt is what refuses now");
    check(registry.contains(identity) && !bool(returned) && FixtureResourceCharge::live == 3 &&
              FixtureCharge::live == 1,
          "the refusal left the owner parked, with its charge and its three activation pins");
    std::string retire_error;
    check(registry.retire_all("test", FixtureRetire{}, retire_error) && registry.size() == 0, "drain");
    check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0, "and release everything");
    made.controller.reset();
}

void test_a_restore_onto_the_publication_cut_is_verified_and_a_refused_activation_restores_the_prior_state() {
    // Slice 6 blocker 2: at an A-return the live session stands on ANOTHER conversation's cut, so the serving
    // scope owes the controller a restore onto THIS owner's publication cut before `activate_retained` will
    // mutate. This cell drives the whole sequence the serving path now runs - capture the other conversation at
    // a drained boundary, advance the chain to the endpoint, publish the owner's own result, restore the
    // publication image - and then shows the refusal half: a controller that still refuses after the restore
    // leaves the owner parked, charged, unquarantined, and the prior conversation state exactly as captured.
    FixtureRegistry registry(kOneSlot);
    std::string error;
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "park A");
    FixtureController& c = *made.controller;
    const core::StateCut other = fixture_cut(90, kPublishedForwarded + 6, kPublishedForwarded + 6, 0x91);
    c.live_cut = other;   // B's turns are what the live session stands on now
    // The chain has to catch the published cut before any boundary can land it.
    RepairOwnerEvidence after_unit{};
    uint64_t advanced = 0;
    for (uint64_t units = 1; c.private_cut.forwarded < c.published_cut.forwarded; ++units) {
        LiveSessionFacts idle = idle_session(c, true, identity);
        idle.residency_generation = c.residency_generation;
        check(units <= 8 && run_unit(registry, identity, idle, kFixtureCap, after_unit, advanced, error) &&
                  advanced > 0,
              "bounded idle units close A's debt");
    }
    PreparedRetainedResult published{};
    check(registry.publish_retained_result_from_owner(
              identity,
              [](FixtureOutcome& payload, uint64_t record_version, PreparedRetainedResult& result,
                 std::string& why) {
                  result = payload.controller->prepare_retained_result(record_version, why);
                  return result.produced();
              },
              published, error),
          "the registry publishes the parked payload's own result");
    // The serving scope captures the other conversation at the drained boundary, BEFORE the restore it is about
    // to run; the registry's prior-state and session facts are these pre-restore observations.
    const ConversationStateFacts prior_b = captured_prior_state(c);
    LiveSessionFacts boundary = activation_session(c, c.residency_generation,
                                                   core::serving_raw::Phase::drained, identity);
    // The restore: the owner's publication image is installed onto the live transaction, and the session is
    // verified to match the owner record before the controller is asked to mutate.
    c.live_cut = c.published_cut;
    SuspendedOwnerFacts record;
    check(registry.entry_facts(identity, record) && c.live_cut == record.published_cut &&
              c.published_key == identity,
          "the restored session now matches the owner record's publication: the guard is satisfied, not bypassed");
    c.activation_denied = true;   // the controller still refuses to mutate at this boundary
    auto attempt = [prior_b](FixtureOutcome& payload, const RetainedActivationFacts&,
                             RepairOwnerEvidence& after_evidence, ConversationStateFacts& restored_out,
                             std::string& attempt_error) {
        // The refusal half of the real `activate_retained`: the capture/restore primitive puts the session back
        // exactly where the prior capture took it, and reports it field by field.
        return payload.controller->activate_retained(prior_b, restored_out, after_evidence, attempt_error);
    };
    std::unique_ptr<FixtureOutcome> returned;
    ConversationStateFacts restored{};
    RepairOwnerEvidence after{};
    std::vector<OwnerReservationHandle> charges;
    check(!registry.activate_parked_owner(identity, boundary, prior_b, attempt, returned, restored, after, charges,
                                          error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_private_unit &&
              error.find("the controller refused the activation") != std::string::npos,
          "the controller's own refusal is reported, not hidden");
    check(ConversationStateFacts::identical(prior_b, restored) && c.live_cut == prior_b.cut,
          "the refused restore put the prior conversation state back exactly, and the session verification holds");
    check(registry.contains(identity) && !bool(returned) && registry.activations_completed() == 0 &&
              registry.size() == 1 && registry.entry_facts(identity, record) && !record.quarantined &&
              record.result_published,
          "the owner stays parked, unquarantined and still carries its published result");
    check(FixtureResourceCharge::live == 3 && FixtureCharge::live == 1,
          "and its charge is intact: a refused activation frees nothing");
    // The evidence values this cell read are aliases of the owner's own ledger lease (the same aliasing the
    // production `RepairOwnerFacts::private_charge` carries), so they are dropped before the leak assertion,
    // exactly as the A->B->A cell drops its own.
    after_unit = RepairOwnerEvidence{};
    after = RepairOwnerEvidence{};
    charges.clear();
    std::string retire_error;
    check(registry.retire_all("test", FixtureRetire{}, retire_error) && registry.size() == 0, "drain");
    check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0, "and release everything");
    made.controller.reset();
}

// ---- slice 6 contract cells: activation installs the recorded publication, approximate or not ----------
// The reviewed contract change to `activate_retained`'s boundary (resident_repair_controller.inc): exactness is
// required of the certified private checkpoint R, never of the publication D, because suffix reuse leaves D
// approximate by construction. What the gates require instead is that the installed cut IS the recorded
// publication field by field, so an image that relabels D is refused by name. These two cells drive that contract
// through the production gate and the production registry; the CUDA-bound controller body itself is still not
// compiled on this host, so the boundary conjunct itself is exercised here only in its shared-gate form.
void test_an_activation_installs_an_approximate_publication_and_preserves_its_approximation() {
    FixtureRegistry registry(kOneSlot);
    std::string error;
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "park A");
    FixtureController& c = *made.controller;
    check(c.published_cut.exact_frontier < c.published_cut.forwarded,
          "the fixture publication is approximate at its own frontier, as a suffix-reuse CURRENT is");
    const core::StateCut other = fixture_cut(90, kPublishedForwarded + 6, kPublishedForwarded + 6, 0x91);
    c.live_cut = other;   // B's conversation is what the live session stands on
    RepairOwnerEvidence after_unit{};
    uint64_t advanced = 0;
    for (uint64_t units = 1; c.private_cut.forwarded < c.published_cut.forwarded; ++units) {
        LiveSessionFacts idle = idle_session(c, true, identity);
        idle.residency_generation = c.residency_generation;
        check(units <= 8 && run_unit(registry, identity, idle, kFixtureCap, after_unit, advanced, error) &&
                  advanced > 0,
              "bounded units catch the chain to the approximate publication");
    }
    PreparedRetainedResult published{};
    check(registry.publish_retained_result_from_owner(
              identity,
              [](FixtureOutcome& payload, uint64_t record_version, PreparedRetainedResult& result,
                 std::string& why) {
                  result = payload.controller->prepare_retained_result(record_version, why);
                  return result.produced();
              },
              published, error),
          "the result is prepared on behalf of the parked payload (the blocker 1 path)");
    check(published.certified_cut().exact_frontier == published.certified_cut().forwarded &&
              published.published_cut().exact_frontier < published.published_cut().forwarded,
          "R is exact at its own frontier while D stays approximate: the shape the boundary now accepts");
    const ConversationStateFacts prior_b = captured_prior_state(c);   // retained capture, taken at B's cut
    c.live_cut = c.published_cut;                                    // the scope restored D's image
    LiveSessionFacts boundary = activation_session(c, c.residency_generation,
                                                   core::serving_raw::Phase::drained, identity);
    SuspendedOwnerFacts record;
    check(registry.entry_facts(identity, record) && boundary.session_cut == record.published_cut,
          "the live session now stands on the owner's recorded publication");
    bool attempt_ran = false;
    auto attempt = [&](FixtureOutcome& payload, const RetainedActivationFacts&,
                       RepairOwnerEvidence& after_evidence, ConversationStateFacts& restored_out,
                       std::string& attempt_error) {
        attempt_ran = true;
        return payload.controller->activate_retained(prior_b, restored_out, after_evidence, attempt_error);
    };
    std::unique_ptr<FixtureOutcome> returned;
    ConversationStateFacts restored{};
    RepairOwnerEvidence after{};
    std::vector<OwnerReservationHandle> charges;
    check(registry.activate_parked_owner(identity, boundary, prior_b, attempt, returned, restored, after, charges,
                                         error),
          "an approximate publication is activatable: exactness is asked of R, not of D");
    check(attempt_ran && bool(returned) && registry.activations_completed() == 1 && registry.size() == 0,
          "the activation ran once and the payload came back to the serving scope");
    check(after.published_cut().exact_frontier < after.published_cut().forwarded &&
              after.published_cut() == published.published_cut() &&
              after.published_serial() == published.published_serial() && after.live_coherent() &&
              after.token_debt() == 0,
          "the installed publication kept its approximation shape: activation never marked D exact and never "
          "relabelled its cut, serial or frontier");
    check(ConversationStateFacts::identical(prior_b, restored),
          "and the retained capture of the other conversation is what the activation reports back");
    after_unit = RepairOwnerEvidence{};
    after = RepairOwnerEvidence{};
    charges.clear();
    std::string retire_error;
    check(bool(returned) && FixtureRetire{}(*returned, retire_error), "retire the activated owner deliberately");
    check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0, "and release every charge exactly once");
    returned.reset();
    made.controller.reset();
}

void test_an_installed_image_that_claims_an_exactness_the_record_does_not_carry_is_refused_by_name() {
    // The discriminating refusal: the restore put an image onto the session whose cut CLAIMS exactness at the
    // published position, while the owner record carries an approximate publication. The installed cut is not the
    // recorded publication, so the gate refuses before anything mutates and the attempt never runs.
    FixtureRegistry registry(kOneSlot);
    std::string error;
    FixtureOwner made = make_yielded_owner();
    const core::WindowKey identity = made.owner->published.key;
    check(registry.park_retained_owner(identity, made.owner, park_facts(registry, *made.controller, identity), error),
          "park A");
    FixtureController& c = *made.controller;
    const core::StateCut other = fixture_cut(90, kPublishedForwarded + 6, kPublishedForwarded + 6, 0x91);
    c.live_cut = other;
    RepairOwnerEvidence after_unit{};
    uint64_t advanced = 0;
    for (uint64_t units = 1; c.private_cut.forwarded < c.published_cut.forwarded; ++units) {
        LiveSessionFacts idle = idle_session(c, true, identity);
        idle.residency_generation = c.residency_generation;
        check(units <= 8 && run_unit(registry, identity, idle, kFixtureCap, after_unit, advanced, error) &&
                  advanced > 0,
              "the chain catches the published position");
    }
    PreparedRetainedResult published{};
    check(registry.publish_retained_result_from_owner(
              identity,
              [](FixtureOutcome& payload, uint64_t record_version, PreparedRetainedResult& result,
                 std::string& why) {
                  result = payload.controller->prepare_retained_result(record_version, why);
                  return result.produced();
              },
              published, error),
          "publish the payload's own result");
    const ConversationStateFacts prior_b = captured_prior_state(c);
    core::StateCut relabelled = c.published_cut;
    relabelled.exact_frontier = relabelled.forwarded;   // an image claiming a D that was never certified exact
    c.live_cut = relabelled;
    LiveSessionFacts boundary = activation_session(c, c.residency_generation,
                                                   core::serving_raw::Phase::drained, identity);
    bool attempt_ran = false;
    auto attempt = [&](FixtureOutcome&, const RetainedActivationFacts&, RepairOwnerEvidence&,
                       ConversationStateFacts&, std::string&) {
        attempt_ran = true;
        return true;
    };
    std::unique_ptr<FixtureOutcome> returned;
    ConversationStateFacts restored{};
    RepairOwnerEvidence after{};
    std::vector<OwnerReservationHandle> charges;
    check(!registry.activate_parked_owner(identity, boundary, prior_b, attempt, returned, restored, after, charges,
                                          error) &&
              registry.last_refusal() == RepairOwnerRefusal::refused_stale_publication && !attempt_ran,
          "a restored image that relabels the publication is refused by name, before the attempt runs");
    SuspendedOwnerFacts parked_record;
    check(registry.contains(identity) && !bool(returned) && registry.activations_completed() == 0 &&
              registry.entry_facts(identity, parked_record) && !parked_record.quarantined &&
              parked_record.result_published,
          "the owner stays parked, unquarantined, with its published result and its payload");
    check(FixtureResourceCharge::live == 3 && FixtureCharge::live == 1, "and every charge is intact");
    after_unit = RepairOwnerEvidence{};
    after = RepairOwnerEvidence{};
    charges.clear();
    std::string retire_error;
    check(registry.retire_all("test", FixtureRetire{}, retire_error) && registry.size() == 0, "drain");
    check(FixtureCharge::live == 0 && FixtureResourceCharge::live == 0, "and release everything");
    made.controller.reset();
}

#if defined(__linux__)
void test_registry_destructor_failstops_instead_of_dropping_an_unretired_owner() {
    // Forked probe: the child parks a live owner and lets the registry go out of scope.  The named failstop
    // must fire (_Exit(1)) before any payload destructor runs, so no controller is ever dropped unretired.
    pid_t child = ::fork();
    check(child >= 0, "fork for the failstop probe");
    if (child == 0) {
        {
            FixtureRegistry scoped(kOneSlot);
            FixtureOwner made = make_yielded_owner();
            const core::WindowKey identity = made.owner->published.key;
            std::string scoped_error;
            if (!scoped.park_retained_owner(identity, made.owner, park_facts(scoped, *made.controller, identity),
                                            scoped_error))
                std::_Exit(7);
            std::fprintf(stderr, "parked=%d\n", int(scoped.size()));
            std::fflush(stderr);
        }   // ~ServingRepairOwnerRegistry must failstop here, never drop the controller
        std::_Exit(0);
    }
    int status = 0;
    check(::waitpid(child, &status, 0) == child, "wait for the failstop probe");
    check(WIFEXITED(status) && WEXITSTATUS(status) == 1,
          "the registry destructor takes the named failstop exit(1), not a silent drop");
}
#endif

}  // namespace

int main() {
    test_private_window_needs_a_closed_quiescent_fence();
    test_private_window_refusals_are_stable();
    test_default_evidence_is_refused_not_assumed();
    test_contradictory_snapshots_cannot_be_invented();
    test_evidence_derives_debt_and_coherence();
    test_yield_gate_refuses_every_unsafe_owner();
    test_private_unit_gate_requires_the_suspended_state_and_a_cap();
    test_a_live_yielded_owner_is_parked_and_its_charge_is_pinned();
    test_a_retired_payload_is_refused();
    test_park_refusals();
    test_one_bounded_unit_advances_a_parked_owner();
    test_advance_refuses_outside_a_legal_idle_window();
    test_advance_refuses_a_record_that_no_longer_matches();
    test_advance_refuses_a_unit_that_breaks_the_contract();
    test_publication_is_refused_once_the_live_session_moved_on();
    test_resume_refusals_on_a_parked_owner();
    test_close_and_shutdown_refuse_then_retire();
    test_scope_guard_fires_when_an_owner_is_lost_and_stays_inert_when_registered();
    test_idle_admission_is_hard_gated_on_demand_not_on_any_number();
    test_d1_yield_is_reachable_at_response_completion();
    test_d1_admitted_response_still_needs_the_controller_s_own_certified_cut();
    test_d2_idle_period_rechecks_demand_before_every_unit();
    test_d3_yield_record_decision_certifies_only_a_complete_clean_pair();
    test_selectors_have_no_compiled_default();
    test_age_and_identity_records();
    test_two_owners_park_and_survive_each_other();
    test_prepared_retained_result_is_refused_when_it_is_not_a_certified_chain_cut();
    test_publish_if_current_is_accepted_for_the_current_record_and_refused_otherwise();
    test_activation_lands_only_at_a_drained_boundary_and_restores_the_other_conversation();
    test_capacity_is_admitted_before_the_yield_and_a_yield_without_it_is_refused();
    test_a_parked_owner_survives_a_second_conversation_and_is_activated_afterwards();
    test_the_registry_prepares_and_publishes_the_parked_owner_s_own_result();
    test_a_restore_onto_the_publication_cut_is_verified_and_a_refused_activation_restores_the_prior_state();
    test_an_activation_installs_an_approximate_publication_and_preserves_its_approximation();
    test_an_installed_image_that_claims_an_exactness_the_record_does_not_carry_is_refused_by_name();
#if defined(__linux__)
    test_registry_destructor_failstops_instead_of_dropping_an_unretired_owner();
#endif
    // The two invariants that make every cell above mean something: across all of them no controller was ever
    // dropped unretired (the terminal rule at resident_repair_controller.inc:70-71) and no payload was ever
    // destroyed while it still held a controller.
    check(FixtureController::unretired_drops == 0, "no controller was ever dropped unretired");
    check(FixtureOutcome::dropped_holding_live_controller == 0, "no payload died holding a live controller");
    check(FixtureCharge::live == 0, "no fixture ledger charge leaked out of any cell");
    std::printf("serving_repair_owners_test passed: production gates and registry exercised against fixture "
                "payloads, including slice 3's certified-yield reachability gate, the idle window's per-unit "
                "demand recheck, slice 4's retained-yield record decision and slice 5's publish-if-current / "
                "activate gates and capacity reservation; the real ResidentRepairController bodies "
                "(prepare_retained_result, publish_retained_if_current, activate_retained), the real serving token "
                "loop, the retained-yield comparator and the device capture/restore in generate.cpp (they do not "
                "compile on this host: cuda_runtime.h is absent), a real WindowBudget ledger and the CUDA runtime "
                "are NOT exercised by this cell\n");
    return 0;
}
