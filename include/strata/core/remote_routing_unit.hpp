#pragma once

#include "strata/core/remote_experts.hpp"
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>

namespace strata::core {

// One serialized controller-owned primary work unit. NOT concurrent admission
// authorization, a GPU drain implementation, or authenticated topology binding.
// The driver supplies its exact actual helper list and real runtime owner. Each
// RoutingRead also retains the helper's own bound runtime. This API cannot
// compare that private ownership identity to runtime below.
//
// Acquire ALL reads before route scans, RemoteExpertOpt::begin, callbacks or
// submission. Busy defers this NEW unit only; continue servicing admitted work.
// Do not hold a unit across adapt_owned (which needs exclusive helper writers).
// Helpers/RemoteOpt mapped outputs cannot be shared by overlapping units unless
// a separate real per-unit output ownership protocol has been implemented.
class RemoteRoutingUnit final {
public:
    static constexpr std::size_t max_helpers = 3;
    using Admission = RemoteExperts::Admission;

    // Assertions supplied by the trusted actual controller, not certificates
    // minted by this class. Serial integration must derive every field from real
    // successful calls/joins and checked device restoration. A false/throwing
    // operation cannot be converted into completion by merely releasing waits.
    struct CheckedCompletion {
        bool primary_and_helper_consumers_complete = false;
        bool dispatch_healthy = false;
        bool unit_host_callbacks_complete = false;
        bool owning_device_restored = false;
    };

    RemoteRoutingUnit() noexcept = default;
    RemoteRoutingUnit(const RemoteRoutingUnit&) = delete;
    RemoteRoutingUnit& operator=(const RemoteRoutingUnit&) = delete;
    RemoteRoutingUnit& operator=(RemoteRoutingUnit&&) = delete;

    // Move construction transfers the entire obligation into an async unit.
    // Caller serializes the move against all uses; no raw pointer to this holder
    // may have been captured by producers. Moved-from object is terminal/inert.
    RemoteRoutingUnit(RemoteRoutingUnit&& other) noexcept
        : runtime_(std::move(other.runtime_)), reads_(std::move(other.reads_)),
          helpers_(other.helpers_), count_(other.count_), state_(other.state_) {
        other.helpers_.fill(nullptr);
        other.count_ = 0;
        other.state_ = State::retired;
    }

    ~RemoteRoutingUnit() noexcept {
        // Must happen in the destructor BODY, before automatic RoutingRead
        // destructors could release exclusion or actual runtime ownership.
        if (state_ == State::attempted) {
            std::fputs("strata: remote routing unit destroyed before checked completion; worker restart required\n", stderr);
            std::_Exit(1);
        }
        clear_unattempted();
    }

    // No allocations, user callbacks, GPU calls or retry/wait loops. The existing
    // primitive briefly locks its metadata mutex: this is not a wait-free API.
    // Nonempty lists require nonnull unique entries. An empty exact topology is
    // valid, but still retains runtime and the attempted-work obligation.
    Admission try_acquire(RemoteExperts* const* helpers, std::size_t count,
                          const std::shared_ptr<NativeRuntimeAttachment>& runtime) noexcept {
        if (state_ != State::empty || !runtime || count > max_helpers || (count && !helpers))
            return Admission::invalid;
        for (std::size_t i = 0; i < count; ++i) {
            if (!helpers[i]) return Admission::invalid;
            for (std::size_t j = 0; j < i; ++j)
                if (helpers[i] == helpers[j]) return Admission::invalid;
        }
        runtime_ = runtime;
        for (std::size_t i = 0; i < count; ++i) {
            const auto admitted = helpers[i]->try_routing_read(reads_[i]);
            if (admitted != Admission::acquired) {
                // No external work was attempted, so acquired prefixes may be
                // rolled back. Never downgrade unbound/unknown to legacy use.
                clear_unattempted();
                return admitted;
            }
            helpers_[i] = helpers[i];
            ++count_;
        }
        state_ = State::acquired;
        return Admission::acquired;
    }

    // Call BEFORE the first producer/route-plan operation, including one that
    // may throw. Failure means do not enter that operation.
    bool mark_attempted() noexcept {
        if (state_ != State::acquired) return false;
        state_ = State::attempted;
        return true;
    }

    bool rollback_unattempted() noexcept {
        if (state_ == State::attempted || state_ == State::retired) return false;
        clear_unattempted();
        return true;
    }

    // No synchronization is performed here. On any incomplete assertion leave
    // guards/runtime and attempted state intact for checked recovery or failstop.
    bool retire_after_checked_completion(const CheckedCompletion& completion) noexcept {
        if (state_ != State::attempted || !completion.primary_and_helper_consumers_complete ||
            !completion.dispatch_healthy || !completion.unit_host_callbacks_complete ||
            !completion.owning_device_restored)
            return false;
        state_ = State::acquired; // caller has supplied the actual completion cut
        clear_unattempted();
        state_ = State::retired;
        return true;
    }

    bool acquired() const noexcept { return state_ == State::acquired; }
    bool attempted() const noexcept { return state_ == State::attempted; }
    std::size_t helper_count() const noexcept { return count_; }

private:
    enum class State { empty, acquired, attempted, retired };
    void clear_unattempted() noexcept {
        while (count_) {
            --count_;
            reads_[count_].reset();
            helpers_[count_] = nullptr;
        }
        runtime_.reset();
        if (state_ != State::retired) state_ = State::empty;
    }

    // Declaration order keeps the explicit runtime reference until after reads
    // on ordinary teardown. Attempted destruction stops before ANY member dtor.
    std::shared_ptr<NativeRuntimeAttachment> runtime_;
    std::array<RemoteExperts::RoutingRead, max_helpers> reads_{};
    std::array<RemoteExperts*, max_helpers> helpers_{};
    std::size_t count_ = 0;
    State state_ = State::empty;
};

} // namespace strata::core
