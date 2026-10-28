#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

namespace strata::core {
class Verifier;
class ReservationLease;
// Serialized publication cut only. This object neither drains devices nor owns
// verifier/runtime resources. Actual callers must retain those owners and provide
// checked completion before commit. No callback, CUDA call or owner destruction
// occurs while this mutex is held.
class VerifierCompletionGate final : public std::enable_shared_from_this<VerifierCompletionGate> {
public:
    static constexpr std::size_t max_stages=8;
    class Token {
    public:
        Token()=default;
        Token(const Token&)=delete;Token& operator=(const Token&)=delete;
        Token(Token&&)=default;Token& operator=(Token&&)=delete;
        bool present()const noexcept{return bool(owner_);}
    private:
        friend class VerifierCompletionGate;
        std::shared_ptr<VerifierCompletionGate> owner_;
        uint64_t operation_=0,epoch_=0;
    };
    class Cut {
    public:
        Cut()=default;
        Cut(const Cut&)=delete;Cut& operator=(const Cut&)=delete;
        Cut(Cut&&)=default;Cut& operator=(Cut&&)=delete;
        // A committed historical cut remains valid after a later release. It
        // cannot authorize another unit, CUDA work or arbitrary output mutation.
        bool committed()const noexcept{return bool(owner_);}
    private:
        friend class VerifierCompletionGate;
        std::shared_ptr<const VerifierCompletionGate> owner_;
        uint64_t operation_=0,epoch_=0;
    };
    // Caller admits this conservative envelope BEFORE factory creation; the
    // actual exclusive host lease remains retained through every Token/Cut.
    // Includes ONE serialized caller Token/Cut scratch pair. External containers
    // retaining additional historical Cut objects require separate admission.
    static constexpr uint64_t metadata_bytes()noexcept{return sizeof(VerifierCompletionGate)+sizeof(Token)+sizeof(Cut)+4096;}
    // Begin only after all reversible admission succeeds. Destroying a live
    // Token leaves the gate busy forever: no implicit cancellation or reuse.
    bool try_begin(Token& output) {
        if(output.present())return false;
        auto self=shared_from_this();
        std::lock_guard<std::mutex> lock(mu_);
        if(unhealthy_||active_||next_operation_==UINT64_MAX)return false;
        active_=++next_operation_;
        output.operation_=active_;output.epoch_=epoch_;output.owner_=std::move(self);
        return true;
    }
    // Trusted controller must already have successful launcher/stream/helper/
    // callback completion and checked device restoration. No GPU checks here.
    bool commit_after_checked_completion(Token& token,Cut& output) {
        if(output.committed())return false;
        std::shared_ptr<VerifierCompletionGate> consumed;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if(token.owner_.get()!=this||!active_||token.operation_!=active_||
               token.epoch_!=epoch_||unhealthy_)return false;
            output.operation_=active_;output.epoch_=epoch_;output.owner_=token.owner_;
            active_=0;token.operation_=token.epoch_=0;
            consumed=std::move(token.owner_);
        } // Release consumed reference outside the gate.
        return true;
    }
    bool healthy_advisory()const noexcept {
        std::lock_guard<std::mutex> lock(mu_);return !unhealthy_;
    }
private:
    friend class Verifier;
    VerifierCompletionGate()=default;
    void fail_stage(std::size_t index)noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        if(index<count_)failed_[index]=true;
        // Invalid binding is also terminal; never silently ignore it.
        if(!unhealthy_){unhealthy_=true;++epoch_;}
    }
    std::shared_ptr<const ReservationLease> metadata_; // destroyed last
    std::array<const Verifier*,max_stages> identities_{}; // values only, never dereferenced
    std::array<bool,max_stages> failed_{};
    std::size_t count_=0;
    mutable std::mutex mu_;
    uint64_t epoch_=1,next_operation_=0,active_=0;
    bool unhealthy_=false;
};
}
