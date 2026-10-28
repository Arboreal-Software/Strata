#pragma once
#include "strata/core/window_budget.hpp"
#include "strata/core/window_hash.hpp"
#include <functional>
#include <string>
namespace strata::core {
class ParsedVerifiedInputs;
class ExpectedRunningExecutable {
public:
    ExpectedRunningExecutable(const ExpectedRunningExecutable&)=delete;
    ExpectedRunningExecutable& operator=(const ExpectedRunningExecutable&)=delete;
    uint64_t bytes()const noexcept{return bytes_;}
    const StateDigest& digest()const noexcept{return digest_;}
private:
    friend class ParsedVerifiedInputs;
    ExpectedRunningExecutable(uint64_t n,StateDigest d,std::shared_ptr<const ParsedVerifiedInputs> p,
        std::shared_ptr<const ReservationLease> lease)
        :lease_(std::move(lease)),bytes_(n),digest_(d),plan_(std::move(p)){}
    std::shared_ptr<const ReservationLease> lease_;
    uint64_t bytes_=0;
    StateDigest digest_{};
    std::shared_ptr<const ParsedVerifiedInputs> plan_;
};
struct RunningEngineLimits {
    uint64_t max_file_bytes=0,block_bytes=0,max_read_calls=0,max_elapsed_ms=0;
};
struct RunningEngineOptions {std::function<bool()> cancelled;};
struct RunningEngineObservation {
    uint64_t bytes=0,device=0,inode=0;
    StateDigest content{};
    uint16_t elf_machine=0;
};
class RunningEngineReceipt {
public:
    ~RunningEngineReceipt();
    RunningEngineReceipt(const RunningEngineReceipt&)=delete;
    RunningEngineReceipt& operator=(const RunningEngineReceipt&)=delete;
    // Partial main-image observation only, same process only. No ProfileArtifact
    // engine capability: loaded libraries/JIT/loader trust remain unresolved.
    const RunningEngineObservation* observation()const noexcept;
    bool partial()const noexcept{return true;}
private:
    RunningEngineReceipt()=default;
    friend bool collect_running_engine(std::shared_ptr<const ExpectedRunningExecutable>,
        const RunningEngineLimits&,WindowBudget,std::shared_ptr<const RunningEngineReceipt>&,
        std::string&,const RunningEngineOptions&);
    std::shared_ptr<const ReservationLease> lease_;
    std::shared_ptr<const ExpectedRunningExecutable> expected_;
    RunningEngineObservation observation_;
    int fd_=-1;
    int64_t pid_=0;
};
// Linux native ELF64 image only (host x86_64/aarch64, little endian, ET_EXEC or
// ET_DYN with bounded ELF program-header table and nonzero entry). Internally
// opens /proc/self/exe, never argv[0] or caller FD. Retains FD+plan+metadata
// quota; transient hash workspace/IO slot is released before success publication.
// Trusted host/kernel/process: stat rechecks do not defeat malicious same-UID
// or root in-place/mount tampering. This is NOT full runtime dependency closure.
// Cooperative cancellation/deadline/read-call limits surround regular-file IO;
// they cannot forcibly interrupt a blocked kernel syscall. Output unchanged on
// failure. Uncertain descriptor close is fail-stop, never retried.
bool collect_running_engine(std::shared_ptr<const ExpectedRunningExecutable>,const RunningEngineLimits&,
    WindowBudget,std::shared_ptr<const RunningEngineReceipt>&,std::string&,
    const RunningEngineOptions& = {});
}
