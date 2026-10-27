// Fault-injection harness for the real drain implementation. No CUDA runtime is
// linked: these fake handles must never reach a GPU destructor or a real driver.
#include "strata/core/verify.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>

static std::map<cudaStream_t, cudaError_t> streams;
static cudaError_t event_status = cudaSuccess;
static int waits = 0, current_device = 0;
static int fail_get = 0, fail_set_at = 0, set_calls = 0;
extern "C" cudaError_t cudaGetDevice(int* d) { if(fail_get){--fail_get;return cudaErrorUnknown;}*d = current_device; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int d) { if(++set_calls==fail_set_at)return cudaErrorUnknown;current_device = d; return cudaSuccess; }
extern "C" cudaError_t cudaStreamQuery(cudaStream_t s) { return streams.at(s); }
extern "C" cudaError_t cudaEventQuery(cudaEvent_t) { return event_status; }
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t) { ++waits; return event_status; }
extern "C" const char* cudaGetErrorString(cudaError_t) { return "injected GPU error"; }

namespace strata::core {
// Isolated fixture cleanup: only C++ fields, since all GPU handles are fake.
Verifier::~Verifier() = default;
bool Verifier::handoff_failed(std::string& err) const {
    if (herr_h_ && *herr_h_) { err = "injected handoff failure"; return true; }
    return false;
}
struct VerifierDrainTestPeer {
    static void init(Verifier& v, int id) {
        v.device_ = id;
        v.cs_ = reinterpret_cast<cudaStream_t>(uintptr_t(id * 2 + 1));
        v.copy_ = reinterpret_cast<cudaStream_t>(uintptr_t(id * 2 + 2));
        streams[v.cs_] = streams[v.copy_] = cudaSuccess;
    }
    static void pending(Verifier& v) {
        v.commit_pending_ = true;
        v.commit_done_ = reinterpret_cast<cudaEvent_t>(uintptr_t(1));
    }
    static void copy_status(Verifier& v, cudaError_t s) { streams[v.copy_] = s; }
    static void compute_status(Verifier& v, cudaError_t s) { streams[v.cs_] = s; }
    static void chain(Verifier& a, Verifier& b) { a.next_ = &b; }
    static bool pending(const Verifier& v) { return v.commit_pending_; }
    static void fail_commit(Verifier& v) { v.commit_failure_ = "event recording failed after launch"; }
    static void release(Verifier& v) { v.released_.store(true); }
};
}
using namespace strata::core;
using Peer = VerifierDrainTestPeer;
using Drain = Verifier::SnapshotDrain;
static void check(bool b, const char* message) {
    if (!b) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
int main() {
    std::string error;
    Verifier first, second;
    check(first.poll_snapshot_drain(error) == Drain::unsupported, "uninitialized verifier cannot certify drain");
    Peer::init(first, 0); Peer::init(second, 1); Peer::chain(first, second);
    Peer::copy_status(second, cudaErrorNotReady);
    check(first.poll_snapshot_drain(error) == Drain::pending, "other-stage copy retains pending drain");
    check(current_device == 0, "poll restores calling device");
    Peer::copy_status(second, cudaSuccess);
    Peer::pending(second); event_status = cudaErrorNotReady;
    check(first.poll_snapshot_drain(error) == Drain::pending, "pending commit prevents drain");
    event_status = cudaSuccess;
    check(first.poll_snapshot_drain(error) == Drain::drained, "all streams and commit drain");
    check(!Peer::pending(static_cast<const Verifier&>(second)), "successful event clears pending");
    Peer::pending(second); event_status = cudaErrorUnknown;
    check(!first.wait_commit(error), "chain reports commit wait failure");
    check(Peer::pending(static_cast<const Verifier&>(second)), "failed event retains pending state");
    const int failed_waits = waits; event_status = cudaSuccess;
    check(!first.wait_commit(error) && waits == failed_waits, "repeat wait cannot erase terminal failure");
    check(first.poll_snapshot_drain(error) == Drain::unhealthy, "failed commit cannot become snapshot");
    Verifier copy_failure; Peer::init(copy_failure, 2);
    Peer::compute_status(copy_failure, cudaErrorNotReady); Peer::copy_status(copy_failure, cudaErrorUnknown);
    check(copy_failure.poll_snapshot_drain(error) == Drain::unhealthy, "pending compute cannot hide copy failure");
    Peer::compute_status(copy_failure, cudaSuccess); Peer::copy_status(copy_failure, cudaSuccess);
    check(copy_failure.poll_snapshot_drain(error) == Drain::unhealthy, "later successful query cannot erase failure");
    Verifier record_failure; Peer::init(record_failure, 3); Peer::fail_commit(record_failure);
    check(!record_failure.wait_commit(error), "record failure is retained without pending event");
    check(record_failure.poll_snapshot_drain(error) == Drain::unhealthy, "launched work without event cannot certify cut");
    Verifier released; Peer::init(released, 4); Peer::release(released);
    check(released.poll_snapshot_drain(error) == Drain::unhealthy, "watchdog-released state is unusable");
    Verifier selection_failure; Peer::init(selection_failure,5); fail_get=1;
    check(selection_failure.poll_snapshot_drain(error)==Drain::unhealthy,"initial device error cannot certify drain");
    check(selection_failure.poll_snapshot_drain(error)==Drain::unhealthy,"device selection failure remains sticky");
    Verifier restore_failure;Peer::init(restore_failure,6);set_calls=0;fail_set_at=2;current_device=0;
    check(restore_failure.poll_snapshot_drain(error)==Drain::unhealthy,"caller-device restore failure prevents drain");
    fail_set_at=0;current_device=0;
    check(restore_failure.poll_snapshot_drain(error)==Drain::unhealthy,"later success cannot clear failed restoration");
    Verifier wait_device_failure;Peer::init(wait_device_failure,7);Peer::pending(wait_device_failure);fail_get=1;
    const auto prior_waits=waits;
    check(!wait_device_failure.wait_commit(error)&&waits==prior_waits&&Peer::pending(static_cast<const Verifier&>(wait_device_failure)),
          "failed device selection preserves pending commit without synchronizing wrong device");
    std::puts("verify_drain_test passed");
}
