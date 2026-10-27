// The real capture orchestrator/native adapters with synthetic saved state.
// Device copies and verifier queries are injected; this is not GPU qualification.
#include "strata/core/window_capture.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/verify.hpp"
#include "window_native_test_fixture.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>

using namespace strata::core;
using namespace strata::core::window_test;
static int current_device = 0, copies = 0, fail_at = 0;
static std::map<const Verifier*,Verifier::SnapshotDrain> drains;
static WindowExecution* active_gate = nullptr;
static WindowKey active_key;
static void check(bool ok,const char* why) {
    if(!ok) { std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1); }
}
extern "C" cudaError_t cudaGetDevice(int* d) { *d=current_device;return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int d) { current_device=d;return cudaSuccess; }
namespace strata::core {
Verifier::~Verifier() = default;
Verifier::SnapshotDrain Verifier::poll_snapshot_drain(std::string&) { return drains.at(this); }
struct VerifierDrainTestPeer {
    static void bind(Verifier& v, SessionState& s,int device) { v.ss_=&s;v.device_=device; }
};
bool conversation_snapshot_save_private(SavedConversation& out,const ConversationView& view,
                                const SessionState& session,const ModelGeometry&,std::string& error) {
    ++copies;
    check(active_gate && !active_gate->resume(active_key),"capture hold blocks resume during each copy");
    auto directory=std::move(out.stage_images);
    out=saved(int(session.layer_lo),int(session.layer_hi),1,session.ple_hist!=nullptr);
    out.stage_images=std::move(directory);
    out.live.ids=view.ids;
    if(copies==fail_at) { error="injected stage-copy failure";return false; }
    return true;
}
}
int main() {
    auto meta=metadata();active_key=meta.key;
    auto g=geometry();
    WindowResources capacity{{100,100},32*1024*1024,0,0,0,2}, baseline{{0,0},0,0,0,0,0};
    WindowBudget budget(capacity,baseline);
    auto source=budget.reserve(WindowResources{{1,1},1024,0,0,0,0},ReservationPurpose::foreground);
    WindowExecution gate(meta.key,meta.cut,2,std::make_shared<int>(1),source.lease);
    active_gate=&gate;
    check(gate.begin_quiesce(meta.key),"source quiesces");
    WindowRead read;check(gate.acquire_read(meta.key,read),"source read held");
    QsaState qsa[2]{};qsa[0].kv_int8=qsa[1].kv_int8=true;
    SessionState sessions[2];Verifier verifiers[2];
    std::vector<NativeCaptureStage> stages;
    for(int i=0;i<2;++i) {
        auto& s=sessions[i];s.layer_lo=i*4;s.layer_hi=i*4+4;s.qsa_states=qsa;s.qsa_ord0=i;s.qsa_alloc=1;
        s.ple_prev[0]=8;s.ple_prev[1]=9;s.ple_token=10;
        if(i==0) s.ple_hist=reinterpret_cast<float*>(uintptr_t(1));
        VerifierDrainTestPeer::bind(verifiers[i],s,i);
        drains[&verifiers[i]]=Verifier::SnapshotDrain::drained;
        stages.push_back({i,&s,&verifiers[i]});
    }
    BundleHandle output;
    check(bool(capture_native_window(read,g,stages,meta,1024,1<<24,budget,output)),"two-stage native capture succeeds");
    check(copies==2 && output.get()->devices.size()==2 && output.get()->pending_tokens==meta.pending_tokens,
          "all stage parts and pending accepted input preserved");
    check(current_device==0,"capture restores calling device");
    const auto* previous=output.get();const auto held=budget.snapshot().reserved.host_bytes;
    check(held>1024,"immutable result retains capture reservation");
    fail_at=0;copies=0;drains[&verifiers[1]]=Verifier::SnapshotDrain::pending;
    check(capture_native_window(read,g,stages,meta,1024,1<<24,budget,output).status==NativeCaptureStatus::pending && copies==0,
          "no stage copies until every verifier drained");
    drains[&verifiers[1]]=Verifier::SnapshotDrain::drained;
    auto wrong=meta;wrong.pending_tokens[0]++;
    check(capture_native_window(read,g,stages,wrong,1024,1<<24,budget,output).status==NativeCaptureStatus::invalid && copies==0,
          "pending history mismatch rejected before copies");
    auto wrong_stage=stages;wrong_stage[1].verifier=&verifiers[0];
    check(capture_native_window(read,g,wrong_stage,meta,1024,1<<24,budget,output).status==NativeCaptureStatus::invalid && copies==0,
          "foreign verifier binding rejected before copies");
    auto baseline_high=baseline;baseline_high.host_bytes=capacity.host_bytes;
    check(budget.observe_baseline(baseline_high),"observed external growth recorded");
    check(capture_native_window(read,g,stages,meta,1024,1<<24,budget,output).status==NativeCaptureStatus::budget_refused && copies==0,
          "resource refusal happens before copy");
    check(budget.observe_baseline(baseline),"external baseline restored");
    copies=0;fail_at=2;
    auto failed=capture_native_window(read,g,stages,meta,1024,1<<24,budget,output);
    check(failed.status==NativeCaptureStatus::unhealthy && bool(failed.failure), "second-stage failure returns retirement handle");
    check(copies==2 && output.get()==previous && budget.snapshot().reserved.host_bytes>held,
          "failed capture preserves prior result and retains potentially in-flight destination");
    failed.failure->retire_after_teardown();
    check(budget.snapshot().reserved.host_bytes==held, "confirmed teardown releases failed transaction only");
    output={};check(budget.snapshot().reserved.host_bytes==1024,"last bundle releases capture charge");
    read={};check(!gate.resume(meta.key),"failed device cannot resume even after capture storage retirement");
    std::puts("window_capture_test passed");
}
