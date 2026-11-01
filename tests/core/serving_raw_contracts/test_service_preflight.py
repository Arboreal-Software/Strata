#!/usr/bin/env python3
"""Run actual fixed-service predicate and prebuilt retirement guard at CPU seams.

No native/GPU state, real allocation, drain or completion authority is simulated.
The recording stubs prove refusal selection and checked-teardown control flow only.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


PRELUDE = r'''
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace core { [[noreturn]] void retirement_failed(const char* why){std::fprintf(stderr,"FATAL %s\n",why);std::terminate();} }
namespace strata { namespace core = ::core; }
namespace strata::program { struct SuffixRawTransaction {}; }
struct Geometry {int n_expert=2;};
struct Verifier;
struct Binding {void* optimizer=nullptr;void* session=nullptr;Geometry* geometry=nullptr;Verifier* next=nullptr;const void* next_user=nullptr;};
struct Verifier {Binding binding;void* adapt_gate_=nullptr;Binding serial_routing_binding()const{return binding;}
VERIFIER_GATE_GETTER
};
struct AdaptGate {std::atomic<bool> active{false};};
struct Stage {Verifier ver;};
struct NativeStage {void* session=nullptr;};
struct Sessions {bool good=true;std::vector<NativeStage> native;bool healthy()const{return good;}const auto& stages()const{return native;}};
struct Source {
    bool failed=false;void* host_res=this;int remote_count=0;void* peer=nullptr;void* lookahead=nullptr;void* fs=nullptr;
    struct {bool on=false;} pcie_model;
    void* host_execution_observer=nullptr;std::vector<float> usage;
    bool evict_sync=false,res_dirty=false,dres_check=false;int n_expert=2,cache_blob=64;
};
struct Drive {Source d;};
struct Split {int n=2;};
struct DriverWindowRuntime {
    std::shared_ptr<Drive> drive=std::make_shared<Drive>();
    std::shared_ptr<Geometry> geometry_owner=std::make_shared<Geometry>();
    std::shared_ptr<Verifier> primary_verifier=std::make_shared<Verifier>();
    Sessions sessions;std::atomic<unsigned> background_retirement{0};bool owned=true;
    void* remote_opt=nullptr;void* foresight=nullptr;
    std::shared_ptr<AdaptGate> adapt_gate=std::make_shared<AdaptGate>();
    std::vector<std::shared_ptr<Stage>> stages;std::shared_ptr<Split> split_drive;
    int session0=0,session1=0;
    DriverWindowRuntime(){sessions.native={{&session0}};primary_verifier->binding.session=&session0;primary_verifier->binding.geometry=geometry_owner.get();}
    bool owns_expert_dispatch(Source* source)const{return owned&&source==&drive->d;}
    void dual(){
        stages.push_back(std::make_shared<Stage>());sessions.native.push_back({&session1});split_drive=std::make_shared<Split>();
        primary_verifier->binding.next=&stages[0]->ver;primary_verifier->binding.next_user=split_drive.get();
        stages[0]->ver.binding.session=&session1;stages[0]->ver.binding.geometry=geometry_owner.get();
    }
};
namespace strata::program::resident_repair_detail {
constexpr size_t maximum_stages=8;
enum class RuntimePhase {ready,drained,retired};
struct RecordingDispatch {unsigned retire_calls=0;bool permit=true;bool retire_initialized(std::string&){++retire_calls;return permit;}};
struct ResidentRepairRuntime {RuntimePhase phase_=RuntimePhase::ready;std::shared_ptr<RecordingDispatch> dispatch_=std::make_shared<RecordingDispatch>();};
'''

TESTS = r'''
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){try{
    using namespace strata::program::resident_repair_detail;
    DriverWindowRuntime single;
    require(single.adapt_gate&&!single.adapt_gate->active.load()&&!single.primary_verifier->has_adapt_gate()&&
            !resident_service_refusal(single),"allocated inactive unbound single-stage placeholder must admit");
    DriverWindowRuntime dual;dual.dual();
    require(dual.adapt_gate&&!dual.stages[0]->ver.has_adapt_gate()&&!resident_service_refusal(dual),
            "allocated inactive unbound two-stage placeholder must admit");
    using Change=void(*)(DriverWindowRuntime&);
    const Change changes[]={
        [](auto& s){s.drive.reset();}, [](auto& s){s.geometry_owner.reset();},
        [](auto& s){s.primary_verifier.reset();}, [](auto& s){s.sessions.good=false;},
        [](auto& s){s.background_retirement=1;}, [](auto& s){s.owned=false;},
        [](auto& s){s.drive->d.failed=true;}, [](auto& s){s.drive->d.host_res=nullptr;},
        [](auto& s){s.drive->d.remote_count=1;}, [](auto& s){s.remote_opt=&s;},
        [](auto& s){s.drive->d.peer=&s;}, [](auto& s){s.drive->d.lookahead=&s;},
        [](auto& s){s.foresight=&s;}, [](auto& s){s.drive->d.fs=&s;},
        [](auto& s){s.adapt_gate->active.store(true);}, [](auto& s){s.drive->d.pcie_model.on=true;},
        [](auto& s){s.drive->d.host_execution_observer=&s;}, [](auto& s){s.drive->d.usage.push_back(1);},
        [](auto& s){s.drive->d.evict_sync=true;}, [](auto& s){s.drive->d.res_dirty=true;},
        [](auto& s){s.drive->d.dres_check=true;}, [](auto& s){s.drive->d.n_expert=3;},
        [](auto& s){s.drive->d.cache_blob=0;}, [](auto& s){s.primary_verifier->binding.optimizer=&s;},
        [](auto& s){s.primary_verifier->binding.session=nullptr;},
        [](auto& s){s.primary_verifier->binding.geometry=nullptr;},
        [](auto& s){s.dual();s.stages[0].reset();},
        [](auto& s){s.dual();s.split_drive->n=1;},
        [](auto& s){s.dual();s.primary_verifier->binding.next_user=nullptr;},
        [](auto& s){s.dual();s.stages[0]->ver.binding.optimizer=&s;},
        [](auto& s){s.primary_verifier->adapt_gate_=&s;},
        [](auto& s){s.adapt_gate.reset();s.primary_verifier->adapt_gate_=&s;},
        [](auto& s){s.dual();s.stages[0]->ver.adapt_gate_=&s;},
        [](auto& s){s.dual();s.adapt_gate->active.store(true);},
    };
    unsigned refused=0;
    for(auto change:changes){DriverWindowRuntime s;change(s);
        require(resident_service_refusal(s)!=nullptr,"unsupported actual mode escaped preflight");++refused;}

    // Execute the exact controller retirement helper. Refusal must retain the genuine owner; success drops it
    // only after the recording dispatch acknowledged its explicit checked retirement seam.
    auto private_runtime=std::make_shared<ResidentRepairRuntime>();
    const auto dispatch=private_runtime->dispatch_;std::weak_ptr<ResidentRepairRuntime> weak=private_runtime;
    std::string error;dispatch->permit=false;
    require(!ResidentRepairController::retire_preconstructed(private_runtime,error)&&private_runtime&&!weak.expired()&&
            dispatch->retire_calls==1,"retirement refusal must retain the preconstructed owner");
    dispatch->permit=true;
    require(ResidentRepairController::retire_preconstructed(private_runtime,error)&&!private_runtime&&weak.expired()&&
            dispatch->retire_calls==2,"successful checked retirement must precede owner release");
    require(ResidentRepairController::retire_preconstructed(private_runtime,error),"empty prebuilt cleanup must be harmless");
    auto preconstructed=std::make_shared<ResidentRepairRuntime>();
    const auto guarded_dispatch=preconstructed->dispatch_;
    std::weak_ptr<ResidentRepairRuntime> guarded_weak=preconstructed;
    auto preinitialized=std::make_unique<strata::program::SuffixRawTransaction>();
    {
GUARD
        // Simulated pre-install quote/admission refusal exits this scope; the actual production scope guard
        // must invoke actual prebuilt-retirement helper before the operation could return exact fallback.
    }
    require(!preconstructed&&!preinitialized&&guarded_weak.expired()&&guarded_dispatch->retire_calls==1,
            "actual pre-install scope guard failed to retire initialized private runtime");
    std::printf("PASS actual fixed-service predicate (%u negative modes) and prebuilt teardown CPU seams; no native/GPU proof\n",refused);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
'''


def extract(code, marker):
    if code.count(marker) != 1:
        raise SystemExit(f"missing/ambiguous production source: {marker}")
    start = code.index(marker)
    begin = code.index("{", start)
    end, depth = begin + 1, 1
    while depth:
        depth += (code[end] == "{") - (code[end] == "}")
        end += 1
    return code[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sources = {name: (root / "src/program" / name).read_text() for name in
               ("resident_repair_runtime.inc", "resident_repair_controller.inc", "generate.cpp")}
    sources["verify.hpp"]=(root/"include/strata/core/verify.hpp").read_text()
    getter=extract(sources["verify.hpp"],"bool has_adapt_gate() const noexcept")
    predicate = extract(sources["resident_repair_runtime.inc"], "inline const char* resident_service_refusal(")
    retire = extract(sources["resident_repair_controller.inc"], "static bool retire_preconstructed(")
    retire_wrapper = extract(sources["generate.cpp"], "static void retire_prebuilt(")
    guard = extract(sources["generate.cpp"], "struct PreparedRuntimeGuard {") + " prepared_guard{preconstructed,preinitialized};"
    generated = (PRELUDE.replace("VERIFIER_GATE_GETTER",getter) + predicate + "\nstruct ResidentRepairController {\n" + retire + "\n};\n}\n" +
                 "namespace resident_repair_detail = strata::program::resident_repair_detail;\n" +
                 "namespace rrd = strata::program::resident_repair_detail;\n" +
                 "struct ServingSuffixRepairOperation {\n" + retire_wrapper + "\n};\n" + TESTS.replace("GUARD", guard))
    temporary = None
    if args.build_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="actual-service-preflight-")
        build = Path(temporary.name)
    else:
        build = args.build_dir
        build.mkdir(parents=True, exist_ok=True)
    fixture, executable = build / "actual-service-preflight.cpp", build / "actual-service-preflight"
    fixture.write_text(generated)
    for name, code in sources.items():
        print(f"{name} sha256={hashlib.sha256(code.encode()).hexdigest()}", flush=True)
    subprocess.run([args.compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wconversion", "-Werror",
                    str(fixture), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
