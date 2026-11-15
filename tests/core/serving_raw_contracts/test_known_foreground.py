#!/usr/bin/env python3
"""Execute actual controller known-token method with recording CPU seam stubs.

This checks prompt/history/receipt logic, not CUDA, native state, real drains, or
completion authority. The extracted production method is never reimplemented.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


PRELUDE = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace core {
struct Cut {uint64_t id=0,forwarded=0,accepted=0,exact_frontier=0,output_watermark=0,history=0;};
struct Key {uint64_t append_epoch=0;};
struct Region {uint64_t end=0;};
struct RepairSpatialCoverage {
    uint64_t begin=0,end=0;std::vector<int> roots;
    const std::vector<int>& ancestry()const{return roots;}
};
struct NativeRepairSuffixData {
    RepairSpatialCoverage coverage;std::vector<int> region_roots;
    explicit NativeRepairSuffixData(const RepairSpatialCoverage& c):coverage(c){}
};
enum class RepairAncestryStatus {ok};
RepairAncestryStatus make_repair_spatial_coverage(const std::vector<int>& roots,uint64_t begin,
    uint64_t end,RepairSpatialCoverage& out){out={begin,end,roots};return RepairAncestryStatus::ok;}
uint64_t window_history_digest(const std::vector<int32_t>& h){uint64_t v=1;for(auto t:h)v=v*131+uint32_t(t);return v;}
}
struct Metadata {
    core::Cut cut;core::Key key;
    std::vector<int32_t> accepted_tokens,pending_tokens;
    std::vector<core::Region> approximation;
};
struct Published {Metadata metadata;uint64_t serial=1;std::shared_ptr<const core::NativeRepairSuffixData> data;};
struct Dispatch {
    unsigned calls=0;int32_t consumed=-1;uint64_t position=0;bool done=false;
    bool run_foreground_one(int32_t token,uint64_t pos,int32_t& pick,std::string&){
        ++calls;consumed=token;position=pos;pick=42;done=true;return true;
    }
};
struct Geometry {int32_t n_vocab=100;};
struct Service {std::shared_ptr<Geometry> geometry_owner_=std::make_shared<Geometry>();};
struct Runtime {
    std::shared_ptr<Service> service_=std::make_shared<Service>();
    std::shared_ptr<Dispatch> dispatch_=std::make_shared<Dispatch>();
};
struct Live {
    uint64_t max_tokens_=16;unsigned begins=0,completions=0;bool running=false;
    bool begin_driver_work(){++begins;running=true;return true;}
};
struct Controller {
    struct Receipt {uint64_t operation=0;};
    struct KnownReceipt {uint64_t operation=0,publication=0,cut=0;int32_t consumed=-1,next=-1;};
    std::shared_ptr<Runtime> runtime_=std::make_shared<Runtime>();
    Live live_;std::shared_ptr<Published> published_=std::make_shared<Published>();
    bool busy_=false,finished_=false;uint64_t foreground_count_=0,known_count_=0,next_cut_=10;
    Receipt last_;KnownReceipt last_known_;
    bool check_idle(uint64_t serial)const{return !busy_&&!live_.running&&serial==published_->serial;}
    uint64_t mint_cut(){return next_cut_++;}
    [[noreturn]] static void fatal(const char* why){throw std::runtime_error(why);}
    void complete(Live& live,const core::Cut&){
        if(!live.running||!runtime_->dispatch_->done)fatal("test seam completed before dispatch");
        ++live.completions;live.running=false;runtime_->dispatch_->done=false;
    }
    Controller(){
        auto& m=published_->metadata;m.accepted_tokens={3,5,7};m.pending_tokens={7};
        m.cut={9,2,3,0,7,0};m.key.append_epoch=4;m.approximation={{2}};
        core::RepairSpatialCoverage coverage{0,2,{11}};
        auto data=std::make_shared<core::NativeRepairSuffixData>(coverage);data->region_roots={99};published_->data=data;
    }
'''

POSTLUDE = r'''
};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){try{
    Controller c;std::string error;
    const auto* first=c.foreground_known(1,1,9,error);
    require(first&&error.empty(),"real known-token method refused valid pending input");
    require(c.runtime_->dispatch_->calls==1&&c.runtime_->dispatch_->consumed==7&&
            c.runtime_->dispatch_->position==2,"known path must consume request pending at actual position");
    const auto& m=c.published_->metadata;
    require(m.accepted_tokens==std::vector<int32_t>({3,5,7,9})&&m.pending_tokens==std::vector<int32_t>({9}),
            "supplied next request token must win over computed greedy pick 42");
    require(m.cut.forwarded==3&&m.cut.accepted==4&&m.cut.exact_frontier==0&&m.cut.output_watermark==7,
            "known request work must advance history without exact certificate or output effect");
    require(m.key.append_epoch==5&&m.cut.history==core::window_history_digest(m.accepted_tokens),
            "known request append identity and history must describe actual accepted input");
    require(m.approximation.size()==1&&m.approximation[0].end==3&&c.published_->data->coverage.end==3&&
            c.published_->data->coverage.roots==std::vector<int>({11})&&
            c.published_->data->region_roots==std::vector<int>({99}),"known descendants must preserve causal ancestry");
    require(c.foreground_count_==1&&c.known_count_==1&&c.last_.operation==0&&
            c.live_.begins==1&&c.live_.completions==1,"known work must use dispatch seam without a generated receipt");
    const auto cut=m.cut.id;const auto accepted=m.accepted_tokens;
    require(c.foreground_known(999,1,9,error)==first&&c.runtime_->dispatch_->calls==1&&c.live_.begins==1,
            "historical exact-last-operation retry must return original acknowledgement without forwarding");
    require(!c.foreground_known(1,1,10,error),"retry cannot substitute a new known token");error.clear();
    require(!c.foreground_known(2,2,10,error),"stale selection must refuse before work");error.clear();
    require(!c.foreground_known(1,2,100,error),"out-of-vocabulary request token must refuse before work");error.clear();
    require(c.runtime_->dispatch_->calls==1&&c.live_.begins==1&&c.published_->metadata.cut.id==cut&&
            c.published_->metadata.accepted_tokens==accepted,"refused inputs must not mutate selected history or submit work");
    const auto* second=c.foreground_known(1,2,11,error);
    require(second&&second->cut>cut&&c.runtime_->dispatch_->calls==2&&c.runtime_->dispatch_->consumed==9&&
            c.published_->metadata.pending_tokens[0]==11&&c.published_->metadata.cut.output_watermark==7,
            "repeated actual known forwards must preserve request sequence and output effects");
    std::puts("PASS actual foreground_known CPU logic; recording dispatch/drain stubs, no native/GPU proof");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--expected-controller-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = (root / "src/program/resident_repair_controller.inc").read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if args.expected_controller_sha256 and digest != args.expected_controller_sha256:
        raise SystemExit("controller snapshot differs from the requested source pin")
    code = source.decode()
    marker = "const KnownReceipt* foreground_known("
    if code.count(marker) != 1:
        raise SystemExit("actual known-token method is missing or ambiguous")
    start = code.index(marker)
    begin = code.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (code[end] == "{") - (code[end] == "}")
        end += 1
    method = code[start:end]
    owned_tmp = None
    if args.build_dir is None:
        owned_tmp = tempfile.TemporaryDirectory(prefix="actual-known-foreground-")
        build = Path(owned_tmp.name)
    else:
        build = args.build_dir
        build.mkdir(parents=True, exist_ok=True)
    fixture = build / "actual-known-foreground.cpp"
    executable = build / "actual-known-foreground"
    fixture.write_text(PRELUDE + method + POSTLUDE)
    print(f"controller_sha256={digest} method_sha256={hashlib.sha256(method.encode()).hexdigest()}", flush=True)
    subprocess.run([args.compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wconversion", "-Werror",
                    str(fixture), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
    if owned_tmp is not None:
        owned_tmp.cleanup()


if __name__ == "__main__":
    main()
