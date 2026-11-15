#!/usr/bin/env python3
"""Execute actual fresh retained watermark initialization and exact continuation; recording CPU metadata only."""
import argparse
from pathlib import Path
import subprocess
from test_retained_request_continuation import take

def run(build):
 root=Path(__file__).resolve().parents[3];text=(root/'src/program/generate.cpp').read_text()
 begin=text.index('if(completed && facts.retain_foreground) {');end=text.index('if(!cohort_metadata_shape_supported',begin)
 init=text[begin:end];init=init[init.index('out.controller='):]
 if 'out.request_watermark_base=out.controller->initial_output_watermark_;' not in init:raise ValueError('fresh imported watermark authority missing')
 method=take(text,'    static bool exact_continuation_metadata(')
 pre='''#include "strata/core/window_repair_suffix.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
struct Controller {uint64_t initial_output_watermark_=2;};
struct Harness {struct Outcome {std::shared_ptr<Controller> controller;uint64_t request_watermark_base=0,generated=0,max_generated=64;bool retired=false,certified_exact=false;strata::core::SnapshotDescriptor published;};
'''
 tail='''};
int main(){try{for(uint64_t inherited: {0ULL,1ULL,2ULL,63ULL}){
 Harness::Outcome out;bool completed=true;std::shared_ptr<Controller> repair=std::make_shared<Controller>();repair->initial_output_watermark_=inherited;
 ACTUAL_INIT
 if(out.request_watermark_base!=inherited||!out.controller)throw std::runtime_error("fresh base not imported");
 out.controller.reset();out.retired=out.certified_exact=true;out.generated=3;
 auto& m=out.published;m.accepted_tokens={10,20,30};m.pending_tokens={30};m.cut.id=9;m.cut.accepted=3;m.cut.forwarded=m.cut.exact_frontier=2;m.cut.output_watermark=inherited+3;
 strata::core::SnapshotDescriptor next;std::string error;
 if(!Harness::exact_continuation_metadata(out,{10,20,30},40,4,10,next,error)||next.cut.output_watermark!=inherited+4)throw std::runtime_error("authentic nonzero base continuation refused");
 out.request_watermark_base=inherited+1;
 if(Harness::exact_continuation_metadata(out,{10,20,30},40,4,10,next,error))throw std::runtime_error("wrong base accepted");
 if(inherited){out.request_watermark_base=0;if(Harness::exact_continuation_metadata(out,{10,20,30},40,4,10,next,error))throw std::runtime_error("old zero base accepted");}
 }std::puts("PASS actual fresh initialization and exact continuation cumulative watermark; CPU metadata only");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\\n",e.what());return 1;}}
'''
 build=Path(build);build.mkdir(parents=True,exist_ok=True);source=build/'actual-fresh-watermark.cpp';binary=build/'actual-fresh-watermark'
 source.write_text(pre+method+tail.replace('ACTUAL_INIT',init))
 subprocess.run(['c++','-std=c++20','-O0','-Wall','-Wextra','-I'+str(root/'include'),str(source),str(root/'src/core/window_hash.cpp'),'-o',str(binary)],check=True)
 subprocess.run([str(binary)],check=True,timeout=20)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);a=p.parse_args();run(a.build_dir)
