#!/usr/bin/env python3
"""Actual core native-layout/freeze/validator/full-payload comparator regression. CPU only; no native GPU proof."""
import argparse
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[3]
CPP=r'''
#include "window_native_test_fixture.hpp"
#include "resident_repair_compare.inc"
#include <cstdio>
#include <stdexcept>
using namespace strata::core;
namespace q=strata::program::clean_recompute_qualification;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
struct Image {SnapshotDescriptor metadata;WindowStateLayout layout;BundleHandle bundle;};
Image image(SnapshotDescriptor metadata,int split=4,int format=1,bool mutate=false){
 auto geometry=window_test::geometry();
 std::vector<NativeWindowStage> stages{{0,split,std::vector<int>(1,format),true},{split,8,std::vector<int>(split==1?2:1,format),false}};
 auto native=window_test::saved(0,split,format,true);native.stage_images.push_back(window_test::saved(split,8,format,false));
 native.cvec=metadata.steering_enabled;for(auto& child:native.stage_images)child.cvec=metadata.steering_enabled;
 Image out;check(bool(native_window_layout(geometry,stages,metadata.key,metadata.cut,1024,1<<24,out.layout,metadata.steering_enabled)),"actual independent native layout");
 std::vector<NativePleTokens> ple(stages.size(),NativePleTokens{{8,9},10});
 const auto described=describe_native_window(native,geometry,out.layout,metadata,ple,out.metadata);
 if(!described)throw std::runtime_error("actual native inventory: "+described.detail);
 if(mutate){ConversationBuffer bytes;bytes.resize(out.metadata.devices[0].tensors[0].payload.size(),255);out.metadata.devices[0].tensors[0].payload=std::move(bytes);}
 check(bool(BundleHandle::freeze(out.metadata,out.metadata.key,out.layout,out.bundle)),"actual own-key immutable freeze");return out;
}
int main(){try{
 auto source_meta=window_test::metadata();auto destination_meta=source_meta;
 destination_meta.key.topology[0]=3;destination_meta.key.session_id="main";destination_meta.cut.id=42;
 auto a=image(source_meta),b=image(destination_meta);q::Report report;
 check(!q::compare(a.bundle,b.bundle,a.layout,report)&&report.first[0].reason==q::Difference::invalid_clean&&report.tensors_compared==0,"reproduce old same-layout invalid_clean");
 check(q::compare_physical_bridge(a.bundle,b.bundle,a.layout,b.layout,report)&&report.completed&&report.equal&&report.tensors_compared>0&&report.bytes_compared>0,"actual distinct owner layouts compare every tensor byte");
 uint64_t count=0,bytes=0;for(const auto& s:a.layout.stages)for(const auto& t:s.tensors){++count;bytes+=t.bytes;}
 check(report.tensors_compared==count&&report.bytes_compared==bytes,"complete actual tensor inventory");
 auto wrong=b.layout;wrong.topology=a.layout.topology;
 check(!q::compare_physical_bridge(a.bundle,b.bundle,a.layout,wrong,report)&&report.first[0].reason==q::Difference::invalid_clean,"wrong topology to own layout refuses");
 wrong=b.layout;wrong.model_profile[0]^=1;
 check(!q::compare_physical_bridge(a.bundle,b.bundle,a.layout,wrong,report),"wrong own model binding refuses");
 auto other_model=destination_meta;other_model.key.model_profile[0]=9;auto model=image(other_model);
 check(!q::compare_physical_bridge(a.bundle,model.bundle,a.layout,model.layout,report),"independently valid foreign model refuses");
 auto partition=image(destination_meta,1);
 check(!q::compare_physical_bridge(a.bundle,partition.bundle,a.layout,partition.layout,report),"actual differing stage geometry refuses");
 auto tensor=image(destination_meta,4,0);
 check(!q::compare_physical_bridge(a.bundle,tensor.bundle,a.layout,tensor.layout,report),"actual differing quantized tensor geometry refuses");
 auto changed=image(destination_meta,4,1,true);
 check(!q::compare_physical_bridge(a.bundle,changed.bundle,a.layout,changed.layout,report)&&report.completed&&report.bytes_compared==bytes,"mutated payload full comparison refuses");
 auto watermark=destination_meta;++watermark.cut.output_watermark;auto wm=image(watermark);
 check(!q::compare_physical_bridge(a.bundle,wm.bundle,a.layout,wm.layout,report),"watermark semantic mismatch refuses");
 auto history=destination_meta;history.accepted_tokens.back()=11;history.pending_tokens={11};history.cut.history=window_history_digest(history.accepted_tokens);auto h=image(history);
 check(!q::compare_physical_bridge(a.bundle,h.bundle,a.layout,h.layout,report),"actual accepted/pending history mismatch refuses");
 auto approx=destination_meta;approx.cut.exact_frontier=8;ApproximationRegion region;region.begin=8;region.end=9;region.first_unresolved_unix_ms=1;region.ancestor[0]=1;approx.approximation={region};auto ap=image(approx);
 check(!q::compare_physical_bridge(a.bundle,ap.bundle,a.layout,ap.layout,report)&&report.first[0].reason==q::Difference::not_exact,"approximate bridge refuses");
 check(!q::compare_preserved_selection(a.bundle,b.bundle,a.layout,report),"old preservation remains own-layout strict");
 std::puts("PASS actual independent native layouts/own validation/full payload: old invalid_clean reproduced and bridge negatives retained; CPU only");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
'''
def take(text,marker,semi=False):
 first=text.index(marker);opening=text.index('{',first);depth=0
 for at in range(opening,len(text)):
  if text[at]=='{':depth+=1
  elif text[at]=='}':
   depth-=1
   if depth==0:return text[first:at+1]+(';' if semi else '')+'\n'
 raise ValueError(marker)
EMITTER_MAIN=r'''int main(){try{
 auto metadata=window_test::metadata();metadata.steering_enabled=false;auto destination_metadata=metadata;
 destination_metadata.key.topology[0]=3;destination_metadata.key.session_id="main";destination_metadata.cut.id=42;
 auto a=image(metadata),b=image(destination_metadata);ServingRawSession slot;
 ServingQualification admission;admission.binary.fill('b');admission.binary.back()=0;admission.source.fill('c');admission.source.back()=0;
 uint64_t ordinal=3;auto actual=a.bundle,copy=b.bundle;auto destination_layout=b.layout;
 ACTUAL_CALL
 check(bridge,"actual emitted bridge must pass");
 const auto sl=qualification_layout_hash(a.layout),dl=qualification_layout_hash(b.layout);
 const auto ss=qualification_hex(qualification_state_hash(*actual.get(),sl));
 const auto ds=qualification_hex(qualification_state_hash(*copy.get(),dl));
 uint64_t count=0,bytes=0;for(const auto& stage:a.layout.stages)for(const auto& tensor:stage.tensors){++count;bytes+=tensor.bytes;}
 std::printf("TEST_EXPECT source_layout=%s destination_layout=%s source_state=%s destination_state=%s tensors=%llu bytes=%llu\n",
  qualification_hex(sl).data(),qualification_hex(dl).data(),ss.data(),ds.data(),(unsigned long long)count,(unsigned long long)bytes);
 return 0;}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
'''
def run(build,source=ROOT,overlay=None):
 source=Path(source);overlay=Path(overlay) if overlay else source;build=Path(build);build.mkdir(parents=True,exist_ok=True)
 caller=(overlay/'src/program/serving_slot_qualification.inc').read_text()
 for guard in ('source_tx.identity_matches(actual.get()->key)','destination_tx.identity_matches(copy.get()->key)',
               'source_tx.geometry_identity_!=destination_tx.geometry_identity_','slot.cut()!=actual.get()->cut||main.cut()!=copy.get()->cut',
               'native_window_layout(main.geometry(),main.initial_config(),copy.get()->key,copy.get()->cut',
               'report_serving_comparison(slot,admission,ordinal,"owned_slot_bridge"'):
  if guard not in caller:raise ValueError('actual bridge producer authentication missing '+guard)
 reporter=(overlay/'src/program/serving_suffix_qualification.inc').read_text()
 if reporter.index('owned_slot_bridge_binding req=')>reporter.index('phase=%s repaired_cut='):raise ValueError('bridge binding must precede comparison')
 flags=['c++','-std=c++20','-O0','-Wall','-Wextra','-I'+str(source/'include'),'-I'+str(source/'src/core')]
 objects=[]
 for index,rel in enumerate(('src/core/window_native.cpp','src/core/window_state.cpp','src/core/window_hash.cpp')):
  obj=build/('core'+str(index)+'.o');subprocess.run(flags+['-c',str(source/rel),'-o',str(obj)],check=True);objects.append(str(obj))
 comparator=(overlay/'src/program/resident_repair_compare.inc').read_text()
 variants=[('production',comparator),
  ('old_single_layout_bridge',comparator.replace('detail::compare_impl(source,destination,source_actual,destination_actual,report,detail::Purpose::clean_exact)','detail::compare_impl(source,destination,source_actual,source_actual,report,detail::Purpose::clean_exact)')),
  ('omitted_own_clean_validation',comparator.replace('if(!bv){report.validation_error=bv.error;report.note({Difference::invalid_clean});return false;}','/* negative: invalid destination own layout accepted */')),
  ('omitted_physical_model_layout_gate',comparator.replace('if(!same_physical_layout(actual,clean_actual)){report.note({Difference::native_layout});return false;}','/* negative: different independently valid model accepted */'))]
 for name,body in variants:
  if name!='production' and body==comparator:raise ValueError('ineffective mutant '+name)
  include=build/(name+'.inc');include.write_text(body);cpp=build/(name+'.cpp');binary=build/name
  cpp.write_text(CPP.replace('#include "resident_repair_compare.inc"','#include "'+str(include.resolve())+'"'))
  subprocess.run(flags+[str(cpp),*objects,'-o',str(binary)],check=True)
  result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
  if (result.returncode==0)!=(name=='production'):raise ValueError(name+': '+result.stdout+result.stderr)
  print('PASS '+name+': '+(result.stdout.strip() if name=='production' else 'required actual comparator control rejects mutated production'))
 # Real reporter, actual hash helpers, actual call expression, real validated core images.
 prelude=r'''
namespace core=strata::core;
namespace clean_recompute_qualification=strata::program::clean_recompute_qualification;
[[noreturn]] void failstop(const char* why){throw std::runtime_error(why);}
struct ServingRawSession {
 core::ModelGeometry shape=window_test::geometry();
 std::vector<core::NativeWindowStage> stages{{0,4,{1},true},{4,8,{1},false}};
 const core::ModelGeometry& geometry()const{return shape;}
 const std::vector<core::NativeWindowStage>& initial_config()const{return stages;}
 uint64_t max_tokens()const{return 1024;}uint64_t max_payload_bytes()const{return 1<<24;}
};
struct ServingQualification {bool activation_oracle=false;std::array<char,65> binary{},source{};};
'''
 actual_helpers=''.join(take(reporter,marker,semi) for marker,semi in (
  ('static std::array<char,65> qualification_hex(',False),('static void qualification_u64(',False),
  ('static void qualification_string(',False),('static core::StateDigest qualification_layout_hash(',False),
  ('static core::StateDigest qualification_state_hash(',False),('struct ServingComparisonWitness {',True),
  ('static bool report_serving_comparison(',False)))
 call_start=caller.index('const bool bridge=report_serving_comparison(');actual_call=caller[call_start:caller.index(';',call_start)+1]
 if '"physical_bridge",&destination_layout)' not in actual_call:raise ValueError('actual caller lost independent destination layout')
 base=CPP[:CPP.index('int main()')].replace('#include "resident_repair_compare.inc"','#include "'+str((build/'production.inc').resolve())+'"')
 for name,call in [('reporter',actual_call),('dropped_destination_argument',actual_call.replace(',&destination_layout','')),
                   ('wrong_destination_argument',actual_call.replace('&destination_layout','&a.layout')),
                   ('cross_owner_wrong_phase',actual_call.replace('"owned_slot_bridge"','"owned_slot_frontier"'))]:
  cpp=build/(name+'.cpp');binary=build/name;cpp.write_text(base+prelude+actual_helpers+EMITTER_MAIN.replace('ACTUAL_CALL',call))
  subprocess.run(flags+[str(cpp),*objects,'-o',str(binary)],check=True)
  result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
  if name!='reporter':
   if result.returncode==0:raise ValueError('actual emitter/caller mutant passed '+name)
   print('PASS '+name+': actual extracted reporter/call rejects mutation');continue
  if result.returncode:raise ValueError(result.stdout+result.stderr)
  def fields(line):return dict(item.split('=',1) for item in line.split()[1:])
  expected=fields(next(x for x in result.stdout.splitlines() if x.startswith('TEST_EXPECT ')))
  lines=result.stderr.splitlines();binding=next(x for x in lines if x.startswith('owned_slot_bridge_binding '));row=next(x for x in lines if x.startswith('suffix_repair_qualification '));b=fields(binding);r=fields(row)
  if lines.index(binding)>=lines.index(row):raise ValueError('binding ordering changed')
  for key in ('source_layout','destination_layout','source_state','destination_state'):
   if b[key]!=expected[key]:raise ValueError('emitted own-layout/state digest differs '+key)
  if b['source_layout']==b['destination_layout'] or b['source_state']==b['destination_state']:raise ValueError('fixture lost distinct actual owner bindings')
  if r['layout']!=b['source_layout'] or r['repaired']!=b['source_state'] or r['clean']!=r['repaired']:raise ValueError('source comparison basis differs')
  for key in ('tensors','bytes'):
   if r[key]!=expected[key] or int(r[key])<=0:raise ValueError('full emitted inventory differs')
  if r['completed']!='1' or r['equal']!='1' or r['mismatches']!='0' or b['source_cut']!='3' or b['destination_cut']!='42':raise ValueError('actual emitted bridge evidence differs')
  print('PASS reporter: actual call+hashes+complete byte inventory and distinct own-layout binding emitted')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--source',type=Path,default=ROOT);p.add_argument('--overlay',type=Path);a=p.parse_args();run(a.build_dir,a.source,a.overlay)
