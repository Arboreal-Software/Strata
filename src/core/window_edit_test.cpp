#include "strata/core/window_edit.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;
namespace {
void check(bool ok,const char* what) { if(!ok){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);} }
BundleHandle bundle(std::vector<int32_t> ids,uint64_t c,uint64_t f) {
    SnapshotDescriptor s;
    s.key = {1,1,"session","thread","window",1,1,1,1,{1},{2}};
    s.cut = {1,ids.size(),c,f,17,window_history_digest(ids)};
    s.accepted_tokens = std::move(ids);
    s.pending_tokens.assign(s.accepted_tokens.begin()+c,s.accepted_tokens.end());
    s.native_layout = "edit-fixture";
    if(f<c) s.approximation.push_back({f,c,100,{3}});
    TensorLayout t{StatePartKind::gdn_recurrence_and_convolution,0,0,"f32","fixture",{4},{1},0,4,4};
    TensorPart p; p.layout=t; p.payload.resize(4,23);
    s.devices.push_back({0,0,1,s.cut,{std::move(p)}});
    WindowStateLayout l;
    l.native_layout=s.native_layout;l.model_profile=s.key.model_profile;l.topology=s.key.topology;
    l.layers={WindowStateLayout::LayerKind::gdn};l.stages={{0,0,1,false,{t}}};
    l.max_tokens=100;l.max_payload_bytes=100;
    BundleHandle out;
    check(bool(BundleHandle::freeze(s,s.key,l,out)),"planning fixture is a validated immutable bundle");
    return out;
}
}
int main() {
    auto source=bundle({1,2,3,4,5,6,7},6,6);
    const auto key=source.get()->key;
    auto target=key;++target.revision;
    auto cp=bundle({1,2,3},2,2);
    ExactWindowEdit plan;
    check(bool(plan_exact_window_edit(source,key,target,{1,2,3,4,8,9},5,100,{cp},plan)),"exact edit planned");
    check(plan.replay_begin==2 && plan.gap_end==4 && plan.replay_end==5,"checkpoint gap and changed remainder both replayed");
    check(plan.pending_tokens==std::vector<int32_t>{9} && plan.output_watermark==17,"pending input and emitted watermark preserved");
    cp={};
    check(bool(plan.checkpoint),"plan retains checkpoint source lease");
    check(bool(plan_exact_window_edit(source,key,target,{1,2,3,4,8,9},5,100,{},plan)) && plan.replay_begin==0,
          "without a complete recurrent checkpoint replay starts at zero");
    auto approximate=bundle({1,2,3,4,5,6,7},6,2);
    auto later_cp=bundle({1,2,3,4,5},4,4);
    check(bool(plan_exact_window_edit(approximate,key,target,{1,2,3,4,5,6,7},6,100,{later_cp},plan)),"approximate parent edit planned");
    check(plan.unchanged_exact_prefix==2 && plan.replay_begin==0,"matching approximate text cannot certify an exact prefix");
    check(bool(plan_exact_window_edit(source,key,target,{9,2,3},2,100,{later_cp},plan)) && plan.unchanged_exact_prefix==0,
          "edit at start requires clean replay");
    check(bool(plan_exact_window_edit(source,key,target,{1,2},1,100,{later_cp},plan)) && plan.replay_begin==0 && plan.replay_end==1,
          "deletion rejects checkpoints beyond new forwarded cut");
    check(bool(plan_exact_window_edit(source,key,target,{1,2,3,4,5,6,7},6,100,{},plan)) && plan.replay_begin==6,
          "same-token revision can reuse a complete exact endpoint");
    auto stale=key;++stale.selection_epoch;
    check(!plan_exact_window_edit(source,stale,target,{1},0,100,{},plan),"stale source selection rejected");
    check(!plan_exact_window_edit(source,key,key,{1},0,100,{},plan),"edit cannot reuse content revision");
    auto foreign=target;foreign.model_profile[0]++;
    check(!plan_exact_window_edit(source,key,foreign,{1},0,100,{},plan),"foreign template/model profile rejected");
    auto reused=target;++reused.window_generation;
    check(!plan_exact_window_edit(source,key,reused,{1},0,100,{},plan),"delete/recreate ABA rejected");
    std::puts("window_edit_test passed");
}
