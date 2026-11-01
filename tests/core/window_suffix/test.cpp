#include "strata/core/window_suffix.hpp"
#include "window_native_test_fixture.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>
using namespace strata::core;
static thread_local long fail_after=-1;
void* operator new(size_t n){if(fail_after==0)throw std::bad_alloc();if(fail_after>0)--fail_after;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](size_t n){return ::operator new(n);}
void* operator new(size_t n,std::align_val_t a){if(fail_after==0)throw std::bad_alloc();if(fail_after>0)--fail_after;void* p=nullptr;if(!posix_memalign(&p,size_t(a),n?n:1))return p;throw std::bad_alloc();}
void* operator new[](size_t n,std::align_val_t a){return ::operator new(n,a);}
void operator delete(void* p)noexcept{std::free(p);}void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,size_t)noexcept{std::free(p);}void operator delete[](void* p,size_t)noexcept{std::free(p);}
void operator delete(void* p,std::align_val_t)noexcept{std::free(p);}void operator delete[](void* p,std::align_val_t)noexcept{std::free(p);}
void operator delete(void* p,size_t,std::align_val_t)noexcept{std::free(p);}void operator delete[](void* p,size_t,std::align_val_t)noexcept{std::free(p);}
static void check(bool v,const char* why){if(!v){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
static WindowResources host(uint64_t n){WindowResources r;r.device_bytes={0};r.host_bytes=n;return r;}
static NativeSuffixLimits limits(){return {1024,1<<24,1<<26};}
struct Fixture {
 ModelGeometry g=window_test::geometry();
 WindowBudget captures{host(64<<20),host(0)};
 std::vector<NativeWindowStage> stages;
 SnapshotDescriptor donor_meta,prefix_meta;
 WindowKey target;
 std::vector<int32_t> tokens{1,42,3,4,5,6,7,8,9};
 BundleHandle donor,prefix;
 Fixture(bool split=false){
  if(split)stages={{0,4,{1},true},{4,8,{1},false}};else stages={{0,8,{1,1},true}};
  auto& d=donor_meta;d.key.engine_generation=1;d.key.window_generation=1;d.key.session_id="s";d.key.thread_id="t";d.key.window_id="w";
  d.key.revision=d.key.append_epoch=d.key.selection_epoch=d.key.provenance_epoch=1;d.key.model_profile[0]=1;d.key.topology[0]=2;
  d.accepted_tokens={1,2,3,4,5,6,7,8,9};d.pending_tokens={9};d.cut={3,9,8,8,7,window_history_digest(d.accepted_tokens)};d.steering_enabled=false;
  target=d.key;target.revision=target.append_epoch=target.selection_epoch=2;
  auto& p=prefix_meta;p.key=target;p.accepted_tokens.assign(tokens.begin(),tokens.begin()+4);p.cut={4,4,4,4,7,window_history_digest(p.accepted_tokens)};p.steering_enabled=false;
  donor=make(d,false);prefix=make(p,true);
 }
 BundleHandle make(const SnapshotDescriptor& metadata,bool fresh,int format=1){
  auto config=stages;for(auto& stage:config)for(auto& f:stage.qsa_formats)f=format;
  WindowStateLayout layout;check(bool(native_window_layout(g,config,metadata.key,metadata.cut,limits().max_tokens,limits().max_payload_bytes,layout,metadata.steering_enabled)),"fixture actual native layout");
  auto r=captures.reserve(host(8<<20),ReservationPurpose::foreground);check(bool(r),"fixture genuine capture admission before data allocation");
  SnapshotDescriptor image=metadata;image.native_layout=layout.native_layout;
  for(const auto& stage:layout.stages){DevicePart part;part.id=stage.id;part.layer_begin=stage.layer_begin;part.layer_end=stage.layer_end;part.cut=image.cut;
   for(const auto& tensor:stage.tensors){TensorPart value;value.layout=tensor;value.payload.resize(size_t(tensor.bytes),uint8_t((fresh?150:10)+unsigned(tensor.kind)+stage.id));
    if(tensor.kind==StatePartKind::ple_previous_tokens){const std::array<int32_t,2> ids=fresh?std::array<int32_t,2>{101,102}:std::array<int32_t,2>{11,12};check(value.payload.visit(0,8,[&](uint8_t* p,size_t n,size_t at){std::memcpy(p,reinterpret_cast<const uint8_t*>(ids.data())+at,n);return true;}),"fixture actual PLE prev sidecar");}
    if(tensor.kind==StatePartKind::ple_current_token){const int32_t id=fresh?103:13;check(value.payload.visit(0,4,[&](uint8_t* p,size_t n,size_t at){std::memcpy(p,reinterpret_cast<const uint8_t*>(&id)+at,n);return true;}),"fixture PLE current sidecar");}
    part.tensors.push_back(std::move(value));}
   image.devices.push_back(std::move(part));}
  BundleHandle bundle;check(bool(BundleHandle::freeze(image,image.key,layout,bundle,r.lease)),"fixture immutable native capture");return bundle;
 }
 NativeSuffixTarget request(){return {target,tokens,5,1000};}
};
static const TensorPart& tensor(const DevicePart& stage,StatePartKind kind,int64_t layer){for(const auto& t:stage.tensors)if(t.layout.kind==kind&&t.layout.layer==layer)return t;std::abort();}
static void success(bool split){
 Fixture f(split);WindowBudget budget(host(64<<20),host(0));PreparedNativeSuffix result;uint64_t quote=0;
 fail_after=0;auto quoted=quote_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),quote);fail_after=-1;
 check(bool(quoted)&&quote>0,"quote performs no allocations");
 check(bool(prepare_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),budget,result)),"actual host merge succeeds");
 const auto& merged=*result.bundle().get();check(merged.cut.forwarded==8&&merged.cut.exact_frontier==4&&merged.cut.accepted==9&&merged.cut.output_watermark==7&&merged.pending_tokens==std::vector<int32_t>{9},"approximate cut and pending fresh input");
 check(merged.key==f.target&&merged.accepted_tokens==f.tokens&&merged.cut.history==window_history_digest(f.tokens),"new target identity/history");
 check(merged.approximation.size()==1&&merged.approximation[0].begin==4&&merged.approximation[0].end==8&&merged.approximation[0].first_unresolved_unix_ms==1000&&merged.approximation[0].ancestor==f.donor.get()->cut.history,"old history lineage and age");
 check(result.stats().reused_suffix_tokens==4&&result.stats().exact_prefix_tokens==4&&result.stats().pending_tokens==1&&result.stats().logical_host_charge==quote,"honest merge counters");
 for(size_t i=0;i<merged.devices.size();++i){check(merged.devices[i].cut==merged.cut,"all stages same approximate cut");
  for(const auto& t:merged.devices[i].tensors){const auto& old=tensor(f.donor.get()->devices[i],t.layout.kind,t.layout.layer);const auto kind=t.layout.kind;
   const bool kv=kind==StatePartKind::qsa_k||kind==StatePartKind::qsa_v||kind==StatePartKind::qsa_k_scale||kind==StatePartKind::qsa_v_scale;
   if(kv||kind==StatePartKind::qsa_pooled){const auto& prefix=tensor(f.prefix.get()->devices[i],kind,t.layout.layer);size_t keep=kv?prefix.payload.size():size_t(f.g.idx_key_dim)*sizeof(float);
    check(t.payload.visit(0,t.payload.size(),[&](const uint8_t* p,size_t n,size_t at){for(size_t j=0;j<n;++j){uint8_t expected=0;const size_t offset=at+j;const auto& source=offset<keep?prefix.payload:old.payload;if(!source.read(&expected,offset,1)||p[j]!=expected)return false;}return true;}),"encoded KV/scales and completed pooled rows splice bytes");
   }else check(t.payload==old.payload,"complete old GDN/PLE/index endpoint retained");
  }
 }
 WindowStateLayout layout;check(bool(native_window_layout(f.g,f.stages,f.target,merged.cut,1024,1<<24,layout,false))&&bool(validate_window_state(merged,f.target,layout)),"merged native descriptor validates");
 check(budget.snapshot().reserved.host_bytes==quote,"real result retained reservation");auto keep=result.bundle();result={};check(budget.snapshot().reserved.host_bytes==quote,"escaped immutable result owns charge");keep={};check(budget.snapshot().reserved.host_bytes==0,"final result returns charge");
 check(f.donor.get()->cut.exact_frontier==8&&f.donor.get()->approximation.empty(),"old exact source unchanged");
}
static void refusals(){
 Fixture f;WindowBudget budget(host(64<<20),host(0));PreparedNativeSuffix out;
 check(bool(prepare_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),budget,out)),"sentinel merge");const auto* saved=out.bundle().get();const auto held=budget.snapshot().reserved.host_bytes;
 auto refusal=[&](NativeSuffixTarget target,const std::vector<NativeWindowStage>& stages,NativeSuffixLimits lim){
  uint64_t q=919;auto quote=quote_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,stages,target,lim,q);check(!quote&&q==919,"quote refusal unchanged");
  auto result=prepare_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,stages,target,lim,budget,out);check(!result&&out.bundle().get()==saved&&budget.snapshot().reserved.host_bytes==held,"merge refusal unchanged result and quota");return result;
 };
 check(refusal({f.target,f.tokens,f.prefix.get()->cut.id,1000},f.stages,limits()).status==NativeSuffixStatus::stale,"composed state needs newer cut than prefix, output unchanged");
 auto key=f.target;key.append_epoch=1;check(refusal({key,f.tokens,5,1000},f.stages,limits()).status==NativeSuffixStatus::stale,"append epoch required");
 auto changed=f.tokens;changed[6]=99;refusal({f.target,changed,5,1000},f.stages,limits());
 auto formats=f.stages;formats[0].qsa_formats[0]=17;refusal(f.request(),formats,limits());formats[0].qsa_formats[0]=0;refusal(f.request(),formats,limits());
 auto lim=limits();lim.max_payload_bytes=1;refusal(f.request(),f.stages,lim);lim=limits();lim.max_metadata_bytes=1;refusal(f.request(),f.stages,lim);
 refusal({f.target,f.tokens,5,0},f.stages,limits());
 // Multiple disjoint edits before aligned B are rejected even though suffix matches.
 auto multiple=f.tokens;multiple[3]=77;auto prefix_meta=f.prefix_meta;prefix_meta.accepted_tokens.assign(multiple.begin(),multiple.begin()+4);prefix_meta.cut.history=window_history_digest(prefix_meta.accepted_tokens);auto prefix=f.make(prefix_meta,true);
 uint64_t q=919;check(!quote_int8_native_suffix(f.donor,f.donor.get()->key,prefix,prefix.get()->key,f.g,f.stages,{f.target,multiple,5,1000},limits(),q)&&q==919,"multiple edits exact fallback");
 {auto metadata=f.donor_meta;metadata.steering_enabled=true;auto steered=f.make(metadata,false);uint64_t quote=9;
  check(quote_int8_native_suffix(steered,steered.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),quote).reason==NativeSuffixReason::unsupported_flags&&quote==9,"real steered source unsupported");}
 {auto donor=f.make(f.donor_meta,false,0);auto result=prepare_int8_native_suffix(donor,donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),budget,out);
  check(result.status==NativeSuffixStatus::exact_fallback&&out.bundle().get()==saved&&budget.snapshot().reserved.host_bytes==held,"trusted runtime format cannot be overridden by imported FP16 layout");}
 auto no_edit=f.donor.get()->accepted_tokens;prefix_meta.accepted_tokens.assign(no_edit.begin(),no_edit.begin()+4);prefix_meta.cut.history=window_history_digest(prefix_meta.accepted_tokens);prefix=f.make(prefix_meta,true);
 check(!quote_int8_native_suffix(f.donor,f.donor.get()->key,prefix,prefix.get()->key,f.g,f.stages,{f.target,no_edit,5,1000},limits(),q),"no edit not misreported suffix experiment");
}
static void boundaries_and_ancestry(){
 Fixture f;
 for(uint64_t remainder=1;remainder<4;++remainder){auto meta=f.prefix_meta;meta.cut.accepted=meta.cut.forwarded=meta.cut.exact_frontier=4+remainder;meta.accepted_tokens.assign(f.tokens.begin(),f.tokens.begin()+4+remainder);meta.cut.history=window_history_digest(meta.accepted_tokens);auto prefix=f.make(meta,true);uint64_t q=7;check(!quote_int8_native_suffix(f.donor,f.donor.get()->key,prefix,prefix.get()->key,f.g,f.stages,f.request(),limits(),q)&&q==7,"each prefix group remainder falls back");}
 for(uint64_t remainder=1;remainder<4;++remainder){auto meta=f.donor_meta;auto tokens=f.tokens;
  for(uint64_t i=0;i<remainder;++i){meta.accepted_tokens.push_back(int32_t(10+i));tokens.push_back(int32_t(10+i));}
  meta.cut.accepted=meta.accepted_tokens.size();meta.cut.forwarded=meta.cut.exact_frontier=meta.cut.accepted-1;meta.pending_tokens={meta.accepted_tokens.back()};meta.cut.history=window_history_digest(meta.accepted_tokens);auto donor=f.make(meta,false);uint64_t quote=7;
  check(!quote_int8_native_suffix(donor,donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,{f.target,tokens,5,1000},limits(),quote)&&quote==7,"each donor endpoint remainder falls back");}
 {auto meta=f.prefix_meta;meta.accepted_tokens.assign(f.tokens.begin(),f.tokens.begin()+8);meta.cut.accepted=meta.cut.forwarded=meta.cut.exact_frontier=8;meta.cut.history=window_history_digest(meta.accepted_tokens);auto prefix=f.make(meta,true);uint64_t quote=7;
  check(!quote_int8_native_suffix(f.donor,f.donor.get()->key,prefix,prefix.get()->key,f.g,f.stages,f.request(),limits(),quote),"suffix less than four refuses");}
 {auto meta=f.prefix_meta;meta.cut.exact_frontier=0;meta.approximation.push_back({0,4,99,{1}});auto prefix=f.make(meta,true);uint64_t quote=7;
  check(quote_int8_native_suffix(f.donor,f.donor.get()->key,prefix,prefix.get()->key,f.g,f.stages,f.request(),limits(),quote).reason==NativeSuffixReason::approximate_parent,"approximate new prefix refuses");}
 auto meta=f.donor_meta;meta.cut.exact_frontier=4;meta.approximation.push_back({4,8,99,{1}});auto parent=f.make(meta,false);uint64_t q=7;
 check(!quote_int8_native_suffix(parent,parent.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),q)&&q==7,"approximate donor refuses without lineage erasure");
 WindowKey stale=f.donor.get()->key;++stale.revision;check(quote_int8_native_suffix(f.donor,stale,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),q).status==NativeSuffixStatus::stale,"stale source explicit");
}
static void budget_and_allocation(){
 Fixture f;uint64_t quote=0;check(bool(quote_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),quote)),"budget quote");
 WindowBudget tiny(host(quote-1),host(0));PreparedNativeSuffix out;auto result=prepare_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),tiny,out);
 check(result.status==NativeSuffixStatus::budget_refused&&!out&&tiny.snapshot().reserved.host_bytes==0,"budget refusal no output");
 WindowBudget budget(host(64<<20),host(0));bool witnessed=false;
 for(long offset:{0,1,2,8,32,128,512,1024}){fail_after=offset;auto r=prepare_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,f.request(),limits(),budget,out);fail_after=-1;
  if(!r){witnessed=true;check(!out&&budget.snapshot().reserved.host_bytes==0,"allocation failure releases introduced reservation");}else {out={};check(budget.snapshot().reserved.host_bytes==0,"successful sample releases");}}
 check(witnessed,"real allocation refusal observed");
}
static void serving_metadata_ceiling(){
 Fixture f(true); // Real two-stage native operand layouts, not an arithmetic-only substitute for the producer.
 const uint64_t scratch=(1ULL<<20)+128*uint64_t(f.tokens.size());
 auto too_small=limits();too_small.max_metadata_bytes=scratch;uint64_t unchanged=919;
 const auto rejected=quote_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,
     f.g,f.stages,f.request(),too_small,unchanged);
 check(rejected.status==NativeSuffixStatus::exact_fallback&&rejected.reason==NativeSuffixReason::metadata_bound&&unchanged==919,
       "serving scratch allowance cannot cover real two-stage native metadata; refusal preserves quote");
 const uint64_t aggregate_cap=64ULL<<20;auto admitted=limits();admitted.max_metadata_bytes=aggregate_cap;
 uint64_t quote=0;
 check(bool(quote_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,
       f.g,f.stages,f.request(),admitted,quote)),"same actual two-stage operands quote under explicit host ceiling");
 WindowBudget budget(host(aggregate_cap),host(0));PreparedNativeSuffix prepared;
 check(bool(prepare_int8_native_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,
       f.g,f.stages,f.request(),admitted,budget,prepared)),"same actual operands prepare through genuine budget reservation");
 check(quote<aggregate_cap&&budget.snapshot().reserved.host_bytes==quote&&prepared.stats().logical_host_charge==quote,
       "producer reserves actual quote, never the aggregate metadata ceiling");
 check(prepared.bundle().get()->approximation.size()==1&&f.donor.get()->approximation.empty()&&
       prepared.bundle().get()->cut.exact_frontier==4&&prepared.bundle().get()->cut.forwarded==8,
       "larger metadata ceiling preserves actual approximate provenance and exact donor");
 prepared={};check(budget.snapshot().reserved.host_bytes==0,"actual quoted producer reservation retires");
 std::printf("metadata ceiling regression: scratch=%llu host_cap=%llu actual_quote=%llu two_stages=1\n",
     (unsigned long long)scratch,(unsigned long long)aggregate_cap,(unsigned long long)quote);
}
int main(){success(false);success(true);refusals();boundaries_and_ancestry();budget_and_allocation();serving_metadata_ceiling();std::puts("PASS native INT8 suffix prepared cases");}
