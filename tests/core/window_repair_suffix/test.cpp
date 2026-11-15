#include "strata/core/window_repair_suffix.hpp"
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
 std::vector<int32_t> tokens{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
 BundleHandle donor,prefix;
 Fixture(bool split=false){
  if(split)stages={{0,4,{1},true},{4,8,{1},false}};else stages={{0,8,{1,1},true}};
  auto& d=donor_meta;d.key.engine_generation=1;d.key.window_generation=1;d.key.session_id="s";d.key.thread_id="t";d.key.window_id="w";
  d.key.revision=d.key.append_epoch=d.key.selection_epoch=d.key.provenance_epoch=1;d.key.model_profile[0]=1;d.key.topology[0]=2;
  d.accepted_tokens=tokens;d.pending_tokens={17};d.cut={3,17,16,4,7,window_history_digest(d.accepted_tokens)};d.approximation={{4,8,1000,d.cut.history},{8,16,2000,d.cut.history}};d.steering_enabled=false;
  target=d.key;target.selection_epoch=2;
  auto& p=prefix_meta;p.key=d.key;p.accepted_tokens.assign(tokens.begin(),tokens.begin()+8);p.cut={4,8,8,8,0,window_history_digest(p.accepted_tokens)};p.steering_enabled=false;
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
};
static const TensorPart& tensor(const DevicePart& stage,StatePartKind kind,int64_t layer){for(const auto& t:stage.tensors)if(t.layout.kind==kind&&t.layout.layer==layer)return t;std::abort();}

static void exercise(bool split){
 Fixture f(split);WindowBudget budget(host(128<<20),host(0));
 std::array<RepairCausalRoot,3> roots{};
 for(size_t i=0;i<3;++i){auto& r=roots[i];r.identity[0]=uint8_t(i+1);r.original_history=f.donor.get()->cut.history;
  r.engine_generation=r.window_generation=r.runtime_incarnation=1;r.original_cut=1;r.original_begin=4;r.original_count=12;
  r.first_unresolved_unix_ms=1000*(i+1);r.monotonic_birth={1,uint64_t(i+1)};}
 RepairAncestry ancestry;check(make_repair_ancestry(roots,{64,RepairAncestry::retained_bytes()},ancestry)==RepairAncestryStatus::ok,"bounded root table");
 RepairSpatialCoverage coverage;check(make_repair_spatial_coverage(ancestry,4,16,coverage)==RepairAncestryStatus::ok,"bound donor spatial interval");
 std::array<RepairRegionRootBinding,2> bindings{{{0,0},{1,1}}};NativeRepairSuffixTarget target{f.target,5};
 uint64_t quote=0;fail_after=0;auto q=quote_int8_native_repair_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,coverage,bindings,target,limits(),quote);fail_after=-1;
 check(bool(q)&&quote,"no-allocation quote");PreparedNativeRepairSuffix out;
 check(bool(prepare_int8_native_repair_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,coverage,bindings,target,limits(),budget,out)),"prepare");
 const auto& image=*out.bundle().get();check(image.cut.exact_frontier==8&&image.cut.forwarded==16&&image.cut.accepted==17&&image.cut.output_watermark==7,"complete donor counts retained");
 check(image.accepted_tokens==f.tokens&&image.pending_tokens==std::vector<int32_t>{17},"current history and pending retained");
 check(out.data()->coverage.begin()==8&&out.data()->coverage.end()==16&&out.data()->coverage.ancestry().roots().size()==3,"trimmed spatial range retains both causal roots");
 check(out.data()->coverage.ancestry().roots()[0]==roots[0]&&out.data()->region_roots.size()==1&&out.data()->region_roots[0].root==1,"root removed from spatial map still retained unchanged");
 check(repair_ancestry_age(out.data()->coverage.ancestry(),{1,10}).oldest_unresolved_unix_ms==1000,"global oldest birth preserved");
 for(size_t i=0;i<image.devices.size();++i)for(const auto& t:image.devices[i].tensors){const auto kind=t.layout.kind;const auto& old=tensor(f.donor.get()->devices[i],kind,t.layout.layer);
  bool kv=kind==StatePartKind::qsa_k||kind==StatePartKind::qsa_v||kind==StatePartKind::qsa_k_scale||kind==StatePartKind::qsa_v_scale;
  if(kv||kind==StatePartKind::qsa_pooled){const auto& exact=tensor(f.prefix.get()->devices[i],kind,t.layout.layer);size_t prefix_bytes=kv?exact.payload.size():size_t(2*f.g.idx_key_dim)*sizeof(float);
   check(t.payload.visit(0,t.payload.size(),[&](const uint8_t* b,size_t n,size_t at){for(size_t j=0;j<n;++j){uint8_t v=0;size_t off=at+j;if(!(off<prefix_bytes?exact.payload:old.payload).read(&v,off,1)||v!=b[j])return false;}return true;}),"prefix encoded bytes only");
  }else check(t.payload==old.payload,"current recurrent PLE tail spare position unchanged");}
 const auto* saved=out.bundle().get();auto held=budget.snapshot().reserved.host_bytes;
 auto refuse=[&](std::span<const RepairRegionRootBinding> b,NativeRepairSuffixTarget t,NativeSuffixLimits lim){uint64_t untouched=123;
  auto r=quote_int8_native_repair_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,coverage,b,t,lim,untouched);check(!r&&untouched==123,"quote refusal unchanged");
  check(!prepare_int8_native_repair_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,coverage,b,t,lim,budget,out)&&out.bundle().get()==saved&&budget.snapshot().reserved.host_bytes==held,"prepare refusal unchanged");};
 auto wrong=bindings;wrong[0].root=1;refuse(wrong,target,limits());refuse(bindings,{f.target,4},limits());auto lim=limits();lim.max_metadata_bytes=1;refuse(bindings,target,lim);
 // New operand refusals preserve the already prepared result and its charge.
 auto operand_refusal=[&](const BundleHandle& d,const WindowKey& expected,const BundleHandle& p,const std::vector<NativeWindowStage>& stages){uint64_t q=321;
  check(!quote_int8_native_repair_suffix(d,expected,p,p.get()->key,f.g,stages,coverage,bindings,target,limits(),q)&&q==321,"operand quote refusal unchanged");
  check(!prepare_int8_native_repair_suffix(d,expected,p,p.get()->key,f.g,stages,coverage,bindings,target,limits(),budget,out)&&out.bundle().get()==saved&&budget.snapshot().reserved.host_bytes==held,"operand prepare refusal unchanged");};
 auto bad_meta=f.prefix_meta;bad_meta.accepted_tokens[1]=99;bad_meta.cut.history=window_history_digest(bad_meta.accepted_tokens);
 operand_refusal(f.donor,f.donor.get()->key,f.make(bad_meta,true),f.stages);
 bad_meta=f.prefix_meta;bad_meta.cut.exact_frontier=4;bad_meta.approximation={{4,8,1000,roots[0].original_history}};
 operand_refusal(f.donor,f.donor.get()->key,f.make(bad_meta,true),f.stages);
 auto stale=f.donor.get()->key;++stale.append_epoch;operand_refusal(f.donor,stale,f.prefix,f.stages);
 auto formats=f.stages;formats[0].qsa_formats[0]=0;operand_refusal(f.donor,f.donor.get()->key,f.prefix,formats);
 bad_meta=f.prefix_meta;bad_meta.accepted_tokens.resize(9,9);bad_meta.cut={4,9,9,9,0,window_history_digest(bad_meta.accepted_tokens)};
 operand_refusal(f.donor,f.donor.get()->key,f.make(bad_meta,true),f.stages);
 bad_meta=f.prefix_meta;bad_meta.accepted_tokens.assign(f.tokens.begin(),f.tokens.begin()+16);bad_meta.cut={4,16,16,16,0,window_history_digest(bad_meta.accepted_tokens)};
 operand_refusal(f.donor,f.donor.get()->key,f.make(bad_meta,true),f.stages);
 // Generic freeze is not trusted native admission. Supply a self-consistent
 // zero-sized tensor specification, then require actual native rejection.
 for(bool bad_prefix:{false,true}){
  const auto& source=bad_prefix?f.prefix:f.donor;
  auto admission=f.captures.reserve(host(8<<20),ReservationPurpose::foreground);check(bool(admission),"malformed DATA fixture admission before descriptor copy");
  SnapshotDescriptor malformed=*source.get();WindowStateLayout self_layout;
  check(bool(native_window_layout(f.g,f.stages,malformed.key,malformed.cut,1024,1<<24,self_layout,false)),"malformed fixture starts with native inventory");
  auto& spec=self_layout.stages[0].tensors[0];spec.bytes=0;spec.valid_elements=0;spec.shape={0};spec.strides={1};
  malformed.devices[0].tensors[0].layout=spec;malformed.devices[0].tensors[0].payload=ConversationBuffer{};
  BundleHandle frozen;check(bool(BundleHandle::freeze(malformed,malformed.key,self_layout,frozen,admission.lease)),"generic caller-provided layout freeze permits zero tensor DATA");
  const auto& d=bad_prefix?f.donor:frozen;const auto& prefix=bad_prefix?frozen:f.prefix;
  uint64_t q=0;check(bool(quote_int8_native_repair_suffix(d,d.get()->key,prefix,prefix.get()->key,f.g,f.stages,coverage,bindings,target,limits(),q)),"bounded metadata quote does not authenticate native tensor meaning");
  const auto* paired=out.data();auto result=prepare_int8_native_repair_suffix(d,d.get()->key,prefix,prefix.get()->key,f.g,f.stages,coverage,bindings,target,limits(),budget,out);
  check(!result&&result.reason==NativeSuffixReason::invalid_operand&&out.bundle().get()==saved&&out.data()==paired&&budget.snapshot().reserved.host_bytes==held,"trusted native validation rejects before candidate clone; saved bytes paired DATA and budget unchanged");
 }
 // SECOND attachment: synthetic DATA-only foreground transition, no native
 // execution claimed. Current endpoint bytes deliberately differ from original.
 {
  auto current=f.donor_meta;current.key=image.key;++current.key.append_epoch;
  current.accepted_tokens=f.tokens;current.accepted_tokens.insert(current.accepted_tokens.end(),{18,19,20,21});
  current.pending_tokens={21};current.cut={6,21,20,8,11,window_history_digest(current.accepted_tokens)};
  current.approximation=image.approximation;current.approximation.back().end=20;
  auto current_bundle=f.make(current,true);
  auto exact=f.prefix_meta;exact.key=current.key;exact.accepted_tokens.assign(current.accepted_tokens.begin(),current.accepted_tokens.begin()+12);
  exact.cut={7,12,12,12,0,window_history_digest(exact.accepted_tokens)};auto exact_bundle=f.make(exact,false);
  RepairSpatialCoverage current_coverage;check(make_repair_spatial_coverage(out.data()->coverage.ancestry(),8,20,current_coverage)==RepairAncestryStatus::ok,"synthetic appended DATA coverage inherits full roots");
  auto key=current.key;++key.selection_epoch;PreparedNativeRepairSuffix second;
  check(bool(prepare_int8_native_repair_suffix(current_bundle,current.key,exact_bundle,exact.key,f.g,f.stages,current_coverage,out.data()->region_roots,{key,8},limits(),budget,second)),"second actual DATA producer invocation");
  const auto& m=*second.bundle().get();check(m.cut.forwarded==20&&m.cut.exact_frontier==12&&m.cut.accepted==21&&m.cut.output_watermark==11&&m.accepted_tokens==current.accepted_tokens&&m.pending_tokens==current.pending_tokens&&m.cut.history==current.cut.history,"second preserves CURRENT history pending watermark");
  check(second.data()->coverage.ancestry().roots().size()==3&&second.data()->coverage.ancestry().roots()[0]==roots[0]&&second.data()->coverage.ancestry().roots()[2]==roots[2],"second retains trimmed root0 and never-spatially-bound root2");
  for(size_t i=0;i<m.devices.size();++i)for(const auto& t:m.devices[i].tensors){const auto kind=t.layout.kind;
   bool kv=kind==StatePartKind::qsa_k||kind==StatePartKind::qsa_v||kind==StatePartKind::qsa_k_scale||kind==StatePartKind::qsa_v_scale;
   const auto& now=tensor(current_bundle.get()->devices[i],kind,t.layout.layer);
   if(kv||kind==StatePartKind::qsa_pooled){const auto& e=tensor(exact_bundle.get()->devices[i],kind,t.layout.layer);size_t prefix_bytes=kv?e.payload.size():size_t(3*f.g.idx_key_dim)*sizeof(float);
    check(t.payload.visit(0,t.payload.size(),[&](const uint8_t* bytes,size_t n,size_t at){for(size_t j=0;j<n;++j){uint8_t expected=0;size_t off=at+j;if(!(off<prefix_bytes?e.payload:now.payload).read(&expected,off,1)||bytes[j]!=expected)return false;}return true;}),"second exact prefix and CURRENT suffix bytes");
   }else {check(t.payload==now.payload,"second retains CURRENT full endpoint");check(!(t.payload==tensor(f.donor.get()->devices[i],kind,t.layout.layer).payload),"second endpoint differs from original donor");}
  }
 }
 check(budget.snapshot().reserved.host_bytes==held,"second temporary result releases own charge");
 WindowBudget empty(host(0),host(0));check(!prepare_int8_native_repair_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,coverage,bindings,target,limits(),empty,out)&&out.bundle().get()==saved,"global budget refusal preserves existing output");
 // Exercise real refusal rollback across progressively later allocation sites.
 bool reached=false;for(long point=0;point<4096;++point){PreparedNativeRepairSuffix trial;fail_after=point;
  auto r=prepare_int8_native_repair_suffix(f.donor,f.donor.get()->key,f.prefix,f.prefix.get()->key,f.g,f.stages,coverage,bindings,target,limits(),budget,trial);fail_after=-1;
  if(r){reached=true;break;}check(!trial&&budget.snapshot().reserved.host_bytes==held,"allocation refusal has no output or leaked reservation");}
 check(reached,"fault sweep reaches successful preparation");
 auto keep=out.bundle();out={};check(budget.snapshot().reserved.host_bytes==held,"escaped bundle retains roots and charge");keep={};check(budget.snapshot().reserved.host_bytes==0,"final release returns charge");
}
int main(){exercise(false);exercise(true);std::puts("native repair suffix DATA fixture PASS");}
