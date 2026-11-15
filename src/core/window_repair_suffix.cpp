#include "strata/core/window_repair_suffix.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
namespace strata::core {
namespace {
using Status=NativeSuffixStatus;using Reason=NativeSuffixReason;
NativeSuffixResult fail(Reason reason,Status status=Status::exact_fallback) noexcept{return {status,reason};}
bool add(uint64_t& n,uint64_t extra) noexcept {if(extra>UINT64_MAX-n)return false;n+=extra;return true;}
bool times(uint64_t& n,uint64_t count,uint64_t width) noexcept {if(width&&count>UINT64_MAX/width)return false;return add(n,count*width);}
bool string_bytes(uint64_t& n,const std::string& text) noexcept {return add(n,text.size())&&add(n,1);}
bool key_bytes(uint64_t& n,const WindowKey& key) noexcept {
 return add(n,sizeof(WindowKey))&&string_bytes(n,key.session_id)&&string_bytes(n,key.thread_id)&&string_bytes(n,key.window_id);
}
bool same_owner(const WindowKey& a,const WindowKey& b) noexcept {
 return a.engine_generation==b.engine_generation&&a.window_generation==b.window_generation&&a.session_id==b.session_id&&a.thread_id==b.thread_id&&a.window_id==b.window_id&&a.provenance_epoch==b.provenance_epoch&&a.model_profile==b.model_profile&&a.topology==b.topology;
}
bool geometry(const ModelGeometry& g) noexcept {
 if(std::endian::native!=std::endian::little||g.family!=ModelFamily::Qwen4Exp||g.n_layers<=0||g.n_layers>4096||g.qsa_interval<=0||g.idx_block!=4||g.ssm_d_conv<=1||g.head_dim>INT32_MAX||g.head_dim%64)return false;
 const int64_t dimensions[]={g.n_embd,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
 for(auto value:dimensions)if(value<=0)return false;return true;
}
bool flags(const SnapshotDescriptor& s) noexcept {
 return s.speculation_disabled&&!s.steering_enabled&&s.continuation.mode==ContinuationMode::new_generation_policy&&s.continuation.adapter.empty()&&s.continuation.payload.empty();
}
// Count existing bounded metadata without copies/strings/layout construction.
// This is a conservative logical envelope; standard allocator/node/RSS overhead
// is not certified by it. Explicit headroom follows the native capture contract.
bool metadata(const SnapshotDescriptor& s,uint64_t& n) noexcept {
 if(!add(n,sizeof(SnapshotDescriptor))||!key_bytes(n,s.key)||!string_bytes(n,s.native_layout)||
    !times(n,s.accepted_tokens.size(),sizeof(int32_t))||!times(n,s.pending_tokens.size(),sizeof(int32_t))||
    !times(n,s.devices.size(),sizeof(DevicePart)))return false;
 for(const auto& stage:s.devices){
  if(!times(n,stage.tensors.size(),sizeof(TensorPart)))return false;
  for(const auto& tensor:stage.tensors){const auto& t=tensor.layout;
   if(!string_bytes(n,t.dtype)||!string_bytes(n,t.native_layout)||!times(n,t.shape.size(),sizeof(uint64_t))||!times(n,t.strides.size(),sizeof(uint64_t)))return false;
  }
 }
 return true;
}
const TensorPart* find(const DevicePart& stage,StatePartKind kind,int64_t layer) noexcept {
 for(const auto& tensor:stage.tensors)if(tensor.layout.kind==kind&&tensor.layout.layer==layer)return &tensor;return nullptr;
}
bool copy(ConversationBuffer& target,size_t at,const ConversationBuffer& source,size_t count) {
 if(at>target.size()||count>target.size()-at||count>source.size())return false;
 return target.visit(at,count,[&](uint8_t* p,size_t n,size_t offset){return source.read(p,offset-at,n);});
}
struct ChargeOwner {std::shared_ptr<const ReservationLease> admission;std::shared_ptr<const NativeRepairSuffixData> data;};
}
NativeSuffixResult quote_int8_native_repair_suffix(const BundleHandle& donor,const WindowKey& expected_donor,
 const BundleHandle& prefix,const WindowKey& expected_prefix,const ModelGeometry& g,
 const std::vector<NativeWindowStage>& stages,const RepairSpatialCoverage& coverage,std::span<const RepairRegionRootBinding> bindings,NativeRepairSuffixTarget target,NativeSuffixLimits limits,uint64_t& output) noexcept {
 if(!limits.max_tokens||!limits.max_payload_bytes||!limits.max_metadata_bytes)return fail(Reason::invalid_limits);
 if(!donor||!prefix||donor.get()->key!=expected_donor||prefix.get()->key!=expected_prefix)return fail(Reason::stale_operand,Status::stale);
 const auto& old=*donor.get();const auto& fresh=*prefix.get();const auto& key=target.key;
 if(!same_owner(key,old.key)||(!same_owner(fresh.key,old.key)||fresh.key.revision!=old.key.revision||fresh.key.append_epoch>old.key.append_epoch)||key.revision!=old.key.revision||key.append_epoch!=old.key.append_epoch||key.selection_epoch<old.key.selection_epoch||
    !target.cut_id||target.cut_id<=old.cut.id||target.cut_id<=fresh.cut.id)return fail(Reason::invalid_target_identity,Status::stale);
 if(!geometry(g)||g.n_qsa_layers()<=0||stages.empty()||stages.size()>size_t(g.n_layers))return fail(Reason::unsupported_geometry);
 if(!flags(old)||!flags(fresh))return fail(Reason::unsupported_flags);
 if(old.approximation.empty()||!fresh.approximation.empty()||fresh.cut.exact_frontier!=fresh.cut.forwarded)return fail(Reason::approximate_parent);
 const uint64_t n=old.cut.accepted,p=old.cut.exact_frontier,b=fresh.cut.forwarded,f=old.cut.forwarded;
 if(!n||n>limits.max_tokens||old.accepted_tokens.size()!=n||fresh.cut.accepted!=b||fresh.accepted_tokens.size()!=b||!fresh.pending_tokens.empty()||f>n||b>n||
    old.pending_tokens.size()!=n-f)return fail(Reason::invalid_boundaries);
 if(p%4||b%4||f%4||p>=b||b>=f||f-b<4)return fail(Reason::invalid_boundaries);
 if(!coverage.valid()||coverage.begin()!=p||coverage.end()!=f||coverage.ancestry().empty()||
    old.approximation.size()>RepairAncestry::capacity||bindings.size()!=old.approximation.size())return fail(Reason::invalid_operand);
 const auto roots=coverage.ancestry().roots();
 uint64_t frontier=p;
 for(size_t i=0;i<bindings.size();++i){const auto& binding=bindings[i];const auto& region=old.approximation[i];
  if(binding.region!=i||binding.root>=roots.size()||region.begin!=frontier||region.end<=region.begin||region.end>f)return fail(Reason::invalid_operand);
  const auto& root=roots[binding.root];
  if(root.original_history!=region.ancestor||root.first_unresolved_unix_ms!=region.first_unresolved_unix_ms||
     root.engine_generation!=old.key.engine_generation||root.window_generation!=old.key.window_generation)return fail(Reason::invalid_operand);
  frontier=region.end;
 }
 if(frontier!=f)return fail(Reason::invalid_operand);
 for(size_t i=0;i<size_t(n);++i)if(old.accepted_tokens[i]<0||
    (i>=size_t(f)&&old.pending_tokens[i-size_t(f)]!=old.accepted_tokens[i])||
    (i<size_t(b)&&fresh.accepted_tokens[i]!=old.accepted_tokens[i]))return fail(Reason::invalid_operand);
 if(old.devices.size()!=stages.size()||fresh.devices.size()!=stages.size())return fail(Reason::invalid_operand);
 int64_t end=0;uint64_t tensors=0,payload=0,payload_peak=0;
 for(size_t i=0;i<stages.size();++i){const auto& stage=stages[i];
  if(stage.layer_begin!=end||stage.layer_end<=end||stage.layer_end>g.n_layers||stage.qsa_formats.size()>size_t(g.n_layers)+1)return fail(Reason::unsupported_geometry);
  end=stage.layer_end;for(auto format:stage.qsa_formats)if(format!=1)return fail(Reason::unsupported_format);
  if(old.devices[i].cut!=old.cut||fresh.devices[i].cut!=fresh.cut)return fail(Reason::invalid_operand);
  const uint64_t bound=8*uint64_t(stage.layer_end-stage.layer_begin+1)+3;
  if(old.devices[i].tensors.size()>bound||fresh.devices[i].tensors.size()>bound)return fail(Reason::metadata_bound);
  if(!add(tensors,bound))return fail(Reason::overflow);
  for(const auto& tensor:old.devices[i].tensors){
   if(tensor.payload.size()!=tensor.layout.bytes||tensor.layout.shape.size()!=1||tensor.layout.strides.size()!=1)return fail(Reason::invalid_operand);
   if(!add(payload,tensor.payload.size()))return fail(Reason::overflow);
   const ConversationBuffer empty;const size_t peak=empty.allocation_peak(tensor.payload.size());
   if(peak==SIZE_MAX||!add(payload_peak,peak))return fail(Reason::overflow);
  }
  for(const auto& tensor:fresh.devices[i].tensors)if(tensor.payload.size()!=tensor.layout.bytes||tensor.layout.shape.size()!=1||tensor.layout.strides.size()!=1)return fail(Reason::invalid_operand);
 }
 if(end!=g.n_layers)return fail(Reason::unsupported_geometry);
 if(payload>limits.max_payload_bytes)return fail(Reason::payload_bound);
 uint64_t meta=0;if(!metadata(old,meta)||!metadata(fresh,meta)||!key_bytes(meta,key))return fail(Reason::overflow);
 // Three resolved layouts, candidate/frozen descriptor directories/tokens,
 // sequential validation sets and control owner. Extra native metadata headroom
 // is explicitly logical (not an allocator/RSS maximum or admission itself).
 uint64_t introduced=sizeof(ChargeOwner)+sizeof(PreparedNativeRepairSuffix)+sizeof(ReservationLease)+6*sizeof(WindowResources)+8*64*sizeof(uint64_t)+4096;
 if(!times(introduced,meta,8)||!times(introduced,tensors,4*(sizeof(TensorLayout)+8*sizeof(void*)))||
    !times(introduced,uint64_t(g.n_layers),3*sizeof(WindowStateLayout::LayerKind))||
    !times(introduced,stages.size(),1024*1024)||!times(introduced,n,128))return fail(Reason::overflow);
 if(!times(introduced,3,sizeof(NativeRepairSuffixData))||!times(introduced,3*RepairAncestry::capacity,sizeof(RepairRegionRootBinding)+sizeof(ApproximationRegion)))return fail(Reason::overflow);
 if(introduced>limits.max_metadata_bytes)return fail(Reason::metadata_bound);
 if(!times(introduced,payload_peak,2)||introduced>SIZE_MAX)return fail(Reason::overflow);
 output=introduced;return {Status::ok,Reason::none};
}
NativeSuffixResult prepare_int8_native_repair_suffix(const BundleHandle& donor,const WindowKey& expected_donor,
 const BundleHandle& prefix,const WindowKey& expected_prefix,const ModelGeometry& g,
 const std::vector<NativeWindowStage>& stages,const RepairSpatialCoverage& coverage,std::span<const RepairRegionRootBinding> bindings,NativeRepairSuffixTarget target,NativeSuffixLimits limits,
 WindowBudget& budget,PreparedNativeRepairSuffix& output) {
 uint64_t quote=0;auto preflight=quote_int8_native_repair_suffix(donor,expected_donor,prefix,expected_prefix,g,stages,coverage,bindings,target,limits,quote);if(!preflight)return preflight;
 // Input BundleHandles remain immutable and retained by their call owners for
 // every read. Original capture payload/ownership charges are NOT reallocated.
 try {
  // Existing ledger admission transients precede charge: snapshot/request shape
  // copies (<=64 devices), reserve's lease/control metadata. They are priced in
  // quote, not claimed to allocate after charge. No image/layout payload yet.
  WindowResources request;request.device_bytes.resize(budget.snapshot().capacity.device_bytes.size());request.host_bytes=quote;
  auto reserved=budget.reserve(request,ReservationPurpose::foreground);if(!reserved)return fail(Reason::none,reserved.error==ReservationError::allocation_failed?Status::allocation_failed:Status::budget_refused);
  auto charge=std::make_shared<ChargeOwner>();charge->admission=std::move(reserved.lease);
  const auto& old=*donor.get();const auto& fresh=*prefix.get();const uint64_t b=fresh.cut.forwarded,f=old.cut.forwarded;
  StateCut cut=old.cut;cut.id=target.cut_id;cut.exact_frontier=b;
  WindowStateLayout old_layout,prefix_layout,target_layout;
  auto checked=native_window_layout(g,stages,old.key,old.cut,limits.max_tokens,limits.max_payload_bytes,old_layout,false);
  if(!checked)return fail(Reason::invalid_operand,checked.error==WindowStateError::limit_exceeded?Status::allocation_failed:Status::exact_fallback);
  checked=native_window_layout(g,stages,fresh.key,fresh.cut,limits.max_tokens,limits.max_payload_bytes,prefix_layout,false);if(!checked)return fail(Reason::invalid_operand,checked.error==WindowStateError::limit_exceeded?Status::allocation_failed:Status::exact_fallback);
  checked=validate_window_state(old,expected_donor,old_layout);if(!checked)return fail(Reason::invalid_operand,checked.error==WindowStateError::limit_exceeded?Status::allocation_failed:Status::exact_fallback);
  checked=validate_window_state(fresh,expected_prefix,prefix_layout);if(!checked)return fail(Reason::invalid_operand,checked.error==WindowStateError::limit_exceeded?Status::allocation_failed:Status::exact_fallback);
  SnapshotDescriptor merged;
  merged.key=target.key;merged.accepted_tokens=old.accepted_tokens;
  cut.history=window_history_digest(merged.accepted_tokens);merged.cut=cut;
  merged.pending_tokens=old.pending_tokens;merged.native_layout=old.native_layout;
  merged.speculation_disabled=true;merged.steering_enabled=false;
  RepairSpatialCoverage advanced;
  if(advance_repair_spatial_begin(coverage,b,advanced)!=RepairAncestryStatus::ok)return fail(Reason::invalid_operand);
  auto data=std::make_shared<NativeRepairSuffixData>(advanced);
  for(size_t j=0;j<old.approximation.size();++j){auto region=old.approximation[j];if(region.end<=b)continue;
   region.begin=std::max(region.begin,b);data->region_roots.push_back({uint32_t(merged.approximation.size()),bindings[j].root});merged.approximation.push_back(region);
  }
  charge->data=data;
  checked=native_window_layout(g,stages,target.key,cut,limits.max_tokens,limits.max_payload_bytes,target_layout,false);if(!checked)return fail(Reason::invalid_operand,checked.error==WindowStateError::limit_exceeded?Status::allocation_failed:Status::exact_fallback);
  merged.devices.reserve(old.devices.size());
  for(size_t i=0;i<old.devices.size();++i){
   const auto& source=old.devices[i];DevicePart stage;stage.id=source.id;stage.layer_begin=source.layer_begin;stage.layer_end=source.layer_end;stage.cut=cut;stage.tensors.reserve(source.tensors.size());
   for(const auto& tensor:source.tensors){
    TensorPart part;part.layout=tensor.layout;part.payload.resize(tensor.payload.size());
    if(!copy(part.payload,0,tensor.payload,tensor.payload.size()))return fail(Reason::invalid_operand);
    const auto kind=tensor.layout.kind;
    const bool kv=kind==StatePartKind::qsa_k||kind==StatePartKind::qsa_v||kind==StatePartKind::qsa_k_scale||kind==StatePartKind::qsa_v_scale;
    if(kv||kind==StatePartKind::qsa_pooled){
     const auto* p=find(fresh.devices[i],kind,tensor.layout.layer);if(!p)return fail(Reason::invalid_operand);
     uint64_t bytes=p->layout.bytes;
     if(kind==StatePartKind::qsa_pooled){bytes=0;if(!times(bytes,b/4,uint64_t(g.idx_key_dim))||bytes>UINT64_MAX/sizeof(float))return fail(Reason::overflow);bytes*=sizeof(float);}
     if(bytes>SIZE_MAX||bytes>p->payload.size()||bytes>part.payload.size()||!copy(part.payload,0,p->payload,size_t(bytes)))return fail(Reason::invalid_operand);
    }
    stage.tensors.push_back(std::move(part));
   }
   merged.devices.push_back(std::move(stage));
  }
  checked=validate_window_state(merged,target.key,target_layout);if(!checked)return fail(Reason::invalid_operand,checked.error==WindowStateError::limit_exceeded?Status::allocation_failed:Status::exact_fallback);
  PreparedNativeRepairSuffix prepared;
  checked=BundleHandle::freeze(merged,target.key,target_layout,prepared.bundle_,charge);
  if(!checked)return fail(Reason::invalid_operand,checked.error==WindowStateError::limit_exceeded?Status::allocation_failed:Status::exact_fallback);
  prepared.data_=std::move(data);static_assert(std::is_nothrow_move_assignable_v<PreparedNativeRepairSuffix>);output=std::move(prepared);
  return {Status::ok,Reason::none};
 }catch(const std::bad_alloc&){return fail(Reason::none,Status::allocation_failed);}
 catch(const std::length_error&){return fail(Reason::overflow,Status::allocation_failed);}
}
} // namespace strata::core
