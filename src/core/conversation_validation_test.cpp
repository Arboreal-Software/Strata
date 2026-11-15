// Host-only validation fixtures. CUDA is linked but must never be initialized:
// invalid restores must return before the first CUDA call or destination write.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "conversation_copy.hpp"
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <barrier>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <cstring>

#if defined(CONVERSATION_TEST_TRANSFERS)
// Real source methods, host allocations and modeled CUDA completion only.
// No CUDA context, physical DMA, device-fit, or recovery claim.
#include <cerrno>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>
namespace {
std::atomic<int> copy_calls{0},sync_calls{0},fail_copy{0},fail_sync{0};
std::atomic<size_t> copied_bytes{0};
thread_local int recorded_device=0;
struct Stream {int device;};
struct Event {int device;};
std::mutex gate_mu;
std::condition_variable gate_cv;
bool gate_host=false,gate_seen=false,join_seen=false,restore_done=false,gate_timeout=false;
std::atomic<bool> fail_launch{false},fail_event{false};
int fault_marker=-1;
strata::core::ConversationCheckpoint* fault_checkpoint=nullptr;
std::atomic<void*> watched_control{nullptr};
std::atomic<bool> owner_drop_seen{false},owner_free_seen{false},owner_free_early{false};
void marker(char value){if(fault_marker>=0)(void)::write(fault_marker,&value,1);}
}
// Observe the actual final allocation free, not just an ownership callback.
void operator delete(void* p) noexcept {
    if(p&&p==watched_control.load()){
        owner_free_early=!owner_drop_seen.load();owner_free_seen=true;
    }
    std::free(p);
}
void operator delete(void* p,std::size_t) noexcept {::operator delete(p);}
// This Linux/ELF test-only public-symbol boundary injects an actual failed
// std::async thread launch; production uses the ordinary standard library.
extern "C" int pthread_create(pthread_t* t,const pthread_attr_t* a,void* (*fn)(void*),void* arg) noexcept {
    if(fail_launch.exchange(false)){
        marker(fault_checkpoint&&fault_checkpoint->pending?'L':'B');return EAGAIN;
    }
    using Create=int (*)(pthread_t*,const pthread_attr_t*,void* (*)(void*),void*);
    static const auto actual=reinterpret_cast<Create>(dlsym(RTLD_NEXT,"pthread_create"));
    return actual?actual(t,a,fn,arg):EAGAIN;
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst,const void* src,size_t n,cudaMemcpyKind){
    if(++copy_calls==fail_copy){marker('C');return cudaErrorInvalidValue;}
    copied_bytes+=n;std::memcpy(dst,src,n);return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetDevice(int* d){*d=recorded_device;return cudaSuccess;}
extern "C" cudaError_t __wrap_cudaSetDevice(int d){recorded_device=d;return cudaSuccess;}
extern "C" cudaError_t __wrap_cudaMalloc(void** p,size_t n){*p=std::malloc(n);return *p?cudaSuccess:cudaErrorMemoryAllocation;}
extern "C" cudaError_t __wrap_cudaMallocHost(void** p,size_t n){return __wrap_cudaMalloc(p,n);}
extern "C" cudaError_t __wrap_cudaFree(void* p){std::free(p);return cudaSuccess;}
extern "C" cudaError_t __wrap_cudaFreeHost(void* p){return __wrap_cudaFree(p);}
extern "C" cudaError_t __wrap_cudaMemGetInfo(size_t* free,size_t* total){*free=size_t(8)<<30;*total=size_t(16)<<30;return cudaSuccess;}
extern "C" cudaError_t __wrap_cudaPointerGetAttributes(cudaPointerAttributes* a,const void*){
    *a={};a->type=cudaMemoryTypeHost;return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* s,unsigned int){
    *s=reinterpret_cast<cudaStream_t>(new Stream{recorded_device});return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaStreamDestroy(cudaStream_t s){delete reinterpret_cast<Stream*>(s);return cudaSuccess;}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst,const void* src,size_t n,cudaMemcpyKind kind,cudaStream_t s){
    if(s&&reinterpret_cast<Stream*>(s)->device!=recorded_device)return cudaErrorInvalidDevice;
    return __wrap_cudaMemcpy(dst,src,n,kind);
}
extern "C" cudaError_t __wrap_cudaMemsetAsync(void* dst,int value,size_t n,cudaStream_t s){
    if(s&&reinterpret_cast<Stream*>(s)->device!=recorded_device)return cudaErrorInvalidDevice;
    std::memset(dst,value,n);return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t s){
    return !s||reinterpret_cast<Stream*>(s)->device==recorded_device?cudaSuccess:cudaErrorInvalidDevice;
}
extern "C" cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* e,unsigned int){
    *e=reinterpret_cast<cudaEvent_t>(new Event{recorded_device});return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t e,cudaStream_t s){
    return reinterpret_cast<Event*>(e)->device==recorded_device&&
        (!s||reinterpret_cast<Stream*>(s)->device==recorded_device)?cudaSuccess:cudaErrorInvalidDevice;
}
extern "C" cudaError_t __wrap_cudaEventSynchronize(cudaEvent_t e){
    if(reinterpret_cast<Event*>(e)->device!=recorded_device)return cudaErrorInvalidDevice;
    if(fail_event.exchange(false)){marker('E');return cudaErrorUnknown;}
    std::unique_lock<std::mutex> lock(gate_mu);
    if(gate_host){gate_seen=true;gate_cv.notify_all();gate_cv.wait(lock,[]{return !gate_host;});}
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t e){delete reinterpret_cast<Event*>(e);return cudaSuccess;}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize(){return ++sync_calls==fail_sync?cudaErrorUnknown:cudaSuccess;}
extern "C" cudaError_t __wrap_cudaGetLastError(){return cudaSuccess;}
extern "C" const char* __wrap_cudaGetErrorString(cudaError_t e){
    return e==cudaSuccess?"success":"recorded CUDA error";
}
extern "C" cudaError_t __wrap_cudaStreamWaitEvent(cudaStream_t,cudaEvent_t,unsigned int){
    std::fputs("unsupported recorded stream-wait branch\n",stderr);std::abort();
}
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
struct Pools {
    QsaState st;
    std::array<std::vector<uint8_t>, 8> data;
    Pools(const ModelGeometry& g, int format) {
        st.max_cells = 96; st.n_pages = st.n_slots = 24; st.idx_pooled_rows = 26;
        st.kv_int8 = format == 1; st.kv_q4 = format == 2;
        const size_t per = format == 2 ? strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                      : g.head_dim * (format == 1 ? 1 : 2);
        data[0].resize(96 * g.n_head_kv * per, 0xa5); data[1] = data[0];
        data[2].resize(format == 1 ? 96 * g.n_head_kv * (g.head_dim / 64) * 2 : 0, 0xa5); data[3] = data[2];
        data[4].resize(26 * g.idx_key_dim * 4, 0xa5);
        data[5].resize(3 * g.idx_key_dim * 4, 0xa5);
        data[6].resize(g.idx_key_dim * 4, 0xa5); data[7].resize(4, 0xa5);
        if (format == 2) { st.k_q4 = data[0].data(); st.v_q4 = data[1].data(); }
        else if (format == 1) {
            st.k_q = (int8_t*) data[0].data(); st.v_q = (int8_t*) data[1].data();
            st.k_scale = (uint16_t*) data[2].data(); st.v_scale = (uint16_t*) data[3].data();
        } else { st.k_pool = (uint16_t*) data[0].data(); st.v_pool = (uint16_t*) data[1].data(); }
        st.idx_pooled = (float*) data[4].data(); st.idx_tail = (float*) data[5].data();
        st.idx_dead = (float*) data[6].data(); st.idx_block_pos = (int32_t*) data[7].data();
    }
    ConversationKv image(const ModelGeometry& g, bool index) const {
        ConversationKv k;
        k.format = qsa_kv_format(st); k.cells = 12; k.page_size = 4;
        k.heads = g.n_head_kv; k.head_dim = g.head_dim; k.idx_dim = g.idx_key_dim;
        k.pooled_rows = index ? 3 : 0;
        k.k.resize(data[0].size() / 8, 13); k.v = k.k;
        k.k_scale.resize(data[2].size() / 8, 13); k.v_scale = k.k_scale;
        k.pooled.resize(k.pooled_rows * g.idx_key_dim * 4, 13);
        return k;
    }
};

void fixture(int format, int experts, bool zero_qsa, bool ple) {
    ModelGeometry g;
    g.n_layers = zero_qsa ? 3 : 8; g.n_expert = experts;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::string error;
    ConversationStateSizes z;
    check(conversation_state_sizes(g, z, error), "checked canonical/pruned/zero-QSA sizing");
    Pools first(g, format), last(g, format), draft(g, format);
    std::array<QsaState, 2> layers{first.st, last.st};
    std::vector<uint8_t> gdn(z.gdn, 0xa5), history(ple ? z.ple : 0, 0xa5);
    SessionState ss;
    ss.max_cells = 96; ss.gdn_state = (float*) gdn.data();
    // the whole-model carve, as session_init leaves a one-GPU session
    ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
    ss.ple_hist = ple ? (float*) history.data() : nullptr;
    ss.qsa_states = zero_qsa ? nullptr : layers.data();
    SavedConversation image;
    image.geometry = {g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                      g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                      g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    image.layer_hi = g.n_layers;
    auto checkpoint = [&](size_t tokens) {
        ConversationCheckpoint c;
        c.ids.resize(tokens);
        for (size_t i = 0; i < tokens; ++i) c.ids[i] = (int32_t) i + 1;
        c.imgs = {{1, 44}};
        c.gdn.resize(z.gdn, 13); c.ple.resize(ple ? z.ple : 0, 13);
        c.tails.resize(g.n_qsa_layers() * z.tail, 13); c.dead.resize(g.n_qsa_layers() * z.dead, 13);
        c.block_pos.resize(g.n_qsa_layers() * z.block_pos, 13);
        return c;
    };
    image.live = checkpoint(9); image.checkpoints.push_back(checkpoint(5));
    image.kv.reserve((size_t) g.n_qsa_layers() + 1);
    if (!zero_qsa) { image.kv.push_back(first.image(g, true)); image.kv.push_back(last.image(g, true)); }
    image.kv.push_back(draft.image(g, false));
    check(conversation_snapshot_validate(image, ss, g, draft.st, error), "complete image validates without CUDA");
    size_t estimate = 0;
    check(conversation_snapshot_bytes({image.live.ids,image.live.imgs,image.checkpoints,true},ss,g,draft.st,estimate,error),
          "capture estimate works without CUDA");
    // Public capture reserves the future live-checkpoint and whole-image
    // completion controls plus one per actual K/V part. This fixture currently
    // holds synchronous payloads, so its resident image has none of those yet.
    const size_t pending_control_peak = (image.kv.size() + 2) * ConversationPending::allocation_bytes();
    check(estimate == image.bytes() + pending_control_peak, "estimate covers checkpoint/indexer/spare payloads");
    auto unchanged = [&] {
        auto pristine = [](const auto& bytes) { return std::all_of(bytes.begin(),bytes.end(),[](uint8_t b){return b==0xa5;}); };
        if (!pristine(gdn) || !pristine(history) || ss.ple_prev[0] != -1 || ss.ple_prev[1] != -1) return false;
        for (const auto* p : {&first,&last,&draft}) for (const auto& bytes : p->data) if (!pristine(bytes)) return false;
        return true;
    };
    auto reject = [&](const std::function<void(SavedConversation&)>& mutate, const char* label) {
        auto bad = image; mutate(bad);
        check(conversation_snapshot_restore(bad,ss,g,draft.st,error) == ConversationRestore::invalid,label);
        check(unchanged(), "invalid restore did not touch any layer or running-state buffer");
    };
    reject([](auto& s){s.kv.back().k.pop_back();}, "late draft corruption rejected before first layer write");
    reject([](auto& s){s.geometry[16] = 128;}, "different expert geometry rejected");
    reject([](auto& s){s.live.stage_parts.emplace_back();}, "layer-split live state rejected before writes");
    reject([](auto& s){s.checkpoints[0].stage_parts.emplace_back();}, "layer-split checkpoint rejected before writes");
    reject([](auto& s){s.kv.pop_back();}, "missing KV layer rejected");
    reject([](auto& s){s.live.gdn.pop_back();}, "bad live recurrence rejected");
    reject([](auto& s){s.checkpoints.back().gdn.pop_back();}, "bad retained checkpoint rejected");
    reject([](auto& s){s.checkpoints.back().ids[0] = 99;}, "foreign checkpoint prefix rejected");
    reject([](auto& s){s.checkpoints.back().imgs[0].hash++;}, "foreign checkpoint image rejected");
    reject([](auto& s){s.live.imgs[0].start = -1;}, "negative image position rejected");
    reject([](auto& s){s.live.ids[0] = -1;}, "negative live token rejected");
    reject([](auto& s){s.live.ple.push_back(0);}, "PLE size/enabled mismatch rejected");
    reject([](auto& s){s.live.tails.push_back(0);}, "tail size/zero-QSA mismatch rejected");
    if (!zero_qsa) {
        reject([](auto& s){s.kv[1].pooled.pop_back();}, "late indexer spare row corruption rejected");
        reject([](auto& s){s.live.dead.pop_back();}, "missing indexer spare key rejected");
        reject([](auto& s){s.live.block_pos.pop_back();}, "missing indexer metadata rejected");
        ss.qsa_states = nullptr;
        check(!conversation_snapshot_validate(image,ss,g,draft.st,error), "missing QSA targets rejected");
        ss.qsa_states = layers.data();
        reject([](auto& s){s.layer_lo = 4;}, "snapshot from another layer range rejected");
        // #216's carve: a split stage owning layers [4, 8) holds GDN rows 3..5 and QSA ordinal 1 only
        SessionState stage = ss;
        stage.layer_lo = 4; stage.gdn_alloc = 3; stage.qsa_ord0 = 1; stage.qsa_alloc = 1;
        ConversationStateSizes zs;
        check(conversation_session_sizes(g,stage,zs,error) && zs.gdn == z.gdn / 6 * 3 && zs.tail == z.tail,
              "carved session sizes cover the owned rows only");
        check(!conversation_checkpoint_validate(image.live,stage,g,error), "whole-model checkpoint rejected by a carve");
        ConversationCheckpoint part = image.live;
        part.gdn.resize(zs.gdn); part.tails.resize(zs.tail); part.dead.resize(zs.dead);
        part.block_pos.resize(zs.block_pos);
        check(conversation_checkpoint_validate(part,stage,g,error), "carve-sized checkpoint validates");
        check(!conversation_snapshot_validate(image,stage,g,draft.st,error), "whole-model snapshot rejected by a carve");
        stage.qsa_alloc = 2;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the last QSA ordinal rejected");
        stage.qsa_alloc = 1; stage.gdn_alloc = 7;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the GDN rows rejected");
    }
    auto bad_geometry = g; bad_geometry.ssm_state_size = std::numeric_limits<int64_t>::max();
    check(!conversation_state_sizes(bad_geometry,z,error), "running-state arithmetic overflow rejected");
    bad_geometry = g; bad_geometry.qsa_interval = 0;
    check(!conversation_state_sizes(bad_geometry,z,error), "zero layer interval rejected before division");
    bad_geometry = g; bad_geometry.n_head_kv = std::numeric_limits<int64_t>::max();
    check(conversation_kv_bytes(draft.st,bad_geometry,9,false)==0, "KV byte overflow rejected");
#if defined(CONVERSATION_TEST_TRANSFERS)
    auto reset = [&] {
        for (auto* p : {&first,&last,&draft}) for (auto& bytes : p->data) std::fill(bytes.begin(),bytes.end(),0xa5);
        std::fill(gdn.begin(),gdn.end(),0xa5); std::fill(history.begin(),history.end(),0xa5);
        ss.ple_prev[0] = ss.ple_prev[1] = -1;
        copy_calls = sync_calls = fail_copy = fail_sync = 0;
    };
    reset(); fail_sync = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "pre-transfer CUDA synchronization failure is fatal");
    check(copy_calls==0 && unchanged(), "pre-sync failure did not mutate state");
    reset(); fail_copy = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "first CUDA transfer failure is fatal");
    check(unchanged(), "first-copy failure did not mutate state");
    reset(); fail_copy = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "partial CUDA transfer failure is fatal, not an invalid-image fallback");
    check(!unchanged(), "fault fixture genuinely produced partial state");
    reset(); fail_sync = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "post-transfer CUDA synchronization failure is fatal");
    check(copy_calls>0 && !unchanged(), "post-sync failure occurred after state writes");
    reset();
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::restored,
          "injected host transfer backend completes a valid restore");
    check(gdn==image.live.gdn && history==image.live.ple && ss.ple_prev[0]==8 && ss.ple_prev[1]==9,
          "successful restore publishes correct running state and PLE window");
    uint64_t fingerprint = 0;
    check(conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back verifies complete draft payload");
    draft.data[0][0] ^= 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back detects corrupted draft byte");
    draft.data[0][0] ^= 1;
    fail_copy = copy_calls + 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back transfer failure is not a successful fingerprint");
    reset(); copied_bytes = 0;
    const ConversationView view{image.live.ids,image.live.imgs,image.checkpoints,true};
    SavedConversation full,incremental;
    check(conversation_snapshot_save(full,view,ss,g,draft.st,error), "full capture through host transfer backend");
    const size_t full_copies = copied_bytes;
    ConversationKvReuse reuse{full.kv,9,9};
    size_t peak = 0, reused = 0;
    check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "incremental capture admission without transfers");
    copied_bytes = 0;
    const bool saved_incrementally = conversation_snapshot_save(incremental,view,ss,g,draft.st,error,std::move(reuse),&reused);
    if (!saved_incrementally) std::fprintf(stderr,"capture error: %s\n",error.c_str());
    check(saved_incrementally, "incremental capture through host backend");
    check(reused > 0 && copied_bytes + reused == full_copies, "reused byte count measures transfers actually omitted");
    check(incremental.bytes() <= peak && incremental.live.gdn == full.live.gdn, "running state refreshed within admitted allocation");
    for (size_t i=0;i<full.kv.size();++i)
        check(incremental.kv[i].k==full.kv[i].k && incremental.kv[i].v==full.kv[i].v &&
              incremental.kv[i].k_scale==full.kv[i].k_scale && incremental.kv[i].v_scale==full.kv[i].v_scale &&
              incremental.kv[i].pooled==full.kv[i].pooled, "incremental and full capture agree across every payload");
    reuse = {full.kv,9,9};
    reuse.kv.back().k.pop_back();
    copy_calls = sync_calls = 0;
    check(!conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "malformed retained draft rejected before admission");
    check(!copy_calls && !sync_calls, "invalid retained storage performs no CUDA calls");
    reuse = {full.kv,9,9};
    fail_copy = copy_calls + 1;
    SavedConversation unpublished;
    check(!conversation_snapshot_save(unpublished,view,ss,g,draft.st,error,std::move(reuse)), "incremental capture copy failure is reported");
    check(unpublished.kv.empty() && unpublished.live.ids.empty(), "failed incremental capture cannot publish a partial image");
    copy_calls = sync_calls = fail_copy = fail_sync = 0;
    SavedConversation private_capture;
    check(conversation_snapshot_save_private(private_capture,view,ss,g,error),
          "transaction-owned native capture succeeds without draft state");
    check(private_capture.kv.size()==size_t(g.n_qsa_layers()) && private_capture.live.ids==view.ids,
          "private capture includes owned layers and forwarded history");
    copy_calls = sync_calls = 0; fail_copy = 1;
    SavedConversation retained_partial;
    check(!conversation_snapshot_save_private(retained_partial,view,ss,g,error),
          "transaction-owned capture reports injected copy failure");
    check(retained_partial.live.ids==view.ids && !retained_partial.live.gdn.empty(),
          "failed private capture leaves transfer destinations owned by caller");
#endif
}
struct CompletionObservation {std::atomic<unsigned> joins{0},drops{0};};
CompletionObservation* completion_observation=nullptr;
bool completion_join(const void* key) noexcept {
    static_cast<CompletionObservation*>(const_cast<void*>(key))->joins++;
    return true;
}
bool completion_probe(const void*) noexcept {return true;}
void completion_drop(uint64_t tag){
    if(tag!=7)std::_Exit(70);
    completion_observation->drops++;
#if defined(CONVERSATION_TEST_TRANSFERS)
    owner_drop_seen=true;
#endif
}
void completion_owner_fixture(){
    CompletionObservation observation;completion_observation=&observation;
    check(ConversationPending::allocation_bytes()==sizeof(ConversationPending),"portable exact intrusive completion quote");
    std::printf("completion control: object=%zu align=%zu handle=%zu quote=%zu\n",
                sizeof(ConversationPending),alignof(ConversationPending),sizeof(ConversationPendingRef),
                ConversationPending::allocation_bytes());
    auto owner=conversation_pending(&completion_join,&observation,&completion_drop,7,&completion_probe);
    check(owner->references()==1,"intrusive first reference");
    check(!owner->slot_ready(),"new control has no unearned device-ready proof");
    owner->publish_slot_ready();
    check(owner->slot_ready()&&observation.joins==0,"control-local slot readiness does not certify/read host payload");
    auto copy=owner;auto moved=std::move(copy);
    check(!copy&&owner->references()==2,"intrusive copy/move count exactly two owners");
    std::thread offthread([held=std::move(moved)]() mutable {held.reset();});offthread.join();
    check(owner->references()==1&&observation.drops==0,"off-thread alias does not release another owner");
#if defined(CONVERSATION_TEST_TRANSFERS)
    watched_control=owner.get();owner_drop_seen=false;owner_free_seen=false;owner_free_early=false;
#endif
    owner.reset();
    check(observation.joins==1&&observation.drops==1,"final owner checks host completion and releases once");
#if defined(CONVERSATION_TEST_TRANSFERS)
    check(owner_free_seen&&!owner_free_early,"actual intrusive control free follows checked completion/drop");
    watched_control=nullptr;
#endif
    std::vector<ConversationCheckpoint> roots(1);
    roots[0].stage_parts.resize(1);
    roots[0].stage_parts[0].gdn={1,2,3};
    roots[0].stage_parts[0].pending=conversation_pending(&completion_join,&observation);
    const auto copied=conversation_detail::copy_checkpoints_beating(roots);
    check(observation.joins==2&&copied[0].stage_parts[0].gdn==roots[0].stage_parts[0].gdn,
          "actual checkpoint copying recursively consumes child completion before host read");
    CompletionObservation split_observation;
    std::vector<ConversationCheckpoint> staged(1);
    staged[0].ids={1};staged[0].stage_parts.resize(1);
    staged[0].stage_parts[0].ids={1};staged[0].stage_parts[0].gdn={42};
    staged[0].stage_parts[0].dev_tag=13; // metadata carry only, NOT a runtime slot certificate
    staged[0].stage_parts[0].pending=conversation_pending(&completion_join,&split_observation);
    const auto* actual=staged[0].stage_parts[0].pending.get();
    const auto* payload=staged[0].stage_parts[0].gdn.data();
    auto split=conversation_checkpoints_split(std::move(staged),1);
    check(split_observation.joins==0&&split.parts[0][0].pending.get()==actual&&split.parts[0][0].dev_tag==13,
          "actual split transfers the same pending owner/tag with payload, without premature host wait");
    std::vector<ConversationCheckpoint> merged;
    check(conversation_checkpoints_merge(std::move(split),merged)&&split_observation.joins==0&&
          merged[0].stage_parts[0].pending.get()==actual&&merged[0].stage_parts[0].gdn.data()==payload&&
          merged[0].stage_parts[0].gdn.size()==1,
          "actual merge preserves payload/control identity without copying unfinished host data");
    merged.clear();
    check(split_observation.joins==1,"actual merged payload destruction checks the transferred completion");
}
#if defined(CONVERSATION_TEST_TRANSFERS)
struct BigCheckpointSource {
    ModelGeometry g;
    SessionState ss;
    std::vector<uint8_t> bytes;
    explicit BigCheckpointSource(uint8_t value){
        g.n_layers=1;g.ssm_state_size=512;g.ssm_v_heads=1;g.ssm_conv_channels=8;
        ConversationStateSizes z;std::string error;
        check(conversation_state_sizes(g,z,error),"recorded large checkpoint shape");
        bytes.assign(z.gdn,value);ss.max_cells=96;ss.layer_hi=1;ss.gdn_alloc=1;
        ss.gdn_state=reinterpret_cast<float*>(bytes.data());
    }
};
bool observed_park_join(const void* key) noexcept {
    {std::lock_guard<std::mutex> lock(gate_mu);join_seen=true;}gate_cv.notify_all();
    return conversation_park_join_checked(key);
}
std::atomic<unsigned> slot_drops{0};
void observed_slot_drop(uint64_t tag){slot_drops++;conversation_park_slot_release(tag);}
void async_checkpoint_fixture(){
    BigCheckpointSource first(0x51),second(0x77);
    for(int wave=0;wave<4;++wave){
        ConversationCheckpoint a,b;a.ids=b.ids={1,2};
        std::string ea,eb;bool okay_a=false,okay_b=false;
        std::barrier start(3);
        std::thread t0([&]{recorded_device=0;start.arrive_and_wait();okay_a=conversation_checkpoint_save(a,first.ss,first.g,ea);});
        std::thread t1([&]{recorded_device=1;start.arrive_and_wait();okay_b=conversation_checkpoint_save(b,second.ss,second.g,eb);});
        start.arrive_and_wait();t0.join();t1.join();
        check(okay_a&&okay_b,"actual concurrent stage-thread saves complete both publications");
        a.wait_pending();b.wait_pending();
        check(a.gdn==first.bytes&&b.gdn==second.bytes,"concurrent checked saves preserve each stage's distinct bytes");
    }
    recorded_device=0;
}
void slot_host_readiness_fixture(){
    ModelGeometry g;g.n_layers=4;g.ssm_state_size=2;g.ssm_v_heads=2;g.ssm_conv_channels=8;
    g.n_head_kv=1;g.head_dim=64;g.idx_key_dim=8;
    Pools pools(g,1);ConversationStateSizes z;std::string error;
    check(conversation_state_sizes(g,z,error),"recorded slot snapshot shape");
    std::vector<uint8_t> gdn(z.gdn,0x51);SessionState ss;
    ss.max_cells=96;ss.layer_hi=4;ss.gdn_alloc=g.n_gdn_layers();ss.qsa_alloc=1;
    ss.gdn_state=reinterpret_cast<float*>(gdn.data());ss.qsa_states=&pools.st;
    std::vector<int32_t> ids{1,2,3,4,5,6,7,8,9};std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;ConversationView view{ids,images,checkpoints,false};
    ConversationKv retained;
    {
        SavedConversation image;
        {std::lock_guard<std::mutex> lock(gate_mu);gate_host=true;gate_seen=join_seen=restore_done=gate_timeout=false;}
        check(conversation_snapshot_save(image,view,ss,g,nullptr,error),"actual async slot snapshot publishes after device drain");
        check(bool(image.pending)&&image.kv.size()==1&&image.kv[0].pending==image.pending,
              "image and retained layer share one genuine portable completion/slot owner");
        image.pending->join=&observed_park_join;image.pending->release=&observed_slot_drop;
        slot_drops=0;std::atomic<bool> validated{false};
        std::thread host_reader([&]{std::string why;validated=conversation_snapshot_validate(image,ss,g,nullptr,why);});
        std::thread safety([&]{
            std::unique_lock<std::mutex> lock(gate_mu);
            if(!gate_cv.wait_for(lock,std::chrono::seconds(2),[]{return restore_done;})){
                gate_timeout=true;gate_host=false;gate_cv.notify_all();
            }
        });
        {std::unique_lock<std::mutex> lock(gate_mu);
         gate_cv.wait_for(lock,std::chrono::seconds(2),[]{return (join_seen&&gate_seen)||gate_timeout;});}
        std::fill(gdn.begin(),gdn.end(),0x33);
        std::fill(pools.data[0].begin(),pools.data[0].end(),0x33);
        const auto restored=conversation_snapshot_restore(image,ss,g,nullptr,error);
        bool before_host=false;
        {std::lock_guard<std::mutex> lock(gate_mu);before_host=gate_host&&!gate_timeout&&join_seen&&gate_seen&&!validated;
         restore_done=true;gate_host=false;}gate_cv.notify_all();
        safety.join();host_reader.join();
        check(before_host&&restored==ConversationRestore::restored,"exact all-slot restore returns while actual host D2H/validator are gated");
        check(std::all_of(gdn.begin(),gdn.end(),[](uint8_t x){return x==0x51;})&&
              std::all_of(pools.data[0].begin(),pools.data[0].begin()+image.kv[0].k.size(),[](uint8_t x){return x==0xa5;}),
              "slot restore reads actual drained device bytes, not unfinished host buffers");
        check(validated,"public host validator finishes only after checked host completion");
        retained=image.kv[0];
    }
    check(slot_drops==0&&conversation_kv_slot_ready(retained),"retained layer keeps real slot after complete image destruction");
    retained.pending.reset();
    check(slot_drops==1,"last retained control releases the actual slot exactly once");
    (void)conversation_park_vram_release_all();
}
// Each new route runs in a fresh child before any singleton copy worker is
// created in its parent. Readiness comes only from actual save/slot methods.
struct RestoreRouteSource {
    ModelGeometry g;
    Pools pools,draft;
    SessionState ss;
    std::vector<uint8_t> gdn;
    std::vector<int32_t> ids{1,2,3,4,5,6,7,8,9};
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    static ModelGeometry geometry(bool large){
        ModelGeometry result;result.n_layers=4;result.ssm_state_size=large?512:2;
        result.ssm_v_heads=1;result.ssm_conv_channels=8;
        result.n_head_kv=1;result.head_dim=64;result.idx_key_dim=8;return result;
    }
    explicit RestoreRouteSource(bool large):g(geometry(large)),pools(g,1),draft(g,1){
        ConversationStateSizes z;std::string error;
        check(conversation_state_sizes(g,z,error),"public restore route actual shape");
        gdn.assign(z.gdn,0x51);ss.max_cells=96;ss.layer_hi=4;
        ss.gdn_alloc=g.n_gdn_layers();ss.qsa_alloc=1;
        ss.gdn_state=reinterpret_cast<float*>(gdn.data());ss.qsa_states=&pools.st;
    }
    void save(SavedConversation& image){
        std::string error;ConversationView view{ids,images,checkpoints,false};
        check(conversation_snapshot_save(image,view,ss,g,draft.st,error),"public route genuine main plus draft capture");
    }
    void dirty(){
        std::fill(gdn.begin(),gdn.end(),0x33);
        std::fill(pools.data[0].begin(),pools.data[0].end(),0x33);
        std::fill(draft.data[0].begin(),draft.data[0].end(),0x33);
    }
};
void begin_host_gate(){
    std::lock_guard<std::mutex> lock(gate_mu);
    gate_host=true;gate_seen=join_seen=restore_done=gate_timeout=false;
}
bool wait_host_gate(){
    std::unique_lock<std::mutex> lock(gate_mu);
    return gate_cv.wait_for(lock,std::chrono::seconds(2),[]{return gate_seen;});
}
void end_host_gate(){
    {std::lock_guard<std::mutex> lock(gate_mu);gate_host=false;}gate_cv.notify_all();
}
// 0: public reference whole; 1/2: reference/pointer prefix; 3: pointer
// whole/history; 4: pointer whole/later image; 5: distinct live_pending.
void public_restore_route_fixture(int route){
    RestoreRouteSource source(true);SavedConversation image,later;std::string error;
    if(route==4){
        recorded_device=1;source.save(later);later.wait_pending();
        // Its host bytes are complete. Drop every later K/V slot keeper so
        // the root can acquire the sole 64 MiB park slot; retain live state.
        later.pending.reset();
        for(auto& layer:later.kv)layer.pending.reset();
        recorded_device=0;
    }
    source.save(image);image.wait_pending();
    check(conversation_checkpoint_slot_ready(image.live)&&image.kv.size()==2&&
          conversation_kv_slot_ready(image.kv[0])&&conversation_kv_slot_ready(image.kv[1]),
          "public route begins with genuine live/main/draft exact slot certificates");
    ConversationCheckpoint historical;
    ConversationCheckpoint* unrelated=&historical;
    if(route==4){recorded_device=1;unrelated=&later.live;}
    else unrelated->ids={1,2,3,4,5};
    begin_host_gate();
    check(conversation_checkpoint_save(*unrelated,source.ss,source.g,error)&&bool(unrelated->pending),
          "unconsumed checkpoint has actual asynchronous host work");
    check(wait_host_gate(),"unconsumed actual checkpoint D2H reaches the existing event gate");
    if(route==4){
        later.live_pending=later.live.pending;
        image.stage_images.push_back(std::move(later));recorded_device=0;
    }else if(route==5)image.live_pending=historical.pending;
    else image.checkpoints.push_back(std::move(historical));
    check(conversation_checkpoint_slot_ready(image.live)&&conversation_kv_slot_ready(image.kv[0])&&
          conversation_kv_slot_ready(image.kv[1]),"unrelated publication does not replace consumed exact slots");
    source.dirty();
    // A baseline prejoin must not hang the test: release its real transfer on
    // timeout, then assert that the operation missed the required gated interval.
    std::thread safety([]{
        std::unique_lock<std::mutex> lock(gate_mu);
        if(!gate_cv.wait_for(lock,std::chrono::seconds(2),[]{return restore_done;})){
            gate_timeout=true;gate_host=false;gate_cv.notify_all();
        }
    });
    // The only worker in flight is stopped inside the existing event gate:
    // no async memcpy can race these before-write counter observations.
    const auto geometry=image.geometry;image.geometry[0]++;
    const int copies=copy_calls.load(),syncs=sync_calls.load();
    check(conversation_snapshot_restore(image,source.ss,source.g,&source.draft.st,error)==ConversationRestore::invalid&&
          conversation_snapshot_restore_prefix(image,5,source.ss,source.g,&source.draft.st,error)==ConversationRestore::invalid&&
          conversation_snapshot_restore_prefix(image,5,source.ss,source.g,source.draft.st,error)==ConversationRestore::invalid&&
          copy_calls==copies&&sync_calls==syncs,
          "public whole/both prefix reject incompatible metadata before native calls or payload writes");
    image.geometry=geometry;
    if(route==4){
        check(conversation_snapshot_restore(image,source.ss,source.g,source.draft.st,error)==ConversationRestore::invalid&&
              copy_calls==copies&&sync_calls==syncs,
              "public reference whole rejects layer-split images before host waits or native writes");
    }
    ConversationRestore restored=ConversationRestore::invalid;
    if(route==0)restored=conversation_snapshot_restore(image,source.ss,source.g,source.draft.st,error);
    else if(route==1)restored=conversation_snapshot_restore_prefix(image,5,source.ss,source.g,source.draft.st,error);
    else if(route==2)restored=conversation_snapshot_restore_prefix(image,5,source.ss,source.g,&source.draft.st,error);
    else restored=conversation_snapshot_restore(image,source.ss,source.g,&source.draft.st,error);
    bool before_host=false;
    {std::lock_guard<std::mutex> lock(gate_mu);before_host=gate_host&&gate_seen&&!gate_timeout;
     restore_done=true;gate_host=false;}gate_cv.notify_all();safety.join();
    check(before_host&&restored==ConversationRestore::restored,
          "public slot-aware restore returns without waiting for unconsumed host work");
    std::array<uint64_t,5> main_sizes{},draft_sizes{};
    const int64_t upto=(route==1||route==2)?5:9;
    check(conversation_kv_part_sizes(source.pools.st,source.g,upto,true,main_sizes,error)&&
          conversation_kv_part_sizes(source.draft.st,source.g,upto,false,draft_sizes,error),
          "actual layout supplies restored main and draft byte extents");
    check(std::all_of(source.pools.data[0].begin(),source.pools.data[0].begin()+main_sizes[0],[](uint8_t x){return x==0xa5;})&&
          std::all_of(source.draft.data[0].begin(),source.draft.data[0].begin()+draft_sizes[0],[](uint8_t x){return x==0xa5;}),
          "public routes copy actual captured main and draft slot contents");
    check(std::all_of(source.gdn.begin(),source.gdn.end(),[&](uint8_t x){return x==((route==1||route==2)?0x33:0x51);}),
          "whole restores live endpoint while prefix leaves endpoint for checkpoint rewind");
    image.wait_pending();historical.wait_pending();later.wait_pending();
}
// Revoke a real K/V certificate conservatively (never manufacture readiness).
// The consumed layer's real pending control must still gate its host fallback.
void public_prefix_fallback_fixture(int mode){
    const bool failed=mode==1,slot_route=mode>=2;
    RestoreRouteSource source(false);SavedConversation image;std::string error;
    if(failed){fail_event=true;}
    else begin_host_gate();
    source.save(image);
    check(image.kv.size()==2&&bool(image.pending)&&image.kv[0].pending==image.pending,
          "prefix fallback retains actual captured layer completion");
    if(!failed)check(wait_host_gate(),"consumed K/V D2H reaches the actual event gate");
    if(slot_route){
        check(conversation_kv_slot_ready(image.kv[0])&&conversation_kv_slot_ready(image.kv[1]),
              "both public prefix routes have actual drained main/draft slots while their host copy is gated");
    }else image.kv[0].dev_tag=0;
    // The control retains its original real tag and allocation; only the image's
    // optional certificate is withdrawn, forcing the existing checked fallback.
    image.pending->join=&observed_park_join;
    source.dirty();
    if(failed){
        (void)conversation_snapshot_restore_prefix(image,5,source.ss,source.g,&source.draft.st,error);
        std::_Exit(42);
    }
    // A baseline prejoin must not hang the test: release its real transfer on
    // timeout, then assert that the operation missed the required gated interval.
    std::thread safety([]{
        std::unique_lock<std::mutex> lock(gate_mu);
        if(!gate_cv.wait_for(lock,std::chrono::seconds(2),[]{return restore_done;})){
            gate_timeout=true;gate_host=false;gate_cv.notify_all();
        }
    });
    const auto geometry=image.geometry;image.geometry[0]++;
    const int copies=copy_calls.load(),syncs=sync_calls.load();
    check(conversation_snapshot_restore_prefix(image,5,source.ss,source.g,&source.draft.st,error)==ConversationRestore::invalid&&
          conversation_snapshot_restore(image,source.ss,source.g,source.draft.st,error)==ConversationRestore::invalid&&
          copy_calls==copies&&sync_calls==syncs&&!join_seen,
          "invalid consumed-fallback metadata is rejected without joining or writing");
    image.geometry=geometry;
    ConversationRestore result=ConversationRestore::invalid;
    if(slot_route){
        if(mode==2)result=conversation_snapshot_restore_prefix(image,5,source.ss,source.g,&source.draft.st,error);
        else result=conversation_snapshot_restore_prefix(image,5,source.ss,source.g,source.draft.st,error);
        bool before_host=false;
        {std::lock_guard<std::mutex> lock(gate_mu);before_host=gate_host&&gate_seen&&!gate_timeout&&!join_seen;
         restore_done=true;gate_host=false;}gate_cv.notify_all();safety.join();
        check(before_host&&result==ConversationRestore::restored,
              "public pointer/reference prefix restores from actual slots before consumed host D2H completes");
        std::array<uint64_t,5> sizes{};
        check(conversation_kv_part_sizes(source.pools.st,source.g,5,true,sizes,error)&&
              std::all_of(source.pools.data[0].begin(),source.pools.data[0].begin()+sizes[0],[](uint8_t x){return x==0xa5;}),
              "gated public prefix copies actual main slot bytes");
        image.wait_pending();return;
    }
    std::thread restore([&]{
        result=conversation_snapshot_restore_prefix(image,5,source.ss,source.g,&source.draft.st,error);
        {std::lock_guard<std::mutex> lock(gate_mu);restore_done=true;}gate_cv.notify_all();
    });
    bool waited=false;
    {std::unique_lock<std::mutex> lock(gate_mu);
     const bool signaled=gate_cv.wait_for(lock,std::chrono::seconds(2),[]{return join_seen||restore_done;});
     waited=signaled&&join_seen&&!restore_done&&gate_host&&gate_seen;}
    end_host_gate();restore.join();safety.join();
    check(waited&&result==ConversationRestore::restored,"actual consumed host fallback waits then restores after checked completion");
    std::array<uint64_t,5> sizes{};
    check(conversation_kv_part_sizes(source.pools.st,source.g,5,true,sizes,error)&&
          std::all_of(source.pools.data[0].begin(),source.pools.data[0].begin()+sizes[0],[](uint8_t x){return x==0xa5;}),
          "consumed prefix fallback reads completed actual host bytes");
    image.wait_pending();
}
void public_restore_routes_child_fixture(){
    for(int route=0;route<10;++route){
        int receipt[2];check(pipe(receipt)==0,"public route receipt pipe");
        const auto child=fork();check(child>=0,"fresh public restore route child");
        if(child==0){
            close(receipt[0]);if(route==7)fault_marker=receipt[1];
            if(route<6)public_restore_route_fixture(route);
            else public_prefix_fallback_fixture(route-6);
            std::_Exit(0);
        }
        close(receipt[1]);int status=0;
        check(waitpid(child,&status,0)==child,"public restore route child reaped");
        char observed[4]{};const auto n=read(receipt[0],observed,sizeof(observed));close(receipt[0]);
        if(route==7)check(n==1&&observed[0]=='E'&&WIFEXITED(status)&&WEXITSTATUS(status)==1,
                         "consumed prefix event failure is observed and fails closed at real checked completion");
        else check(n==0&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"genuine public restore route assertions passed");
    }
}
void failure_child_fixture(char mode){
    int receipt[2];check(pipe(receipt)==0,"failure receipt pipe");
    const auto child=fork();check(child>=0,"failure child created");
    if(child==0){
        close(receipt[0]);fault_marker=receipt[1];
        BigCheckpointSource source(0x66);ConversationCheckpoint c;c.ids={1,2};std::string error;
        fault_checkpoint=&c;
        if(mode=='C')fail_copy=copy_calls+1;
        if(mode=='L')fail_launch=true;
        if(mode=='E')fail_event=true;
        if(!conversation_checkpoint_save(c,source.ss,source.g,error))std::_Exit(43);
        c.wait_pending();std::_Exit(42); // each injected failure must fail-stop before this point
    }
    close(receipt[1]);int status=0;check(waitpid(child,&status,0)==child,"failure child reaped");
    char observed[4]{};const auto n=read(receipt[0],observed,sizeof(observed));close(receipt[0]);
    check(n==1&&observed[0]==mode&&WIFEXITED(status)&&WEXITSTATUS(status)==1,
          "actual copy/async-launch/event failure reached its named boundary and retained ownership by fail-stop");
}
void async_source_child_fixture(){
    const auto child=fork();check(child>=0,"recording source child created before test threads");
    if(child==0){
        setenv("STRATA_CKPT_ASYNC","1",1);setenv("STRATA_CKPT_THREADS","1",1);
        setenv("STRATA_CKPT_VRAM_SLOTS","2",1);setenv("STRATA_PARK_ASYNC","1",1);setenv("STRATA_PARK_VRAM_MIB","64",1);
        public_restore_routes_child_fixture();
        async_checkpoint_fixture();slot_host_readiness_fixture();
        failure_child_fixture('C');failure_child_fixture('L');failure_child_fixture('E');
        std::puts("checked state: actual concurrent saves/slot-host distinction/three named failures passed (recording CUDA)");
        std::fflush(stdout);std::_Exit(0);
    }
    int status=0;check(waitpid(child,&status,0)==child,"recording source child reaped");
    check(WIFEXITED(status)&&WEXITSTATUS(status)==0,"actual-source recording groups passed without contaminating original default fixtures");
}
#endif

}

int main() {
#if defined(CONVERSATION_TEST_TRANSFERS)
    async_source_child_fixture();
#endif
    completion_owner_fixture();
    for (int format : {0,1,2}) for (int experts : {256,512})
        for (bool zero_qsa : {false,true}) for (bool ple : {false,true}) fixture(format,experts,zero_qsa,ple);
    std::printf("conversation_validation_test: %d host-only checks passed\n", checks);
}
