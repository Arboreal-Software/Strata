#!/usr/bin/env python3
"""Execute actual prefix recorder/callback/RAW bridge and Prefill STOP/join seams.

Recording CUDA/cache/Prefill numeric stubs establish control and lifetime rules,
not native tensor equivalence, allocation fit, physical drains, or GPU STOP proof.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


def take(source, marker, semi=False):
    begin = source.index(marker)
    opening = source.index('{', begin)
    depth = 0
    for at in range(opening, len(source)):
        if source[at] == '{':
            depth += 1
        elif source[at] == '}':
            depth -= 1
            if not depth:
                return source[begin:at+1] + (';' if semi else '') + '\n'
    raise RuntimeError('unclosed actual source: ' + marker)


PRELUDE = r'''
#include "serving_prefix_replay.hpp"
#include "strata/core/window_device.hpp"
#include "strata/core/runtime_policy_observation.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <cmath>
#include <future>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
namespace seam {
int device=0,sync_device=-1,upload_device=-1,throw_stage=-1,fail_stage=-1,device_error=0;
int primary_chunks=0,lends=0,uploads=0,refills=0,legacy_callbacks=0,destruct_fd=-1;
}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* p){*p=seam::device;return seam::device_error?cudaErrorInvalidDevice:cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaSetDevice(int n){seam::device=n;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize(){return seam::device==seam::sync_device?cudaErrorUnknown:cudaSuccess;}
void require(bool value,const char* why){if(!value){std::fprintf(stderr,"FAIL %s\n",why);std::exit(9);}}
namespace strata::core {
[[noreturn]] void retirement_failed(const char* why){std::fprintf(stderr,"actual prefix fatal: %s\n",why);std::fflush(stderr);std::_Exit(70);}
struct OnDevice {int previous;explicit OnDevice(int n):previous(seam::device){if(n>=0)seam::device=n;}~OnDevice(){seam::device=previous;}};
constexpr int32_t kNotResident=-1;
struct BlobPart {};
struct SessionState {int32_t ple_prev[2]{-1,-1};int32_t ple_token=-1;
    void* qsa_states=nullptr;void* qsa_state_arena=nullptr;void* qsa_buf_arena=nullptr;void* gdn_state=nullptr;
    void* ple_hist=nullptr;void* block_arena=nullptr;void* moe_arena=nullptr;void* R=nullptr;
    int64_t max_cells=128,k=2,layer_lo=0,layer_hi=0,qsa_ord0=0,qsa_alloc=1,gdn_ord0=0,gdn_alloc=1;};
struct ExpertCache {
    int device=0;int64_t size=256;std::array<size_t,257> offsets{};std::vector<int> copied;
    int64_t slots()const{return size;}const size_t* slot_offsets()const{return offsets.data();}
    bool fill_slot_queued_parts(int32_t slot,const BlobPart*,int,std::string&){require(seam::device==device,"actual refill stage device");copied.push_back(slot);return true;}
    bool fill_slot_queued(int32_t slot,const uint8_t*,std::string&,int64_t){return fill_slot_queued_parts(slot,nullptr,0,dummy);}
    bool fill_slot_blocking(int32_t slot,const uint8_t* p,std::string& e,int64_t n){return fill_slot_queued(slot,p,e,n);}
    bool sync_queued(std::string&){require(seam::device==device,"actual queued wait device");++seam::refills;return true;}
    std::string dummy;
};
struct ExpertSource {std::array<uint8_t,8> blob{};
    int blob_parts(int64_t,int64_t,BlobPart*){return 0;}
    const uint8_t* blob_stable(int64_t,int64_t){return blob.data();}
    void note_async_read(const uint8_t*,void*){}
};
struct Verifier {enum class SnapshotDrain {drained,failed};bool good=true;
    SnapshotDrain poll_snapshot_drain(std::string&){return good?SnapshotDrain::drained:SnapshotDrain::failed;}};
}
namespace strata::kernels::cpu {struct Layout {uint64_t blob_bytes(int64_t)const{return 8;}};inline Layout expert_layout(){return {};}}
namespace strata::prefill {
namespace mmq {inline int direct_mode(){return 0;}}
struct Prefill {
    struct Impl {core::SessionState* ss=nullptr;core::ExpertSource* src=nullptr;core::ExpertCache* cache=nullptr;
        const int32_t* host_res=nullptr;int device=0;bool help_pp=false,pp=false,borrow_spill=false;};
    ACTUAL_ACTIVITY_TYPE
    std::unique_ptr<Impl> impl_=std::make_unique<Impl>();
    Prefill* next_=nullptr;int64_t stage_lb_=0,stage_le_=2,force_chunk_=0,stopped_at_=-1;
    std::vector<int> tap_layers_;const float* hand_in_=nullptr;const float* const* embd_rows=nullptr;
    std::future<bool> next_run_;std::string next_err_;
    std::function<bool(const float*,int64_t,int64_t,std::string&)> on_chunk,on_taps;
    std::function<bool(int64_t,std::string&)> on_stage_chunk;
    std::function<bool()> should_stop,stop_at_chunk;
    std::function<void(int64_t,const int32_t*,int64_t)> on_routes;
    struct Stats {int experts_direct=0,experts_resident=0,experts_streamed=0;} stats_;
    core::PrefillCpuShareObservation observed;
    Prefill(){observed.explicit_resolved=true;observed.mode=core::NumericPolicyMode::disabled;}
    core::PrefillCpuShareObservation cpu_share_policy()const{return observed;}
    ACTUAL_ACTIVITY
    static double pipe_setting(){return 0;}
    int64_t chunk()const{return 16;}int64_t stopped_at()const{return stopped_at_;}
    void force_chunk(int64_t n,bool,bool){force_chunk_=n;}
    bool run_impl(const int64_t* tokens,int64_t n,int64_t pos0,std::string& err){
        auto& ss=*impl_->ss;int32_t prev[2]{ss.ple_prev[0],ss.ple_prev[1]};const int64_t C=16;
        if(hand_in_==nullptr){stopped_at_=-1;force_chunk_=0;}
        for(int64_t c0=0;c0<n;c0+=C){
            ACTUAL_SHOULD_STOP
            ACTUAL_GRACEFUL_STOP
            const int64_t T=std::min(C,n-c0),p0=pos0+c0;
            if(seam::throw_stage==impl_->device)throw std::runtime_error("recording numeric exception");
            ACTUAL_PREV_UPDATE
            if(on_stage_chunk&&!on_stage_chunk(p0+T,err))return false;
            if(impl_->device==0)++seam::primary_chunks;
            if(next_){
                if(next_run_.valid()&&!next_run_.get()){err=next_err_;return false;}
                next_->hand_in_=reinterpret_cast<const float*>(uintptr_t(1));
                next_run_=std::async(std::launch::async,[this,tokens,c0,T,p0]{return next_->run_impl(tokens+c0,T,p0,next_err_);});
            }else if(on_chunk&&!on_chunk(nullptr,T,p0,err))return false;
            if(seam::fail_stage==impl_->device){err="recording downstream error";return false;}
        }
        ACTUAL_PLE_EPILOGUE
        return true;
    }
    ACTUAL_DRAIN
    ACTUAL_RUN
};
}
namespace strata::program {
class SuffixRawTransaction {public:
    core::StateCut cut_;bool running=false,begin_permit=true,complete_permit=true;int begins=0,completes=0;
    bool begin_driver_work(){++begins;if(!begin_permit)return false;running=true;return true;}
    bool complete_driver_work(bool actual,const core::StateCut& cut,std::span<const core::StateCut> stages){
        ++completes;require(running&&actual&&stages.size()==2,"actual bridge completion operands");
        for(const auto& stage:stages)require(stage==cut,"common bridge cut operands");
        if(!complete_permit)return false;
        cut_=cut;running=false;return true;}
};
struct Bindings {struct Native {core::SessionState* session=nullptr;core::Verifier* verifier=nullptr;int device=0;int64_t begin=0,end=0,covered=0;bool ple_commit=false;};
    std::array<Native,8> native{};std::array<core::Verifier*,8> chain{};size_t native_count=2,chain_count=2;};
struct Context {struct {int max_context=128;} options;bool binding=true;int primary_device=0;Bindings bindings;
    struct Owner {struct {bool good=true;bool healthy()const{return good;}} sessions;
        struct Drive {struct {bool failed=false;} d;};std::unique_ptr<Drive> drive=std::make_unique<Drive>();};
    std::shared_ptr<Owner> owner=std::make_shared<Owner>();};
namespace suffix_driver_detail {using Bindings=::strata::program::Bindings;
    template<class C> bool bind(const C& c,Bindings& out,const char*&){out=c.bindings;return c.binding;}
    bool present(const core::StateDigest& h){return std::any_of(h.begin(),h.end(),[](auto x){return x!=0;});}}
class SuffixExperimentDriver {public:
    struct Step {core::StateCut cut;int32_t next=-1;};
    [[noreturn]] static void failstop(const char* why){core::retirement_failed(why);}
    ACTUAL_BRIDGE
};
}
namespace core=strata::core;
using namespace strata::program;
using Clock=std::chrono::steady_clock;
struct Sentinel {~Sentinel(){if(seam::destruct_fd>=0&&::write(seam::destruct_fd,"D",1)!=1)std::_Exit(92);}};
template<class F> void fails(F&& body,const char* origin){
    int p[2],e[2];require(pipe(p)==0&&pipe(e)==0,"fault pipes");auto pid=fork();require(pid>=0,"fault fork");
    if(!pid){close(p[0]);close(e[0]);require(dup2(e[1],2)>=0,"fault stderr");close(e[1]);
        seam::destruct_fd=p[1];Sentinel owner;body();std::_Exit(91);}
    close(p[1]);close(e[1]);int status=0;require(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==70,"actual prefix noreturn70");
    char bytes[4096];require(read(p[0],bytes,sizeof bytes)==0,"actual fatal retains owner sentinel without unwind");close(p[0]);
    auto n=read(e[0],bytes,sizeof bytes);close(e[0]);require(n>0&&std::string(bytes,size_t(n)).find(origin)!=std::string::npos,"actual fatal source origin");
}
int main(int argc,char**){
    PrefixReplayPlan refused_plan;
    require(!refused_plan.reject("original policy refusal"),"initial refusal");
    require(!refused_plan.seal({}),"refused transcript cannot seal");
    require(std::string(refused_plan.reason)=="original policy refusal","seal preserves original refusal cause");
    uint64_t quote=0;require(PrefixReplayPlan::quote(64,8,quote)&&quote==0,"whollyT1 no extra quote");
    require(PrefixReplayPlan::quote(96,8,quote)&&quote>0,"actual typed plan envelope");
    require(!PrefixReplayPlan::quote(UINT64_MAX,8,quote),"actual quote overflow refusal");
    setenv("STRATA_PREFILL_CPU_SHARE","0",1);setenv("STRATA_PREFILL_HELP","0",1);
    unsetenv("STRATA_PREFILL_HELPER_SOURCE");unsetenv("STRATA_CKPT_REREAD");unsetenv("STRATA_PIPELINE_SWITCH");
    struct Options {int prefill_chunk=16,short_read=64,batch=0,pipeline_windows=0,peer_device=-1;bool prefill_auto=false,native_serial_serving=true,native_owned_batch_serving=false;} o;
    int64_t read_from=0,resume=0,reread_to=0,req_short_read=64;struct {int64_t chunk=0;} resume_plan;
    std::vector<int> req_imgs;bool pipe=false,remote_caches=false,remote_opt=false,fs_swap=false;
    struct {bool on=false;} ram_adapt2;struct {bool valid()const{return false;}} peer;
    bool use_mtp=false,batch_mtp=false;
    int pl_pw=0,primary_device=0,n_stages=2;struct {int64_t n_expert=2;} g;
    core::ExpertSource source;auto* srcp=&source;std::vector<int32_t> host_res{0,1,2,3,4,5,6,7};
    core::ExpertCache primary_cache,secondary_cache;secondary_cache.device=1;
    core::SessionState primary_session,secondary_session;
    strata::prefill::Prefill sp,secondary;sp.next_=&secondary;secondary.stage_lb_=2;secondary.stage_le_=4;
    sp.impl_->ss=&primary_session;sp.impl_->src=srcp;sp.impl_->cache=&primary_cache;sp.impl_->host_res=host_res.data();
    secondary.impl_->ss=&secondary_session;secondary.impl_->src=srcp;secondary.impl_->cache=&secondary_cache;
    secondary.impl_->host_res=host_res.data();secondary.impl_->device=1;
    struct Part {core::ExpertCache* cache;const core::SessionState* ses;strata::prefill::Prefill* sp;int dev;int64_t lb,le;
        int32_t first_now=0;int64_t lent_chunk=0;std::vector<std::pair<int32_t,int32_t>> lent;};
    using PfPart=Part; // preserve the extracted production lambdas' actual type name
    std::vector<Part> pf_parts{{&primary_cache,&primary_session,&sp,-1,0,2,0,0,{}},{&secondary_cache,&secondary_session,&secondary,1,2,4,0,0,{}}};
    struct Runtime {uint64_t remote_policy_incarnation=77;};
    auto owned_runtime=std::make_shared<Runtime>();
    struct Active {core::SnapshotDescriptor published;};std::unique_ptr<Active> serving_suffix_active;
    bool sfx_prefix_origin_replay=false,ajob=false;int a_swapped=0,adapt_nswaps=0;
    struct {std::shared_ptr<const void> credit=std::make_shared<int>(2);PrefixReplayPlan prefix;} sfx_qualification;
    sfx_qualification.prefix.boundary=96;sfx_qualification.prefix.request=2;
    auto& sfx_prefix_plan=sfx_qualification.prefix;
    constexpr bool sfx_prefix_mode=true;bool sfx_prefix_replaying=false,trace=false;
    auto tr=[](auto...){return;};bool refill_parts=false,refill_serial=false;auto refill_blocking=[](){return false;};
    int primary_table=0,secondary_table=1;auto* d_res=&primary_table;
    struct Stage {int dev=1;int* d_res=nullptr;};std::vector<std::unique_ptr<Stage>> stages;
    stages.push_back(std::make_unique<Stage>());stages[0]->d_res=&secondary_table;
    auto res_put=[](const int*){++seam::uploads;return seam::device==seam::upload_device?cudaErrorUnknown:cudaSuccess;};
    ACTUAL_UPLOAD
    ACTUAL_REFILL_ISSUE
    ACTUAL_REFILL_WAIT
    ACTUAL_REFILL_ONE
    ACTUAL_REFILL
    auto lend=[&](int64_t,std::string&,int64_t){++seam::lends;
        for(auto& p:pf_parts){if(p.lent.empty()){int32_t i=int32_t(p.lb*g.n_expert);p.lent.emplace_back(i,host_res[size_t(i)]);host_res[size_t(i)]=core::kNotResident;}p.lent_chunk=16;}return true;};
    if(argc>1)resume=16;
    ACTUAL_RESET_WITNESS
    ACTUAL_POLICY
    ACTUAL_PREPARE
    ACTUAL_LOANS
    ACTUAL_BASE
    ACTUAL_CALLBACK
    ACTUAL_CHECKED_REFILL
    std::string error;
    if(argc>1){require(sfx_prefix_policy()!=nullptr,"actual checkpoint ancestry refuses");return 0;}
    require(!sfx_prefix_policy(),"actual supported fixed numeric profile");
    o.native_serial_serving=false;o.batch=2;
    require(sfx_prefix_policy()!=nullptr,"unresolved batch writer refuses prefix replay");
    o.native_owned_batch_serving=true;
    require(!sfx_prefix_policy(),"resolved owned two-slot main-prefix profile");
    o.native_owned_batch_serving=false;o.native_serial_serving=true;o.batch=0;
    for(int mode=0;mode<11;++mode){
        o.prefill_chunk=mode==0?32:16;o.short_read=mode==1?0:64;
        if(mode==2)setenv("STRATA_PREFILL_CPU_SHARE","auto",1);
        sp.impl_->help_pp=mode==3;secondary.impl_->pp=mode==4;secondary.tap_layers_=mode==5?std::vector<int>{1}:std::vector<int>{};
        secondary.observed.mode=mode==6?core::NumericPolicyMode::adaptive:core::NumericPolicyMode::disabled;
        secondary.impl_->host_res=mode==7?nullptr:host_res.data();secondary.impl_->device=mode==8?0:1;
        secondary.on_routes=mode==9?std::function<void(int64_t,const int32_t*,int64_t)>([](int64_t,const int32_t*,int64_t){}):nullptr;
        if(mode==10)setenv("STRATA_PIPELINE_SWITCH","unused",1);
        require(sfx_prefix_policy()!=nullptr,"actual unsupported runtime profile refuses");
        setenv("STRATA_PREFILL_CPU_SHARE","0",1);unsetenv("STRATA_PIPELINE_SWITCH");
    }
    o.prefill_chunk=16;o.short_read=64;sp.impl_->help_pp=false;secondary.impl_->pp=false;secondary.tap_layers_.clear();
    secondary.observed.mode=core::NumericPolicyMode::disabled;secondary.impl_->host_res=host_res.data();secondary.impl_->device=1;secondary.on_routes=nullptr;
    // A real allocator refusal under a child address-space ceiling. No cache loan
    // occurs and the original admission owner remains held; no guessed failure.
    {auto pid=fork();require(pid>=0,"allocation refusal fork");if(!pid){
        FILE* f=std::fopen("/proc/self/statm","r");unsigned long pages=0;require(f&&std::fscanf(f,"%lu",&pages)==1,"current address space");std::fclose(f);
        rlimit limit;limit.rlim_cur=rlim_t(pages)*rlim_t(sysconf(_SC_PAGESIZE))+(4u<<20);limit.rlim_max=limit.rlim_cur;
        require(setrlimit(RLIMIT_AS,&limit)==0,"bounded actual allocation refusal");sfx_qualification.prefix.boundary=65536;
        require(!sfx_prefix_prepare()&&sfx_qualification.prefix.refused&&sfx_qualification.prefix.credit&&seam::lends==0,
            "actual preparation catches allocation refusal before original loan and retains credit");std::_Exit(0);}
        int status=0;require(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"actual allocation refusal child");}
    {PrefixReplayPlan denied;denied.boundary=96;require(!denied.prepare({},host_res)&&denied.events.empty()&&denied.base.empty(),"actual missing admission refuses before allocation");}
    PrefixReplayEvent window;window.to=16;require(sfx_qualification.prefix.record(window),"actual leading windows recorded without vector allocation");
    require(sfx_qualification.prefix.events.empty()&&!sfx_qualification.prefix.active,"whollyT1 no replay vector/callback");
    require(sfx_prefix_prepare(),"actual transcript setup under admitted owner");
    require(lend(80,error,0)&&sfx_prefix_loans(true),"actual canonical loan recording");
    auto canonical=pf_parts[1].lent;pf_parts[1].lent[0].second++;
    require(!sfx_prefix_loans(false),"actual changed canonical loan refuses");pf_parts[1].lent=canonical;sfx_qualification.prefix.refused=false;
    PrefixReplayEvent batch;batch.kind=PrefixReplayEvent::Kind::prefill;batch.from=16;batch.to=96;
    batch.chunk=16;batch.planned=true;batch.lend=true;batch.lend_tokens=80;
    require(sfx_qualification.prefix.record(batch),"actual batch call recorded");
    std::vector<int32_t> history(100);for(size_t i=0;i<history.size();++i)history[i]=int32_t(i%10);
    core::WindowKey origin_key;origin_key.engine_generation=1;origin_key.window_generation=1;
    origin_key.session_id="owned";origin_key.thread_id="conversation";origin_key.window_id="window";origin_key.revision=2;
    sfx_qualification.prefix.origin_key=origin_key;sfx_qualification.prefix.origin_runtime_incarnation=77;
    require(sfx_qualification.prefix.seal(history)&&sfx_qualification.prefix.matches(history,2,99),"actual bound sealed prefix");
    sfx_qualification.prefix.events.back().chunk=8;require(!sfx_qualification.prefix.matches(history,2,99),"actual post-seal numeric-call mutation refuses");sfx_qualification.prefix.events.back().chunk=16;
    auto changed=history;changed[4]++;require(!sfx_qualification.prefix.matches(changed,2,99),"actual changed conditioning refuses");
    require(!sfx_qualification.prefix.matches(history,3,99),"actual stale request refuses");
    seam::upload_device=1;require(!refill(error)&&error.find("original refill residency upload")!=std::string::npos,"ORIGINAL second-stage upload failure propagates");
    require(seam::uploads==2,"failed original upload not retried into fake success");seam::upload_device=-1;
    require(refill(error)&&sfx_prefix_base(sfx_qualification.prefix,error),"actual same-slot refill returns complete mapping");
    host_res[2]++;require(!sfx_prefix_base(sfx_qualification.prefix,error),"actual full base mapping drift refuses");host_res[2]--;
    secondary.impl_->help_pp=true;require(!sfx_prefix_base(sfx_qualification.prefix,error),"actual secondary helper refuses before replay");secondary.impl_->help_pp=false;
    auto sentinel=std::make_shared<Sentinel>();
    sp.on_chunk=[sentinel](const float*,int64_t,int64_t,std::string&){++seam::legacy_callbacks;return true;};
    secondary.on_stage_chunk=[sentinel](int64_t,std::string&){++seam::legacy_callbacks;return true;};
    sp.should_stop=[](){return true;};secondary.should_stop=[](){return true;}; // witnessed STOP stays latched
    struct Adapter {decltype(sfx_prefix_base)* check;decltype(sfx_prefix_replay_piece)* run;decltype(sfx_prefix_checked_refill)* refill;};
    Adapter adapter{&sfx_prefix_base,&sfx_prefix_replay_piece,&sfx_prefix_checked_refill};
    PrefixReplayCallbacks callbacks{&adapter,
        [](void* p,const PrefixReplayPlan& plan,std::string& e){return (*static_cast<Adapter*>(p)->check)(plan,e);},
        [](void* p,const PrefixReplayEvent& event,std::span<const int32_t> h,PrefixReplayAbort abort,PrefixReplayProgress& progress,std::string& e){return (*static_cast<Adapter*>(p)->run)(event,h,abort,progress,e);},
        [](void* p,std::string& e){return (*static_cast<Adapter*>(p)->refill)(e);}};
    auto check_view=[&](PrefixReplayView prefix,uint64_t ordinal=2){
        auto& admission=sfx_qualification;core::SnapshotDescriptor original;original.accepted_tokens=history;
        const uint64_t forwarded=99;
        auto failstop=[](const char* why){core::retirement_failed(why);};
        ACTUAL_PREFIX_GATE
    };
    fails([&]{check_view({});},"requires its sealed request transcript");
    check_view({&sfx_qualification.prefix,&callbacks});
    Context context;core::Verifier primary_verifier,secondary_verifier;
    context.bindings.native[0]={&primary_session,&primary_verifier,0};context.bindings.native[1]={&secondary_session,&secondary_verifier,1};
    context.bindings.chain[0]=&primary_verifier;context.bindings.chain[1]=&secondary_verifier;
    auto init=[&](SuffixRawTransaction& tx){tx.cut_={4,100,16,16,7,core::window_history_digest(history)};
        primary_session.ple_prev[0]=history[14];primary_session.ple_prev[1]=history[15];
        secondary_session.ple_prev[0]=secondary_session.ple_prev[1]=-1;seam::primary_chunks=0;};
    PrefixReplayAbort never{nullptr,[](void*){return false;}};
    SuffixRawTransaction tx;init(tx);uint64_t next_cut=5;bool stopped=false;
    PrefixReplayAbort before{nullptr,[](void*){return true;}};
    auto step=SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,before,stopped);
    require(stopped&&tx.begins==0&&tx.completes==0&&tx.cut_.forwarded==16&&next_cut==5,"actual pre-start abort cannot begin or mint a cut");stopped=false;
    step=SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,never,stopped);
    require(step.cut.forwarded==96&&step.cut.exact_frontier==96&&step.cut.output_watermark==7&&next_cut==6&&!stopped&&!tx.running,
        "actual RAW bridge common endpoint/cut/watermark");
    require(!seam::legacy_callbacks&&sp.should_stop()&&secondary.should_stop()&&!sfx_prefix_replaying,
        "actual callback suppression/restoration keeps original STOP latched");
    require(sfx_prefix_checked_refill(error)&&sfx_prefix_base(sfx_qualification.prefix,error),"actual replay checked refill/base");
    init(tx);next_cut=6;PrefixReplayAbort stop{nullptr,[](void*){return seam::primary_chunks>=1;}};
    step=SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,stop,stopped);
    require(stopped&&step.cut.forwarded==32&&tx.cut_.exact_frontier==32&&primary_session.ple_prev[0]==history[30]&&
        secondary_session.ple_prev[1]==history[31]&&!sp.next_run_.valid(),"actual graceful break/public join/PLE/common endpoint");
    require(sfx_prefix_checked_refill(error),"checked refill after verified graceful STOP");
    // Original should_stop branch is the negative control: it bypasses actual
    // PLE epilogue after one real synthetic chunk and must not certify a cut.
    primary_session.ple_prev[0]=primary_session.ple_prev[1]=-1;seam::primary_chunks=0;
    sp.should_stop=[](){return seam::primary_chunks>=1;};secondary.should_stop=nullptr;sp.stop_at_chunk=nullptr;
    require(!sp.run(sfx_qualification.prefix.tokens.data()+16,80,16,error)&&primary_session.ple_prev[1]==-1,
        "original should_stop negative demonstrates missing PLE epilogue");
    sp.should_stop=[](){return true;};secondary.should_stop=[](){return true;};
    fails([&]{init(tx);next_cut=8;seam::throw_stage=0;SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,never,stopped);},"callback exception retains owners");
    fails([&]{init(tx);next_cut=8;seam::fail_stage=1;SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,never,stopped);},"actual run or downstream join failed");
    fails([&]{init(tx);next_cut=8;seam::sync_device=1;SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,never,stopped);},"callback device completion uncertain");
    auto original_run=callbacks.run_piece;
    auto bridge_with=[&](auto mutate){
        struct Swap {PrefixReplayCallbacks original;Context* context;core::SessionState* replacement;bool layout;} swap{
            callbacks,&context,&secondary_session,false};
        mutate(swap);
        PrefixReplayCallbacks changed_callbacks=callbacks;changed_callbacks.user=&swap;
        changed_callbacks.run_piece=[](void* p,const PrefixReplayEvent& event,std::span<const int32_t> h,PrefixReplayAbort abort,PrefixReplayProgress& progress,std::string& e){
            auto& state=*static_cast<Swap*>(p);const auto& original=state.original;
            const bool ran=original.run_piece(original.user,event,h,abort,progress,e);
            if(state.layout)state.context->bindings.native[1].session->max_cells++;
            else state.context->bindings.native[1].session=state.replacement;
            return ran;};
        init(tx);next_cut=8;SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,changed_callbacks,history,100,tx.cut_.history,never,stopped);
    };
    fails([&]{core::SessionState replacement=secondary_session;bridge_with([&](auto& state){state.replacement=&replacement;});},"executed native identity/layer/allocation layout changed");
    fails([&]{bridge_with([](auto& state){state.layout=true;});},"executed native identity/layer/allocation layout changed");
    callbacks.run_piece=[](void*,const PrefixReplayEvent&,std::span<const int32_t>,PrefixReplayAbort,PrefixReplayProgress& progress,std::string&){progress.stages=2;progress.stage_end={96,95};return true;};
    fails([&]{init(tx);next_cut=8;SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,never,stopped);},"stage endpoints disagree");callbacks.run_piece=original_run;
    fails([&]{init(tx);next_cut=8;secondary_verifier.good=false;SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,never,stopped);},"verifier drain incomplete");
    fails([&]{init(tx);next_cut=8;tx.complete_permit=false;SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,never,stopped);},"actual common cut refused");
    fails([&]{init(tx);next_cut=4;SuffixExperimentDriver::replay_prefill_piece(context,tx,next_cut,batch,callbacks,history,100,tx.cut_.history,never,stopped);},"binding/cut bounds");
    // Move the sealed production transcript out of its original recording scope.
    // The old callback pointers are deliberately unusable; only data/credit survive.
    auto original_digest=sfx_prefix_plan.digest;
    std::weak_ptr<Sentinel> original_request_owner;
    auto durable=[&]{
        auto holder=std::make_shared<PrefixReplayPlan>();
        PrefixReplayPlan recorded=std::exchange(sfx_prefix_plan,PrefixReplayPlan{});
        auto& sfx_prefix_plan=recorded;
        ACTUAL_CALLBACK
        auto request_owner=std::make_shared<Sentinel>();original_request_owner=request_owner;
        struct OriginalAdapter {decltype(sfx_prefix_replay_piece) run;std::shared_ptr<Sentinel> owner;};
        OriginalAdapter original_adapter{sfx_prefix_replay_piece,request_owner};
        (void)original_adapter;
        auto wrong_key=origin_key;wrong_key.revision++;
        require(!retain_recorded_prefix_origin(holder,sfx_prefix_plan,history,2,99,wrong_key,77,0),
                "actual transfer refuses wrong content origin before consuming transcript");
        require(!retain_recorded_prefix_origin(holder,sfx_prefix_plan,history,4,99,origin_key,77,0),
                "consumer ordinal cannot replace original ordinal");
        require(retain_recorded_prefix_origin(holder,sfx_prefix_plan,history,2,99,origin_key,77,0),
                "actual park transfer moves sealed data without callback ownership");
        return holder;
    }();
    require(original_request_owner.expired(),"actual original callback adapter scope dies while retained data survives");
    std::weak_ptr<const void> origin_credit=durable->credit;
    sfx_qualification.credit.reset();callbacks.user=nullptr;callbacks.run_piece=nullptr;
    require(!origin_credit.expired()&&!sfx_prefix_plan.active&&durable->request==2&&durable->digest==original_digest,
            "origin scope releases credit and callbacks while durable owner preserves origin and digest");
    require(!durable->origin_matches(origin_key,78,0)&&!durable->origin_matches(origin_key,77,1),
            "actual runtime and residency origin substitutions refuse");
    {
        // Recreate the ACTUAL request-local adapter against A3's owned origin.
        auto& sfx_prefix_plan=*durable;
        const bool sfx_prefix_origin_replay=true,sfx_prefix_reset_from_zero=false;
        serving_suffix_active=std::make_unique<Active>();serving_suffix_active->published.key=origin_key;
        ACTUAL_POLICY
        ACTUAL_LOANS
        ACTUAL_BASE
        ACTUAL_CALLBACK
        ACTUAL_CHECKED_REFILL
        struct ReboundAdapter {decltype(sfx_prefix_base)* check;decltype(sfx_prefix_replay_piece)* run;decltype(sfx_prefix_checked_refill)* refill;};
        ReboundAdapter rebound_adapter{&sfx_prefix_base,&sfx_prefix_replay_piece,&sfx_prefix_checked_refill};
        PrefixReplayCallbacks rebound{&rebound_adapter,
            [](void* p,const PrefixReplayPlan& plan,std::string& e){return (*static_cast<ReboundAdapter*>(p)->check)(plan,e);},
            [](void* p,const PrefixReplayEvent& event,std::span<const int32_t> h,PrefixReplayAbort abort,PrefixReplayProgress& progress,std::string& e){return (*static_cast<ReboundAdapter*>(p)->run)(event,h,abort,progress,e);},
            [](void* p,std::string& e){return (*static_cast<ReboundAdapter*>(p)->refill)(e);}};
        core::SnapshotDescriptor final;final.key=origin_key;final.cut.forwarded=99;final.accepted_tokens=history;
        PrefixReplayView origin_view{durable.get(),&rebound,2};
        check_view(origin_view,4);
        require(activation_prefix_view_matches(96,origin_view,final,77,0),"actual activation gate accepts origin2 under consumer4");
        require(!activation_prefix_view_matches(96,{},final,77,0),"actual activation gate refuses silent T1 substitution");
        auto wrong_view=origin_view;wrong_view.origin_request=4;
        require(!activation_prefix_view_matches(96,wrong_view,final,77,0),"actual activation gate refuses origin relabelling");
        final.accepted_tokens[2]++;
        require(!activation_prefix_view_matches(96,origin_view,final,77,0),"actual activation gate refuses wrong origin conditioning");
        final.accepted_tokens=history;
        require(!sfx_prefix_policy(),"A3 reset/replay uses recorded from0 provenance despite continuation admission");
        require(refill(error)&&sfx_prefix_base(sfx_prefix_plan,error),"A3 current execution owners bind to retained origin");
        PrefixReplayProgress progress;
        require(sfx_prefix_replay_piece(batch,history,never,progress,error)&&progress.stage_end[0]==96&&progress.stage_end[1]==96,
                "new A3 callbacks execute actual original batch after old callback pointers are destroyed");
        require(sfx_prefix_checked_refill(error)&&sfx_prefix_base(sfx_prefix_plan,error),"new A3 adapter restores original mapping");
        serving_suffix_active->published.key.revision++;
        require(!sfx_prefix_base(sfx_prefix_plan,error),"actual A3 callback refuses wrong content origin");
        serving_suffix_active->published.key=origin_key;
        sfx_prefix_plan.origin_source[0]='x';
        require(!sfx_prefix_plan.matches(history,2,99),"source provenance mutation invalidates sealed origin");
        sfx_prefix_plan.origin_source[0]=0;
        sfx_prefix_plan.request=4;
        require(!sfx_prefix_plan.matches(history,4,99),"relabelled origin ordinal invalidates sealed digest");
        sfx_prefix_plan.request=2;
        require(sfx_prefix_plan.matches(history,2,99),"original transcript stays valid after A3 replay");
    }
    durable.reset();require(origin_credit.expired(),"retiring durable transcript releases its original metadata credit");
    std::puts("PASS actual prefix recorder/profile/loan/upload/callback/STOP/RAW-bridge CPU seams; no native proof");
    return 0;
}
'''


def run(build):
    root = Path(__file__).resolve().parents[3]
    paths = ['src/program/generate.cpp', 'src/program/serving_prefix_replay.hpp',
             'src/program/suffix_experiment_execution.inc', 'include/strata/prefill/prefill.hpp',
             'src/prefill/prefill.cpp', 'src/core/window_hash.cpp', 'src/program/serving_suffix_qualification.inc']
    pins = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
    generate = (root / paths[0]).read_text()
    prefill_h = (root / paths[3]).read_text()
    prefill = (root / paths[4]).read_text()
    execution = (root / paths[2]).read_text()
    qualification = (root / paths[6]).read_text()
    replacement = {
        'ACTUAL_PREFIX_GATE': qualification[qualification.index('    if(admission.prefix.active&&'):qualification.index('    // Every owner/buffer remains in scope')],
        'ACTUAL_ACTIVITY_TYPE': take(prefill_h, '    struct ReplayActivity', True),
        'ACTUAL_ACTIVITY': take(prefill, 'Prefill::ReplayActivity Prefill::replay_activity()').replace('Prefill::ReplayActivity Prefill::', 'ReplayActivity ', 1),
        'ACTUAL_SHOULD_STOP': take(prefill, '        if (should_stop && should_stop())'),
        'ACTUAL_GRACEFUL_STOP': take(prefill, '        if (c0 > 0 && hand_in_ == nullptr && stop_at_chunk && stop_at_chunk())'),
        'ACTUAL_PREV_UPDATE': take(prefill, '        for (int64_t t = 0; t < T; ++t) { prev[0]'),
        'ACTUAL_PLE_EPILOGUE': 'ss.ple_prev[0] = prev[0];\nss.ple_prev[1] = prev[1];',
        'ACTUAL_DRAIN': take(prefill, 'bool Prefill::drain_pipeline(').replace('Prefill::', '', 1),
        'ACTUAL_RUN': take(prefill, 'bool Prefill::run(').replace('Prefill::', '', 1),
        'ACTUAL_BRIDGE': take(execution, 'template<class Context>\nSuffixExperimentDriver::Step SuffixExperimentDriver::replay_prefill_piece(').replace('SuffixExperimentDriver::Step SuffixExperimentDriver::', 'static Step ', 1),
        'ACTUAL_UPLOAD': take(generate, '        auto res_upload =') + ';',
        'ACTUAL_REFILL_ISSUE': take(generate, '            auto shared_refill_issue =') + ';',
        'ACTUAL_REFILL_WAIT': take(generate, '            auto shared_refill_wait =') + ';',
        'ACTUAL_REFILL_ONE': take(generate, '            auto shared_refill_one =') + ';',
        'ACTUAL_REFILL': take(generate, '            auto shared_prefix_refill =') + ';' + take(generate, '            auto refill =') + ';',
        'ACTUAL_RESET_WITNESS': generate[generate.index('            const bool sfx_prefix_reset_from_zero='):generate.index('            auto sfx_prefix_policy =')],
        'ACTUAL_POLICY': take(generate, '            auto sfx_prefix_policy =') + ';',
        'ACTUAL_PREPARE': take(generate, '            auto sfx_prefix_prepare =') + ';',
        'ACTUAL_LOANS': take(generate, '            auto sfx_prefix_loans =') + ';',
        'ACTUAL_BASE': take(generate, '            auto sfx_prefix_base =') + ';',
        'ACTUAL_CALLBACK': take(generate, '            auto sfx_prefix_replay_piece =') + ';',
        'ACTUAL_CHECKED_REFILL': take(generate, '            auto sfx_prefix_checked_refill =') + ';',
    }
    # Keep even these two epilogue assignments byte-bound to the actual source.
    epilogue = '    ss.ple_prev[0] = prev[0];\n    ss.ple_prev[1] = prev[1];'
    if prefill.count(epilogue) != 1:
        raise RuntimeError('actual PLE epilogue changed')
    replacement['ACTUAL_PLE_EPILOGUE'] = epilogue
    code = PRELUDE
    for key, value in replacement.items():
        code = code.replace(key, value)
    path = build / 'actual-prefix-replay.cpp'
    binary = build / 'actual-prefix-replay'
    path.write_text(code)
    subprocess.run(['c++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Wconversion', '-Werror',
                    '-pthread', '-DSTRATA_SERVING_SUFFIX_REPAIR_AVAILABLE=1',
                    '-I' + str(root / 'include'), '-I' + str(root / 'src/program'),
                    '-isystem', '/usr/local/cuda/targets/x86_64-linux/include', str(path),
                    str(root / 'src/core/window_hash.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=45)
    subprocess.run([str(binary), 'checkpoint-ancestry'], check=True, timeout=10)
    if any(hashlib.sha256((root / p).read_bytes()).hexdigest() != h for p, h in pins.items()):
        raise RuntimeError('actual prefix sources changed during fixture')
    print('source_pins=' + repr(pins), flush=True)
    print('actual_extraction_sha256=' + hashlib.sha256(''.join(replacement.values()).encode()).hexdigest(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    if args.build_dir:
        args.build_dir.mkdir(parents=True, exist_ok=True)
        run(args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='strata-prefix-replay-') as directory:
            run(Path(directory))


if __name__ == '__main__':
    main()
