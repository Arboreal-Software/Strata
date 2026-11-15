#include "strata/core/window_disk.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
using namespace strata::core;
namespace fs=std::filesystem;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
struct Directory {std::string path;Directory(){char name[]="/tmp/strata-raw2-XXXXXX";auto* p=::mkdtemp(name);check(p,"mkdtemp");path=p;}~Directory(){fs::remove_all(path);}};
WindowDiskLimits limits(){return {1<<20,64<<20,4096,1024,128,16,256,64,8,8};}
WindowBudget budget(uint64_t disk=16<<20,uint64_t host=128<<20){return WindowBudget(WindowResources{{0},host,0,disk,2,0},WindowResources{{0},0,0,0,0,0});}
struct Fixture {
    SnapshotDescriptor s;WindowStateLayout l;BundleHandle bundle;
    Fixture(){s.key.engine_generation=3;s.key.window_generation=4;s.key.session_id="session";s.key.thread_id="thread";s.key.window_id="window";s.key.model_profile[0]=7;s.key.topology[0]=9;
        s.key.revision=5;s.key.append_epoch=6;s.key.selection_epoch=8;s.key.provenance_epoch=10;
        s.cut={11,3,2,1,12,{}};s.accepted_tokens={13,14,15};s.pending_tokens={15};s.cut.history=window_history_digest(s.accepted_tokens);s.approximation={{1,2,123,{4}}};s.native_layout="disk-fixture-native-v2";
        l.native_layout=s.native_layout;l.model_profile=s.key.model_profile;l.topology=s.key.topology;l.max_tokens=1024;l.max_payload_bytes=1<<20;l.layers={WindowStateLayout::LayerKind::gdn,WindowStateLayout::LayerKind::gdn};
        for(uint32_t i=0;i<2;++i){l.stages.push_back({i,int64_t(i),int64_t(i+1),i==0,{}});s.devices.push_back({i,int64_t(i),int64_t(i+1),s.cut,{}});add(i,StatePartKind::gdn_recurrence_and_convolution,i);}
        for(auto k:{StatePartKind::ple_history,StatePartKind::ple_previous_tokens,StatePartKind::ple_current_token})add(0,k,-1);
        freeze();
    }
    void add(uint32_t i,StatePartKind kind,int64_t layer){TensorLayout t{kind,layer,i,"u8",s.native_layout,{2,4},{4,1},0,8,8};l.stages[i].tensors.push_back(t);TensorPart p;p.layout=t;p.payload.resize(8,uint8_t(kind)+20);s.devices[i].tensors.push_back(std::move(p));}
    void freeze(){check(bool(BundleHandle::freeze(s,s.key,l,bundle)),"fixture freeze");}
};
WindowDiskOperation operation(int n){WindowDiskOperation o{3,4,{}};o.nonce[0]=uint8_t(n);return o;}
StateDigest catalog_id(){StateDigest d{};d[0]=1;return d;}
std::shared_ptr<WindowDiskStore> opened(const Directory& d,WindowBudget b=budget(),WindowDiskLimits l=limits()){std::shared_ptr<WindowDiskStore> store;const auto r=WindowDiskStore::open(d.path,l,b,store);if(!r)std::fprintf(stderr,"open: %s\n",r.detail.c_str());check(bool(r),"store open");return store;}
void compare(const SnapshotDescriptor& a,const SnapshotDescriptor& b){check(a.key==b.key&&a.cut==b.cut&&a.accepted_tokens==b.accepted_tokens&&a.pending_tokens==b.pending_tokens&&a.native_layout==b.native_layout&&a.speculation_disabled==b.speculation_disabled&&a.steering_enabled==b.steering_enabled,"complete key/cut/token/flags roundtrip");check(a.approximation.size()==b.approximation.size()&&a.approximation[0].ancestor==b.approximation[0].ancestor&&a.approximation[0].begin==b.approximation[0].begin&&a.approximation[0].end==b.approximation[0].end&&a.approximation[0].first_unresolved_unix_ms==b.approximation[0].first_unresolved_unix_ms,"approximation roundtrip");check(a.continuation.mode==b.continuation.mode&&a.continuation.adapter==b.continuation.adapter&&a.continuation.payload==b.continuation.payload,"continuation roundtrip");check(a.devices.size()==b.devices.size(),"all stages roundtrip");for(size_t i=0;i<a.devices.size();++i){const auto& x=a.devices[i];const auto& y=b.devices[i];check(x.id==y.id&&x.layer_begin==y.layer_begin&&x.layer_end==y.layer_end&&x.cut==y.cut&&x.tensors.size()==y.tensors.size(),"stage metadata roundtrip");for(size_t j=0;j<x.tensors.size();++j)check(x.tensors[j].layout==y.tensors[j].layout&&x.tensors[j].payload==y.tensors[j].payload,"RAW tensor and layout roundtrip");}}
#ifdef STRATA_DISK_SYSCALL_TEST
static thread_local bool fail_next_allocation=false;
static thread_local int allocation_countdown=-1;
void* operator new(std::size_t n){
    if(fail_next_allocation||allocation_countdown==0){fail_next_allocation=false;allocation_countdown=-1;throw std::bad_alloc();}
    if(allocation_countdown>0)--allocation_countdown;
    if(auto* p=std::malloc(n?n:1))return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n){return ::operator new(n);}
__attribute__((noinline)) void operator delete(void* p) noexcept{std::free(p);}
__attribute__((noinline)) void operator delete(void* p,std::size_t) noexcept{std::free(p);}
__attribute__((noinline)) void operator delete[](void* p) noexcept{std::free(p);}
__attribute__((noinline)) void operator delete[](void* p,std::size_t) noexcept{std::free(p);}
#endif
#ifdef STRATA_DISK_SYSCALL_TEST
static bool fail_directory_sync=false, fail_reads=false, short_reads=false, fail_stats=false;
static int stat_success_before_fail=0;
extern "C" int __real_fsync(int);
extern "C" ssize_t __real_read(int,void*,size_t);
extern "C" int __real_fstat(int,struct stat*);
extern "C" int __wrap_fsync(int fd){struct stat st{};if(fail_directory_sync&&__real_fstat(fd,&st)==0&&S_ISDIR(st.st_mode)){errno=EIO;return -1;}return __real_fsync(fd);}
extern "C" ssize_t __wrap_read(int fd,void* p,size_t n){if(fail_reads){errno=EIO;return -1;}if(short_reads)return 0;return __real_read(fd,p,n);}
extern "C" int __wrap_fstat(int fd,struct stat* st){if(fail_stats&&stat_success_before_fail--==0){errno=EIO;return -1;}return __real_fstat(fd,st);}

static bool fail_statat=false;
static int statat_hits=0,statat_error=EIO;
static bool statat_eintr_once=false;
extern "C" int __real_fstatat(int,const char*,struct stat*,int);
extern "C" int __wrap_fstatat(int fd,const char* name,struct stat* st,int flags){
    if(fail_statat){++statat_hits;if(statat_eintr_once){statat_eintr_once=false;errno=EINTR;}else errno=statat_error;return -1;}
    return __real_fstatat(fd,name,st,flags);
}
void binding_regression(const std::string& mode,Fixture& f){
    Directory dir;auto store=opened(dir);WindowDiskArtifact out;
    auto a=store->save(operation(100),f.bundle,f.s.key,f.l,catalog_id(),0,out);check(bool(a),"binding artifact A");
    const auto old=out.id();
    if(mode=="Bcatalog"){
        int failures=0,successes=0;
        for(int fault=0;fault<500;++fault){
            WindowDiskReceipt output;output.operation="previous-operation";output.artifact="previous-artifact";output.catalog="previous-catalog";output.catalog_revision=987;
            auto old_output=output;
            allocation_countdown=fault;auto r=store->catalog(catalog_id(),output);allocation_countdown=-1;
            if(!r){
                ++failures;const bool unchanged=output.operation==old_output.operation&&output.artifact==old_output.artifact&&output.catalog==old_output.catalog&&output.catalog_revision==old_output.catalog_revision&&output.phase==old_output.phase&&output.fingerprint==old_output.fingerprint;
                if(!unchanged)std::fprintf(stderr,"Bcatalog fault=%d error=%d output_changed=1\n",fault,int(r.error));
                check(unchanged,"catalog allocation failure leaves entire caller receipt unchanged");
            }else{++successes;check(output.artifact==a.receipt.artifact&&output.catalog_revision==1,"successful catalog output");}
        }
        check(failures&&successes,"catalog allocation sweep reached failure and success");
        std::fprintf(stderr,"Bcatalog failures=%d successes=%d\n",failures,successes);return;
    }
    if(mode=="Breturn"){
        f.s.devices[0].tensors[0].payload.visit(0,1,[](uint8_t* data,size_t,size_t){data[0]^=1;return true;});f.freeze();
        WindowDiskOptions o;bool armed=false;o.fault=[&](const char* phase){
            if(std::strcmp(phase,"receipt_dir_flush")==0){armed=true;fail_next_allocation=true;}
            return 0;
        };
        auto b=store->save(operation(101),f.bundle,f.s.key,f.l,catalog_id(),1,out,o);
        fail_next_allocation=false;
        std::fprintf(stderr,"Breturn armed=%d error=%d phase=%d output_changed=%d\n",armed,int(b.error),int(b.receipt.phase),out.id()!=old);
        check(armed&&b.error==WindowDiskError::budget_refused,"failure at final success result construction");
        check(out.id()==old,"final result allocation failure preserves caller output");
        check(store->query(operation(101)).receipt.phase==WindowDiskPhase::catalog_linked,"failed return does not deny durable commit");
        return;
    }
    WindowDiskOptions o;bool intent=false;o.fault=[&](const char* phase){
        if(std::strcmp(phase,"intent_publish")==0)intent=true;
        const bool hit=mode=="Bafter"?(intent&&std::strcmp(phase,"after_link")==0):std::strcmp(phase,"intent_file_flush")==0;
        if(hit){if(mode=="Boom")throw std::bad_alloc();return EIO;}return 0;
    };
    auto b=store->save(operation(101),f.bundle,f.s.key,f.l,catalog_id(),1,out,o);
    auto q=store->query(operation(101));
    std::fprintf(stderr,"%s error=%d phase=%d query=%d\n",mode.c_str(),int(b.error),int(b.receipt.phase),int(q.receipt.phase));
    check(out.id()==old,"failed B leaves A output");
    if(mode=="Bafter")check(b.receipt.phase==WindowDiskPhase::published_durability_unknown&&q.receipt.phase==WindowDiskPhase::durable_artifact,"own linked intent authorizes same-content binding");
    else check(b.receipt.phase==WindowDiskPhase::not_published&&q.receipt.phase==WindowDiskPhase::not_published,"unlinked B cannot claim A artifact publication");
}

void rereview_regression(const std::string& mode,Fixture& f){
    Directory dir;
    if(mode=="Rquota"||mode=="Rquota_plus"){
        auto lim=limits();const uint64_t base=lim.max_entries*1024+4096;
        const uint64_t operation_bytes=lim.max_entries*1024+lim.io_block_bytes+8192;
        lim.max_host_bytes=base+operation_bytes+(mode=="Rquota_plus"?512:0);
        auto store=opened(dir,budget(),lim);WindowDiskArtifact out;
        auto r=store->save(operation(91),f.bundle,f.s.key,f.l,catalog_id(),0,out);
        size_t entries=0;for(const auto& entry:fs::directory_iterator(dir.path))if(entry.path().filename()!=".manager-lock")++entries;
        std::fprintf(stderr,"quota mode=%s result=%d phase=%d entries=%zu\n",mode.c_str(),int(r.error),int(r.receipt.phase),entries);
        if(mode=="Rquota")check(r.error==WindowDiskError::budget_refused&&!out&&entries==0,"handle quota refusal precedes publication");
        else {
            check(bool(r)&&bool(out)&&r.receipt.phase==WindowDiskPhase::catalog_linked,"extra512 admits returned handle");
            const auto old_id=out.id();auto retry=store->save(operation(91),f.bundle,f.s.key,f.l,catalog_id(),0,out);
            check(retry.error==WindowDiskError::budget_refused&&retry.receipt.phase==WindowDiskPhase::catalog_linked&&out.id()==old_id,"retry refusal preserves stronger history and old output handle");
        }
        return;
    }
    auto store=opened(dir);WindowDiskArtifact out;bool armed=false;
    const bool catalog=mode.find("catalog")!=std::string::npos,oom=mode.find("oom")!=std::string::npos,existing=mode=="Rexisting";
    statat_error=mode.find("absent")!=std::string::npos?ENOENT:EIO;
    statat_eintr_once=mode.find("eintr")!=std::string::npos;
    if(existing){auto prior=store->save(operation(89),f.bundle,f.s.key,f.l,catalog_id(),1,out);check(prior.receipt.phase==WindowDiskPhase::durable_artifact,"existing publication fixture");}
    WindowDiskOptions opts;opts.fault=[&](const char* name){
        if(std::string(name)==(catalog?"catalog_publish":"artifact_publish"))armed=true;
        if((armed&&std::string(name)=="after_link")||(existing&&std::string(name)=="artifact_recover_flush")){
            fail_statat=true;
            if(oom)throw std::bad_alloc();
            return ENOSPC; // Probe's EIO must not replace the original failure.
        }
        return 0;
    };
    auto r=store->save(operation(90),f.bundle,f.s.key,f.l,catalog_id(),0,out,opts);
    fail_statat=false;
    auto recovered=store->query(operation(90));
    std::fprintf(stderr,"mode=%s result=%d phase=%d recovered=%d probes=%d errno=%d\n",mode.c_str(),int(r.error),int(r.receipt.phase),int(recovered.receipt.phase),statat_hits,r.system_error);
    check(!out&&statat_hits>0,"uncertain catch actually probed; output unchanged");
    check(r.error==(oom?WindowDiskError::budget_refused:WindowDiskError::io)&&r.system_error==(oom?0:ENOSPC),"original error survives probe failure");
    check(r.receipt.phase==(catalog?WindowDiskPhase::catalog_durability_unknown:WindowDiskPhase::published_durability_unknown),"uninspectable link publication remains uncertain");
    check(bool(recovered)&&recovered.receipt.phase==(catalog?WindowDiskPhase::catalog_linked:WindowDiskPhase::durable_artifact),"clean query recovers actual link effect");
}

void review_regression(const std::string& mode,Fixture& f){
    Directory dir;auto store=opened(dir);WindowDiskArtifact out;
    if(mode=="F1"){
        auto saved=store->save(operation(80),f.bundle,f.s.key,f.l,catalog_id(),1,out);
        check(saved.receipt.phase==WindowDiskPhase::durable_artifact,"F1 fixture");
        WindowDiskOptions o;o.fault=[](const char* s){return std::string(s)=="delete_dir_flush"?EIO:0;};
        check(store->erase(saved.receipt.artifact,o).receipt.phase==WindowDiskPhase::retirement_durability_unknown,"F1 uncertain erase");
        store.reset();fail_directory_sync=true;std::shared_ptr<WindowDiskStore> reopened;
        auto r=WindowDiskStore::open(dir.path,limits(),budget(),reopened);fail_directory_sync=false;
        check(r.error==WindowDiskError::io&&r.system_error==EIO&&!reopened,"F1 restart must confirm directory durability");
        reopened=opened(dir);check(reopened->query(operation(80)).receipt.phase==WindowDiskPhase::retired_artifact,"F1 confirmed retirement");
    }else if(mode=="F2"){
        auto saved=store->save(operation(81),f.bundle,f.s.key,f.l,catalog_id(),1,out);
        check(bool(store->erase(saved.receipt.artifact)),"F2 erase");
        auto retry=store->save(operation(81),f.bundle,f.s.key,f.l,catalog_id(),1,out);
        check(retry.error==WindowDiskError::conflict&&retry.receipt.phase==WindowDiskPhase::retired_artifact,"F2 retry preserves retirement");
        Fixture changed;changed.s.devices[0].tensors[0].payload.visit(0,1,[](uint8_t* p,size_t,size_t){*p=99;return true;});changed.freeze();
        check(bool(store->save(operation(82),changed.bundle,changed.s.key,changed.l,catalog_id(),0,out)),"F2 linked fixture");
        retry=store->save(operation(82),f.bundle,f.s.key,f.l,catalog_id(),0,out);
        check(retry.error==WindowDiskError::conflict&&retry.receipt.phase==WindowDiskPhase::catalog_linked,"F2 mismatch preserves linked history");
    }else{
        check(bool(store->save(operation(83),f.bundle,f.s.key,f.l,catalog_id(),0,out)),"F3 fixture");
        BundleHandle loaded=f.bundle;const auto* prior=loaded.get();fail_reads=true;
        auto r=store->load(out,f.s.key,f.l,loaded);fail_reads=false;
        check(r.error==WindowDiskError::io&&r.system_error==EIO&&loaded.get()==prior,"F3 physical read errno preserved");
        fail_reads=true;r=store->query(operation(83));fail_reads=false;
        check(r.error==WindowDiskError::io&&r.system_error==EIO&&r.receipt.phase==WindowDiskPhase::unknown,"F3 query read failure unknown");
        short_reads=true;r=store->load(out,f.s.key,f.l,loaded);short_reads=false;
        check(r.error==WindowDiskError::invalid&&loaded.get()==prior,"F3 EOF is invalid and preserves output");
        fail_stats=true;stat_success_before_fail=1;r=store->load(out,f.s.key,f.l,loaded);fail_stats=false;
        check(r.error==WindowDiskError::io&&r.system_error==EIO&&loaded.get()==prior,"F3 second fstat errno preserved");
    }
    std::puts("review regression passed");
}
#endif
int main(){Fixture f;
#ifdef STRATA_DISK_SYSCALL_TEST
    if(const char* mode=std::getenv("STRATA_DISK_REVIEW")){if(mode[0]=='B')binding_regression(mode,f);else if(mode[0]=='R')rereview_regression(mode,f);else review_regression(mode,f);return 0;}
#endif
    {
        Directory dir;auto b=budget();auto store=opened(dir,b);WindowDiskArtifact artifact;
        auto saved=store->save(operation(1),f.bundle,f.s.key,f.l,catalog_id(),0,artifact);
        if(!saved)std::fprintf(stderr,"save: %s\n",saved.detail.c_str());
        check(bool(saved)&&saved.receipt.phase==WindowDiskPhase::catalog_linked,"transaction publishes and catalogs");
        check(b.snapshot().reserved.disk_bytes>0,"published files retain disk reservations after API return");
        BundleHandle loaded;check(bool(store->load(artifact,f.s.key,f.l,loaded)),"RAW load");compare(f.s,*loaded.get());
        check(b.snapshot().reserved.io_slots==0,"loaded immutable bundle retains memory but releases operation IO slot");
        auto wrong=f.s.key;wrong.model_profile[0]++;const auto* unchanged=loaded.get();check(!store->load(artifact,wrong,f.l,loaded)&&loaded.get()==unchanged,"profile mismatch preserves output");wrong=f.s.key;wrong.topology[0]++;check(!store->load(artifact,wrong,f.l,loaded),"topology mismatch");
        WindowDiskArtifact again;check(bool(store->save(operation(1),f.bundle,f.s.key,f.l,catalog_id(),0,again))&&again.id()==artifact.id(),"same operation/input retries idempotently");
        Fixture changed;changed.s.devices[1].tensors[0].payload.resize(9);changed.s.devices[1].tensors[0].payload.resize(8,99); // modify below without changing extent
        changed.s.devices[1].tensors[0].payload.visit(0,1,[](uint8_t* p,size_t,size_t){*p=99;return true;});changed.freeze();
        check(store->save(operation(1),changed.bundle,changed.s.key,changed.l,catalog_id(),0,again).error==WindowDiskError::conflict,"same operation/different input refused");
        auto conflict=store->save(operation(2),changed.bundle,changed.s.key,changed.l,catalog_id(),0,again);
        check(conflict.error==WindowDiskError::conflict&&conflict.receipt.phase==WindowDiskPhase::durable_artifact,"CAS conflict retains new durable artifact");
        WindowDiskReceipt current;check(bool(store->catalog(catalog_id(),current))&&current.artifact==artifact.id()&&current.catalog_revision==1,"CAS conflict preserves old good catalog");
        WindowDiskArtifact unlinked;check(bool(store->acquire(conflict.receipt.artifact,unlinked)),"acquire uncataloged published artifact");check(store->erase(unlinked.id()).error==WindowDiskError::conflict,"lease blocks deletion");const auto retired_id=unlinked.id();unlinked={};check(bool(store->erase(retired_id)),"unleased uncataloged artifact deletes");check(store->query(operation(2)).receipt.phase==WindowDiskPhase::retired_artifact,"query does not mistake historical publication for no effect after deletion");
        check(store->erase(artifact.id()).error==WindowDiskError::conflict,"catalog/lease protects old good");
        check(!store->acquire("../outside",again),"opaque ID blocks traversal");
        std::shared_ptr<WindowDiskStore> second;check(WindowDiskStore::open(dir.path,limits(),b,second).error==WindowDiskError::conflict,"one manager owns namespace");
        again={};artifact={};loaded={};store.reset();check(b.snapshot().reserved.disk_bytes==0,"closing manager releases only in-memory accounting");
        store=opened(dir,b);check(b.snapshot().reserved.disk_bytes>0&&store->query(operation(1)).receipt.phase==WindowDiskPhase::catalog_linked,"restart reaccounts and reconciles receipt");
    }
    {
        Directory dir;auto store=opened(dir);WindowDiskArtifact artifact;
        check(bool(store->save(operation(7),f.bundle,f.s.key,f.l,catalog_id(),0,artifact)),"corruption fixture");
        const auto path=dir.path+"/artifact-"+artifact.id()+".raw2";
        std::ifstream input(path,std::ios::binary);std::vector<char> original((std::istreambuf_iterator<char>(input)),{});input.close();
        auto replace=[&](const std::vector<char>& bytes){std::ofstream output(path,std::ios::binary|std::ios::trunc);output.write(bytes.data(),bytes.size());};
        BundleHandle loaded=f.bundle;const auto* keep=loaded.get();
        auto bytes=original;bytes.back()^=1;replace(bytes);check(!store->load(artifact,f.s.key,f.l,loaded)&&loaded.get()==keep,"RAW payload checksum failure preserves old bundle");
        bytes=original;bytes[8]=3;replace(bytes);check(!store->load(artifact,f.s.key,f.l,loaded),"unknown format version refused");
        bytes=original;bytes.resize(bytes.size()-1);replace(bytes);check(!store->load(artifact,f.s.key,f.l,loaded),"truncation refused");
        bytes=original;for(size_t i=72;i<80;++i)bytes[i]=char(255);replace(bytes);check(!store->load(artifact,f.s.key,f.l,loaded),"malicious UINT64_MAX string length bounded before allocation");
        replace(original);artifact={};store.reset();auto small=limits();small.max_tokens=2;store=opened(dir,budget(),small);check(bool(store->acquire(fs::path(path).filename().string().substr(9,64),artifact)),"bounded reader acquires fixture");check(!store->load(artifact,f.s.key,f.l,loaded),"token bound enforced before vector allocation");
    }
    {
        Directory dir;std::ofstream outside(dir.path+"/untrusted");outside<<"x";outside.close();std::shared_ptr<WindowDiskStore> store;check(!WindowDiskStore::open(dir.path,limits(),budget(),store),"unexpected namespace files refused");
        fs::remove(dir.path+"/untrusted");fs::create_symlink("/etc/passwd",dir.path+"/artifact-"+std::string(64,'0')+".raw2");check(!WindowDiskStore::open(dir.path,limits(),budget(),store),"symlink substitution refused");
    }
    for(const char* phase:{"write","intent_file_flush","intent_publish","intent_dir_flush","artifact_file_flush","artifact_publish","artifact_dir_flush","catalog_file_flush","catalog_publish","catalog_dir_flush","receipt_publish","receipt_dir_flush"}){
        Directory dir;auto store=opened(dir);WindowDiskArtifact out;WindowDiskOptions fault;fault.fault=[&](const char* step){return std::strcmp(step,phase)==0?ENOSPC:0;};
        auto failed=store->save(operation(3),f.bundle,f.s.key,f.l,catalog_id(),0,out,fault);check(!failed&&!out&&failed.system_error==ENOSPC,"injected disk-full failure preserves errno and refuses result");
        if(std::string(phase)=="artifact_dir_flush"){
            check(failed.receipt.phase==WindowDiskPhase::published_durability_unknown,"post-publication failure is not no effect");
            WindowDiskOptions recovery_fault;recovery_fault.fault=[](const char* step){return std::string(step)=="recover_dir_flush"?EIO:0;};
            auto query=store->query(operation(3),recovery_fault);
            check(!query&&query.receipt.phase==WindowDiskPhase::published_durability_unknown,"failed recovery fsync never reports not published");
        }
        if(std::string(phase)=="catalog_dir_flush")check(failed.receipt.phase==WindowDiskPhase::catalog_durability_unknown,"post-catalog failure reports ambiguous CAS publication");
        auto retry=store->save(operation(3),f.bundle,f.s.key,f.l,catalog_id(),0,out);if(!retry)std::fprintf(stderr,"retry %s: %s\n",phase,retry.detail.c_str());check(bool(retry),"safe retry after each IO fault");
    }
    for(const char* phase:{"write","intent_file_flush","intent_publish","intent_dir_flush","after_intent_publish","artifact_file_flush","artifact_publish","artifact_dir_flush","after_artifact_publish","catalog_file_flush","catalog_publish","catalog_dir_flush","after_catalog_publish","receipt_file_flush","receipt_publish","receipt_dir_flush","after_receipt_publish","artifact_link_gap"}){
        Directory dir;const auto pid=fork();check(pid>=0,"fork crash test");if(!pid){auto store=opened(dir);WindowDiskArtifact out;bool artifact=false;WindowDiskOptions crash;crash.fault=[&](const char* step){if(std::string(step)=="artifact_publish")artifact=true;if(std::string(step)==phase||(std::string(phase)=="artifact_link_gap"&&artifact&&std::string(step)=="after_link"))_exit(77);return 0;};store->save(operation(4),f.bundle,f.s.key,f.l,catalog_id(),0,out,crash);_exit(78);}
        int status=0;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==77,"crashed at requested cut");auto b=budget();auto store=opened(dir,b);uint64_t retained=0;for(const auto& entry:fs::directory_iterator(dir.path))retained+=entry.file_size();check(b.snapshot().reserved.disk_bytes==retained,"restart exactly accounts retained temp/published file bytes");auto q=store->query(operation(4));check(bool(q),"crash outcome query");WindowDiskArtifact out;auto retry=store->save(operation(4),f.bundle,f.s.key,f.l,catalog_id(),0,out);if(!retry)std::fprintf(stderr,"crash retry %s: %s\n",phase,retry.detail.c_str());check(bool(retry),"crash retry reconciles without replacing old good");
    }
    {
        Directory dir;auto store=opened(dir);WindowDiskArtifact out;WindowDiskOptions cancelled;cancelled.cancelled=[] {return true;};check(store->save(operation(5),f.bundle,f.s.key,f.l,catalog_id(),0,out,cancelled).error==WindowDiskError::cancelled,"pre-publication cancellation");
        bool stop=false;cancelled.cancelled=[&]{return stop;};cancelled.fault=[&](const char* phase){if(std::string(phase)=="after_artifact_publish")stop=true;return 0;};auto r=store->save(operation(5),f.bundle,f.s.key,f.l,catalog_id(),0,out,cancelled);check(r.error==WindowDiskError::cancelled&&r.receipt.phase!=WindowDiskPhase::not_published,"post-publication cancellation retains receipt");check(store->query(operation(5)).receipt.phase==WindowDiskPhase::durable_artifact,"cancelled publication reconciles");
    }
    {
        Directory dir;auto b=budget();auto store=opened(dir,b);WindowDiskArtifact out;
        check(bool(store->save(operation(11),f.bundle,f.s.key,f.l,catalog_id(),0,out)),"directory read failure fixture");
        const auto charged=b.snapshot().reserved.disk_bytes;int calls=0;WindowDiskOptions failure;
        failure.fault=[&](const char* step){return std::string(step)=="readdir"&&++calls==4?EIO:0;};
        const auto q=store->query(operation(11),failure);
        check(q.error==WindowDiskError::io&&q.system_error==EIO&&q.receipt.phase==WindowDiskPhase::unknown&&b.snapshot().reserved.disk_bytes==charged,
              "partial readdir error is not EOF/no-publication and releases no quota");
    }
    {
        Directory dir;auto b=budget();auto store=opened(dir,b);WindowDiskArtifact out;
        WindowDiskOptions initial;initial.fault=[](const char* step){return std::string(step)=="artifact_file_flush"?EIO:0;};
        check(!store->save(operation(12),f.bundle,f.s.key,f.l,catalog_id(),0,out,initial),"partial temp fixture");
        const auto charged=b.snapshot().reserved.disk_bytes;WindowDiskOptions failed_delete;
        failed_delete.fault=[](const char* step){return std::string(step)=="temporary_delete_dir_flush"?EIO:0;};
        check(!store->save(operation(12),f.bundle,f.s.key,f.l,catalog_id(),0,out,failed_delete)&&b.snapshot().reserved.disk_bytes==charged,
              "temp unlink with failed directory fsync retains charge");
        WindowDiskOptions failed_reconcile;failed_reconcile.fault=[](const char* step){return std::string(step)=="reconcile_delete_dir_flush"?EIO:0;};
        check(!store->query(operation(12),failed_reconcile)&&b.snapshot().reserved.disk_bytes==charged,"ambiguous temp deletion stays charged until confirmed");
        check(bool(store->query(operation(12)))&&b.snapshot().reserved.disk_bytes<charged,"confirmed absent temp releases quota");
        check(bool(store->save(operation(12),f.bundle,f.s.key,f.l,catalog_id(),0,out)),"temp cleanup retry safely publishes");
    }
    {
        Directory dir;auto b=budget();auto store=opened(dir,b);WindowDiskArtifact out;bool artifact=false;WindowDiskOptions gap;
        gap.fault=[&](const char* step){if(std::string(step)=="artifact_publish")artifact=true;return artifact&&std::string(step)=="after_link"?EIO:0;};
        auto saved=store->save(operation(13),f.bundle,f.s.key,f.l,catalog_id(),0,out,gap);check(!saved&&saved.receipt.phase==WindowDiskPhase::published_durability_unknown,"live publication gap fixture");
        const auto charged=b.snapshot().reserved.disk_bytes;WindowDiskOptions failed_sync;failed_sync.fault=[](const char* step){return std::string(step)=="temporary_delete_dir_flush"?EIO:0;};
        check(!store->query(operation(13),failed_sync)&&b.snapshot().reserved.disk_bytes==charged,"publication recovery transfers same charge without a release gap");
        check(store->query(operation(13)).receipt.phase==WindowDiskPhase::durable_artifact,"hardlink-gap recovery confirms artifact");
    }
    for(const char* phase:{"retirement_file_flush","retirement_publish","retirement_dir_flush","delete_unlink","delete_dir_flush"}){
        Directory dir;auto b=budget();auto store=opened(dir,b);WindowDiskArtifact out;
        auto saved=store->save(operation(8),f.bundle,f.s.key,f.l,catalog_id(),1,out);
        check(saved.error==WindowDiskError::conflict&&saved.receipt.phase==WindowDiskPhase::durable_artifact,"uncataloged retirement fixture");
        const auto charge=b.snapshot().reserved.disk_bytes;WindowDiskOptions fault;fault.fault=[&](const char* step){return std::string(step)==phase?EIO:0;};
        const auto failed=store->erase(saved.receipt.artifact,fault);
        check(!failed&&b.snapshot().reserved.disk_bytes>=charge,"failed retirement retains original disk charge");
        if(std::string(phase)=="delete_dir_flush")check(failed.receipt.phase==WindowDiskPhase::retirement_durability_unknown,"post-unlink failure reports uncertain retirement");
        else check(failed.receipt.phase==WindowDiskPhase::durable_artifact,"failed durable marker cannot claim deleted or never published");
        auto queried=store->query(operation(8));check(bool(queried),"retirement query reconciles");
        if(queried.receipt.phase!=WindowDiskPhase::retired_artifact)check(bool(store->erase(saved.receipt.artifact)),"retirement retry preserves publication history");
        check(store->query(operation(8)).receipt.phase==WindowDiskPhase::retired_artifact,"retired remains distinct from absent operation");
        check(store->query(operation(99)).receipt.phase==WindowDiskPhase::not_published,"unknown operation has no publication history");
    }
    {
        Directory dir;auto store=opened(dir);WindowDiskArtifact artifact;
        auto saved=store->save(operation(9),f.bundle,f.s.key,f.l,catalog_id(),0,artifact);check(bool(saved),"catalog deletion fixture");
        const auto id=artifact.id();artifact={};check(store->erase(id).error==WindowDiskError::conflict,"catalog alone prevents retirement without leases");
        store.reset();std::shared_ptr<WindowDiskStore> refused;
        check(WindowDiskStore::open(dir.path,limits(),budget(100),refused).error==WindowDiskError::budget_refused&&!refused,"restart reaccounts retained disk before new admission");
        uint64_t retained=0;for(const auto& entry:fs::directory_iterator(dir.path))retained+=entry.file_size();
        store=opened(dir,budget(retained+100));WindowDiskArtifact new_artifact;
        check(store->save(operation(10),f.bundle,f.s.key,f.l,catalog_id(),1,new_artifact).error==WindowDiskError::budget_refused,
              "existing retained artifacts plus new operation overhead consume quota together");
        check(store->query(operation(9)).receipt.phase==WindowDiskPhase::catalog_linked,"quota-refused restart/save preserves old good");
    }
    {
        Directory dir;auto tiny=budget(100);auto store=opened(dir,tiny);WindowDiskArtifact out;check(store->save(operation(6),f.bundle,f.s.key,f.l,catalog_id(),0,out).error==WindowDiskError::budget_refused,"disk quota refuses write");
    }
    {
        Directory dir;auto path=dir.path+"/model";std::string bytes(4<<20,'a');{std::ofstream file(path,std::ios::binary);file.write(bytes.data(),bytes.size());}StateDigest a{},b{};check(bool(window_disk_full_digest(path,bytes.size(),a)),"full model hash");{std::fstream file(path,std::ios::in|std::ios::out|std::ios::binary);file.seekp(2<<20);file.put('z');}check(bool(window_disk_full_digest(path,bytes.size(),b))&&a!=b,"same-size middle model edit changes full digest");
        struct stat before{};check(::stat(path.c_str(),&before)==0,"model timestamp fixture");bool changed=false;WindowDiskOptions during_hash;
        during_hash.fault=[&](const char* phase){if(!changed&&std::string(phase)=="model_hash_chunk"){
            changed=true;std::fstream file(path,std::ios::in|std::ios::out|std::ios::binary);file.seekp(3<<20);file.put('q');file.close();
            timespec stamps[2]={before.st_atim,before.st_mtim};stamps[1].tv_nsec=(stamps[1].tv_nsec+1)%1000000000;
            check(::utimensat(AT_FDCWD,path.c_str(),stamps,0)==0,"set changed same-second nanosecond timestamp");
        }return 0;};
        const auto old=b;check(!window_disk_full_digest(path,bytes.size(),b,during_hash)&&b==old,"same-size same-second mutation detected via nanosecond metadata; output unchanged");
    }
    {
        Directory dir;Fixture segmented;const size_t bytes=ConversationBuffer::segment_bytes+17;
        auto& tensor=segmented.s.devices[1].tensors[0];tensor.payload.resize(bytes,71);tensor.layout.bytes=bytes;tensor.layout.valid_elements=bytes;tensor.layout.shape={bytes};tensor.layout.strides={1};
        segmented.l.stages[1].tensors[0]=tensor.layout;segmented.l.max_payload_bytes=32<<20;segmented.freeze();
        auto bounds=limits();bounds.max_file_bytes=32<<20;bounds.max_host_bytes=256<<20;bounds.io_block_bytes=4096;
        auto store=opened(dir,budget(128<<20,512<<20),bounds);WindowDiskArtifact artifact;BundleHandle loaded;
        check(bool(store->save(operation(14),segmented.bundle,segmented.s.key,segmented.l,catalog_id(),0,artifact)),"multi-segment RAW artifact saves without flattening");
        check(bool(store->load(artifact,segmented.s.key,segmented.l,loaded)),"multi-segment RAW artifact loads privately");compare(segmented.s,*loaded.get());
    }
    {
        Directory dir;auto bound=limits();bound.max_entries=3;const auto pid=fork();check(pid>=0,"fork namespace-headroom case");
        if(!pid){auto store=opened(dir,budget(),bound);bool artifact=false;WindowDiskArtifact out;WindowDiskOptions crash;
            crash.fault=[&](const char* step){if(std::string(step)=="artifact_publish")artifact=true;if(artifact&&std::string(step)=="after_link")_exit(77);return 0;};
            store->save(operation(15),f.bundle,f.s.key,f.l,catalog_id(),0,out,crash);_exit(78);
        }
        int status=0;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==77,"publication reached bounded spare entry");
        auto store=opened(dir,budget(),bound);check(store->query(operation(15)).receipt.phase==WindowDiskPhase::durable_artifact,"exact-entry-bound crash remains recoverable");
        WindowDiskArtifact out;check(store->save(operation(15),f.bundle,f.s.key,f.l,catalog_id(),0,out).error==WindowDiskError::budget_refused,
              "catalog publication refuses without consuming reserved namespace headroom");
    }
    {
        Directory dir;const auto pid=fork();check(pid>=0,"fork reentrant output-owner fixture");
        if(!pid){
            alarm(10); // old implementation deadlocks; bound that regression without sleeps
            WindowBudget single_io(WindowResources{{0},128<<20,0,16<<20,1,0},WindowResources{{0},0,0,0,0,0});
            auto store=opened(dir,single_io);WindowDiskArtifact artifact;
            check(bool(store->save(operation(16),f.bundle,f.s.key,f.l,catalog_id(),0,artifact)),"output-owner fixture saved");
            bool reentered=false;
            auto owner=std::shared_ptr<const void>(new int(1),[&](const void* p){
                check(bool(store->query(operation(16))),"prior bundle owner reenters after load unlock and IO release");
                reentered=true;delete static_cast<const int*>(p);
            });
            BundleHandle previous;check(bool(BundleHandle::freeze(f.s,f.s.key,f.l,previous,owner)),"custom old output owner");owner.reset();
            check(bool(store->load(artifact,f.s.key,f.l,previous))&&reentered,"load replacement retires previous owner outside mutex");
            bool source_reentered=false;
            auto source_owner=std::shared_ptr<const void>(new int(2),[&](const void* p){
                check(bool(store->query(operation(17))),"failed save source retires after mutex and IO release");
                source_reentered=true;delete static_cast<const int*>(p);
            });
            BundleHandle source;check(bool(BundleHandle::freeze(f.s,f.s.key,f.l,source,source_owner)),"custom source owner");source_owner.reset();
            WindowDiskOptions failure;failure.fault=[&](const char* step){if(std::string(step)=="write"){source={};return EIO;}return 0;};
            WindowDiskArtifact output;check(!store->save(operation(17),source,f.s.key,f.l,catalog_id(),1,output,failure)&&source_reentered,
                "source retained across callback cancellation then released outside lock on failure");
            alarm(0);std::exit(0);
        }
        int status=0;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"reentrant owner test completes without deadlock");
    }
    {
        Directory first,second;auto a_budget=budget();auto a=opened(first,a_budget);auto b=opened(second);WindowDiskArtifact prior,current;
        check(bool(a->save(operation(18),f.bundle,f.s.key,f.l,catalog_id(),0,prior)),"other-manager output fixture");a.reset();
        check(bool(b->save(operation(19),f.bundle,f.s.key,f.l,catalog_id(),0,current)),"current-manager fixture");
        check(bool(b->acquire(current.id(),prior))&&a_budget.snapshot().reserved.disk_bytes==0,"acquire replacement retires previous manager ownership");
        a=opened(first,a_budget);check(bool(a->acquire(current.id(),prior)),"reacquire old-manager artifact");a.reset();
        check(bool(b->save(operation(20),f.bundle,f.s.key,f.l,catalog_id(),1,prior))&&a_budget.snapshot().reserved.disk_bytes==0,"save replacement retires previous manager ownership");
    }
    {
        Directory dir;auto lim=limits();lim.max_host_bytes=1;std::shared_ptr<WindowDiskStore> refused;
        check(WindowDiskStore::open(dir.path,lim,budget(),refused).error==WindowDiskError::budget_refused&&!refused&&fs::is_empty(dir.path),"hard host envelope refuses startup before lock creation");
        auto b=budget();auto occupied=b.reserve(WindowResources{{0},0,0,0,2,0},ReservationPurpose::foreground);check(bool(occupied),"occupy IO");
        check(WindowDiskStore::open(dir.path,limits(),b,refused).error==WindowDiskError::budget_refused&&fs::is_empty(dir.path),"IO refusal before startup filesystem effect");
        occupied.lease.reset();auto store=opened(dir,b);WindowDiskArtifact artifact;
        check(bool(store->save(operation(90),f.bundle,f.s.key,f.l,catalog_id(),0,artifact)),"admission fixture");
        const auto baseline=store->host_usage().reserved_bytes;BundleHandle decoded;
        check(bool(store->load(artifact,f.s.key,f.l,decoded))&&store->host_usage().reserved_bytes>baseline,"load retains local host allowance");
        decoded={};check(store->host_usage().reserved_bytes==baseline,"decoded owner releases allowance");
        occupied=b.reserve(WindowResources{{0},0,0,0,2,0},ReservationPurpose::foreground);int scans=0;WindowDiskOptions probe;probe.fault=[&](const char* step){if(std::string(step)=="readdir")++scans;return 0;};
        WindowDiskArtifact keep=artifact;WindowDiskReceipt cat;BundleHandle output=f.bundle;
        check(store->query(operation(90),probe).error==WindowDiskError::budget_refused&&scans==0,"query IO refusal precedes reconcile");
        check(store->save(operation(91),f.bundle,f.s.key,f.l,catalog_id(),1,keep,probe).error==WindowDiskError::budget_refused&&scans==0,"save IO refusal precedes reconcile");
        check(store->erase(artifact.id(),probe).error==WindowDiskError::budget_refused&&scans==0,"erase IO refusal precedes reconcile");
        check(store->acquire(artifact.id(),keep).error==WindowDiskError::budget_refused&&keep.id()==artifact.id(),"acquire needs IO admission");
        check(store->catalog(catalog_id(),cat).error==WindowDiskError::budget_refused,"catalog needs IO admission");
        check(store->load(artifact,f.s.key,f.l,output).error==WindowDiskError::budget_refused&&output.get()==f.bundle.get(),"load needs IO before reads");
        occupied.lease.reset();check(store->host_usage().reserved_bytes==baseline,"refusals leak no allowance");
        check(bool(store->load(artifact,f.s.key,f.l,decoded)),"retained decoded fixture");artifact={};keep={};store.reset();
        check(b.snapshot().reserved.host_bytes>0,"decoded charge survives store destruction");decoded={};check(b.snapshot().reserved.host_bytes==0,"last decoded owner releases budget");
    }
    {
        Directory dir;auto lim=limits();lim.max_host_bytes=1024*1024;auto store=opened(dir,budget(),lim);WindowDiskArtifact artifact;
        check(bool(store->save(operation(92),f.bundle,f.s.key,f.l,catalog_id(),0,artifact)),"bounded retained fixture");
        std::vector<BundleHandle> retained;bool refused=false;
        for(int i=0;i<16;++i){BundleHandle next;auto r=store->load(artifact,f.s.key,f.l,next);if(!r){check(r.error==WindowDiskError::budget_refused&&!next,"retained bundles collectively refuse new load");refused=true;break;}retained.push_back(std::move(next));check(store->host_usage().reserved_bytes<=lim.max_host_bytes,"retained envelope never exceeds bound");}
        check(refused&&!retained.empty(),"retained host cap exercised");retained.clear();BundleHandle next;check(bool(store->load(artifact,f.s.key,f.l,next)),"released host allowance admits again");
    }
    std::puts("window_disk_test passed");
}
