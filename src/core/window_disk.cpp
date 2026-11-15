#include "strata/core/window_disk.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <type_traits>
#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {
struct Failure { WindowDiskError code; std::string text; int system_error; };
[[noreturn]] void fail(WindowDiskError e,const char* text){throw Failure{e,text,e==WindowDiskError::io?errno:0};}
void require(bool ok,const char* why){if(!ok)fail(WindowDiskError::invalid,why);}
std::string hex(const StateDigest& d){static const char h[]="0123456789abcdef";std::string s; s.reserve(64);for(auto b:d){s+=h[b>>4];s+=h[b&15];}return s;}
bool opaque(const std::string& s){return s.size()==64&&std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
uint64_t sum(uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,"disk size overflow");return a+b;}
uint64_t product(uint64_t a,uint64_t b){require(!a||b<=UINT64_MAX/a,"disk size multiplication overflow");return a*b;}
void limits_ok(const WindowDiskLimits& l){require(l.max_file_bytes>=56&&l.max_host_bytes&&l.max_string_bytes&&l.max_tokens&&l.max_tensors&&l.max_entries&&l.io_block_bytes&&l.io_block_bytes<=SIZE_MAX&&l.max_stages&&l.max_rank,"zero or unsupported disk bounds");}
void checkpoint(const WindowDiskOptions& o,const char* name){if(o.fault){const int e=o.fault(name);if(e){errno=e;fail(WindowDiskError::io,name);}}}
void cancelled(const WindowDiskOptions& o){if(o.cancelled&&o.cancelled())fail(WindowDiskError::cancelled,"disk operation cancelled");}
#ifndef _WIN32
struct Fd {int n=-1;Fd()=default;explicit Fd(int x):n(x){}Fd(const Fd&)=delete;Fd& operator=(const Fd&)=delete;Fd(Fd&& o)noexcept:n(std::exchange(o.n,-1)){}~Fd(){if(n>=0)::close(n);}};
void write_all(int fd,const void* data,size_t bytes,const WindowDiskLimits& limits,const WindowDiskOptions& options){
    auto* p=static_cast<const uint8_t*>(data);
    while(bytes){cancelled(options);checkpoint(options,"write");const size_t part=std::min<uint64_t>(bytes,limits.io_block_bytes);const auto n=::write(fd,p,part);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)fail(WindowDiskError::io,"write failed");
        p+=n;bytes-=size_t(n);}
}
void read_all(int fd,void* data,size_t bytes){auto* p=static_cast<uint8_t*>(data);while(bytes){const auto n=::read(fd,p,bytes);if(n<0&&errno==EINTR)continue;if(n<0)fail(WindowDiskError::io,"disk object read failed");if(n==0)fail(WindowDiskError::invalid,"truncated disk object");p+=n;bytes-=size_t(n);}}
#endif
// Explicit canonical RAW encoding. A dry hashing pass computes length/digest;
// the write pass visits immutable payload segments without flattening them.
struct Encoder {
    std::function<void(const void*,size_t)> emit;
    uint64_t count=0;
    void bytes(const void* p,size_t n){count=sum(count,n);emit(p,n);}
    void u64(uint64_t n){uint8_t b[8];for(int i=0;i<8;++i)b[i]=uint8_t(n>>(8*i));bytes(b,8);}
    void u32(uint32_t n){uint8_t b[4];for(int i=0;i<4;++i)b[i]=uint8_t(n>>(8*i));bytes(b,4);}
    void text(const std::string& s){u64(s.size());bytes(s.data(),s.size());}
    void digest(const StateDigest& d){bytes(d.data(),d.size());}
};
void key(Encoder& e,const WindowKey& k){e.u64(k.engine_generation);e.u64(k.window_generation);e.text(k.session_id);e.text(k.thread_id);e.text(k.window_id);e.u64(k.revision);e.u64(k.append_epoch);e.u64(k.selection_epoch);e.u64(k.provenance_epoch);e.digest(k.model_profile);e.digest(k.topology);}
void cut(Encoder& e,const StateCut& c){e.u64(c.id);e.u64(c.accepted);e.u64(c.forwarded);e.u64(c.exact_frontier);e.u64(c.output_watermark);e.digest(c.history);}
void layout(Encoder& e,const TensorLayout& t){e.u32(uint32_t(t.kind));e.u64(uint64_t(t.layer));e.u32(t.device_part);e.text(t.dtype);e.text(t.native_layout);e.u64(t.shape.size());for(auto n:t.shape)e.u64(n);e.u64(t.strides.size());for(auto n:t.strides)e.u64(n);e.u64(t.logical_begin);e.u64(t.valid_elements);e.u64(t.bytes);}
void bounds(const SnapshotDescriptor& s,const WindowDiskLimits& l){
    for(const auto* p:{&s.key.session_id,&s.key.thread_id,&s.key.window_id,&s.native_layout,&s.continuation.adapter})require(p->size()<=l.max_string_bytes,"metadata string exceeds bound");
    require(s.accepted_tokens.size()<=l.max_tokens&&s.pending_tokens.size()<=l.max_tokens&&s.devices.size()<=l.max_stages&&s.approximation.size()<=l.max_regions&&s.continuation.payload.size()<=l.max_file_bytes,"metadata count exceeds bound");
    uint64_t tensors=0;for(const auto& d:s.devices){tensors=sum(tensors,d.tensors.size());require(tensors<=l.max_tensors,"tensor count exceeds bound");for(const auto& t:d.tensors)require(t.layout.shape.size()<=l.max_rank&&t.layout.strides.size()<=l.max_rank&&t.layout.dtype.size()<=l.max_string_bytes&&t.layout.native_layout.size()<=l.max_string_bytes,"tensor metadata exceeds bound");}
}
void encode(Encoder& e,const SnapshotDescriptor& s){
    key(e,s.key);cut(e,s.cut);e.text(s.native_layout);
    e.u64(s.accepted_tokens.size());for(auto n:s.accepted_tokens)e.u32(uint32_t(n));
    e.u64(s.pending_tokens.size());for(auto n:s.pending_tokens)e.u32(uint32_t(n));
    e.u32(s.speculation_disabled?1:0);e.u32(s.steering_enabled?1:0);
    e.u32(uint32_t(s.continuation.mode));e.text(s.continuation.adapter);e.u64(s.continuation.payload.size());e.bytes(s.continuation.payload.data(),s.continuation.payload.size());
    e.u64(s.approximation.size());for(const auto& a:s.approximation){e.u64(a.begin);e.u64(a.end);e.u64(a.first_unresolved_unix_ms);e.digest(a.ancestor);}
    e.u64(s.devices.size());for(const auto& d:s.devices){e.u32(d.id);e.u64(uint64_t(d.layer_begin));e.u64(uint64_t(d.layer_end));cut(e,d.cut);e.u64(d.tensors.size());for(const auto& t:d.tensors){layout(e,t.layout);e.u64(t.payload.size());t.payload.visit(0,t.payload.size(),[&](const uint8_t* p,size_t n,size_t){e.bytes(p,n);return true;});}}
}
constexpr char magic[8]={'S','T','R','A','W','V','2','\0'};
#ifndef _WIN32
struct Decoder {
    int fd;uint64_t remaining;const WindowDiskLimits& limits;const WindowDiskOptions& options;WindowHasher hash;
    void bytes(void* p,size_t n){require(n<=remaining,"field exceeds file extent");auto* out=static_cast<uint8_t*>(p);size_t left=n;while(left){cancelled(options);const auto count=size_t(std::min<uint64_t>(left,limits.io_block_bytes));read_all(fd,out,count);hash.update(out,count);out+=count;left-=count;}remaining-=n;}
    uint64_t u64(){uint8_t b[8];bytes(b,8);uint64_t n=0;for(int i=0;i<8;++i)n|=uint64_t(b[i])<<(8*i);return n;}
    uint32_t u32(){uint8_t b[4];bytes(b,4);uint32_t n=0;for(int i=0;i<4;++i)n|=uint32_t(b[i])<<(8*i);return n;}
    uint64_t count(uint64_t cap,uint64_t minimum){const auto n=u64();require(n<=cap&&n<=SIZE_MAX&&(!minimum||n<=remaining/minimum),"bounded count exceeds limit/file");return n;}
    std::string text(){const auto n=count(limits.max_string_bytes,1);std::string s(size_t(n),'\0');bytes(s.data(),s.size());return s;}
    StateDigest digest(){StateDigest d;bytes(d.data(),d.size());return d;}
};
WindowKey key(Decoder& d){WindowKey k;k.engine_generation=d.u64();k.window_generation=d.u64();k.session_id=d.text();k.thread_id=d.text();k.window_id=d.text();k.revision=d.u64();k.append_epoch=d.u64();k.selection_epoch=d.u64();k.provenance_epoch=d.u64();k.model_profile=d.digest();k.topology=d.digest();return k;}
StateCut cut(Decoder& d){StateCut c;c.id=d.u64();c.accepted=d.u64();c.forwarded=d.u64();c.exact_frontier=d.u64();c.output_watermark=d.u64();c.history=d.digest();return c;}
TensorLayout layout(Decoder& d){TensorLayout t;const auto kind=d.u32();require(kind<=uint32_t(StatePartKind::ple_current_token),"unknown tensor kind");t.kind=StatePartKind(kind);const auto layer=d.u64();require(layer<=uint64_t(INT64_MAX)||layer==UINT64_MAX,"invalid layer ordinal");t.layer=layer==UINT64_MAX?-1:int64_t(layer);t.device_part=d.u32();t.dtype=d.text();t.native_layout=d.text();const auto rank=d.count(d.limits.max_rank,8);t.shape.resize(size_t(rank));for(auto& n:t.shape)n=d.u64();const auto strides=d.count(d.limits.max_rank,8);t.strides.resize(size_t(strides));for(auto& n:t.strides)n=d.u64();t.logical_begin=d.u64();t.valid_elements=d.u64();t.bytes=d.u64();return t;}
SnapshotDescriptor decode(Decoder& d,const WindowKey& expected){
    SnapshotDescriptor s;s.key=key(d);require(s.key==expected,"window generation/profile/topology identity mismatch");s.cut=cut(d);s.native_layout=d.text();
    const auto accepted=d.count(d.limits.max_tokens,4);s.accepted_tokens.resize(size_t(accepted));for(auto& n:s.accepted_tokens){const auto v=d.u32();require(v<=INT32_MAX,"invalid token");n=int32_t(v);}
    const auto pending=d.count(d.limits.max_tokens,4);s.pending_tokens.resize(size_t(pending));for(auto& n:s.pending_tokens){const auto v=d.u32();require(v<=INT32_MAX,"invalid token");n=int32_t(v);}
    const auto speculation=d.u32(),steering=d.u32();require(speculation<=1&&steering<=1,"invalid feature flag");s.speculation_disabled=speculation;s.steering_enabled=steering;
    const auto mode=d.u32();require(mode<=uint32_t(ContinuationMode::exact_continuation),"unknown continuation");s.continuation.mode=ContinuationMode(mode);s.continuation.adapter=d.text();const auto continuation=d.count(d.limits.max_file_bytes,1);s.continuation.payload.resize(size_t(continuation));d.bytes(s.continuation.payload.data(),s.continuation.payload.size());
    const auto regions=d.count(d.limits.max_regions,56);s.approximation.resize(size_t(regions));for(auto& a:s.approximation){a.begin=d.u64();a.end=d.u64();a.first_unresolved_unix_ms=d.u64();a.ancestor=d.digest();}
    const auto devices=d.count(d.limits.max_stages,100);s.devices.resize(size_t(devices));uint64_t tensors=0;
    for(auto& stage:s.devices){stage.id=d.u32();const auto lo=d.u64(),hi=d.u64();require(lo<=INT64_MAX&&hi<=INT64_MAX,"invalid stage range");stage.layer_begin=int64_t(lo);stage.layer_end=int64_t(hi);stage.cut=cut(d);const auto n=d.count(d.limits.max_tensors,64);tensors=sum(tensors,n);require(tensors<=d.limits.max_tensors,"total tensor count exceeds bound");stage.tensors.resize(size_t(n));for(auto& t:stage.tensors){t.layout=layout(d);const auto bytes=d.count(d.limits.max_file_bytes,1);require(bytes==t.layout.bytes,"RAW tensor extent mismatch");t.payload.resize(size_t(bytes));t.payload.visit(0,size_t(bytes),[&](uint8_t* p,size_t size,size_t){d.bytes(p,size);return true;});}}
    require(d.remaining==0,"trailing window bytes");return s;
}
#endif
} // namespace
#ifndef _WIN32
struct HostAllowance {std::atomic<uint64_t> used{0};uint64_t limit;explicit HostAllowance(uint64_t n):limit(n){}};
struct DiskReservation {
    std::shared_ptr<HostAllowance> allowance;
    std::shared_ptr<const ReservationLease> budget;
    uint64_t host=0;
    ~DiskReservation(){if(host)allowance->used.fetch_sub(host);}
};
struct DiskFile {std::shared_ptr<const DiskReservation> charge;uint64_t bytes=0;};
struct WindowDiskState {
    Fd directory,lock;WindowDiskLimits limits;WindowBudget budget;std::mutex mutex;
    std::shared_ptr<HostAllowance> host;
    std::shared_ptr<const DiskReservation> metadata;
    bool startup_reconciled=false;
    std::map<std::string,std::shared_ptr<DiskFile>> files;
    WindowDiskState(WindowDiskLimits l,WindowBudget b):limits(l),budget(std::move(b)),host(std::make_shared<HostAllowance>(l.max_host_bytes)){}
};
namespace {
struct ArtifactHold {std::shared_ptr<WindowDiskState> state;std::shared_ptr<DiskFile> file;std::shared_ptr<const DiskReservation> metadata;};
std::string art_name(const std::string& id){return "artifact-"+id+".raw2";}
std::string op_name(const std::string& id,bool done=false){return "operation-"+id+(done?".done":".intent");}
std::string revision_hex(uint64_t n){std::string s(16,'0');const char* h="0123456789abcdef";for(int i=15;i>=0;--i){s[size_t(i)]=h[n&15];n>>=4;}return s;}
std::string cat_name(const std::string& id,uint64_t revision){return "catalog-"+id+"-"+revision_hex(revision)+".cat2";}
StateDigest digest_bytes(const void* p,size_t n){WindowHasher h;h.update(p,n);return h.digest();}
std::string temporary(const std::string& target){return "temporary-"+hex(digest_bytes(target.data(),target.size()))+".tmp2";}
bool named(const std::string& n,const std::string& prefix,const std::string& suffix){return n.size()==prefix.size()+64+suffix.size()&&n.starts_with(prefix)&&n.ends_with(suffix)&&opaque(n.substr(prefix.size(),64));}
bool filename(const std::string& n){
    if(n==".manager-lock")return true;
    if(named(n,"artifact-",".raw2")||named(n,"operation-",".intent")||named(n,"operation-",".done")||named(n,"temporary-",".tmp2")||named(n,"retired-",".mark2"))return true;
    return n.size()==94&&n.starts_with("catalog-")&&opaque(n.substr(8,64))&&n[72]=='-'&&n.ends_with(".cat2")&&
        std::all_of(n.begin()+73,n.begin()+89,[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
WindowResources charge_vector(WindowDiskState& s,uint64_t host,uint64_t disk,uint64_t io=0){WindowResources r;r.device_bytes.resize(s.budget.snapshot().capacity.device_bytes.size());r.host_bytes=host;r.disk_bytes=disk;r.io_slots=io;return r;}
std::shared_ptr<const DiskReservation> reserve(WindowDiskState& s,uint64_t host,uint64_t disk,uint64_t io=0){
    auto lease=std::make_shared<DiskReservation>();lease->allowance=s.host;
    auto used=s.host->used.load();
    do {if(host>s.host->limit||used>s.host->limit-host)fail(WindowDiskError::budget_refused,"store retained plus transient host allowance refused");}
    while(!s.host->used.compare_exchange_weak(used,used+host));
    lease->host=host;
    auto r=s.budget.reserve(charge_vector(s,host,disk,io),ReservationPurpose::foreground);
    if(!r)fail(WindowDiskError::budget_refused,"disk/host/IO reservation refused");
    lease->budget=std::move(r.lease);return lease;
}
// Reserved envelope, not allocator-resident measurement. Each scan entry covers
// stat/map node, bounded <=94-byte filename and linked-temp vector copies.
uint64_t operation_host(const WindowDiskState& s){return sum(product(s.limits.max_entries,1024),sum(s.limits.io_block_bytes,8192));}
std::shared_ptr<const DiskReservation> admit_io(WindowDiskState& s){return reserve(s,operation_host(s),0,1);}

void account(WindowDiskState& s,const std::string& name,uint64_t bytes){
    require(bytes<=s.limits.max_file_bytes,"cache object exceeds configured file bound");
    const auto old=s.files.find(name);if(old!=s.files.end()){require(bytes<=old->second->bytes,"cache object changed outside manager");return;}
    if(s.files.size()>=s.limits.max_entries)fail(WindowDiskError::budget_refused,"cache entry bound reached");
    auto file=std::make_shared<DiskFile>();file->charge=reserve(s,0,bytes);file->bytes=bytes;s.files.emplace(name,std::move(file));
}
void remove_temp(WindowDiskState&,const std::string&,const WindowDiskOptions&);
void flush_dir(WindowDiskState&,const WindowDiskOptions&,const char*);
WindowDiskReceipt read_record(WindowDiskState&,const std::string&);
void reconcile(WindowDiskState& s,const WindowDiskOptions& options={}){
    Fd copied(::dup(s.directory.n));if(copied.n<0)fail(WindowDiskError::io,"directory duplication failed");
    if(::lseek(copied.n,0,SEEK_SET)<0)fail(WindowDiskError::io,"directory rewind failed");
    DIR* dir=::fdopendir(copied.n);if(!dir)fail(WindowDiskError::io,"directory scan failed");copied.n=-1;
    std::map<std::string,struct stat> seen;
    try {while(true){
        errno=0;dirent* entry=nullptr;
        const int injected=options.fault?options.fault("readdir"):0;
        if(injected)errno=injected;else {errno=0;entry=::readdir(dir);}
        if(!entry){if(errno)fail(WindowDiskError::io,"cache directory read failed");break;}
        const std::string name=entry->d_name;if(name=="."||name=="..")continue;
        require(filename(name),"unexpected cache namespace entry");struct stat st{};
        if(::fstatat(s.directory.n,name.c_str(),&st,AT_SYMLINK_NOFOLLOW)!=0)fail(WindowDiskError::io,"cache stat failed");
        require(S_ISREG(st.st_mode)&&st.st_nlink>=1&&st.st_nlink<=2&&st.st_size>=0,"cache entry is linked or not a regular file");
        if(name==".manager-lock"){require(st.st_nlink==1&&st.st_size==0,"invalid manager lock file");continue;}
        require(seen.size()<s.limits.max_entries,"cache entry bound exceeded");seen.emplace(name,st);
    }}catch(...){::closedir(dir);throw;}::closedir(dir);
    // A crash can fall between linkat publication and temporary unlink. Accept
    // only the exact deterministic temp/final pair, never an arbitrary hardlink.
    std::vector<std::string> linked_temps;
    for(const auto& [name,st]:seen)if(st.st_nlink==2){
        const auto pair=std::find_if(seen.begin(),seen.end(),[&](const auto& other){return other.first!=name&&other.second.st_dev==st.st_dev&&other.second.st_ino==st.st_ino;});
        require(pair!=seen.end(),"unowned cache hardlink");
        bool owned=name==temporary(pair->first)||pair->first==temporary(name);
        const auto& final=name.starts_with("artifact-")?name:pair->first;
        const auto& temp=name.starts_with("temporary-")?name:pair->first;
        if(!owned&&final.starts_with("artifact-"))for(const auto& [intent,info]:seen){
            (void)info;if(!intent.starts_with("operation-")||!intent.ends_with(".intent"))continue;
            const auto r=read_record(s,intent);
            if(intent==op_name(r.operation)&&art_name(r.artifact)==final&&temporary(final+"-"+r.operation)==temp){owned=true;break;}
        }
        require(owned,"hardlink is not owned by the publishing operation");
        if(name.starts_with("temporary-"))linked_temps.push_back(name);
    }
    for(const auto& name:linked_temps){
        const auto& stat=seen.at(name);
        const auto final=std::find_if(seen.begin(),seen.end(),[&](const auto& other){return other.first!=name&&other.second.st_dev==stat.st_dev&&other.second.st_ino==stat.st_ino;});
        if(!s.files.contains(final->first)){
            if(s.files.contains(name)){
                // Preserve the SAME disk reservation across publication recovery.
                std::string key=final->first;auto node=s.files.extract(name);node.key()=std::move(key);s.files.insert(std::move(node));
            }else account(s,final->first,uint64_t(stat.st_size));
        }
        remove_temp(s,name,options);seen.erase(name);
    }
    for(const auto& [name,st]:seen)account(s,name,uint64_t(st.st_size));
    // A failed deletion-directory flush may have left an absent but still
    // charged artifact. Confirm directory durability before releasing that charge.
    const bool missing=std::any_of(s.files.begin(),s.files.end(),[&](const auto& file){return !seen.contains(file.first);});
    if(missing||!s.startup_reconciled)flush_dir(s,options,"reconcile_delete_dir_flush");
    s.startup_reconciled=true;
    for(auto it=s.files.begin();it!=s.files.end();)it=seen.contains(it->first)?std::next(it):s.files.erase(it);
}
Fd read_file(WindowDiskState& s,const std::string& name){
    Fd fd(::openat(s.directory.n,name.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK));if(fd.n<0)fail(WindowDiskError::io,"cache object open failed");
    struct stat st{};if(::fstat(fd.n,&st)!=0)fail(WindowDiskError::io,"cache object stat failed");require(S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size>=0&&uint64_t(st.st_size)<=s.limits.max_file_bytes,"unsafe cache object");return fd;
}
void remove_temp(WindowDiskState& s,const std::string& name,const WindowDiskOptions& options){
    checkpoint(options,"temporary_unlink");
    if(::unlinkat(s.directory.n,name.c_str(),0)!=0&&errno!=ENOENT)fail(WindowDiskError::io,"temporary removal failed");
    flush_dir(s,options,"temporary_delete_dir_flush");
    s.files.erase(name); // never release an ambiguously deleted temp's charge
}
Fd create_temp(WindowDiskState& s,const std::string& name,uint64_t size,const WindowDiskOptions& options){
    // Manager lock and deterministic target-owned name fence restart cleanup.
    if(s.files.contains(name))remove_temp(s,name,options);
    // linkat temporarily exposes both names. Reserve one namespace entry so a
    // crash at that cut remains within the restart scan bound.
    if(s.files.size()>=s.limits.max_entries-1)fail(WindowDiskError::budget_refused,"atomic publication requires one spare namespace entry");
    account(s,name,size);Fd fd(::openat(s.directory.n,name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));
    if(fd.n<0){s.files.erase(name);fail(WindowDiskError::io,"exclusive temporary creation failed");}return fd;
}
void flush_file(int fd,const WindowDiskOptions& o,const char* phase){checkpoint(o,phase);if(::fsync(fd)!=0)fail(WindowDiskError::io,phase);}
void flush_dir(WindowDiskState& s,const WindowDiskOptions& o,const char* phase){checkpoint(o,phase);if(::fsync(s.directory.n)!=0)fail(WindowDiskError::io,phase);}
void publish(WindowDiskState& s,const std::string& temp,const std::string& final,const WindowDiskOptions& o,const char* phase,bool* linked=nullptr){
    std::string final_key=final; // prepare map key before the publication cut
    checkpoint(o,phase);
    // linkat is atomic and refuses an existing immutable destination. Unlink the
    // operation-owned temporary only after publication. Never replace old good.
    if(::linkat(s.directory.n,temp.c_str(),s.directory.n,final.c_str(),0)!=0)fail(errno==EEXIST?WindowDiskError::conflict:WindowDiskError::io,phase);
    if(linked)*linked=true; // Record the effect before callbacks or later bookkeeping can fail.
    // No allocation after link: prebuilt key is moved into the map node.
    checkpoint(o,"after_link");
    auto node=s.files.extract(temp);
    node.key()=std::move(final_key);s.files.insert(std::move(node));
    if(::unlinkat(s.directory.n,temp.c_str(),0)!=0)fail(WindowDiskError::io,"published temporary unlink failed");
}
std::vector<uint8_t> record(const WindowDiskReceipt& r){
    const std::string body="STRATA-RAW2-RECEIPT\n"+r.operation+"\n"+r.artifact+"\n"+r.catalog+"\n"+hex(r.fingerprint)+"\n"+revision_hex(r.catalog_revision)+"\n";
    const auto text=body+hex(digest_bytes(body.data(),body.size()))+"\n";return {text.begin(),text.end()};
}
uint8_t nibble(char c){if(c>='0'&&c<='9')return uint8_t(c-'0');require(c>='a'&&c<='f',"bad hex record");return uint8_t(c-'a'+10);}
StateDigest unhex(const std::string& s){require(opaque(s),"bad digest record");StateDigest d{};for(size_t i=0;i<32;++i)d[i]=uint8_t(nibble(s[2*i])*16+nibble(s[2*i+1]));return d;}
WindowDiskReceipt read_record(WindowDiskState& s,const std::string& name){
    auto fd=read_file(s,name);struct stat st{};if(::fstat(fd.n,&st)!=0)fail(WindowDiskError::io,"cache object stat failed");require(st.st_size>0&&st.st_size<=1024,"record length invalid");std::string text(size_t(st.st_size),'\0');read_all(fd.n,text.data(),text.size());
    std::vector<std::string> fields;size_t at=0;while(at<text.size()){auto end=text.find('\n',at);require(end!=std::string::npos,"incomplete receipt");fields.push_back(text.substr(at,end-at));at=end+1;}
    require(fields.size()==7&&fields[0]=="STRATA-RAW2-RECEIPT"&&opaque(fields[1])&&opaque(fields[2])&&opaque(fields[3])&&fields[5].size()==16,"invalid receipt fields");
    const auto body=text.substr(0,text.size()-65);require(hex(digest_bytes(body.data(),body.size()))==fields[6],"receipt checksum mismatch");
    WindowDiskReceipt r;r.operation=fields[1];r.artifact=fields[2];r.catalog=fields[3];r.fingerprint=unhex(fields[4]);for(char c:fields[5])r.catalog_revision=(r.catalog_revision<<4)|nibble(c);return r;
}
bool equal_record(const WindowDiskReceipt& a,const WindowDiskReceipt& b){return a.operation==b.operation&&a.artifact==b.artifact&&a.catalog==b.catalog&&a.fingerprint==b.fingerprint&&a.catalog_revision==b.catalog_revision;}
void write_record(WindowDiskState& s,const std::string& name,const WindowDiskReceipt& r,const WindowDiskOptions& o,const char* type,bool* linked=nullptr){
    if(s.files.contains(name)){
        require(equal_record(read_record(s,name),r),"operation/catalog fingerprint conflict");
        auto existing=read_file(s,name);const auto file_phase=std::string(type)+"_file_flush",dir_phase=std::string(type)+"_dir_flush";
        flush_file(existing.n,o,file_phase.c_str());flush_dir(s,o,dir_phase.c_str());return;
    }
    const auto bytes=record(r);const auto temp=temporary(name);auto fd=create_temp(s,temp,bytes.size(),o);
    write_all(fd.n,bytes.data(),bytes.size(),s.limits,o);const std::string file_phase=std::string(type)+"_file_flush",publish_phase=std::string(type)+"_publish",after_phase=std::string("after_")+type+"_publish",dir_phase=std::string(type)+"_dir_flush";
    flush_file(fd.n,o,file_phase.c_str());publish(s,temp,name,o,publish_phase.c_str(),linked);checkpoint(o,after_phase.c_str());flush_dir(s,o,dir_phase.c_str());
}
std::string operation_id(const WindowDiskOperation& o){require(o.engine_generation&&o.window_generation,"operation generation is zero");WindowHasher h;constexpr char domain[]="strata-window-disk-operation-v2";h.update(domain,sizeof(domain));Encoder e{[&](const void* p,size_t n){h.update(p,n);}};e.u64(o.engine_generation);e.u64(o.window_generation);e.digest(o.nonce);return hex(h.digest());}
StateDigest fingerprint(const std::string& artifact,const std::string& catalog,uint64_t revision){WindowHasher h;Encoder e{[&](const void* p,size_t n){h.update(p,n);}};e.text("strata-window-save-input-v2");e.text(artifact);e.text(catalog);e.u64(revision);return h.digest();}
struct Header {uint64_t body=0;StateDigest digest{};};
std::array<uint8_t,56> header(const Header& h){std::array<uint8_t,56> b{};size_t at=0;Encoder e{[&](const void* p,size_t n){std::memcpy(b.data()+at,p,n);at+=n;}};e.bytes(magic,8);e.u32(2);e.u32(0);e.u64(h.body);e.digest(h.digest);return b;}
Header read_header(int fd,uint64_t size){std::array<uint8_t,56> b{};require(size>=b.size(),"truncated header");read_all(fd,b.data(),b.size());require(std::memcmp(b.data(),magic,8)==0&&b[8]==2&&b[9]==0&&b[10]==0&&b[11]==0&&b[12]==0&&b[13]==0&&b[14]==0&&b[15]==0,"unsupported RAW-v2 schema/codec");Header h;for(int i=0;i<8;++i)h.body|=uint64_t(b[size_t(16+i)])<<(8*i);require(h.body==size-56,"file/header extent mismatch");std::copy(b.begin()+24,b.end(),h.digest.begin());return h;}
Header verify_artifact(WindowDiskState& s,const std::string& id,const WindowDiskOptions& o){
    auto fd=read_file(s,art_name(id));struct stat st{};if(::fstat(fd.n,&st)!=0)fail(WindowDiskError::io,"cache object stat failed");const auto h=read_header(fd.n,uint64_t(st.st_size));require(hex(h.digest)==id,"artifact identity mismatch");
    WindowHasher hash;std::vector<uint8_t> buffer(size_t(std::min<uint64_t>(s.limits.io_block_bytes,h.body)));uint64_t left=h.body;
    while(left){cancelled(o);const auto n=size_t(std::min<uint64_t>(left,buffer.size()));read_all(fd.n,buffer.data(),n);hash.update(buffer.data(),n);left-=n;}
    require(hash.digest()==h.digest,"artifact checksum mismatch");return h;
}
WindowDiskReceipt latest(WindowDiskState& s,const std::string& id){WindowDiskReceipt out;out.catalog=id;const auto prefix="catalog-"+id+"-";for(const auto& [name,file]:s.files){(void)file;if(name.starts_with(prefix)&&name.ends_with(".cat2")){auto r=read_record(s,name);require(r.catalog==id&&name==cat_name(id,r.catalog_revision),"catalog name/record mismatch");if(r.catalog_revision>out.catalog_revision)out=std::move(r);}}return out;}
WindowDiskReceipt query_locked(WindowDiskState& s,const std::string& id,const WindowDiskOptions& o){
    WindowDiskReceipt r;r.operation=id;if(!s.files.contains(op_name(id)))return r;
    r=read_record(s,op_name(id));require(r.operation==id,"operation record identity mismatch");
    if(!s.files.contains(art_name(r.artifact))){if(s.files.contains("retired-"+r.artifact+".mark2")){const auto retired=read_record(s,"retired-"+r.artifact+".mark2");require(retired.artifact==r.artifact,"retirement marker identity mismatch");r.phase=WindowDiskPhase::retired_artifact;}return r;}
    r.phase=WindowDiskPhase::published_durability_unknown;verify_artifact(s,r.artifact,o);
    auto fd=read_file(s,art_name(r.artifact));flush_file(fd.n,o,"recover_file_flush");flush_dir(s,o,"recover_dir_flush");r.phase=WindowDiskPhase::durable_artifact;
    const auto cat=cat_name(r.catalog,r.catalog_revision);
    if(s.files.contains(cat)&&equal_record(read_record(s,cat),r)){r.phase=WindowDiskPhase::catalog_linked;}
    if(s.files.contains(op_name(id,true)))require(r.phase==WindowDiskPhase::catalog_linked&&equal_record(read_record(s,op_name(id,true)),r),"receipt lacks matching catalog publication");
    return r;
}
WindowDiskResult result(const Failure& f,WindowDiskReceipt receipt={}){return {f.code,f.text,std::move(receipt),f.system_error};}
} // namespace
WindowDiskResult WindowDiskStore::open(const std::string& root,const WindowDiskLimits& limits,WindowBudget budget,std::shared_ptr<WindowDiskStore>& out){
    try {limits_ok(limits);auto state=std::make_shared<WindowDiskState>(limits,std::move(budget));state->metadata=reserve(*state,sum(product(limits.max_entries,1024),4096),0);auto io=admit_io(*state);state->directory.n=::open(root.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);if(state->directory.n<0)fail(WindowDiskError::io,"cache root open failed");
        state->lock.n=::openat(state->directory.n,".manager-lock",O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK,0600);if(state->lock.n<0||::flock(state->lock.n,LOCK_EX|LOCK_NB)!=0)fail(WindowDiskError::conflict,"cache already has a manager");
        reconcile(*state);
        out=std::shared_ptr<WindowDiskStore>(new WindowDiskStore(std::move(state)));return {};
    }catch(const Failure& f){return result(f);}catch(const std::bad_alloc&){return {WindowDiskError::budget_refused,"cache metadata allocation failed",{}};}
}
WindowDiskHostUsage WindowDiskStore::host_usage() const {return {state_->host->used.load(),state_->host->limit};}
WindowDiskResult WindowDiskStore::save(const WindowDiskOperation& operation,const BundleHandle& input,const WindowKey& expected,
    const WindowStateLayout& resolved,const StateDigest& catalog_id,uint64_t expected_revision,WindowDiskArtifact& out,const WindowDiskOptions& options){
    // Input retirement can invoke a caller-owned deleter. Declare the retained
    // source before the lock so failure unwinding also retires it after unlock.
    BundleHandle source=input;
    std::unique_lock lock(state_->mutex);auto& s=*state_;WindowDiskReceipt r,history;bool have_history=false,publishing_catalog=false,artifact_linked=false,catalog_linked=false,intent_bound=false;std::shared_ptr<const DiskReservation> io;
    std::string artifact_name,catalog_name;
    auto preserve_history=[&]{if(have_history&&(!equal_record(history,r)||int(history.phase)>int(r.phase)))r=std::move(history);};
    // Recovery cannot allocate filenames while handling bad_alloc. ENOENT alone
    // proves absence; any other failed probe leaves the publication unknown.
    enum class Presence { absent, present, unknown };
    auto probe=[&](const std::string& name) noexcept {
        struct stat st{};int rc;
        do {rc=::fstatat(s.directory.n,name.c_str(),&st,AT_SYMLINK_NOFOLLOW);}while(rc<0&&errno==EINTR);
        return rc==0?Presence::present:(errno==ENOENT?Presence::absent:Presence::unknown);
    };
    auto recover_publication=[&] {
        if((intent_bound||artifact_linked)&&!artifact_name.empty()&&int(r.phase)<int(WindowDiskPhase::durable_artifact)) {
            const auto seen=probe(artifact_name);
            if(artifact_linked||seen==Presence::present)r.phase=WindowDiskPhase::published_durability_unknown;
            else if(seen==Presence::unknown&&r.phase==WindowDiskPhase::not_published)r.phase=WindowDiskPhase::unknown;
        }
        if(publishing_catalog&&r.phase!=WindowDiskPhase::catalog_linked) {
            const auto seen=probe(catalog_name);
            if(catalog_linked||seen!=Presence::absent)r.phase=WindowDiskPhase::catalog_durability_unknown;
        }
        preserve_history();
    };
    try {r.phase=WindowDiskPhase::unknown;r.operation=operation_id(operation);io=admit_io(s);reconcile(s,options);
        if(s.files.contains(op_name(r.operation))){r=query_locked(s,r.operation,options);history=r;have_history=true;}else r.phase=WindowDiskPhase::not_published;
        require(bool(source)&&operation.engine_generation==expected.engine_generation&&operation.window_generation==expected.window_generation,"save operation generation/source mismatch");
        const auto checked=validate_window_state(*source.get(),expected,resolved);require(bool(checked),checked.detail.c_str());bounds(*source.get(),s.limits);
        require(expected_revision!=UINT64_MAX,"catalog revision exhausted");r.phase=WindowDiskPhase::not_published;r.operation=operation_id(operation);r.catalog=hex(catalog_id);r.catalog_revision=expected_revision+1;
        WindowHasher hash;Encoder sizing{[&](const void* p,size_t n){
            require(sizing.count<=s.limits.max_file_bytes-56,"encoded artifact exceeds file bound");
            const auto* data=static_cast<const uint8_t*>(p);while(n){cancelled(options);const auto part=size_t(std::min<uint64_t>(n,s.limits.io_block_bytes));hash.update(data,part);data+=part;n-=part;}
        }};encode(sizing,*source.get());Header h{sizing.count,hash.digest()};
        require(sum(h.body,56)<=s.limits.max_file_bytes,"encoded artifact exceeds file bound");r.artifact=hex(h.digest);r.fingerprint=fingerprint(r.artifact,r.catalog,expected_revision);
        if(s.files.contains(op_name(r.operation))){const auto prior=read_record(s,op_name(r.operation));if(!equal_record(prior,r)){r=query_locked(s,r.operation,options);fail(WindowDiskError::conflict,"operation id reused with different input fingerprint");}intent_bound=true;}
        if(s.files.contains("retired-"+r.artifact+".mark2"))fail(WindowDiskError::conflict,"artifact explicitly retired; operation cannot recreate it");
        // Admit and construct the returned owner before the first save publication.
        // A quota refusal must not persist an intent/artifact/catalog as a side effect.
        auto handle_charge=reserve(s,512,0);
        auto returned_owner=std::make_shared<ArtifactHold>(ArtifactHold{state_,{},std::move(handle_charge)});
        WindowDiskArtifact handle;handle.id_=r.artifact;handle.owner_=returned_owner;
        artifact_name=art_name(r.artifact);catalog_name=cat_name(r.catalog,r.catalog_revision);
        // Existing artifact content alone cannot bind this operation. A successful
        // own intent link (recorded before callbacks), or verified matching prior
        // intent above, is the publication prerequisite for recovery probes.
        cancelled(options);write_record(s,op_name(r.operation),r,options,"intent",&intent_bound);
        if(s.files.contains(art_name(r.artifact))){r.phase=WindowDiskPhase::published_durability_unknown;verify_artifact(s,r.artifact,options);}
        else {
            const auto name=art_name(r.artifact),temp=temporary(name+"-"+r.operation);auto fd=create_temp(s,temp,h.body+56,options);const auto bytes=header(h);write_all(fd.n,bytes.data(),bytes.size(),s.limits,options);
            Encoder writer{[&](const void* p,size_t n){write_all(fd.n,p,n,s.limits,options);}};encode(writer,*source.get());require(writer.count==h.body,"immutable source changed during save");
            flush_file(fd.n,options,"artifact_file_flush");cancelled(options);publish(s,temp,name,options,"artifact_publish",&artifact_linked);r.phase=WindowDiskPhase::published_durability_unknown;checkpoint(options,"after_artifact_publish");
        }
        // Also reconfirm a previously published/durability-ambiguous artifact.
        {auto fd=read_file(s,art_name(r.artifact));flush_file(fd.n,options,"artifact_recover_flush");}
        flush_dir(s,options,"artifact_dir_flush");r.phase=WindowDiskPhase::durable_artifact;cancelled(options);
        const auto cat=cat_name(r.catalog,r.catalog_revision);
        if(s.files.contains(cat)){
            if(!equal_record(read_record(s,cat),r))fail(WindowDiskError::conflict,"catalog compare-and-swap conflict");
            publishing_catalog=true;flush_dir(s,options,"catalog_dir_flush");
        }else {
            if(latest(s,r.catalog).catalog_revision!=expected_revision)fail(WindowDiskError::conflict,"catalog compare-and-swap conflict");
            publishing_catalog=true;write_record(s,cat,r,options,"catalog",&catalog_linked);
        }
        r.phase=WindowDiskPhase::catalog_linked;write_record(s,op_name(r.operation,true),r,options,"receipt");
        returned_owner->file=s.files.at(artifact_name);
        // Complete all throwing receipt copies before replacing caller output.
        WindowDiskResult success{WindowDiskError::none,{},r};
        static_assert(std::is_nothrow_move_constructible_v<WindowDiskResult>);
        static_assert(std::is_nothrow_move_assignable_v<WindowDiskArtifact>);
        io.reset();lock.unlock(); // prior output may retire another manager/resources
        out=std::move(handle);return success;
    }catch(const Failure& f){
        recover_publication();return result(f,r);
    }catch(const std::bad_alloc&){recover_publication();return {WindowDiskError::budget_refused,"save allocation failed; query operation before retry",r};}
}
WindowDiskResult WindowDiskStore::query(const WindowDiskOperation& operation,const WindowDiskOptions& options){
    std::lock_guard lock(state_->mutex);auto& s=*state_;WindowDiskReceipt r;
    try {r.phase=WindowDiskPhase::unknown;r.operation=operation_id(operation);auto io=admit_io(s);reconcile(s,options);
        if(s.files.contains(op_name(r.operation))){r=read_record(s,op_name(r.operation));r.phase=WindowDiskPhase::unknown;if(s.files.contains(art_name(r.artifact)))r.phase=WindowDiskPhase::published_durability_unknown;}
        r=query_locked(s,r.operation,options);return {WindowDiskError::none,{},r};
    }catch(const Failure& f){return result(f,r);}catch(const std::bad_alloc&){return {WindowDiskError::budget_refused,"query metadata allocation failed",r};}
}
WindowDiskResult WindowDiskStore::catalog(const StateDigest& id,WindowDiskReceipt& out){
    std::lock_guard lock(state_->mutex);WindowDiskReceipt r;r.phase=WindowDiskPhase::unknown;
    try{auto io=admit_io(*state_);reconcile(*state_);r=latest(*state_,hex(id));
        if(!r.artifact.empty()){r.phase=WindowDiskPhase::published_durability_unknown;auto recovered=query_locked(*state_,r.operation,{});require(recovered.phase==WindowDiskPhase::catalog_linked&&equal_record(recovered,r),"catalog has no matching operation receipt");r=std::move(recovered);}
        WindowDiskReceipt candidate=r;
        WindowDiskResult success{WindowDiskError::none,{},r};
        static_assert(std::is_nothrow_move_assignable_v<WindowDiskReceipt>);
        static_assert(std::is_nothrow_move_constructible_v<WindowDiskResult>);
        out=std::move(candidate);return success;
    }catch(const Failure& f){return result(f,r);}catch(const std::bad_alloc&){return {WindowDiskError::budget_refused,"catalog allocation failed",r};}
}
WindowDiskResult WindowDiskStore::acquire(const std::string& id,WindowDiskArtifact& out){std::unique_lock lock(state_->mutex);try{require(opaque(id),"invalid artifact id");auto io=admit_io(*state_);reconcile(*state_);const auto it=state_->files.find(art_name(id));require(it!=state_->files.end(),"artifact absent");auto fd=read_file(*state_,it->first);WindowDiskArtifact handle;handle.id_=id;handle.owner_=std::make_shared<ArtifactHold>(ArtifactHold{state_,it->second,reserve(*state_,512,0)});lock.unlock();out=std::move(handle);return {};}catch(const Failure& f){return result(f);}catch(const std::bad_alloc&){return {WindowDiskError::budget_refused,"artifact handle allocation failed",{}};}}
WindowDiskResult WindowDiskStore::load(const WindowDiskArtifact& artifact,const WindowKey& expected,const WindowStateLayout& resolved,BundleHandle& out,const WindowDiskOptions& options){
    std::unique_lock lock(state_->mutex);auto& s=*state_;
    try {require(bool(artifact)&&opaque(artifact.id()),"missing artifact lease");auto hold=std::static_pointer_cast<const ArtifactHold>(artifact.owner_);require(hold->state==state_,"artifact belongs to another manager");
        auto io=admit_io(s);auto fd=read_file(s,art_name(artifact.id()));struct stat st{};if(::fstat(fd.n,&st)!=0)fail(WindowDiskError::io,"cache object stat failed");const uint64_t size=uint64_t(st.st_size);
        // Conservative host peak for decoded metadata, segmented mutable payload,
        // immutable deep copy and codec workspace. Bound before any payload allocation.
        const auto host=sum(product(size,8),sum(product(s.limits.max_tensors,1024),s.limits.io_block_bytes));require(host<=s.limits.max_host_bytes,"load host peak exceeds configured bound");auto reservation=reserve(s,host,0);
        const auto h=read_header(fd.n,size);require(hex(h.digest)==artifact.id(),"artifact/header identity mismatch");Decoder decoder{fd.n,h.body,s.limits,options,{}};
        auto decoded=decode(decoder,expected);require(decoder.hash.digest()==h.digest,"RAW payload checksum mismatch");
        BundleHandle candidate;
        const auto checked=BundleHandle::freeze(decoded,expected,resolved,candidate,reservation);require(bool(checked),checked.detail.c_str());
        io.reset();lock.unlock();
        out=std::move(candidate); // arbitrary prior allocation_owner retires outside the store lock
        return {};
    }catch(const Failure& f){return result(f);}catch(const std::bad_alloc&){return {WindowDiskError::budget_refused,"load allocation failed",{}};}
}
WindowDiskResult WindowDiskStore::erase(const std::string& id,const WindowDiskOptions& options){
    std::lock_guard lock(state_->mutex);auto& s=*state_;WindowDiskReceipt retired;
    try{retired.phase=WindowDiskPhase::unknown;require(opaque(id),"invalid artifact id");auto io=admit_io(s);reconcile(s,options);
        const auto name=art_name(id);auto it=s.files.find(name);require(it!=s.files.end(),"artifact absent");
        if(it->second.use_count()!=1)fail(WindowDiskError::conflict,"artifact still leased");
        for(const auto& [path,file]:s.files){(void)file;if(path.starts_with("catalog-")&&read_record(s,path).artifact==id)fail(WindowDiskError::conflict,"artifact retained by catalog history");}
        retired.operation=std::string(64,'0');retired.artifact=id;retired.catalog=std::string(64,'0');retired.fingerprint=unhex(id);
        retired.phase=WindowDiskPhase::published_durability_unknown;verify_artifact(s,id,options);flush_dir(s,options,"retire_artifact_confirm");retired.phase=WindowDiskPhase::durable_artifact;
        write_record(s,"retired-"+id+".mark2",retired,options,"retirement");cancelled(options);
        checkpoint(options,"delete_unlink");if(::unlinkat(s.directory.n,name.c_str(),0)!=0)fail(WindowDiskError::io,"artifact deletion failed");
        retired.phase=WindowDiskPhase::retirement_durability_unknown;
        flush_dir(s,options,"delete_dir_flush");s.files.erase(it);retired.phase=WindowDiskPhase::retired_artifact;
        return {WindowDiskError::none,{},retired};
    }catch(const Failure& f){return result(f,retired);}catch(const std::bad_alloc&){return {WindowDiskError::budget_refused,"delete metadata allocation failed",retired};}
}
WindowDiskResult window_disk_full_digest(const std::string& path,uint64_t max_bytes,StateDigest& out,const WindowDiskOptions& options){try{Fd fd(::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK));if(fd.n<0)fail(WindowDiskError::io,"model input open failed");struct stat before{},after{};if(::fstat(fd.n,&before)!=0)fail(WindowDiskError::io,"model input stat failed");require(S_ISREG(before.st_mode)&&before.st_size>=0&&uint64_t(before.st_size)<=max_bytes,"model input size/type refused");std::array<uint8_t,65536> buffer{};uint64_t remaining=uint64_t(before.st_size);WindowHasher h;while(remaining){auto n=size_t(std::min<uint64_t>(remaining,buffer.size()));cancelled(options);read_all(fd.n,buffer.data(),n);h.update(buffer.data(),n);remaining-=n;checkpoint(options,"model_hash_chunk");}if(::fstat(fd.n,&after)!=0)fail(WindowDiskError::io,"model input final stat failed");
#if defined(__APPLE__)
    const bool timestamps=before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec&&before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec&&before.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec&&before.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec;
#else
    const bool timestamps=before.st_mtim.tv_sec==after.st_mtim.tv_sec&&before.st_mtim.tv_nsec==after.st_mtim.tv_nsec&&before.st_ctim.tv_sec==after.st_ctim.tv_sec&&before.st_ctim.tv_nsec==after.st_ctim.tv_nsec;
#endif
    require(before.st_size==after.st_size&&before.st_dev==after.st_dev&&before.st_ino==after.st_ino&&timestamps,"model input changed during full hash");out=h.digest();return {};}catch(const Failure& f){return result(f);}catch(const std::bad_alloc&){return {WindowDiskError::budget_refused,"model hash allocation failed",{}};}}
#else
struct WindowDiskState {};
WindowDiskHostUsage WindowDiskStore::host_usage() const {return {};}
namespace {WindowDiskResult unsupported(){return {WindowDiskError::unsupported,"RAW disk-v2 durability is not implemented on Windows",{}};}}
WindowDiskResult WindowDiskStore::open(const std::string&,const WindowDiskLimits&,WindowBudget,std::shared_ptr<WindowDiskStore>&){return unsupported();}
WindowDiskResult WindowDiskStore::save(const WindowDiskOperation&,const BundleHandle&,const WindowKey&,const WindowStateLayout&,const StateDigest&,uint64_t,WindowDiskArtifact&,const WindowDiskOptions&){return unsupported();}
WindowDiskResult WindowDiskStore::query(const WindowDiskOperation&,const WindowDiskOptions&){return unsupported();}
WindowDiskResult WindowDiskStore::catalog(const StateDigest&,WindowDiskReceipt&){return unsupported();}
WindowDiskResult WindowDiskStore::acquire(const std::string&,WindowDiskArtifact&){return unsupported();}
WindowDiskResult WindowDiskStore::load(const WindowDiskArtifact&,const WindowKey&,const WindowStateLayout&,BundleHandle&,const WindowDiskOptions&){return unsupported();}
WindowDiskResult WindowDiskStore::erase(const std::string&,const WindowDiskOptions&){return unsupported();}
WindowDiskResult window_disk_full_digest(const std::string&,uint64_t,StateDigest&,const WindowDiskOptions&){return unsupported();}
#endif
} // namespace strata::core
