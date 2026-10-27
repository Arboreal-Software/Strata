#include "strata/core/verified_inputs.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>
#ifdef __linux__
#include <sys/random.h>
#endif
namespace strata::core {
namespace {
#ifdef __linux__
struct Failure {VerifiedArtifactError code;int number;std::string text;};
[[noreturn]] void fail(VerifiedArtifactError c,const std::string& t,int e=0){throw Failure{c,e,t};}
void need(bool b,const char* t){if(!b)fail(VerifiedArtifactError::invalid,t);}
uint64_t add(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,"group resource overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){need(!a||b<=UINT64_MAX/a,"group resource overflow");return a*b;}
#ifdef __linux__
void close_checked(int& fd){if(fd<0)return;int n=std::exchange(fd,-1);if(::close(n)){std::fprintf(stderr,"strata: verified input close uncertain errno=%d\n",errno);std::_Exit(1);}}
struct Fd {int n=-1;~Fd(){close_checked(n);}};
void io_ok(bool b,const char* step){if(!b)fail(VerifiedArtifactError::io,step,errno);}
void checkpoint(const VerifiedArtifactOptions& o,const char* step){if(o.cancelled&&o.cancelled())fail(VerifiedArtifactError::cancelled,"input group cancelled");if(o.fault){int e=o.fault(step);if(e)fail(VerifiedArtifactError::io,step,e);}}
std::shared_ptr<const ReservationLease> reserve(WindowBudget& b,uint64_t host,uint64_t disk,uint64_t io){WindowResources r;r.device_bytes.resize(b.snapshot().capacity.device_bytes.size());r.host_bytes=host;r.disk_bytes=disk;r.io_slots=io;auto got=b.reserve(r,ReservationPurpose::foreground);if(!got)fail(VerifiedArtifactError::budget_refused,"whole input group admission refused");return got.lease;}
#endif
bool safe_relative(const std::string& p,const VerifiedInputLimits& l){
    if(p.empty()||p.size()>l.max_path_bytes||p.front()=='/'||p.back()=='/'||p.find('\0')!=std::string::npos)return false;
    uint64_t depth=0;size_t at=0;while(at<p.size()){auto end=p.find('/',at);if(end==std::string::npos)end=p.size();auto part=p.substr(at,end-at);if(part.empty()||part=="."||part==".."||++depth>l.max_depth)return false;for(unsigned char c:part)if(c<=32||c==127||c=='\\'||c==',')return false;at=end+1;}return true;
}
std::string parent(const std::string& p){auto i=p.rfind('/');return i==std::string::npos?"":p.substr(0,i);}
std::string child(const std::string& p,const std::string& n){return p.empty()?n:p+"/"+n;}
std::vector<std::string> fields(const std::string& line){std::istringstream in(line);std::vector<std::string> out;std::string s;while(in>>s){need(out.size()<20,"metadata column bound exceeded");out.push_back(std::move(s));}return out;}
uint64_t number(const std::string& s){uint64_t n=0;auto r=std::from_chars(s.data(),s.data()+s.size(),n);need(r.ec==std::errc()&&r.ptr==s.data()+s.size(),"invalid unsigned metadata field");return n;}
void integer(const std::string& s){int64_t n=0;auto r=std::from_chars(s.data(),s.data()+s.size(),n);need(r.ec==std::errc()&&r.ptr==s.data()+s.size(),"invalid signed metadata field");}
#endif
}
struct VerifiedInputGroup {
    enum class Phase {building,live,sealed,retiring,retired};
    mutable std::mutex mutex;
    Phase phase=Phase::building;
    std::shared_ptr<const ReservationLease> retained;
    std::shared_ptr<const ReservationLease> disk;
    std::shared_ptr<const ReservationLease> construction_io;
    std::optional<WindowBudget> outer_budget;
    std::vector<std::shared_ptr<const VerifiedArtifact>> artifacts;
    std::map<std::string,size_t> aliases;
    std::set<std::string> directories;
    std::vector<std::string> created_files,created_dirs;
    std::string root_name;
    int root_fd=-1,parent_fd=-1;
    void cleanup() noexcept {
#ifdef __linux__
        for(auto i=created_files.rbegin();i!=created_files.rend();++i)if(::unlinkat(root_fd,i->c_str(),0)){std::fprintf(stderr,"strata: input alias cleanup uncertain\n");std::_Exit(1);}
        for(auto i=created_dirs.rbegin();i!=created_dirs.rend();++i)if(::unlinkat(root_fd,i->c_str(),AT_REMOVEDIR)){std::fprintf(stderr,"strata: input directory cleanup uncertain\n");std::_Exit(1);}
        close_checked(root_fd);
        if(!root_name.empty()&&::unlinkat(parent_fd,root_name.c_str(),AT_REMOVEDIR)){std::fprintf(stderr,"strata: input root cleanup uncertain\n");std::_Exit(1);}
        close_checked(parent_fd);
        artifacts.clear(); // checked snapshot close while cleanup IO is still held
        created_files.clear();created_dirs.clear();root_name.clear();disk.reset(); // metadata remains charged until the retired object dies
#endif
    }
    ~VerifiedInputGroup(){
        if(phase==Phase::building){cleanup();return;}
        if(phase!=Phase::retired){std::fprintf(stderr,"strata: unretired input group lifetime contract violated\n");std::_Exit(1);}
    }
    std::string path(const std::string& alias)const{return "/proc/self/fd/"+std::to_string(root_fd)+"/"+alias;}
};
std::string VerifiedInputLease::loader_path()const{return group_?group_->path(alias_):std::string();}
int VerifiedInputLease::readonly_fd()const{return group_&&index_<group_->artifacts.size()?group_->artifacts.at(index_)->readonly_fd():-1;}
uint64_t VerifiedInputLease::size()const{return group_&&index_<group_->artifacts.size()?group_->artifacts.at(index_)->size():0;}
StateDigest VerifiedInputLease::content_digest()const{return group_&&index_<group_->artifacts.size()?group_->artifacts.at(index_)->content_digest():StateDigest{};}
ProfileArtifact VerifiedInputLease::profile_artifact(ProfileRole role,std::string name)const{
    if(!group_||index_>=group_->artifacts.size())throw std::invalid_argument("empty/directory input lease");
    const auto& a=group_->artifacts.at(index_);
    return {role,std::move(name),a->size(),a->content_digest(),a->verification_digest(),true,std::make_shared<VerifiedInputLease>(*this)};
}
size_t VerifiedInputBundle::alias_count()const{return group_->aliases.size();}
bool VerifiedInputBundle::resolve(const std::string& alias,VerifiedInputLease& out)const{
    VerifiedInputLease candidate;
    {std::lock_guard lock(group_->mutex);if(group_->phase!=VerifiedInputGroup::Phase::live)return false;auto i=group_->aliases.find(alias);if(i==group_->aliases.end())return false;candidate.group_=group_;candidate.index_=i->second;candidate.alias_=alias;}
    out=std::move(candidate);return true;
}
bool VerifiedInputBundle::resolve_directory(const std::string& alias,VerifiedInputLease& out)const{
    VerifiedInputLease candidate;
    {std::lock_guard lock(group_->mutex);if(group_->phase!=VerifiedInputGroup::Phase::live||!group_->directories.contains(alias))return false;candidate.group_=group_;candidate.index_=SIZE_MAX;candidate.alias_=alias;}
    out=std::move(candidate);return true;
}
bool VerifiedInputBundle::retired()const{std::lock_guard lock(group_->mutex);return group_->phase==VerifiedInputGroup::Phase::retired;}
VerifiedInputRetirement VerifiedInputBundle::retire_checked()const{
#ifdef __linux__
    {std::lock_guard lock(group_->mutex);if(group_->phase==VerifiedInputGroup::Phase::retired)return VerifiedInputRetirement::retired;if(group_->phase==VerifiedInputGroup::Phase::retiring)return VerifiedInputRetirement::busy_retiring;group_->phase=VerifiedInputGroup::Phase::sealed;if(group_.use_count()!=1)return VerifiedInputRetirement::busy_leases;group_->phase=VerifiedInputGroup::Phase::retiring;}
    std::shared_ptr<const ReservationLease> io;
    try{io=reserve(*group_->outer_budget,0,0,1);}catch(...){std::lock_guard lock(group_->mutex);group_->phase=VerifiedInputGroup::Phase::sealed;return VerifiedInputRetirement::budget_refused;}
    group_->cleanup();
    {std::lock_guard lock(group_->mutex);group_->phase=VerifiedInputGroup::Phase::retired;}
    return VerifiedInputRetirement::retired;
#else
    return VerifiedInputRetirement::retired;
#endif
}
VerifiedArtifactResult VerifiedInputBundle::prepare(const VerifiedInputManifest& manifest,const std::string& private_dir,
    const VerifiedInputLimits& l,WindowBudget budget,std::shared_ptr<const VerifiedInputBundle>& out,const VerifiedArtifactOptions& options){
#ifndef __linux__
    (void)manifest;(void)private_dir;(void)l;(void)budget;(void)out;(void)options;return {VerifiedArtifactError::unsupported,0,"verified input views require Linux"};
#else
    try{
        need(!out||out->retired(),"prior output bundle must be explicitly retired before replacement");
        need(l.artifact.block_bytes&&l.artifact.block_bytes<=SIZE_MAX&&l.artifact.metadata_host_bytes>=4096&&l.artifact.max_file_bytes,"invalid artifact suballocation bounds");
        need(l.max_files&&l.max_aliases&&l.max_path_bytes&&l.max_path_bytes<=4096&&l.max_depth&&l.max_metadata_bytes&&l.view_host_bytes&&l.view_disk_bytes,"invalid input group bounds");
        need(!manifest.files.empty()&&manifest.files.size()<=l.max_files&&manifest.packs.size()<=l.max_files&&manifest.gguf_groups.size()<=l.max_files,"input group count bound");
        auto bundle=std::shared_ptr<VerifiedInputBundle>(new VerifiedInputBundle);auto group=std::make_shared<VerifiedInputGroup>();bundle->group_=group;group->outer_budget=budget;
        std::vector<VerifiedInputFile> unique;std::map<std::string,size_t> sources;uint64_t disk=l.view_disk_bytes;
        for(const auto& file:manifest.files){need(file.source.size()>1&&file.source.front()=='/'&&safe_relative(file.source.substr(1),l),"source path must be normalized absolute");need(file.expected_bytes&&file.expected_bytes<=l.artifact.max_file_bytes&&std::any_of(file.expected_digest.begin(),file.expected_digest.end(),[](auto b){return b!=0;}),"missing input size/full digest");need(!file.aliases.empty()&&file.aliases.size()<=l.max_aliases,"missing/oversize aliases");auto [s,inserted]=sources.emplace(file.source,unique.size());if(inserted){unique.push_back({file.source,file.expected_bytes,file.expected_digest,{}});disk=add(disk,add(file.expected_bytes,l.artifact.snapshot_disk_overhead_bytes));}else need(unique[s->second].expected_bytes==file.expected_bytes&&unique[s->second].expected_digest==file.expected_digest,"conflicting source declarations");
            for(const auto& alias:file.aliases){need(safe_relative(alias,l),"unsafe input alias");need(group->aliases.size()<l.max_aliases&&group->aliases.emplace(alias,s->second).second,"duplicate/oversize alias");for(auto d=parent(alias);!d.empty();d=parent(d))group->directories.insert(d);}}
        for(const auto& d:group->directories)need(!group->aliases.contains(d),"file/directory alias collision");
        std::set<std::string> pack_names;
        for(const auto& pack:manifest.packs){need(pack_names.insert(pack.pack_directory).second,"duplicate pack closure");need(safe_relative(pack.pack_directory,l)&&group->aliases.contains(child(pack.pack_directory,"index.txt")),"pack index missing");if(!pack.primary_gguf.empty())need(safe_relative(pack.primary_gguf,l)&&group->aliases.contains(pack.primary_gguf),"primary GGUF alias missing");}
        uint64_t group_refs=0;
        for(const auto& list:manifest.gguf_groups){group_refs=add(group_refs,list.size());need(group_refs<=l.max_aliases,"GGUF reference count bound");need(!list.empty()&&list.size()<=l.max_files,"invalid GGUF group");std::set<std::string> seen;for(const auto& a:list)need(group->aliases.contains(a)&&seen.insert(a).second,"missing/duplicate GGUF alias");}
        const auto packs=manifest.packs;const auto groups=manifest.gguf_groups;
        const auto private_root=private_dir; // inputs survive synchronous callback mutation
        // Explicit conservative map/vector/path metadata envelope, not RSS proof.
        const auto metadata=add(add(l.view_host_bytes,add(mul(unique.size(),l.artifact.metadata_host_bytes),mul(add(group->aliases.size(),group->directories.size()),add(mul(l.max_path_bytes,8),1024)))),mul(group_refs,add(l.max_path_bytes,128)));
        const auto transient=add(mul(l.max_metadata_bytes,16),add(l.artifact.block_bytes,l.artifact.metadata_host_bytes));
        group->retained=reserve(budget,metadata,0,0);group->disk=reserve(budget,0,disk,0);group->construction_io=reserve(budget,transient,0,1); // ALL before any filesystem access
        auto source_text=[&](const std::string& alias){const auto& file=unique.at(group->aliases.at(alias));need(file.expected_bytes<=l.max_metadata_bytes&&file.expected_bytes<=SIZE_MAX,"metadata file exceeds group bound");checkpoint(options,"closure_read");Fd fd;fd.n=::open(file.source.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);io_ok(fd.n>=0,"closure open");struct stat st{};io_ok(::fstat(fd.n,&st)==0,"closure stat");need(S_ISREG(st.st_mode)&&st.st_size>=0&&uint64_t(st.st_size)==file.expected_bytes,"closure extent mismatch");std::string text(size_t(file.expected_bytes),'\0');size_t at=0;while(at<text.size()){checkpoint(options,"closure_read");auto n=::read(fd.n,text.data()+at,std::min<size_t>(text.size()-at,size_t(l.artifact.block_bytes)));if(n<0&&errno==EINTR)continue;io_ok(n>=0,"closure read");need(n!=0,"closure truncated");at+=size_t(n);}WindowHasher hash;hash.update(text.data(),text.size());need(hash.digest()==file.expected_digest,"closure full digest mismatch");return text;};
        // Validate reopen closure on verified metadata BEFORE any snapshot/view write.
        for(const auto& pack:packs){std::istringstream index(source_text(child(pack.pack_directory,"index.txt")));std::string line;bool header=false;size_t rows=0;std::set<std::string> names;
            while(std::getline(index,line)){auto f=fields(line);need(!f.empty(),"empty pack index row");if(f[0]=="#"){if(f.size()>=2&&f[1]=="align"){need(f.size()==7&&f[3]=="pool"&&f[5]=="tensors"&&number(f[2])&&number(f[4])&&number(f[6]),"invalid pack index header");header=true;}continue;}need(f.size()==19&&names.insert(f[0]).second,"invalid/duplicate pack index row");for(size_t k=1;k<f.size();++k)integer(f[k]);const auto id=number(f[1]);need(id<4,"unknown pack file ID");const char* names_by_id[]={"dense.bin","embd.bin","experts.bin","extra.bin"};const auto alias=child(pack.pack_directory,names_by_id[id]);need(group->aliases.contains(alias),"undeclared pack reopen");const auto size=unique[group->aliases.at(alias)].expected_bytes;const auto off=number(f[3]),bytes=number(f[4]);need(off<=size&&bytes<=size-off,"pack source extent mismatch");++rows;}need(header&&rows,"missing pack header/rows");
            const auto native=child(pack.pack_directory,"native_experts.txt");if(group->aliases.contains(native)){need(!pack.primary_gguf.empty(),"native expert primary missing");std::istringstream input(source_text(native));bool version_seen=false;while(std::getline(input,line)){if(line.empty())continue;if(line[0]=='#'){const std::string tag="# strata native experts v";if(line.starts_with(tag)){auto end=line.find_first_not_of("0123456789",tag.size());auto version=number(line.substr(tag.size(),end-tag.size()));need(version>=1&&version<=5,"unsupported native expert grammar");need(!version_seen,"duplicate expert version");version_seen=true;if(version==5)need(group->aliases.contains(child(pack.pack_directory,"hadamard.txt")),"missing Hadamard sidecar");}continue;}auto f=fields(line);need(f.size()==5||f.size()==8||f.size()==9,"unsupported native expert row");for(size_t k=0;k<std::min<size_t>(8,f.size());++k)number(f[k]);if(f.size()<9)continue;std::vector<std::string> parts;size_t start=0;for(;;){auto comma=f[8].find(',',start);parts.push_back(f[8].substr(start,comma==std::string::npos?comma:comma-start));if(comma==std::string::npos)break;start=comma+1;}need(parts.size()==1||parts.size()==3,"invalid native expert shard roles");for(const auto& name:parts)if(!name.empty()){need(safe_relative(name,l),"unsafe native expert reopen");need(group->aliases.contains(child(parent(pack.primary_gguf),name)),"undeclared native expert reopen");}}need(version_seen,"native expert version missing");}}
        WindowResources capacity;capacity.device_bytes.resize(budget.snapshot().capacity.device_bytes.size());capacity.host_bytes=add(metadata,transient);capacity.disk_bytes=disk;capacity.io_slots=1;WindowResources zero;zero.device_bytes.resize(capacity.device_bytes.size());WindowBudget prepaid(capacity,zero);
        group->artifacts.reserve(unique.size());for(const auto& record:unique){const auto* file=&record;checkpoint(options,"snapshot_start");auto one=l.artifact;one.max_file_bytes=file->expected_bytes;std::shared_ptr<const VerifiedArtifact> artifact;auto result=VerifiedArtifact::snapshot(file->source,private_root,file->expected_digest,one,prepaid,artifact,options);if(!result)fail(result.error,result.detail,result.system_error);need(artifact->size()==file->expected_bytes,"snapshot declared extent mismatch");group->artifacts.push_back(std::move(artifact));}
        checkpoint(options,"view_create");group->parent_fd=::open(private_root.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);io_ok(group->parent_fd>=0,"view parent open");
        std::string root_name;for(int attempt=0;attempt<16;++attempt){unsigned char random[16];ssize_t got;do{got=::getrandom(random,sizeof random,0);}while(got<0&&errno==EINTR);io_ok(got==sizeof random,"view random name");root_name="strata-inputs-";constexpr char h[]="0123456789abcdef";for(auto b:random){root_name+=h[b>>4];root_name+=h[b&15];}if(::mkdirat(group->parent_fd,root_name.c_str(),0700)==0){group->root_name=std::move(root_name);break;}if(errno!=EEXIST)fail(VerifiedArtifactError::io,"view directory create",errno);}need(!group->root_name.empty(),"view name collisions");
        group->root_fd=::openat(group->parent_fd,group->root_name.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);io_ok(group->root_fd>=0,"view directory open");group->created_dirs.reserve(group->directories.size());group->created_files.reserve(group->aliases.size());
        for(const auto& d:group->directories){group->created_dirs.push_back(d);if(::mkdirat(group->root_fd,d.c_str(),0700)){const int e=errno;group->created_dirs.pop_back();fail(VerifiedArtifactError::io,"view child create",e);}}
        for(const auto& [alias,index]:group->aliases){checkpoint(options,"alias_create");const auto target=group->artifacts[index]->loader_path();group->created_files.push_back(alias);if(::symlinkat(target.c_str(),group->root_fd,alias.c_str())){const int e=errno;group->created_files.pop_back();fail(VerifiedArtifactError::io,"view alias create",e);}}
        for(const auto& list:groups){std::vector<std::string> paths;for(const auto& alias:list)paths.push_back(group->path(alias));strata::GgufModel model(paths);}
        checkpoint(options,"group_publish");group->construction_io.reset();group->phase=VerifiedInputGroup::Phase::live;out=std::move(bundle);return {};
    }catch(const Failure& f){return {f.code,f.number,f.text};}catch(const std::bad_alloc&){return {VerifiedArtifactError::budget_refused,0,"input group metadata allocation failed"};}catch(const std::exception& e){return {VerifiedArtifactError::invalid,0,e.what()};}catch(...){return {VerifiedArtifactError::invalid,0,"input group callback failed"};}
#endif
}
} // namespace strata::core
