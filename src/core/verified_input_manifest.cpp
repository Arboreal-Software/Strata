#include "strata/core/verified_input_manifest.hpp"
#include "strata/core/expert_profile_inventory.hpp"
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <set>
#include <stdexcept>
#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace strata::core {
namespace {
struct Failure {VerifiedArtifactError code;int number;std::string text;};
void need(bool b,const char* t){if(!b)throw Failure{VerifiedArtifactError::invalid,0,t};}
uint64_t add(uint64_t a,uint64_t b){need(b<=UINT64_MAX-a,"manifest resource overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){need(!a||b<=UINT64_MAX/a,"manifest resource overflow");return a*b;}
uint64_t number(const std::string& s){need(!s.empty(),"missing decimal");uint64_t n=0;auto r=std::from_chars(s.data(),s.data()+s.size(),n);need(r.ec==std::errc()&&r.ptr==s.data()+s.size(),"invalid/overflow decimal");return n;}
StateDigest digest_hex(const std::string& s){need(s.size()==64,"SHA256 must be 64 lowercase hex characters");StateDigest d{};auto digit=[](char c){need((c>='0'&&c<='9')||(c>='a'&&c<='f'),"invalid SHA256 hex");return c<='9'?c-'0':c-'a'+10;};for(size_t i=0;i<32;++i)d[i]=uint8_t(16*digit(s[2*i])+digit(s[2*i+1]));need(std::any_of(d.begin(),d.end(),[](auto x){return x!=0;}),"zero expected digest");return d;}
bool identifier(const std::string& s){return !s.empty()&&s.size()<=64&&std::all_of(s.begin(),s.end(),[](unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.';});}
bool relative(const std::string& s,uint64_t depth,uint64_t bytes){if(s.empty()||s.size()>bytes||s.front()=='/'||s.back()=='/')return false;size_t at=0;uint64_t n=0;while(at<s.size()){auto e=s.find('/',at);if(e==std::string::npos)e=s.size();auto part=s.substr(at,e-at);if(part.empty()||part=="."||part==".."||++n>depth)return false;for(unsigned char c:part)if(c<=32||c==127||c=='\\'||c==',')return false;at=e+1;}return true;}
std::string parent(const std::string& s){auto p=s.rfind('/');return p==std::string::npos?"":s.substr(0,p);}
const std::set<std::string>& roles(){static const std::set<std::string> r{"pack","native-primary","native-group","dense-group","head-primary","head-group","embedding","ple","expert-profile"};return r;}
VerifiedArtifactResult result(const Failure& f){return {f.code,f.number,f.text};}
#ifdef __linux__
struct Fd {int n=-1;~Fd(){if(n>=0&&::close(n)){std::fprintf(stderr,"strata: manifest close uncertain; worker restart required\n");std::_Exit(1);}}};
void io(bool b,const char* text){if(!b)throw Failure{VerifiedArtifactError::io,errno,text};}
void checkpoint(const VerifiedArtifactOptions& o,const char* step){if(o.cancelled&&o.cancelled())throw Failure{VerifiedArtifactError::cancelled,0,"manifest cancelled"};if(o.fault){int e=o.fault(step);if(e)throw Failure{VerifiedArtifactError::io,e,step};}}
std::shared_ptr<const ReservationLease> reserve(WindowBudget& b,uint64_t host,uint64_t slots){WindowResources r;r.device_bytes.resize(b.snapshot().capacity.device_bytes.size());r.host_bytes=host;r.io_slots=slots;auto got=b.reserve(r,ReservationPurpose::foreground);if(!got)throw Failure{VerifiedArtifactError::budget_refused,0,"manifest host/IO admission refused"};return got.lease;}
#endif
std::vector<std::string> paths(const std::vector<VerifiedInputLease>& v){std::vector<std::string> r;r.reserve(v.size());for(const auto& lease:v)r.push_back(lease.loader_path());return r;}
}
std::vector<std::string> VerifiedDriverBindings::native_paths()const{return paths(native);}
std::vector<std::string> VerifiedDriverBindings::dense_paths()const{return paths(dense);}
std::vector<std::string> VerifiedDriverBindings::head_paths()const{return paths(head);}
VerifiedArtifactResult ParsedVerifiedInputs::read(const std::string& path,const StateDigest& expected,
    const VerifiedManifestParseLimits& bounds,WindowBudget budget,std::shared_ptr<const ParsedVerifiedInputs>& out,const VerifiedArtifactOptions& options){
#ifndef __linux__
    (void)path;(void)expected;(void)bounds;(void)budget;(void)out;(void)options;return {VerifiedArtifactError::unsupported,0,"verified manifest reader requires Linux"};
#else
    try{
        need(bounds.max_bytes&&bounds.max_bytes<=SIZE_MAX&&bounds.max_records&&bounds.max_line_bytes&&bounds.max_line_bytes<=bounds.max_bytes&&bounds.max_field_bytes&&bounds.max_field_bytes<=bounds.max_line_bytes,"invalid manifest parser bounds");
        need(!path.empty()&&path.size()<=4096&&path.find('\0')==std::string::npos,"invalid manifest path");need(std::any_of(expected.begin(),expected.end(),[](auto b){return b!=0;}),"missing full manifest digest");
        auto memory=reserve(budget,add(mul(bounds.max_bytes,16),add(mul(bounds.max_records,1024),4096)),0);auto admission=reserve(budget,0,1);
        auto plan=std::shared_ptr<ParsedVerifiedInputs>(new ParsedVerifiedInputs);plan->parse_owner_=memory;
        std::string text;
        {checkpoint(options,"manifest_open");Fd fd;fd.n=::open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK);io(fd.n>=0,"manifest open");struct stat st{};io(::fstat(fd.n,&st)==0,"manifest stat");need(S_ISREG(st.st_mode)&&st.st_size>0&&uint64_t(st.st_size)<=bounds.max_bytes,"manifest file bound/type");text.resize(size_t(st.st_size));size_t at=0;while(at<text.size()){checkpoint(options,"manifest_read");auto n=::read(fd.n,text.data()+at,std::min<size_t>(text.size()-at,65536));if(n<0&&errno==EINTR)continue;io(n>=0,"manifest read");need(n!=0,"manifest truncated");at+=size_t(n);} }
        WindowHasher hash;hash.update(text.data(),text.size());plan->digest_=hash.digest();if(plan->digest_!=expected)return {VerifiedArtifactError::hash_mismatch,0,"full manifest digest mismatch"};need(text.back()=='\n'&&text.find('\0')==std::string::npos&&text.find('\r')==std::string::npos,"manifest requires LF-terminated NUL-free records");
        std::map<std::string,uint64_t> limits;std::map<std::string,VerifiedInputFile> files;std::vector<std::pair<std::string,std::string>> aliases;std::map<std::string,std::map<uint64_t,std::string>> groups;uint64_t records=0;unsigned schema=0;size_t at=0;
        while(at<text.size()){const auto end=text.find('\n',at);need(end-at<=bounds.max_line_bytes&&++records<=bounds.max_records,"manifest line/record bound");std::vector<std::string> f;for(size_t p=at;;){auto e=text.find('\t',p);if(e==std::string::npos||e>end)e=end;need(e>p&&e-p<=bounds.max_field_bytes&&f.size()<8,"manifest field bound/empty field");f.push_back(text.substr(p,e-p));if(e==end)break;p=e+1;}at=end+1;
            if(records==1){need(f.size()==2&&f[0]=="STRATA_VERIFIED_INPUT_MANIFEST"&&(f[1]=="1"||f[1]=="2"),"unsupported manifest schema");schema=unsigned(f[1][0]-'0');continue;}
            if(f[0]=="LIMIT"){need(f.size()==3&&limits.emplace(f[1],number(f[2])).second,"duplicate/malformed limit");}
            else if(f[0]=="FILE"){need(f.size()==5&&identifier(f[1]),"malformed FILE");VerifiedInputFile file{f[4],number(f[2]),digest_hex(f[3]),{}};need(files.emplace(f[1],std::move(file)).second,"duplicate FILE id");}
            else if(f[0]=="ALIAS"){need(f.size()==3&&identifier(f[1]),"malformed ALIAS");aliases.emplace_back(f[1],f[2]);}
            else if(f[0]=="GGUF"){need(f.size()==4&&identifier(f[1]),"malformed GGUF");need(groups[f[1]].emplace(number(f[2]),f[3]).second,"duplicate GGUF ordinal");}
            else if(f[0]=="PACK"){need(f.size()==3,"malformed PACK");plan->manifest_.packs.push_back({f[1],f[2]});}
            else if(f[0]=="BIND"){need(f.size()==3&&roles().contains(f[1])&&(schema==2||f[1]!="expert-profile")&&plan->binds_.emplace(f[1],f[2]).second,"unknown/duplicate BIND role");}
            else need(false,"unknown manifest record");
        }
        const std::vector<std::string> limit_keys={"max_files","max_aliases","max_path_bytes","max_depth","max_metadata_bytes","view_host_bytes","view_disk_bytes","artifact_max_file_bytes","artifact_block_bytes","artifact_metadata_host_bytes","artifact_disk_overhead_bytes","allow_full_copy"};
        need(limits.size()==limit_keys.size(),"missing/unknown manifest limits");for(const auto& key:limit_keys)need(limits.contains(key),"missing/unknown manifest limit");
        auto& l=plan->limits_;l={limits.at("max_files"),limits.at("max_aliases"),limits.at("max_path_bytes"),limits.at("max_depth"),limits.at("max_metadata_bytes"),limits.at("view_host_bytes"),limits.at("view_disk_bytes"),{limits.at("artifact_max_file_bytes"),limits.at("artifact_block_bytes"),limits.at("artifact_metadata_host_bytes"),limits.at("artifact_disk_overhead_bytes"),limits.at("allow_full_copy")!=0}};
        need(l.max_files&&l.max_aliases&&l.max_path_bytes&&l.max_path_bytes<=4096&&l.max_depth&&l.max_metadata_bytes&&l.view_host_bytes&&l.view_disk_bytes&&l.artifact.max_file_bytes&&l.artifact.block_bytes&&l.artifact.block_bytes<=SIZE_MAX&&l.artifact.metadata_host_bytes>=4096&&limits.at("allow_full_copy")<=1,"invalid manifest limits");
        need(!files.empty()&&files.size()<=l.max_files&&aliases.size()<=l.max_aliases&&groups.size()<=l.max_files,"manifest inventory count bound");std::set<std::string> alias_names,dirs;
        for(const auto& [id,alias]:aliases){need(files.contains(id)&&relative(alias,l.max_depth,l.max_path_bytes)&&alias_names.insert(alias).second,"missing FILE/unsafe/duplicate alias");files.at(id).aliases.push_back(alias);plan->original_by_alias_.emplace(alias,files.at(id).source);for(auto d=parent(alias);!d.empty();d=parent(d))dirs.insert(d);}
        for(const auto& d:dirs)need(!alias_names.contains(d),"alias file/directory collision");
        std::map<std::string,std::pair<uint64_t,StateDigest>> source_claims;
        for(auto& [id,file]:files){(void)id;need(file.source.size()>1&&file.source.front()=='/'&&relative(file.source.substr(1),l.max_depth,l.max_path_bytes)&&file.expected_bytes&&file.expected_bytes<=l.artifact.max_file_bytes&&!file.aliases.empty(),"FILE source/extent/aliases invalid");auto [old,inserted]=source_claims.emplace(file.source,std::make_pair(file.expected_bytes,file.expected_digest));need(inserted||old->second==std::make_pair(file.expected_bytes,file.expected_digest),"conflicting duplicate source");plan->manifest_.files.push_back(std::move(file));}
        uint64_t refs=0;for(const auto& [name,ordered]:groups){need(!ordered.empty()&&ordered.size()<=l.max_files,"invalid GGUF group");std::vector<std::string> values;std::set<std::string> seen;for(const auto& [ordinal,alias]:ordered){need(ordinal==values.size()&&alias_names.contains(alias)&&seen.insert(alias).second,"missing/duplicate/noncontiguous GGUF group");values.push_back(alias);}refs=add(refs,values.size());need(refs<=l.max_aliases,"GGUF reference bound");plan->groups_[name]=values;plan->manifest_.gguf_groups.push_back(std::move(values));}
        if(schema==1){need(plan->binds_.size()==8,"every v1 binding required");plan->binds_.emplace("expert-profile","-");}
        need(plan->binds_.size()==roles().size()&&plan->manifest_.packs.size()==1,"every binding and one pack required");auto& b=plan->binds_;const auto& pack=plan->manifest_.packs.front();need(relative(pack.pack_directory,l.max_depth,l.max_path_bytes)&&alias_names.contains(pack.pack_directory+"/index.txt")&&pack.pack_directory==b.at("pack")&&pack.primary_gguf==b.at("native-primary"),"pack binding/primary mismatch");
        auto group_check=[&](const std::string& role){need(plan->groups_.contains(b.at(role)),"binding group missing");};group_check("native-group");group_check("dense-group");need(plan->groups_.at(b.at("native-group")).front()==b.at("native-primary"),"native primary must be first shard");
        need((b.at("head-primary")=="-")==(b.at("head-group")=="-"),"partial disabled head binding");if(b.at("head-group")!="-"){group_check("head-group");need(plan->groups_.at(b.at("head-group")).front()==b.at("head-primary"),"head primary must be first shard");}
        for(const auto& role:{"embedding","ple","expert-profile"})need(b.at(role)=="-"||alias_names.contains(b.at(role)),"optional binding alias missing");
        if(b.at("expert-profile")!="-") {
            const auto& alias=b.at("expert-profile");
            const auto file=std::find_if(plan->manifest_.files.begin(),plan->manifest_.files.end(),[&](const auto& f){return std::find(f.aliases.begin(),f.aliases.end(),alias)!=f.aliases.end();});
            need(file!=plan->manifest_.files.end(),"profile FILE missing");
            plan->profile_reader_owner_=reserve(budget,add(mul(file->expected_bytes,4),4096),0);
        }
        // Workspace and IO released before replacing arbitrary old output owners.
        std::string().swap(text);admission.reset();out=std::move(plan);return {};
    }catch(const Failure& f){return result(f);}catch(const std::bad_alloc&){return {VerifiedArtifactError::budget_refused,0,"manifest metadata allocation failed"};}catch(const std::exception& e){return {VerifiedArtifactError::invalid,0,e.what()};}catch(...){return {VerifiedArtifactError::invalid,0,"manifest callback failed"};}
#endif
}
VerifiedArtifactResult ParsedVerifiedInputs::validate_overrides(const VerifiedDriverOverrides& overrides)const{
    try{for(const auto& [role,actual]:overrides.paths){need(roles().contains(role),"unknown caller override role");std::vector<std::string> expected;const auto& value=binds_.at(role);if(role=="pack")expected.push_back(parent(original_by_alias_.at(value+"/index.txt")));else if(role=="native-group"||role=="dense-group"||role=="head-group"){if(value!="-")for(const auto& a:groups_.at(value))expected.push_back(original_by_alias_.at(a));}else if(value!="-")expected.push_back(original_by_alias_.at(value));need(actual==expected,"caller override differs from verified manifest source");}return {};}catch(const Failure& f){return result(f);}catch(const std::exception& e){return {VerifiedArtifactError::invalid,0,e.what()};}
}
VerifiedArtifactResult ParsedVerifiedInputs::prepare_bundle(const std::string& directory,WindowBudget budget,
    const VerifiedDriverOverrides& overrides,std::shared_ptr<const VerifiedInputBundle>& out,const VerifiedArtifactOptions& options)const{
    const auto retained=shared_from_this();
    auto checked=validate_overrides(overrides);if(!checked)return checked;
    return VerifiedInputBundle::prepare(manifest_,directory,limits_,std::move(budget),out,options);
}
VerifiedArtifactResult ParsedVerifiedInputs::bind(const std::shared_ptr<const VerifiedInputBundle>& bundle,const VerifiedDriverOverrides& overrides,VerifiedDriverBindings& out)const{
    try{auto accepted=validate_overrides(overrides);if(!accepted)return accepted;need(bool(bundle),"missing prepared input bundle");need(bundle->alias_count()==original_by_alias_.size(),"bundle contains extra/missing aliases");VerifiedDriverBindings candidate;candidate.plan_=shared_from_this();candidate.manifest_digest=digest_;
        auto acquire=[&](const std::string& alias){VerifiedInputLease lease;need(bundle->resolve(alias,lease),"bundle alias unavailable");auto file=std::find_if(manifest_.files.begin(),manifest_.files.end(),[&](const auto& f){return std::find(f.aliases.begin(),f.aliases.end(),alias)!=f.aliases.end();});need(file!=manifest_.files.end()&&lease.size()==file->expected_bytes&&lease.content_digest()==file->expected_digest,"bundle differs from verified manifest");return lease;};
        need(bundle->resolve_directory(binds_.at("pack"),candidate.pack),"pack directory unavailable");
        // Bind/validate EVERY file alias, including pack sidecars, against this plan.
        for(const auto& file:manifest_.files)for(const auto& alias:file.aliases){auto checked=acquire(alias);(void)checked;}
        candidate.pack_index=acquire(binds_.at("pack")+"/index.txt");
        const char* payload_names[]={"dense.bin","embd.bin","experts.bin","extra.bin"};
        for(size_t id=0;id<candidate.pack_payloads.size();++id) {
            const auto alias=binds_.at("pack")+"/"+payload_names[id];
            if(original_by_alias_.contains(alias))candidate.pack_payloads[id]=acquire(alias);
        }
        const auto layout_alias=binds_.at("pack")+"/native_experts.txt";
        const auto hadamard_alias=binds_.at("pack")+"/hadamard.txt";
        candidate.native_expert_layout_absent=!original_by_alias_.contains(layout_alias);
        candidate.hadamard_absent=!original_by_alias_.contains(hadamard_alias);
        if(!candidate.native_expert_layout_absent)candidate.native_expert_layout=acquire(layout_alias);
        if(!candidate.hadamard_absent)candidate.hadamard=acquire(hadamard_alias);
        candidate.native_primary=acquire(binds_.at("native-primary"));
        auto group=[&](const std::string& role,std::vector<VerifiedInputLease>& output){for(const auto& alias:groups_.at(binds_.at(role)))output.push_back(acquire(alias));};group("native-group",candidate.native);group("dense-group",candidate.dense);
        candidate.head_disabled=binds_.at("head-group")=="-";if(!candidate.head_disabled){candidate.head_primary=acquire(binds_.at("head-primary"));group("head-group",candidate.head);}
        candidate.embedding_from_native=binds_.at("embedding")=="-";if(!candidate.embedding_from_native)candidate.embedding=acquire(binds_.at("embedding"));candidate.ple_disabled=binds_.at("ple")=="-";if(!candidate.ple_disabled)candidate.ple=acquire(binds_.at("ple"));
        candidate.expert_profile_disabled=binds_.at("expert-profile")=="-";candidate.expert_profile_disabled_minted_=candidate.expert_profile_disabled;if(!candidate.expert_profile_disabled)candidate.expert_profile=acquire(binds_.at("expert-profile"));candidate.expert_profile_workspace_=profile_reader_owner_;
        out=std::move(candidate);return {};
    }catch(const Failure& f){return result(f);}catch(const std::bad_alloc&){return {VerifiedArtifactError::budget_refused,0,"binding metadata allocation failed"};}catch(const std::exception& e){return {VerifiedArtifactError::invalid,0,e.what()};}
}
bool VerifiedDriverBindings::confirms_expert_profile_disabled() const noexcept {
    return expert_profile_disabled_minted_&&bool(plan_)&&expert_profile_disabled&&!expert_profile;
}
bool ExpertProfileCapture::create_disabled(uint64_t layers,uint64_t experts,
    std::shared_ptr<const ReservationLease> lease,std::shared_ptr<const VerifiedDriverBindings> bindings,
    std::unique_ptr<ExpertProfileCapture>& out,std::string& error) {
    try {
        uint64_t required=0;if(!expert_profile_disabled_host_bound(layers,experts,required,error))return false;
        if(!bindings||!bindings->confirms_expert_profile_disabled()||!lease||
           lease->purpose()!=ReservationPurpose::foreground||lease->resources().host_bytes<required) {
            error="disabled profile requires verified absent role and prepaid metadata";return false;
        }
        if((out&&lease==out->inventory_.metadata_reservation)||lease==bindings->expert_profile_workspace_||
           lease==bindings->plan_->parse_owner_||lease==bindings->plan_->profile_reader_owner_) {
            error="disabled profile metadata must have a distinct reservation";return false;
        }
        auto candidate=std::make_unique<ExpertProfileCapture>();
        candidate->inventory_.metadata_reservation=std::move(lease);
        candidate->inventory_.retained_input_bindings=std::move(bindings);
        candidate->inventory_.origin=ExpertProfileOrigin::verified_disabled;
        candidate->inventory_.disabled_layers=layers;candidate->inventory_.disabled_experts=experts;
        // No file-shaped fields are populated; construction is its own initial
        // attempt. Any later real reader attempt increments this and revokes it.
        candidate->attempt_generation_=1;candidate->complete_=true;
        candidate->disabled_claim_current_=[](const VerifiedDriverBindings& b) noexcept {return b.confirms_expert_profile_disabled();};
        error.clear();out.swap(candidate);return true;
    } catch(const std::bad_alloc&) {error.clear();return false;}
    catch(const std::exception& e) {try{error=e.what();}catch(...){error.clear();}return false;}
}
VerifiedArtifactResult VerifiedDriverBindings::preflight_expert_profile(WindowBudget budget,const VerifiedArtifactOptions& options) {
#ifndef __linux__
    (void)budget;(void)options;return {VerifiedArtifactError::unsupported,0,"profile preflight requires Linux"};
#else
    try {
        if(expert_profile_disabled)return {};
        need(bool(expert_profile),"missing retained expert profile");
        const auto retained=expert_profile;const uint64_t bytes=retained.size();
        need(bytes>=24,"expert profile header truncated");
        // Bound loader ranked/raw vectors by the already admitted full file
        // extent. Keep the allowance through model readers, not just this read.
        need(bool(expert_profile_workspace_)&&expert_profile_workspace_->resources().host_bytes>=add(mul(bytes,4),4096),"missing prepaid profile reader allowance");
        auto admission=reserve(budget,4096,1);
        unsigned char header[24]{};size_t at=0;
        while(at<sizeof header) {
            checkpoint(options,"expert_profile_header");
            const auto n=::pread(retained.readonly_fd(),header+at,sizeof header-at,off_t(at));
            if(n<0&&errno==EINTR)continue;
            io(n>=0,"expert profile header read");need(n!=0,"expert profile header truncated");at+=size_t(n);
        }
        auto word=[&](size_t offset){uint64_t n=0;for(size_t i=0;i<4;++i)n|=uint64_t(header[offset+i])<<(8*i);return n;};
        need(std::memcmp(header,"STRP",4)==0&&word(4)==1,"unsupported expert profile header");
        const auto layers=word(8),experts=word(12),wanted=word(16),ranked=word(20);
        need(layers&&layers<=65535&&experts&&experts<=65535,"expert profile geometry outside writer format");
        need(ranked<=wanted&&wanted<=mul(layers,experts),"expert profile count exceeds geometry");
        need(add(24,mul(ranked,4))<=bytes,"expert profile ranked list truncated");
        // Current writer appends a slot table; current reader ignores it. Its
        // complete bytes remain bound by the snapshot's expected full digest.
        admission.reset();return {};
    }catch(const Failure& f){return result(f);}catch(const std::bad_alloc&){return {VerifiedArtifactError::budget_refused,0,"profile preflight allocation failed"};}catch(const std::exception& e){return {VerifiedArtifactError::invalid,0,e.what()};}
#endif
}
} // namespace strata::core
