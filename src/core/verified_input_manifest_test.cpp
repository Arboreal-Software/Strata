#include "strata/core/verified_input_manifest.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace strata::core;
namespace fs=std::filesystem;
static StateDigest hash(const std::string& s){WindowHasher h;h.update(s.data(),s.size());return h.digest();}
static std::string hex(const StateDigest& d){std::string s;const char* h="0123456789abcdef";for(auto b:d){s+=h[b>>4];s+=h[b&15];}return s;}
static void write(const std::string& p,const std::string& s){std::ofstream f(p,std::ios::binary|std::ios::trunc);f<<s;f.close();assert(f);}
static WindowBudget budget(uint64_t host=32<<20,uint64_t disk=32<<20,uint64_t io=1){return WindowBudget(WindowResources{{0},host,0,disk,io,0},WindowResources{{0},0,0,0,0,0});}
static VerifiedArtifactOptions options(){VerifiedArtifactOptions o;o.fault=[](const char* s){return std::string(s)=="clone"?EOPNOTSUPP:0;};return o;}
struct Fixture {
    std::string dir,path,text;
    Fixture(){char p[]="/tmp/strata-manifest-XXXXXX";auto* m=mkdtemp(p);assert(m);dir=m;path=dir+"/manifest.tsv";
        std::string gguf;auto n=[&](uint64_t v,int count){for(int i=0;i<count;++i)gguf+=char(v>>(i*8));};auto str=[&](std::string s){n(s.size(),8);gguf+=s;};n(0x46554747,4);n(3,4);n(0,8);n(1,8);str("general.architecture");n(8,4);str("qwen4exp");
        std::vector<std::pair<std::string,std::string>> files={{"model.gguf",gguf},{"index.txt","# align 16 pool 16 tensors 1\nw 0 2 0 4 0 4 1 1 0 0 0 0 0 4 0 0 0 0\n"},{"dense.bin","data"}};
        text="STRATA_VERIFIED_INPUT_MANIFEST\t1\n";
        for(const auto& [key,value]:std::vector<std::pair<std::string,uint64_t>>{{"max_files",32},{"max_aliases",64},{"max_path_bytes",256},{"max_depth",8},{"max_metadata_bytes",4096},{"view_host_bytes",8192},{"view_disk_bytes",8192},{"artifact_max_file_bytes",1<<20},{"artifact_block_bytes",64},{"artifact_metadata_host_bytes",4096},{"artifact_disk_overhead_bytes",4096},{"allow_full_copy",1}})text+="LIMIT\t"+key+"\t"+std::to_string(value)+"\n";
        for(const auto& [name,bytes]:files){write(dir+"/"+name,bytes);text+="FILE\t"+name+"\t"+std::to_string(bytes.size())+"\t"+hex(hash(bytes))+"\t"+dir+"/"+name+"\n";text+="ALIAS\t"+name+"\t"+(name=="model.gguf"?"models/":"pack/")+name+"\n";}
        text+="GGUF\tmain\t0\tmodels/model.gguf\nPACK\tpack\tmodels/model.gguf\n";
        for(const auto& [role,value]:std::vector<std::pair<std::string,std::string>>{{"pack","pack"},{"native-primary","models/model.gguf"},{"native-group","main"},{"dense-group","main"},{"head-primary","-"},{"head-group","-"},{"embedding","-"},{"ple","-"}})text+="BIND\t"+role+"\t"+value+"\n";
        write(path,text);
    }
    ~Fixture(){fs::remove_all(dir);}
};
static const VerifiedManifestParseLimits bounds{65536,256,4096,4096};
static std::string replace(std::string text,const std::string& from,const std::string& to){auto at=text.find(from);assert(at!=std::string::npos);text.replace(at,from.size(),to);return text;}
int main(){Fixture f;auto b=budget();std::shared_ptr<const ParsedVerifiedInputs> plan;auto r=ParsedVerifiedInputs::read(f.path,hash(f.text),bounds,b,plan);if(!r)std::cerr<<r.detail<<'\n';assert(r&&plan&&plan->digest()==hash(f.text)&&b.snapshot().reserved.host_bytes>0&&b.snapshot().reserved.io_slots==0);
    auto rejects=[&](const std::string& text){write(f.path,text);auto output=plan;auto prior=b.snapshot().reserved;auto result=ParsedVerifiedInputs::read(f.path,hash(text),bounds,b,output);assert(!result&&output==plan&&b.snapshot().reserved==prior);};
    rejects(replace(f.text,"MANIFEST\t1","MANIFEST\t2"));rejects(f.text+"BIND\tpack\tpack\n");rejects(replace(f.text,"BIND\tple\t-\n",""));rejects(replace(f.text,"BIND\tembedding\t-","BIND\tunknown\t-"));rejects(replace(f.text,"max_files\t32","max_files\t18446744073709551616"));rejects(replace(f.text,"GGUF\tmain\t0","GGUF\tmain\t1"));rejects(replace(f.text,"ALIAS\tdense.bin\tpack/dense.bin","ALIAS\tdense.bin\t../dense.bin"));rejects(replace(f.text,"head-primary\t-","head-primary\tmodels/model.gguf"));rejects(replace(f.text,"PACK\tpack\tmodels/model.gguf","PACK\tpack\tmissing"));rejects(replace(f.text,"artifact_block_bytes\t64","artifact_block_bytes\t0"));rejects(replace(f.text,"allow_full_copy\t1","allow_full_copy\t2"));rejects(f.text+"UNKNOWN\tx\n");
    write(f.path,f.text);auto expected=hash(f.text);expected[0]^=1;auto output=plan;r=ParsedVerifiedInputs::read(f.path,expected,bounds,b,output);assert(r.error==VerifiedArtifactError::hash_mismatch&&output==plan);output.reset();
    for(int kind=0;kind<2;++kind){auto small=budget(kind==0?1:32<<20,32<<20,kind==1?0:1);int opened=0;VerifiedArtifactOptions o;o.fault=[&](const char*){++opened;return 0;};std::shared_ptr<const ParsedVerifiedInputs> out;r=ParsedVerifiedInputs::read(f.path,hash(f.text),bounds,small,out,o);assert(r.error==VerifiedArtifactError::budget_refused&&!out&&opened==0&&small.snapshot().reserved.host_bytes==0);}
    {auto tiny=bounds;tiny.max_bytes=8;tiny.max_line_bytes=8;tiny.max_field_bytes=8;std::shared_ptr<const ParsedVerifiedInputs> none;auto refused=ParsedVerifiedInputs::read(f.path,hash(f.text),tiny,b,none);assert(!refused&&!none);}
    for(const std::string fault:{"manifest_open","manifest_read"}){VerifiedArtifactOptions o;o.fault=[&](const char* step){return std::string(step)==fault?EIO:0;};auto old=plan;auto reserved=b.snapshot().reserved;auto failed=ParsedVerifiedInputs::read(f.path,hash(f.text),bounds,b,old,o);assert(failed.error==VerifiedArtifactError::io&&failed.system_error==EIO&&old==plan&&b.snapshot().reserved==reserved);}
    {VerifiedArtifactOptions o;o.cancelled=[](){return true;};auto old=plan;auto cancelled=ParsedVerifiedInputs::read(f.path,hash(f.text),bounds,b,old,o);assert(cancelled.error==VerifiedArtifactError::cancelled&&old==plan);}
    VerifiedDriverOverrides overrides;overrides.paths={{"pack",{f.dir}},{"native-primary",{f.dir+"/model.gguf"}},{"native-group",{f.dir+"/model.gguf"}},{"dense-group",{f.dir+"/model.gguf"}}};assert(plan->validate_overrides(overrides));auto wrong=overrides;wrong.paths["native-primary"]={"/unlisted/model.gguf"};assert(!plan->validate_overrides(wrong));wrong=overrides;wrong.paths["unknown"]={};assert(!plan->validate_overrides(wrong));
    {int effects=0;auto o=options();o.fault=[&](const char*){++effects;return 0;};std::shared_ptr<const VerifiedInputBundle> untouched;auto refused=plan->prepare_bundle(f.dir,b,wrong,untouched,o);assert(!refused&&!untouched&&effects==0);}
    std::shared_ptr<const VerifiedInputBundle> bundle;assert(plan->prepare_bundle(f.dir,b,overrides,bundle,options()));VerifiedDriverBindings bindings;assert(plan->bind(bundle,overrides,bindings));assert(bindings.frontend==ProfileFrontend::raw_token_ids&&bindings.head_disabled&&bindings.ple_disabled&&bindings.embedding_from_native&&bindings.manifest_digest==plan->digest());assert(bindings.native_paths().size()==1&&bindings.pack_path().starts_with("/proc/self/fd/")&&bindings.native_paths()[0]!=f.dir+"/model.gguf");
    {strata::GgufModel model(bindings.native_paths());assert(model.meta().get("general.architecture")->s=="qwen4exp");std::ifstream file(bindings.pack_path()+"/dense.bin");std::string bytes;file>>bytes;assert(bytes=="data");}
    assert(bindings.pack_index&&bindings.pack_index.loader_path()==bindings.pack_path()+"/index.txt");
    assert(bindings.native_expert_layout_absent&&bindings.hadamard_absent&&!bindings.native_expert_layout&&!bindings.hadamard);
    assert(bindings.pack_payloads[0]&&bindings.pack_payloads[0].size()==4&&bindings.pack_payloads[0].content_digest()==hash("data"));
    assert(bindings.pack_payloads[0].loader_path()==bindings.pack_path()+"/dense.bin");
    for(size_t id=1;id<bindings.pack_payloads.size();++id)assert(!bindings.pack_payloads[id]);
    auto unchanged=bindings.pack_path();assert(!plan->bind(bundle,wrong,bindings)&&bindings.pack_path()==unchanged);
    auto extra=plan->manifest();extra.files[0].aliases.push_back("pack/unlisted.bin");std::shared_ptr<const VerifiedInputBundle> extra_bundle;assert(VerifiedInputBundle::prepare(extra,f.dir,plan->limits(),b,extra_bundle,options()));assert(extra_bundle->alias_count()==bundle->alias_count()+1);assert(!plan->bind(extra_bundle,overrides,bindings)&&bindings.pack_path()==unchanged);assert(extra_bundle->retire_checked()==VerifiedInputRetirement::retired);extra_bundle.reset();
    {auto altered=plan->manifest();for(auto& file:altered.files)if(file.source==f.dir+"/dense.bin")file.expected_digest=hash("evil");write(f.dir+"/dense.bin","evil");std::shared_ptr<const VerifiedInputBundle> different;assert(VerifiedInputBundle::prepare(altered,f.dir,plan->limits(),b,different,options()));assert(different->alias_count()==bundle->alias_count());assert(!plan->bind(different,overrides,bindings)&&bindings.pack_path()==unchanged);assert(different->retire_checked()==VerifiedInputRetirement::retired);different.reset();write(f.dir+"/dense.bin","data");}
    const auto metadata=b.snapshot().reserved.host_bytes;std::weak_ptr<const ParsedVerifiedInputs> retained=plan;plan.reset();assert(!retained.expired()&&b.snapshot().reserved.host_bytes==metadata);assert(bundle->retire_checked()==VerifiedInputRetirement::busy_leases);{auto held=retained.lock();VerifiedDriverBindings refused;assert(!held->bind(bundle,overrides,refused));}bindings={};assert(retained.expired());assert(bundle->retire_checked()==VerifiedInputRetirement::retired);bundle.reset();assert(b.snapshot().reserved.host_bytes==0&&b.snapshot().reserved.disk_bytes==0&&b.snapshot().reserved.io_slots==0);

    // Each canonical file ID retains its own immutable source authority, even
    // after the bindings object is dropped and retirement is requested.
    {Fixture all;const char* names[]={"dense.bin","embd.bin","experts.bin","extra.bin"};
        for(size_t id=1;id<4;++id){const std::string value="payload-"+std::to_string(id);write(all.dir+"/"+names[id],value);
            all.text+="FILE\t"+std::string(names[id])+"\t"+std::to_string(value.size())+"\t"+hex(hash(value))+"\t"+all.dir+"/"+names[id]+"\nALIAS\t"+names[id]+"\tpack/"+names[id]+"\n";}
        for(const auto& [name,value]:std::vector<std::pair<std::string,std::string>>{{"native_experts.txt","# strata native experts v1\n"},{"hadamard.txt","binding-only fixture"}}) {
            write(all.dir+"/"+name,value);all.text+="FILE\t"+name+"\t"+std::to_string(value.size())+"\t"+hex(hash(value))+"\t"+all.dir+"/"+name+"\nALIAS\t"+name+"\tpack/"+name+"\n";
        }
        write(all.path,all.text);auto ledger=budget();std::shared_ptr<const ParsedVerifiedInputs> p;
        assert(ParsedVerifiedInputs::read(all.path,hash(all.text),bounds,ledger,p));
        std::shared_ptr<const VerifiedInputBundle> group;assert(p->prepare_bundle(all.dir,ledger,{},group,options()));
        VerifiedDriverBindings binding;assert(p->bind(group,{},binding));
        assert(!binding.native_expert_layout_absent&&!binding.hadamard_absent);
        assert(binding.native_expert_layout.content_digest()==hash("# strata native experts v1\n"));
        assert(binding.hadamard.content_digest()==hash("binding-only fixture"));
        assert(binding.native_expert_layout.loader_path()==binding.pack_path()+"/native_experts.txt");
        assert(binding.hadamard.loader_path()==binding.pack_path()+"/hadamard.txt");
        for(size_t id=0;id<4;++id){const std::string value=id?"payload-"+std::to_string(id):"data";
            assert(binding.pack_payloads[id]&&binding.pack_payloads[id].loader_path()==binding.pack_path()+"/"+names[id]);
            assert(binding.pack_payloads[id].size()==value.size()&&binding.pack_payloads[id].content_digest()==hash(value));}
        auto held=binding.pack_payloads[2];binding={};p.reset();
        assert(group->retire_checked()==VerifiedInputRetirement::busy_leases);
        write(all.dir+"/experts.bin","changed original");
        std::ifstream input(held.loader_path());std::string actual;input>>actual;assert(actual=="payload-2");input.close();
        held={};assert(group->retire_checked()==VerifiedInputRetirement::retired);group.reset();
        assert(ledger.snapshot().reserved.host_bytes==0&&ledger.snapshot().reserved.disk_bytes==0);
    }

    // Schema1 is unchanged and cannot silently authorize an external profile.
    {Fixture old;std::shared_ptr<const ParsedVerifiedInputs> p;auto ledger=budget();assert(ParsedVerifiedInputs::read(old.path,hash(old.text),bounds,ledger,p));VerifiedDriverOverrides extra;extra.paths["expert-profile"]={old.dir+"/profile.bin"};assert(!p->validate_overrides(extra));}
    auto profile_case=[&](std::string bytes,bool valid) {
        Fixture pf;write(pf.dir+"/profile.bin",bytes);
        pf.text=replace(pf.text,"MANIFEST\t1","MANIFEST\t2");
        pf.text+="FILE\tprofile\t"+std::to_string(bytes.size())+"\t"+hex(hash(bytes))+"\t"+pf.dir+"/profile.bin\nALIAS\tprofile\tprofiles/expert.bin\nBIND\texpert-profile\tprofiles/expert.bin\n";
        write(pf.path,pf.text);auto ledger=budget();std::shared_ptr<const ParsedVerifiedInputs> p;
        {auto small=budget(16*bounds.max_bytes+1024*bounds.max_records+4096);std::shared_ptr<const ParsedVerifiedInputs> untouched;auto no_room=ParsedVerifiedInputs::read(pf.path,hash(pf.text),bounds,small,untouched);assert(no_room.error==VerifiedArtifactError::budget_refused&&!untouched&&small.snapshot().reserved.host_bytes==0&&small.snapshot().reserved.io_slots==0);}
        assert(ParsedVerifiedInputs::read(pf.path,hash(pf.text),bounds,ledger,p));
        VerifiedDriverOverrides ov;ov.paths["expert-profile"]={pf.dir+"/profile.bin"};
        auto bad=ov;bad.paths["expert-profile"]={"/wrong/profile.bin"};int effects=0;auto opts=options();opts.fault=[&](const char*){++effects;return EIO;};
        std::shared_ptr<const VerifiedInputBundle> group;assert(!p->prepare_bundle(pf.dir,ledger,bad,group,opts)&&!group&&effects==0);
        assert(p->prepare_bundle(pf.dir,ledger,ov,group,options()));VerifiedDriverBindings binding;assert(p->bind(group,ov,binding)&&!binding.expert_profile_disabled);
        const auto prior=ledger.snapshot().reserved;
        for(int failure=0;failure<2;++failure){auto limited=budget(failure==0?1:32<<20,32<<20,failure==1?0:1);int reads=0;VerifiedArtifactOptions hook;hook.fault=[&](const char*){++reads;return 0;};auto result=binding.preflight_expert_profile(limited,hook);assert(result.error==VerifiedArtifactError::budget_refused&&reads==0&&limited.snapshot().reserved.host_bytes==0&&limited.snapshot().reserved.io_slots==0&&ledger.snapshot().reserved==prior);}
        auto checked=binding.preflight_expert_profile(ledger);assert(bool(checked)==valid&&ledger.snapshot().reserved.io_slots==0);
        if(valid){assert(ledger.snapshot().reserved==prior&&prior.host_bytes>=bytes.size()*4+4096);
            write(pf.dir+"/profile.bin","changed live source");
#ifdef STRATA_TEST_PROFILE_READ
            std::vector<std::pair<int32_t,int32_t>> ranked;int64_t slots=0;std::string error;
            assert(read_expert_profile(binding.expert_profile.loader_path(),2,4,ranked,slots,error)&&ranked.size()==2&&slots>=2);
#endif
        }else assert(ledger.snapshot().reserved==prior);
        assert(group->retire_checked()==VerifiedInputRetirement::busy_leases);binding={};assert(group->retire_checked()==VerifiedInputRetirement::retired);group.reset();p.reset();assert(ledger.snapshot().reserved.host_bytes==0);
    };
    auto profile_bytes=[](uint32_t layers,uint32_t experts,uint32_t want,uint32_t count){std::string data="STRP";for(uint32_t n:{1u,layers,experts,want,count})for(int i=0;i<4;++i)data+=char(n>>(8*i));return data;};
    auto padded=profile_bytes(2,4,3,2);padded+=std::string("\0\0\1\0\1\0\2\0",8);padded+=std::string(32,'\xff');profile_case(padded,true);
    profile_case(profile_bytes(2,4,3,2),false); // ranked payload truncated
    profile_case(profile_bytes(2,4,UINT32_MAX,UINT32_MAX),false);
    profile_case(profile_bytes(UINT32_MAX,UINT32_MAX,1,1),false);
    profile_case(profile_bytes(2,4,1,2)+std::string(8,'\0'),false);
#ifdef STRATA_TEST_PROFILE_READ
    {Fixture written;std::string error;std::vector<std::pair<int32_t,int32_t>> pairs{{0,1},{1,2}};assert(write_expert_profile(written.dir+"/writer.bin",2,4,pairs,error));std::ifstream input(written.dir+"/writer.bin",std::ios::binary);std::string actual((std::istreambuf_iterator<char>(input)),{});assert(actual.size()==24+8+32);profile_case(actual,true);}
#endif
    std::puts("verified_input_manifest_test passed");
}
