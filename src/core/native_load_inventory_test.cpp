#include "strata/core/native_dense.hpp"
#include "strata/core/verified_input_manifest.hpp"
#include <filesystem>
#include <iterator>
#include "strata/core/weights.hpp"
#include "strata/core/geometry.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <stdexcept>
#include "strata/artifact/gguf_reader.hpp"
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
static int allocations=0,copies=0,fail_alloc=0,fail_copy=0;
static bool fail_register=false,fail_free=false;
static int gpu_quantized=0;
static int current_device=0,expected_device=0,drains=0,drain_marker=-1;
static bool fail_drain=false,fail_conversion=false;
static WindowBudget* observed_budget=nullptr;
static uint64_t observed_charge=0,observed_conversion_bytes=0;
static int conversion_allocations=0,conversion_live=0;
static std::weak_ptr<const void> observed_input;
static void* conversion_pointer=nullptr;
static void observe_conversion_payment(){
    if(observed_budget && (observed_budget->snapshot().reserved.host_bytes!=observed_charge||observed_input.expired()))std::abort();
}
void* operator new(size_t n){
    if(observed_budget&&(n==1088||n==2176)&&n<=observed_conversion_bytes){
        observe_conversion_payment();++conversion_allocations;
        if(fail_conversion)throw std::bad_alloc();
    }
    void* p=std::malloc(n?n:1);if(!p)throw std::bad_alloc();
    if(observed_budget&&(n==1088||n==2176)&&n<=observed_conversion_bytes){conversion_pointer=p;++conversion_live;}
    return p;
}
void operator delete(void* p) noexcept {
    if(p&&p==conversion_pointer){observe_conversion_payment();if(fail_drain)std::abort();--conversion_live;conversion_pointer=nullptr;}
    std::free(p);
}
void operator delete(void* p,size_t) noexcept {::operator delete(p);}
// Link this existing recording test with --wrap=malloc,--wrap=free. These
// wrappers observe the actual single arena request/free, never a fake parser.
static WindowBudget* preflight_budget=nullptr;
static uint64_t preflight_quote=0,preflight_base=0;
static int preflight_mallocs=0,preflight_frees=0;
static bool preflight_fail_malloc=false;
static void* preflight_pointer=nullptr;
extern "C" void* __real_malloc(size_t) __attribute__((weak));
extern "C" void __real_free(void*) __attribute__((weak));
extern "C" void* __wrap_malloc(size_t n){
    const bool observed=preflight_budget&&uint64_t(n)==preflight_quote;
    if(observed){
        ++preflight_mallocs;
        if(preflight_budget->snapshot().reserved.host_bytes!=preflight_base+preflight_quote)std::abort();
        if(preflight_fail_malloc)return nullptr;
    }
    void* p=__real_malloc?__real_malloc(n):std::malloc(n);
    if(observed)preflight_pointer=p;
    return p;
}
extern "C" void __wrap_free(void* p){
    if(p&&p==preflight_pointer){
        if(!preflight_budget||preflight_budget->snapshot().reserved.host_bytes!=preflight_base+preflight_quote)std::abort();
        ++preflight_frees;preflight_pointer=nullptr;
    }
    if(__real_free)__real_free(p);else std::free(p);
}

static std::map<void*,size_t> live;
static std::set<const void*> registered;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
extern "C" cudaError_t cudaMalloc(void** p,size_t n){observe_conversion_payment();if(observed_budget&&current_device!=expected_device)std::abort();if(++allocations==fail_alloc){*p=nullptr;return cudaErrorMemoryAllocation;}*p=std::malloc(n);live[*p]=n;return cudaSuccess;}
extern "C" cudaError_t cudaFree(void* p){if(!p)return cudaSuccess;if(fail_free)return cudaErrorUnknown;check(live.erase(p)==1,"free belongs to one allocation");std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaMemcpy(void* d,const void* s,size_t n,cudaMemcpyKind){if(++copies==fail_copy)return cudaErrorUnknown;std::memcpy(d,s,n);return cudaSuccess;}
extern "C" cudaError_t cudaGetDevice(int* p){*p=current_device;return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){
    observe_conversion_payment();++drains;
    if(fail_drain){if(conversion_live!=1||current_device!=expected_device)std::abort();if(drain_marker>=0)(void)::write(drain_marker,"D",1);return cudaErrorUnknown;}
    return cudaSuccess;
}
extern "C" cudaError_t cudaGetLastError(){return cudaSuccess;}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected";}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* p,unsigned){*p=reinterpret_cast<cudaEvent_t>(uintptr_t(1));return cudaSuccess;}
namespace strata::platform {void advise_willneed(const void*,uint64_t){}}
namespace strata::core {
WeightTable::~WeightTable()=default;
bool WeightTable::read_bf16_host(const std::string&,std::vector<uint16_t>& out,std::string&)const {out.assign(1024,0);return true;} // canonical producer intentionally unresolved
}
namespace strata::kernels {
bool native_mmvq_supported(int t) noexcept{return t==8;}
size_t native_mmvq_weight_bytes(int,int a,int b){return size_t(a)*b/32*34;}
size_t native_q8_1_bytes(int a,int b){return size_t(a)*b/32*36;}
bool native_q8_0_packed_enabled(){return true;}
bool native_q8_0_packed_eligible(int,int){return true;}
void native_q8_0_pack_host(const void* src,void* dst,int a,int b){std::memcpy(dst,src,size_t(a)*b/32*34);}
void native_q8_0_packed_register(const void* a,const void*,int,int){registered.insert(a);if(fail_register)throw std::bad_alloc();}
void native_q8_0_packed_unregister(const void* a){registered.erase(a);}
void fused_gr_quantize_q8_0(const uint16_t*,uint8_t* p,int64_t n,void*){++gpu_quantized;std::memset(p,0,size_t(n)*34);}
}
template<class T>static void put(std::vector<uint8_t>& b,T v){auto n=b.size();b.resize(n+sizeof v);std::memcpy(b.data()+n,&v,sizeof v);}
static void str(std::vector<uint8_t>& b,const std::string& s){put<uint64_t>(b,s.size());b.insert(b.end(),s.begin(),s.end());}
static std::string fixture(bool empty=false,bool inject=false){
    std::vector<uint8_t> b;put<uint32_t>(b,0x46554747);put<uint32_t>(b,3);put<uint64_t>(b,empty?0:2);put<uint64_t>(b,7);
    str(b,"general.architecture");put<uint32_t>(b,8);str(b,"qwen4exp");
    const auto& g=kCompiledGeometry;
    for(const auto& [key,value]:std::vector<std::pair<std::string,int64_t>>{{"block_count",g.n_layers},{"embedding_length",g.n_embd},{"expert_count",g.n_expert},{"expert_used_count",g.n_expert_used},{"attention.head_count",g.n_head},{"attention.head_count_kv",g.n_head_kv}}){str(b,"qwen4exp."+key);put<uint32_t>(b,10);put<uint64_t>(b,value);}
    uint64_t offset=0;
    if(!empty)for(const auto* name:{"blk.0.attn_q.weight",inject?"blk.0.hc_attn_inject.weight":"blk.0.hc_attn_down.weight"}){const bool f32=inject&&std::string(name).find("inject")!=std::string::npos;str(b,name);put<uint32_t>(b,2);put<uint64_t>(b,32);put<uint64_t>(b,32);put<uint32_t>(b,f32?0:8);put<uint64_t>(b,offset);offset+=f32?4096:1088;}
    b.resize((b.size()+31)/32*32+offset,0);
    const std::string path="/tmp/strata_native_inventory_"+std::to_string(getpid())+(empty?"_empty":inject?"_inject":"_full")+".gguf";std::ofstream(path,std::ios::binary).write(reinterpret_cast<const char*>(b.data()),b.size());return path;
}

static std::string bf16_fixture(){
    std::vector<uint8_t> b;put<uint32_t>(b,0x46554747);put<uint32_t>(b,3);put<uint64_t>(b,2);put<uint64_t>(b,7);
    str(b,"general.architecture");put<uint32_t>(b,8);str(b,"qwen4exp");
    const auto& g=kCompiledGeometry;
    for(const auto& [key,value]:std::vector<std::pair<std::string,int64_t>>{{"block_count",g.n_layers},{"embedding_length",g.n_embd},{"expert_count",g.n_expert},{"expert_used_count",g.n_expert_used},{"attention.head_count",g.n_head},{"attention.head_count_kv",g.n_head_kv}}){str(b,"qwen4exp."+key);put<uint32_t>(b,10);put<uint64_t>(b,value);}
    str(b,"blk.0.attn_q.weight");put<uint32_t>(b,2);put<uint64_t>(b,32);put<uint64_t>(b,32);put<uint32_t>(b,30);put<uint64_t>(b,0);
    str(b,"blk.1.attn_k.weight");put<uint32_t>(b,2);put<uint64_t>(b,64);put<uint64_t>(b,32);put<uint32_t>(b,30);put<uint64_t>(b,2048);
    b.resize((b.size()+31)/32*32);
    for(int n=0;n<3072;++n)put<uint16_t>(b,n%2?0xbf80:0x3f80); // exact alternating -1/+1
    const std::string path="/tmp/strata_bf16_inventory_"+std::to_string(getpid())+".gguf";
    std::ofstream(path,std::ios::binary).write(reinterpret_cast<const char*>(b.data()),b.size());return path;
}
static void bf16_refs(WeightTable& table){
    static float canonical[2048]{};
    auto& refs=const_cast<std::map<std::string,WeightRef>&>(table.all());
    WeightRef q;q.ne0=q.ne1=32;q.elements=1024;q.kind=WeightKind::Bf16InF32;q.bytes=2048;q.data=canonical;
    refs["blk.0.attn_q.weight"]=q;
    q.ne0=64;q.elements=2048;q.bytes=4096;refs["blk.1.attn_k.weight"]=q;
}
static void check_bf16_blocks(const void* p,size_t bytes){
    const auto* q=static_cast<const uint8_t*>(p);
    for(size_t b=0;b<bytes;b+=34){
        check(q[b]==0x08&&q[b+1]==0x20,"actual BF16 Q8 half scale is rounded 1/127");
        for(size_t j=0;j<32;++j)check(q[b+2+j]==(j%2?uint8_t(129):uint8_t(127)),"actual BF16 Q8 signed values match literal rounding oracle");
    }
}

struct Probe {bool* destroyed;explicit Probe(bool* p):destroyed(p){}~Probe(){check(live.empty(),"input owner releases after GPU allocations");*destroyed=true;}};
static void refs(WeightTable& table,bool inject=false,int fallback=0) {
    static float canonical[1024]{};auto& r=const_cast<std::map<std::string,WeightRef>&>(table.all());
    WeightRef q;q.ne0=q.ne1=32;q.elements=1024;q.code_bits=8;q.data=canonical;r["blk.0.attn_q.weight"]=q;
    q.code_bits=0;q.kind=WeightKind::Bf16InF32;q.bytes=2048;r[inject?"blk.0.hc_attn_inject.weight":"blk.0.hc_attn_down.weight"]=q;
    if(fallback){q.file_id=3;q.src_off=1234;q.src_bytes=2048;if(fallback==2){q.data=nullptr;q.resident=false;}r["blk.0.hc_ffn_down.weight"]=q;}
}
static std::shared_ptr<const ReservationLease> reserve(WindowBudget& b,uint64_t n){return b.reserve({{0},n,0,0,0,0},ReservationPurpose::foreground).lease;}
static StateDigest preflight_hash(const std::string& bytes){WindowHasher h;h.update(bytes.data(),bytes.size());return h.digest();}
static std::string preflight_hex(const StateDigest& digest){
    std::string out;constexpr char digits[]="0123456789abcdef";
    for(auto b:digest){out+=digits[b>>4];out+=digits[b&15];}return out;
}
static std::string preflight_read(const std::string& path){
    std::ifstream input(path,std::ios::binary);return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
static void preflight_write(const std::string& path,const std::string& bytes){
    std::ofstream out(path,std::ios::binary);out.write(bytes.data(),std::streamsize(bytes.size()));out.close();check(bool(out),"preflight source fixture written");
}
static std::string preflight_split(const std::string& source,unsigned shard){
    strata::GgufFile original(source);const auto raw=preflight_read(source);
    strata::Cursor cursor(reinterpret_cast<const uint8_t*>(raw.data()),raw.size());
    (void)cursor.read<uint32_t>();(void)cursor.read<uint32_t>();
    const auto tensors=cursor.read<uint64_t>(),metadata=cursor.read<uint64_t>();
    for(uint64_t i=0;i<metadata;++i){(void)cursor.str();(void)strata::read_value(cursor,strata::MetaType(cursor.read<uint32_t>()));}
    std::vector<uint8_t> b;put<uint32_t>(b,0x46554747);put<uint32_t>(b,3);put<uint64_t>(b,tensors);put<uint64_t>(b,metadata+3);
    b.insert(b.end(),raw.begin()+24,raw.begin()+std::ptrdiff_t(cursor.pos()));
    for(const auto& item:std::array<std::pair<const char*,uint32_t>,3>{{{"split.count",2},{"split.no",shard},{"split.tensors.count",2}}}){
        str(b,item.first);put<uint32_t>(b,4);put<uint32_t>(b,item.second);
    }
    for(const auto& tensor:original.tensors()){
        str(b,tensor.name);put<uint32_t>(b,uint32_t(tensor.shape.size()));for(auto n:tensor.shape)put<uint64_t>(b,n);
        put<uint32_t>(b,tensor.type);put<uint64_t>(b,tensor.offset);
    }
    b.resize((b.size()+31)/32*32);b.insert(b.end(),raw.begin()+std::ptrdiff_t(original.data_start()),raw.end());
    return {reinterpret_cast<const char*>(b.data()),b.size()};
}
struct PreflightInputFixture {
    std::string dir;
    WindowBudget budget;
    std::shared_ptr<const ParsedVerifiedInputs> plan;
    std::shared_ptr<const VerifiedInputBundle> bundle;
    std::shared_ptr<VerifiedDriverBindings> bindings;
    std::vector<std::string> paths;
    PreflightInputFixture(const std::vector<std::string>& files,bool driver_factory)
        :budget({{0},uint64_t(128)<<20,0,uint64_t(32)<<20,1,0},
                 {{0},NativeDense::inventory_preflight_setup_bytes(),0,0,0,0}) {
        char name[]="/tmp/strata-inventory-preflight-XXXXXX";char* made=mkdtemp(name);
        check(made,"preflight private fixture directory");dir=made;
        if(driver_factory){
            std::string text="STRATA_VERIFIED_INPUT_MANIFEST\t1\n";
            for(const auto& item:std::vector<std::pair<std::string,uint64_t>>{{"max_files",32},{"max_aliases",64},
                {"max_path_bytes",256},{"max_depth",8},{"max_metadata_bytes",4096},{"view_host_bytes",8192},
                {"view_disk_bytes",8192},{"artifact_max_file_bytes",1<<20},{"artifact_block_bytes",4096},
                {"artifact_metadata_host_bytes",4096},{"artifact_disk_overhead_bytes",4096},{"allow_full_copy",1}})
                text+="LIMIT\t"+item.first+"\t"+std::to_string(item.second)+"\n";
            std::vector<std::pair<std::string,std::string>> all;
            for(size_t i=0;i<files.size();++i)all.push_back({"model"+std::to_string(i)+".gguf",files[i]});
            all.push_back({"index.txt","# align 16 pool 16 tensors 1\nw 0 2 0 4 0 4 1 1 0 0 0 0 0 4 0 0 0 0\n"});
            all.push_back({"dense.bin","data"});
            for(const auto& item:all){
                preflight_write(dir+"/"+item.first,item.second);
                text+="FILE\t"+item.first+"\t"+std::to_string(item.second.size())+"\t"+preflight_hex(preflight_hash(item.second))+"\t"+dir+"/"+item.first+"\n";
                text+="ALIAS\t"+item.first+"\t"+(item.first.starts_with("model")?"models/":"pack/")+item.first+"\n";
            }
            text+="ALIAS\tmodel0.gguf\tmodels/other-spelling.gguf\n";
            for(size_t i=0;i<files.size();++i)text+="GGUF\tmain\t"+std::to_string(i)+"\tmodels/model"+std::to_string(i)+".gguf\n";
            text+="PACK\tpack\tmodels/model0.gguf\n";
            for(const auto& item:std::vector<std::pair<std::string,std::string>>{{"pack","pack"},{"native-primary","models/model0.gguf"},
                {"native-group","main"},{"dense-group","main"},{"head-primary","-"},{"head-group","-"},{"embedding","-"},{"ple","-"}})
                text+="BIND\t"+item.first+"\t"+item.second+"\n";
            preflight_write(dir+"/manifest.tsv",text);
            check(bool(ParsedVerifiedInputs::read(dir+"/manifest.tsv",preflight_hash(text),{65536,256,4096,4096},budget,plan)),
                  "actual manifest parser publishes genuine input plan");
            check(bool(plan->prepare_bundle(dir,budget,{},bundle)),"actual verified plan prepares immutable bundle");
            bindings=std::make_shared<VerifiedDriverBindings>();
            check(bool(plan->bind(bundle,{},*bindings)),"actual binding factory publishes positive ordered dense role");
        }else{
            // Malformed-header safety cases deliberately use genuine artifact
            // snapshots without claiming the successful driver GGUF validation.
            VerifiedInputManifest manifest;
            for(size_t i=0;i<files.size();++i){
                const auto file=dir+"/bad"+std::to_string(i);preflight_write(file,files[i]);
                manifest.files.push_back({file,files[i].size(),preflight_hash(files[i]),{"bad"+std::to_string(i)}});
            }
            VerifiedInputLimits limits{32,64,256,8,4096,8192,8192,{1<<20,4096,4096,4096,true}};
            check(bool(VerifiedInputBundle::prepare(manifest,dir,limits,budget,bundle)),"actual snapshot bundle authenticates malformed test bytes");
            bindings=std::make_shared<VerifiedDriverBindings>();
            for(size_t i=0;i<files.size();++i){VerifiedInputLease lease;check(bundle->resolve("bad"+std::to_string(i),lease),"actual malformed artifact lease resolved");bindings->dense.push_back(std::move(lease));}
        }
        paths=bindings->dense_paths();
    }
    ~PreflightInputFixture(){
        paths.clear();bindings.reset();plan.reset();
        check(bundle->retire_checked()==VerifiedInputRetirement::retired,"preflight input group retires after actual parser/result owners");bundle.reset();
        check(budget.snapshot().reserved.host_bytes==0,"all preflight/group reservations released; dedicated setup remains baseline only");
        std::filesystem::remove_all(dir);
    }
};
static std::string preflight_capture_report(const NativeInventoryPreflight& result){
    int receipt[2];check(pipe(receipt)==0,"parser diagnostic receipt pipe");
    const int saved=dup(STDERR_FILENO);check(saved>=0&&dup2(receipt[1],STDERR_FILENO)>=0,"parser diagnostic capture installed");
    close(receipt[1]);result.report_error("");
    check(dup2(saved,STDERR_FILENO)>=0,"parser diagnostic output restored");close(saved);
    std::string text;char bytes[512];ssize_t n;
    while((n=read(receipt[0],bytes,sizeof bytes))>0)text.append(bytes,size_t(n));
    close(receipt[0]);return text;
}
static void paid_preflight_fixture(const std::string& bf16,const std::string& empty){
    check(__real_malloc&&__real_free,"actual arena allocator recording requires malloc/free link wrapping");
    NativeDense::set_layer_range(-1,-1);
    const auto first=preflight_split(bf16,0),second=preflight_split(empty,1);
    PreflightInputFixture input({first,second},true);WeightTable table;bf16_refs(table);
    const NativeInventoryLimits initial{2,4,2,256};
    const auto baseline=input.budget.snapshot().reserved.host_bytes;
    uint64_t request=0;
    {
        auto parsed=NativeDense::inventory_conversion_limits_admitted(input.paths,table,false,initial,input.budget,input.bindings);
        check(bool(parsed)&&parsed.limits().max_bf16_q8_bytes==2176,"actual admitted immutable two-shard parser derives exact BF16 maximum");
        request=parsed.requested_bytes();
        check(request>0&&parsed.allocation_attempts()==1&&parsed.parser_constructions()==2&&parsed.workspace_bytes()==request,
              "serial actual parsers reuse one maximum arena and retain the real debit through result scope");
        check(input.budget.snapshot().reserved.host_bytes==baseline+request,"actual temporary workspace is a distinct live same-ledger charge");
        check(input.budget.snapshot().reserved.io_slots==1,"actual synchronous descriptor reads retain an existing-ledger IO admission");
    }
    check(input.budget.snapshot().reserved.host_bytes==baseline,"success releases parser storage/debit before final inventory admission");
    check(input.budget.snapshot().reserved.io_slots==0,"last result cleanup releases IO before subsequent startup work");
    for(int stage=0;stage<2;++stage){
        auto parsed=NativeDense::inventory_conversion_limits_admitted(input.paths,table,false,initial,input.budget,input.bindings,stage,stage+1);
        check(bool(parsed)&&parsed.limits().max_bf16_q8_bytes==uint64_t(stage?2176:1088),
              "actual admitted per-stage selection preserves exact smaller/larger conversion peaks");
    }
    auto arm=[&]{preflight_budget=&input.budget;preflight_quote=request;preflight_base=baseline;
        preflight_mallocs=preflight_frees=0;preflight_pointer=nullptr;preflight_fail_malloc=false;};
    arm();
    {
        auto parsed=NativeDense::inventory_conversion_limits_admitted(input.paths,table,false,initial,input.budget,input.bindings);
        check(bool(parsed)&&preflight_mallocs==1&&preflight_frees==0,"actual malloc happens only after charged admission");
        uint64_t inventory=0;std::string error;
        check(native_inventory_host_bound(parsed.limits(),inventory,error),"unchanged inventory quote validates scalar candidate separately");
    }
    check(preflight_frees==1&&preflight_pointer==nullptr&&input.budget.snapshot().reserved.host_bytes==baseline,
          "actual free observer sees workspace debit alive until the final arena free");
    preflight_budget=nullptr;
    {
        const auto snapshot=input.budget.snapshot();
        auto fill=reserve(input.budget,snapshot.capacity.host_bytes-snapshot.baseline.host_bytes-snapshot.reserved.host_bytes-(request-1));
        check(bool(fill),"real foreground filler leaves one byte below actual workspace quote");
        preflight_budget=&input.budget;preflight_quote=request;preflight_base=input.budget.snapshot().reserved.host_bytes;
        preflight_mallocs=preflight_frees=0;
        auto parsed=NativeDense::inventory_conversion_limits_admitted(input.paths,table,false,initial,input.budget,input.bindings);
        check(!parsed&&parsed.requested_bytes()==request&&parsed.allocation_attempts()==0&&parsed.parser_constructions()==0&&preflight_mallocs==0,
              "genuine ledger refuses before actual arena allocation/full parser construction");
        check(initial.max_bf16_q8_bytes==0&&initial.bf16_control_bytes==0,"refusal leaves supplied limits unchanged");
        preflight_budget=nullptr;
    }
    arm();preflight_fail_malloc=true;
    {
        auto parsed=NativeDense::inventory_conversion_limits_admitted(input.paths,table,false,initial,input.budget,input.bindings);
        check(!parsed&&parsed.allocation_attempts()==1&&parsed.parser_constructions()==0&&preflight_mallocs==1&&
              parsed.workspace_bytes()==request,"actual malloc failure retains the real debit through failure owner cleanup");
        parsed.report_error("native inventory controlled allocation failure: ");
    }
    preflight_fail_malloc=false;preflight_budget=nullptr;
    check(input.budget.snapshot().reserved.host_bytes==baseline,"actual allocation failure returns ledger to entry after result destruction");
    for(int mismatch=0;mismatch<3;++mismatch){
        auto paths=input.paths;
        if(mismatch==0)std::swap(paths[0],paths[1]);
        if(mismatch==1){VerifiedInputLease alias;check(input.bundle->resolve("models/other-spelling.gguf",alias),"same-source alternate spelling resolved");paths[0]=alias.loader_path();}
        if(mismatch==2)paths[0].insert(std::string("/proc/self/fd/").size(),"0");
        auto parsed=NativeDense::inventory_conversion_limits_admitted(paths,table,false,initial,input.budget,input.bindings);
        check(!parsed&&parsed.allocation_attempts()==0&&parsed.parser_constructions()==0,"ordered/exact-spelling authority rejects before parser allocation");
    }
    VerifiedInputLease nothing;
    check(nothing.matches_loader_path("")&&!nothing.matches_loader_path("x"),"empty matcher exactly preserves empty loader_path equality");
    for(size_t i=0;i<input.paths.size();++i)check(input.bindings->dense[i].matches_loader_path(input.paths[i]),"actual positive lease exact spelling matches without path construction");
    {
        PreflightInputFixture metadata({preflight_read(empty)},true);
        auto parsed=NativeDense::inventory_conversion_limits_admitted(metadata.paths,table,false,initial,metadata.budget,metadata.bindings);
        check(bool(parsed)&&parsed.requested_bytes()==0&&parsed.workspace_bytes()==0&&parsed.allocation_attempts()==0&&parsed.parser_constructions()==1,
              "genuine zero-tensor metadata-only input needs only prepaid fixed objects, no invented payload claim");
    }
    std::string duplicate=first;const std::string old="blk.1.attn_k.weight",same="blk.0.attn_q.weight";
    const auto at=duplicate.find(old);check(at!=std::string::npos&&old.size()==same.size(),"duplicate source mutation is exact name only");duplicate.replace(at,old.size(),same);
    std::string duplicate_bad_rank=duplicate;
    const auto repeated=duplicate_bad_rank.find(same,duplicate_bad_rank.find(same)+same.size());
    check(repeated!=std::string::npos,"second duplicate tensor located");
    const uint32_t zero_rank=0;std::memcpy(duplicate_bad_rank.data()+repeated+same.size(),&zero_rank,sizeof zero_rank);
    const std::string truncated=first.substr(0,27);
    std::vector<uint8_t> discarded;put<uint32_t>(discarded,0x46554747);put<uint32_t>(discarded,3);put<uint64_t>(discarded,0);put<uint64_t>(discarded,1);
    str(discarded,"x");put<uint32_t>(discarded,9);put<uint32_t>(discarded,8);put<uint64_t>(discarded,65);
    for(int i=0;i<64;++i)str(discarded,"a");
    put<uint64_t>(discarded,99);
    const std::string bad_discarded(reinterpret_cast<const char*>(discarded.data()),discarded.size());
    for(const auto& bad:std::array<std::string,4>{duplicate,duplicate_bad_rank,truncated,bad_discarded}){
        PreflightInputFixture malformed({bad},false);
        const auto held=malformed.budget.snapshot().reserved.host_bytes;
        auto ordinary=initial;std::string ordinary_error;
        check(!NativeDense::inventory_conversion_limits(malformed.paths,table,false,ordinary,ordinary_error),
              "unchanged ordinary parser supplies actual malformed-error reference");
        {
            auto parsed=NativeDense::inventory_conversion_limits_admitted(malformed.paths,table,false,initial,malformed.budget,malformed.bindings);
            check(!parsed&&initial.max_bf16_q8_bytes==0,"actual bounded parser rejects duplicate/truncated/discarded malformed bytes without changing input limits");
            check(parsed.parser_constructions()==0&&parsed.allocation_attempts()==0,
                  "actual malformed count pass allocates no parser payload");
            check(preflight_capture_report(parsed)==ordinary_error+"\n",
                  "actual admitted parser preserves ordinary malformed diagnostic and duplicate-before-rank precedence");
        }
        check(malformed.budget.snapshot().reserved.host_bytes==held,"actual malformed parser cleanup preserves entry ledger and genuine source owner");
    }
    {
        // The original GGUF validator accepts the directory; the actual BF16
        // consumer must reject its short payload while the arena debit is live.
        PreflightInputFixture short_payload({first.substr(0,first.size()-1),second},true);
        const auto held=short_payload.budget.snapshot().reserved.host_bytes;
        {
            auto parsed=NativeDense::inventory_conversion_limits_admitted(short_payload.paths,table,false,initial,
                short_payload.budget,short_payload.bindings);
            check(!parsed&&parsed.allocation_attempts()==1&&parsed.parser_constructions()==1&&parsed.workspace_bytes()>0&&
                  short_payload.budget.snapshot().reserved.host_bytes==held+parsed.workspace_bytes(),
                  "actual consumer truncation retains its separately admitted arena through failure reporting");
            parsed.report_error("native inventory controlled short payload: ");
        }
        check(short_payload.budget.snapshot().reserved.host_bytes==held,"actual consumer failure frees arena before refund and retains input authority");
    }
    std::puts("native inventory: genuine admitted preflight/path/constructor/allocation/free/failure boundary passed (recording allocator)");
}

int main(){
    setenv("STRATA_HC_Q8","1",1);setenv("STRATA_HC_Q8_INJECT","1",1);
    const auto path=fixture(),empty=fixture(true),injected=fixture(false,true);
    NativeInventoryLimits limits{2,4,2,256};uint64_t bytes=0;std::string error;
    check(native_inventory_host_bound(limits,bytes,error),"metadata bound");
    {auto invalid=limits;invalid.max_records=UINT64_MAX;uint64_t unchanged=7;check(!native_inventory_host_bound(invalid,unchanged,error)&&unchanged==7,"bad bounds preserve output");}
    for(int scenario=0;scenario<12;++scenario){
        allocations=copies=0;fail_alloc=scenario>=1&&scenario<=4?scenario:0;
        fail_copy=scenario>=5&&scenario<=7?scenario-4:0;fail_register=scenario==8;
        WindowBudget budget({{0},bytes+4096,0,0,0,0},{{0},0,0,0,0,0});bool destroyed=false;
        auto owner=std::make_shared<Probe>(&destroyed);std::weak_ptr<Probe> weak=owner;
        WeightTable table;refs(table,scenario==9,scenario==10?1:scenario==11?2:0);
        {
            NativeDense dense;check(!dense.loaded_inventory(),"no default receipt");
            auto tiny=reserve(budget,bytes-1);check(!dense.enable_inventory_capture(limits,tiny,owner,error)&&!dense.loaded_inventory(),"underadmission unchanged");tiny.reset();
            auto lease=reserve(budget,bytes);check(dense.enable_inventory_capture(limits,lease,owner,error),"admitted inventory enables");
            check(!dense.enable_inventory_capture(limits,lease,owner,error),"duplicate enable refuses");lease.reset();owner.reset();
            check(!destroyed&&!weak.expired(),"configured receipt retains actual input owner");
            const auto chosen=scenario==9?injected:path;const bool ok=dense.load({empty,chosen},table,error);
            if(scenario>=1&&scenario<=8)check(!ok&&!dense.loaded_inventory(),"failed actual load publishes no inventory");
            else {
                check(ok,error.c_str());const auto* inv=dense.loaded_inventory();check(inv&&inv->sources.size()==2&&inv->sources[1]==chosen,"actual selected second argument retained");
                check(inv->records.size()==size_t(scenario>=10?3:2),"actual source operations counted");
                const auto& native=inv->records[0];strata::GgufFile gguf(chosen);const auto& t=gguf.tensors()[0];
                check(native.source_ordinal==1&&native.source_offset==gguf.data_start()+t.offset&&native.source_bytes==1088&&native.source_type==8&&native.source_shape==std::vector<uint64_t>({32,32}),"exact loader source metadata retained");
                check(native.q8_packed_helper_bytes==1088&&native.runtime_bytes==1088,"packed helper recorded without fake tensor source");
                if(scenario==9)check(inv->records[1].source_type==0&&inv->records[1].source_bytes==4096&&inv->records[1].runtime_bytes==1088&&inv->records[1].conversion==NativeLoadConversion::f32_to_q8_0_host,"actual F32 to Q8 conversion distinct extents");
                if(scenario>=10){const auto& r=inv->records.back();check(inv->has_unresolved_canonical&&r.source==NativeLoadSource::unresolved_canonical&&r.canonical.file_id==3&&r.canonical.source_offset==1234&&r.canonical.source_bytes==2048,"canonical fallback explicitly unresolved");check(r.conversion==(scenario==10?NativeLoadConversion::bf16_to_q8_0_device:NativeLoadConversion::bf16_to_q8_0_host),"actual canonical CPU/GPU conversion recorded");}
                check(dense.release_checked(error)&&!dense.loaded_inventory(),"retirement revokes receipt");
            }
        }
        check(destroyed&&weak.expired()&&budget.snapshot().reserved.host_bytes==0&&live.empty()&&registered.empty(),"cleanup releases input and real reservation last");
    }
    fail_alloc=fail_copy=0;fail_register=false;
    // Disabled load remains the legacy path: no receipt/configuration required.
    {WeightTable table;refs(table);NativeDense dense;check(dense.load({path},table,error)&&!dense.loaded_inventory(),"default load has no inventory");}
    for(int kind=0;kind<3;++kind) {
        WindowBudget budget({{0},bytes,0,0,0,0},{{0},0,0,0,0,0});bool gone=false;
        auto owner=std::make_shared<Probe>(&gone);auto bounded=limits;
        if(kind==0)bounded.max_sources=1;
        if(kind==1)bounded.max_rank=1;
        if(kind==2)bounded.max_string_bytes=8;
        NativeDense dense;auto lease=reserve(budget,bytes);
        check(!dense.enable_inventory_capture(bounded,{},owner,error)&&!dense.enable_inventory_capture(bounded,lease,{},error),"missing reservation/owner refuses without configuring");
        check(dense.enable_inventory_capture(bounded,lease,owner,error),"bounded capture enabled after refusal");
        WeightTable table;refs(table);const int before=allocations;
        check(!dense.load({empty,path},table,error)&&!dense.loaded_inventory()&&allocations==before,"source/rank/string bound refuses before device allocations");
    }
    // A count-bound failure cannot publish even when earlier operations uploaded.
    {WindowBudget budget({{0},bytes,0,0,0,0},{{0},0,0,0,0,0});bool gone=false;auto owner=std::make_shared<Probe>(&gone);auto small=limits;small.max_records=1;NativeDense dense;check(dense.enable_inventory_capture(small,reserve(budget,bytes),owner,error),"bounded capture enabled");WeightTable table;refs(table);check(!dense.load({empty,path},table,error)&&!dense.loaded_inventory(),"record bound failure is not partial success");}
    // Separate direct destruction misuse from model-owner quarantine: both keep
    // the actual receipt/input/host charge when cleanup became uncertain.
    for(int direct=0;direct<3;++direct){pid_t pid=fork();check(pid>=0,"fork");if(pid==0){WindowBudget budget({{0},bytes,0,0,0,0},{{0},0,0,0,0,0});bool gone=false;auto owner=std::make_shared<Probe>(&gone);std::weak_ptr<Probe> weak=owner;auto* dense=new NativeDense;check(dense->enable_inventory_capture(limits,reserve(budget,bytes),owner,error),"uncertainty enabled");owner.reset();WeightTable table;refs(table);check(dense->load({empty,path},table,error),"uncertainty load");fail_free=true;if(direct!=2)check(!dense->release_checked(error)&&!dense->loaded_inventory(),"failed release revokes receipt");if(direct)delete dense;check(!gone&&!weak.expired()&&budget.snapshot().reserved.host_bytes==bytes,"uncertain direct/retained-object cleanup retains input and charge");std::_Exit(0);}int status=0;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"uncertainty lifetime test");}
    {pid_t pid=fork();check(pid>=0,"fork unwind failure");if(pid==0){WindowBudget budget({{0},bytes,0,0,0,0},{{0},0,0,0,0,0});bool gone=false;auto owner=std::make_shared<Probe>(&gone);std::weak_ptr<Probe> weak=owner;auto* dense=new NativeDense;check(dense->enable_inventory_capture(limits,reserve(budget,bytes),owner,error),"unwind capture enabled");owner.reset();WeightTable table;refs(table);fail_copy=copies+1;fail_free=true;check(!dense->load({empty,path},table,error)&&!dense->loaded_inventory(),"failed upload and uncertain local cleanup");fail_free=false;const auto retained=live.size();check(!dense->release_checked(error)&&live.size()==retained,"uncertain unwind cannot retry frees");delete dense;check(!gone&&!weak.expired()&&budget.snapshot().reserved.host_bytes==bytes&&live.size()==retained,"unwind uncertainty retains actual input and charge");std::_Exit(0);}int status=0;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"unwind uncertainty lifetime test");}

    // Actual BF16 projection, ordinary-supported and now admitted by the same
    // real inventory path. Only the new maximum/control delta is charged here;
    // original identity/canonical tests and all their assertions remain above.
    const auto bf16=bf16_fixture();
    for(int stage=0;stage<2;++stage){
        WeightTable table;bf16_refs(table);NativeDense::set_layer_range(stage?1:-1,stage?2:-1);
        auto bounded=limits;
        check(NativeDense::inventory_conversion_limits({bf16},table,false,bounded,error),"actual BF16 preflight");
        check(bounded.max_bf16_q8_bytes==2176,"maximum serial BF16 payload is actual converted bytes");
        uint64_t peak=0;check(native_inventory_host_bound(bounded,peak,error),"conversion peak quote");
        check(peak==bytes+2176+NativeDense::inventory_bf16_control_bytes()&&bounded.bf16_control_bytes==NativeDense::inventory_bf16_control_bytes(),"exact disjoint payload/control delta, not setup slack");
        {WindowBudget scarce({{0},peak-1,0,0,0,0},{{0},0,0,0,0,0});
         const int before=allocations;auto refused=reserve(scarce,peak);
         check(!refused&&scarce.snapshot().reserved.host_bytes==0&&allocations==before,
               "genuine WindowBudget peak refusal precedes conversion/device allocation and preserves ledger");}
        WindowBudget budget({{0},peak,0,0,0,0},{{0},0,0,0,0,0});bool gone=false;
        auto owner=std::make_shared<Probe>(&gone);NativeDense dense;
        const int before=allocations;
        auto tiny=reserve(budget,peak-1);
        check(!dense.enable_inventory_capture(bounded,tiny,owner,error)&&allocations==before&&!dense.loaded_inventory(),"one byte under peak refuses without device allocation or published state");tiny.reset();
        auto forged=bounded;--forged.bf16_control_bytes;
        auto admitted=reserve(budget,peak);
        check(!dense.enable_inventory_capture(forged,admitted,owner,error)&&!dense.loaded_inventory(),"actual control-type quote cannot be forged smaller");
        check(dense.enable_inventory_capture(bounded,std::move(admitted),owner,error),"exact foreground peak enables BF16");
        observed_input=owner;owner.reset();observed_budget=&budget;observed_charge=peak;
        observed_conversion_bytes=2176;conversion_allocations=conversion_live=0;
        current_device=expected_device=stage?3:0;const int before_drains=drains;
        check(dense.load({bf16},table,error),error.c_str());
        check(conversion_allocations==(stage?1:2)&&conversion_live==0&&drains-before_drains==(stage?1:2),"serial stage-filtered conversion drained before actual buffer free");
        const auto* inv=dense.loaded_inventory();check(inv&&inv->records.size()==size_t(stage?1:2),"only actual stage operations published");
        strata::GgufFile file(bf16);
        for(const auto& r:inv->records){
            const size_t ordinal=r.name=="blk.0.attn_q.weight"?0:1;const auto& t=file.tensors()[ordinal];
            check(r.source_ordinal==0&&r.source_type==30&&r.source_offset==file.data_start()+t.offset&&r.source_bytes==(ordinal?4096:2048)&&r.source_shape==t.shape,"actual BF16 source ordinal/range/type/shape receipt");
            check(r.runtime_layout==NativeLoadLayout::transformed_q8_0&&r.conversion==NativeLoadConversion::bf16_to_q8_0_host&&r.runtime_bytes==(ordinal?2176:1088),"actual Q8 runtime layout/bytes and BF16 host conversion receipt");
            int64_t runtime_type=-7;
            check(native_load_runtime_type(r,runtime_type)&&runtime_type==8&&r.source_type==30,
                  "actual NativeDense receipt traverses the collector's real production classifier, runtime8/source30");
            for(int malformed=0;malformed<10;++malformed){
                auto bad=r;
                switch(malformed){
                    case 0:bad.conversion=NativeLoadConversion::identity;break;
                    case 1:bad.source_type=8;break;
                    case 2:bad.source=NativeLoadSource::unresolved_canonical;break;
                    case 3:++bad.source_bytes;break;
                    case 4:++bad.runtime_bytes;break;
                    case 5:bad.runtime_layout=static_cast<NativeLoadLayout>(99);break;
                    case 6:bad.source_shape[0]=0;break;
                    case 7:bad.source_shape={UINT64_MAX,2};break;
                    case 8:bad.source_offset=UINT64_MAX;break;
                    case 9:bad.q8_packed_helper_bytes=34;break;
                }
                int64_t prior=123;
                check(!native_load_runtime_type(bad,prior)&&prior==123,"actual production BF16 classifier rejects malformed receipt without changing prior output");
            }
            check_bf16_blocks(table.all().at(r.name).native_data,size_t(r.runtime_bytes));
        }
        check(dense.release_checked(error),"BF16 owning-device retirement");
        observed_budget=nullptr;current_device=expected_device=0;
    }
    NativeDense::set_layer_range(0,1);
    {WeightTable table;bf16_refs(table);auto bounded=limits;check(NativeDense::inventory_conversion_limits({bf16},table,false,bounded,error)&&bounded.max_bf16_q8_bytes==1088,"primary filter exact smaller BF16 peak");}
    NativeDense::set_layer_range(-1,-1);
    for(int fault=0;fault<2;++fault){
        WeightTable table;bf16_refs(table);auto bounded=limits;
        check(NativeDense::inventory_conversion_limits({bf16},table,false,bounded,error),"fault actual preflight");
        uint64_t peak=0;check(native_inventory_host_bound(bounded,peak,error),"fault peak");
        WindowBudget budget({{0},peak,0,0,0,0},{{0},0,0,0,0,0});bool gone=false;auto owner=std::make_shared<Probe>(&gone);
        {NativeDense dense;check(dense.enable_inventory_capture(bounded,reserve(budget,peak),owner,error),"fault paid owner");
         observed_budget=&budget;observed_input=owner;observed_charge=peak;observed_conversion_bytes=1088;
         conversion_allocations=conversion_live=0;const int before=allocations;
         fail_conversion=fault==0;fail_copy=fault==1?copies+1:0;
         check(!dense.load({bf16},table,error)&&!dense.loaded_inventory(),"real BF16 allocation/copy fault publishes no inventory");
         check(conversion_allocations==1&&conversion_live==0&&(fault||allocations==before),"failed conversion allocates no device; failed copy drains then frees scratch");
         fail_conversion=false;fail_copy=0;observed_budget=nullptr;}
        check(live.empty()&&budget.snapshot().reserved.host_bytes==0,"BF16 fault releases exact admitted envelope after cleanup");
    }
    {int receipt[2];check(pipe(receipt)==0,"BF16 drain receipt");auto child=fork();check(child>=0,"BF16 drain child");
     if(!child){close(receipt[0]);WeightTable table;bf16_refs(table);auto bounded=limits;
      check(NativeDense::inventory_conversion_limits({bf16},table,false,bounded,error),"uncertain BF16 preflight");uint64_t peak=0;check(native_inventory_host_bound(bounded,peak,error),"uncertain peak");
      WindowBudget budget({{0},peak,0,0,0,0},{{0},0,0,0,0,0});auto owner=std::make_shared<int>(1);NativeDense dense;
      check(dense.enable_inventory_capture(bounded,reserve(budget,peak),owner,error),"uncertain paid conversion");
      observed_budget=&budget;observed_input=owner;observed_charge=peak;observed_conversion_bytes=1088;
      conversion_live=0;fail_drain=true;drain_marker=receipt[1];(void)dense.load({bf16},table,error);std::_Exit(42);}
     close(receipt[1]);int status=0;check(waitpid(child,&status,0)==child,"uncertain BF16 child reaped");char marker=0;
     check(read(receipt[0],&marker,1)==1&&marker=='D'&&WIFEXITED(status)&&WEXITSTATUS(status)==1,"actual uncertain owning-device drain fail-stops before host scratch or credit free");close(receipt[0]);}
    paid_preflight_fixture(bf16,empty);
    std::remove(bf16.c_str());
    std::remove(path.c_str());std::remove(empty.c_str());std::remove(injected.c_str());std::puts("native_load_inventory_test passed");
}
