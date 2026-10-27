#include "strata/core/native_dense.hpp"
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
static std::map<void*,size_t> live;
static std::set<const void*> registered;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
extern "C" cudaError_t cudaMalloc(void** p,size_t n){if(++allocations==fail_alloc){*p=nullptr;return cudaErrorMemoryAllocation;}*p=std::malloc(n);live[*p]=n;return cudaSuccess;}
extern "C" cudaError_t cudaFree(void* p){if(!p)return cudaSuccess;if(fail_free)return cudaErrorUnknown;check(live.erase(p)==1,"free belongs to one allocation");std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaMemcpy(void* d,const void* s,size_t n,cudaMemcpyKind){if(++copies==fail_copy)return cudaErrorUnknown;std::memcpy(d,s,n);return cudaSuccess;}
extern "C" cudaError_t cudaGetDevice(int* p){*p=0;return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){return cudaSuccess;}
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

struct Probe {bool* destroyed;explicit Probe(bool* p):destroyed(p){}~Probe(){check(live.empty(),"input owner releases after GPU allocations");*destroyed=true;}};
static void refs(WeightTable& table,bool inject=false,int fallback=0) {
    static float canonical[1024]{};auto& r=const_cast<std::map<std::string,WeightRef>&>(table.all());
    WeightRef q;q.ne0=q.ne1=32;q.elements=1024;q.code_bits=8;q.data=canonical;r["blk.0.attn_q.weight"]=q;
    q.code_bits=0;q.kind=WeightKind::Bf16InF32;q.bytes=2048;r[inject?"blk.0.hc_attn_inject.weight":"blk.0.hc_attn_down.weight"]=q;
    if(fallback){q.file_id=3;q.src_off=1234;q.src_bytes=2048;if(fallback==2){q.data=nullptr;q.resident=false;}r["blk.0.hc_ffn_down.weight"]=q;}
}
static std::shared_ptr<const ReservationLease> reserve(WindowBudget& b,uint64_t n){return b.reserve({{0},n,0,0,0,0},ReservationPurpose::foreground).lease;}
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
    std::remove(path.c_str());std::remove(empty.c_str());std::remove(injected.c_str());std::puts("native_load_inventory_test passed");
}
