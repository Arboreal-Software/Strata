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
#include <unistd.h>
using namespace strata::core;
static int allocations=0,copies=0,fail_alloc=0,fail_copy=0;
static bool fail_register=false;
static std::map<void*,size_t> live;
static std::set<const void*> registered;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
extern "C" cudaError_t cudaMalloc(void** p,size_t n){if(++allocations==fail_alloc){*p=nullptr;return cudaErrorMemoryAllocation;}*p=std::malloc(n);live[*p]=n;return cudaSuccess;}
extern "C" cudaError_t cudaFree(void* p){if(!p)return cudaSuccess;check(live.erase(p)==1,"free belongs to one allocation");std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaMemcpy(void* d,const void* s,size_t n,cudaMemcpyKind){if(++copies==fail_copy)return cudaErrorUnknown;std::memcpy(d,s,n);return cudaSuccess;}
extern "C" cudaError_t cudaGetDevice(int* p){*p=0;return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){return cudaSuccess;}
extern "C" cudaError_t cudaGetLastError(){return cudaSuccess;}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected";}
namespace strata::kernels {
bool native_mmvq_supported(int t) noexcept{return t==8;}
size_t native_mmvq_weight_bytes(int,int a,int b){return size_t(a)*b/32*34;}
size_t native_q8_1_bytes(int a,int b){return size_t(a)*b/32*36;}
bool native_q8_0_packed_enabled(){return true;}
bool native_q8_0_packed_eligible(int,int){return true;}
void native_q8_0_pack_host(const void* src,void* dst,int a,int b){std::memcpy(dst,src,size_t(a)*b/32*34);}
void native_q8_0_packed_register(const void* a,const void*,int,int){registered.insert(a);if(fail_register)throw std::bad_alloc();}
void native_q8_0_packed_unregister(const void* a){registered.erase(a);}
void fused_gr_quantize_q8_0(const uint16_t*,uint8_t*,int64_t,void*){throw std::runtime_error("unexpected quantize branch");}
}
template<class T>static void put(std::vector<uint8_t>& b,T v){auto n=b.size();b.resize(n+sizeof v);std::memcpy(b.data()+n,&v,sizeof v);}
static void str(std::vector<uint8_t>& b,const std::string& s){put<uint64_t>(b,s.size());b.insert(b.end(),s.begin(),s.end());}
struct FixtureDirectory {
    char directory[64] = "/tmp/strata-native-load-XXXXXX";
    FixtureDirectory(){if(!::mkdtemp(directory))throw std::runtime_error("create private fixture directory");}
    FixtureDirectory(const FixtureDirectory&)=delete;
    FixtureDirectory& operator=(const FixtureDirectory&)=delete;
    ~FixtureDirectory(){
        // Only remove this guard's known file; never recursively follow entries.
        char file[96];std::snprintf(file,sizeof file,"%s/model.gguf",directory);
        std::remove(file);::rmdir(directory);
    }
};
static std::string fixture(const FixtureDirectory& directory){
    std::vector<uint8_t> b;put<uint32_t>(b,0x46554747);put<uint32_t>(b,3);put<uint64_t>(b,2);put<uint64_t>(b,7);
    str(b,"general.architecture");put<uint32_t>(b,8);str(b,"qwen4exp");
    const auto& g=kCompiledGeometry;
    for(const auto& [key,value]:std::vector<std::pair<std::string,int64_t>>{{"block_count",g.n_layers},{"embedding_length",g.n_embd},{"expert_count",g.n_expert},{"expert_used_count",g.n_expert_used},{"attention.head_count",g.n_head},{"attention.head_count_kv",g.n_head_kv}}){str(b,"qwen4exp."+key);put<uint32_t>(b,10);put<uint64_t>(b,value);}
    uint64_t offset=0;
    for(const auto* name:{"blk.0.attn_q.weight","blk.0.hc_attn_down.weight"}){str(b,name);put<uint32_t>(b,2);put<uint64_t>(b,32);put<uint64_t>(b,32);put<uint32_t>(b,8);put<uint64_t>(b,offset);offset+=1088;}
    b.resize((b.size()+31)/32*32+offset,0);
    const std::string path=std::string(directory.directory)+"/model.gguf";
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("open fixture file");
    out.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));
    if(!out)throw std::runtime_error("write fixture file");
    out.close();if(!out)throw std::runtime_error("close fixture file");
    return path;
}
int main() try {
    setenv("STRATA_HC_Q8","1",1);FixtureDirectory directory;const auto path=fixture(directory);
    // Simultaneously live fixtures cannot overwrite or remove one another.
    {FixtureDirectory other;const auto second=fixture(other);
        check(path!=second,"overlapping fixture paths are distinct");
        std::ifstream first(path,std::ios::binary),next(second,std::ios::binary);
        check(first.good()&&next.good(),"both overlapping fixtures remain readable");}
    check(std::ifstream(path,std::ios::binary).good(),"other fixture cleanup preserves original");
    for(int scenario=std::getenv("STRATA_NATIVE_TEST_FAULTS_ONLY")?1:0;scenario<9;++scenario){
        allocations=copies=0;fail_alloc=scenario>=1&&scenario<=4?scenario:0;
        fail_copy=scenario>=5&&scenario<=7?scenario-4:0;fail_register=scenario==8;
        WeightTable table;float canonical=0;auto& refs=const_cast<std::map<std::string,WeightRef>&>(table.all());
        WeightRef q;q.ne0=q.ne1=32;q.elements=1024;q.code_bits=8;q.data=&canonical;refs["blk.0.attn_q.weight"]=q;
        q.code_bits=0;q.kind=WeightKind::Bf16InF32;q.bytes=2048;refs["blk.0.hc_attn_down.weight"]=q;
        {
            NativeDense dense;check(dense.ready_for_adoption(),"untouched canonical-only native owner is admissible");std::string error;const bool loaded=dense.load({path},table,error);
            if(scenario==0){check(loaded,error.c_str());check(dense.ready_for_adoption(),"complete native load admissible");
                const auto count=allocations;check(!dense.load({path},table,error)&&allocations==count&&dense.ready_for_adoption(),"duplicate load refuses without invalidating live model");check(allocations==4&&copies==3,"fixture exercised primary packed HC and scratch allocations");check(dense.allocated_bytes()==3300,"allocation estimate includes all four allocations");check(dense.owns_weight_span(refs.at("blk.0.attn_q.weight").native_data,1088),"complete load publishes owned native span");}
            else {check(!loaded,"injected loader failure propagated");check(!dense.ready_for_adoption(),"failed loader cannot be adopted");
                const auto count=allocations;check(!dense.load({path},table,error)&&allocations==count,"failed load cannot retry into partial state");check(!dense.owns_weight_span(refs.at("blk.0.attn_q.weight").native_data,1),"failed load publishes no valid native span");}
        }
        check(live.empty()&&registered.empty(),"failure/success teardown retains then releases every allocation and registration");
    }
    std::puts("native_dense_load_test passed");
} catch(const std::exception& error){
    std::fprintf(stderr,"FAIL: fixture setup: %s\n",error.what());return 1;
}
