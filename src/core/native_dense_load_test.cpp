#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <map>
#include <set>
#include <vector>
#include <stdexcept>
#include <unistd.h>
using namespace strata::core;
static int allocations=0,copies=0,fail_alloc=0,fail_copy=0;
static bool fail_register=false,register_fault_hit=false;
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
// Public CPU target links no CUDA runtime. Unused canonical staging is refused;
// event tokens model ordering calls only and do not represent device completion.
extern "C" cudaError_t cudaSetDevice(int device){return device==0?cudaSuccess:cudaErrorInvalidDevice;}
extern "C" cudaError_t cudaHostAlloc(void** p,size_t,unsigned){*p=nullptr;return cudaErrorNotSupported;}
extern "C" cudaError_t cudaFreeHost(void* p){check(p==nullptr,"canonical staging unsupported by fixture");return cudaSuccess;}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event,unsigned){
    static uintptr_t next=1;*event=reinterpret_cast<cudaEvent_t>(next++);return cudaSuccess;
}
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t,cudaEvent_t event,unsigned){return event?cudaSuccess:cudaErrorInvalidResourceHandle;}
extern "C" cudaError_t cudaEventRecord(cudaEvent_t event,cudaStream_t){return event?cudaSuccess:cudaErrorInvalidResourceHandle;}
namespace strata::kernels {
bool native_mmvq_supported(int t) noexcept{return t==8;}
size_t native_mmvq_weight_bytes(int,int a,int b){return size_t(a)*b/32*34;}
size_t native_q8_1_bytes(int a,int b){return size_t(a)*b/32*36;}
bool native_q8_0_packed_enabled(){return true;}
bool native_q8_0_packed_eligible(int,int){return true;}
void native_q8_0_pack_host(const void* src,void* dst,int a,int b){std::memcpy(dst,src,size_t(a)*b/32*34);}
void native_q8_0_packed_register(const void* a,const void*,int,int){registered.insert(a);if(fail_register){register_fault_hit=true;throw std::bad_alloc();}}
void native_q8_0_packed_unregister(const void* a){registered.erase(a);}
void fused_gr_quantize_q8_0(const uint16_t*,uint8_t*,int64_t,void*){throw std::runtime_error("unexpected quantize branch");}
void fused_gr_dequant_q8_0(const uint8_t*,uint16_t*,int64_t,void*){throw std::runtime_error("unexpected dequantize branch");}
}
template<class T>static void put(std::vector<uint8_t>& b,T v){auto n=b.size();b.resize(n+sizeof v);std::memcpy(b.data()+n,&v,sizeof v);}
static void str(std::vector<uint8_t>& b,const std::string& s){put<uint64_t>(b,s.size());b.insert(b.end(),s.begin(),s.end());}
class TempGguf {
    char path_[sizeof("/tmp/strata-native-load-XXXXXX")]="/tmp/strata-native-load-XXXXXX";
    int fd_=-1;
public:
    TempGguf(){fd_=::mkstemp(path_);if(fd_<0)throw std::runtime_error("create exclusive fixture");}
    TempGguf(const TempGguf&)=delete;
    TempGguf& operator=(const TempGguf&)=delete;
    ~TempGguf(){if(fd_>=0)(void)::close(fd_);(void)::unlink(path_);}
    const char* path()const{return path_;}
    void write_and_close(const std::vector<uint8_t>& bytes){
        size_t at=0;
        while(at<bytes.size()){
            const auto n=::write(fd_,bytes.data()+at,bytes.size()-at);
            if(n<0&&errno==EINTR)continue;
            if(n<=0)throw std::runtime_error("write fixture");
            at+=size_t(n);
        }
        const int fd=fd_;fd_=-1; // close may consume descriptor even on error: never retry
        if(::close(fd)!=0)throw std::runtime_error("close fixture");
    }
};
static void fixture(TempGguf& file){
    std::vector<uint8_t> b;put<uint32_t>(b,0x46554747);put<uint32_t>(b,3);put<uint64_t>(b,2);put<uint64_t>(b,7);
    str(b,"general.architecture");put<uint32_t>(b,8);str(b,"qwen4exp");
    const auto& g=kCompiledGeometry;
    for(const auto& [key,value]:std::vector<std::pair<std::string,int64_t>>{{"block_count",g.n_layers},{"embedding_length",g.n_embd},{"expert_count",g.n_expert},{"expert_used_count",g.n_expert_used},{"attention.head_count",g.n_head},{"attention.head_count_kv",g.n_head_kv}}){str(b,"qwen4exp."+key);put<uint32_t>(b,10);put<uint64_t>(b,value);}
    uint64_t offset=0;
    for(const auto* name:{"blk.0.attn_q.weight","blk.0.hc_attn_down.weight"}){str(b,name);put<uint32_t>(b,2);put<uint64_t>(b,32);put<uint64_t>(b,32);put<uint32_t>(b,8);put<uint64_t>(b,offset);offset+=1088;}
    b.resize((b.size()+31)/32*32+offset,0);
    file.write_and_close(b);
}
int main(int argc,char** argv) try {
    // Main assertions unwind fixture ownership; fatal CUDA mock invariants retain
    // their existing process-failure behavior and cannot promise cleanup.
    const auto check=[](bool ok,const char* why){if(!ok)throw std::runtime_error(why);};
    setenv("STRATA_HC_Q8","1",1);TempGguf input;fixture(input);const std::string path=input.path();
    if(argc==2&&(std::string(argv[1])=="--fixture-ready"||std::string(argv[1])=="--fixture-throw")){
        check(std::puts(path.c_str())>=0&&std::fflush(stdout)==0,"publish fixture path");
        char byte=0;ssize_t n;do{n=::read(STDIN_FILENO,&byte,1);}while(n<0&&errno==EINTR);
        check(n==1,"fixture barrier");
        if(std::string(argv[1])=="--fixture-throw")throw std::runtime_error("injected fixture unwind");
    }
    for(int scenario=std::getenv("STRATA_NATIVE_TEST_FAULTS_ONLY")?1:0;scenario<9;++scenario){
        allocations=copies=0;fail_alloc=scenario>=1&&scenario<=4?scenario:0;
        fail_copy=scenario>=5&&scenario<=7?scenario-4:0;fail_register=scenario==8;register_fault_hit=false;
        WeightTable table;float canonical=0;auto& refs=const_cast<std::map<std::string,WeightRef>&>(table.all());
        WeightRef q;q.ne0=q.ne1=32;q.elements=1024;q.code_bits=8;q.data=&canonical;refs["blk.0.attn_q.weight"]=q;
        q.code_bits=0;q.kind=WeightKind::Bf16InF32;q.bytes=2048;refs["blk.0.hc_attn_down.weight"]=q;
        {
            NativeDense dense;check(dense.ready_for_adoption(),"untouched canonical-only native owner is admissible");std::string error;const bool loaded=dense.load({path},table,error);
            // Check reachability before duplicate-load refusal can obscure which
            // failure the first load actually encountered.
            if(fail_alloc)check(allocations==fail_alloc,"requested allocation fault was reached");
            if(fail_copy)check(copies==fail_copy,"requested copy fault was reached");
            if(scenario==8)check(register_fault_hit,"requested registration fault was reached");
            if(scenario==0){check(loaded,error.c_str());check(dense.ready_for_adoption(),"complete native load admissible");
                const auto count=allocations;check(!dense.load({path},table,error)&&allocations==count&&dense.ready_for_adoption(),"duplicate load refuses without invalidating live model");check(allocations==4&&copies==3,"fixture exercised primary packed HC and scratch allocations");check(dense.allocated_bytes()==3300,"allocation estimate includes all four allocations");check(dense.owns_weight_span(refs.at("blk.0.attn_q.weight").native_data,1088),"complete load publishes owned native span");}
            else {check(!loaded,"injected loader failure propagated");check(!dense.ready_for_adoption(),"failed loader cannot be adopted");
                const auto count=allocations;check(!dense.load({path},table,error)&&allocations==count,"failed load cannot retry into partial state");check(!dense.owns_weight_span(refs.at("blk.0.attn_q.weight").native_data,1),"failed load publishes no valid native span");}
        }
        check(live.empty()&&registered.empty(),"failure/success teardown retains then releases every allocation and registration");
    }
    std::puts("native_dense_load_test passed");
} catch(const std::exception& error){
    std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;
}
