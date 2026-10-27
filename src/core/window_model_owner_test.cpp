#include "strata/core/window_model_owner.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
static int device=0,drains=0,releases=0,frees=0;
static bool fail_drain=false,fail_native=false,fail_free=false;
static bool fail_upload=false;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
extern "C" cudaError_t cudaGetDevice(int* p){*p=device;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int n){device=n;return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(){++drains;return fail_drain?cudaErrorUnknown:cudaSuccess;}
extern "C" cudaError_t cudaHostAlloc(void** p,size_t n,unsigned int){*p=std::malloc(n);return *p?cudaSuccess:cudaErrorMemoryAllocation;}
extern "C" cudaError_t cudaFreeHost(void* p){std::free(p);return cudaSuccess;}
extern "C" cudaError_t cudaMemcpy(void* dst,const void* src,size_t n,cudaMemcpyKind){if(fail_upload)return cudaErrorUnknown;std::memcpy(dst,src,n);return cudaSuccess;}
namespace strata::platform {void advise_willneed(int,uint64_t,uint64_t){}}
extern "C" cudaError_t cudaFree(void* p){check(device==1,"arena release uses owning device");++frees;if(fail_free)return cudaErrorUnknown;std::free(p);return cudaSuccess;}
namespace strata::core {
struct NativeDenseReleaseTestAccess {static void fail_load(NativeDense& d){d.load_attempted_=true;}};
NativeDense::~NativeDense()=default;
NativeHead::~NativeHead()=default;
uint64_t NativeHead::allocated_bytes() const{return 0;}
bool NativeHead::release_checked(std::string&){return true;}
NativeEmbed::~NativeEmbed()=default;
bool NativeEmbed::release_checked(std::string&){return true;}
bool NativeDense::owns_weight_span(const void*,uint64_t) const{return false;}
bool NativeDense::release_checked(std::string&){check(device==1,"native release uses owning device");++releases;return !fail_native;}
}
static bool load_fixture(WeightTable& weights,void* arena,std::string& error) {
    char path[]="/tmp/strata-model-load-XXXXXX";
    check(mkdtemp(path)!=nullptr,"temporary canonical pack");
    const std::filesystem::path dir(path);
    {std::ofstream f(dir/"index.txt");f<<"# align 32 pool 32 tensors 1\nfixture.weight 0 2 0 32 0 32 8 1 0 0 0 0 0 0 0 0 0 0\n";}
    {std::ofstream f(dir/"dense.bin",std::ios::binary);float values[8]={1,2,3,4,5,6,7,8};f.write(reinterpret_cast<const char*>(values),sizeof values);}
    const bool loaded=weights.load(path,arena,32,error);
    std::filesystem::remove_all(dir);
    return loaded;
}
static ModelStageResources resources(WindowBudget& budget) {
    ModelStageResources r;r.device=1;r.arena=std::malloc(32);r.arena_bytes=32;
    r.weights=std::make_unique<WeightTable>();r.native=std::make_unique<NativeDense>();
    std::string error;const auto before=drains;
    check(load_fixture(*r.weights,r.arena,error)&&r.weights->load_complete(),"real canonical loader completes fixture");
    check(static_cast<float*>(r.arena)[7]==8,"canonical fixture bytes uploaded");drains=before;
    r.reservation=budget.reserve({{0,32},0,0,0,0,0},ReservationPurpose::foreground).lease;return r;
}
int main() {
    const WindowResources capacity{{100,100},0,0,0,0,0},zero{{0,0},0,0,0,0,0};
    {
        WindowBudget budget(capacity,zero);auto input=resources(budget);const auto* table=input.weights.get();const auto* base=static_cast<const char*>(input.arena);
        std::shared_ptr<const OwnedModelStage> out;std::string error;
        fail_drain=true;
        check(!load_fixture(*input.weights,input.arena,error)&&!input.weights->load_complete()&&!input.weights->all().empty(),"final drain failure leaves metadata but no readiness");
        fail_drain=false;drains=0;
        check(!adopt_model_stage(input,out,error)&&input.arena&&!out,"failed canonical load cannot be adopted");
        check(load_fixture(*input.weights,input.arena,error),"successful retry restores readiness");drains=0;
        check(!input.weights->load("/nonexistent/strata-fixture",input.arena,32,error)&&!input.weights->load_complete(),"early retry failure revokes prior readiness");
        check(!adopt_model_stage(input,out,error),"stale successful metadata cannot be adopted");
        fail_upload=true;
        check(!load_fixture(*input.weights,input.arena,error)&&!input.weights->load_complete(),"upload failure does not publish readiness");fail_upload=false;
        check(load_fixture(*input.weights,input.arena,error),"canonical fixture reload succeeds");drains=0;
        NativeDenseReleaseTestAccess::fail_load(*input.native);
        check(!adopt_model_stage(input,out,error)&&input.arena&&!out,"partial native loader state cannot transfer into model owner");
        input.native=std::make_unique<NativeDense>();
        auto& refs=const_cast<std::map<std::string,WeightRef>&>(input.weights->all());
        auto saved=refs.at("fixture.weight");refs.clear();
        check(!adopt_model_stage(input,out,error)&&input.arena&&!out,"empty model metadata refused");
        refs["fixture.weight"]=saved;
        refs.at("fixture.weight").data=base+31;refs.at("fixture.weight").bytes=2;
        check(!adopt_model_stage(input,out,error)&&input.arena&&!out,"one-byte canonical overflow refused");
        refs.at("fixture.weight").data=&device;
        check(!adopt_model_stage(input,out,error)&&input.arena&&!out,"external canonical allocation refused");
        refs.at("fixture.weight")=saved;refs.at("fixture.weight").data=nullptr;
        check(!adopt_model_stage(input,out,error)&&input.arena&&!out,"resident null data refused");
        refs.at("fixture.weight")=saved;
        WeightRef skipped;skipped.resident=false;refs["skipped.weight"]=skipped;
        input.arena_bytes=33;
        check(!adopt_model_stage(input,out,error)&&input.arena&&!out&&drains==0,"undercharged adoption leaves caller resources untouched");
        input.arena_bytes=32;
        check(adopt_model_stage(input,out,error)&&!input.arena&&!input.weights&&out->healthy(),"loaded ownership transfers atomically");
        check(&out->weights()==table,"owner retains actual model metadata");
        check(out->owns_weight_span(base,32,false)&&out->owns_weight_span(base+31,1,false),"canonical spans contained in retained arena");
        check(!out->owns_weight_span(base+31,2,false)&&!out->owns_weight_span(base,33,false)&&
              !out->owns_weight_span(base,0,false)&&!out->owns_weight_span(base,1,true),"oversize empty and unowned native spans refused");
        auto reader=out;out.reset();check(frees==0&&budget.snapshot().reserved.device_bytes[1]==32,"reader retains model and full reservation");
        reader.reset();check(drains==1&&releases==1&&frees==1&&device==0&&budget.snapshot().reserved==zero,"last reader drains and releases on owner device then restores caller");
    }
    for(int failure=0;failure<3;++failure) {
        const auto pid=fork();check(pid>=0,"fork failure isolation");
        if(pid==0) {
            WindowBudget budget(capacity,zero);auto input=resources(budget);std::shared_ptr<const OwnedModelStage> out;std::string error;
            check(adopt_model_stage(input,out,error),"failure model adopted");
            fail_drain=failure==0;fail_native=failure==1;fail_free=failure==2;
            const auto before=frees;out.reset();
            check(model_stage_quarantine_count()==1&&budget.snapshot().reserved.device_bytes[1]==32,"uncertain teardown retains allocation and reservation");
            check(frees==before+(failure==2?1:0),"teardown stops before dependent frees");
            ModelStageResources empty;check(!adopt_model_stage(empty,out,error),"quarantine fences later model adoption");
            std::exit(0);
        }
        int status=0;check(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"isolated teardown failure passed");
    }
    std::puts("window_model_owner_test passed");
}
