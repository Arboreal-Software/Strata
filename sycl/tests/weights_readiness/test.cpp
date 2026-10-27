#include "strata/core/weights.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unistd.h>
using strata::core::WeightTable;
namespace strata::platform {void advise_willneed(int,uint64_t,uint64_t){}}
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
int main(){
    char path[]="/tmp/strata-sycl-weights-XXXXXX";check(mkdtemp(path),"temporary pack");
    const std::filesystem::path dir(path);
    {std::ofstream f(dir/"index.txt");f<<"# align 32 pool 32 tensors 1\nfixture.weight 0 2 0 32 0 32 8 1 0 0 0 0 0 0 0 0 0 0\n";}
    {std::ofstream f(dir/"dense.bin",std::ios::binary);float values[8]={1,2,3,4,5,6,7,8};f.write(reinterpret_cast<const char*>(values),sizeof values);}
    float arena[8]={};std::string err;WeightTable table;
    check(!table.load_complete(),"fresh table is incomplete");
    check(table.load(path,arena,sizeof arena,err)&&table.load_complete()&&arena[7]==8,"real SYCL loader parses and uploads fixture");
    sycl::fail_drain=true;
    check(!table.load(path,arena,sizeof arena,err)&&!table.load_complete()&&!table.all().empty(),"failed final drain cannot publish readiness despite metadata");
    sycl::fail_drain=false;
    check(table.load(path,arena,sizeof arena,err)&&table.load_complete(),"successful retry restores readiness");
    check(!table.load("/nonexistent/strata-sycl-fixture",arena,sizeof arena,err)&&!table.load_complete(),"early failure revokes previous readiness");
    sycl::fail_copy=true;
    check(!table.load(path,arena,sizeof arena,err)&&!table.load_complete(),"failed upload cannot publish readiness");sycl::fail_copy=false;
    check(table.load(path,arena,sizeof arena,err)&&table.load_complete(),"reload after copy failure");
    uint64_t bytes=123;
    check(!table.load_hc_q8(path,bytes,err)&&bytes==0&&table.load_complete(),"unsupported HC request refuses without invalidating canonical data");
    check(table.auxiliary_bytes()==0&&table.auxiliary_device()==-1&&!table.owns_auxiliary_span(arena,4),"canonical SYCL table never claims auxiliary storage");
    check(table.release_auxiliary_checked(err)&&!table.load_complete(),"empty auxiliary retirement revokes readiness");
    check(table.release_auxiliary_checked(err),"empty auxiliary retirement is idempotent");
    check(!table.load(path,arena,sizeof arena,err),"retired table cannot be reloaded");
    check(sycl::allocations==0,"all staging allocations released");
    std::filesystem::remove_all(dir);std::puts("SYCL weights readiness CPU seam test passed");
}
