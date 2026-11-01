#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <initializer_list>
using cudaError_t=int;
constexpr int cudaSuccess=0;
static int upload_status,sync_status,order,latches,upload_order,sync_order,failure_order;
static int exec_t=1,cs_=2;
static bool strict_retirement_;
static cudaError_t cudaGraphUpload(int,int){upload_order=++order;return upload_status;}
static cudaError_t cudaStreamSynchronize(int){sync_order=++order;return sync_status;}
static const char* cudaGetErrorString(int e){return e?"failure":"success";}
static void latch_copy_query_failure(){++latches;}
// Test seam ONLY: production retirement_failed is noreturn _Exit. Throwing here
// observes whether it was reached after BOTH operations without executing GPU.
struct Failed {};
[[noreturn]] static void retirement_failed(const char*){failure_order=++order;throw Failed{};}
static bool capture_tail(int T){
#include "capture_tail.inc"
}
static void check(bool b){if(!b)std::abort();}
int main(){for(bool strict:{false,true})for(int upload:{0,1})for(int sync:{0,1}){
 strict_retirement_=strict;upload_status=upload;sync_status=sync;order=latches=upload_order=sync_order=failure_order=0;
 bool returned=false,failed=false;try{returned=capture_tail(1);}catch(const Failed&){failed=true;}
 check(upload_order==1&&sync_order==2);
 if(strict&&(upload||sync))check(failed&&!returned&&latches==0&&failure_order==3);
 else check(returned&&!failed&&!latches&&!failure_order);
 }std::puts("actual extracted capture completion policy PASS");}
