#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
// Fixture state models already loaded allocations. This test does not qualify load().
#define private public
#include "strata/core/native_head.hpp"
#undef private
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>
using namespace strata::core;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static size_t packed_bytes=0;static bool packed_fail=false;static int packed_calls=0;
namespace strata::kernels {
size_t native_head_packed_allocation_bytes(const void*){return packed_bytes;}
bool native_head_packed_release_checked(const void*){++packed_calls;if(packed_fail)return false;packed_bytes=0;return true;}
}
static void* allocate(size_t n){return sycl::malloc_host(n,dpct::get_in_order_queue());}
static void fill(NativeHead& head){head.weights_=allocate(128);head.scratch_=allocate(36);head.bytes_=100;head.allocated_weight_bytes_=128;head.scratch_bytes_=36;head.device_=1;head.load_complete_=true;}
int main(){
 {NativeHead head;check(!head.loaded()&&head.allocated_bytes()==0,"fresh head empty");}
 {
   NativeHead head;fill(head);packed_bytes=64;std::string err;
   check(head.loaded()&&head.ready_for_adoption()&&head.allocated_bytes()==228,"actual padded weight plus scratch plus packed bytes");
   dpct::select_device(0);check(!head.release_checked(err)&&sycl::allocations==2,"wrong device refuses before frees");
   dpct::select_device(1);check(head.release_checked(err)&&head.allocated_bytes()==0&&!head.loaded(),"checked head release");
   check(head.release_checked(err),"head release idempotent");
 }
 {
   NativeHead head;fill(head);dpct::select_device(0);head.unload();
   check(dpct::get_current_device_id()==0&&!head.release_started_&&head.allocated_bytes()==0,"unload selects owner and restores caller, permits reload");
 }
 for(int failure=0;failure<2;++failure){
   void* weights;void* scratch;int before;
   {NativeHead head;fill(head);weights=head.weights_;scratch=head.scratch_;dpct::select_device(1);std::string err;
    packed_fail=failure==0;sycl::fail_free=failure==1;
    check(!head.release_checked(err)&&head.release_failed_&&!head.ready_for_adoption(),"failed release fences head");
    before=sycl::free_calls;const auto packed_before=packed_calls;
    check(!head.release_checked(err)&&sycl::free_calls==before&&packed_calls==packed_before,"failed release never retries");
   }
   check(sycl::free_calls==before,"destructor never retries uncertain free");
   packed_fail=false;sycl::fail_free=false;
   // Test-owned cleanup after proving production refusal/no retry; no uncertainty in this fake allocator.
   sycl::free(weights,dpct::get_in_order_queue());sycl::free(scratch,dpct::get_in_order_queue());
 }
 for(bool device_owned:{false,true}){
   NativeEmbed embed;void* p=allocate(64);embed.host_=device_owned?nullptr:p;embed.dev_=p;embed.device_owned_=device_owned;embed.bytes_=64;embed.device_=1;embed.load_complete_=true;
   check(embed.allocated_device_bytes()==(device_owned?64:0)&&embed.allocated_host_bytes()==(device_owned?0:64)&&embed.allocated_pinned_bytes()==embed.allocated_host_bytes(),"embedding resource categories");
   set_native_embed(&embed);dpct::select_device(1);std::string err;
   check(embed.release_checked(err)&&!native_embed()&&!embed.ready_for_adoption()&&embed.allocated_host_bytes()==0&&embed.allocated_device_bytes()==0,"embedding alias released once and global detached");
 }
 {
   void* retained;int before;
   {NativeEmbed embed;retained=allocate(64);embed.host_=retained;embed.dev_=retained;embed.bytes_=64;embed.device_=1;embed.load_complete_=true;
    std::string err;sycl::fail_free=true;check(!embed.release_checked(err)&&embed.allocated_host_bytes()==64,"failed embedding free retains allocation charge");before=sycl::free_calls;
    check(!embed.release_checked(err)&&sycl::free_calls==before,"embedding sticky free failure");}
   check(sycl::free_calls==before,"embedding destructor does not retry");sycl::fail_free=false;sycl::free(retained,dpct::get_in_order_queue());
 }
 check(sycl::allocations==0,"fixture leaves no allocations");std::puts("SYCL native lifetime CPU seam test passed");
}
