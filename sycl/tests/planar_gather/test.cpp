#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <vector>
static void check(bool b,const char* s){if(!b){std::fprintf(stderr,"FAIL: %s\n",s);std::exit(1);}}
static bool fail_launch=false,allocate_null=false;
static bool fail_allocate=false,fail_insert=false,fail_drain=false,fail_free_source=false,fail_free_new=false;
static void* source_pointer=nullptr;
static std::set<void*> live;
static int frees_source=0,frees_new=0;
namespace sycl {
struct int4 { int v[4]; };
template<int> struct range {size_t n;explicit range(size_t v):n(v){}};
template<int> struct id {size_t n;size_t operator[](int)const{return n;}};
struct queue {
 template<class F> void parallel_for(range<1> r,F f){for(size_t i=0;i<r.n;++i){f(id<1>{i});if(fail_launch)throw std::runtime_error("injected launch failure");}}
 void wait_and_throw(){if(fail_drain)throw std::runtime_error("injected drain failure");}
};
void free(void* p,queue&){if(p==source_pointer){++frees_source;if(fail_free_source)throw std::runtime_error("source free failed");}else{++frees_new;if(fail_free_new)throw std::runtime_error("new free failed");}check(live.erase(p)==1,"allocation freed once");std::free(p);}
}
namespace dpct {sycl::queue& get_in_order_queue(){static sycl::queue q;return q;}}
namespace strata {void* malloc_device_guarded(size_t n,sycl::queue&,const char*){if(fail_allocate)throw std::bad_alloc();if(allocate_null)return nullptr;void* p=std::malloc(n);check(p,"allocation");live.insert(p);return p;}}
struct Q6Planar {size_t blocks;int n_in;const uint8_t *ql,*qh,*sc,*d;};
struct Registry {
 std::map<const void*,Q6Planar> rows;
 auto find(const void* p){return rows.find(p);}
 auto end(){return rows.end();}
 size_t count(const void* p){return rows.count(p);}
 auto emplace(const void* p,Q6Planar v){if(fail_insert)throw std::bad_alloc();return rows.emplace(p,v);}
 void erase(const void* p){rows.erase(p);}
};
Registry& q6_planar_registry(){static Registry r;return r;}
struct Q6PlanarLayout {size_t o_qh,o_sc,o_d,total;};
Q6PlanarLayout q6_planar_layout(size_t nb){auto a=[](size_t n){return(n+255)&~size_t(255);};const auto qh=a(nb*128),sc=a(qh+nb*64),d=a(sc+nb*16);return{qh,sc,d,d+nb*2};}
Q6Planar q6_planar_make(void* p,size_t nb,int n_in,int){auto* b=static_cast<uint8_t*>(p);auto l=q6_planar_layout(nb);return{nb,n_in,b,b+l.o_qh,b+l.o_sc,b+l.o_d};}
bool native_q6_k_planar_enabled(){return true;}
struct Q6Packed {void* base;int n_in,n_out;bool release_failed=false;};
std::map<const void*,Q6Packed>& q6_packed_registry(){static std::map<const void*,Q6Packed> r;return r;}
struct Q81Block {};
static bool g_q6_bypass=false,g_multi_exact=false;
static int launches=0;
void validate_shape(int,int,int){}
void validate_pointer(const void*){}
void validate_stream(void*){}
void launch_check(){}
namespace strata {sycl::queue* q_of(void*){return &dpct::get_in_order_queue();}}
template<int N> void q6_packed_launch(const Q6Packed&,const Q81Block*,float*,sycl::queue*){++launches;}
#include "planar_transaction.inc"
static void reset(){fail_launch=allocate_null=false;fail_allocate=fail_insert=fail_drain=fail_free_source=fail_free_new=false;frees_source=frees_new=0;q6_planar_registry().rows.clear();q6_packed_registry().clear();while(q6_planar_quarantine){auto* n=q6_planar_quarantine;q6_planar_quarantine=n->next;delete n;}for(void* p:live)std::free(p);live.clear();}
int main(){
 for(int mode=0;mode<6;++mode){
  reset();const auto layout=q6_planar_layout(2);auto* src=static_cast<uint8_t*>(std::malloc(layout.total));source_pointer=src;live.insert(src);
  for(size_t i=0;i<layout.total;++i)src[i]=uint8_t(i*13+i/127);
  q6_planar_registry().emplace(src,q6_planar_make(src,2,256,2));
  int32_t rows[]={1,0};
  fail_allocate=mode==1;allocate_null=mode==2;fail_launch=mode==3;fail_drain=mode==4;fail_insert=mode==5;
  void* result=nullptr;bool threw=false;try{result=native_q6_k_planar_gather(src,256,rows,2,"gather fixture");}catch(...){threw=true;}
  check(frees_source==0&&frees_new==0,"gather never frees borrowed input or uncertain candidate");
  if(mode==0){
   check(!threw&&result&&result!=src&&live.size()==2&&q6_planar_registry().count(result)==1&&!q6_planar_quarantine,"success publishes completed candidate");
   auto* dst=static_cast<uint8_t*>(result);
   const size_t offsets[]={0,layout.o_qh,layout.o_sc,layout.o_d},strides[]={128,64,16,2};
   for(int plane=0;plane<4;++plane)for(int row=0;row<2;++row)
    check(std::memcmp(dst+offsets[plane]+row*strides[plane],src+offsets[plane]+rows[row]*strides[plane],strides[plane])==0,"numeric plane row bytes unchanged");
   check(native_head_packed_allocation_bytes(src)==0,"successful main candidate not charged as source auxiliary");
  }else if(mode<=2){check(!threw&&!result&&live.size()==1&&!q6_planar_quarantine,"no allocation produces no retained candidate or charge");}
  else{
   check(threw&&!result&&live.size()==2&&q6_planar_quarantine&&q6_planar_registry().rows.size()==1,"failure retains candidate without publishing");
   check(native_head_packed_allocation_bytes(src)==layout.total,"candidate charge attached to source owner");
   check(!native_head_packed_release_checked(src),"source release fenced while candidate work uncertain");
   bool fenced=false;g_q6_bypass=true;try{q6_packed_mmvq(src,nullptr,nullptr,1,1,9,nullptr);}catch(...){fenced=true;}
   check(fenced&&launches==0,"numeric fallback fenced");
   fenced=false;try{native_q6_k_planar_gather(src,256,rows,2,"retry");}catch(...){fenced=true;}
   check(fenced&&live.size()==2&&frees_source==0&&frees_new==0,"failed gather cannot retry or allocate again");
   native_q6_k_unpack(src);check(q6_planar_registry().count(src)==1,"legacy release retains quarantined input registry");
  }
 }
 reset();std::puts("SYCL planar gather CPU transaction test passed");
}
