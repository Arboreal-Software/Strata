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
static bool fail_allocate=false,fail_insert=false,fail_drain=false,fail_free_source=false,fail_free_new=false;
static void* source_pointer=nullptr;
static std::set<void*> live;
static int frees_source=0,frees_new=0;
namespace sycl {
template<int> struct range {size_t n;explicit range(size_t v):n(v){}};
template<int> struct id {size_t n;size_t operator[](int)const{return n;}};
struct queue {
 template<class F> void parallel_for(range<1> r,F f){for(size_t i=0;i<r.n;++i)f(id<1>{i});}
 void wait_and_throw(){if(fail_drain)throw std::runtime_error("injected drain failure");}
};
void free(void* p,queue&){if(p==source_pointer){++frees_source;if(fail_free_source)throw std::runtime_error("source free failed");}else{++frees_new;if(fail_free_new)throw std::runtime_error("new free failed");}check(live.erase(p)==1,"allocation freed once");std::free(p);}
}
namespace dpct {sycl::queue& get_in_order_queue(){static sycl::queue q;return q;}}
namespace strata {void* malloc_device_guarded(size_t n,sycl::queue&,const char*){if(fail_allocate)throw std::bad_alloc();void* p=std::malloc(n);check(p,"allocation");live.insert(p);return p;}}
struct Q6Planar {size_t blocks;};
struct Registry {
 std::map<const void*,Q6Planar> rows;
 size_t count(const void* p){return rows.count(p);}
 auto emplace(const void* p,Q6Planar v){if(fail_insert)throw std::bad_alloc();return rows.emplace(p,v);}
 void erase(const void* p){rows.erase(p);}
};
Registry& q6_planar_registry(){static Registry r;return r;}
struct Q6PlanarLayout {size_t o_qh,o_sc,o_d,total;};
Q6PlanarLayout q6_planar_layout(size_t nb){auto a=[](size_t n){return(n+255)&~size_t(255);};const auto qh=a(nb*128),sc=a(qh+nb*64),d=a(sc+nb*16);return{qh,sc,d,d+nb*2};}
Q6Planar q6_planar_make(void*,size_t nb,int,int){return{nb};}
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
static void reset(){fail_allocate=fail_insert=fail_drain=fail_free_source=fail_free_new=false;frees_source=frees_new=0;q6_planar_registry().rows.clear();q6_packed_registry().clear();while(q6_planar_quarantine){auto* n=q6_planar_quarantine;q6_planar_quarantine=n->next;delete n;}for(void* p:live)std::free(p);live.clear();}
int main(){
 for(int mode=0;mode<6;++mode){
  reset();auto* src=static_cast<uint8_t*>(std::malloc(210));source_pointer=src;live.insert(src);for(int i=0;i<210;++i)src[i]=uint8_t(i);
  fail_allocate=mode==1;fail_insert=mode==2;fail_drain=mode==3;fail_free_source=mode==4;fail_free_new=mode==5;if(mode==5)fail_insert=true;
  void* result=nullptr;bool threw=false;try{result=native_q6_k_planar_adopt(src,256,1,"fixture");}catch(...){threw=true;}
  if(mode==0){
   check(!threw&&result!=src&&live.size()==1&&frees_source==1&&q6_planar_registry().count(result)==1,"success publishes registry before releasing old allocation");
   const auto* b=static_cast<const uint8_t*>(result);for(int i=0;i<128;++i)check(b[i]==uint8_t(i),"low quantization plane unchanged");for(int i=0;i<64;++i)check(b[256+i]==uint8_t(128+i),"high quantization plane unchanged");
   for(int g=0;g<8;++g){const int so=8*(g>>2)+(g&3);check(b[512+2*g]==uint8_t(192+so)&&b[512+2*g+1]==uint8_t(196+so),"scale permutation unchanged");}
   check(b[768]==208&&b[769]==209,"delta unchanged");
  }else if(mode<=2){check(!threw&&result==src&&frees_source==0&&live.size()==1&&q6_planar_registry().rows.empty()&&!q6_planar_quarantine,"allocation/registry refusal retains valid original and cleans candidate");}
  else {check(threw&&q6_planar_quarantine&&q6_planar_quarantine->bytes==770&&live.size()==2,"uncertain operation quarantines both allocations and charge");const int before_source=frees_source,before_new=frees_new;bool fenced=false;try{native_q6_k_planar_adopt(src,256,1,"retry");}catch(...){fenced=true;}check(fenced&&frees_source==before_source&&frees_new==before_new,"quarantine fences future adoption without free retries");check(frees_source==(mode==4?1:0),"source free not attempted before commit or retried");check(q6_planar_registry().rows.size()==size_t(mode==4),"registry retains uncertain committed allocation only");check(native_head_packed_allocation_bytes(src)==770&&!native_head_packed_release_checked(src),"quarantined candidate remains charged and cannot release");bool numeric_fenced=false;try{q6_packed_mmvq(src,nullptr,nullptr,1,1,1,nullptr);}catch(...){numeric_fenced=true;}check(numeric_fenced&&launches==0,"quarantined source cannot dispatch numeric fallback");}
 }
 reset();
 source_pointer=std::malloc(210);live.insert(source_pointer);void* packed=std::malloc(770);live.insert(packed);
 q6_packed_registry()[source_pointer]=Q6Packed{packed,256,1,false};fail_free_new=true;
 check(!native_head_packed_release_checked(source_pointer)&&q6_packed_registry().at(source_pointer).release_failed,"failed packed free retains tombstone");
 check(native_head_packed_allocation_bytes(source_pointer)==770,"failed packed allocation remains charged");
 const int before=frees_new;check(!native_head_packed_release_checked(source_pointer)&&frees_new==before,"checked packed release does not retry");
 native_q6_k_unpack(source_pointer);check(frees_new==before&&q6_packed_registry().count(source_pointer)==1,"legacy unpack preserves failed tombstone");
 bool fenced=false;g_q6_bypass=true;try{q6_packed_mmvq(source_pointer,nullptr,nullptr,1,1,9,nullptr);}catch(...){fenced=true;}
 check(fenced&&launches==0,"numeric lookup rejects failed copy before bypass, shape or fallback");
 reset();std::puts("SYCL planar adoption CPU transaction test passed");
}
