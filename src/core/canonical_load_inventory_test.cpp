#include "strata/core/weights.hpp"
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
// Only typed holder lifetime is mocked; real verified binding/path authentication
// belongs to driver admission. No VerifiedInputLease functions linked here.
namespace strata::core {class VerifiedDriverBindings {public:~VerifiedDriverBindings(){++destroyed;}static inline int destroyed=0;};}
using namespace strata::core;
static int calls=0,fail=0,host_live=0,device_live=0;
static bool step(){return ++calls!=fail;}
int cudaHostAlloc(void** p,size_t n,unsigned){if(!step())return 1;*p=std::malloc(n);assert(*p);++host_live;return 0;}
int cudaFreeHost(void* p){if(!step())return 1;std::free(p);--host_live;return 0;}
int cudaMemcpy(void* d,const void* s,size_t n,int){if(!step())return 1;std::memcpy(d,s,n);return 0;}
int cudaDeviceSynchronize(){return step()?0:1;}int cudaGetDevice(int* d){if(!step())return 1;*d=0;return 0;}int cudaSetDevice(int){return step()?0:1;}
int cudaMalloc(void** p,size_t n){if(!step())return 1;*p=std::malloc(n);assert(*p);++device_live;return 0;}int cudaFree(void* p){if(!step())return 1;std::free(p);--device_live;return 0;}
namespace strata::platform {void advise_willneed(int,uint64_t,uint64_t){}}
WindowBudget budget(){return WindowBudget({{1<<20},1<<24,1<<24,0,0,0},{{0},0,0,0,0,0});}
struct Fixture{
 std::string path;Fixture(){char p[]="/tmp/canonical-inventory-XXXXXX";path=mkdtemp(p);assert(!path.empty());}
 ~Fixture(){std::filesystem::remove_all(path);}
 void write(bool hc=false,bool duplicate=false,bool promoted=false){std::ofstream idx(path+"/index.txt");idx<<"# align 16 pool 256 tensors 8\n";
  for(int k=0;k<6;++k){int src=k==1||k==2||k==3?4:2;int dst=k==2?4:2;idx<<"kind"<<k<<" "<<k%4<<" "<<k<<" "<<k*8<<" "<<src<<" "<<k*16<<" "<<dst<<" 1 1 0 0 0 0 0 0 0 0 0 0\n";}
  idx<<"quant 0 0 64 6 96 8 4 1 4 0 4 0 0 4 4 0 1 0\n";
  if(hc)idx<<"blk.0.hc_attn_down.weight 0 "<<(promoted?1:4)<<" 128 "<<(promoted?128:64)<<" 112 64 32 1 0 0 0 0 0 0 0 0 0 0\n";
  if(duplicate)idx<<"kind0 0 0 0 2 192 2 1 1 0 0 0 0 0 0 0 0 0 0\n";
  std::vector<uint8_t> data(256);float one=1;for(int k=0;k<6;++k)std::memcpy(data.data()+k*8,&one,4);for(int i=0;i<32;++i){uint16_t x=0x3f80;if(promoted)std::memcpy(data.data()+128+i*4,&one,4);else std::memcpy(data.data()+128+i*2,&x,2);}uint16_t h=0x3c00;std::memcpy(data.data()+68,&h,2);for(const auto* name:{"dense.bin","embd.bin","experts.bin","extra.bin"}){std::ofstream file(path+"/"+name,std::ios::binary);file.write(reinterpret_cast<const char*>(data.data()),data.size());}
 }
};
void enable(WeightTable& w,WindowBudget& b,const std::string& path,std::weak_ptr<const VerifiedDriverBindings>* weak=nullptr){CanonicalInventoryLimits l{16,256};uint64_t n=0;std::string e;assert(canonical_inventory_host_bound(l,n,e));auto lease=b.reserve({{0},n,0,0,0,0},ReservationPurpose::foreground);assert(lease);auto bindings=std::make_shared<VerifiedDriverBindings>();if(weak)*weak=bindings;assert(w.enable_inventory_capture(l,std::move(lease.lease),bindings,path,e));assert(!w.enable_inventory_capture(l,{},bindings,path,e));}
int load_case(int fault,bool hc,bool promoted=false){Fixture fixture;fixture.write(hc,false,promoted);auto b=budget();std::weak_ptr<const VerifiedDriverBindings> weak;auto w=std::make_unique<WeightTable>();enable(*w,b,fixture.path,&weak);std::vector<uint8_t> arena(256);std::string e;std::set<std::string> skip{"kind5"};if(hc)skip.insert("blk.0.hc_attn_down.weight");calls=0;fail=fault;bool ok=w->load(fixture.path,arena.data(),arena.size(),e,&skip);if(ok&&hc){uint64_t bytes=0;ok=w->load_hc_q8(fixture.path,bytes,e);}
 int used=calls;if(fault){assert(!ok);assert(!w->loaded_inventory());auto charged=b.snapshot().reserved.host_bytes;w.reset();assert(!weak.expired());assert(b.snapshot().reserved.host_bytes==charged);return used;}
 assert(ok);auto* inv=w->loaded_inventory();assert(inv&&inv->records.size()==size_t(hc?8:7));for(int k=0;k<6;++k){auto& r=inv->records[k];assert(r.source_index_kind==k&&r.raw16==(k>=4)&&r.chosen==(k!=5));assert(r.source_offset==uint64_t(k*8)&&r.file_id==k%4);assert(r.plane_count==(k==5?0u:1u));}
 assert(inv->records[1].planes[0].conversion==CanonicalPlaneConversion::bf16_high_bits);assert(inv->records[3].planes[0].conversion==CanonicalPlaneConversion::f32_to_f16);assert(inv->records[4].planes[0].conversion==CanonicalPlaneConversion::copy);assert(inv->records[6].plane_count==2&&inv->records[6].planes[1].source_bytes==2&&inv->records[6].planes[1].runtime_bytes==4&&inv->records[6].planes[1].conversion==CanonicalPlaneConversion::f16_to_f32);
 if(hc){assert(inv->hc_completed&&inv->hc_records.size()==1&&inv->hc_records[0].canonical_row==7&&!inv->records[7].chosen);assert(inv->hc_records[0].conversion==(promoted?CanonicalHcConversion::bf16_in_f32_to_q8_host:CanonicalHcConversion::bf16_raw16_to_q8_host));assert(inv->hc_records[0].codes_bytes==32&&inv->hc_records[0].scales_bytes==4);}
 uint16_t bf=0,fp=0;float scale=0;std::memcpy(&bf,arena.data()+inv->records[1].runtime_offset,2);std::memcpy(&fp,arena.data()+inv->records[3].runtime_offset,2);std::memcpy(&scale,arena.data()+inv->records[6].runtime_offset+4,4);assert(bf==0x3f80&&fp==0x3c00&&scale==1.0f);
 w.reset();assert(weak.expired()&&b.snapshot().reserved.host_bytes==0);assert(!host_live&&!device_live);return used;}
void retirement_failure(int offset){Fixture f;f.write(true);auto b=budget();auto w=std::make_unique<WeightTable>();std::weak_ptr<const VerifiedDriverBindings> weak;enable(*w,b,f.path,&weak);std::vector<uint8_t>a(256);std::set<std::string> skip{"blk.0.hc_attn_down.weight"};std::string e;calls=0;fail=0;assert(w->load(f.path,a.data(),a.size(),e,&skip));uint64_t bytes=0;assert(w->load_hc_q8(f.path,bytes,e));const auto charged=b.snapshot().reserved.host_bytes;fail=calls+offset;w.reset();assert(!weak.expired()&&b.snapshot().reserved.host_bytes==charged);}
int main(){for(int i=1;i<=4;++i){pid_t p=fork();assert(p>=0);if(!p){retirement_failure(i);_Exit(0);}int status;waitpid(p,&status,0);assert(WIFEXITED(status)&&WEXITSTATUS(status)==0);}load_case(0,true,true);int ordinary=load_case(0,false),hc=load_case(0,true);for(int i=1;i<=hc;++i){pid_t pid=fork();assert(pid>=0);if(!pid){load_case(i,true);_Exit(0);}int status;waitpid(pid,&status,0);if(!WIFEXITED(status)||WEXITSTATUS(status)){std::cerr<<"failure position "<<i<<'\n';std::abort();}}
 {Fixture f;f.write(false,true);auto b=budget();WeightTable w;enable(w,b,f.path);calls=0;fail=0;std::vector<uint8_t>a(256);std::string e;assert(!w.load(f.path,a.data(),a.size(),e)&&calls==0&&!w.loaded_inventory());f.write();assert(w.load(f.path,a.data(),a.size(),e)&&w.loaded_inventory());assert(!w.load(f.path+"/wrong",a.data(),a.size(),e)&&!w.loaded_inventory());}
 {Fixture f;f.write();WeightTable w;std::string e;std::vector<uint8_t>a(256);assert(w.load(f.path,a.data(),a.size(),e));assert(!w.loaded_inventory());auto b=budget();CanonicalInventoryLimits l{16,256};assert(!w.enable_inventory_capture(l,{},std::make_shared<VerifiedDriverBindings>(),f.path,e));}
 {Fixture f;f.write();auto b=budget();WeightTable w;CanonicalInventoryLimits l{16,256};std::string e;uint64_t bound=0;assert(canonical_inventory_host_bound(l,bound,e));auto inputs=std::make_shared<VerifiedDriverBindings>();auto lease=b.reserve({{0},bound-1,0,0,0,0},ReservationPurpose::foreground);assert(lease);assert(!w.enable_inventory_capture(l,lease.lease,inputs,f.path,e));assert(!w.loaded_inventory());auto invalid=l;invalid.max_records=UINT64_MAX;assert(!canonical_inventory_host_bound(invalid,bound,e));}
 {Fixture f;f.write(true);auto b=budget();WeightTable w;enable(w,b,f.path);std::string e;std::vector<uint8_t>a(256);std::set<std::string> skip{"blk.0.hc_attn_down.weight"};assert(w.load(f.path,a.data(),a.size(),e,&skip)&&w.loaded_inventory());uint64_t bytes=0;assert(!w.load_hc_q8(f.path+"/wrong",bytes,e)&&!w.loaded_inventory());}
 {Fixture f;f.write();auto b=budget();WeightTable w;enable(w,b,f.path);std::filesystem::remove(f.path+"/dense.bin");std::vector<uint8_t>a(256);std::string e;assert(!w.load(f.path,a.data(),a.size(),e)&&!w.loaded_inventory());}
 std::cout<<"actual WeightTable inventory passed "<<hc+4<<" injected runtime failures; canonical calls "<<ordinary<<", kinds0..5/planes/skips/HC/retry/lifetime/default checks\n";
}
