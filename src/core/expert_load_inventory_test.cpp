#include "strata/core/expert_load_inventory.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <new>
#include <cstdlib>
static bool allocation_fault=false;static size_t allocation_count=0,allocation_failure=0;
__attribute__((noinline)) void* operator new(size_t n){if(allocation_fault&&++allocation_count>=allocation_failure&&allocation_failure)throw std::bad_alloc();if(auto* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
__attribute__((noinline)) void* operator new[](size_t n){return ::operator new(n);}
__attribute__((noinline)) void operator delete(void* p)noexcept{std::free(p);}
__attribute__((noinline)) void operator delete[](void* p)noexcept{std::free(p);}
__attribute__((noinline)) void operator delete(void* p,size_t)noexcept{std::free(p);}
__attribute__((noinline)) void operator delete[](void* p,size_t)noexcept{std::free(p);}
namespace strata::core {class VerifiedDriverBindings {};}
// Only GGML format lookup is mocked: actual layout/Hadamard parser, selected
// fields, bounds, commit, generations and genuine budget remain production code.
namespace strata::kernels::cpu {
bool native_fmt(int gu,int down,int64_t,int64_t,NativeFmt& f,std::string& e){if(gu<0||down<0){e="mock unsupported type";return false;}f.gu_type=gu;f.d_type=down;f.gu_act=8;f.d_act=9;f.n_embd=f.n_ff=4;f.gu_row=f.d_row=1;f.up_off=4;f.down_off=8;f.bytes=12;f.act_bytes=f.h_bytes=4;return true;}
}
using namespace strata::core;using namespace strata::kernels::cpu;
struct Fixture{std::string path;Fixture(){char p[]="/tmp/expert-inventory-XXXXXX";path=mkdtemp(p);assert(!path.empty());}~Fixture(){std::filesystem::remove_all(path);}void write(int version=5,bool duplicate=false,int block=2){std::ofstream out(path+"/native_experts.txt");out<<"# strata native experts v"<<version<<" (n_expert 3)\n";out<<"1 11 12 36 12 400 500 600 c.gguf,d.gguf,e.gguf\n0 11 12 0 12 100 200 300 a.gguf,b.gguf,a.gguf\n";if(duplicate)out<<"0 11 12 0 12 100 200 300 a.gguf\n";if(version>=5){std::ofstream h(path+"/hadamard.txt");h<<"# strata hadamard v1\nblock "<<block<<"\nsigns 4 1 -1 1 -1\nlayer 0 1 1 1\nlayer 1 0 0 1\n";}else std::filesystem::remove(path+"/hadamard.txt");}};
ExpertInventoryLimits limits(){return {4,256,1024,4096,4,32};}
WindowBudget budget(){return WindowBudget({{0},1<<24,0,0,0,0},{{0}});}
std::unique_ptr<ExpertLayoutCapture> capture(WindowBudget& b,const Fixture& f,bool native=true,bool had=true,ExpertInventoryLimits l=limits(),std::weak_ptr<const VerifiedDriverBindings>* weak=nullptr){uint64_t charge;std::string e;assert(expert_inventory_host_bound(l,charge,e));auto lease=b.reserve({{0},charge,0,0,0,0},ReservationPurpose::foreground);assert(lease);auto binding=std::make_shared<VerifiedDriverBindings>();if(weak)*weak=binding;std::unique_ptr<ExpertLayoutCapture> c;assert(ExpertLayoutCapture::create(l,std::move(lease.lease),binding,{f.path,native,had},c,e));return c;}
int main(){
 Fixture f;f.write();auto b=budget();std::weak_ptr<const VerifiedDriverBindings> weak;auto c=capture(b,f,true,true,limits(),&weak);std::string e;
 assert(expert_layout_load(f.path,2,2,e,c.get()));auto* inv=c->completed_inventory();assert(inv&&inv->native&&inv->effective_experts==3&&inv->requested_experts==2&&inv->layers.size()==2&&inv->total_bytes==72);
 assert(inv->layers[0].layer==1&&inv->layers[0].source_order==0&&inv->layers[0].shard_names[1]=="d.gguf"&&inv->layers[1].gguf_offsets[2]==300);
 assert(inv->hadamard_block==2&&inv->signs.size()==1&&inv->signs[0].values==std::vector<int8_t>({1,-1,1,-1}));assert(inv->hadamard_roles==std::vector<uint8_t>({7,4}));assert(!inv->layers[0].format.gu_signs&&!inv->layers[0].format.d_signs);assert(expert_layout().fmt[0].gu_signs&&expert_layout().fmt[0].gu_signs[1]==-1);
 auto lg=expert_layout_generation(),hg=hadamard_generation();
 f.write(5,true);assert(!expert_layout_load(f.path,2,2,e,c.get())&&!c->completed_inventory());assert(expert_layout_generation()==lg&&hadamard_generation()==hg);
 f.write(5,false,8);assert(!expert_layout_load(f.path,2,2,e,c.get()));assert(expert_layout_generation()==lg&&hadamard_generation()==hg&&hadamard_spec().block==2);
 f.write();assert(expert_layout_load(f.path,2,2,e,c.get()));hadamard_set({});assert(!c->completed_inventory());assert(expert_layout_load(f.path,2,2,e,c.get()));
 // Legacy failed validation mutates Hadamard; generation must revoke receipt.
 f.write();{std::ifstream file(f.path+"/native_experts.txt");std::string text((std::istreambuf_iterator<char>(file)),{});auto pos=text.find("0 11 12");assert(pos!=std::string::npos);text.replace(pos,7,"0 42 12");std::ofstream(f.path+"/native_experts.txt")<<text;}assert(!expert_layout_load(f.path,2,2,e));assert(!c->completed_inventory());f.write();assert(expert_layout_load(f.path,2,2,e,c.get()));
 {Fixture canonical;auto other=capture(b,canonical,false,false);assert(expert_layout_load(canonical.path,2,3,e,other.get()));assert(other->completed_inventory()&&!other->completed_inventory()->native&&other->completed_inventory()->layers.size()==2);assert(other->completed_inventory()->canonical.n_embd==uint64_t(H)&&other->completed_inventory()->canonical.offsets[2]==O_GU_SCALES&&other->completed_inventory()->canonical.bytes[3]==BLOB-O_D_SCALES);assert(!c->completed_inventory());}
 for(int mode=0;mode<10;++mode){f.write();auto l=limits();if(mode==0)l.max_sign_elements=3;if(mode==1)l.max_line_bytes=8;if(mode==2)l.max_metadata_bytes=8;if(mode==3)l.max_layers=1;auto bad=capture(b,f,true,true,l);auto before_l=expert_layout_generation(),before_h=hadamard_generation();if(mode==4)std::filesystem::remove(f.path+"/hadamard.txt");if(mode==5){std::filesystem::remove(f.path+"/native_experts.txt");std::filesystem::create_symlink("native_experts.txt",f.path+"/native_experts.txt");}if(mode>=6){auto name=mode==9?"/hadamard.txt":"/native_experts.txt";std::ifstream file(f.path+name);std::string text((std::istreambuf_iterator<char>(file)),{});if(mode==6)text.replace(text.find("v5"),2,"v9999999999999999999999999");if(mode==7)text.replace(text.find("n_expert 3"),10,"n_expert 99999999999999999999999");if(mode==8)text.replace(text.find("1 11 12"),7,"1 9999999999 12");if(mode==9)text.replace(text.find("v1"),2,"v999999999999999999999");std::ofstream(f.path+name)<<text;}assert(!expert_layout_load(f.path,2,2,e,bad.get())&&!bad->completed_inventory());assert(before_l==expert_layout_generation()&&before_h==hadamard_generation());if(mode==5)std::filesystem::remove(f.path+"/native_experts.txt");}
 f.write(4);{auto plain=capture(b,f,true,false);assert(expert_layout_load(f.path,2,2,e,plain.get())&&plain->completed_inventory()->hadamard_block==0);}
 {auto absent=capture(b,f,false,false);assert(!expert_layout_load(f.path,2,2,e,absent.get()));}
 {HadamardSpec prior;prior.block=64;HadamardParseLimits l{2,8,1024,1,4};f.write();assert(!hadamard_parse(f.path+"/hadamard.txt",2,prior,e,&l)&&prior.block==64);}
 {std::unique_ptr<ExpertLayoutCapture> unchanged;auto inputs=std::make_shared<VerifiedDriverBindings>();auto l=limits();assert(!ExpertLayoutCapture::create(l,{},inputs,{f.path,true,true},unchanged,e)&&!unchanged);}
 {auto existing=capture(b,f);auto* prior=existing.get();auto inputs=std::make_shared<VerifiedDriverBindings>();uint64_t bound=0;assert(expert_inventory_host_bound(limits(),bound,e));auto too_small=b.reserve({{0},bound-1,0,0,0,0},ReservationPurpose::foreground);assert(too_small);assert(!ExpertLayoutCapture::create(limits(),too_small.lease,inputs,{f.path,true,true},existing,e)&&existing.get()==prior);auto invalid=limits();invalid.max_layers=UINT64_MAX;bound=123;assert(!expert_inventory_host_bound(invalid,bound,e)&&bound==123);}
 size_t injected=0;
 {f.write();auto faults=capture(b,f);allocation_count=0;allocation_failure=0;allocation_fault=true;bool loaded=expert_layout_load(f.path,2,2,e,faults.get());allocation_fault=false;assert(loaded);const auto count=allocation_count;
  for(size_t failure=1;failure<=count;++failure){auto old_layout=expert_layout_generation(),old_hadamard=hadamard_generation();allocation_count=0;allocation_failure=failure;allocation_fault=true;loaded=expert_layout_load(f.path,2,2,e,faults.get());allocation_fault=false;assert(!loaded&&!faults->completed_inventory());assert(expert_layout_generation()==old_layout&&hadamard_generation()==old_hadamard);++injected;}
 }
 c.reset();assert(weak.expired()&&b.snapshot().reserved.host_bytes==0);
 std::cout<<"injected allocation failures "<<injected<<"; actual expert layout/Hadamard inventory: transactional commit, parsed order/signs, membership/IO/bounds, legacy generation invalidation and retained admission lifetime passed\n";
}
