#include "strata/core/on_device.hpp"
#include <cassert>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <xmmintrin.h>
static int current=0,gets=0,sets=0,fail_get=0,fail_set=0;
static std::vector<int> selected;
int cudaGetDevice(int* p){++gets;if(gets==fail_get){std::fprintf(stderr,"FAULT get %d\n",gets);return 7;}*p=current;return 0;}
int cudaSetDevice(int d){++sets;if(sets==fail_set){std::fprintf(stderr,"FAULT set %d\n",sets);return 7;}selected.push_back(d);current=d;return 0;}
int cudaDeviceSynchronize(){std::_Exit(74);} // Selection may never synchronize.
using strata::core::OnDevice;
static bool nested(){
 const OnDevice parent(1,true);
 {const OnDevice child(2,true);assert(current==2);}
 std::fprintf(stderr,"AFTER CHILD\n");assert(current==1);return true;
}
using Clock=std::chrono::steady_clock;
using cudaStream_t=void*;
constexpr int cudaErrorNotReady=34;
int cudaStreamQuery(void*){std::_Exit(77);}
double ms_since(Clock::time_point p){return std::chrono::duration<double,std::milli>(Clock::now()-p).count();}
struct Verifier {
 bool strict_retirement_=true,retirement_init_started_=false;
 int device_=-1;cudaStream_t cs_=nullptr,copy_=nullptr;
 std::atomic<bool> released_{false};uint32_t flag=0;
 uint32_t *h_flag_=&flag,*h_flagA_=nullptr,*h_flagB_=nullptr,*hflag_in_h_=nullptr;
 void trace_ev(const char*,int,int,int64_t){}
 bool release_gpu_waits(int);
};
#include "release.inc"
int main(int argc,char** argv){
 assert(argc==2);const char* c=argv[1];
 if(!std::strcmp(c,"no-work")){
  Verifier v;assert(v.release_gpu_waits(1));assert(v.released_&&v.flag==UINT32_MAX);assert(gets==0&&sets==0);return 0;
 }else if(!std::strcmp(c,"partial-no-device")){
  Verifier v;v.retirement_init_started_=true;v.release_gpu_waits(1);
 }else if(!std::strcmp(c,"get")){fail_get=1;OnDevice d(1,true);}
 else if(!std::strcmp(c,"select")){fail_set=1;OnDevice d(1,true);}
 else if(!std::strcmp(c,"restore")){fail_set=2;OnDevice d(1,true);}
 else if(!std::strcmp(c,"nested-restore")){fail_set=3;nested();}
 else if(!std::strcmp(c,"parent-restore")){fail_set=4;nested();}
 else if(!std::strcmp(c,"worker")){fail_set=1;std::thread t([]{OnDevice child(2,true);std::fprintf(stderr,"WORKER EFFECT\n");});t.join();}
 else if(!std::strcmp(c,"invalid")){OnDevice d(-1,true);}
 else if(!std::strcmp(c,"success")){
  {OnDevice d(0,true);assert(gets==1&&sets==0);}
  assert(nested());assert(current==0);assert((selected==std::vector<int>{1,2,1,0}));return 0;
 }else if(!std::strcmp(c,"legacy")){
  {OnDevice d(-1);assert(gets==0&&sets==0);}
  fail_get=1;{OnDevice d(1);}assert(current==0&&sets==0);
  fail_get=0;fail_set=1;{OnDevice d(1);}assert(current==0&&sets==1);
  fail_set=3;{OnDevice d(1);}assert(current==1&&sets==3);return 0;
 }else return 75;
 std::fprintf(stderr,"UNEXPECTED EFFECT\n");return 76;
}
