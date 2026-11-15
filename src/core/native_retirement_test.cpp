// Prepared source-only CPU fixture. Link actual native_retirement.cpp; no GPU library.
#include "strata/core/native_retirement.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
using namespace strata::core;
namespace {
int current=7,gets=0,sets=0,syncs=0,fail_get=0,fail_set=0,fail_sync=0;
std::vector<int> drained;
void reset(){current=7;gets=sets=syncs=fail_get=fail_set=fail_sync=0;drained.clear();}
}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* d){if(++gets==fail_get)return cudaErrorUnknown;*d=current;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaSetDevice(int d){if(++sets==fail_set)return cudaErrorUnknown;current=d;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize(){if(++syncs==fail_sync)return cudaErrorUnknown;drained.push_back(current);return cudaSuccess;}
int main(){
 const std::array<int,2> devices{0,1};
 auto owner=std::make_shared<int>(1);std::weak_ptr<int> weak=owner;
 {
  NativeRetirementContext c;const std::array<int,2> duplicate{0,0};
  assert(!c.prepare(owner,duplicate)&&!c.prepared()&&!c.generation());
  assert(!c.prepare({},devices)&&!c.drain_after_producers_closed());
  assert(c.prepare(owner,devices)&&c.prepared()&&c.covers(0)&&c.covers(1)&&!c.covers(7));
  assert(gets==0&&sets==0&&syncs==0);const auto generation=c.generation();
  NativeRetirementContext foreign;assert(foreign.prepare(owner,devices)&&foreign.generation()!=generation);
  owner.reset();assert(!weak.expired());
  assert(c.drain_after_producers_closed()&&c.drained()&&!c.prepared());
  assert(drained==std::vector<int>({0,1})&&current==7&&syncs==2);
  const auto total=gets+sets+syncs;assert(c.drain_after_producers_closed()&&total==gets+sets+syncs);
  assert(!c.prepare(weak.lock(),devices)&&c.generation()==generation);
 }
 assert(weak.expired());
 for(int failure=0;failure<6;++failure){
  reset();auto held=std::make_shared<int>(2);NativeRetirementContext c;assert(c.prepare(held,devices));
  if(failure==0)fail_get=1;
  else if(failure<=3)fail_set=failure; // first/second selection and final restoration
  else fail_sync=failure-3;
  assert(!c.drain_after_producers_closed()&&!c.drained()&&!c.prepared());
  if(failure!=3)assert(current==7);
  const auto total=gets+sets+syncs;assert(!c.drain_after_producers_closed()&&gets+sets+syncs==total);
  assert(!c.prepare(held,devices));
 }
}
