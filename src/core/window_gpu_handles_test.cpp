#include "strata/core/window_gpu_handles.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
using strata::core::OwnedWindowGpuHandles;
enum Op {get,set,stream_create,event_create,sync_device,stream_destroy,event_destroy};
struct Shared {int n=0, fail=-1, device=9, destroyed_scope=0;bool oom=false,partial=false;Op ops[128]{};int devices[128]{};cudaStream_t stream_out=nullptr;cudaEvent_t event_out=nullptr;};
static Shared* s;
static cudaError_t call(Op op) {int i=s->n++;assert(i<128);s->ops[i]=op;s->devices[i]=s->device;return i==s->fail?(s->oom?cudaErrorMemoryAllocation:static_cast<cudaError_t>(999)):cudaSuccess;}
extern "C" cudaError_t cudaGetDevice(int* d){auto e=call(get);if(e==cudaSuccess)*d=s->device;return e;}
extern "C" cudaError_t cudaSetDevice(int d){auto e=call(set);if(e==cudaSuccess)s->device=d;return e;}
extern "C" cudaError_t cudaDeviceSynchronize(){return call(sync_device);}
extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t* p,unsigned){auto e=call(stream_create);if(e==cudaSuccess||s->partial)*p=reinterpret_cast<cudaStream_t>(uintptr_t(100+s->n));return e;}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* p,unsigned){auto e=call(event_create);if(e==cudaSuccess||s->partial)*p=reinterpret_cast<cudaEvent_t>(uintptr_t(200+s->n));return e;}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t){return call(stream_destroy);}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t){return call(event_destroy);}
static void reset(){*s=Shared{};s->stream_out=reinterpret_cast<cudaStream_t>(999);s->event_out=reinterpret_cast<cudaEvent_t>(999);}
struct Scope {~Scope(){s->destroyed_scope=1;}};
static void wait_failed(pid_t pid){int status=0;assert(waitpid(pid,&status,0)==pid);assert(WIFEXITED(status)&&WEXITSTATUS(status)==1);assert(s->destroyed_scope==0);assert(s->n==s->fail+1);}
int main(){
 s=static_cast<Shared*>(mmap(nullptr,sizeof(Shared),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));assert(s!=MAP_FAILED);reset();
 std::string error;
 {OwnedWindowGpuHandles empty(2,3);assert(empty.metadata_capacity_bytes()>0);}assert(s->n==0);
 {OwnedWindowGpuHandles empty(0,0);cudaStream_t out=reinterpret_cast<cudaStream_t>(999);assert(!empty.create_stream(0,0,out,error));assert(uintptr_t(out)==999&&s->n==0);empty.retire_after_producers_closed();assert(!empty.declare_consumer_device(0,error));assert(s->n==0);}
 reset();
 {OwnedWindowGpuHandles h(2,3);cudaStream_t a=nullptr;cudaEvent_t b=nullptr;assert(h.create_stream(0,0,a,error));assert(h.create_event(1,0,b,error));assert(h.declare_consumer_device(2,error));assert(h.declare_consumer_device(2,error));assert(h.stream_count()==1&&h.event_count()==1&&h.device_count()==3);assert(s->device==9);int start=s->n;cudaStream_t unchanged=reinterpret_cast<cudaStream_t>(999);assert(!h.create_stream(0,0,unchanged,error));assert(uintptr_t(unchanged)==999&&s->n==start);assert(!h.declare_consumer_device(3,error)&&s->n==start);h.drain_after_producers_closed();assert(s->n==start+8);assert(h.stream_count()==1&&h.event_count()==1);for(int i=start;i<s->n;++i)assert(s->ops[i]!=stream_destroy&&s->ops[i]!=event_destroy);auto drained=s->n;h.drain_after_producers_closed();assert(!h.create_stream(0,0,unchanged,error)&&!h.declare_consumer_device(0,error));assert(s->n==drained);h.retire_after_producers_closed();assert(s->n==start+14);assert(s->ops[start]==get);for(int i=0;i<3;++i){assert(s->ops[start+2+2*i]==sync_device);assert(s->devices[start+2+2*i]==i);}assert(s->ops[start+10]==stream_destroy&&s->devices[start+10]==0);assert(s->ops[start+12]==event_destroy&&s->devices[start+12]==1);assert(s->device==9&&h.stream_count()==0&&h.event_count()==0);start=s->n;h.retire_after_producers_closed();assert(!h.create_event(0,0,b,error)&&!h.declare_consumer_device(0,error));assert(s->n==start);}
 // Creation OOM before a returned handle is recoverable, preserving earlier owned handles.
 for(bool event:{false,true}){reset();OwnedWindowGpuHandles h(3,3);cudaStream_t first=nullptr;assert(h.create_stream(0,0,first,error));s->fail=s->n+2;s->oom=true;cudaStream_t a=reinterpret_cast<cudaStream_t>(999);cudaEvent_t b=reinterpret_cast<cudaEvent_t>(999);assert(!(event?h.create_event(1,0,b,error):h.create_stream(1,0,a,error)));assert(uintptr_t(a)==999&&uintptr_t(b)==999&&s->device==9);assert(h.stream_count()==1&&h.event_count()==0);s->fail=-1;assert(event?h.create_event(1,0,b,error):h.create_stream(1,0,a,error));}
 // Every get/select/create/restore uncertainty fail-stops without unwinding any later owner.
 for(bool event:{false,true})for(int point=0;point<4;++point){reset();s->fail=point;auto pid=fork();assert(pid>=0);if(!pid){Scope sentinel;OwnedWindowGpuHandles h(2,2);if(event)h.create_event(0,0,s->event_out,error);else h.create_stream(0,0,s->stream_out,error);_exit(77);}wait_failed(pid);assert(uintptr_t(s->stream_out)==999&&uintptr_t(s->event_out)==999);}
 // OOM with a partially returned handle is uncertain, never a recoverable refusal.
 for(bool event:{false,true}){reset();s->fail=2;s->oom=true;s->partial=true;auto pid=fork();assert(pid>=0);if(!pid){Scope sentinel;OwnedWindowGpuHandles h(1,1);cudaStream_t a=nullptr;cudaEvent_t b=nullptr;if(event)h.create_event(0,0,b,error);else h.create_stream(0,0,a,error);_exit(77);}wait_failed(pid);}
 // Inject each retirement call after two successful creations, including all drains,
 // each partial destruction and final device restoration. No later call may execute.
 for(int point=0;point<14;++point){reset();auto pid=fork();assert(pid>=0);if(!pid){Scope sentinel;OwnedWindowGpuHandles h(2,3);cudaStream_t a=nullptr;cudaEvent_t b=nullptr;assert(h.create_stream(0,0,a,error));assert(h.create_event(1,0,b,error));assert(h.declare_consumer_device(2,error));s->fail=s->n+point;h.retire_after_producers_closed();_exit(77);}wait_failed(pid);}
 reset();{OwnedWindowGpuHandles h(2,1);cudaStream_t a=nullptr;assert(!h.create_stream(-1,0,a,error));assert(s->n==0);assert(h.create_stream(0,0,a,error));auto n=s->n;cudaEvent_t b=reinterpret_cast<cudaEvent_t>(999);assert(!h.create_event(1,0,b,error));assert(uintptr_t(b)==999&&s->n==n);}
 munmap(s,sizeof(Shared));std::puts("window_gpu_handles_test passed");
}
