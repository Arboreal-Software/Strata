#include "strata/core/session.hpp"
#include "strata/kernels/ngram.hpp"
#include <cassert>
#include <iostream>
#include <vector>
using namespace strata::core;
static int fail_at=0, calls=0;
static int capture_failure=0;
static std::vector<uintptr_t> destroyed;
extern "C" cudaError_t cudaGraphExecDestroy(cudaGraphExec_t h) {
    ++calls; if (calls==fail_at) return cudaErrorUnknown;
    destroyed.push_back(reinterpret_cast<uintptr_t>(h)); return cudaSuccess;
}
extern "C" cudaError_t cudaGraphDestroy(cudaGraph_t h) {
    ++calls; if (calls==fail_at) return cudaErrorUnknown;
    destroyed.push_back(reinterpret_cast<uintptr_t>(h)); return cudaSuccess;
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t h) {
    ++calls; if (calls==fail_at) return cudaErrorUnknown;
    destroyed.push_back(reinterpret_cast<uintptr_t>(h)); return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t) { return "injected"; }
extern "C" cudaError_t cudaStreamCreate(cudaStream_t* s) { *s=reinterpret_cast<cudaStream_t>(100); return capture_failure==1?cudaErrorUnknown:cudaSuccess; }
extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { return capture_failure==2?cudaErrorUnknown:cudaSuccess; }
extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* g) { *g=reinterpret_cast<cudaGraph_t>(101); return capture_failure==3?cudaErrorUnknown:cudaSuccess; }
extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t* x, cudaGraph_t, unsigned long long) { *x=reinterpret_cast<cudaGraphExec_t>(102); return capture_failure==4?cudaErrorUnknown:cudaSuccess; }
namespace strata::kernels {
bool PleTable::is_open() const { return false; }
PleTable* PleReadContext::table() const { return nullptr; }
}
namespace strata::core {
WeightTable::~WeightTable() = default;
bool block_layer_pre(const WeightTable&,const ModelGeometry&,int64_t,int64_t,int32_t,const GdnBuffers&,const QsaState&,const QsaBuffers&,const MoEBuffers&,int64_t,const BlockBuffers&,void*,std::string&,const Doorbell*,const PleRun*,int,int) {return capture_failure!=5;}
bool block_layer_post(const WeightTable&,const ModelGeometry&,int64_t,int64_t,const MoEBuffers&,const BlockBuffers&,const float*,void*,std::string&) {return true;}
}
static void reset(int failure) { fail_at=failure; calls=0; destroyed.clear(); }
static SessionGraphs graphs() {
    SessionGraphs g; g.n=1; g.allocated_slots=3; g.captured=g.split_captured=true;
    uintptr_t id=1;
    auto fill=[&](cudaGraphExec_t*& a){a=new cudaGraphExec_t[3](); a[0]=reinterpret_cast<cudaGraphExec_t>(id++); a[2]=reinterpret_cast<cudaGraphExec_t>(id++);};
    fill(g.execs); fill(g.posts); fill(g.preA); fill(g.preB);
    for(auto& a:g.preP)fill(a);
    g.pending_graph=reinterpret_cast<cudaGraph_t>(id++); g.pending_stream=reinterpret_cast<cudaStream_t>(id); return g;
}
// Only test fixture arrays are reclaimed after sticky failure; never retry CUDA handles.
static void discard_arrays(SessionGraphs& g) {
    delete[]g.execs;delete[]g.posts;delete[]g.preA;delete[]g.preB;
    for(auto a:g.preP)delete[]a;
}
int main(){
    std::string err;
    for(int failure=0;failure<=20;++failure){
        reset(failure);auto g=graphs();bool ok=session_graphs_free_checked(g,err);
        assert(ok==(failure==0));assert(!g.captured&&!g.split_captured);
        if(ok){assert(calls==20);assert(!g.execs&&!g.pending_graph&&!g.pending_stream&&!g.allocated_slots);assert(session_graphs_free_checked(g,err));assert(calls==20);}
        else{
            assert(calls==failure&&g.retirement_failed&&g.retirement_started&&!err.empty());
            assert(destroyed.size()==size_t(failure-1));
            auto check=[&](cudaGraphExec_t* a){if(a)for(int i=0;i<3;++i)if(a[i])assert(reinterpret_cast<uintptr_t>(a[i])>=uintptr_t(failure));};
            check(g.execs);check(g.posts);check(g.preA);check(g.preB);for(auto a:g.preP)check(a);
            assert(!session_graphs_free_checked(g,err));session_graphs_free(g);assert(calls==failure);
            discard_arrays(g);
        }
    }
    for(int failure=0;failure<=3;++failure){
        reset(failure);TokenGraph g;g.exec=reinterpret_cast<cudaGraphExec_t>(1);g.pending_graph=reinterpret_cast<cudaGraph_t>(2);g.pending_stream=reinterpret_cast<cudaStream_t>(3);g.captured=true;
        assert(token_graph_free_checked(g,err)==(failure==0));assert(!g.captured);
        if(failure){assert(g.retirement_failed);assert(bool(g.exec)==(failure==1));assert(bool(g.pending_graph)==(failure<=2));assert(g.pending_stream);assert(!token_graph_free_checked(g,err));token_graph_free(g);assert(calls==failure);}
        else{assert(!g.exec&&!g.pending_graph&&!g.pending_stream);assert(token_graph_free_checked(g,err));assert(calls==3);}
    }
    // Raw-graph destroy uncertainty during capture must never be retried by retirement.
    reset(0);auto g=graphs();g.retirement_failed=true;assert(!session_graphs_free_checked(g,err));assert(calls==0);discard_arrays(g);
    TokenGraph t;t.pending_graph=reinterpret_cast<cudaGraph_t>(1);t.retirement_failed=true;assert(!token_graph_free_checked(t,err));assert(calls==0);
    for(int failure=0;failure<=7;++failure) {
        reset(failure==6?1:failure==7?2:0);capture_failure=failure;
        ModelGeometry geometry;geometry.n_layers=1;geometry.qsa_interval=1;
        SessionState state;QsaState qsa;state.qsa_states=&qsa;
        WeightTable weights;SessionGraphs capture;
        const bool ok=session_capture(weights,geometry,state,nullptr,capture,err);
        assert(ok==(failure==0));
        if(failure==1||failure==2)assert(capture.pending_stream);
        if(failure==3)assert(capture.pending_stream&&capture.pending_graph);
        if(failure==4)assert(capture.execs[0]&&capture.pending_graph);
        if(failure==6||failure==7) {
            assert(capture.retirement_failed);const int old=calls;
            assert(!session_graphs_free_checked(capture,err));assert(calls==old);discard_arrays(capture);
        } else { fail_at=0;assert(session_graphs_free_checked(capture,err));assert(!capture.pending_stream&&!capture.pending_graph); }
    }
    std::cout<<"SessionGraphs/TokenGraph retirement: all injected failures passed\n";
}
