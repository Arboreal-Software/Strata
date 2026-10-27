#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static int phase=0,destroy_calls=0,fail_destroy_at=0,body_calls=0;
static std::set<void*> raw_live,exec_live,queue_live;
namespace sycl {
struct queue {queue(){queue_live.insert(this);}~queue(){queue_live.erase(this);}int get_context(){return 0;}int get_device(){return 0;}};
namespace ext::oneapi::experimental {
enum class graph_state {modifiable,executable};
template<graph_state> struct command_graph;
template<> struct command_graph<graph_state::modifiable> {
 bool recording=false;
 command_graph(int,int){if(phase==2)throw std::bad_alloc();raw_live.insert(this);}
 void begin_recording(queue&){recording=true;if(phase==3)throw std::runtime_error("begin failed");}
 void end_recording(){if(phase==6)throw std::runtime_error("end failed");recording=false;}
 int finalize(){if(phase==8)throw std::runtime_error("finalize failed");return 1;}
 ~command_graph()noexcept(false){check(!recording,"no destruction while producer is recording");raw_live.erase(this);if(++destroy_calls==fail_destroy_at||phase==10)throw std::runtime_error("raw graph delete failed");}
};
template<> struct command_graph<graph_state::executable> {
 explicit command_graph(int){if(phase==9)throw std::bad_alloc();exec_live.insert(this);}
 ~command_graph()noexcept(false){exec_live.erase(this);if(++destroy_calls==fail_destroy_at)throw std::runtime_error("executable delete failed");}
};
}}
namespace dpct {
using queue_ptr=sycl::queue*;
namespace experimental {
using command_graph_ptr=sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable>*;
using command_graph_exec_ptr=sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>*;
}
struct device {
 queue_ptr create_queue(bool){if(phase==1)throw std::bad_alloc();return new sycl::queue;}
 void destroy_queue(queue_ptr q){if(++destroy_calls==fail_destroy_at||phase==7)throw std::runtime_error("queue destroy failed");delete q;}
};
device& get_current_device(){static device d;return d;}
}
#include "graph_ownership.inc"
static void clear_faults(){phase=0;fail_destroy_at=0;}
static void cleanup_raw(dpct::experimental::command_graph_ptr p){if(raw_live.count(p)){p->recording=false;delete p;}}
static void cleanup_exec(dpct::experimental::command_graph_exec_ptr p){if(exec_live.count(p))delete p;}
static void cleanup(TokenGraph& t){clear_faults();cleanup_exec(t.exec);cleanup_raw(t.pending_graph);if(queue_live.count(t.pending_queue))delete t.pending_queue;t={};}
static void cleanup(SessionGraphs& g){clear_faults();auto release=[&](auto*& a){if(!a)return;for(int64_t i=0;i<g.allocated_slots;++i)cleanup_exec(a[i]);delete[]a;};release(g.execs);release(g.posts);release(g.preA);release(g.preB);for(auto& a:g.preP)release(a);cleanup_raw(g.pending_graph);if(queue_live.count(g.pending_queue))delete g.pending_queue;g={};}
int main(){
 for(int p=0;p<=10;++p){
  phase=p;destroy_calls=body_calls=0;TokenGraph t;std::string error;
  const bool ok=capture_owned_graph(t,t.exec,[&](dpct::queue_ptr){++body_calls;if(p==5)throw std::runtime_error("body threw");return p!=4;},error);
  if(p==0){check(ok&&t.exec&&!t.pending_graph&&!t.pending_queue,"successful capture transfers executable and clears temporary handles");t.captured=true;check(token_graph_free_checked(t,error)&&!t.exec&&!t.retirement_started,"successful retirement resets for reuse");}
  else {
   check(!ok,"injected capture failure refused");
   const bool sticky=p==3||p==6||p==7||p==10;
   check(t.retirement_failed==sticky,"uncertainty marked only for operations with ambiguous retirement/producer state");
   if(sticky){const int before=destroy_calls;check(!token_graph_free_checked(t,error)&&destroy_calls==before,"sticky capture uncertainty prevents all retirement retries");check(!capture_owned_graph(t,t.exec,[](auto*){return true;},error),"sticky capture cannot be reused");}
   else {clear_faults();check(token_graph_free_checked(t,error),"recoverable failed capture retires retained handles");}
  }
  cleanup(t);check(raw_live.empty()&&exec_live.empty()&&queue_live.empty(),"capture fixture cleaned");
 }
 for(int failure=0;failure<=20;++failure){
  clear_faults();destroy_calls=0;SessionGraphs g;g.allocated_slots=2;g.n=0;g.captured=g.split_captured=true;
  auto fill=[](dpct::experimental::command_graph_exec_ptr*& array){array=new dpct::experimental::command_graph_exec_ptr[2]();for(int i=0;i<2;++i)array[i]=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(1);};
  fill(g.execs);fill(g.posts);fill(g.preA);fill(g.preB);for(auto& pre:g.preP)fill(pre);
  g.pending_graph=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable>(0,0);g.pending_queue=new sycl::queue;
  fail_destroy_at=failure;std::string error;const bool ok=session_graphs_free_checked(g,error);
  check(!g.captured&&!g.split_captured,"retirement disables replay immediately");
  if(!failure){check(ok&&destroy_calls==20&&!g.allocated_slots&&!g.retirement_started,"all allocated slots retire even with zero completed layers");}
  else {check(!ok&&g.retirement_failed&&destroy_calls==failure,"partial retirement stops on first failure");const int before=destroy_calls;session_graphs_free(g);check(destroy_calls==before,"legacy wrapper does not retry sticky failure");}
  cleanup(g);check(raw_live.empty()&&exec_live.empty()&&queue_live.empty(),"retirement fixture cleaned");
 }
 {SessionGraphs g;g.execs=new dpct::experimental::command_graph_exec_ptr[1]();std::string error;const auto before=destroy_calls;check(!session_graphs_free_checked(g,error)&&g.retirement_failed&&destroy_calls==before,"unknown array capacity refuses cleanup");delete[]g.execs;}
 std::puts("SYCL graph ownership CPU seam test passed");
}
