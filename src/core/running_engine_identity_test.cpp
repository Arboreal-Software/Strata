// Compile BOTH this test and running_engine_identity.cpp with STRATA_RUNNING_ENGINE_IDENTITY_TEST.
// Actual strict manifest factory; only current-process syscall path is mocked. No actual /proc executable reads.
#include "strata/core/verified_input_manifest.hpp"
#include "strata/core/running_engine_identity.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace strata::core;
namespace fs=std::filesystem;
static StateDigest hash(const std::string& s){WindowHasher h;h.update(s.data(),s.size());return h.digest();}
static std::string hex(const StateDigest& d){std::string s;const char* h="0123456789abcdef";for(auto b:d){s+=h[b>>4];s+=h[b&15];}return s;}
static void write(const std::string& p,const std::string& s){std::ofstream f(p,std::ios::binary|std::ios::trunc);f<<s;f.close();assert(f);}
static WindowBudget budget(uint64_t host=32<<20,uint64_t disk=32<<20,uint64_t io=1){return WindowBudget(WindowResources{{0},host,0,disk,io,0},WindowResources{{0},0,0,0,0,0});}
static VerifiedArtifactOptions options(){VerifiedArtifactOptions o;o.fault=[](const char* s){return std::string(s)=="clone"?EOPNOTSUPP:0;};return o;}
struct Fixture {
    std::string dir,path,text;
    Fixture(){char p[]="/tmp/strata-manifest-XXXXXX";auto* m=mkdtemp(p);assert(m);dir=m;path=dir+"/manifest.tsv";
        std::string gguf;auto n=[&](uint64_t v,int count){for(int i=0;i<count;++i)gguf+=char(v>>(i*8));};auto str=[&](std::string s){n(s.size(),8);gguf+=s;};n(0x46554747,4);n(3,4);n(0,8);n(1,8);str("general.architecture");n(8,4);str("qwen4exp");
        std::vector<std::pair<std::string,std::string>> files={{"model.gguf",gguf},{"index.txt","# align 16 pool 16 tensors 1\nw 0 2 0 4 0 4 1 1 0 0 0 0 0 4 0 0 0 0\n"},{"dense.bin","data"}};
        text="STRATA_VERIFIED_INPUT_MANIFEST\t1\n";
        for(const auto& [key,value]:std::vector<std::pair<std::string,uint64_t>>{{"max_files",32},{"max_aliases",64},{"max_path_bytes",256},{"max_depth",8},{"max_metadata_bytes",4096},{"view_host_bytes",8192},{"view_disk_bytes",8192},{"artifact_max_file_bytes",1<<20},{"artifact_block_bytes",64},{"artifact_metadata_host_bytes",4096},{"artifact_disk_overhead_bytes",4096},{"allow_full_copy",1}})text+="LIMIT\t"+key+"\t"+std::to_string(value)+"\n";
        for(const auto& [name,bytes]:files){write(dir+"/"+name,bytes);text+="FILE\t"+name+"\t"+std::to_string(bytes.size())+"\t"+hex(hash(bytes))+"\t"+dir+"/"+name+"\n";text+="ALIAS\t"+name+"\t"+(name=="model.gguf"?"models/":"pack/")+name+"\n";}
        text+="GGUF\tmain\t0\tmodels/model.gguf\nPACK\tpack\tmodels/model.gguf\n";
        for(const auto& [role,value]:std::vector<std::pair<std::string,std::string>>{{"pack","pack"},{"native-primary","models/model.gguf"},{"native-group","main"},{"dense-group","main"},{"head-primary","-"},{"head-group","-"},{"embedding","-"},{"ple","-"}})text+="BIND\t"+role+"\t"+value+"\n";
        write(path,text);
    }
    ~Fixture(){fs::remove_all(dir);}
};
static const VerifiedManifestParseLimits bounds{65536,256,4096,4096};
static std::string replace(std::string text,const std::string& from,const std::string& to){auto at=text.find(from);assert(at!=std::string::npos);text.replace(at,from.size(),to);return text;}

#include <algorithm>
#include <cstring>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <csignal>
namespace strata::core::running_engine_test {
static std::string image;
static int opened=0,closed=0,stats=0,open_attempts=0,pid_calls=0,stat_fail_at=0;
static int64_t pid=123;
static bool read_error=false,changed=false,not_regular=false,open_error=false,close_error=false,pid_change=false;
static WindowBudget* close_budget=nullptr;
static WindowResources close_expected_resources;
static int close_expected_fd=-1,close_attempts=0,close_event_fd=-1,close_errno=EIO;
struct CloseEvent {int fd,attempt,error;};
static int interrupts=0;
int open_self(){++open_attempts;stats=0;pid_calls=0;if(open_error){errno=EACCES;return -1;}++opened;return 900+opened;}
int stat_fd(int,struct stat* s){++stats;if(stats==stat_fail_at){errno=EIO;return -1;}*s={};s->st_mode=not_regular?S_IFIFO:S_IFREG;s->st_dev=2;s->st_ino=3;s->st_size=image.size();if(changed&&stats==2)++s->st_ino;return 0;}
ssize_t read_at(int,void* p,size_t n,off_t off){
 if(interrupts>0){--interrupts;errno=EINTR;return -1;}
 if(read_error){errno=EIO;return -1;}
 if(off<0||uint64_t(off)>=image.size())return 0;
 n=std::min<size_t>({n,image.size()-size_t(off),17});std::memcpy(p,image.data()+off,n);return ssize_t(n);
}
int close_fd(int fd){
 ++closed;
 if(close_error){
  const CloseEvent event{fd,++close_attempts,close_errno};
  if(::write(close_event_fd,&event,sizeof(event))!=ssize_t(sizeof(event)))std::_Exit(86);
  if(fd!=close_expected_fd||close_attempts!=1)std::_Exit(87);
  if(!close_budget||!(close_budget->snapshot().reserved==close_expected_resources))std::_Exit(83);
  errno=close_errno;return -1;
 }
 return 0;
}
int64_t process_id(){return pid_change&&++pid_calls>=2?pid+1:pid;}
}
static std::string native_elf(){
 std::string e(256,'\0');auto put=[&](size_t at,uint64_t v,size_t n){for(size_t j=0;j<n;++j)e[at+j]=char(v>>(8*j));};
 e.replace(0,4,"\177ELF",4);e[4]=2;e[5]=1;e[6]=1;put(16,3,2);
#if defined(__aarch64__)
 put(18,183,2);
#else
 put(18,62,2);
#endif
 put(20,1,4);put(24,120,8);put(32,64,8);put(52,64,2);put(54,56,2);put(56,1,2);
 put(64,1,4);put(68,1,4);put(96,256,8);put(104,256,8);return e;
}
int main(){
 using namespace strata::core::running_engine_test;
 Fixture f;auto b=budget(32<<20,32<<20,1);std::string error;image=native_elf();
 auto claim=[&](const std::string& expected){
  auto text=replace(f.text,"MANIFEST\t1\n","MANIFEST\t3\n")+"BIND\texpert-profile\t-\nMAIN_IMAGE\t"+std::to_string(expected.size())+"\t"+hex(hash(expected))+"\n";
  write(f.path,text);std::shared_ptr<const ParsedVerifiedInputs> plan;assert(ParsedVerifiedInputs::read(f.path,hash(text),bounds,b,plan));return plan->expected_running_executable(b);
 };
 auto expected=claim(image);RunningEngineLimits limits{1<<20,4096,128,1000};
 std::shared_ptr<const RunningEngineReceipt> out;
 const auto claim_host=b.snapshot().reserved.host_bytes;
 interrupts=1;assert(collect_running_engine(expected,limits,b,out,error));assert(out&&out->partial()&&out->observation());
 assert(out->observation()->content==hash(image)&&out->observation()->bytes==image.size()&&closed==0);
 assert(b.snapshot().reserved.io_slots==0);
 {auto next_loader=b.reserve({{0},0,0,0,1,0},ReservationPurpose::foreground);assert(next_loader);}
 // Receipt retains metadata only; read block capacity is no longer charged.
 assert(b.snapshot().reserved.host_bytes==claim_host+4096+sizeof(RunningEngineReceipt));
 ++pid;assert(!out->observation());--pid;assert(out->observation());
 const int retained_fd=900+opened;
 auto saved=out;
 auto failure=[&](const std::shared_ptr<const ExpectedRunningExecutable>& e,const RunningEngineLimits& lim,
                  WindowBudget q,const RunningEngineOptions& opts,const char* diagnostic){
  const auto held=q.snapshot().reserved;const auto active=opened-closed;
  assert(!collect_running_engine(e,lim,q,out,error,opts));
  assert(out==saved&&q.snapshot().reserved==held&&opened-closed==active);
  assert(error.find(diagnostic)!=std::string::npos);
 };
 open_error=true;failure(expected,limits,b,{},"open actual");open_error=false;
 for(int at:{1,2}){stat_fail_at=at;failure(expected,limits,b,{},at==1?"initial stat":"changed during");stat_fail_at=0;}
 read_error=true;failure(expected,limits,b,{},"short/error read");read_error=false;
 changed=true;failure(expected,limits,b,{},"changed during");changed=false;
 not_regular=true;failure(expected,limits,b,{},"type/size mismatch");not_regular=false;
 pid_change=true;failure(expected,limits,b,{},"process changed");pid_change=false;
 auto short_limit=limits;short_limit.max_read_calls=1;failure(expected,short_limit,b,{},"read-call bound");
 RunningEngineOptions cancelled;cancelled.cancelled=[] {return true;};int before=open_attempts;
 failure(expected,limits,b,cancelled,"cancelled");assert(open_attempts==before);
 auto different=image;different[255]=1;auto wrong=claim(different);failure(wrong,limits,b,{},"digest mismatch");
 auto good=image;image[4]=1;auto bad_elf=claim(image);failure(bad_elf,limits,b,{},"not native ELF64");image=good;
 for(int branch=0;branch<5;++branch){
  auto put=[&](size_t at,uint64_t v,size_t n){for(size_t j=0;j<n;++j)image[at+j]=char(v>>(8*j));};
  if(branch==0)put(32,250,8); // PH table outside file
  if(branch==1)put(56,1025,2); // bounded PH count
  if(branch==2)put(96,257,8); // file extent exceeds memory/file
  if(branch==3)put(68,0,4); // entry has no executable load
  if(branch==4)put(24,300,8); // entry outside file-backed segment
  auto malformed=claim(image);
  failure(malformed,limits,b,{},branch<2?"program geometry":branch==2?"load extent":"entry not");
  image=good;
 }
 auto tiny=budget(64);failure(expected,limits,tiny,{},"receipt quota");
 auto workspace_only=budget(4096+sizeof(RunningEngineReceipt)+1);before=open_attempts;
 failure(expected,limits,workspace_only,{},"workspace/IO quota");assert(open_attempts==before);
 // Every failure scenario gets its own child and parent-observed close events.
 // Compare the entire ledger, not a nonzero total masked by other live claims.
 for(bool during_cleanup:{false,true})for(int injected_errno:{EIO,EINTR}){
  const auto before_close=b.snapshot().reserved;
  int pipes[2],events[2];assert(pipe(pipes)==0);assert(pipe(events)==0);
  const pid_t child=fork();assert(child>=0);
  if(child==0){
   ::close(pipes[0]);::close(events[0]);
   assert(dup2(pipes[1],STDERR_FILENO)>=0);::close(pipes[1]);
   close_error=true;close_budget=&b;close_event_fd=events[1];close_errno=injected_errno;
   close_attempts=0;close_expected_resources=before_close;
   close_expected_fd=during_cleanup?900+opened+1:retained_fd;
   const uint64_t metadata=4096+sizeof(RunningEngineReceipt);
   if(during_cleanup){
    // Existing receipt/claims remain unchanged; new receipt plus still-live
    // read/hash workspace and its IO slot are charged at the failed FD close.
    close_expected_resources.host_bytes+=metadata+limits.block_bytes+65536;
    close_expected_resources.io_slots+=1;
    read_error=true;(void)collect_running_engine(expected,limits,b,out,error);std::_Exit(84);
   }
   // Explicit operation delta: remove unrelated owners' baseline, then require
   // this closing receipt's complete metadata charge still present.
   assert(close_expected_resources.host_bytes>=metadata);
   auto unrelated=before_close;unrelated.host_bytes-=metadata;
   close_expected_resources=unrelated;close_expected_resources.host_bytes+=metadata;
   saved.reset();out.reset();std::_Exit(85);
  }
  ::close(pipes[1]);::close(events[1]);
  int status=0;pid_t reaped=0;
  auto await_child=[&](int milliseconds){
   const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(milliseconds);
   do {
    reaped=waitpid(child,&status,WNOHANG);
    if(reaped==child)return true;
    if(reaped<0&&errno!=EINTR)return false;
    usleep(1000);
   }while(std::chrono::steady_clock::now()<deadline);
   return false;
  };
  const bool timely=await_child(3000);
  if(!timely){(void)kill(child,SIGKILL);(void)await_child(1000);}
  // Never block reading a pipe while the child might still hold its writer.
  if(reaped!=child){::close(pipes[0]);::close(events[0]);assert(false&&"close child not reaped within bound");}
  std::string diagnostic;char buf[256];ssize_t n;
  while((n=::read(pipes[0],buf,sizeof(buf)))>0)diagnostic.append(buf,size_t(n));
  ::close(pipes[0]);
  CloseEvent event{};const auto event_bytes=::read(events[0],&event,sizeof(event));
  char extra=0;const auto extra_bytes=::read(events[0],&extra,1);::close(events[0]);
  assert(timely&&WIFEXITED(status)&&WEXITSTATUS(status)==1);
  assert(event_bytes==ssize_t(sizeof(event))&&extra_bytes==0);
  assert(event.fd==(during_cleanup?900+opened+1:retained_fd)&&event.attempt==1&&event.error==injected_errno);
  assert(diagnostic=="strata: running image descriptor close uncertain; worker restart required\n");
  assert(b.snapshot().reserved==before_close);
 }

 saved.reset();out.reset();assert(opened==closed);expected.reset();wrong.reset();bad_elf.reset();assert(b.snapshot().reserved.host_bytes==0&&b.snapshot().reserved.io_slots==0);
}
