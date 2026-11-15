namespace strata::core {
struct RemoteOwnershipTestAccess {
    static void prime(RemoteExperts& r,float* x,float* out,void* metadata) {
        r.device_=1;r.stream_=(void*)1;r.read_stream_=(void*)2;r.n_expert_=2;
        r.owned_.resize(80);r.group_of_.resize(80);r.layers_present_={1};
        r.h_x_=x;r.h_out_=out;r.h_meta_=metadata;
    }
    static void no_streams(RemoteExperts& r){r.stream_=nullptr;r.read_stream_=nullptr;}
    static void no_hits(RemoteExperts& r){r.layers_present_[0]=0;}
    static void exhaust(RemoteExperts& r){r.next_generation_=UINT64_MAX;}
    static bool pending(RemoteExperts& r){return r.compute_generation_!=0;}
    static uint32_t readers(RemoteExperts& r){return r.copy_readers_;}
    static bool closed(RemoteExperts& r){return r.owned_closed_;}
};
}
using namespace strata::core;
[[noreturn]] void assertion_failed(const char* s){std::fprintf(stderr,"ASSERT: %s\n",s);std::fflush(stderr);std::_Exit(86);}
void require(bool b,const char* s){if(!b)assertion_failed(s);}
struct Owner final:NativeRuntimeAttachment {
    RemoteExperts* reenter=nullptr; int* destroyed=nullptr;
    ~Owner(){if(reenter)(void)reenter->owned_routing_healthy();if(destroyed)++*destroyed;}
};
struct Fixture {
    float x[32]{},y[32]{}; alignas(16) unsigned char meta[4096]{};
    std::shared_ptr<Owner> owner=std::make_shared<Owner>();
    std::string error;
    RemoteExperts r;
    Fixture(bool bind=true){gpu={};if(bind)require(r.set_strict_retirement(true,{0,1},error),"strict retirement");RemoteOwnershipTestAccess::prime(r,x,y,meta);if(bind)require(r.bind_owned_routing(owner,2,error),"bind");}
    bool begin(){int32_t ids[10]={0,1,1,1,1,1,1,1,1,1};return r.begin(0,x,ids,1,10,nullptr,nullptr,error);}
};
struct OwningRuntime final:NativeRuntimeAttachment {
    float x[32]{},y[32]{};alignas(16) unsigned char meta[4096]{};
    int* destroyed=nullptr;
    RemoteExperts helper; // actual strong runtime ownership of the helper
    explicit OwningRuntime(int& count):destroyed(&count){std::string error;require(helper.set_strict_retirement(true,{0,1},error),"actual strict setup");RemoteOwnershipTestAccess::prime(helper,x,y,meta);}
    ~OwningRuntime(){++*destroyed;}
    bool begin(std::string& error){int32_t ids[10]={0,1,1,1,1,1,1,1,1,1};return helper.begin(0,x,ids,1,10,nullptr,nullptr,error);}
};
static_assert(!std::is_copy_constructible_v<RemoteExperts::RoutingRead>);
static_assert(std::is_nothrow_move_constructible_v<RemoteExperts::RoutingRead>);
static_assert(!std::is_copy_constructible_v<RemoteExperts::PlacementWrite>);
static_assert(std::is_nothrow_move_constructible_v<RemoteExperts::PlacementWrite>);
RemoteExperts* in_copy=nullptr;
void writer_during_copy(){RemoteExperts::PlacementWrite w;require(in_copy->try_placement_write(w)==RemoteExperts::Admission::busy,"copy excludes writer without deadlock");}
RemoteExperts* overlapping_copy=nullptr;
void fail_copy_restore_during_finish(){
    gpu.during_sync=nullptr;
    int32_t l=0,e=0;uint8_t data[8]{},hit=0;uint8_t* dst=data;std::string error;
    gpu.fail_set=gpu.set_calls+2; // nested copy select succeeds, its restore fails
    require(!overlapping_copy->copy_cached_blobs(&l,&e,&dst,1,&hit,error),"overlapping copy restoration fails");
}
void ordinary() {
    {
        gpu={};RemoteExperts unopened;std::string error;
        require(unopened.set_strict_retirement(true,{0,1},error),"preopen strict config");
        unopened.close();require(gpu.get_calls==0 && gpu.sync_calls==0 && gpu.free_calls==0,"unopened has no invented device/drain");
    }
    {
        Fixture partial(false);require(partial.r.set_strict_retirement(true,{0,1},partial.error),"prebind partial strict");
        RemoteOwnershipTestAccess::no_streams(partial.r);partial.r.close();
        require(gpu.sync_calls==0 && gpu.free_calls>0,"prebind buffer cleanup without fabricated stream");
    }
    {
        gpu={};int destroyed=0;std::string error;
        auto runtime=std::make_shared<OwningRuntime>(destroyed);
        require(runtime->helper.bind_owned_routing(runtime,2,error),"bind actual containing runtime");
        auto weak=std::weak_ptr<OwningRuntime>(runtime);RemoteExperts::RoutingRead scope;
        require(runtime->helper.try_routing_read(scope)==RemoteExperts::Admission::acquired && runtime->begin(error),"actual runtime compute");
        scope.reset();runtime.reset();
        require(!weak.expired() && destroyed==0,"pending compute retains actual helper and backing storage");
        auto drain=weak.lock();require(drain->helper.finish(drain->x,error),"actual retained helper finish");
        drain.reset();require(weak.expired() && destroyed==1,"successful drain removes actual ownership cycle");
    }
    {
        Fixture f(false);RemoteExperts::RoutingRead g;
        require(f.r.try_routing_read(g)==RemoteExperts::Admission::unbound,"legacy unbound");
        require(!f.r.bind_owned_routing(f.owner,2,f.error),"binding requires actual strict setup");
        RemoteOwnershipTestAccess::no_hits(f.r);
        require(f.begin() && f.r.finish(f.x,f.error),"legacy no outer guard required");
    }
    {
        Fixture f;require(!f.r.bind_owned_routing(f.owner,2,f.error),"no rebind");
        require(!f.begin(),"managed begin requires outer pre-plan scope");
        RemoteExperts::RoutingRead g,g2;RemoteExperts::PlacementWrite w;
        require(f.r.try_routing_read(g)==RemoteExperts::Admission::acquired,"routing admit");
        require(f.r.try_routing_read(g2)==RemoteExperts::Admission::busy,"shared optimizer is serial");
        require(f.r.holds(0,0),"routing mapping");
        require(f.r.try_placement_write(w)==RemoteExperts::Admission::busy,"routing blocks writer before begin");
        require(f.begin(),"compute submit");
        const int copies=gpu.copy_calls;
        require(!f.begin() && gpu.copy_calls==copies,"second begin has no effects");
        g.reset();
        require(f.r.try_placement_write(w)==RemoteExperts::Admission::busy,"inner read survives outer scope");
        require(f.r.finish(f.x,f.error),"compute drain");
        require(!f.r.finish(f.x,f.error),"duplicate finish refused");
        require(f.r.try_placement_write(w)==RemoteExperts::Admission::acquired,"writer after exact completion");
        require(!f.r.holds(0,0),"writer excludes mapping reads");
        require(!w.mark_attempted(),"must drain old readers before overwrite");
        const int syncs=gpu.sync_calls;
        require(w.drain_before_overwrite(f.error) && gpu.sync_calls==syncs+2,"both old streams drained");
        require(w.mark_attempted(),"first attempt");
        require(!w.mark_attempted() && !w.finish_publication(),"no repeated attempt/no early publish");
        require(w.drain_copies(f.error),"copy drain");
        require(w.finish_publication() && !w,"publish consumes writer");
        require(!w.finish_publication(),"stale consumed writer");
        require(f.r.owned_routing_healthy(),"healthy after writer");
        RemoteOwnershipTestAccess::no_hits(f.r);
        require(f.r.try_routing_read(g)==RemoteExperts::Admission::acquired && f.begin(),"no-hit admitted pair");
        require(RemoteOwnershipTestAccess::pending(f.r),"no-hit retains pair");
        require(f.r.finish(f.x,f.error),"no-hit retires pair");g.reset();
        RemoteOwnershipTestAccess::exhaust(f.r);
        require(f.r.try_routing_read(g)==RemoteExperts::Admission::exhausted,"generation no wrap");
        require(f.r.try_placement_write(w)==RemoteExperts::Admission::exhausted,"writer generation no wrap");
    }
    {
        Fixture f;int32_t l=0,e=0;uint8_t data[8]{},hit=0;uint8_t* dst=data;
        in_copy=&f.r;gpu.during_sync=writer_during_copy;
        require(f.r.copy_cached_blobs(&l,&e,&dst,1,&hit,f.error) && hit==1,"copy checked completion");
        gpu.during_sync=nullptr;in_copy=nullptr;
        require(RemoteOwnershipTestAccess::readers(f.r)==0,"copy retires after mutex");
        gpu.fail_copy=gpu.copy_calls+1;
        require(!f.r.copy_cached_blobs(&l,&e,&dst,1,&hit,f.error),"failed submission drains borrowed destination");
        require(f.r.owned_routing_healthy(),"drained refusal not uncertain DMA");
        RemoteExperts::PlacementWrite w;
        require(f.r.try_placement_write(w)==RemoteExperts::Admission::acquired,"no stranded copy");
        require(!f.r.copy_cached_blobs(&l,&e,&dst,1,&hit,f.error) && hit==0,"writer refuses copy");w.reset();
    }
    {
        Fixture first,second;RemoteExperts::RoutingRead g;RemoteExperts::PlacementWrite a,b;
        require(second.r.try_routing_read(g)==RemoteExperts::Admission::acquired,"second peer read");
        require(first.r.try_placement_write(a)==RemoteExperts::Admission::acquired,"first peer writer");
        require(second.r.try_placement_write(b)==RemoteExperts::Admission::busy,"all peer acquisition busy");
        a.reset();g.reset();
        require(first.r.owned_routing_healthy(),"unattempted rollback healthy");
    }
    {
        Fixture f;int destroyed=0;f.owner->destroyed=&destroyed;f.owner->reenter=&f.r;
        RemoteExperts::RoutingRead a;
        require(f.r.try_routing_read(a)==RemoteExperts::Admission::acquired,"owner reader");
        auto weak=std::weak_ptr<Owner>(f.owner);f.owner.reset();
        RemoteExperts::RoutingRead moved(std::move(a));a.reset();
        require(!a && !weak.expired(),"move preserves actual owner");
        moved.reset();require(destroyed==1 && weak.expired(),"last owner release reenters outside mutex");
    }
}
void child_case(int mode){
    if(mode==12 || mode==13){
        Fixture partial(false);require(partial.r.set_strict_retirement(true,{0,1},partial.error),"prebind partial strict failure");
        if(mode==12)RemoteOwnershipTestAccess::no_streams(partial.r);
        gpu.fail_free=mode==12;gpu.fail_destroy=mode==13;
        partial.r.close();assertion_failed("prebind strict cleanup ignored release failure");
    }
    if(mode==9){
        gpu={};int destroyed=0;std::string error;
        auto runtime=std::make_shared<OwningRuntime>(destroyed);
        require(runtime->helper.bind_owned_routing(runtime,2,error),"bind actual unknown runtime");
        auto weak=std::weak_ptr<OwningRuntime>(runtime);RemoteExperts::PlacementWrite writer;
        require(runtime->helper.try_placement_write(writer)==RemoteExperts::Admission::acquired &&
                writer.drain_before_overwrite(error) && writer.mark_attempted(),"actual unknown attempt");
        runtime.reset();writer.reset();
        require(!weak.expired() && destroyed==0,"unknown retains actual helper cache and backing buffers");
        auto retained=weak.lock();require(!retained->helper.owned_routing_healthy(),"actual unknown helper stays fenced");
        std::_Exit(0);
    }
    Fixture f;RemoteExperts::RoutingRead g;RemoteExperts::PlacementWrite w;
    if(mode==0){ // unretired pair cannot close, even if outer guard was dropped
        require(f.r.try_routing_read(g)==RemoteExperts::Admission::acquired && f.begin(),"missing finish setup");
        g.reset();f.r.close();assertion_failed("unsafe close returned");
    }
    if(mode==1){
        int32_t l=0,e=0;uint8_t data[8]{},hit=0;uint8_t* dst=data;gpu.fail_sync=1;
        (void)f.r.copy_cached_blobs(&l,&e,&dst,1,&hit,f.error);assertion_failed("uncertain D2H returned");
    }
    if(mode==10 || mode==11){
        gpu.fail_free=mode==10;gpu.fail_destroy=mode==11;
        f.r.close();assertion_failed("unchecked managed resource retirement returned");
    }
    const auto count=RemoteExperts::owned_routing_quarantines();
    if(mode==2){
        require(f.r.try_placement_write(w)==RemoteExperts::Admission::acquired,"unknown writer");
        require(w.drain_before_overwrite(f.error) && w.mark_attempted(),"unknown attempt");
        auto weak=std::weak_ptr<Owner>(f.owner);f.owner.reset();w.reset();
        require(!weak.expired() && !f.r.owned_routing_healthy(),"attempted abandonment retains runtime");
    }else if(mode==3){
        require(f.r.try_routing_read(g)==RemoteExperts::Admission::acquired && f.begin(),"failed finish setup");
        gpu.fail_sync=gpu.sync_calls+1;
        require(!f.r.finish(f.x,f.error),"finish failure");g.reset();
        require(f.r.try_placement_write(w)==RemoteExperts::Admission::unknown,"failed finish never idle");
    }else if(mode==4){
        require(f.r.try_placement_write(w)==RemoteExperts::Admission::acquired,"restore failure setup");
        gpu.fail_set=gpu.set_calls+2;
        require(!w.drain_before_overwrite(f.error),"writer checks restore");w.reset();
    }else if(mode==5){
        require(f.r.try_routing_read(g)==RemoteExperts::Admission::acquired,"failed submission setup");
        gpu.fail_copy=1;require(!f.begin(),"compute submission fails");g.reset();
        require(RemoteOwnershipTestAccess::pending(f.r),"failed compute retains exact operation");
    }
    if(mode==6){
        require(f.r.try_routing_read(g)==RemoteExperts::Admission::acquired,"badalloc compute setup");
        gpu.fail_copy=1;gpu.arm_alloc_on_copy_error=true;
        bool caught=false;try{(void)f.begin();}catch(const std::bad_alloc&){caught=true;}
        fail_allocations=false;require(caught,"persistent allocation failure reached actual diagnostic");g.reset();
        require(RemoteOwnershipTestAccess::pending(f.r),"exception retains compute owner");
    }else if(mode==7){
        require(f.r.try_placement_write(w)==RemoteExperts::Admission::acquired,"badalloc writer setup");
        gpu.fail_sync=1;gpu.arm_alloc_on_sync_error=true;
        bool caught=false;try{(void)w.drain_before_overwrite(f.error);}catch(const std::bad_alloc&){caught=true;}
        fail_allocations=false;require(caught,"persistent failure reached writer diagnostic");w.reset();
    }
    if(mode==8){
        require(f.r.try_routing_read(g)==RemoteExperts::Admission::acquired && f.begin(),"overlap finish setup");
        overlapping_copy=&f.r;gpu.during_sync=fail_copy_restore_during_finish;
        require(!f.r.finish(f.x,f.error),"successful compute drain cannot hide concurrent copy poison");
        g.reset();require(RemoteOwnershipTestAccess::pending(f.r),"poisoned completion retains compute");
    }
    require(RemoteExperts::owned_routing_quarantines()==count+1,"one quarantine accounted");
    require(!f.r.owned_routing_healthy(),"unknown never healthy");
    std::_Exit(0); // intentionally quarantined owner; no pretending teardown succeeded
}
struct ChildResult {
 int status=0;std::string diagnostics;bool timed_out=false,reaped=false,no_child=false,stderr_eof=false,status_lost=false;int kills=0;
 const char* failure=nullptr;
};
enum class ChildBehavior {normal,hang_open,hang_closed,status_loss};
static ChildResult scenario(int mode,bool inject=false,bool noisy=false,
                           ChildBehavior behavior=ChildBehavior::normal){
 // This single-threaded harness owns child reaping. Refuse inherited auto-reap
 // or custom handlers before creating a PID; a concurrent reaper would break
 // the wait-before-kill ownership contract even after a successful preflight.
 struct sigaction disposition{};
 require(sigaction(SIGCHLD,nullptr,&disposition)==0&&disposition.sa_handler==SIG_DFL&&
       !(disposition.sa_flags&SA_NOCLDWAIT),"exclusive child status ownership required");
 int fds[2];require(pipe(fds)==0,"diagnostic pipe");
 const int flags=fcntl(fds[0],F_GETFL,0);
 if(flags<0||fcntl(fds[0],F_SETFL,flags|O_NONBLOCK)<0){close(fds[0]);close(fds[1]);require(false,"nonblocking diagnostic pipe");}
 std::fflush(nullptr);
 const auto pid=fork();
 if(pid<0){close(fds[0]);close(fds[1]);require(false,"fork");}
 if(!pid){
  close(fds[0]);require(dup2(fds[1],STDERR_FILENO)>=0,"diagnostic redirect");close(fds[1]);
  if(behavior==ChildBehavior::status_loss)std::_Exit(0);
  if(behavior!=ChildBehavior::normal){
   if(behavior==ChildBehavior::hang_closed)close(STDERR_FILENO);
   for(;;)pause();
  }
  if(noisy)std::fputs("FAIL: injected teardown diagnostic noise\nAddressSanitizer: injected diagnostic marker only\n",stderr);
  if(inject)assertion_failed("injected fatal oracle control");
  child_case(mode);std::_Exit(0);
 }
 close(fds[1]);
 // No allocating diagnostics operation occurs while an unreaped child is live.
 // Cleanup targets only our own unreaped PID: never a recycled/reaped PID.
 struct ChildOwner {
  pid_t pid;int fd;ChildBehavior behavior;int status=0,kills=0;
  bool owns_pid=true,reaped=false,status_lost=false,actually_reaped=false;
  pid_t wait(int flags) noexcept {
   const auto result=waitpid(pid,&status,flags);
   if(result==pid){
    actually_reaped=true;
    // Safe syscall seam: consume the real child's status FIRST, then simulate
    // loss. No risky recycled-PID signal is needed to test disarming.
    if(behavior==ChildBehavior::status_loss){errno=ECHILD;return -1;}
   }
   return result;
  }
  void observe(pid_t result,int error) noexcept {
   if(result==pid){reaped=true;owns_pid=false;}
   else if(result<0&&error==ECHILD){status_lost=true;owns_pid=false;}
  }
  void finish() noexcept {
   if(owns_pid){
    pid_t result;do{result=wait(WNOHANG);}while(result<0&&errno==EINTR);
    observe(result,errno);
    if(owns_pid&&result==0){
     // Test seam refuses to signal a PID whose real status was already reaped.
     if(actually_reaped)std::_Exit(86);
     ++kills;(void)kill(pid,SIGKILL);
     do{result=wait(0);}while(result<0&&errno==EINTR);
     observe(result,errno);
    }
   }
   if(fd>=0){close(fd);fd=-1;}
  }
  ~ChildOwner(){finish();}
 } owner{pid,fds[0],behavior};
 ChildResult out;std::array<char,65536> diagnostics{};size_t used=0;bool eof=false;
 const auto deadline=std::chrono::steady_clock::now()+
     std::chrono::milliseconds(behavior==ChildBehavior::normal?3000:150);
 while(!eof||!owner.reaped){
  if(std::chrono::steady_clock::now()>=deadline){out.timed_out=true;break;}
  // Bound work per pass so a noisy writer cannot starve the wait/deadline check.
  for(int chunk=0;!eof&&chunk<16;++chunk){
   char buffer[1024];const auto n=read(owner.fd,buffer,sizeof buffer);
   if(n<0){if(errno==EINTR)continue;if(errno==EAGAIN||errno==EWOULDBLOCK)break;out.failure="diagnostic read";break;}
   if(!n){eof=true;break;}
   if(size_t(n)>diagnostics.size()-used){out.failure="diagnostic limit";break;}
   std::memcpy(diagnostics.data()+used,buffer,size_t(n));used+=size_t(n);
  }
  if(out.failure)break;
  if(!owner.reaped){
   const auto waited=owner.wait(WNOHANG);const int wait_error=errno;
   owner.observe(waited,wait_error);
   if(waited<0&&wait_error!=EINTR){out.failure="wait child";break;}
  }
  if(eof&&owner.reaped)break;
  pollfd ready{owner.fd,POLLIN|POLLHUP,0};
  const int polled=poll(eof?nullptr:&ready,eof?0:1,10);
  if(polled<0&&errno!=EINTR){out.failure="diagnostic poll";break;}
 }
 // EOF is not process completion. Timeout/error must kill and reap even when
 // stderr was closed already. SIGKILL cleanup waits for kernel reap before any
 // failure is reported; this is not a promise to interrupt unkillable kernel IO.
 out.stderr_eof=eof;
 owner.finish();out.status=owner.status;out.reaped=owner.reaped;
 out.status_lost=owner.status_lost;out.kills=owner.kills;
 int probe=0;errno=0;out.no_child=waitpid(pid,&probe,WNOHANG)==-1&&errno==ECHILD;
 out.diagnostics.assign(diagnostics.data(),used);
 return out;
}

static bool accepted(const ChildResult& child,bool fatal){
    const char* exact="strata: remote helper ownership or DMA completion uncertain; worker restart required\n";
    return !child.timed_out && !child.status_lost && !child.failure && child.reaped && child.no_child &&
           WIFEXITED(child.status) && WEXITSTATUS(child.status)==(fatal?1:0) && child.diagnostics==(fatal?exact:"");
}
void fork_case(int mode,bool fatal){
    const auto child=scenario(mode);
    require(accepted(child,fatal),"exact child status/diagnostic, completed cleanup required");
}
void cleanup_controls(){
    for(auto behavior:{ChildBehavior::hang_open,ChildBehavior::hang_closed}){
        const auto child=scenario(0,false,false,behavior);
        require(child.timed_out && child.reaped && child.no_child && child.kills==1 && !child.failure &&
                WIFSIGNALED(child.status) && WTERMSIG(child.status)==SIGKILL && !accepted(child,true),
                "timeout must kill/reap before reporting, EOF is not completion");
    }
    const auto lost=scenario(0,false,false,ChildBehavior::status_loss);
    require(lost.status_lost && lost.no_child && !lost.reaped && lost.kills==0 && !accepted(lost,true),
            "lost wait ownership disarms PID without qualifying result");
    const auto assertion=scenario(0,true);
    require(assertion.reaped && assertion.no_child && WIFEXITED(assertion.status) &&
            WEXITSTATUS(assertion.status)==86 && !accepted(assertion,true),"assertion cannot impersonate fatal");
    const auto noisy=scenario(0,false,true);
    require(noisy.reaped && noisy.no_child && WIFEXITED(noisy.status) && WEXITSTATUS(noisy.status)==1 &&
            !accepted(noisy,true),"fatal status plus noise is rejected");
}
int main(){alarm(30);ordinary();for(int i=0;i<14;++i)fork_case(i,i<2 || i>=10);cleanup_controls();
    std::puts("PASS remote routing ownership: ordinary + 14 isolated failure cases + cleanup controls");}
