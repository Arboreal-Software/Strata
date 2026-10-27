#include "strata/core/verified_artifact.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core;
namespace fs=std::filesystem;
static bool fail_close=false;
static int* close_count=nullptr;
extern "C" int __real_close(int);
extern "C" int __wrap_close(int fd){if(fail_close){++*close_count;const int r=__real_close(fd);(void)r;errno=EIO;return -1;}return __real_close(fd);}
struct Fixture {
    std::string dir,path,data=std::string(1200,'a');
    Fixture(){char p[]="/tmp/strata-artifact-XXXXXX";auto* made=mkdtemp(p);assert(made);dir=made;path=dir+"/source";write(data);}
    void write(const std::string& bytes){std::ofstream out(path,std::ios::binary|std::ios::trunc);out<<bytes;out.close();assert(out);}
    StateDigest digest() const{WindowHasher h;h.update(data.data(),data.size());return h.digest();}
    ~Fixture(){fs::remove_all(dir);}
};
static WindowBudget budget(uint64_t disk=1<<20,uint64_t host=1<<20,uint64_t io=1){return WindowBudget(WindowResources{{0},host,0,disk,io,0},WindowResources{{0},0,0,0,0,0});}
static VerifiedArtifactLimits limits(){return {1<<20,64,4096,4096,true};}
static VerifiedArtifactOptions copy(){VerifiedArtifactOptions o;o.fault=[](const char* step){return std::string(step)=="clone"?EOPNOTSUPP:0;};return o;}
static std::string read(const VerifiedArtifact& a){std::string out(size_t(a.size()),'\0');assert(pread(a.readonly_fd(),out.data(),out.size(),0)==ssize_t(out.size()));return out;}
int main(){
    {
        Fixture f;auto b=budget();std::shared_ptr<const VerifiedArtifact> a;
        const auto r=VerifiedArtifact::snapshot(f.path,f.dir,f.digest(),limits(),b,a,copy());if(!r)std::fprintf(stderr,"snapshot: %s errno=%d\n",r.detail.c_str(),r.system_error);assert(r&&a);
        assert(a->snapshot_method()==ArtifactSnapshotMethod::full_copy&&a->content_digest()==f.digest()&&read(*a)==f.data);
        assert(b.snapshot().reserved.io_slots==0&&b.snapshot().reserved.disk_bytes==f.data.size()+4096&&b.snapshot().reserved.host_bytes==4096);
        assert((fcntl(a->readonly_fd(),F_GETFL)&O_ACCMODE)==O_RDONLY);struct stat st{};assert(fstat(a->readonly_fd(),&st)==0&&st.st_nlink==0&&(st.st_mode&0777)==0400);
        errno=0;assert(pwrite(a->readonly_fd(),"x",1,0)==-1&&errno==EBADF);assert(ftruncate(a->readonly_fd(),0)==-1);
        assert(fs::file_size(f.path)==f.data.size());f.write(std::string(f.data.size(),'z'));assert(read(*a)==f.data);
        fs::rename(f.path,f.path+".old");f.write("new path");assert(read(*a)==f.data);
        int loader=open(a->loader_path().c_str(),O_RDONLY|O_CLOEXEC);assert(loader>=0);void* map=mmap(nullptr,a->size(),PROT_READ,MAP_PRIVATE,loader,0);assert(map!=MAP_FAILED);assert(close(loader)==0);
        auto profile=verified_profile_artifact(ProfileRole::dense,"dense",a);auto held=a;a.reset();assert(b.snapshot().reserved.disk_bytes>0&&static_cast<const char*>(map)[0]=='a');assert(munmap(map,held->size())==0);held.reset();assert(b.snapshot().reserved.disk_bytes>0);profile={};assert(b.snapshot().reserved.disk_bytes==0&&b.snapshot().reserved.host_bytes==0);
    }
    {
        Fixture f;std::shared_ptr<const VerifiedArtifact> a;auto b=budget();auto r=VerifiedArtifact::snapshot(f.path,f.dir,f.digest(),limits(),b,a);assert(r&&read(*a)==f.data);std::printf("actual snapshot method: %s\n",a->snapshot_method()==ArtifactSnapshotMethod::reflink?"FICLONE":"fallback copy (FICLONE unsupported)");
    }
    for(const char* step:{"source_open","source_stat","directory_open","temporary_open","copy_reset","copy_read","copy_write","snapshot_stat","snapshot_flush","snapshot_readonly_mode","readonly_open","readonly_stat","hash_read"}){
        Fixture f;auto b=budget();std::shared_ptr<const VerifiedArtifact> output;auto o=copy();o.fault=[&](const char* s){if(std::string(s)==step)return EIO;return std::string(s)=="clone"?EOPNOTSUPP:0;};
        auto r=VerifiedArtifact::snapshot(f.path,f.dir,f.digest(),limits(),b,output,o);assert(!r&&r.error==VerifiedArtifactError::io&&r.system_error==EIO&&!output);assert(b.snapshot().reserved.disk_bytes==0&&b.snapshot().reserved.host_bytes==0&&b.snapshot().reserved.io_slots==0);assert(std::distance(fs::directory_iterator(f.dir),fs::directory_iterator{})==1);
    }
    {
        Fixture f;auto b=budget();std::shared_ptr<const VerifiedArtifact> a;assert(VerifiedArtifact::snapshot(f.path,f.dir,f.digest(),limits(),b,a,copy()));auto keep=a;auto wrong=f.digest();wrong[0]^=1;auto r=VerifiedArtifact::snapshot(f.path,f.dir,wrong,limits(),b,a,copy());assert(r.error==VerifiedArtifactError::hash_mismatch&&a==keep);
        bool stop=false;auto o=copy();o.progress=[&](uint64_t,uint64_t){stop=true;};o.cancelled=[&]{return stop;};r=VerifiedArtifact::snapshot(f.path,f.dir,f.digest(),limits(),b,a,o);assert(r.error==VerifiedArtifactError::cancelled&&a==keep);
        auto no_copy=limits();no_copy.allow_full_copy=false;r=VerifiedArtifact::snapshot(f.path,f.dir,f.digest(),no_copy,b,a,copy());assert(r.error==VerifiedArtifactError::unsupported&&a==keep);
        auto bad_clone=copy();bad_clone.fault=[](const char* step){return std::string(step)=="clone"?EIO:0;};r=VerifiedArtifact::snapshot(f.path,f.dir,f.digest(),limits(),b,a,bad_clone);assert(r.error==VerifiedArtifactError::io&&r.system_error==EIO&&a==keep);
    }
    for(int kind=0;kind<3;++kind){Fixture f;auto b=budget(kind==0?1:1<<20,kind==1?1:1<<20,kind==2?0:1);std::shared_ptr<const VerifiedArtifact> out;int effects=0;auto o=copy();o.fault=[&](const char* step){if(std::string(step)=="temporary_open")++effects;return 0;};auto r=VerifiedArtifact::snapshot(f.path,f.dir,f.digest(),limits(),b,out,o);assert(r.error==VerifiedArtifactError::budget_refused&&!out&&effects==0&&b.snapshot().reserved.host_bytes==0);}
    {
        Fixture f;auto expected=f.digest();auto o=copy();bool changed=false;o.fault=[&](const char* step){if(std::string(step)=="hash_read"&&!changed){f.write(std::string(f.data.size(),'q'));changed=true;}return std::string(step)=="clone"?EOPNOTSUPP:0;};std::shared_ptr<const VerifiedArtifact> a;assert(VerifiedArtifact::snapshot(f.path,f.dir,expected,limits(),budget(),a,o)&&read(*a)==f.data);
    }
    {
        close_count=static_cast<int*>(mmap(nullptr,sizeof(int),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));assert(close_count!=MAP_FAILED);*close_count=0;Fixture f;pid_t pid=fork();assert(pid>=0);if(!pid){fail_close=true;std::shared_ptr<const VerifiedArtifact> a;VerifiedArtifact::snapshot(f.path,f.dir,f.digest(),limits(),budget(),a,copy());_Exit(42);}int status=0;assert(waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==1&&*close_count==1);munmap(close_count,sizeof(int));close_count=nullptr;
    }
    {
        // Real header-only GgufModel parser/mmap, explicit two-shard proc paths.
        Fixture first,second;auto tiny=[](uint32_t shard){std::string bytes;auto n=[&](uint64_t value,int width){for(int i=0;i<width;++i)bytes+=char(value>>(i*8));};auto text=[&](const std::string& value){n(value.size(),8);bytes+=value;};n(0x46554747,4);n(3,4);n(0,8);n(4,8);text("general.architecture");n(8,4);text("qwen4exp");for(const auto& [key,value]:std::vector<std::pair<std::string,uint32_t>>{{"split.count",2},{"split.no",shard},{"split.tensors.count",0}}){text(key);n(4,4);n(value,4);}return bytes;};
        first.data=tiny(0);second.data=tiny(1);first.write(first.data);second.write(second.data);auto b=budget();std::shared_ptr<const VerifiedArtifact> a,z;
        assert(VerifiedArtifact::snapshot(first.path,first.dir,first.digest(),limits(),b,a,copy()));assert(VerifiedArtifact::snapshot(second.path,second.dir,second.digest(),limits(),b,z,copy()));
        {strata::GgufModel model(std::vector<std::string>{a->loader_path(),z->loader_path()});assert(model.size()==2&&model.meta().get("general.architecture")->s=="qwen4exp");first.write("changed source");assert(model.shard(0).version()==3);}
        // Owners released only after actual loader mappings/FDs retired.
        a.reset();z.reset();assert(b.snapshot().reserved.disk_bytes==0);
    }
    std::puts("verified_artifact_test passed");
}
