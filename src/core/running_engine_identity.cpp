#include "strata/core/running_engine_identity.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef STRATA_RUNNING_ENGINE_IDENTITY_TEST
// Compile-time test seam only; never exported by normal production compilation.
namespace strata::core::running_engine_test {
int open_self();int stat_fd(int,struct stat*);ssize_t read_at(int,void*,size_t,off_t);
int close_fd(int);int64_t process_id();
}
#endif
#endif
namespace strata::core {
namespace {
void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
#if defined(__linux__)
#ifdef STRATA_RUNNING_ENGINE_IDENTITY_TEST
namespace os=running_engine_test;
#else
namespace os {
int open_self(){return ::open("/proc/self/exe",O_RDONLY|O_CLOEXEC);}
int stat_fd(int fd,struct stat* s){return ::fstat(fd,s);}
ssize_t read_at(int fd,void* p,size_t n,off_t off){return ::pread(fd,p,n,off);}
int close_fd(int fd){return ::close(fd);}
int64_t process_id(){return ::getpid();}
}
#endif
void close_once(int fd) noexcept {
    if(fd>=0&&os::close_fd(fd)!=0){std::fputs("strata: running image descriptor close uncertain; worker restart required\n",stderr);std::_Exit(1);}
}
struct Fd {int value=-1;~Fd(){close_once(value);}};
bool unchanged(const struct stat& a,const struct stat& b){
    return a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_mode==b.st_mode&&a.st_size==b.st_size&&
        a.st_mtim.tv_sec==b.st_mtim.tv_sec&&a.st_mtim.tv_nsec==b.st_mtim.tv_nsec&&
        a.st_ctim.tv_sec==b.st_ctim.tv_sec&&a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;
}
uint64_t little(const uint8_t* p,size_t n){uint64_t v=0;for(size_t i=0;i<n;++i)v|=uint64_t(p[i])<<(8*i);return v;}
#endif
}
RunningEngineReceipt::~RunningEngineReceipt(){
#if defined(__linux__)
    close_once(fd_);
#endif
}
const RunningEngineObservation* RunningEngineReceipt::observation()const noexcept {
#if defined(__linux__)
    return fd_>=0&&pid_==os::process_id()?&observation_:nullptr;
#else
    return nullptr;
#endif
}
bool collect_running_engine(std::shared_ptr<const ExpectedRunningExecutable> expected,
    const RunningEngineLimits& l,WindowBudget budget,std::shared_ptr<const RunningEngineReceipt>& out,
    std::string& error,const RunningEngineOptions& options){
#if !defined(__linux__)
    (void)expected;(void)l;(void)budget;(void)out;(void)options;
    error="actual running-image observation requires Linux";return false;
#else
    try {
        need(expected&&expected->bytes()>=64&&expected->bytes()<=l.max_file_bytes&&
            l.max_file_bytes<=(2ull<<30)&&l.block_bytes>=4096&&l.block_bytes<=(1ull<<20)&&
            l.max_read_calls&&l.max_read_calls<=1000000&&l.max_elapsed_ms&&l.max_elapsed_ms<=300000,"running image bounds/claim invalid");
        const auto start=std::chrono::steady_clock::now();
        auto checkpoint=[&]{need(!options.cancelled||!options.cancelled(),"running image cancelled");
            need(std::chrono::steady_clock::now()-start<std::chrono::milliseconds(l.max_elapsed_ms),"running image deadline");};
        checkpoint();
        WindowResources charge;charge.device_bytes.resize(budget.snapshot().capacity.device_bytes.size());
        charge.host_bytes=4096+sizeof(RunningEngineReceipt);
        auto admitted=budget.reserve(charge,ReservationPurpose::foreground);
        need(bool(admitted),"running image receipt quota refused");
        auto lease=std::move(admitted.lease);
        std::shared_ptr<RunningEngineReceipt> candidate;
        {
        // Transient IO/hash quota is released only AFTER FD failure cleanup,
        // hash state and buffer destruction. Retained FD metadata has its own
        // lease, so later model loaders can use a single-slot IO budget.
        charge.host_bytes=l.block_bytes+65536;charge.io_slots=1;
        auto workspace=budget.reserve(charge,ReservationPurpose::foreground);
        need(bool(workspace),"running image workspace/IO quota refused");
        auto workspace_lease=std::move(workspace.lease);
        std::vector<uint8_t> block(size_t(l.block_bytes));WindowHasher hash;
        Fd fd;fd.value=os::open_self();need(fd.value>=0,"open actual running image failed");
        const auto pid=os::process_id();need(pid>0,"invalid process identity");
        struct stat before{};need(os::stat_fd(fd.value,&before)==0,"running image initial stat failed");
        need(S_ISREG(before.st_mode)&&before.st_size>=0&&uint64_t(before.st_size)==expected->bytes(),"running image type/size mismatch");
        uint64_t calls=0;
        auto read=[&](void* dst,size_t bytes,uint64_t offset){
            size_t at=0;while(at<bytes){checkpoint();need(calls++<l.max_read_calls,"running image read-call bound");
                const auto n=os::read_at(fd.value,static_cast<uint8_t*>(dst)+at,bytes-at,off_t(offset+at));
                if(n<0&&errno==EINTR)continue;
                need(n>0&&uint64_t(n)<=bytes-at,"running image short/error read");at+=size_t(n);
            }checkpoint();
        };
        std::array<uint8_t,64> elf{};
        for(uint64_t offset=0;offset<expected->bytes();){
            const auto n=size_t(std::min<uint64_t>(block.size(),expected->bytes()-offset));
            read(block.data(),n,offset);if(offset==0)std::copy_n(block.data(),elf.size(),elf.data());
            hash.update(block.data(),n);offset+=n;
        }
        const auto actual_digest=hash.digest();
        need(actual_digest==expected->digest(),"actual running image digest mismatch");
        need(std::memcmp(elf.data(),"\177ELF",4)==0&&elf[4]==2&&elf[5]==1&&elf[6]==1,"running image is not native ELF64 little endian");
        const auto type=little(&elf[16],2),machine=little(&elf[18],2);
#if defined(__x86_64__)
        need(machine==62,"running image ELF machine differs from host");
#elif defined(__aarch64__)
        need(machine==183,"running image ELF machine differs from host");
#else
        need(false,"running image native machine unsupported");
#endif
        const auto entry=little(&elf[24],8),phoff=little(&elf[32],8),phnum=little(&elf[56],2);
        need((type==2||type==3)&&little(&elf[20],4)==1&&little(&elf[52],2)==64&&
            little(&elf[54],2)==56&&entry&&phnum&&phnum<=1024&&phoff>=64&&
            phoff<=expected->bytes()&&phnum<=(expected->bytes()-phoff)/56,"invalid native ELF program geometry");
        bool executable_entry=false;
        for(uint64_t i=0;i<phnum;++i){
            std::array<uint8_t,56> ph{};read(ph.data(),ph.size(),phoff+i*56);
            if(little(ph.data(),4)!=1)continue; // PT_LOAD
            const auto offset=little(&ph[8],8),va=little(&ph[16],8),filesz=little(&ph[32],8),memsz=little(&ph[40],8);
            need(filesz<=memsz&&offset<=expected->bytes()&&filesz<=expected->bytes()-offset&&memsz<=UINT64_MAX-va,"invalid ELF load extent");
            if((little(&ph[4],4)&1)&&entry>=va&&entry-va<filesz)executable_entry=true;
        }
        need(executable_entry,"ELF entry not in executable file-backed load segment");
        struct stat after{};need(os::stat_fd(fd.value,&after)==0&&unchanged(before,after),"running image changed during observation");
        checkpoint();need(os::process_id()==pid,"process changed during image observation");
        candidate=std::shared_ptr<RunningEngineReceipt>(new RunningEngineReceipt);
        candidate->lease_=std::move(lease);candidate->expected_=std::move(expected);candidate->pid_=pid;
        candidate->observation_={uint64_t(after.st_size),uint64_t(after.st_dev),uint64_t(after.st_ino),actual_digest,uint16_t(machine)};
        candidate->fd_=fd.value;fd.value=-1;
        } // hash/buffer destruction precedes transient workspace/IO lease release
        out=std::move(candidate);error.clear();return true;
    }catch(const std::exception& e){try{error=e.what();}catch(...){error.clear();}return false;}
    catch(...){error.clear();return false;}
#endif
}
}
