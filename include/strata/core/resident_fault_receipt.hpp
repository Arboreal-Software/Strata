#pragma once
// Test-only bounded output receipt. No model/source IO, completion authority or
// global fault arming. Compile OFF unless explicit qualification build + LinuxCUDA.
#if defined(STRATA_RESIDENT_FAULT_QUALIFICATION) && defined(__linux__) && !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_USE_SYCL) && !defined(STRATA_GPU_SYCL)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstddef>
#include <cstring>
namespace strata::core::resident_fault_receipt {
inline bool valid_fd(int fd)noexcept {
    struct stat s{};
    return fd>=0&&fstat(fd,&s)==0&&S_ISREG(s.st_mode)&&s.st_uid==geteuid()&&s.st_nlink==1&&(s.st_mode&0777)==0600&&s.st_size>=0&&s.st_size<=16384;
}
inline bool write_record(int fd,const char* bytes,std::size_t size)noexcept {
    if(!bytes||!size||size>1024||!valid_fd(fd))return false;
    struct stat s{};if(fstat(fd,&s)!=0||size>16384-std::size_t(s.st_size))return false;
    std::size_t offset=0;unsigned interruptions=0;
    while(offset<size){const auto n=write(fd,bytes+offset,size-offset);
        if(n<0&&errno==EINTR){if(++interruptions>16)return false;continue;}if(n<=0)return false;offset+=std::size_t(n);}
    return fsync(fd)==0;
}
class File final {
public:
    File()=default;File(const File&)=delete;File& operator=(const File&)=delete;
    ~File(){if(fd_>=0)(void)close(fd_);}
    bool open_new(const char* path)noexcept {
        if(fd_>=0||!path||std::strcmp(path,"/output/RESIDENT-FAULT.jsonl")!=0)return false;
        const int f=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
        if(f<0)return false;
        struct stat s{};
        if(fchmod(f,0600)!=0||!valid_fd(f)||fstat(f,&s)!=0||s.st_size!=0||fsync(f)!=0){(void)close(f);return false;}
        const int directory=open("/output",O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
        if(directory<0){(void)close(f);return false;}
        const bool durable=fsync(directory)==0;(void)close(directory);
        if(!durable){(void)close(f);return false;}
        fd_=f;return true;
    }
    int get()const noexcept{return fd_;}
private:int fd_=-1;
};
} // namespace strata::core::resident_fault_receipt
#endif
