#include "strata/core/verified_artifact.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef __linux__
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace strata::core {
namespace {
#ifdef __linux__
struct Failure {VerifiedArtifactError code;int number;const char* text;};
[[noreturn]] void fail(VerifiedArtifactError c,const char* t,int e=0){throw Failure{c,e,t};}
void require(bool b,const char* t){if(!b)fail(VerifiedArtifactError::invalid,t);}
void close_checked(int& fd) noexcept {
    if(fd<0)return;
    const int owned=std::exchange(fd,-1);
    if(::close(owned)!=0){const int e=errno;std::fprintf(stderr,"strata: artifact close uncertain errno=%d; worker restart required\n",e);std::fflush(stderr);std::_Exit(1);}
}
struct Fd {int fd=-1;~Fd(){close_checked(fd);}};
int injection(const VerifiedArtifactOptions& o,const char* step){return o.fault?o.fault(step):0;}
void checkpoint(const VerifiedArtifactOptions& o,const char* step){if(const int e=injection(o,step))fail(VerifiedArtifactError::io,step,e);}
void cancel(const VerifiedArtifactOptions& o){if(o.cancelled&&o.cancelled())fail(VerifiedArtifactError::cancelled,"artifact verification cancelled");}
uint64_t add(uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,"artifact reservation overflow");return a+b;}
std::shared_ptr<const ReservationLease> reserve(WindowBudget& budget,uint64_t host,uint64_t disk,uint64_t io){
    WindowResources resources;resources.device_bytes.resize(budget.snapshot().capacity.device_bytes.size());resources.host_bytes=host;resources.disk_bytes=disk;resources.io_slots=io;
    auto r=budget.reserve(resources,ReservationPurpose::foreground);if(!r)fail(VerifiedArtifactError::budget_refused,"artifact host/disk/IO admission refused");return std::move(r.lease);
}
struct stat stat_checked(int fd,const VerifiedArtifactOptions& o,const char* step){checkpoint(o,step);struct stat st{};if(::fstat(fd,&st)!=0)fail(VerifiedArtifactError::io,step,errno);return st;}
void regular(const struct stat& s,uint64_t cap){require(S_ISREG(s.st_mode)&&s.st_size>0&&uint64_t(s.st_size)<=cap,"artifact is not a bounded nonempty regular file");}
void evidence_u64(WindowHasher& h,uint64_t n){uint8_t b[8];for(int i=0;i<8;++i)b[i]=uint8_t(n>>(8*i));h.update(b,8);}
#endif
}
VerifiedArtifact::~VerifiedArtifact(){
#ifdef __linux__
    close_checked(fd_); // lease releases only after checked descriptor cleanup
#endif
}
std::string VerifiedArtifact::loader_path() const {
#ifdef __linux__
    return fd_<0?std::string():"/proc/self/fd/"+std::to_string(fd_);
#else
    return {};
#endif
}
VerifiedArtifactResult VerifiedArtifact::snapshot(const std::string& source,const std::string& directory,
    const StateDigest& expected,const VerifiedArtifactLimits& limits,WindowBudget budget,
    std::shared_ptr<const VerifiedArtifact>& out,const VerifiedArtifactOptions& options){
#ifndef __linux__
    (void)source;(void)directory;(void)expected;(void)limits;(void)budget;(void)out;(void)options;
    return {VerifiedArtifactError::unsupported,0,"private verified artifact snapshot requires Linux"};
#else
    try {
        require(limits.max_file_bytes&&limits.max_file_bytes<=uint64_t(std::numeric_limits<off_t>::max())&&limits.block_bytes&&limits.block_bytes<=SIZE_MAX&&limits.metadata_host_bytes>=4096,"invalid artifact limits");
        require(!source.empty()&&!directory.empty()&&source.size()<=4096&&directory.size()<=4096&&source.find('\0')==std::string::npos&&directory.find('\0')==std::string::npos,"invalid artifact paths");
        require(std::any_of(expected.begin(),expected.end(),[](uint8_t b){return b!=0;}),"missing expected full-content SHA256");
        auto io=reserve(budget,add(limits.block_bytes,limits.metadata_host_bytes),0,1);
        auto candidate=std::shared_ptr<VerifiedArtifact>(new VerifiedArtifact);
        std::vector<uint8_t> buffer(size_t(limits.block_bytes));
        cancel(options);checkpoint(options,"source_open");Fd src;
        src.fd=::open(source.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);
        if(src.fd<0)fail(VerifiedArtifactError::io,"source open",errno);
        const auto before=stat_checked(src.fd,options,"source_stat");regular(before,limits.max_file_bytes);const auto bytes=uint64_t(before.st_size);
        candidate->retained_=reserve(budget,limits.metadata_host_bytes,add(bytes,limits.snapshot_disk_overhead_bytes),0);
        cancel(options);checkpoint(options,"directory_open");Fd dir;
        dir.fd=::open(directory.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
        if(dir.fd<0)fail(VerifiedArtifactError::io,"private directory open",errno);
        checkpoint(options,"temporary_open");Fd writable;
        writable.fd=::openat(dir.fd,".",O_TMPFILE|O_EXCL|O_RDWR|O_CLOEXEC,0600);
        if(writable.fd<0){const int e=errno;fail(e==EOPNOTSUPP||e==EINVAL||e==EISDIR?VerifiedArtifactError::unsupported:VerifiedArtifactError::io,"private O_TMPFILE creation",e);}
        cancel(options);const int injected=injection(options,"clone");int rc=-1,clone_error=injected;
        if(!injected){rc=::ioctl(writable.fd,FICLONE,src.fd);clone_error=errno;}
        if(rc==0)candidate->method_=ArtifactSnapshotMethod::reflink;
        else {
            if(clone_error!=EXDEV&&clone_error!=EOPNOTSUPP&&clone_error!=ENOTTY&&clone_error!=EINVAL)fail(VerifiedArtifactError::io,"artifact clone",clone_error);
            if(!limits.allow_full_copy)fail(VerifiedArtifactError::unsupported,"clone unavailable and full copy not admitted",clone_error);
            checkpoint(options,"copy_reset");if(::ftruncate(writable.fd,0)!=0)fail(VerifiedArtifactError::io,"copy reset",errno);
            uint64_t at=0;while(at<bytes){cancel(options);const size_t n=size_t(std::min<uint64_t>(bytes-at,buffer.size()));checkpoint(options,"copy_read");ssize_t got;
                do{got=::pread(src.fd,buffer.data(),n,off_t(at));}while(got<0&&errno==EINTR);
                if(got<0)fail(VerifiedArtifactError::io,"source copy read",errno);
                if(!got)fail(VerifiedArtifactError::invalid,"source truncated during copy");
                size_t done=0;while(done<size_t(got)){cancel(options);checkpoint(options,"copy_write");ssize_t written;
                    do{written=::pwrite(writable.fd,buffer.data()+done,size_t(got)-done,off_t(at+done));}while(written<0&&errno==EINTR);
                    if(written<=0)fail(VerifiedArtifactError::io,"snapshot copy write",written<0?errno:EIO);
                    done+=size_t(written);}
                at+=uint64_t(got);
            }
        }
        cancel(options);const auto created=stat_checked(writable.fd,options,"snapshot_stat");regular(created,limits.max_file_bytes);require(created.st_size==before.st_size&&created.st_nlink==0,"snapshot extent/link mismatch");
        checkpoint(options,"snapshot_flush");if(::fsync(writable.fd)!=0)fail(VerifiedArtifactError::io,"snapshot flush",errno);
        checkpoint(options,"snapshot_readonly_mode");if(::fchmod(writable.fd,0400)!=0)fail(VerifiedArtifactError::io,"snapshot mode",errno);
        const auto path="/proc/self/fd/"+std::to_string(writable.fd);checkpoint(options,"readonly_open");
        candidate->fd_=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
        if(candidate->fd_<0)fail(VerifiedArtifactError::io,"readonly snapshot reopen",errno);
        const auto readonly=stat_checked(candidate->fd_,options,"readonly_stat");require(readonly.st_dev==created.st_dev&&readonly.st_ino==created.st_ino&&readonly.st_size==created.st_size&&readonly.st_nlink==0&&(readonly.st_mode&0777)==0400,"readonly snapshot identity mismatch");
        close_checked(writable.fd);close_checked(src.fd);close_checked(dir.fd);
        WindowHasher hash;uint64_t at=0;while(at<bytes){cancel(options);checkpoint(options,"hash_read");const size_t n=size_t(std::min<uint64_t>(bytes-at,buffer.size()));ssize_t got;
            do{got=::pread(candidate->fd_,buffer.data(),n,off_t(at));}while(got<0&&errno==EINTR);
            if(got<0)fail(VerifiedArtifactError::io,"snapshot hash read",errno);
                if(!got)fail(VerifiedArtifactError::invalid,"snapshot truncated during verification");
            hash.update(buffer.data(),size_t(got));at+=uint64_t(got);if(options.progress)options.progress(at,bytes);}
        candidate->content_=hash.digest();if(candidate->content_!=expected)fail(VerifiedArtifactError::hash_mismatch,"snapshot full-content digest mismatch");
        candidate->bytes_=bytes;WindowHasher proof;constexpr char domain[]="strata-verified-private-artifact-v1";proof.update(domain,sizeof(domain));proof.update(candidate->content_.data(),candidate->content_.size());evidence_u64(proof,bytes);evidence_u64(proof,uint64_t(candidate->method_));candidate->verification_=proof.digest();
        cancel(options);std::vector<uint8_t>().swap(buffer);io.reset();out=std::move(candidate);return {};
    }catch(const Failure& f){return {f.code,f.number,f.text};}
    catch(const std::bad_alloc&){return {VerifiedArtifactError::budget_refused,0,"artifact metadata allocation failed"};}
    catch(const std::exception& e){return {VerifiedArtifactError::invalid,0,e.what()};}
    catch(...){return {VerifiedArtifactError::invalid,0,"artifact callback failed"};}
#endif
}
ProfileArtifact verified_profile_artifact(ProfileRole role,std::string name,std::shared_ptr<const VerifiedArtifact> owner){
    if(!owner)throw std::invalid_argument("missing verified artifact owner");
    return {role,std::move(name),owner->size(),owner->content_digest(),owner->verification_digest(),true,std::move(owner)};
}
} // namespace strata::core
