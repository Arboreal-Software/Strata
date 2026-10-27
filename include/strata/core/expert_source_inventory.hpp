#pragma once
#include "strata/core/expert_load_inventory.hpp"
#include "strata/core/expert_source_observer.hpp"
#include "strata/core/verified_input_manifest.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <limits>
#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace strata::core {
struct ExpertSourceInventoryLimits {uint64_t max_sources=0,max_layers=0,max_string_bytes=0,max_workers=0;};
struct ExpertOpenedSource {
    std::string path;
    uint64_t expected_bytes=0,device=0,inode=0;
    int retained_fd=-1;
    bool selected=false;
};
struct ExpertSourceSpan {
    uint64_t layer=0,role=0,source=0,source_offset=0,source_stride=0,bytes_per_expert=0,experts=0;
    // mapped_file_ordinal is the actual File maps_ index, absent for Arena worker streams.
    uint64_t destination_offset=0,destination_stride=0,mapped_file_ordinal=UINT64_MAX;
    ExpertSourceTransfer transfer=ExpertSourceTransfer::mapped_packed;
    bool complete=false;
};
struct ExpertSourceInventory {
    std::shared_ptr<const ReservationLease> metadata_reservation;
    std::shared_ptr<const VerifiedDriverBindings> retained_input_bindings;
    std::shared_ptr<const ExpertLayoutCapture> retained_layout;
    ExpertSourceInventoryLimits limits;
    std::vector<ExpertOpenedSource> sources; // binding ordinals, actual selected spans below
    std::vector<ExpertSourceSpan> spans; // fixed layer*3 slots; unselected slots remain incomplete
    std::string pack_path,primary_gguf;
    uint64_t layout_generation=0,hadamard_generation=0,layers=0,experts=0;
    bool packed_present=false,from_gguf=false,arena=false,padded=false;
};
inline bool expert_source_inventory_host_bound(const ExpertSourceInventoryLimits& l,uint64_t& out,std::string& error) {
    if(!l.max_sources||l.max_sources>1024||!l.max_layers||l.max_layers>65536||
       !l.max_string_bytes||l.max_string_bytes>4096||!l.max_workers||l.max_workers>256) {
        error="expert source inventory bounds outside limits";return false;
    }
    // Includes preallocated span/source metadata and bounded factory string temporaries.
    const uint64_t n=8192+sizeof(ExpertSourceInventory)+
        l.max_sources*(sizeof(ExpertOpenedSource)+4*l.max_string_bytes+256)+
        l.max_layers*3*(sizeof(ExpertSourceSpan)+128)+l.max_workers*(l.max_string_bytes+512);
    out=n;return true; // administrative bounds make all products above fit uint64
}
class ExpertSourceCapture : public ExpertSourceReadObserver {
public:
    // Factory is inline and only instantiated by opt-in callers. Default engine
    // source paths introduce no out-of-line verified-input dependency.
    static bool create(const ExpertSourceInventoryLimits& limits,std::shared_ptr<const ReservationLease> lease,
                       std::shared_ptr<const VerifiedDriverBindings> bindings,
                       std::shared_ptr<const ExpertLayoutCapture> layout,
                       std::shared_ptr<ExpertSourceCapture>& out,std::string& error) {
#if !defined(__linux__)
        (void)limits;(void)lease;(void)bindings;(void)layout;(void)out;
        error="verified expert source capture requires Linux immutable inputs";return false;
#else
        uint64_t required=0;
        if(!expert_source_inventory_host_bound(limits,required,error))return false;
        const auto* parsed=layout?layout->completed_inventory():nullptr;
        if(out||!lease||lease->purpose()!=ReservationPurpose::foreground||lease->resources().host_bytes<required||
           !bindings||!parsed||parsed->retained_input_bindings!=bindings||parsed->requested_layers<=0||
           uint64_t(parsed->requested_layers)>limits.max_layers||parsed->effective_experts<=0) {
            error="expert source capture requires separate admitted envelope and matching completed verified layout";return false;
        }
        if(bindings->native.size()>limits.max_sources-uint64_t(bool(bindings->pack_payloads[2]))) {
            error="expert source binding count exceeds admission";return false;
        }
        try {
            auto p=std::shared_ptr<ExpertSourceCapture>(new ExpertSourceCapture);
            auto& r=p->receipt_;r.metadata_reservation=std::move(lease);r.retained_input_bindings=std::move(bindings);
            r.retained_layout=std::move(layout);r.limits=limits;r.layers=uint64_t(parsed->requested_layers);r.experts=uint64_t(parsed->effective_experts);
            r.layout_generation=parsed->layout_generation;r.hadamard_generation=parsed->hadamard_generation;
            r.sources.reserve(size_t(limits.max_sources));r.spans.resize(size_t(r.layers*3));
            r.pack_path=r.retained_input_bindings->pack_path();r.primary_gguf=r.retained_input_bindings->native_primary.loader_path();
            if(r.pack_path!=parsed->claims.expected_pack_path||r.pack_path.size()>limits.max_string_bytes||r.primary_gguf.size()>limits.max_string_bytes){error="expert source binding path mismatch/bound";return false;}
            auto add=[&](const VerifiedInputLease& input)->bool {
                if(!input)return true;
                const auto path=input.loader_path();
                for(const auto& old:r.sources)if(old.path==path)return old.retained_fd==input.readonly_fd();
                if(r.sources.size()>=limits.max_sources||path.size()>limits.max_string_bytes)return false;
                struct stat st{};const int fd=input.readonly_fd();
                if(fd<0||fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0||uint64_t(st.st_size)!=input.size())return false;
                r.sources.push_back({path,uint64_t(st.st_size),uint64_t(st.st_dev),uint64_t(st.st_ino),fd,false});return true;
            };
            r.packed_present=bool(r.retained_input_bindings->pack_payloads[2]);
            if(!add(r.retained_input_bindings->pack_payloads[2])){error="expert packed binding invalid or exceeds bounds";return false;}
            for(const auto& file:r.retained_input_bindings->native)if(!add(file)){error="expert native binding invalid or exceeds bounds";return false;}
            if(r.sources.empty()){error="expert source capture has no declared sources";return false;}
            out=std::move(p);return true;
        }catch(const std::bad_alloc&){error="expert source capture metadata allocation failed";return false;}
#endif
    }
private:
    friend class FileExpertSource;
    friend class ArenaExpertSource;
    bool claim_source() noexcept {bool expected=false;return attached_.compare_exchange_strong(expected,true);}
    bool begin(const std::string& pack,const std::string& primary,int64_t layers,int64_t experts,bool arena,std::string& error) {
        if(attempted_||!layout_valid()||pack!=receipt_.pack_path||
           (!receipt_.packed_present&&primary!=receipt_.primary_gguf)||layers<=0||experts<=0||
           uint64_t(layers)!=receipt_.layers||uint64_t(experts)!=receipt_.experts) {
            error="expert source capture requires fresh admission and unchanged verified layout/bindings";return false;
        }
        attempted_=true;receipt_.arena=arena;receipt_.from_gguf=!receipt_.packed_present;return true;
    }
public:
    bool allows(const std::string& path) const noexcept {for(const auto& f:receipt_.sources)if(f.path==path)return true;return false;}
    bool validate_fd(const std::string& path,int fd) const noexcept {
#if defined(__linux__)
        for(const auto& f:receipt_.sources)if(f.path==path) {
            struct stat a{},b{};
            return fd>=0&&f.retained_fd>=0&&!fstat(fd,&a)&&!fstat(f.retained_fd,&b)&&S_ISREG(a.st_mode)&&
                a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_size==b.st_size&&a.st_size>=0&&
                uint64_t(a.st_dev)==f.device&&uint64_t(a.st_ino)==f.inode&&uint64_t(a.st_size)==f.expected_bytes;
        }
#else
        (void)path;(void)fd;
#endif
        return false;
    }
    bool range_allowed(const std::string& path,uint64_t off,uint64_t per,uint64_t count) const noexcept {
        if(!per||!count||count>UINT64_MAX/per)return false;
        for(const auto& f:receipt_.sources)if(f.path==path)return off<=f.expected_bytes&&per*count<=f.expected_bytes-off;
        return false;
    }
    uint64_t max_sources() const noexcept{return receipt_.limits.max_sources;}
    bool record(uint64_t layer,uint64_t role,const std::string& path,uint64_t off,uint64_t per,
                uint64_t source_stride,uint64_t dst,uint64_t dst_stride,ExpertSourceTransfer transfer,uint64_t mapped_file_ordinal=UINT64_MAX) noexcept {
        if(!attempted_||retired_||layer>=receipt_.layers||role>=3||!per||source_stride<per||dst_stride<per)return false;
        size_t source=0;for(;source<receipt_.sources.size();++source)if(receipt_.sources[source].path==path)break;
        if(source==receipt_.sources.size())return false;
        const uint64_t count=receipt_.experts,extent=receipt_.sources[source].expected_bytes;
        if(count-1>(UINT64_MAX-per)/source_stride)return false;
        const uint64_t bytes=(count-1)*source_stride+per;
        if(off>extent||bytes>extent-off||count-1>(UINT64_MAX-per)/dst_stride||dst>UINT64_MAX-((count-1)*dst_stride+per))return false;
        std::lock_guard lock(mutex_);
        auto& span=receipt_.spans[size_t(layer*3+role)];if(span.complete)return false;
        span={layer,role,source,off,source_stride,per,count,dst,dst_stride,mapped_file_ordinal,transfer,true};receipt_.sources[source].selected=true;return true;
    }
private:
    bool publish(std::string& error) {
        if(!attempted_||retired_||!layout_valid()){error="expert source layout changed before publication";return false;}
        for(uint64_t l=0;l<receipt_.layers;++l)for(uint64_t r=0;r<(receipt_.from_gguf?3u:1u);++r)
            if(!receipt_.spans[size_t(l*3+r)].complete){error="expert source inventory has an incomplete actual read/map";return false;}
        complete_=true;return true;
    }
    void invalidate() noexcept {complete_=false;if(attempted_)retired_=true;}
public:
    const ExpertSourceInventory* completed_inventory() const noexcept {return complete_&&!retired_&&layout_valid()?&receipt_:nullptr;}
    uint64_t experts() const noexcept{return receipt_.experts;}
    uint64_t max_workers() const noexcept{return receipt_.limits.max_workers;}
    bool packed_present() const noexcept{return receipt_.packed_present;}
    void mark_padded(bool value) noexcept{receipt_.padded=value;}
private:
    bool layout_valid() const noexcept {
        const auto* p=receipt_.retained_layout?receipt_.retained_layout->completed_inventory():nullptr;
        return p&&p->layout_generation==receipt_.layout_generation&&p->hadamard_generation==receipt_.hadamard_generation;
    }
    ExpertSourceInventory receipt_;
    std::mutex mutex_;
    std::atomic<bool> attached_{false};
    bool attempted_=false,complete_=false,retired_=false;
};
} // namespace strata::core
