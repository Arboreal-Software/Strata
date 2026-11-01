#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>

namespace strata::prefill {
// Counts supplied by Prefill's actual strict text/no-helper producer. These are
// allocation shapes, not caller policy, memory authority or execution receipts.
struct OwnedWorkspaceShape {
    uint64_t chunk=0,embedding=0,handoff_width=0,routes=0,experts=0,layers=0;
    uint64_t step_count=0,ple_heads=0,stager_slots=0,max_blob=0;
    uint64_t object_bytes=0,stager_job_bytes=0,atomic_bytes=0,upload_slots=0,upload_min_bytes=0;
    uint64_t arena_device_bytes=0;
    bool borrowed=false,successor=false;
};
struct OwnedWorkspaceBytes {uint64_t device=0,host=0,pinned=0;};
inline bool owned_workspace_quote(const OwnedWorkspaceShape& s,OwnedWorkspaceBytes& output)noexcept {
    if(!s.chunk||!s.embedding||!s.routes||!s.experts||!s.layers||!s.step_count||!s.ple_heads||
       !s.stager_slots||!s.max_blob||!s.upload_slots||!s.upload_min_bytes||!s.object_bytes||!s.stager_job_bytes||!s.atomic_bytes||
       (s.successor&&!s.handoff_width)||!s.arena_device_bytes)return false;
    uint64_t host=s.object_bytes;
    auto product=[](uint64_t a,uint64_t b,uint64_t& out){if(b&&a>UINT64_MAX/b)return false;out=a*b;return true;};
    auto add=[&](uint64_t count,uint64_t unit){uint64_t bytes=0;if(!product(count,unit,bytes)||bytes>UINT64_MAX-host)return false;host+=bytes;return true;};
    uint64_t tk=0,tn=0,steps=0,groups=0;
    if(!product(s.chunk,s.routes,tk)||!product(s.chunk,s.embedding,tn)||!product(s.chunk,s.step_count,steps)||
       s.experts>UINT64_MAX-s.experts/16-2)return false;
    groups=s.experts+s.experts/16+2;
    if(groups>UINT64_MAX/2||tk>(UINT64_MAX-2*groups)/3)return false;
    const uint64_t group_words=3*tk+2*groups;
    // Token/step pageable + pinned uploads; routing and count vectors, mapped
    // group tables and their pageable copies. Three generations cover bounded
    // vector replacement transients; all are capped at the admitted maximum T.
    if(!add(s.chunk,8)||!add(steps,8)||!add(tk,3*3*sizeof(int32_t))||
       !add(s.experts,3*2*sizeof(int32_t))||!add(1,3*sizeof(int32_t))||
       !add(group_words,3*2*sizeof(int32_t)))return false;
    // Two PLE upload banks and their index vectors; the four actual pinned-ring
    // slots bound every text upload by the larger PLE or grouping-table upload.
    uint64_t rows=0,upload=std::max(tn,group_words);
    if(!product(s.chunk,s.ple_heads,rows)||!add(tn,2*sizeof(float))||!add(rows,3*2*sizeof(int64_t))||
       !product(upload,sizeof(int32_t),upload)||!add(s.upload_slots,std::max(upload,s.upload_min_bytes)))return false;
    if(s.successor){uint64_t hand=0;if(!product(s.chunk,s.handoff_width,hand)||!add(hand,2*sizeof(float)))return false;}
    // Fixed expert-copy banks, per-layer jobs/ready-array transients and sign
    // directories. Source's persistent expert complement is separately owned.
    uint64_t jobs=0;
    if(!add(s.stager_slots,s.max_blob)||s.atomic_bytes>UINT64_MAX/2||s.stager_job_bytes>UINT64_MAX-2*s.atomic_bytes||
       !product(s.experts,3,jobs)||!add(jobs,s.stager_job_bytes+2*s.atomic_bytes)||
       !add(s.layers,2*sizeof(void*)))return false;
    // Alloc::take's smallest allocation is256 bytes. Its owned-pointer vector
    // therefore cannot exceed this bound, including replacement transients.
    if(!add(s.borrowed?1:s.arena_device_bytes/256+1,3*sizeof(void*)))return false;
    // init's token device buffer is outside carve/bytes_needed_owned even when
    // the main arena aliases already-paid expert-cache storage.
    if(s.chunk>(UINT64_MAX-(2ULL<<20)+1)/sizeof(int32_t))return false;
    const uint64_t token_bytes=s.chunk*sizeof(int32_t);
    const uint64_t token_pages=((token_bytes+(2ULL<<20)-1)/(2ULL<<20))*(2ULL<<20);
    if(!s.borrowed&&s.arena_device_bytes>UINT64_MAX-token_pages)return false;
    OwnedWorkspaceBytes quote;
    quote.device=(s.borrowed?0:s.arena_device_bytes)+token_pages;
    quote.host=host;quote.pinned=host; // conservative pinned SUBSET, not extra host
    if(quote.device>SIZE_MAX||quote.host>SIZE_MAX)return false;
    output=quote;return true;
}
} // namespace strata::prefill
