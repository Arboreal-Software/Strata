#include "strata/core/window_repair_ancestry.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
using namespace strata::core;
static void check(bool value,const char* why){if(!value){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
static RepairAncestryLimits limits(uint32_t count=64){return {count,RepairAncestry::retained_bytes()};}
static RepairCausalRoot root(uint8_t id,uint64_t unix_birth=600,uint64_t mono_birth=100){
    RepairCausalRoot r;r.identity[0]=id;r.original_history[0]=42;
    r.engine_generation=1;r.window_generation=2;r.runtime_incarnation=3;r.original_cut=4;
    r.original_begin=4;r.original_count=4;r.first_unresolved_unix_ms=unix_birth;r.monotonic_birth={7,mono_birth};return r;
}
static void trimming_keeps_endpoint_roots(){
    const std::array records{root(1)};RepairAncestry ancestry;
    check(make_repair_ancestry(records,limits(),ancestry)==RepairAncestryStatus::ok,"initial roots");
    RepairSpatialCoverage initial,trimmed,empty;
    check(make_repair_spatial_coverage(ancestry,4,20,initial)==RepairAncestryStatus::ok,"initial spatial coverage");
    check(advance_repair_spatial_begin(initial,8,trimmed)==RepairAncestryStatus::ok,"trim ALL original root interval");
    check(trimmed.ancestry().roots().size()==1&&trimmed.ancestry().roots()[0]==records[0],"old endpoint root survives its original interval being gone");
    check(advance_repair_spatial_begin(trimmed,20,empty)==RepairAncestryStatus::ok,"empty spatial range");
    check(empty.begin()==empty.end()&&empty.ancestry().roots().size()==1&&empty.ancestry().roots()[0]==records[0],"empty range is NOT exact replacement/root deletion");
    const auto age=repair_ancestry_age(empty.ancestry(),{7,1000});
    check(age.oldest_unresolved_unix_ms==600&&age.oldest_monotonic_age_ns==900,"trim does not refresh age");
    RepairSpatialCoverage refused;
    check(advance_repair_spatial_begin(trimmed,7,refused)==RepairAncestryStatus::invalid_coverage&&!refused.valid(),"no backwards trim");
    check(advance_repair_spatial_begin(trimmed,21,refused)==RepairAncestryStatus::invalid_coverage&&!refused.valid(),"no beyond-end trim");
}
static void deterministic_union_and_conflicts(){
    const auto older=root(3,600,100),younger=root(1,900,500);
    const std::array left{older,younger,older};const std::array right{younger};
    RepairAncestry a,b,ab,ba;
    check(make_repair_ancestry(left,limits(),a)==RepairAncestryStatus::ok&&a.roots().size()==2,"exact duplicates collapse");
    check(make_repair_ancestry(right,limits(),b)==RepairAncestryStatus::ok,"second input");
    check(union_repair_ancestry(a,b,limits(),ab)==RepairAncestryStatus::ok&&union_repair_ancestry(b,a,limits(),ba)==RepairAncestryStatus::ok,"union either order");
    check(ab.roots().size()==2&&ba.roots().size()==2&&ab.roots()[0]==younger&&ab.roots()[1]==older&&ba.roots()[0]==ab.roots()[0]&&ba.roots()[1]==ab.roots()[1],"canonical deterministic identity order");
    const auto age=repair_ancestry_age(ab,{7,1000});
    check(age.oldest_unresolved_unix_ms==600&&age.oldest_monotonic_age_ns==900,"multiroot oldest survives union");
    // Sample distinct original-field changes cannot renew an existing identity.
    for(unsigned which=0;which<4;++which){
        auto conflict=older;
        if(which==0)++conflict.first_unresolved_unix_ms;
        if(which==1)++conflict.monotonic_birth.ticks_ns;
        if(which==2)++conflict.original_begin;
        if(which==3)++conflict.original_history[0];
        const std::array altered{conflict};RepairAncestry c,out;
        check(make_repair_ancestry(altered,limits(),c)==RepairAncestryStatus::ok,"conflicting record valid in isolation");
        check(union_repair_ancestry(a,c,limits(),out)==RepairAncestryStatus::identity_conflict&&out.empty(),"conflict rejected without partial union");
        check(a.roots()[1]==older,"original birth/identity stays unchanged");
    }
    check(union_repair_ancestry(a,b,limits(),ab)==RepairAncestryStatus::output_not_empty&&ab.roots().size()==2,"cannot overwrite existing root table");
}
static void hard_bounds_and_overflow(){
    const std::array one{root(1)};RepairAncestry bytes;
    check(make_repair_ancestry(one,{1,RepairAncestry::retained_bytes()-1},bytes)==RepairAncestryStatus::byte_bound&&bytes.empty(),"actual fixed retained byte bound");
    std::array<RepairCausalRoot,64> full;
    for(size_t i=0;i<full.size();++i)full[i]=root(uint8_t(i+1));
    RepairAncestry maximum;
    check(make_repair_ancestry(full,limits(),maximum)==RepairAncestryStatus::ok&&maximum.roots().size()==64,"hard capacity inclusive");
    const std::array extra{root(65)};RepairAncestry b,overflow;
    check(make_repair_ancestry(extra,limits(),b)==RepairAncestryStatus::ok,"extra source");
    check(union_repair_ancestry(maximum,b,limits(),overflow)==RepairAncestryStatus::count_bound&&overflow.empty(),"union refuses65 unique roots");
    std::array<RepairCausalRoot,65> duplicate_flood;
    for(auto& r:duplicate_flood)r=one[0];
    check(make_repair_ancestry(duplicate_flood,limits(),overflow)==RepairAncestryStatus::count_bound&&overflow.empty(),"input scan itself bounded even if duplicates");
    auto bad=root(4);bad.original_begin=std::numeric_limits<uint64_t>::max()-1;bad.original_count=2;
    const std::array wrapped{bad};
    check(make_repair_ancestry(wrapped,limits(),overflow)==RepairAncestryStatus::overflow&&overflow.empty(),"original interval addition cannot wrap");
    const std::array two{root(1),root(2)};
    check(make_repair_ancestry(two,limits(1),overflow)==RepairAncestryStatus::count_bound&&overflow.empty(),"caller count bound lower than physical capacity");
    check(make_repair_ancestry(one,limits(65),overflow)==RepairAncestryStatus::invalid_limits&&overflow.empty(),"cannot widen hard capacity");
}
static void unknown_and_epoch_age(){
    auto unknown=root(1);unknown.monotonic_birth={0,0};
    const std::array unanchored{unknown};RepairAncestry a;
    check(make_repair_ancestry(unanchored,limits(),a)==RepairAncestryStatus::ok,"unknown anchor retained, not invented");
    auto age=repair_ancestry_age(a,{7,1000});
    check(!age.oldest_monotonic_age_ns&&age.unknown_monotonic_anchors==1&&age.oldest_unresolved_unix_ms==600,"unknown is not young age0");
    auto epoch=root(2);epoch.monotonic_birth={7,0};
    const std::array at_epoch{epoch};RepairAncestry b,mixed;
    check(make_repair_ancestry(at_epoch,limits(),b)==RepairAncestryStatus::ok,"nonzero domain with ticks0 is valid epoch");
    check(repair_ancestry_age(b,{7,1000}).oldest_monotonic_age_ns==1000,"epoch age known");
    check(repair_ancestry_age(b,{7,0}).oldest_monotonic_age_ns==0,"actual known age0 distinct from unknown");
    check(!repair_ancestry_age(b,{8,1000}).oldest_monotonic_age_ns,"foreign clock domain not comparable");
    check(union_repair_ancestry(a,b,limits(),mixed)==RepairAncestryStatus::ok&&!repair_ancestry_age(mixed,{7,1000}).oldest_monotonic_age_ns,"one unknown root blocks full age claim");
    auto future=root(3,500,2000);const std::array later{future};RepairAncestry c;
    check(make_repair_ancestry(later,limits(),c)==RepairAncestryStatus::ok&&!repair_ancestry_age(c,{7,1000}).oldest_monotonic_age_ns,"future monotonic birth is unknown, no subtraction wrap");
    unknown.monotonic_birth={0,1};const std::array malformed{unknown};RepairAncestry rejected;
    check(make_repair_ancestry(malformed,limits(),rejected)==RepairAncestryStatus::invalid_root&&rejected.empty(),"domain0 requires ticks0");
}
int main(){trimming_keeps_endpoint_roots();deterministic_union_and_conflicts();hard_bounds_and_overflow();unknown_and_epoch_age();std::puts("PASS repair ancestry DATA helper (not executor certification)");}
