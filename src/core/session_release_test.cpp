#include "strata/core/session.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<const float*> released;
namespace strata::kernels {
void rope_table_release(const float* table) { released.push_back(table); }
}
static void check(bool ok,const char* why) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}
}
int main() {
    strata::core::QsaState states[4]{};
    float table[4]{};
    for(int i=0;i<4;++i){states[i].owns_rope=true;states[i].cos_tab=&table[i];}
    strata::core::SessionState stage;
    stage.qsa_states=states;stage.qsa_ord0=2;stage.qsa_alloc=2;
    states[3].owns_rope=false; // the second owned layer borrows the primary table
    strata::core::session_release(stage);
    check(released.size()==1 && released[0]==&table[2],"release uses global ordinal of split primary");
    check(states[0].owns_rope && states[1].owns_rope,"unowned earlier layers remain registered");
    check(!states[2].owns_rope,"released ownership cleared");
    strata::core::session_release(stage);
    check(released.size()==1,"release is idempotent");
    stage.qsa_ord0=0;stage.qsa_alloc=1;
    strata::core::session_release(stage);
    check(released.size()==2 && released[1]==&table[0],"single-stage ordinal zero stays unchanged");
    stage.qsa_states=nullptr;
    strata::core::session_release(stage);
    check(released.size()==2,"empty release does not access arrays");
    std::puts("session_release_test passed");
}
