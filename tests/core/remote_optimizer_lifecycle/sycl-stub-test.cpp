#include "strata/core/remote_expert_opt.hpp"
#include <cassert>
namespace strata::core {
// Exact shadow setter only: not a substitute for compiling DPCT/cache headers.
class RemoteExperts { public:
#include "helper-setter.inc"
};
}
int main(){
 using namespace strata::core;std::string e;RemoteExperts helper;RemoteExpertOpt opt;
 assert(!helper.set_strict_retirement(true,{0,1},e));assert(!e.empty());
 assert(helper.set_strict_retirement(false,{},e));
 assert(!opt.set_strict_retirement(true,{0,1},e));assert(!e.empty());
 assert(opt.set_strict_retirement(false,{},e));assert(!opt.attach_checked(helper,e));
 assert(!opt.close_after_callers_joined());assert(!opt.init(e));
}
