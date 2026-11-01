#include "strata/core/window_state.hpp"
#include <array>
#include <memory>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <algorithm>
static bool fail_next=false;
void*operator new(size_t n){if(fail_next){fail_next=false;throw std::bad_alloc();}if(void*p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void*p)noexcept{std::free(p);}void operator delete(void*p,size_t)noexcept{std::free(p);}
static void check(bool v,const char*e){if(!v){std::fprintf(stderr,"FAIL %s\n",e);std::exit(9);}}
namespace strata::core{[[noreturn]]void retirement_failed(const char*e){std::fprintf(stderr,"EXPECTED_OR_UNEXPECTED_FATAL %s\n",e);std::exit(70);}}
namespace strata::program{
struct ServingRawSession{
 enum class Phase{running,drained};Phase state=Phase::running;bool opened=true,good=true;int admitted=0;bool budget=false,oom=false;
 core::WindowKey identity{};core::StateCut current{};std::weak_ptr<const void>last_credit;
 struct Admission{bool okay;const char*detail;explicit operator bool()const{return okay;}};
 bool healthy()const{return good;}bool request_open()const{return opened;}Phase phase()const{return state;}
 const core::WindowKey&key()const{return identity;}const core::StateCut&cut()const{return current;}
 Admission admit_host(uint64_t,std::shared_ptr<const void>&credit){++admitted;if(budget)return {false,"budget_refused"};credit=std::make_shared<int>(1);last_credit=credit;if(oom){oom=false;fail_next=true;}return {true,""};}
 bool drain_work(bool,const core::StateCut&cut,std::string&){current=cut;state=Phase::drained;return true;}
 bool close_request(std::string&){opened=false;return true;}bool open_request(std::string&){opened=true;return true;}bool begin_work(std::string&){state=Phase::running;return true;}
};}
