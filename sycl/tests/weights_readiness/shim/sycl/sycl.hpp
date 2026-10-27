#pragma once
// CPU fault-injection seam only. This is not a SYCL runtime or qualification.
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace sycl {
class exception : public std::runtime_error { public: using std::runtime_error::runtime_error; };
inline bool fail_copy=false,fail_drain=false,fail_free=false;
inline int free_calls=0;
inline int allocations=0;
struct event { void wait() const {} };
struct queue {
    event memcpy(void* dst,const void* src,size_t n) {
        if(fail_copy)throw exception("injected copy failure");
        std::memcpy(dst,src,n);return {};
    }
};
inline void* malloc_host(size_t n,queue&) {
    void* p=std::malloc(n);if(!p)throw exception("host allocation failed");++allocations;return p;
}
inline void free(void* p,queue&) {if(p){++free_calls;if(fail_free)throw exception("injected free failure");--allocations;std::free(p);}}
}
