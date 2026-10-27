#pragma once
// CPU test seam, not a SYCL runtime.
#include <dpct/dpct.hpp>
namespace strata {
inline void* malloc_device_guarded(size_t n,sycl::queue& q){return sycl::malloc_host(n,q);}
}
