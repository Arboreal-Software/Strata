#pragma once
#include <sycl/sycl.hpp>
#include <iostream>
namespace dpct {
using err0=int;
inline constexpr int default_error=999;
inline unsigned selected_device=0;
inline unsigned get_current_device_id(){return selected_device;}
inline void select_device(unsigned d){selected_device=d;}
inline const char* get_error_string_dummy(int){return "injected error";}
inline sycl::queue& get_in_order_queue(){static sycl::queue q;return q;}
struct device {
    void queues_wait_and_throw(){if(sycl::fail_drain)throw sycl::exception("injected drain failure");}
};
inline device& get_current_device(){static device d;return d;}
}
// Mirrors the repository helper's error-mapping behavior for the used seam.
#define DPCT_CHECK_ERROR(expr) [&](){try{expr;return 0;}catch(const std::exception&){return 999;}}()
