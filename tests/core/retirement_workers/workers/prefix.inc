#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#define private public
#include "foresight_swap.hpp"
namespace strata::platform { inline void name_this_thread(const char*) {} }
namespace strata::aux_cpus { inline void pin_current_thread() {} }
namespace strata::core {
struct ExpertSource {
    std::function<void()> callback;
    int blobs=0;uint8_t data[8]={};
    virtual bool warms(){if(callback)callback();return true;}
    bool pinned(int64_t,int32_t){if(callback)callback();return true;}
    const uint8_t* blob(int64_t,int32_t){++blobs;return data;}
};
inline bool valid_retirement_topology(const std::vector<int>&,std::string&){return true;}
inline int device_calls=0;
using cudaError_t=int;
using cudaEvent_t=void*;
constexpr int cudaSuccess=0,cudaErrorNotReady=1,cudaMemcpyDeviceToHost=2;
int cudaEventQuery(void*){++device_calls;return 0;}
int cudaGetLastError(){++device_calls;return 0;}
int cudaMemcpy(void* out,const void* in,size_t n,int){++device_calls;std::memcpy(out,in,n);return 0;}
struct ExpertResourceDevice{ExpertResourceDevice(bool,int){}};
struct OnDevice{explicit OnDevice(int){}};
[[noreturn]] void retirement_failed(const char*){std::abort();}
