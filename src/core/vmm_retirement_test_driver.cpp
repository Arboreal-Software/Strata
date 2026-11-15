// CPU fake for the actual CUDA VMM branch. Uses CUDA SDK declarations only;
// supplies runtime/driver entry points itself, links no CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
static int device=0;
static unsigned creates=0,releases=0,maps=0;
static bool fail_map=false,fail_access=false,fail_release=false,fail_unmap=false;
static uint64_t next_handle=1;
static int fail_create_after=-1;
static unsigned queries=0,syncs=0,unmaps=0,va_frees=0;
static int fail_sync=-1,fail_release_at=-1,fail_unmap_at=-1;
static bool fail_va=false,fail_get=false;
static unsigned sets=0;
static int fail_set_at=-1;
void fake_fail_get(){fail_get=true;} void fake_fail_set_at(int n){fail_set_at=n;} unsigned fake_sets(){return sets;}

unsigned fake_syncs(){return syncs;} unsigned fake_unmaps(){return unmaps;} unsigned fake_va_frees(){return va_frees;}
void fake_fail_sync(int at){fail_sync=at;} void fake_fail_release_at(int at){fail_release_at=at;}
void fake_fail_unmap_at(int at){fail_unmap_at=at;} void fake_fail_va_free(){fail_va=true;}

void fake_fail_create_after(int n){fail_create_after=n;}
unsigned fake_queries(){return queries;}
void fake_device(int d){device=d;}
void fake_fail_next_map(){fail_map=true;}
void fake_fail_next_access(){fail_access=true;}
void fake_fail_next_release(){fail_release=true;}
void fake_fail_next_unmap(){fail_unmap=true;}
unsigned fake_creates(){return creates;}
unsigned fake_releases(){return releases;}
unsigned fake_maps(){return maps;}
static CUresult CUDAAPI attr(int* v,CUdevice_attribute,CUdevice){*v=1;return CUDA_SUCCESS;}
static CUresult CUDAAPI gran(size_t* n,const CUmemAllocationProp*,CUmemAllocationGranularity_flags){*n=4096;return CUDA_SUCCESS;}
static CUresult CUDAAPI reserve(CUdeviceptr* p,size_t,size_t,CUdeviceptr,unsigned long long){*p=0x100000;return CUDA_SUCCESS;}
static CUresult CUDAAPI freeva(CUdeviceptr,size_t){++va_frees;return fail_va?CUDA_ERROR_UNKNOWN:CUDA_SUCCESS;}
static CUresult CUDAAPI create(CUmemGenericAllocationHandle* h,size_t,const CUmemAllocationProp*,unsigned long long){if(fail_create_after==0){fail_create_after=-1;return CUDA_ERROR_OUT_OF_MEMORY;}if(fail_create_after>0)--fail_create_after;++creates;*h=next_handle++;return CUDA_SUCCESS;}
static CUresult CUDAAPI release(CUmemGenericAllocationHandle){if(fail_release||int(releases+1)==fail_release_at){fail_release=false;return CUDA_ERROR_UNKNOWN;}++releases;return CUDA_SUCCESS;}
static CUresult CUDAAPI map(CUdeviceptr,size_t,size_t,CUmemGenericAllocationHandle,unsigned long long){++maps;if(fail_map){fail_map=false;return CUDA_ERROR_UNKNOWN;}return CUDA_SUCCESS;}
static CUresult CUDAAPI unmap(CUdeviceptr,size_t){++unmaps;if(fail_unmap||int(unmaps)==fail_unmap_at){fail_unmap=false;return CUDA_ERROR_UNKNOWN;}return CUDA_SUCCESS;}
static CUresult CUDAAPI access(CUdeviceptr,size_t,const CUmemAccessDesc*,size_t){if(fail_access){fail_access=false;return CUDA_ERROR_UNKNOWN;}return CUDA_SUCCESS;}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* d){++queries;if(fail_get){fail_get=false;return cudaErrorUnknown;}*d=device;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaSetDevice(int d){++sets;if(int(sets)==fail_set_at)return cudaErrorUnknown;device=d;return cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize(){++syncs;return int(syncs)==fail_sync?cudaErrorUnknown:cudaSuccess;}
extern "C" cudaError_t CUDARTAPI cudaGetLastError(){return cudaSuccess;}
static cudaError_t resolve(const char* n,void** p,cudaDriverEntryPointQueryResult* q){
#define ENTRY(name,fn) if(std::strcmp(n,name)==0){*p=reinterpret_cast<void*>(&fn);*q=cudaDriverEntryPointSuccess;return cudaSuccess;}
 ENTRY("cuDeviceGetAttribute",attr)
 ENTRY("cuMemGetAllocationGranularity",gran)
 ENTRY("cuMemAddressReserve",reserve)
 ENTRY("cuMemAddressFree",freeva)
 ENTRY("cuMemCreate",create)
 ENTRY("cuMemRelease",release)
 ENTRY("cuMemMap",map)
 ENTRY("cuMemUnmap",unmap)
 ENTRY("cuMemSetAccess",access)
#undef ENTRY
 return cudaErrorUnknown;
}
#if CUDART_VERSION >= 12050
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(const char* n,void** p,unsigned int,unsigned long long,cudaDriverEntryPointQueryResult* q){return resolve(n,p,q);}
#else
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char* n,void** p,unsigned long long,cudaDriverEntryPointQueryResult* q){return resolve(n,p,q);}
#endif
