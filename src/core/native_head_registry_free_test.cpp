#include <cuda_runtime.h>
extern cudaError_t native_head_test_free(void*);
extern "C" cudaError_t cudaFree(void* p){return native_head_test_free(p);}
