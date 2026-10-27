This CPU seam compiles the actual SYCL `native_head.cpp` against small queue/free
stubs and seeds already-loaded allocation fixtures. It exercises checked release,
padded allocation accounting, mapped-alias versus device embedding ownership,
wrong-device refusal, device restoration and sticky failure behavior. It does not
exercise GGUF loading, registry helper implementation, real SYCL compilation or
asynchronous GPU execution. The production source's unused load/kernel functions
are discarded by the linker; their C++ bodies are still parsed.

```sh
g++ -std=c++20 -O1 -g -fsanitize=address,undefined \
  -ffunction-sections -fdata-sections \
  -Isycl/tests/native_lifetime/shim -Isycl/tests/weights_readiness/shim -Isycl/include -Iinclude \
  sycl/src/core/native_head.cpp sycl/tests/native_lifetime/test.cpp \
  -Wl,--gc-sections -o /tmp/strata-sycl-native-lifetime
/tmp/strata-sycl-native-lifetime
```
