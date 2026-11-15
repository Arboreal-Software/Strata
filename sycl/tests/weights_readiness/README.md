This test compiles the actual SYCL canonical loader with small CPU implementations
of only its allocation, copy and queue-drain interfaces. It checks the shared
`WeightTable` lifetime ABI, real index/data-file loading and readiness publication
under injected copy/drain failure. It is not a SYCL backend build, GPU test or
asynchronous memory-safety qualification.

From the repository root, with a C++20 compiler:

```sh
g++ -std=c++20 -O1 -g -fsanitize=address,undefined \
  -Isycl/tests/weights_readiness/shim -Isycl/include -Iinclude \
  sycl/src/core/weights.cpp sycl/tests/weights_readiness/test.cpp \
  -o /tmp/strata-sycl-weights-readiness
/tmp/strata-sycl-weights-readiness
```

HC requantization is absent from the SYCL implementation. Its shared entrypoint
returns an explicit unsupported error with zero bytes, preserving an otherwise
completed canonical load. No auxiliary allocation is claimed or freed.
