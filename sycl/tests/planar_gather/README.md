# SYCL planar gather ownership seam

`extract.py` extracts the exact production gather transaction, existing adoption/quarantine helpers and packed lookup/release/query functions. `test.cpp` supplies a synchronous CPU queue and tracked allocations. It checks all four gathered plane payloads with reversed rows, thrown/null device allocation refusal, launch failure after partial work, asynchronous wait failure, registry insertion failure, retained candidate bytes, no partial publication, numeric fallback fencing, and no retry/free after quarantine.

Run with a C++20 compiler and sanitizers:

```
python3 sycl/tests/planar_gather/extract.py sycl/src/kernels/cuda/native_mmvq.dp.cpp /tmp/planar_transaction.inc
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I/tmp sycl/tests/planar_gather/test.cpp -o /tmp/planar_gather
ASAN_OPTIONS=detect_leaks=1 /tmp/planar_gather
```

The fixture manually frees retained records only during test reset, to let leak sanitizer check the fixture. Production quarantine intentionally retains them until process exit. Failed candidates are charged to their input planar owner through native_head_packed_allocation_bytes(input); successful returned candidates remain caller-owned main allocations. Failed allocation with no returned pointer has no candidate bytes to retain. Failure to preallocate the bookkeeping record returns nullptr before any device operation; that branch is source-inspected, not allocator-injected here.

This seam does not execute GPU kernels. Full current-source SYCL compilation and hardware qualification remain separate.

MTP still needs a checked caller transaction for freeing its old subset allocation and adopting the successful returned candidate. This test does not qualify that transaction, full MTP accounting, or actual driver ownership integration. Kernel input geometry validation and cross-thread registry locking are unchanged.
