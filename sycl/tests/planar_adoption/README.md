`extract.py` copies the exact marked production adoption transaction from
`sycl/src/kernels/cuda/native_mmvq.dp.cpp` and selected registry release/query/numeric-lookup functions into an include for this CPU test. The
queue executes the conversion loop synchronously; allocator, registry insertion,
drain and free failures are injected. This checks unchanged plane bytes, safe
fallback before source retirement, and retained pointers/registry on uncertainty.
It also verifies retained packed-copy tombstones block free retries and numeric fallback.
It is not a full SYCL kernel build or asynchronous hardware qualification.

```sh
python3 sycl/tests/planar_adoption/extract.py \
  sycl/src/kernels/cuda/native_mmvq.dp.cpp /tmp/planar_transaction.inc
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -I/tmp \
  sycl/tests/planar_adoption/test.cpp -o /tmp/strata-planar-transaction
/tmp/strata-planar-transaction
```
