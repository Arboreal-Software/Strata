This CPU seam compiles the exact graph ownership structures, capture transaction,
and checked retirement functions extracted from the SYCL sources. It injects
queue/graph allocation, begin/end recording, body refusal/exception, finalize,
executable creation, and graph/queue retirement failures. It checks allocated
array capacity separately from the number of completed layer captures.

The fake graph destructors may throw to test no-retry bookkeeping. A real SYCL
destructor declared `noexcept` may terminate rather than expose such a failure;
this fixture does not claim recoverability beyond the backend's API. There is no
whole-SYCL compile, device graph recording, model execution, or GPU qualification.
The caller still must seal producers and drain device users before retirement.

```sh
python3 sycl/tests/graph_ownership/extract.py sycl/src/core/session.cpp \
  sycl/include/strata/core/session.hpp /tmp/graph_ownership.inc
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -I/tmp \
  sycl/tests/graph_ownership/test.cpp -o /tmp/strata-sycl-graph-test
/tmp/strata-sycl-graph-test
```
