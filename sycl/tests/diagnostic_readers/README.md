# SYCL verifier diagnostic reader lifetime

`extract.py` extracts the actual production latest-diagnostic and live-verifier release callbacks. It also checks that destructor retirement precedes device cleanup and that publication remains at the beginning of init. The fake verifier pauses inside either callback while another thread retires it; the test observes unpublication through the real registry before releasing the callback, proving cleanup cannot proceed while that reader is active. It checks the 16-slot best-effort capacity too.

```
python3 sycl/tests/diagnostic_readers/extract.py sycl/src/core/verify.cpp /tmp/callbacks.inc
g++ -std=c++20 -O1 -g -pthread -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude -I/tmp sycl/tests/diagnostic_readers/test.cpp -o /tmp/diagnostic_callbacks
ASAN_OPTIONS=detect_leaks=1 /tmp/diagnostic_callbacks
```

This tests reader lifetime, not synchronization of mutable verifier diagnostic fields, GPU completion, or device teardown error handling.

Full current-source SYCL compilation and hardware qualification remain separate from this extracted CPU fixture.
