# Native RoPE policy selector fixture

`prepare.py REPOSITORY OUTPUT --check` verifies exact production/header pins and extracted selector bodies plus the Metal unsupported-return generator rule. Run it against `tests/core/native_rope_policy/extracted` before compilation. To intentionally refresh after a source change, omit `--check` and review the resulting diff.

Compile test.cpp as C++17 with assertions enabled, include/ on the include path and SELECTOR_FILE defined as the absolute quoted path to cuda.inc or sycl.inc. Compile the CUDA extraction a second time with TEST_HIP_SELECTOR=1 to exercise the HIP policy preprocessor branch. Each executable takes one mode argument (0,1,2,3), run as separate processes because selectors cache their configuration.

The only extraction substitution is std::getenv to a fixture function returning literal values. Tests cover default and opt-in/off policies, geometry refusal, cached-value stability, and mutable native enable. The HIP macro case is not HIP compiler qualification. GPU TUs and Metal generator are freshness inputs only. No actual environment, device/table/dispatch or numerical correctness is observed.

Validation: all three variants and four modes each passed in the bounded Linux CPU fixture with g++ C++17, -O0 -g0 -UNDEBUG -Wall -Wextra, ASan/UBSan and -fno-omit-frame-pointer. Production source matches the frozen tested inputs. Real CUDA/HIP/SYCL/Metal builds and generator execution remain unrun; no driver/schema integration or complete profile claim.
