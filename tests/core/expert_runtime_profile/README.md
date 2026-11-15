# Expert runtime profile CPU fixture

The public target `expert_runtime_profile_test` compiles eleven production units plus the synthetic fixture in `src/core/expert_runtime_profile_test.cpp`. Its source closure matches the prepared standalone candidate. It exercises actual immutable input bundles, expert layout/source receipts and the partial profile collector using tiny generated files.

CUDA API operations and BF16 `native_fmt` selection are explicit test seams. The target uses CUDA headers and `CUDA::cudart` for remaining host link dependencies, and compiles `pinned.cu` as host C++ in this subdirectory only. Function/data sections and `--gc-sections` intentionally discard unexercised engine methods. It does not link engine/kernel archives or a second `native_fmt` implementation. This is not whole-engine or GPU qualification.

Registration requires Linux, CUDA (excluding HIP and gfx906 compatibility), `STRATA_BUILD_WINDOW_STATE=ON` and `STRATA_BUILD_CONVERSATION_TESTS=ON`. Build `expert_runtime_profile_test` and run the identically named CTest in an authorized CPU window. Assertions and ASan/UBSan remain enabled; CTest has a45second timeout and hides CUDA devices. Compilation resource limits must come from the invoking build environment.

No source extraction is used: production source units are compiled directly. Driver snippets/pins in private evidence are contextual and are not compiled by this fixture; no driver freshness check or execution claim is implied.

This public CMake registration is new and unrun. The existing frozen standalone candidate is unchanged. No HIP/SYCL/Metal, real model reads, GPU activity, final policy identity or complete runtime execution authority is qualified by this target.
