# GLM5.1-FP8 from-scratch build — fix changelog

## Fix 1: aiter arch detection fails on GPU-less build node (rocm_final 7/7)
- **Symptom:** `RuntimeError: Get GPU arch from rocminfo failed: ... rocminfo returned non-zero exit status 1`
  during `RUN python3 -c "import torch, aiter, mori, ..."` in the `rocm_final` stage.
- **Root cause:** aiter's `aiter/jit/utils/chip_info.py` `get_gfx_custom_op_core()` reads
  `GPU_ARCHS` (default `"native"`); when unset/`native` it shells out to `rocminfo`, which
  fails on the CPU-only build/login node.
- **Fix:** Added `ENV GPU_ARCHS=gfx942` in `rocm_final` right before the import sanity check.
  Build-time codegen now uses the pinned arch; runtime dispatch is unaffected because
  aiter uses a separate `get_gfx_runtime()` that always re-detects the live GPU.
- **Cache impact:** placed after the wheel-install layers so the expensive
  build_pytorch/aiter/mori/fa/triton stages stay cached; only rocm_final's import check reruns.
- No pins changed.
- **NOTE:** GPU_ARCHS alone was insufficient (see Fix 2/3); rocminfo shim + jax-fallback
  patch were the actual unblockers. GPU_ARCHS is still needed by the jax-fallback patch.

## Fix 2: rocminfo aborts import on GPU-less node (rocm_final)
- **Symptom:** `RuntimeError: Get GPU arch from rocminfo failed` from `get_gfx_runtime()`
  in `aiter/utility/dtypes.py` at aiter import time. `get_gfx_runtime()` ALWAYS calls
  rocminfo and ignores GPU_ARCHS.
- **Fix:** `docker/patches/rocminfo_buildshim.sh` — a transparent rocminfo wrapper.
  Renames real binary to `rocminfo.real`, delegates to it, and only emits a canned
  gfx942/MI300X (304 CU) stanza when the real one fails (no GPU). Runtime on a real GPU
  node delegates to the real binary, so true values are used. Installed in rocm_final
  (propagates to vllm_build/final via FROM chain).
- No pins changed.

## Fix 3: aiter arch_info jax fallback on GPU-less node (rocm_final)
- **Symptom:** `ModuleNotFoundError: No module named 'jax'` from
  `aiter/ops/triton/utils/_triton/arch_info.py` at import. It resolves arch via triton's
  active driver and falls back to `jax._src.lib.gpu_triton` when triton has no active
  target (RuntimeError on a GPU-less node); jax isn't installed.
- **Fix:** `docker/patches/patch_aiter_gpuless_import.py` rewrites that jax fallback to
  read `GPU_ARCHS` (default gfx942). Real GPU nodes take the primary triton path, so
  behavior there is unchanged.
- Rationale: the user's own verification (`docker run ... import aiter`) also runs on this
  GPU-less login node, so aiter must import without a GPU.
- No pins changed.

## Fix 4: vLLM wheel install used wrong directory (vllm_build 7/8)
- **Symptom:** `ERROR: Invalid wheel filename (wrong number of parts): '*'` /
  `dist/*.whl ... file does not exist`.
- **Root cause:** `RUN cd vllm && pip uninstall ... ; cd vllm && pip install dist/*.whl`
  ran a second `cd vllm`, descending into the `vllm/vllm` package source subdir where
  `dist/` does not exist, so the glob never matched.
- **Fix:** collapsed to a single `cd vllm`:
  `RUN cd vllm && (pip uninstall -y vllm 2>/dev/null || true) && pip install --no-deps dist/*.whl`
- No pins changed.

## Fix 5: amd_mori wheel versioned as 0.0.0 (build_mori)
- **Symptom:** verification reported `amd_mori 0.0.0` instead of `1.1.2.dev43+g42e895472`;
  no `python/mori/_version.py` and `mori.__version__ == "unknown"`.
- **Root cause:** mori declares its version dynamically via `setuptools_scm` (a
  `build-system.requires` entry). The Dockerfile builds with `python3 setup.py bdist_wheel`,
  which bypasses PEP 517 build isolation and never installs build requirements, so
  setuptools_scm was absent and the dynamic version silently defaulted to 0.0.0.
- **Fix:** `RUN pip install "setuptools_scm[toml]>=6.2"` in build_mori before the build,
  plus `git fetch --tags --force` so scm can compute the real version, plus a guard that
  fails the build if the wheel is still 0.0.0/0.1.0 (unresolved).
- No pins changed (same commit 42e895472b08).
