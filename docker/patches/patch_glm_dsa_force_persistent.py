#!/usr/bin/env python3
"""Force the AITER sparse-MLA persistent path ON for GLM gqa64 fp8 disagg on gfx942.

ROOT CAUSE (why the baked persistent-kernel gate is fatal for this model):
  The base image bakes the "persistent-kernel gate" (aiter #4076 / vLLM #47567) into
  v1/attention/backends/mla/rocm_aiter_mla_sparse.py::build():

      use_persistent = not is_chunked_continuation.any()
      ...
      work_meta_data=self._mla_work_meta_data if use_persistent else None,

  For a CHUNKED-PREFILL CONTINUATION the gate sets work_meta_data=None -> aiter takes
  the NON-persistent split-KV path. That is fine for gqa_ratio<=16, but GLM-5.1-FP8
  DSA runs gqa_ratio=64, and aiter has NO non-persistent gqa64 fp8 kernel:

      [AITER] asm_mla.cu:949 mla_decode_stage1_asm_fwd:
              fp8/fp8 with gqa_ratio=64 only supports persistent mode

  so the prefill worker hard-crashes on the first long-context (>1 chunk) request
  (observed on 1P1D EP8 job 205754 and 2P2D EP16 job 205755, both aiter0118+qh16fold).

FIX:
  Force use_persistent = True. gqa64 fp8 then always takes the persistent qh16 path
  (via patch_aiter_mla_qh64_fold.py). PR#47766's 6-field sparse-MLA metadata key makes
  the persistent kernel numerically CORRECT across chunked prefill, which is exactly
  the condition the gate was working around -- so with #47766 present the gate is both
  unnecessary and (for gqa64) fatal. Removing it is the correct move here.

  Trade-off / accountability: this trusts #47766 to cover chunked-prefill correctness.
  If long-context NIAH regresses (repetition/garbage past the first chunk boundary),
  that indicates #47766 does NOT fully cover gqa64 chunked prefill and the real fix is
  the aiter-side kernel (aiter #3921 / AITERKER-132) plus a gqa64 non-persistent kernel.
  Either way this unblocks disagg from a deterministic crash so NIAH can measure it.

Idempotent + anchor-based + self-locating (GPU-free). Missing anchor -> hard error
(so we never silently ship the crashing gate).

Usage: patch_glm_dsa_force_persistent.py [<vllm_install_dir>]
"""
import os
import sys

REL = "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"

if len(sys.argv) > 1:
    VLLM = sys.argv[1]
else:
    import importlib.util
    spec = importlib.util.find_spec("vllm")
    VLLM = os.path.dirname(spec.origin)

path = os.path.join(VLLM, REL)
if not os.path.isfile(path):
    print(f"[force-persist] {REL} not found under {VLLM} -- skipping (backend layout differs).")
    sys.exit(0)

src = open(path).read()

if "GLM force-persistent" in src:
    print("[force-persist] already patched -- no-op.")
    sys.exit(0)

OLD = "        use_persistent = not is_chunked_continuation.any()"
NEW = (
    "        use_persistent = True  # GLM force-persistent: gqa64 fp8 has no non-persistent\n"
    "        # kernel (asm_mla.cu:949); keep persistent qh16 path, rely on PR#47766 for\n"
    "        # chunked-prefill correctness. Was: not is_chunked_continuation.any()\n"
    "        _ = is_chunked_continuation"
)

if OLD not in src:
    print("[force-persist] ERROR: 'use_persistent = not is_chunked_continuation.any()' "
          "anchor not found in rocm_aiter_mla_sparse.py (image drift). Aborting.",
          file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
open(path, "w").write(src)

chk = open(path).read()
if "GLM force-persistent" not in chk:
    print("[force-persist] ERROR: post-write verification failed.", file=sys.stderr)
    sys.exit(1)

import py_compile
try:
    py_compile.compile(path, doraise=True)
except Exception as e:  # noqa: BLE001
    print(f"[force-persist] ERROR: patched file fails to compile: {e}", file=sys.stderr)
    sys.exit(1)

print(f"[force-persist] patched {path}: use_persistent forced True "
      "(gqa64 fp8 keeps persistent qh16 path; relies on PR#47766).")
