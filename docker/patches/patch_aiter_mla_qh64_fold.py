#!/usr/bin/env python3
"""Force GLM gqa64 fp8 decode off the broken native-qh64 asm-MLA kernel on gfx942.

ROOT CAUSE: aiter's native qh64 fp8 persistent MLA decode kernel (added in PR #3188),
  hsa/gfx942/mla/mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co, GPU memory-access-faults
  on gfx942 at the first disagg forward (confirmed on aiter e03fa6040+#3188, ac90d5c89,
  origin/main tip, and v0.1.18). In aiter/mla.py the persistent dispatch selects it via
  the "Natively support" branch for:
      get_gfx() in ("gfx942","gfx950") and nhead==64 and fp8/fp8 and max_seqlen_q==1
  Immediately below is the working fallback:
      elif nhead in range(32,128+1,16) and persistent_mode:  # fold gqa64 -> qh16
  which uses mla_a8w8_qh16_qseqlen1_gqaratio16_ps (the pre-#3188 path that ran 1P1D
  disagg fine on aiter 0.1.13.post1, job 204438).

FIX: narrow the native-qh64 clause from ("gfx942","gfx950") to ("gfx950",) so gfx942
  gqa64 fp8 qseqlen1 falls through to the qh16 fold. gfx950 behavior is unchanged.
  Trade-off: qh16 fold reintroduces the split-K run-to-run nondeterminism (+/-1 needle),
  but it RUNS (no GPU fault) -- required to get any disagg forward through.

Idempotent + anchor-based + self-locating (find_spec, GPU-free).

Usage: patch_aiter_mla_qh64_fold.py [<aiter_pkg_dir>]
"""
import os
import sys

if len(sys.argv) > 1:
    AITER = sys.argv[1]
else:
    import importlib.util
    spec = importlib.util.find_spec("aiter")
    AITER = os.path.dirname(spec.origin)

f = os.path.join(AITER, "mla.py")
if not os.path.isfile(f):
    print(f"[qh64-fold] {f} not found -- skipping (aiter layout differs).")
    sys.exit(0)

src = open(f).read()

if "GLM qh16-fold" in src:
    print("[qh64-fold] already patched -- no-op.")
    sys.exit(0)

OLD = (
    "            or (\n"
    "                get_gfx() in (\"gfx942\", \"gfx950\")\n"
    "                and nhead == 64\n"
    "                and q.dtype == dtypes.fp8\n"
    "                and kv_buffer.dtype == dtypes.fp8\n"
    "                and max_seqlen_q == 1\n"
    "            )"
)
NEW = (
    "            or (\n"
    "                get_gfx() == \"gfx950\"  # GLM qh16-fold: gfx942 native qh64 .co faults -> fold to qh16\n"
    "                and nhead == 64\n"
    "                and q.dtype == dtypes.fp8\n"
    "                and kv_buffer.dtype == dtypes.fp8\n"
    "                and max_seqlen_q == 1\n"
    "            )"
)

if OLD not in src:
    print("[qh64-fold] ERROR: gqa64 native-support anchor not found in aiter/mla.py "
          "(aiter version drift). Aborting.", file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
open(f, "w").write(src)

chk = open(f).read()
if "GLM qh16-fold" not in chk:
    print("[qh64-fold] ERROR: post-write verification failed.", file=sys.stderr)
    sys.exit(1)

import py_compile
try:
    py_compile.compile(f, doraise=True)
except Exception as e:  # noqa: BLE001
    print(f"[qh64-fold] ERROR: patched file fails to compile: {e}", file=sys.stderr)
    sys.exit(1)

print(f"[qh64-fold] patched {f}: gfx942 gqa64 fp8 decode now folds to qh16 (avoids broken native qh64 .co)")
