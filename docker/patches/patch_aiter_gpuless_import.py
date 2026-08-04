#!/usr/bin/env python3
"""Make `import aiter` succeed on a GPU-less build/verify node.

aiter v0.1.18 (aiter/ops/triton/utils/_triton/arch_info.py) resolves the GPU
arch at import time via triton's active driver, and falls back to `jax` when
triton has no active target (RuntimeError). On a GPU-less node the triton call
raises and jax is not installed, so `import aiter` aborts.

This rewrites the jax fallback to read the arch from the GPU_ARCHS env var
(default gfx942). On a real GPU node the primary triton path succeeds and this
fallback is never executed, so runtime behavior is unchanged.
"""
import importlib.util
import os
import re
import sys

spec = importlib.util.find_spec("aiter")
if spec is None or not spec.origin:
    print("aiter not importable via find_spec; nothing to patch", file=sys.stderr)
    sys.exit(0)
aiter_dir = os.path.dirname(spec.origin)
target = os.path.join(aiter_dir, "ops", "triton", "utils", "_triton", "arch_info.py")

if not os.path.isfile(target):
    print(f"arch_info.py not found at {target}; nothing to patch")
    sys.exit(0)

src = open(target).read()
if "GPU_ARCHS" in src and "gpu_triton" not in src:
    print("arch_info.py already patched")
    sys.exit(0)

# Replace the entire `from jax..._src.lib import gpu_triton` fallback body with an
# env-based fallback. Match the except block that pulls in jax.
pattern = re.compile(
    r"from jax\._src\.lib import gpu_triton as triton_kernel_call_lib\s*\n"
    r"\s*_CACHED_ARCH\s*=\s*triton_kernel_call_lib\.get_arch_details\(\"0\"\)\.split\(\":\"\)\[0\]"
)
replacement = (
    "import os as _os\n"
    "    _CACHED_ARCH = _os.environ.get(\"GPU_ARCHS\", \"gfx942\").split(\";\")[-1]"
)
new, n = pattern.subn(replacement, src)
if n == 0:
    print("WARN: jax fallback pattern not found; arch_info.py may have changed", file=sys.stderr)
    sys.exit(1)

open(target, "w").write(new)
import py_compile
py_compile.compile(target, doraise=True)
print(f"patched aiter arch_info.py ({n} block): jax fallback -> GPU_ARCHS env")
