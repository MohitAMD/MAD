#!/usr/bin/env python3
"""
Self-healing aiter JIT baton patch (v2 — robust across aiter 0.1.13..0.1.16+).

Same intent as patch_aiter_baton_timeout.py but tolerant of the file_baton.py
log-line/formatting drift in newer aiter releases (0.1.16.post2 changed wait()'s
logger.info to a multi-line pid/pname message, so the old exact-string anchor
missed). Three GPU-free, idempotent edits:

  1. file_baton.py wait(self) -> wait(self, timeout=None): bounded spin, returns
     True if released, False on timeout (was: spin forever -> permanent deadlock
     if the builder died/hung mid cold-kernel build).
  2. file_baton.py release(): tolerate a missing lock file (FileNotFoundError) —
     the observed 1P1D decode crash was release() racing a stolen lock.
  3. core.py mp_lock(): loop; on wait() timeout, remove the abandoned lock and
     retry try_acquire() (become the builder). Timeout via AITER_BATON_TIMEOUT
     (default 600s).

Usage: python3 patch_aiter_baton_selfheal_v2.py [<aiter_pkg_dir>]
"""
import os, sys, re

if len(sys.argv) > 1:
    AITER = sys.argv[1]
else:
    import importlib.util
    spec = importlib.util.find_spec("aiter")
    AITER = os.path.dirname(spec.origin)

fb = os.path.join(AITER, "jit/utils/file_baton.py")
core = os.path.join(AITER, "jit/core.py")

# ---- 1+2. file_baton.py: wait() timeout + release() guard --------------------
with open(fb) as f:
    s = f.read()

if "def wait(self, timeout=None)" in s:
    print("[baton-v2] file_baton.py wait() already patched")
else:
    # Match the whole wait() method regardless of the logger.info wording:
    # from `def wait(self):` through the `time.sleep(self.wait_seconds)` spin line.
    wait_re = re.compile(
        r"    def wait\(self\):.*?while os\.path\.exists\(self\.lock_file_path\):\n"
        r"            time\.sleep\(self\.wait_seconds\)",
        re.DOTALL,
    )
    new_wait = (
        "    def wait(self, timeout=None):\n"
        "        \"\"\"Bounded spin until the baton is released or ``timeout`` s elapse.\n"
        "        Returns True if released, False on timeout. (Patched: the original\n"
        "        spun forever -> deadlock if the builder died/hung without release().)\"\"\"\n"
        "        logger.info(f\"waiting for baton release at {self.lock_file_path}\")\n"
        "        _start = time.monotonic()\n"
        "        while os.path.exists(self.lock_file_path):\n"
        "            if timeout is not None and (time.monotonic() - _start) > timeout:\n"
        "                return False\n"
        "            time.sleep(self.wait_seconds)\n"
        "        return True"
    )
    s2, n = wait_re.subn(new_wait, s, count=1)
    if n != 1:
        print("ERROR: file_baton.py wait() anchor not found (regex)", file=sys.stderr)
        sys.exit(1)
    s = s2

    # Guard release()'s os.remove against a stolen/missing lock.
    if "os.remove(self.lock_file_path)" in s and "except FileNotFoundError" not in s:
        s = s.replace(
            "        os.remove(self.lock_file_path)",
            "        try:\n"
            "            os.remove(self.lock_file_path)\n"
            "        except FileNotFoundError:\n"
            "            pass",
            1,
        )
    with open(fb, "w") as f:
        f.write(s)
    print("[baton-v2] file_baton.py wait()->timeout + release() guarded")

# ---- 3. core.py: mp_lock() self-heal loop -----------------------------------
with open(core) as f:
    c = f.read()

if "AITER_BATON_TIMEOUT" in c:
    print("[baton-v2] core.py mp_lock already patched")
else:
    old_body = '''    baton = FileBaton(lockPath)
    if baton.try_acquire():
        try:
            ret = MainFunc()
        finally:
            if FinalFunc is not None:
                FinalFunc()
            baton.release()
    else:
        baton.wait()
        if WaitFunc is not None:
            ret = WaitFunc()
        ret = None
    return ret'''
    new_body = '''    import os as _os
    _timeout = float(_os.environ.get("AITER_BATON_TIMEOUT", "600"))
    while True:
        baton = FileBaton(lockPath)
        if baton.try_acquire():
            try:
                ret = MainFunc()
            finally:
                if FinalFunc is not None:
                    FinalFunc()
                baton.release()
            return ret
        _released = baton.wait(timeout=_timeout)
        if _released:
            if WaitFunc is not None:
                WaitFunc()
            return None
        logger.warning(
            f"[aiter-baton] wait timed out after {_timeout}s on {lockPath}; "
            "removing abandoned lock and retrying build."
        )
        try:
            _os.remove(lockPath)
        except FileNotFoundError:
            pass
        except Exception:
            pass'''
    if old_body not in c:
        print("ERROR: core.py mp_lock() body anchor not found", file=sys.stderr)
        sys.exit(1)
    c = c.replace(old_body, new_body, 1)
    with open(core, "w") as f:
        f.write(c)
    print("[baton-v2] core.py mp_lock() -> self-healing")

import py_compile
py_compile.compile(fb, doraise=True)
py_compile.compile(core, doraise=True)
print("[baton-v2] OK -- both files compile")
