#!/usr/bin/env python3
"""
Self-healing aiter JIT baton patch.

Root problem: aiter/jit/utils/file_baton.py FileBaton.wait() spins forever
(`while os.path.exists(lock): sleep`) with NO timeout, and release() only runs
in a `finally`. If the worker holding the baton dies/hangs during a cold kernel
build (or a prior job was SIGKILL'd), the lock file is never removed and every
other worker's wait() deadlocks permanently -- observed repeatedly on decode
bring-up (lock_module_gemm_a8w8_blockscale / moe_fmoe_asm / rmsnorm).

Fix (GPU-free, in-place .py edits; idempotent):
  1. file_baton.py: wait(timeout=None) -> returns True if released, False on timeout.
  2. core.py mp_lock(): loop; on wait() timeout, remove the abandoned lock and
     retry try_acquire() (become the builder). Timeout via AITER_BATON_TIMEOUT
     (default 600s -- longer than any legit single-kernel build, short enough to
     recover from a dead builder).

Usage: python3 patch_aiter_baton_timeout.py [<aiter_pkg_dir>]
"""
import os, sys, re

if len(sys.argv) > 1:
    AITER = sys.argv[1]
else:
    # locate aiter without importing it (import needs a GPU on ROCm)
    import importlib.util
    spec = importlib.util.find_spec("aiter")
    AITER = os.path.dirname(spec.origin)

fb = os.path.join(AITER, "jit/utils/file_baton.py")
core = os.path.join(AITER, "jit/core.py")

# ---- 1. file_baton.py: wait() with timeout ----------------------------------
with open(fb) as f:
    s = f.read()

if "def wait(self, timeout=None)" in s:
    print("[baton] file_baton.py already patched")
else:
    old_wait = '''    def wait(self):
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        """
        logger.info(f"waiting for baton release at {self.lock_file_path}")
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)'''
    new_wait = '''    def wait(self, timeout=None):
        """
        Periodically sleeps until the baton is released, or until ``timeout``
        seconds elapse. Returns True if released, False on timeout.

        (Patched: the original spun forever -> permanent deadlock if the builder
        process died/hung without release().)
        """
        logger.info(f"waiting for baton release at {self.lock_file_path}")
        _start = time.monotonic()
        while os.path.exists(self.lock_file_path):
            if timeout is not None and (time.monotonic() - _start) > timeout:
                return False
            time.sleep(self.wait_seconds)
        return True'''
    if old_wait not in s:
        print("ERROR: file_baton.py wait() anchor not found", file=sys.stderr)
        sys.exit(1)
    s = s.replace(old_wait, new_wait, 1)
    with open(fb, "w") as f:
        f.write(s)
    print("[baton] file_baton.py wait() -> timeout-aware")

# ---- 2. core.py: mp_lock() self-heal loop -----------------------------------
with open(core) as f:
    c = f.read()

if "AITER_BATON_TIMEOUT" in c:
    print("[baton] core.py mp_lock already patched")
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
        # not the builder: wait (bounded) for the builder to finish
        _released = baton.wait(timeout=_timeout)
        if _released:
            if WaitFunc is not None:
                WaitFunc()
            return None
        # timed out -> builder likely died; steal the abandoned lock and retry
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
    print("[baton] core.py mp_lock() -> self-healing (steal abandoned lock)")

# ---- verify ------------------------------------------------------------------
import py_compile
py_compile.compile(fb, doraise=True)
py_compile.compile(core, doraise=True)
print("[baton] OK -- both files compile")
