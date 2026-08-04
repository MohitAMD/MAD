#!/usr/bin/env python3
"""Harden vLLM scheduler against stale KV-transfer-finished req_ids.

PROBLEM: v1/core/sched/scheduler.py::_update_from_kv_xfer_finished asserts
`req_id in self.requests` for every finished_recving / finished_sending id the
KV connector reports. On a disagg (MoRIIO) deployment a transfer can complete
(or be reported) AFTER the scheduler already dropped/aborted the request
(e.g. a cold-start indexer JIT stall trips the 60s deferred-write timeout ->
the request aborts -> a late finished_sending id arrives). The bare assert then
kills the whole prefill EngineCore -> the router evicts the prefill -> 503.

FIX: replace the two `assert req_id in self.requests` with a skip-and-warn so a
stale/unknown id can never crash the engine core. Idempotent, anchor-based,
self-skipping. Usage: patch_glm_sched_kv_xfer_stale_guard.py [<vllm_dir>]
"""
import os, sys

if len(sys.argv) > 1:
    VLLM = sys.argv[1]
else:
    import importlib.util
    VLLM = os.path.dirname(importlib.util.find_spec("vllm").origin)

f = os.path.join(VLLM, "v1/core/sched/scheduler.py")
if not os.path.isfile(f):
    print(f"[glm-sched-guard] {f} not found -- skipping"); sys.exit(0)

s = open(f).read()
if "KV recv finished for unknown" in s or "KV send finished for unknown" in s:
    print("[glm-sched-guard] already patched -- no-op"); sys.exit(0)

recv_old = '''            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests'''
recv_new = '''            logger.debug("Finished recving KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.warning(
                    "KV recv finished for unknown/stale req %s; skipping", req_id
                )
                continue'''

send_old = '''            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests'''
send_new = '''            logger.debug("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.warning(
                    "KV send finished for unknown/stale req %s; skipping", req_id
                )
                continue'''

n = 0
for old, new in ((recv_old, recv_new), (send_old, send_new)):
    if old in s:
        s = s.replace(old, new, 1); n += 1
    else:
        print(f"[glm-sched-guard] WARN: anchor not found:\n{old[:80]}...", file=sys.stderr)

if n == 0:
    print("[glm-sched-guard] ERROR: no anchors matched (scheduler refactored)", file=sys.stderr)
    sys.exit(1)

open(f, "w").write(s)
import py_compile
py_compile.compile(f, doraise=True)
print(f"[glm-sched-guard] patched {n}/2 KV-xfer-finished asserts -> skip stale req_ids")
