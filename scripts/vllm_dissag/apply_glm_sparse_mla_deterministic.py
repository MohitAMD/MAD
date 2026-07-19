#!/usr/bin/env python3
"""Force the DETERMINISTIC (non-persistent, single-pass) sparse-MLA decode path.

WHY (root cause, evidence-based):
  For GLM-5.1-FP8 the sparse-MLA decode kernel (aiter mla_decode_fwd) takes a
  PERSISTENT work-stealing split-K path whenever `work_meta_data` is passed. That
  path reduces attention partial sums across dynamically-stolen KV splits, whose
  accumulation ORDER varies run-to-run. In bf16 the reorder error is below the
  needle-selection boundary, but under fp8 KV the lower mantissa AMPLIFIES it ->
  the fp8 1P1D probe showed 20k Δ=3 vs bf16 Δ=0 (same image/config, KV dtype only).

  The MoRIIO transfer + fp8 dequant themselves are already deterministic (raw RDMA
  byte copy, integer block remap, static per-tensor kv_scale). The nondeterminism
  is the split-K accumulation, NOT the connector.

FIX:
  The image already carries the chunked-prefill persistent-kernel GATE
  (apply_glm_dsa_persistent_kernel_gate_fix.py), which introduced a `_use_persistent`
  flag in ROCMAiterMLASparseMetadataBuilder.build() and already falls back to the
  CORRECT non-persistent split-KV path for chunked-prefill continuations. We simply
  EXTEND that same, already-validated fallback: when GLM_SPARSE_MLA_DETERMINISTIC=1,
  force `_use_persistent=False` for ALL batches (including pure decode), so every
  decode step uses the single-pass deterministic reduction.

  This reuses the proven non-persistent code path (no new kernel path invented).
  Cost: decode throughput regression (loses work-stealing) -- opt-in via env only,
  default OFF so normal serving is unaffected.

CAVEAT: some fp8 shapes (e.g. gqa_ratio=64 under DP8/TP1) only support decode_qlen=1
  in PERSISTENT mode; on such a layout forcing non-persistent may error. That is a
  layout-specific limitation of the aiter ASM kernel, surfaced at runtime -- if it
  triggers, the deterministic fix for that shape requires the aiter kernel fix
  (AITERKER-132 / aiter #3921) rather than this vLLM-side gate.

Idempotent + anchor-based + self-skipping. Requires the persistent-kernel gate patch
to be present (it defines `_use_persistent`); if absent, warn+skip (a no-persistent
image needs no determinism gate).

Usage: apply_glm_sparse_mla_deterministic.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"

# PRIMARY (universal) anchor: forward_decode forwards the persistent split-K
# work-stealing metadata to mla_decode_fwd only when work_meta_data is not None.
# Gating THIS condition on the env flag forces the non-persistent single-pass
# kernel path regardless of how build() decided persistence -- works on baked
# fork images that lack the `_use_persistent` gate variable.
OLD_FWD = "        if attn_metadata.work_meta_data is not None:\n            mla_kwargs.update("
NEW_FWD = (
    "        # DETERMINISM GATE: GLM_SPARSE_MLA_DETERMINISTIC=1 forces the\n"
    "        # non-persistent single-pass split-KV reduction (drops the work-stealing\n"
    "        # split-K metadata), eliminating the accumulation-order nondeterminism\n"
    "        # that fp8 KV amplifies.\n"
    "        if attn_metadata.work_meta_data is not None and not (\n"
    "            __import__('os').environ.get('GLM_SPARSE_MLA_DETERMINISTIC', '0') == '1'\n"
    "        ):\n"
    "            mla_kwargs.update("
)

# OPTIONAL (bonus) anchor: if the persistent-kernel gate patch is present, also
# short-circuit build()'s _use_persistent so it skips the wasted metadata launch.
OLD_BUILD = "        _use_persistent = not bool(_is_chunked_continuation.any())"
NEW_BUILD = (
    "        _use_persistent = not bool(_is_chunked_continuation.any())\n"
    "        if __import__('os').environ.get('GLM_SPARSE_MLA_DETERMINISTIC', '0') == '1':\n"
    "            _use_persistent = False"
)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[glm-det-mla] {REL} not found -- skipping (backend layout differs).")
        return 0

    src = open(path).read()

    if "GLM_SPARSE_MLA_DETERMINISTIC" in src:
        print("[glm-det-mla] already patched (determinism gate present) -- no-op.")
        return 0

    if OLD_FWD not in src:
        print(
            "[glm-det-mla] WARN: forward_decode `if attn_metadata.work_meta_data is not "
            "None:\\n            mla_kwargs.update(` anchor not found -- skipping "
            "(sparse backend refactored on this image).",
            file=sys.stderr,
        )
        return 0

    src = src.replace(OLD_FWD, NEW_FWD, 1)

    # Bonus: also gate build()'s _use_persistent when that variable exists (gate-patched
    # images). Optional -- the forward_decode gate above is sufficient on its own.
    if OLD_BUILD in src:
        src = src.replace(OLD_BUILD, NEW_BUILD, 1)
        print("[glm-det-mla] (bonus) also gated build() _use_persistent.")

    open(path, "w").write(src)

    chk = open(path).read()
    if "GLM_SPARSE_MLA_DETERMINISTIC" not in chk or "not (" not in chk:
        print("[glm-det-mla] ERROR: post-write verification failed.", file=sys.stderr)
        return 1

    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[glm-det-mla] ERROR: patched file fails to compile: {e}", file=sys.stderr)
        return 1

    print(f"[glm-det-mla] patched sparse-MLA determinism gate in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
