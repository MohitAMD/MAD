#!/usr/bin/env python3
"""Force DETERMINISTIC, index-stable expert routing (top-k tie-breaking) for GLM MoE.

WHY (root cause, evidence-based):
  The `mori` all2all COMBINE is already an ordered reduction (core::WarpAccum sums in
  fixed slot/node index order in fp32, no float atomics), and vLLM passes weights=None
  so mori does no weighting. So the EP-MoE nondeterminism is NOT unordered combine
  accumulation. The real source is EXPERT ROUTING TIE-BREAKING: GLM (sigmoid grouped
  top-k) selects experts via `torch.topk(..., sorted=False)` / the AITER
  `rocm_aiter_grouped_topk` kernel, neither of which is index-stable on (near-)tied
  gate scores. Tiny numeric jitter in the gate logits can flip which expert wins a
  tie run-to-run -> different expert set -> different output -> borderline needle flip.
  This is the dtype-independent ~Δ1-2 residual floor that survived bf16 KV (203771).

FIX (two coordinated edits in grouped_topk_router.py):
  1. Native `grouped_topk` already supports deterministic selection via
     `use_sorted = envs.VLLM_BATCH_INVARIANT` (sorted=True topk -> stable index-order
     tie-break). Extend that trigger to also fire on GLM_MOE_DETERMINISTIC_ROUTING=1.
  2. GLM's config routes through the AITER kernel (rocm_aiter_grouped_topk) which
     BYPASSES that deterministic native path. When GLM_MOE_DETERMINISTIC_ROUTING=1,
     force `_compute_routing` to use the native (deterministic) grouped_topk instead.

  Safe for GLM because it runs with fused-shared-experts OFF
  (VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 -> num_fused_shared_experts=0), so the
  native path is functionally equivalent minus the tie nondeterminism. Cost: routing
  runs in PyTorch instead of the fused AITER kernel (small vs expert GEMM). Opt-in via
  env only; default OFF.

NOTE: this addresses the routing/tie-break root. Any residual from AITER fused-MoE
  GEMM internals (fp8 accumulation) is a separate, kernel-level concern.

Idempotent + anchor-based + self-skipping. Missing anchor -> warn+skip.

Usage: apply_glm_moe_deterministic_routing.py <vllm_install_dir>
"""
import os
import sys

REL = "model_executor/layers/fused_moe/router/grouped_topk_router.py"

# Edit 1: extend the deterministic-selection trigger in native grouped_topk.
OLD1 = "    use_sorted = envs.VLLM_BATCH_INVARIANT"
NEW1 = (
    "    use_sorted = envs.VLLM_BATCH_INVARIANT or (\n"
    "        __import__('os').environ.get('GLM_MOE_DETERMINISTIC_ROUTING', '0') == '1'\n"
    "    )"
)

# Edit 2: force the native (deterministic) grouped_topk over the AITER kernel when
# the determinism flag is set. Anchored on the comment + selector in _compute_routing.
OLD2 = (
    "        # Select grouped_topk implementation\n"
    "        if rocm_aiter_ops.is_fused_moe_enabled():"
)
NEW2 = (
    "        # Select grouped_topk implementation\n"
    "        _glm_det_routing = (\n"
    "            __import__('os').environ.get('GLM_MOE_DETERMINISTIC_ROUTING', '0') == '1'\n"
    "        )\n"
    "        if rocm_aiter_ops.is_fused_moe_enabled() and not _glm_det_routing:"
)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[glm-det-moe] {REL} not found -- skipping (router layout differs).")
        return 0

    src = open(path).read()

    if "GLM_MOE_DETERMINISTIC_ROUTING" in src:
        print("[glm-det-moe] already patched (deterministic routing present) -- no-op.")
        return 0

    if OLD1 not in src:
        print(
            "[glm-det-moe] WARN: `use_sorted = envs.VLLM_BATCH_INVARIANT` anchor not "
            "found -- skipping (router refactored).",
            file=sys.stderr,
        )
        return 0
    if OLD2 not in src:
        print(
            "[glm-det-moe] ERROR: found the use_sorted anchor but NOT the "
            "`# Select grouped_topk implementation` selector -- refusing partial patch "
            "(would honor sorted=True but still call the AITER kernel). Aborting.",
            file=sys.stderr,
        )
        return 1

    src = src.replace(OLD1, NEW1, 1)
    src = src.replace(OLD2, NEW2, 1)
    open(path, "w").write(src)

    chk = open(path).read()
    if chk.count("GLM_MOE_DETERMINISTIC_ROUTING") < 2 or "and not _glm_det_routing" not in chk:
        print("[glm-det-moe] ERROR: post-write verification failed.", file=sys.stderr)
        return 1

    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[glm-det-moe] ERROR: patched file fails to compile: {e}", file=sys.stderr)
        return 1

    print(f"[glm-det-moe] patched deterministic MoE routing in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
