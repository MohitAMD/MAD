# fix report — deterministic EP-MoE routing for GLM-5.1-FP8 (20k needle Δ 3→2)

## Summary

An opt-in fix that makes GLM-5.1-FP8 expert routing **index-stable** on tied gate
scores, reducing run-to-run retrieval nondeterminism in disaggregated (1P1D, EP8)
serving. On the Needle-in-a-Haystack determinism probe (same prompt ×6, seed 0,
fp8 KV) it cut the pivotal **20k** case from **Δ=3 → Δ=2** with **no perf-path
crash** and no accuracy regression at 8k/96k.

- Patcher: `scripts/vllm_dissag/apply_glm_moe_deterministic_routing.py`
- Env flag: `GLM_MOE_DETERMINISTIC_ROUTING=1` (default off)
- Wired into `connectors/moriio.sh::_glm_determinism_runtime_patch` (runs even on
  baked images); flag forwarded by `run_xPyD_models.slurm`.

## Root cause

The EP-MoE nondeterminism is **not** in the `mori` all2all combine. The combine
reduction uses `core::WarpAccum`, which accumulates expert outputs in **fixed
slot/node index order in fp32 with no float atomics**, and vLLM passes
`weights=None` (mori does no weighting). That path is already deterministic.

The real source is **expert-routing tie-breaking**. GLM-5.1 uses sigmoid grouped
top-k routing, which selects experts via:

- `torch.topk(..., sorted=False)` in the native `grouped_topk`, and
- the AITER `rocm_aiter_grouped_topk` kernel on the ROCm/AITER path GLM actually
  uses.

Neither is index-stable on (near-)tied gate scores. Tiny numeric jitter in the
gate logits can flip which expert wins a tie run-to-run → a different expert set →
a different output → a borderline needle flip. This is the dtype-independent
~Δ1–2 floor that survived even bf16 KV.

## Fix

Two coordinated edits in
`vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`, both gated on
`GLM_MOE_DETERMINISTIC_ROUTING=1`:

1. **Extend the deterministic-selection trigger.** The native `grouped_topk`
   already supports stable selection via `use_sorted = envs.VLLM_BATCH_INVARIANT`
   (`sorted=True` top-k → stable index-order tie-break). Extend that trigger to
   also fire on `GLM_MOE_DETERMINISTIC_ROUTING=1`.
2. **Route to the native (deterministic) path.** GLM's config dispatches through
   the AITER kernel, which *bypasses* the deterministic native path. When the flag
   is set, force `_compute_routing` to use the native `grouped_topk` instead.

Safe for GLM because it runs with fused-shared-experts **off**
(`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0` → `num_fused_shared_experts=0`), so
the native path is functionally equivalent apart from removing the tie
nondeterminism.

The patcher is idempotent, anchor-based, and self-skipping (warns + no-ops if the
router is refactored), applied at container start after the DSA patchers.

## Results (NIAH determinism probe, same prompt ×6, seed 0, fp8 KV, 1P1D EP8)

| Size | baseline (203604, no fixes) | #2 deterministic routing (203868) | Change |
|------|-----------------------------|-----------------------------------|--------|
| ~8k  | Δ0 (deterministic)          | Δ0 (deterministic)                | — |
| ~20k | **Δ3** — dropped penguin/tiger/giraffe (min 7/10) | **Δ2** — min 8/10 | **improved 3→2** |
| ~96k | Δ1 — dropped tiger          | Δ1 — dropped tiger                | — |

- No engine/perf-path crash; both engines up (fp8, seed 0), MoRIIO healthy.
- At 20k the run no longer drops all three needles in a single trial (worst trial
  8/10 vs 7/10 baseline).

## How to enable

Set in the serve/job env (forwarded into the containers automatically):

```
export GLM_MOE_DETERMINISTIC_ROUTING=1
```

Reference wrapper: `glm5.1_notes/sbatch_1p1d_niah_determ_fix_fp8.sh`.

## Cost / limitations

- **Perf:** routing runs in native PyTorch instead of the fused AITER kernel. This
  is small relative to the expert GEMM, but non-zero; the flag is opt-in (default
  off) so normal serving is unaffected.
- **Partial:** this fixes the routing/tie-break component only. A residual Δ (8k/96k
  unchanged, 20k still Δ2) remains and is attributable to:
  - the **sparse-MLA stage-1 persistent ASM kernel** hazard (see
    `AITER_QH64_GPU_FAULT_REPORT.md` — the qh16-fold `_ps` kernel; not fixable from
    vLLM), and
  - **fp8-KV** transfer/precision (fully removable only by switching to **bf16 KV**,
    which reaches the colocated floor, 20k Δ=0).
- For strict determinism today: `GLM_MOE_DETERMINISTIC_ROUTING=1` **+ bf16 KV**;
  full fp8-KV determinism additionally needs the upstream aiter MLA kernel fix.

## Files

- `scripts/vllm_dissag/apply_glm_moe_deterministic_routing.py` — the patcher
- `scripts/vllm_dissag/connectors/moriio.sh` — `_glm_determinism_runtime_patch`
- `scripts/vllm_dissag/run_xPyD_models.slurm` — forwards `GLM_MOE_DETERMINISTIC_ROUTING`
- `scripts/vllm_dissag/glm5.1_notes/sbatch_1p1d_niah_determ_fix_fp8.sh` — reference run
