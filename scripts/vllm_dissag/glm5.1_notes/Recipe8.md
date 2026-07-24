# Recipe 8: GLM-5.1-FP8 on AMD MI300X — WideEP PD-Disaggregated (vLLM v0.25.1 + PR#47766, from-scratch build)

> **STATUS: WORKING.** GLM-5.1-FP8 serves end-to-end in WideEP / PD-disaggregated mode on MI300X (gfx942), validated on **both 1P1D EP8 and 2P2D EP16** across long context (2k–96k). Built **entirely from source** (no prebuilt vLLM base) on **vLLM v0.25.1 + PR#47766**, with **AITER v0.1.18** and **MoRI 42e895472**, unblocking the 2P2D EP16 case that Recipe 7 could not run. Residual: mild ±1–2 NIAH needle nondeterminism (qh16 split-K); no crashes, no GPU faults.

Confluence: https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1814116072

## What this is

GLM-5.1-FP8 (`zai-org`, `GlmMoeDsaForCausalLM`: MLA + DSA sparse attention, 256 experts top-8) served on **AMD MI300X** (gfx942) under **prefill/decode-disaggregated WideEP**, using **AMD MoRI** (MoRI-EP dispatch/combine + MoRI-IO RDMA KV transfer over RoCEv2).

The distinguishing feature is the **fully self-contained, from-scratch build**: a single multi-stage Dockerfile builds the entire pinned stack from the public ROCm OS image — PyTorch, Triton, FlashAttention, AITER v0.1.18, MoRI, and **vLLM v0.25.1 from source** — then applies PR#47766 and the two GLM disagg patches. Nothing is inherited from a prebuilt `vllm-openai-rocm` image.

WideEP disagg runs **DP=8 / TP=1**, so each rank holds all 64 attention heads → **`gqa_ratio=64`** (opposite end from the TP=8 `gqa=8` regime targeted by vLLM#36855 / aiter#2821).

## Build history (from-scratch, pinned)

| Component | Version / ref | From source |
|---|---|---|
| Base OS | `rocm/dev-ubuntu-22.04:7.2.3-complete` (ROCm 7.2.3) | pulled |
| Python | 3.12 | — |
| PyTorch | 2.11.0 ROCm fork @ `d0c8b1f3` (+ vision v0.24.1, audio v2.9.0) | ✅ |
| Triton | ROCm triton @ `0f380657` (+ cherry-pick 555d04f / triton#8991) | ✅ |
| FlashAttention | @ `0e60e394` | ✅ |
| AITER | **v0.1.18** (`d6de776…`) + flydsl 0.2.4 | ✅ |
| MoRI (`amd_mori`) | **1.1.2.dev43+g42e895472** (ROCm/mori @ 42e895472b08) | ✅ |
| vLLM | **0.25.1** from source + **PR#47766** | ✅ |
| vllm-router | raviguptaamd/router `ravgupta/discovery-dp-rank-roundrobin` | ✅ (cargo) |

Image: `glm5.1-fp8-disagg:mi300x-fromscratch`.
- Self-contained: `docker/GLM5.1-FP8.disagg.MI300X.from-scratch.Dockerfile`
- Overlay-on-prebuilt-base variant: `docker/GLM5.1-FP8.disagg.MI300X.share.Dockerfile`

## Accuracy & stability fixes

| Fix | Source | What it fixes | Status |
|---|---|---|---|
| vLLM #47766 — 6-field sparse-MLA metadata key | `docker/patches/patch_pr47766_v024.py` | keeps persistent sparse-MLA correct across chunked-prefill continuations (aiter#4076); eliminates long-context collapse | ✅ |
| Patch A — aiter qh16 fold | `docker/patches/patch_aiter_mla_qh64_fold.py` | native qh64 fp8 decode kernel (#3188) GPU-faults on gfx942 at page_size=1; fold gqa64 fp8 → qh16 | ✅ (aiter #4363 / PR #4365) |
| Patch B — force persistent MLA | `docker/patches/patch_glm_dsa_force_persistent.py` | gqa64 fp8 has no non-persistent kernel; force `use_persistent=True` | ✅ (vLLM #49649) |
| scheduler KV-xfer stale-req guard | `docker/patches/patch_glm_sched_kv_xfer_stale_guard.py` | stale MoRIIO KV-finished req_id must not assert-crash prefill | ✅ |
| DSA indexer boot-warmup | `scripts/vllm_dissag/apply_glm_dsa_indexer_warmup_fix.py` | precompile indexer Triton kernels at boot | ✅ |
| rocm.py GCN-arch circular-import fix | inline (Dockerfile) | concurrent EP-worker boot crash | ✅ |
| MoRIIO/WideEP runtime deps | inline (Dockerfile) | `msgpack quart aiohttp pyzmq blinker` (MoRIIO connector import-time deps) | ✅ |

## Serve flags

```
--trust-remote-code --block-size 1 --kv-cache-dtype fp8 \
--no-enable-prefix-caching --gpu-memory-utilization 0.9
# WideEP disagg: VLLM_ALL2ALL_BACKEND=mori, MoRIIO KV connector, vllm-router (DP-rank round-robin)
# AITER MLA on (VLLM_ROCM_USE_AITER=1); DSA requires --block-size 1, VLLM_ROCM_USE_AITER_MLA=1
# Disagg: prefill DP=8 TP=1 + decode DP=8 TP=1 (EP8 per role; EP16 for 2P2D)
```

## Long-context accuracy (NIAH) — found/10 across 3 trials

Jobs 205803 (1P1D EP8), 205834 (2P2D EP16). Image: vLLM 0.25.1 + PR#47766 + aiter 0.1.18 + qh16-fold + force-persistent + MoRI 42e895472.

| Context | 1P1D EP8 (T1/T2/T3) | 2P2D EP16 (T1/T2/T3) |
|---|---|---|
| 2k  | 10 / 10 / 9  | 10 / 10 / 9 |
| 8k  | 10 / 10 / 10 | 10 / 10 / 10 |
| 20k | 9 / 9 / 8    | 8 / 10 / 8 |
| 35k | 9 / 9 / 10   | 10 / 10 / 10 |
| 96k | 10 / 10 / 10 | 10 / 9 / 10 |

Extremes (8k, 96k) effectively deterministic; residual ±1–2 variance at mid-range (20k–35k) = qh16 split-K nondeterminism (Patch A trade-off), not a crash/correctness issue.

## Accuracy suite (`glm51_suite` pre_release)

Jobs 206023 (1P1D EP8), 206022 (2P2D EP16).

| Benchmark | Metric | 1P1D EP8 | 2P2D EP16 | Status |
|---|---|---|---|---|
| niah_single_2 | retrieval_success | 1.0 | 0.98 | pass |
| aa_lcr_mini | accuracy | 0.40 | 0.50 | pass |
| livecodebench_mini | pass@1 | — | — | hard-fail (harness subprocess exit 1) |
| mmlu_pro_50 / aime_2025_mini / gsm8k_100 / gpqa_diamond_mini | accuracy | — | — | skipped (fast-fail gate) |

Disagg served cleanly; niah + aa_lcr pass on both topologies; the suite fast-fails at `livecodebench_mini` (LCB harness subprocess error, root-cause pending) which gates the rest.

## Deployment configurations

| Topology | EP width | Status |
|---|---|---|
| 1P1D disaggregated | EP8 | ✅ NIAH validated; niah/aa_lcr pass |
| 2P2D disaggregated | EP16 | ✅ NIAH validated; niah/aa_lcr pass (unblocked vs Recipe 7) |
| Single-node TP=8 (colocated) | — | ✅ correctness/determinism control |

## Upstream issues / PRs filed

| Item | Repo | What |
|---|---|---|
| aiter #4363 (issue) + #4365 (PR, Patch A) | ROCm/aiter | qh64 fp8 decode GPU-fault at page_size=1 on gfx942; gate native-qh64 to page_size==64 + repro `op_tests/test_mla_qh64_gfx942_pagesize1.py` |
| vLLM #49649 (issue) | vllm-project/vllm | persistent-kernel gate unsafe for gqa_ratio=64 fp8 |
| aiter #4364 (issue) | ROCm/aiter | qh16 fp8 sparse decode run-to-run nondeterminism (residual ±1–2 needle) |

## Environment knobs

| Var | Value | Purpose |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | 1 | AITER kernels (DSA indexer) |
| `VLLM_ROCM_USE_AITER_MLA` | 1 | AITER MLA path |
| `VLLM_GCN_ARCH` | gfx942 | rocm.py boot fix |
| `--block-size` | 1 | DSA sparse indexer |
| `--kv-cache-dtype` | fp8 | KV precision |
| `AITER_BATON_TIMEOUT` | 1800 | cold-start JIT baton headroom |
| `LOG_WAIT_TIMEOUT_SECONDS` | 9000 | long cold bring-up |
| `VLLM_MORIIO_DEFERRED_TIMEOUT_S` / `VLLM_MORIIO_TRANSFER_TIMEOUT_S` | 600 | MoRIIO KV-write headroom |

## Reproducibility

- **Branch:** `mdeopuja/glm5.1-fp8-recipe8-fromscratch` on ROCm/MAD-private.
- **Dockerfiles:** `docker/GLM5.1-FP8.disagg.MI300X.from-scratch.Dockerfile` (+ `.CHANGELOG.md`), `docker/GLM5.1-FP8.disagg.MI300X.share.Dockerfile`.
- **Patches:** `docker/patches/patch_pr47766_v024.py`, `patch_aiter_mla_qh64_fold.py` (A), `patch_glm_dsa_force_persistent.py` (B), `patch_glm_sched_kv_xfer_stale_guard.py`, `patch_aiter_gpuless_import.py`, `rocminfo_buildshim.sh`, `patch_aiter_baton_selfheal_v2.py`; `scripts/vllm_dissag/apply_glm_dsa_indexer_warmup_fix.py`.
- **AITER repro (for #4365):** `docker/patches/repro_aiter_mla_qh64_gfx942_fault.py`, `run_repro_qh64_fault.sh`.
- **Eval/perf orchestration:** `scripts/vllm_dissag/glm5.1_notes/sbatch_{1p1d,2p2d}_evalsuite_fromscratch.sh`, `run_accsuite_disagg_in_container.sh`, `sbatch_{1p1d,2p2d}_perf_fromscratch.sh`.

## Relationship to other recipes

- **Recipe 7** (vLLM v0.24.0 + PR#47766, minimal in-place patch on the stock image): parent recipe; 2P2D EP16 was blocked (MoRI wide-EP device assert), focus on the determinism study. Recipe 8 rebases to **v0.25.1 from source** with **AITER v0.1.18 + MoRI 42e895472** + qh16-fold + force-persistent, which **unblocks 2P2D EP16** and serves both topologies.
- **Recipe 4** (GLM-5.1-FP8 TP=8, single node): colocated correctness/determinism control.
