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

## Performance (throughput sweep)

Long-context throughput sweep, `/v1/completions`, `ignore_eos`, per-shape warmup=2. Jobs: **206036** (1P1D EP8), **206047** (2P2D EP16). Both jobs were still running the 96k/32k pass at capture time (see note); the main-shape grid is complete.

### 1P1D EP8 (job 206036)

| ISL/OSL | Concurrency | Req/s | Output tok/s | **Total tok/s** | Median TTFT (ms) | Median TPOT (ms) |
|---|---|---|---|---|---|---|
| 8000/1000 | 8 | 0.08 | 81.2 | **730.9** | 3887 | 92.9 |
| 8000/1000 | 32 | 0.31 | 314.5 | **2830.8** | 4776 | 93.8 |
| 8000/1000 | 64 | 0.60 | 601.9 | **5417.0** | 6429 | 93.8 |
| 8000/1000 | 128 | 1.12 | 1116.7 | **10050.6** | 5975 | 94.3 |
| 8000/1000 | 256 | 1.80 | 1796.6 | **16169.5** | 22490 | 94.9 |
| 4000/4000 | 8 | 0.02 | 86.4 | **172.7** | 2020 | 92.2 |
| 4000/4000 | 32 | 0.08 | 339.3 | **678.5** | 4742 | 92.7 |
| 4000/4000 | 64 | 0.17 | 671.3 | **1342.6** | 4744 | 93.1 |
| 4000/4000 | 128 | 0.33 | 1309.8 | **2619.6** | 6656 | 94.4 |
| 4000/4000 | 256 | 0.64 | 2557.5 | **5114.9** | 8867 | 94.8 |
| 8000/4000 | 8 | 0.02 | 84.6 | **253.8** | 4153 | 93.4 |
| 8000/4000 | 32 | 0.08 | 331.0 | **993.0** | 4180 | 94.6 |
| 8000/4000 | 64 | 0.16 | 654.1 | **1962.3** | 4371 | 94.6 |
| 8000/4000 | 128 | 0.32 | 1277.1 | **3831.3** | 6276 | 95.2 |
| 8000/4000 | 256 | 0.61 | 2459.3 | **7377.8** | 6407 | 94.8 |
| 96000/32000 | 8 | 0.004 | 84.5 | **338.2** | 81174 | 92.0 |

### 2P2D EP16 (job 206047)

| ISL/OSL | Concurrency | Req/s | Output tok/s | **Total tok/s** | Median TTFT (ms) | Median TPOT (ms) |
|---|---|---|---|---|---|---|
| 8000/1000 | 8 | 0.08 | 79.4 | **714.1** | 3848 | 96.3 |
| 8000/1000 | 32 | 0.30 | 298.0 | **2682.4** | 5283 | 97.5 |
| 8000/1000 | 64 | 0.57 | 573.4 | **5160.9** | 6603 | 97.1 |
| 8000/1000 | 128 | 1.10 | 1101.1 | **9909.5** | 6164 | 97.3 |
| 8000/1000 | 256 | 1.87 | 1872.6 | **16853.2** | 12838 | 97.6 |
| 4000/4000 | 8 | 0.02 | 83.0 | **166.1** | 2175 | 95.7 |
| 4000/4000 | 32 | 0.08 | 325.4 | **650.8** | 4066 | 97.0 |
| 4000/4000 | 64 | 0.16 | 640.6 | **1281.2** | 4311 | 97.3 |
| 4000/4000 | 128 | 0.32 | 1261.0 | **2522.0** | 5102 | 98.8 |
| 4000/4000 | 256 | 0.62 | 2480.3 | **4960.5** | 8106 | 98.3 |
| 8000/4000 | 8 | 0.02 | 81.5 | **244.5** | 3925 | 96.9 |
| 8000/4000 | 32 | 0.08 | 321.4 | **964.2** | 6005 | 96.9 |
| 8000/4000 | 64 | 0.16 | 637.0 | **1911.0** | 5130 | 97.5 |
| 8000/4000 | 128 | 0.31 | 1233.2 | **3699.7** | 6848 | 98.5 |
| 8000/4000 | 256 | 0.59 | 2379.7 | **7139.1** | 7371 | 98.4 |
| 96000/32000 | 8/32/64/128 | — | — | — | — | — |

> Note: at capture time both jobs were still on the 96k/32k pass — 1P1D has con=8 (338 tok/s, TTFT ~81 s); 2P2D 96k had not yet produced a result. The 96k rows will be filled in when the pass completes.

Peak total throughput observed: **~16.2k tok/s (1P1D)** and **~16.9k tok/s (2P2D)** at 8000/1000 @ con=256. At matched shapes/concurrency the two topologies are within a few percent on throughput; 2P2D shows slightly higher TPOT (~97–99 ms vs ~93–95 ms) reflecting the cross-node EP16 decode.

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
