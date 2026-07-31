# Recipe 8: GLM-5.1-FP8 on AMD MI300X — WideEP PD-Disaggregated (vLLM v0.25.1 + PR#47766, from-scratch build)

> **STATUS: WORKING.** GLM-5.1-FP8 serves end-to-end in WideEP / PD-disaggregated mode on MI300X (gfx942), validated on **both 1P1D EP8 and 2P2D EP16** across long context (2k–96k). Built **entirely from source** (no prebuilt vLLM base) on **vLLM v0.25.1 + PR#47766**, with **AITER v0.1.18** and **MoRI 42e895472**, unblocking the 2P2D EP16 case that Recipe 7 could not run. Residual: mild ±1–2 NIAH needle nondeterminism (qh16 split-K); no crashes, no GPU faults.

Confluence: https://amd.atlassian.net/wiki/spaces/DCGPUAIST/pages/1814116072

## Caveats

- Hard requirement: vLLM pinned to v0.25.1 and AITER v0.1.18. Not tested with previous versions.
- Prefill and Decode nodes must be on the same networking switch for RDMA communication to work.

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

## Salient features

- **WideEP PD-disaggregated** serving: prefill and decode on separate nodes, KV moved over RDMA via MoRI-IO; experts dispatched/combined via MoRI-EP all2all.
- **DSA sparse attention** (block-size 1 indexer) + **MLA**, fp8 Q / fp8 KV, `block_size=1`.
- **Fully from-scratch, reproducible build** — one Dockerfile, all pins from source; auditable end to end.
- **Both 1P1D EP8 and 2P2D EP16 run** (2P2D was blocked in Recipe 7).
- **Cold-start resilience**: aiter JIT baton self-heal (native in v0.1.18) + node-local JIT cache; GPU-less-build import shims (rocminfo shim, arch_info fallback) that no-op on real GPUs.

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

From-scratch image. Initial full-suite run: jobs 206023 (1P1D EP8), 206022 (2P2D EP16). The suite runs fast-fail (`stop_on_hard_fail`), so `livecodebench_mini` hard-failing initially skipped the remaining P1/P2 benchmarks; those were completed in dedicated post-LCB re-runs (LCB skipped) on the same image.

| Benchmark | Metric | 1P1D EP8 | 2P2D EP16 | Status |
|---|---|---|---|---|
| niah_single_2 | retrieval_success | 1.0 | 0.98 | pass |
| aa_lcr_mini | accuracy | 0.40 | 0.50 | pass |
| mmlu_pro_50 | accuracy | 0.821 | 0.797 | pass |
| gsm8k_100 | accuracy | 0.940 | 0.960 | pass |
| gpqa_diamond_mini | accuracy | 0.12 | 0.10 | warn (below 0.45 preferred) |
| aime_2025_mini | exact_match | — | — | not completed (harness / duration limit) |
| livecodebench_mini | pass@1 | — | — | hard-fail (harness metrics-assertion crash) |

Disagg served cleanly; **mmlu_pro (0.82/0.80) and gsm8k (0.94/0.96) pass on both topologies**, plus niah and aa_lcr. The three P1/P2 items that needed follow-up:

- **livecodebench_mini** — hard-fails inside the LiveCodeBench harness (not a serving fault): the code-execution scoring pass crashes with `AssertionError` in `compute_code_generation_metrics.py` and writes no `*_eval.json`. NIAH + AA-LCR passing confirm the image itself is correct.
- **gpqa_diamond_mini** — the suite ships no local GPQA data (official `Idavidrein/gpqa` is gated); items staged from a public mirror scored 0.12/0.10, below the 0.45 preferred threshold. The sub-random result is most likely a scoring artifact (`benchmark_max_tokens=8` truncates a reasoning model before it emits a clear answer letter) rather than a true capability floor — pending audit.
- **aime_2025_mini** — not completed. lm-eval drives AIME over raw `/v1/completions`, which bypasses the chat template's `enable_thinking:false`, so GLM-5.1 emits full ~32k-token reasoning per problem (~35 min/sample). The reliable lm-eval config is `num_concurrent=1` (the async aiohttp path crashes with "Session/Connector is closed" on long generations), so 30 samples run serially (~17.6 h) and exceed the 10 h serving-hold window.

## Single-node TP=8 accuracy vs Recipe 4

Run on the **same from-scratch image** in single-node colocated `vllm serve --tensor-parallel-size 8` mode (no disagg / MoRI / router), thinking-ON. This isolates model/kernel accuracy from the disaggregation transport — the disagg suite above hit MoRIIO KV-transfer degradation late in long runs (dropped decode requests), whereas single-node served cleanly end-to-end. Jobs: **209460** (full 7-benchmark suite) and **209764** (aime + gpqa re-run with the Recipe-4-matched serve flags `--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --chat-template-content-format string` and restored reasoning budgets: aime `max_gen_toks=32768`, gpqa `max_tokens=16384`). LiveCodeBench ran without the harness metrics-assertion crash on this run.

| Benchmark | Metric | Recipe 8 (single-node TP=8) | Recipe 4 (single-node TP=8) | Note |
|---|---|---|---|---|
| niah_single_2 | retrieval_success | 1.000 | 1.000 | match |
| aa_lcr_mini | accuracy | 1.000 | 1.000 | match |
| gsm8k_100 | accuracy | 0.960 | — | R8 only |
| mmlu_pro_50 | accuracy | 0.824 | — | R8 only |
| livecodebench_mini | pass@1 | **0.652** | 0.440 | R8 > R4 |
| gpqa_diamond_mini | accuracy | 0.380 | 0.700 | below R4 (genuine gap) |
| aime_2025_mini | exact_match | 0.167 | 0.367 | below R4 (truncation-limited) |

- **NIAH and AA-LCR match** Recipe 4 (both 1.000); **LiveCodeBench is higher** (0.652 vs 0.440); MMLU-Pro (0.824) and GSM8K (0.960) add coverage Recipe 4 did not report.
- **aime_2025_mini (0.167)** is **truncation-limited**: even at the 32768 generation cap, ~24/30 responses run to the cap mid-reasoning without emitting a `\boxed` answer (only 6/30 boxed — the rise from 2/30 at the earlier 16384 cap confirms the reasoning parser is working). Escaping it needs a larger generation cap (>32768).
- **gpqa_diamond_mini (0.380)** is **not** truncation-limited (~6.8k tokens/sample avg, well under 16384) — a genuine accuracy gap vs Recipe 4's 0.700, most likely a build/numerics difference (vLLM 0.25.1 + qh16-fold / force-persistent + AITER 0.1.18 vs Recipe 4's stock vLLM 0.24.0 + AITER 0.1.13.post1). Pending investigation.

## Performance (throughput sweep)

Long-context throughput sweep, `/v1/completions`, `ignore_eos`, per-shape warmup=2. Jobs: **206036** (1P1D EP8), **206047** (2P2D EP16); extended by **206294** (1P1D con=512 †), **206761** (2P2D 32k/8k ‡), and gap-fills **207569**/**207885** (1P1D 32k ♦◊) and **207570** (2P2D ♠). The main-shape grid is complete; the 96k/32k pass is partial (see note). GPU counts: **1P1D = 16 GPUs** (2 nodes × 8), **2P2D = 32 GPUs** (4 × 8); `Total tok/s/GPU` = Total tok/s ÷ GPU count.

### 1P1D EP8 (job 206036) — 16 GPUs

| ISL/OSL | Concurrency | Req/s | Output tok/s | **Total tok/s** | **Total tok/s/GPU** | Median TTFT (ms) | Median TPOT (ms) |
|---|---|---|---|---|---|---|---|
| 8000/1000 | 8 | 0.08 | 81.2 | **730.9** | 45.7 | 3887 | 92.9 |
| 8000/1000 | 32 | 0.31 | 314.5 | **2830.8** | 176.9 | 4776 | 93.8 |
| 8000/1000 | 64 | 0.60 | 601.9 | **5417.0** | 338.6 | 6429 | 93.8 |
| 8000/1000 | 128 | 1.12 | 1116.7 | **10050.6** | 628.2 | 5975 | 94.3 |
| 8000/1000 | 256 | 1.80 | 1796.6 | **16169.5** | 1010.6 | 22490 | 94.9 |
| 8000/1000 | 512 † | 1.98 | 1983.6 | **17852.8** | 1115.8 | 140330 | 94.2 |
| 4000/4000 | 8 | 0.02 | 86.4 | **172.7** | 10.8 | 2020 | 92.2 |
| 4000/4000 | 32 | 0.08 | 339.3 | **678.5** | 42.4 | 4742 | 92.7 |
| 4000/4000 | 64 | 0.17 | 671.3 | **1342.6** | 83.9 | 4744 | 93.1 |
| 4000/4000 | 128 | 0.33 | 1309.8 | **2619.6** | 163.7 | 6656 | 94.4 |
| 4000/4000 | 256 | 0.64 | 2557.5 | **5114.9** | 319.7 | 8867 | 94.8 |
| 4000/4000 | 512 † | 1.23 | 4930.7 | **9861.3** | 616.3 | 10193 | 94.9 |
| 8000/4000 | 8 | 0.02 | 84.6 | **253.8** | 15.9 | 4153 | 93.4 |
| 8000/4000 | 32 | 0.08 | 331.0 | **993.0** | 62.1 | 4180 | 94.6 |
| 8000/4000 | 64 | 0.16 | 654.1 | **1962.3** | 122.6 | 4371 | 94.6 |
| 8000/4000 | 128 | 0.32 | 1277.1 | **3831.3** | 239.5 | 6276 | 95.2 |
| 8000/4000 | 256 | 0.61 | 2459.3 | **7377.8** | 461.1 | 6407 | 94.8 |
| 32000/2000 | 8 ♦ | 0.04 | 78.3 | **1330.9** | 83.2 | 19662 | 91.9 |
| 32000/2000 | 32 ♦ | 0.15 | 290.6 | **4940.0** | 308.8 | 22694 | 92.7 |
| 32000/2000 | 64 ♦ | 0.27 | 531.8 | **9040.8** | 565.1 | 23140 | 94.2 |
| 32000/2000 | 128 § | 0.40 | 801.8 | **13629.9** | 851.9 | 85490 | 93.5 |
| 32000/8000 | 8 ◊ | 0.01 | 83.9 | **419.3** | 26.2 | 19874 | 92.9 |
| 32000/8000 | 32 ◊ | 0.04 | 329.4 | **1646.9** | 102.9 | 21971 | 92.9 |
| 32000/8000 | 64 ◊ | 0.08 | 640.0 | **3200.0** | 200.0 | 23250 | 93.5 |
| 32000/8000 | 128 § | 0.15 | 1217.9 | **6089.6** | 380.6 | 23728 | 94.4 |
| 96000/32000 | 8 | 0.004 | 84.5 | **338.2** | 21.1 | 81174 | 92.0 |
| 96000/32000 | 32 | 0.01 | 328.6 | **1314.3** | 82.1 | 91338 | 92.7 |

§ 32000/2000 and 32000/8000 @ con=128 from job **206985** (dedicated con=128 sweep, same image); 512/512 successful. The matching 2P2D 32k/2k con=128 point is captured in the 2P2D table (‖, job 206760 = 14562.5 tok/s); the 2P2D 32k/8k ladder is now complete (see ♠).

♦ 1P1D 32000/2000 con 8/32/64 from job **207569** (gap-fill, same image); 100% successful. con=256 wedged on 1P1D (MoRIIO deadlock ceiling — 2P2D runs it fine), so it is omitted.

◊ 1P1D 32000/8000 con 8/32/64 from job **207885** (gap-fill #2, same image); 100% successful. con=256 is omitted (same 1P1D MoRIIO deadlock ceiling as 32k/2k). The 8000/1000 @ con=1024 cell is **not attainable on 1P1D**: it tail-deadlocks reproducibly at ~95–96% completion (the last ~5% of decode KV transfers never drain, hanging >2 h with zero progress). This is a topology capacity ceiling, not a timeout-tunable stall (raising the MoRIIO defer timeout does not help — writes already sit far longer than any timeout and never complete); use 2P2D+ for 8k/1k at con≥512 (see ¶).

† con=512 from job **206294** (dedicated high-concurrency run, same image). At 8000/4000 @ con=512 the run hit a **deterministic MoRIIO KV-transfer saturation ceiling** (`Deferred write task … expired after 600 s`), so no data point — con=512 is beyond sustainable concurrency for the heaviest sub-96k shape on 1P1D EP8.

### 2P2D EP16 (job 206047) — 32 GPUs

| ISL/OSL | Concurrency | Req/s | Output tok/s | **Total tok/s** | **Total tok/s/GPU** | Median TTFT (ms) | Median TPOT (ms) |
|---|---|---|---|---|---|---|---|
| 8000/1000 | 8 | 0.08 | 79.4 | **714.1** | 22.3 | 3848 | 96.3 |
| 8000/1000 | 32 | 0.30 | 298.0 | **2682.4** | 83.8 | 5283 | 97.5 |
| 8000/1000 | 64 | 0.57 | 573.4 | **5160.9** | 161.3 | 6603 | 97.1 |
| 8000/1000 | 128 | 1.10 | 1101.1 | **9909.5** | 309.7 | 6164 | 97.3 |
| 8000/1000 | 256 | 1.87 | 1872.6 | **16853.2** | 526.7 | 12838 | 97.6 |
| 8000/1000 | 512 ¶ | 2.06 | 2056.5 | **18508.7** | 578.4 | 126682 | 97.7 |
| 8000/1000 | 1024 ¶ | 2.17 | 2166.6 | **19499.2** | 609.3 | 349346 | 97.5 |
| 4000/4000 | 8 | 0.02 | 83.0 | **166.1** | 5.2 | 2175 | 95.7 |
| 4000/4000 | 32 | 0.08 | 325.4 | **650.8** | 20.3 | 4066 | 97.0 |
| 4000/4000 | 64 | 0.16 | 640.6 | **1281.2** | 40.0 | 4311 | 97.3 |
| 4000/4000 | 128 | 0.32 | 1261.0 | **2522.0** | 78.8 | 5102 | 98.8 |
| 4000/4000 | 256 | 0.62 | 2480.3 | **4960.5** | 155.0 | 8106 | 98.3 |
| 4000/4000 | 512 ♠ | 1.15 | 4612.7 | **9225.3** | 288.3 | 8891 | 102.6 |
| 8000/4000 | 8 | 0.02 | 81.5 | **244.5** | 7.6 | 3925 | 96.9 |
| 8000/4000 | 32 | 0.08 | 321.4 | **964.2** | 30.1 | 6005 | 96.9 |
| 8000/4000 | 64 | 0.16 | 637.0 | **1911.0** | 59.7 | 5130 | 97.5 |
| 8000/4000 | 128 | 0.31 | 1233.2 | **3699.7** | 115.6 | 6848 | 98.5 |
| 8000/4000 | 256 | 0.59 | 2379.7 | **7139.1** | 223.1 | 7371 | 98.4 |
| 8000/4000 | 512 ♠ | 1.05 | 4216.6 | **12649.7** | 395.3 | 6305 | 105.9 |
| 32000/2000 | 8 ‖ | 0.04 | 75.9 | **1290.1** | 40.3 | 15224 | 96.5 |
| 32000/2000 | 32 ‖ | 0.14 | 280.8 | **4774.0** | 149.2 | 22049 | 97.4 |
| 32000/2000 | 64 ‖ | 0.26 | 527.1 | **8960.9** | 280.0 | 21254 | 97.3 |
| 32000/2000 | 128 ‖ | 0.43 | 856.6 | **14562.5** | 455.1 | 56367 | 97.1 |
| 32000/2000 | 256 ‖ | 0.47 | 938.7 | **15958.4** | 498.7 | 301612 | 97.2 |
| 32000/8000 | 8 ‡ | 0.01 | 81.4 | **407.0** | 12.7 | 18542 | 95.6 |
| 32000/8000 | 32 ♠ | 0.04 | 320.4 | **1602.2** | 50.1 | 22097 | 95.6 |
| 32000/8000 | 64 ♠ | 0.08 | 629.0 | **3145.1** | 98.3 | 21663 | 95.9 |
| 32000/8000 | 128 ♠ | 0.15 | 1203.5 | **6017.4** | 188.0 | 21637 | 96.5 |
| 32000/8000 | 256 ♠ | 0.28 | 2227.4 | **11137.1** | 348.0 | 23902 | 97.0 |
| 96000/32000 | 8 | 0.004 | 82.0 | **327.8** | 10.2 | 76731 | 95.1 |
| 96000/32000 | 32 | 0.01 | 319.4 | **1277.5** | 39.9 | 85310 | 95.8 |
| 96000/32000 | 64 | 0.02 | 625.3 | **2501.0** | 78.2 | 87787 | 95.8 |

‖ 32000/2000 full ladder from job **206760** (dedicated run, same image); con 8/32/64/128/256 all 100% successful. con=512 hit the MoRIIO KV-transfer saturation ceiling (1988/2048, TTFT ~11.6 min) so it is omitted. This supplies the 2P2D 32k/2k con=128 point (14562.5 tok/s).

‡ 32000/8000 con=8 from job **206761** (dedicated run, same image); the rest of the ladder (con 32/64/128/256) was later filled by job **207570** (see ♠).

¶ 8000/1000 con=512/1024 from job **206759** (dedicated high-concurrency run, same image); both 100% successful. con=1024 is the observed **2P2D peak (~19.5k tok/s)**; TTFT grows steeply (349 s median at con=1024) as the offered load exceeds steady-state capacity.

♠ 2P2D gap-fill from job **207570**: 4000/4000 & 8000/4000 @ con=512 (100% successful) and 32000/8000 con 32/64/128/256 (32k/8k@256 shed 1/1024, the rest 100%).

> Note: the 96k/32k pass is **partial** — both jobs hit the 24 h wall (TIMEOUT). Completed: 1P1D con 8/32; 2P2D con 8/32/64. Higher concurrencies (1P1D 64/128, 2P2D 128) did not finish at these very long shapes (32k-token outputs at 96k context → TTFT ~77–91 s, so each concurrency level takes hours). The 96k/32k @ con=128 point is captured separately on 2P4D/4P4D (below), since it deadlocks on 1P1D/2P2D at the default timeout.

### 2P4D EP32 (jobs 208491 + 207083) — 48 GPUs

2P4D = 2 prefill + 4 decode (6 nodes, EP32). The peak-throughput sweep is job **208491** (⧫); the 96k/32k @ con=128 point is job **207083** (partial-shed stress run). GPU count = **48**, so `Total tok/s/GPU` = Total ÷ 48.

| ISL/OSL | Concurrency | Req/s | Output tok/s | **Total tok/s** | **Total tok/s/GPU** | Median TTFT (ms) | Median TPOT (ms) | Successful |
|---|---|---|---|---|---|---|---|---|
| 8000/1000 | 8 ⧫ | 0.08 | 77.0 | **693.1** | 14.4 | 3761 | 98.8 | 32/32 |
| 8000/1000 | 32 ⧫ | 0.30 | 296.8 | **2670.7** | 55.6 | 5965 | 99.3 | 128/128 |
| 8000/1000 | 64 ⧫ | 0.56 | 555.4 | **4999.0** | 104.1 | 5278 | 103.4 | 256/256 |
| 8000/1000 | 128 ⧫ | 0.99 | 993.2 | **8938.8** | 186.2 | 6743 | 108.6 | 512/512 |
| 8000/1000 | 256 ⧫ | 1.65 | 1648.0 | **14832.0** | 309.0 | 11818 | 118.2 | 1024/1024 |
| 8000/1000 | 512 ⧫ | 2.00 | 2003.3 | **18029.6** | 375.6 | 88507 | 138.1 | 2048/2048 |
| 8000/1000 | 1024 ⧫ | 2.13 | 2130.6 | **19175.2** | 399.5 | 311544 | 138.4 | 4096/4096 |
| 4000/4000 | 8 ⧫ | 0.02 | 80.3 | **160.6** | 3.3 | 2273 | 98.9 | 32/32 |
| 4000/4000 | 32 ⧫ | 0.08 | 317.6 | **635.1** | 13.2 | 5048 | 99.0 | 128/128 |
| 4000/4000 | 64 ⧫ | 0.15 | 597.7 | **1195.4** | 24.9 | 4852 | 105.5 | 256/256 |
| 4000/4000 | 256 ⧫ | 0.52 | 2081.0 | **4161.9** | 86.7 | 6740 | 118.0 | 1024/1024 |
| 4000/4000 | 512 ⧫ | 0.85 | 3417.4 | **6834.8** | 142.4 | 7236 | 142.4 | 2048/2048 |
| 8000/4000 | 8 ⧫ | 0.02 | 79.6 | **238.7** | 5.0 | 4170 | 99.1 | 32/32 |
| 8000/4000 | 32 ⧫ | 0.08 | 313.9 | **941.9** | 19.6 | 4532 | 99.9 | 128/128 |
| 8000/4000 | 64 ⧫ | 0.15 | 591.0 | **1772.9** | 36.9 | 6454 | 105.1 | 256/256 |
| 8000/4000 | 128 ⧫ | 0.27 | 1099.1 | **3297.2** | 68.7 | 6658 | 110.3 | 512/512 |
| 8000/4000 | 256 ⧫ | 0.50 | 2005.9 | **6017.8** | 125.4 | 6267 | 119.3 | 1024/1024 |
| 8000/4000 | 512 ⧫ | 0.81 | 3222.5 | **9667.4** | 201.4 | 8243 | 144.4 | 2048/2048 |
| 32000/2000 | 8 ⧫ | 0.04 | 73.6 | **1251.2** | 26.1 | 16854 | 99.3 | 32/32 |
| 32000/2000 | 128 ⧫ | 0.41 | 819.8 | **13937.2** | 290.4 | 25989 | 116.0 | 512/512 |
| 32000/2000 | 256 ⧫ | 0.45 | 906.9 | **15417.3** | 321.2 | 277177 | 115.5 | 1024/1024 |
| 32000/8000 | 128 ⧫ | 0.13 | 1024.4 | **5122.0** | 106.7 | 22865 | 115.1 | 512/512 |
| 32000/8000 | 256 ⧫ | 0.22 | 1728.5 | **8642.7** | 180.1 | 22204 | 131.1 | 1024/1024 |
| 96000/32000 | 128 | 0.01 | 177.5 | **712.0** | 14.8 | 79751 | 113.5 | 270/512 |

**2P4D peak = 19,175 tok/s at 8k/1k @ con=1024.** con=2048 did **not** climb further — it plateaued and tail-stalled at 97% (7923/8192), so throughput rolls off past con=1024.

⧫ Job **208491** peak-finding sweep (from-scratch image, 6 nodes). All listed cells 100% successful. Sweep still in progress — remaining cells (**32k/2k @ 32/64, 32k/8k @ 8/32/64, and 96k/32k @ 8/32/64**) will be appended when complete; the 96k/32k low-con fills are the long pole (32k-token outputs, hours per cell).

**Key finding — 8k/1k is prefill-bound on 2P4D.** The 2P4D 48-GPU peak (19,175 @ con=1024) is essentially tied with (~1.7% below) the **2P2D 32-GPU peak of 19,499** at the same 8k/1k@1024. Doubling decode (2D→4D) did **not** raise 8k/1k throughput and per-GPU efficiency fell sharply (2P2D ≈609 → 2P4D ≈399 tok/s/GPU), because both topologies use only **2 prefill nodes**. For short-ISL shapes the lever is more *prefill* (e.g. 4P2D/4P4D), not more decode.

Note on 96k/32k @ con=128 (job 207083): 270/512 succeeded (~47% load-shed). 96k/32k @ con=128 **deadlocks on 1P1D/2P2D at the default timeout** (decode KV cannot hold 128 concurrent 96k-context sequences → MoRIIO deferred writes expire, `remote blocks never arrived`); **2P4D adds decode KV capacity and clears the fatal deadlock**, though it still partially sheds at this extreme shape. This is a decode-KV-capacity ceiling, not a crash. (The relevant expiry knob is the connector's `defer_timeout` — set by `MORIIO_DEFER_TIMEOUT` — **not** `VLLM_MORIIO_DEFERRED_TIMEOUT_S`/`_TRANSFER_TIMEOUT_S`, which vLLM logs as *"Unknown vLLM environment variable"* and ignores. `MORIIO_DEFER_TIMEOUT` is now forwarded into the container via `connectors/moriio.env` — see the 96k/32k @ con=128 `defer_timeout=7200` runs below.)

### 1P2D EP16 (job 209216 peak; 209754 full-grid in progress) — 24 GPUs

1P2D = 1 prefill + 2 decode (3 nodes, EP16). Peak-throughput hunt on the small shape; GPU count = **24**, so `Total tok/s/GPU` = Total ÷ 24. **Full grid (other shapes) is a separate in-progress sweep (job 209754, con≤1024); only the 8k/1k peak ladder is available so far.**

| ISL/OSL | Concurrency | Req/s | Output tok/s | **Total tok/s** | **Total tok/s/GPU** | Median TTFT (ms) | Median TPOT (ms) |
|---|---|---|---|---|---|---|---|
| 8000/1000 | 128 | 1.06 | 1062.9 | **9565.6** | 398.6 | 6129 | 99.6 |
| 8000/1000 | 256 | 1.76 | 1760.3 | **15842.9** | 660.1 | 20699 | 98.7 |
| 8000/1000 | 512 | 1.92 | 1923.6 | **17312.7** | 721.4 | 142634 | 99.2 |
| 8000/1000 | 1024 | 2.02 | 2021.9 | **18197.3** | 758.2 | 382513 | 98.4 |

**1P2D peak = 18,197 tok/s at 8k/1k @ con=1024.** con=2048 is **not attainable** — it tail-deadlocks (hung at ~97%, 7959/8192, in both jobs 209216 and the dedicated retry 209572; no result). Adding a second decode node to 1P1D lifts the small-shape peak only ~2% (1P1D ~17.9k → 1P2D 18.2k): the 8k/1k peak is prefill/dispatch-bound (both use 1 prefill node), consistent with the 2P2D→2P4D finding below.

### 4P4D EP32 (job 208387) — 64 GPUs

4P4D = 4 prefill + 4 decode (8 nodes, EP32). Single point: **96k/32k @ con=128** with the high MoRIIO deferred-write timeout (`defer_timeout=7200`, confirmed forwarded into the container). GPU count = **64**.

| ISL/OSL | Concurrency | Req/s | Output tok/s | **Total tok/s** | **Total tok/s/GPU** | Median TTFT (ms) | Median TPOT (ms) | Successful |
|---|---|---|---|---|---|---|---|---|
| 96000/32000 | 128 | ~0.003 | 105.5 | **421.95** | 6.6 | 82029 | 112.4 | 222/512 |

**Read:** `defer_timeout=7200` let all 512 requests reach terminal state (vs the default-timeout hard-stall at 95/512 on the earlier 4P4D attempt 207165), but the run still **shed 290/512 (~57%)** and took **18.7 h** (P99 TTFT ~1.83 h, near the 2 h defer window). 4P4D is **worse** than 2P4D at this shape (2P4D 207083 = 270/512 @ 712 tok/s): with the same 4-decode capacity, 4P4D's extra prefill (4 vs 2) floods the decode-side KV transfer harder → more shedding. For 96k/32k @ con=128 the bottleneck is decode-KV capacity; adding prefill hurts. (**2P2D** 96k/32k @ con=128 `defer=7200` (job **209242**, 32 GPU) completed at **59/512 (~88% shed)**, 87.4 tok/s, median TTFT ~12.6 min — *worse* than both 2P4D and 4P4D even with the 2 h defer window, confirming 2-decode capacity cannot hold 128×96k sequences. **1P2D** `defer=7200` (job **209437**) still running; append when complete.)

### 1P1D vs 2P2D comparison (matched shape/concurrency)

Δ×(total) = 2P2D Total tok/s ÷ 1P1D Total tok/s. Δ×(per-GPU) = 2P2D tok/s/GPU ÷ 1P1D tok/s/GPU. Both runs were driven at the **same** concurrency levels, so 2P2D is not given extra load — hence total is ~parity while per-GPU is ~½ (2× the GPUs for the same offered work).

| ISL/OSL | Con | 1P1D Total | 2P2D Total | **Δ× (total)** | 1P1D /GPU | 2P2D /GPU | **Δ× (per-GPU)** |
|---|---|---|---|---|---|---|---|
| 8000/1000 | 8 | 730.9 | 714.1 | 0.98× | 45.7 | 22.3 | 0.49× |
| 8000/1000 | 32 | 2830.8 | 2682.4 | 0.95× | 176.9 | 83.8 | 0.47× |
| 8000/1000 | 64 | 5417.0 | 5160.9 | 0.95× | 338.6 | 161.3 | 0.48× |
| 8000/1000 | 128 | 10050.6 | 9909.5 | 0.99× | 628.2 | 309.7 | 0.49× |
| 8000/1000 | 256 | 16169.5 | 16853.2 | 1.04× | 1010.6 | 526.7 | 0.52× |
| 8000/1000 | 512 | 17852.8 | 18508.7 | 1.04× | 1115.8 | 578.4 | 0.52× |
| 4000/4000 | 8 | 172.7 | 166.1 | 0.96× | 10.8 | 5.2 | 0.48× |
| 4000/4000 | 32 | 678.5 | 650.8 | 0.96× | 42.4 | 20.3 | 0.48× |
| 4000/4000 | 64 | 1342.6 | 1281.2 | 0.95× | 83.9 | 40.0 | 0.48× |
| 4000/4000 | 128 | 2619.6 | 2522.0 | 0.96× | 163.7 | 78.8 | 0.48× |
| 4000/4000 | 256 | 5114.9 | 4960.5 | 0.97× | 319.7 | 155.0 | 0.48× |
| 4000/4000 | 512 | 9861.3 | 9225.3 | 0.94× | 616.3 | 288.3 | 0.47× |
| 8000/4000 | 8 | 253.8 | 244.5 | 0.96× | 15.9 | 7.6 | 0.48× |
| 8000/4000 | 32 | 993.0 | 964.2 | 0.97× | 62.1 | 30.1 | 0.48× |
| 8000/4000 | 64 | 1962.3 | 1911.0 | 0.97× | 122.6 | 59.7 | 0.49× |
| 8000/4000 | 128 | 3831.3 | 3699.7 | 0.97× | 239.5 | 115.6 | 0.48× |
| 8000/4000 | 256 | 7377.8 | 7139.1 | 0.97× | 461.1 | 223.1 | 0.48× |
| 32000/2000 | 8 | 1330.9 | 1290.1 | 0.97× | 83.2 | 40.3 | 0.48× |
| 32000/2000 | 32 | 4940.0 | 4774.0 | 0.97× | 308.8 | 149.2 | 0.48× |
| 32000/2000 | 64 | 9040.8 | 8960.9 | 0.99× | 565.1 | 280.0 | 0.50× |
| 32000/2000 | 128 | 13629.9 | 14562.5 | 1.07× | 851.9 | 455.1 | 0.53× |
| 32000/8000 | 8 | 419.3 | 407.0 | 0.97× | 26.2 | 12.7 | 0.48× |
| 32000/8000 | 32 | 1646.9 | 1602.2 | 0.97× | 102.9 | 50.1 | 0.49× |
| 32000/8000 | 64 | 3200.0 | 3145.1 | 0.98× | 200.0 | 98.3 | 0.49× |
| 32000/8000 | 128 | 6089.6 | 6017.4 | 0.99× | 380.6 | 188.0 | 0.49× |
| 96000/32000 | 8 | 338.2 | 327.8 | 0.97× | 21.1 | 10.2 | 0.48× |
| 96000/32000 | 32 | 1314.3 | 1277.5 | 0.97× | 82.1 | 39.9 | 0.49× |

Peak total throughput observed: **~17.9k tok/s (1P1D, 8000/1000 @ con=512)**, **~18.2k tok/s (1P2D, 8000/1000 @ con=1024)**, **~19.5k tok/s (2P2D, 8000/1000 @ con=1024)**, and **~19.2k tok/s (2P4D, 8000/1000 @ con=1024)**. **Reading it:** at matched concurrency, 2P2D total tok/s is ~0.95–1.07× of 1P1D (parity) while **per-GPU is ~0.48×** — 2P2D uses 2× the GPUs (32 vs 16) for the same offered concurrency, so throughput/GPU roughly halves. To show throughput *scaling* with GPUs, concurrency would need to scale with the deployment. 2P2D also shows slightly higher TPOT (~97–99 ms vs ~93–95 ms), reflecting the cross-node EP16 decode.

## Deployment configurations

| Topology | EP width | Status |
|---|---|---|
| 1P1D disaggregated | EP8 | ✅ NIAH validated; niah/aa_lcr pass |
| 1P2D disaggregated | EP16 | ✅ perf-characterized (peak 18.2k tok/s @ 8k/1k con=1024) |
| 2P2D disaggregated | EP16 | ✅ NIAH validated; niah/aa_lcr pass (unblocked vs Recipe 7) |
| 2P4D disaggregated | EP32 | ✅ perf-characterized (peak 19.2k tok/s @ 8k/1k con=1024) |
| 4P4D disaggregated | EP32 | ✅ perf (96k/32k @ con=128 stress; over-provisioned on prefill for this shape) |
| Single-node TP=8 (colocated) | — | ✅ correctness/determinism control |

## Upstream issues / PRs filed

| Item | Repo | What |
|---|---|---|
| aiter #4363 (issue) + #4365 (PR, Patch A) | ROCm/aiter | qh64 fp8 decode GPU-fault at page_size=1 on gfx942; gate native-qh64 to page_size==64 + repro `op_tests/test_mla_qh64_gfx942_pagesize1.py` |
| vLLM #49649 (issue) | vllm-project/vllm | persistent-kernel gate unsafe for gqa_ratio=64 fp8 |
| vLLM #49755 (PR, Patch B) | vllm-project/vllm | adds `sparse_mla_requires_persistent()` invariant + fail-fast guard in `rocm_aiter_mla_sparse.py` + CPU-only reproducer test (branch `mohitamd/fix-49649-sparse-mla-persistent-guard`) |
| aiter #4364 (issue) | ROCm/aiter | qh16 fp8 sparse decode run-to-run nondeterminism (residual ±1–2 needle) |
| aiter #4378 (PR) | ROCm/aiter | fix for aiter #4364 (qh16 nondeterminism) |

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
| `MORIIO_DEFER_TIMEOUT` (connector `defer_timeout`) | 600 (raise to 7200 for 96k@con128) | MoRIIO deferred-write expiry — the real knob (the `VLLM_MORIIO_*_TIMEOUT_S` vars are ignored by vLLM as unknown env vars). Now forwarded into the container via `connectors/moriio.env`; a submit-time export overrides it. |

## Reproducibility

- **Branch:** `mdeopuja/glm5.1-fp8-recipe8-fromscratch` on ROCm/MAD-private.
- **Dockerfiles:** `docker/GLM5.1-FP8.disagg.MI300X.from-scratch.Dockerfile` (+ `.CHANGELOG.md`), `docker/GLM5.1-FP8.disagg.MI300X.share.Dockerfile`.
- **Patches:** `docker/patches/patch_pr47766_v024.py`, `patch_aiter_mla_qh64_fold.py` (A), `patch_glm_dsa_force_persistent.py` (B), `patch_glm_sched_kv_xfer_stale_guard.py`, `patch_aiter_gpuless_import.py`, `rocminfo_buildshim.sh`, `patch_aiter_baton_selfheal_v2.py`; `scripts/vllm_dissag/apply_glm_dsa_indexer_warmup_fix.py`.
- **AITER repro (for #4365):** `docker/patches/repro_aiter_mla_qh64_gfx942_fault.py`, `run_repro_qh64_fault.sh`.
- **vLLM #49649 repro/fix:** `tests/kernels/attention/test_rocm_aiter_mla_sparse_persistent_guard.py` + patch `vllm_49649_fix.patch` (branch `mohitamd/fix-49649-sparse-mla-persistent-guard`).
- **Eval/perf orchestration:** `scripts/vllm_dissag/glm5.1_notes/sbatch_{1p1d,2p2d}_evalsuite_fromscratch.sh`, `run_accsuite_disagg_in_container.sh`, `sbatch_{1p1d,2p2d}_perf_fromscratch.sh` (gap-fills: `sbatch_{1p1d,2p2d}_perf_gapfill.sh`). Post-LCB accuracy re-runs: `sbatch_{1p1d,2p2d}_evalsuite_remaining.sh` + `run_accsuite_remaining_in_container.sh`.
- **Companion reports (same branch):** `scripts/vllm_dissag/glm5.1_notes/AITER_QH64_GPU_FAULT_REPORT.md`.

## Relationship to other recipes

- **Recipe 7** (vLLM v0.24.0 + PR#47766, minimal in-place patch on the stock image): parent recipe; 2P2D EP16 was blocked (MoRI wide-EP device assert), focus on the determinism study. Recipe 8 rebases to **v0.25.1 from source** with **AITER v0.1.18 + MoRI 42e895472** + qh16-fold + force-persistent, which **unblocks 2P2D EP16** and serves both topologies.
- **Recipe 4** (GLM-5.1-FP8 TP=8, single node): colocated correctness/determinism control.

## Acknowledgements

- Ravi Gupta (Ravi.Gupta@amd.com)
- Shiksha Patel (Shiksha.Patel@amd.com)
- Janet Tseng (Janet.Tseng@amd.com)
- Pradeep Sakhamoori (Pradeep.Sakhamoori@amd.com)
- Eliot Li (Eliot.Li@amd.com)
