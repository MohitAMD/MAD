> **STATUS: MOSTLY COMPLETE.** MTP confirmed working on 1P1D EP8 and 1P2D EP16; 2P2D EP16 expected (topology-agnostic fix) but not yet empirically confirmed (cluster infra: dead node + fair-share QoS throttle). Upstream fix PR: MohitAMD/vllm#2.

# Recipe 18: GLM-5.3-FP8 on AMD MI300X — WideEP PD-Disaggregated MTP (Speculative Decoding) (vLLM v0.27.1 + AITER v0.1.19, Recipe 17 stack)

Recipe 18 adds **MTP (Multi-Token Prediction / speculative decoding) to the Recipe 17 WideEP PD-disaggregated stack**. Recipe 14 enabled MTP single-node (TP=8); Recipe 17 established the WideEP disagg sweep without MTP. Recipe 18 is the union: MTP running across the prefill→decode boundary. The one thing that had to be fixed is a MoRIIO KV-connector block-geometry bug exposed by MTP; everything else is inherited unchanged from Recipe 17.

## Summary
| Field | Value |
|---|---|
| Model | GLM-5.3-FP8 |
| vLLM / AITER | v0.27.1 + PR#176 WideEP overlay / v0.1.19 (prebaked4) |
| MTP | `--speculative-config {"method":"mtp","num_speculative_tokens":3}` (both roles) |
| CUDAGraph | FULL_AND_PIECEWISE on decode |
| Topologies | 1P1D EP8 confirmed; 1P2D EP16 confirmed; 2P2D EP16 expected (infra-blocked) |
| Image | vllm-glm51-v027-aiter019-recipe15-wideep:prebaked4 |
| Enabler | MoRIIO block-offset fix (glm52_recipe15_fix.py sitecustomize; upstream MohitAMD/vllm#2) |

## Key Finding
MTP works on WideEP PD-disagg (validated 1P1D + 1P2D); benefit is workload-dependent (matches Recipe 14): net win on decode-heavy 8k/8k at low concurrency (1P1D con=1: +30% tok/s, 1.36x TPOT; con4/8 1.42-1.45x TPOT), net loss on short-output 8k/1k (0.72-0.86x). Not blocked at the vLLM level — the only blocker was a MoRIIO block-transfer bug (fixed; MohitAMD/vllm#2). Coherent output, 0 token-0 corruption, FULL cudagraph on both topologies.

## The Enabler — MoRIIO block-offset fix
Symptom (job 237103): first prefill→decode KV write crashed `ValueError: local_block_ids longer than remote_block_ids: 8 > 5` in moriio_layout.py compute_block_transfer_offsets.
Root cause: block_size=1 + MTP n=N → prefill reserves N lookahead slots, so local exceeds remote (prompt) by exactly N (8003 vs 8000; 8 vs 5). Block order positional → local[:len(remote)] is prompt KV, trailing are empty lookahead scratch.
Fix: clamp local_block_ids to prompt prefix (symmetric with the existing shorter-local handling). Runtime monkeypatch in glm52_recipe15_fix.py + upstream source fix MohitAMD/vllm#2. On 1P2D the clamp fired 32x, old error absent.

## Enable MTP
```
# INNER single quotes REQUIRED (connector does eval "model_args=(...)"; brace-expansion else shreds JSON)
export EXTRA_VLLM_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
```
Set globally (both roles) so prefill+decode share MTP-layer KV geometry. Keep DECODE_CUDAGRAPH_MODE=FULL_AND_PIECEWISE, VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900. Scripts: sbatch_r18_<topo>_glm53_mtp.sh (+ sbatch_r18_1p1d_glm53_baseline.sh).

## MTP vs baseline A/B (1P1D EP8) — MTP=237737, baseline=239261, 0 failed all cells
### 8k/1k (net loss)
| con | Base tok/s | Base TPOT | MTP tok/s | MTP TPOT | Thpt | TPOT sp |
|--|--|--|--|--|--|--|
| 1 | 287.76 | 28.93 | 236.20 | 35.97 | 0.82x | 0.80x |
| 4 | 1080.34 | 29.99 | 931.50 | 27.41 | 0.86x | 1.09x |
| 8 | 2110.99 | 30.04 | 1520.08 | 32.76 | 0.72x | 0.92x |
| 16 | 3685.08 | 33.64 | 2726.79 | 34.44 | 0.74x | 0.98x |
| 32 | 6383.28 | 37.78 | 4887.25 | 45.66 | 0.77x | 0.83x |
### 8k/8k (MTP wins at low con)
| con | Base tok/s | Base TPOT | MTP tok/s | MTP TPOT | Thpt | TPOT sp |
|--|--|--|--|--|--|--|
| 1 | 68.58 | 29.14 | 88.92 | 21.48 | 1.30x | 1.36x |
| 4 | 263.02 | 29.96 | 273.02 | 21.07 | 1.04x | 1.42x |
| 8 | 514.18 | 30.57 | 561.99 | 21.07 | 1.09x | 1.45x |

## MTP acceptance (n=3)
Prometheus spec_decode counters: drafts=63,400, draft_tokens=190,200 (=x3), accepted=61,134 → 32.1% per-draft-token, 1.96 accepted tok/step. Caveat: synthetic bench (random prompts, ignore_eos) = near worst-case; Recipe 14 natural-text was 62-72%.

## 1P1D EP8 results (job 237737)
8k/1k: con1 236.20(TPOT35.97) / con4 931.50(27.41) / con8 1520.08(32.76) / con16 2726.79(34.44) / con32 4887.25(45.66) — all success.
8k/8k: con1 88.92(21.48) / con4 273.02(21.07) / con8 561.99(21.07).

## 1P2D EP16 results (job 239264) — CONFIRMED (clamp 32x, token-0 PASS, FULL cudagraph)
8k/1k: con1 159.26(60.54) / con4 588.02(65.94) / con8 1194.49(42.19) / con16 2199.87(45.41) / con32 3927.95(49.23).
8k/8k: con1 57.01(29.59) / con4 186.83(28.85) / con8 347.07(30.99).

## 2P2D EP16 — NOT YET CONFIRMED (infra)
Expected to work (topology-agnostic clamp, proven 1P1D+1P2D). Job 239265: (a) first attempt hit dead node useocpm2m-097-132 (froze at container init, deadlocked rendezvous); (b) resubmits rejected by usage-based fair-share QoS throttle (amd-rccl-guest forced to invalid 'low' QoS on amd-rccl). Action: resubmit with --exclude=useocpm2m-097-132 once fair-share recovers (background retry queued).

## Correctness
| Check | 1P1D | 1P2D |
|--|--|--|
| Token-0 (`!`) | PASS (0) | PASS (0) |
| Coherent output | PASS | PASS |
| FULL cudagraph | YES | YES |
| Clamp / old ValueError | fixed | fixed (32x) |

## Known issues
- Fix ships as runtime monkeypatch; source fix upstreamed as MohitAMD/vllm#2 (self-skips once in image).
- num_speculative_tokens>1 re-runs the single MTP layer (may lower acceptance); synthetic bench → 32%.
- MTP net loss on short-output (8k/1k) — enable selectively.
- 2P2D unconfirmed pending cluster infra.

## Relationship
- Recipe 14: MTP single-node (TP=8).
- Recipe 17: WideEP PD-disagg sweep, no MTP.
- Recipe 18: MTP on Recipe 17 WideEP disagg (both roles) + MoRIIO block-offset fix (MohitAMD/vllm#2). Confirmed 1P1D + 1P2D; 2P2D expected.
