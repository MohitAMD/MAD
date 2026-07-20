# aiter bug report — native QH64 fp8 persistent MLA decode kernel GPU-faults on gfx942 (MI300)

## Summary

The native QH64 fp8 persistent MLA decode kernel added in aiter **PR #3188**
("Add native MLA QH64 fp8 persistent decode kernel for gfx942", commit
`04427a5d8`) causes an immediate **GPU memory access fault** on **gfx942 (MI300X)**
the first time it is invoked for the GLM-5.1-FP8 `gqa_ratio=64`, `qseqlen=1`
(decode), fp8-Q/fp8-KV, page_size=1 shape. The prefill engine worker dies before a
single token is produced.

The faulting kernel:

```
_ZN5aiter39mla_a8w8_qh64_qseqlen1_gqaratio64_v3_psE
  hsa/gfx942/mla/mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co   (56520 bytes)
  (lse variant: mla_a8w8_qh64_qseqlen1_gqaratio64_lse_v3_ps.co, 56608 bytes)
```

Dispatch entry (`hsa/gfx942/mla/mla_asm.csv`, added by #3188):

```
fp8,fp8,64,1,1,0,0,0,_ZN5aiter39mla_a8w8_qh64_qseqlen1_gqaratio64_v3_psE,mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co
fp8,fp8,64,1,1,0,0,1,_ZN5aiter43mla_a8w8_qh64_qseqlen1_gqaratio64_lse_v3_psE,mla_a8w8_qh64_qseqlen1_gqaratio64_lse_v3_ps.co
```

Before #3188, this shape was served by folding `gqa_ratio=64 -> qh16`
(`mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co`), which runs correctly. #3188 changed
`aiter/mla.py` to route the shape to the new native qh64 kernel instead, which
faults.

## Impact

Reproduced on **three independent aiter builds**, all faulting identically on the
first fp8 gqa64 decode:

| aiter build | contents | result |
|---|---|---|
| `e03fa6040` + cherry-pick `04427a5d8` | #3188 in isolation | GPU fault (addr `0x7f2901e5b000`, node -017) |
| `ac90d5c89` | #3188 + #4144 batch-gate + intermediate | GPU fault (addr `0x7f4d5ee5b000`, node -014) |
| `dedc19d32` (origin/main tip, 2026-07-19) | #3188 + all later commits incl. #4227 metadata | GPU fault (addr `0x7f55eae13000`, node -020) |

- The `.co` blob (`6f3b953d45f9e072bdbc712e74f1f24e1c22efce`) is **byte-identical
  from #3188 through `origin/main` tip** and is never modified or reverted; the tip
  `mla_asm.csv` still dispatches this shape to it.
- The hypothesis that later metadata changes (#4227 forward-compat
  `get_mla_metadata_v1` + `v1_2_device.cuh`) might avoid the OOB is **disproven** —
  tip reproduces the identical fault. The OOB is in the **kernel binary itself**,
  not the metadata that feeds it.

## Environment

| | |
|---|---|
| GPU | AMD MI300X, `gfx942` |
| aiter | `e03fa6040` + cherry-pick `04427a5d8` (#3188); also `ac90d5c89` |
| flydsl | 0.2.4 |
| vLLM | 0.24.1.dev20+g9bec14e18 (GLM-5.1-FP8 DSA + MoRIIO disagg) |
| model | zai-org/GLM-5.1-FP8 (`GlmMoeDsaForCausalLM`, MLA + DSA sparse attn) |
| topology | 1P1D disaggregated, EP8 (`data_parallel_size=8`, `tensor_parallel_size=1` per role) |
| dtype | fp8 Q + fp8 KV (`--kv-cache-dtype fp8`), block_size=1, seed=0 |

At `data_parallel_size=8, tensor_parallel_size=1` the per-rank MLA head grouping is
`gqa_ratio=64`, which #3188's dispatch now maps to the native qh64 persistent
kernel.

## Exact error

On the first end-to-end warmup completion (a `qseqlen=1` decode step computed in the
prefill role's `execute_dummy_batch`/warmup), the worker loads the kernel and the
GPU faults immediately:

```
[aiter] LoadKernel: _ZN5aiter39mla_a8w8_qh64_qseqlen1_gqaratio64_v3_psE hsaco:
  .../aiter_meta/hsa//gfx942/mla/mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co
Memory access fault by GPU node-2 (Agent handle: 0x...) on address 0x7f...5e5b000. Reason: Unknown.
GPU core dump failed
(EngineCore_DP0) ERROR multiproc_executor.py:284] Worker proc VllmWorker-0 died unexpectedly, shutting down executor.
```

Downstream cascade (secondary, not the cause): sibling DP ranks then fail the
cross-DP coordination all-reduce over gloo (`coordinate_batch_across_dp ->
_synchronize_dp_ranks`, `Connection closed by peer`), ending in `EngineDeadError`.

- Reproduced on 2 independent node sets (`useocpm2m-097-014`, `useocpm2m-097-017`),
  fault addresses `0x7f4d5ee5b000` and `0x7f2901e5b000` — consistent OOB pattern.
- **Decode role never faults**; the fault is isolated to the role that first runs
  the qh64 persistent kernel.
- `dmesg` shows no amdgpu VM-fault/RAS entries beyond the reported access fault →
  clean kernel OOB, not hardware/RAS.

## Reproduction

1. Build aiter with #3188 present (either `e03fa6040` + `git cherry-pick -n
   04427a5d8`, or `ac90d5c89`), flydsl 0.2.4, on the ROCm gfx942 base.
2. Serve GLM-5.1-FP8 1P1D disaggregated, EP8 (DP8/TP1 per role), fp8 KV, page_size=1.
3. Issue any `/v1/completions` request. The prefill worker faults on the first
   decode-shaped forward when it loads `mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co`.

## Workaround

Route `gqa_ratio=64` fp8 back to the `qh16` fold path
(`mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co`) — i.e. revert #3188's `aiter/mla.py`
dispatch change for this shape. That kernel runs correctly (it is the pre-#3188
baseline), though it exhibits the separate run-to-run nondeterminism this
investigation was trying to resolve.

## Ask

Please investigate the OOB in `mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps` on gfx942
for the `qseqlen=1`, `gqa_ratio=64`, fp8/fp8, page_size=1 decode shape (likely a
metadata/index bound or LDS/global addressing bug in the new kernel or its
launch-parameter computation).
