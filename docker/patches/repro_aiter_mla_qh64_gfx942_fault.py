#!/usr/bin/env python3
"""
Minimal reproducer: aiter native QH64 fp8 persistent MLA-decode kernel
(added in ROCm/aiter PR #3188) GPU-memory-access-faults on gfx942 (MI300X)
for the GLM-5.1-FP8 decode shape (gqa_ratio=64, qseqlen=1, fp8-Q/fp8-KV,
page_size=1).

WHY THIS SHAPE
--------------
PR #3188 ("Add native MLA QH64 fp8 persistent decode kernel for gfx942")
routes gfx942 nhead=64 / fp8 / max_seqlen_q==1 to the new native kernel
    hsa/gfx942/mla/mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co
instead of the pre-#3188 qh16 fold. The PR validated correctness ONLY at
page_size=64 (see PR conversation: all tester configs use page_size=64).
Production vLLM serves MLA decode with **page_size=1** (block_size=1), which
#3188's kernel never exercised -> it OOBs on the first decode forward and the
worker dies with "Memory access fault by GPU node-N ... Reason: Unknown".

This standalone script triggers exactly that kernel via aiter's public API
(no vLLM, no disagg, single GPU). It mirrors the metadata build + decode call
that vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py performs for
GLM-5.1-FP8 (DP8/TP1 => per-rank gqa_ratio=64).

EXPECTED RESULT
---------------
On gfx942 with aiter containing #3188 (e.g. v0.1.18, origin/main tip):
    [aiter] LoadKernel: _ZN5aiter39mla_a8w8_qh64_qseqlen1_gqaratio64_v3_psE ...
    Memory access fault by GPU node-2 ... on address 0x7f...5e5b000. Reason: Unknown.
    -> process aborts (SIGABRT / core dump), non-zero exit 134.
This is the bug: the qh64 kernel does an out-of-bounds GPU access for the
page_size=1 decode shape (never covered by #3188's page_size=64 validation).

WORKAROUND
----------
Apply patch_aiter_mla_qh64_fold.py (narrows the native-qh64 dispatch clause in
aiter/mla.py from ("gfx942","gfx950") to ("gfx950",) so gfx942 folds back to
the pre-#3188 qh16 kernel), then re-run -- it no longer faults:
    python3 patch_aiter_mla_qh64_fold.py && python3 repro_aiter_mla_qh64_gfx942_fault.py

SCOPE / CORRECTNESS NOTE
------------------------
This script's job is to demonstrate the GPU FAULT (the crash is deterministic
and unambiguous). It uses random fp8 inputs + unit scales, so the "finite="
flag it prints is only a coarse liveness signal, NOT a correctness check --
do not read a non-finite value here as a qh16-fold bug (the fold is the
production-proven pre-#3188 path). For rigorous numerical A/B validation, run
aiter's own golden-reference harness op_tests/test_mla_persistent.py at
page_size=1, nhead=64, fp8/fp8, decode_qlen=1 (PR #3188 only validated
page_size=64).

HOW TO RUN (single MI300X / gfx942 node, inside the aiter container)
-------------------------------------------------------------------
    srun -p amd-rccl -N1 --gres=gpu:1 --pty \
      docker run --rm --network host --ipc host \
        --device /dev/kfd --device /dev/dri --group-add video \
        -v "$PWD":/repro --entrypoint python3 \
        vllm-disagg:glmv5.1-v0.25.1-pr47766-csfix-morishik-aiter0118 \
        /repro/repro_aiter_mla_qh64_gfx942_fault.py
"""
import sys
import torch

import aiter
from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.jit.utils.chip_info import get_gfx


# --- GLM-5.1-FP8 per-rank decode geometry (DP8/TP1 => gqa_ratio=64) ----------
NUM_HEADS = 64          # gqa_ratio=64 (num_kv_heads=1) -> selects native qh64
NUM_KV_HEADS = 1
KV_LORA_RANK = 512      # MLA absorbed "nope" (== v_head_dim)
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM   # 576 (q / kv_buffer last dim)
V_HEAD_DIM = KV_LORA_RANK                         # 512 (output last dim)
PAGE_SIZE = 1           # <-- the untested-in-#3188 case that faults
MAX_SEQLEN_Q = 1        # decode


def build_decode(batch: int, kv_len: int, device: str = "cuda"):
    """Allocate the fp8 decode inputs + persistent metadata for `batch`
    single-token decode requests each with `kv_len` context tokens."""
    total_s = batch * MAX_SEQLEN_Q
    total_kv = batch * kv_len

    # fp8 Q / paged fp8 KV (page_size=1 -> one token per page).
    q = torch.randn(total_s, NUM_HEADS, QK_HEAD_DIM, device=device).to(dtypes.fp8)
    kv_buffer = torch.randn(
        total_kv, PAGE_SIZE, NUM_KV_HEADS, QK_HEAD_DIM, device=device
    ).to(dtypes.fp8)
    o = torch.empty(total_s, NUM_HEADS, V_HEAD_DIM, dtype=dtypes.bf16, device=device)

    # Ragged/paged index tensors (page_size=1 => one page index per kv token).
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * MAX_SEQLEN_Q
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * kv_len
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(batch, dtype=torch.int32, device=device)  # page_size=1

    # --- persistent MLA metadata (mirrors rocm_aiter_mla_sparse.py) ----------
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        batch, MAX_SEQLEN_Q, NUM_HEADS, dtypes.fp8, dtypes.fp8,
        is_sparse=True, fast_mode=True,
    )
    work_meta_data = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=device)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info_set = torch.empty(work_info_set_size, dtype=work_info_set_type, device=device)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=device)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=device)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=device)

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        NUM_HEADS,        # num_heads_per_head_k (gqa_ratio=64)
        NUM_KV_HEADS,     # num_heads_k
        True,             # is_causal
        work_meta_data, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE, kv_granularity=16,
        max_seqlen_qo=MAX_SEQLEN_Q, uni_seqlen_qo=MAX_SEQLEN_Q, fast_mode=True,
    )
    torch.cuda.synchronize()

    meta = dict(
        work_meta_data=work_meta_data, work_indptr=work_indptr,
        work_info_set=work_info_set, reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map, reduce_partial_map=reduce_partial_map,
    )
    return q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, meta


def main() -> int:
    gfx = get_gfx()
    print(f"[repro] gfx={gfx} aiter={getattr(aiter, '__version__', '?')} "
          f"nhead={NUM_HEADS} gqa_ratio={NUM_HEADS // NUM_KV_HEADS} "
          f"q/kv=fp8 page_size={PAGE_SIZE} max_seqlen_q={MAX_SEQLEN_Q}")
    if gfx != "gfx942":
        print(f"[repro] WARNING: this fault is specific to gfx942; running on {gfx}.")

    sm_scale = 1.0 / (QK_HEAD_DIM ** 0.5)
    # A couple of representative decode shapes; the very first call faults.
    for batch, kv_len in [(1, 512), (4, 1024), (16, 2048)]:
        print(f"[repro] --> mla_decode_fwd persistent  batch={batch} kv_len={kv_len} "
              f"(loads mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co) ...", flush=True)
        (q, kv_buffer, o, qo_indptr, kv_indptr,
         kv_indices, kv_last_page_lens, meta) = build_decode(batch, kv_len)

        mla_decode_fwd(
            q, kv_buffer, o,
            qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
            MAX_SEQLEN_Q,
            page_size=PAGE_SIZE, nhead_kv=NUM_KV_HEADS, sm_scale=sm_scale,
            q_scale=torch.ones(1, device=q.device),
            kv_scale=torch.ones(1, device=q.device),
            **meta,
        )
        torch.cuda.synchronize()  # fault (if any) surfaces here
        # NOTE: 'finite' is a coarse liveness signal only (random fp8 inputs +
        # unit scales); it is NOT a correctness check. See SCOPE note in the
        # module docstring -- use op_tests/test_mla_persistent.py for numerics.
        print(f"[repro]     returned batch={batch} kv_len={kv_len} "
              f"out={tuple(o.shape)} finite={bool(torch.isfinite(o.float()).all())}")

    print("[repro] COMPLETED WITHOUT GPU FAULT -> broken native qh64 kernel was "
          "avoided (qh16 fold / patch active, or non-gfx942).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
