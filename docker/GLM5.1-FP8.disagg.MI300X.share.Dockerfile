# =============================================================================
# GLM-5.1-FP8 — Disaggregated (WideEP) serving on AMD MI300X (gfx942)
# Reference recipe — validated 1P1D EP8 and 2P2D EP16, long-context (up to 96k)
# -----------------------------------------------------------------------------
# Component versions (pinned):
#
#   Base OS image     rocm/dev-ubuntu-22.04:7.2.3-complete
#   ROCm              7.2.3
#   Python            3.12
#   PyTorch           2.11.0  (ROCm fork, branch d0c8b1f3)
#   Triton            ROCm triton @ 0f380657
#   FlashAttention    @ 0e60e394
#   vLLM              0.25.1   + PR#47766 (sparse-MLA 6-field metadata key)
#   AITER             v0.1.18  (d6de77692f62e411375cf2ea0ec17792141d0540)
#   flydsl            0.2.4    (AITER v0.1.18 build dependency)
#   MoRI (amd_mori)   1.1.2.dev43+g42e895472
#
# Model: GLM-5.1-FP8 (GlmMoeDsaForCausalLM — MLA + DeepSeek-style sparse attn),
#        served fp8 weights + fp8 KV cache, block_size=1 (page_size=1).
#
# Topology note (IMPORTANT — this is why the patches below differ from the
# TP=8 GLM-5 fixes in vLLM#36855 / aiter#2821): WideEP disagg runs
# data_parallel_size=8, tensor_parallel_size=1, so EACH rank holds all 64
# attention heads => gqa_ratio=64 (NOT gqa=8). The MLA kernel coverage gap we
# hit is at the gqa=64 end, not the num_heads<16 end.
#
# Build (three inputs assumed already available as the FROM base — see notes):
#   docker build -f docker/GLM5.1-FP8.disagg.MI300X.share.Dockerfile \
#     -t glm5.1-fp8-disagg:mi300x .
# =============================================================================

# -----------------------------------------------------------------------------
# BASE: a vLLM 0.25.1 ROCm 7.2.3 image that already carries:
#   (a) PyTorch 2.11 / Triton / FlashAttention as pinned above, and
#   (b) vLLM PR#47766 (6-field sparse-MLA metadata key) applied to the vLLM
#       install. PR#47766 makes the persistent sparse-MLA kernel numerically
#       correct across chunked-prefill continuations — it is a HARD requirement
#       for the force-persistent patch (Patch 3) to be safe.
#   (c) the WideEP DP-rank round-robin router (vllm-router) for disagg.
#
# If you are starting from stock vLLM 0.25.1, cherry-pick PR#47766 into the vLLM
# install before this stage (it is a multi-file change to the sparse-MLA backend
# metadata builder; a source patch, not a runtime edit).
# -----------------------------------------------------------------------------
ARG BASE_IMAGE=vllm-rocm:0.25.1-pr47766
FROM ${BASE_IMAGE}

ARG PYTORCH_ROCM_ARCH="gfx942"
ARG GPU_ARCHS="gfx942"
ARG MAX_JOBS=32
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} GPU_ARCHS=${GPU_ARCHS} MAX_JOBS=${MAX_JOBS}
SHELL ["/bin/bash", "-c"]
WORKDIR /app

# -----------------------------------------------------------------------------
# 1) AITER v0.1.18  (adds native self-healing JIT baton; pinned kernel set)
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REF=v0.1.18
RUN pip install --no-deps -U "flydsl==0.2.4" && \
    rm -rf /tmp/aiter-src && \
    git clone --recursive "${AITER_REPO}" /tmp/aiter-src && \
    cd /tmp/aiter-src && git checkout "${AITER_REF}" && \
    git submodule sync && git submodule update --init --recursive && \
    (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.18 pip install --no-build-isolation --no-deps -v . && \
    cd / && rm -rf /tmp/aiter-src /opt/vllm_cache/aiter_jit /root/.aiter 2>/dev/null || true

# -----------------------------------------------------------------------------
# 2) MoRI (amd_mori) 1.1.2.dev43+g42e895472  — WideEP all2all + MoRIIO KV xfer.
#    Skip this stage if your base already carries the pinned MoRI.
# -----------------------------------------------------------------------------
# ARG MORI_REF=42e895472
# RUN <build/install amd_mori at ${MORI_REF}>   # see ROCm/mori build docs

# -----------------------------------------------------------------------------
# 3) PATCH A — AITER MLA dispatch: fold gfx942 gqa64 fp8 decode qh64 -> qh16
#    WHY: AITER's native qh64 fp8 persistent decode kernel (#3188,
#    mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps.co) GPU memory-access-FAULTS on
#    gfx942 at page_size=1 (block_size=1). It was validated only at page_size=64.
#    vLLM serves MLA at page_size=1, so the first decode forward crashes the
#    worker. The pre-#3188 qh16 fold path runs correctly. gfx950 is untouched.
#    -> Tracking: file as new AITER issue/PR (see recipe summary).
# -----------------------------------------------------------------------------
RUN python3 - <<'PY'
import importlib.util, os, sys
f = os.path.join(os.path.dirname(importlib.util.find_spec("aiter").origin), "mla.py")
s = open(f).read()
if 'gfx942 native qh64' in s:
    print("[patchA] already applied"); sys.exit(0)
old = '                get_gfx() in ("gfx942", "gfx950")\n                and nhead == 64'
new = '                get_gfx() == "gfx950"  # gfx942 native qh64 fp8 page_size=1 GPU-faults -> fold to qh16\n                and nhead == 64'
assert old in s, "[patchA] aiter/mla.py qh64 dispatch anchor not found (version drift)"
open(f, "w").write(s.replace(old, new, 1))
import py_compile; py_compile.compile(f, doraise=True)
print("[patchA] gfx942 gqa64 fp8 decode now folds to qh16")
PY

# -----------------------------------------------------------------------------
# 4) PATCH B — vLLM sparse-MLA: force the persistent path ON for gqa64.
#    WHY: the base carries the persistent-kernel gate (aiter #4076 / vLLM
#    #47567) which drops to the NON-persistent split-KV MLA path for
#    chunked-prefill continuations. gqa_ratio=64 fp8 has NO non-persistent
#    kernel (asm_mla.cu:949 "gqa_ratio=64 only supports persistent mode"), so
#    the prefill worker crashes on the first >1-chunk request. PR#47766 already
#    makes the PERSISTENT kernel correct across chunked prefill, so keeping it
#    on is both safe and required here.
#    -> Tracking: file as new vLLM issue/PR (see recipe summary).
# -----------------------------------------------------------------------------
RUN python3 - <<'PY'
import importlib.util, os, sys
v = os.path.dirname(importlib.util.find_spec("vllm").origin)
f = os.path.join(v, "v1/attention/backends/mla/rocm_aiter_mla_sparse.py")
s = open(f).read()
if 'force-persistent' in s:
    print("[patchB] already applied"); sys.exit(0)
old = "        use_persistent = not is_chunked_continuation.any()"
new = ("        use_persistent = True  # force-persistent: gqa64 fp8 has no non-persistent\n"
       "        # kernel; keep persistent qh16 path, rely on vLLM PR#47766 for chunked-prefill\n"
       "        # correctness. Was: not is_chunked_continuation.any()\n"
       "        _ = is_chunked_continuation")
assert old in s, "[patchB] rocm_aiter_mla_sparse.py use_persistent anchor not found (version drift)"
open(f, "w").write(s.replace(old, new, 1))
import py_compile; py_compile.compile(f, doraise=True)
print("[patchB] use_persistent forced True (gqa64 keeps persistent path)")
PY

# Scrub the AITER JIT cache so kernels recompile against the patched dispatch.
RUN rm -rf /opt/vllm_cache/aiter_jit /root/.aiter /tmp/vllm_cache*/aiter_jit 2>/dev/null || true

# -----------------------------------------------------------------------------
# Runtime (set by the launcher, shown here for reference):
#   VLLM_ROCM_USE_AITER=1  VLLM_GCN_ARCH=gfx942
#   Disagg: prefill DP=8 TP=1  +  decode DP=8 TP=1  (EP8 per role; EP16 for 2P2D)
#   KV connector: MoRIIO (mori.io)   Router: vllm-router (DP-rank round-robin)
#   Serve containers must use --network host.
# =============================================================================
