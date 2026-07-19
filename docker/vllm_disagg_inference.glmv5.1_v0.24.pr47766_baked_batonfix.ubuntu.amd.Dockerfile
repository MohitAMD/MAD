# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# =============================================================================
# GLM-5.1-FP8 -- baked PR#47766 image + SELF-HEALING aiter JIT baton.
#
# FROM the baked pr47766 image, then patch aiter's FileBaton so a dead/hung builder
# can't deadlock the other workers forever (the recurring decode bring-up hang on
# lock_module_gemm_a8w8_blockscale / moe_fmoe_asm / rmsnorm). wait() gets a timeout;
# mp_lock() steals an abandoned lock and retries. Run with GLM_SKIP_PATCHERS=1.
#
#   docker build -f docker/vllm_disagg_inference.glmv5.1_v0.24.pr47766_baked_batonfix.ubuntu.amd.Dockerfile \
#     -t vllm-disagg:glmv5.1-v0.24-pr47766-baked-batonfix .
# =============================================================================
ARG BASE_IMAGE=vllm-disagg:glmv5.1-v0.24-pr47766-baked
FROM ${BASE_IMAGE}

WORKDIR /app
COPY docker/patches/patch_aiter_baton_timeout.py /tmp/patch_aiter_baton_timeout.py
RUN python3 /tmp/patch_aiter_baton_timeout.py && \
    rm -f /tmp/patch_aiter_baton_timeout.py && \
    echo "PATCH: aiter FileBaton self-heal (timeout+steal, AITER_BATON_TIMEOUT)" >> /app/versions.txt
RUN cat /app/versions.txt 2>/dev/null | tail -25 || true
