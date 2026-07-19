# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# =============================================================================
# GLM-5.1-FP8 -- "GLM_SKIP_PATCHERS=1" variant of the PR#47766 image.
#
# FROM the pr47766 image (stock v0.24.0 + complete 6-field #47766 metadata-key
# patch + vllm-router PR#181 + rocm.py boot fix), then BAKE the GLM DSA/MoRIIO
# runtime patchers IN-SOURCE at build time -- so the served image already carries
# the fixes and is launched with GLM_SKIP_PATCHERS=1 (NO runtime patching, no boot
# cost, and any patcher anchor mismatch fails the BUILD instead of a live run).
#
# Same 7 patchers, same order, as connectors/moriio.sh's _glm_dsa_runtime_patch.
#
#   docker build -f docker/vllm_disagg_inference.glmv5.1_v0.24.pr47766_baked.ubuntu.amd.Dockerfile \
#     -t vllm-disagg:glmv5.1-v0.24-pr47766-baked .
#
# Run with GLM_SKIP_PATCHERS=1.
# =============================================================================
ARG BASE_IMAGE=vllm-disagg:glmv5.1-v0.24-pr47766
FROM ${BASE_IMAGE}

WORKDIR /app
# rocm.py is already patched in the base -> `import vllm` is GPU-free at build,
# so the patchers can resolve the vllm dir without a device.
COPY scripts/vllm_dissag/apply_glm_dsa_kernel_fix.py \
     scripts/vllm_dissag/apply_glm_dsa_moriio_dualkv_fix.py \
     scripts/vllm_dissag/apply_glm_dsa_moriio_engine_fix.py \
     scripts/vllm_dissag/apply_glm_dsa_moriio_gate_fix.py \
     scripts/vllm_dissag/apply_glm_moriio_abort_guard_fix.py \
     scripts/vllm_dissag/apply_glm_dsa_persistent_kernel_gate_fix.py \
     scripts/vllm_dissag/apply_glm_aiter_sampling_oob_fix.py \
     /tmp/patchers/

RUN set -e && \
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')" && \
    echo "Baking GLM DSA/MoRIIO patchers into ${VLLM_DIR}" && \
    for p in apply_glm_dsa_kernel_fix.py \
             apply_glm_dsa_moriio_dualkv_fix.py \
             apply_glm_dsa_moriio_engine_fix.py \
             apply_glm_dsa_moriio_gate_fix.py \
             apply_glm_moriio_abort_guard_fix.py \
             apply_glm_dsa_persistent_kernel_gate_fix.py \
             apply_glm_aiter_sampling_oob_fix.py; do \
      echo "[bake] applying $p" && \
      python3 "/tmp/patchers/$p" "$VLLM_DIR" && \
      echo "PATCH(baked): $p" >> /app/versions.txt ; \
    done && \
    rm -rf /tmp/patchers && \
    echo "GLM_SKIP_PATCHERS_READY=1 (DSA/MoRIIO fixes baked in-source)" >> /app/versions.txt

RUN cat /app/versions.txt 2>/dev/null | tail -25 || true
