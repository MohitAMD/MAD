# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# =============================================================================
# GLM-5.1-FP8 on AMD MI300 -- vLLM v0.24.0 + PR#47766 (COMPLETE, 6-field metadata
# key) applied IN-PLACE as a .py patch (NO vLLM rebuild), then the MoRI-EP / WideEP
# disagg runtime layered on top.
#
# This is the Dockerfile form of build_glm51_v024_patched.sh (psakhamo@amd), which
# does: base v0.24.0 -> patch rocm_aiter_mla_sparse.py to a 6-field sparse-MLA
# metadata key (clamped_seq_lens + clamped_context_lens + seg_lengths) so the aiter
# MLA kernel cannot reuse stale metadata at ISL >~10K. That script's output image was
# base+patch only; here we ALSO add the WideEP disagg bits that are not in the vllm
# image (the router binary + a boot-robustness fix), reusing stock v0.24.0's already-
# bundled MoRIIO KV connector + all2all `mori` backends + MoRI lib.
#
#   docker build -f docker/vllm_disagg_inference.glmv5.1_v0.24.pr47766.ubuntu.amd.Dockerfile \
#     -t vllm-disagg:glmv5.1-v0.24-pr47766 .
#
# vs the .local variant: NO full vLLM source rebuild (fast, minutes) -- it takes
# stock v0.24.0 vLLM and applies ONLY the complete #47766 metadata-key patch, matching
# psakhamo's validated 10/10-deterministic recipe, instead of the 20-commit rebase.
# =============================================================================
ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.24.0
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app
ARG PYTORCH_ROCM_ARCH=gfx942
RUN mkdir -p /app && echo "BASE_IMAGE=${BASE_IMAGE} (v0.24.0 + PR#47766 6-field + WideEP runtime)" >> /app/versions.txt

# -----------------------------------------------------------------------------
# 1. PR#47766 (COMPLETE): 6-field sparse-MLA metadata key. In-place .py edit of the
#    STOCK v0.24.0 vLLM -- no recompile. find_spec() locates vllm without importing
#    it, so this is GPU-free and BuildKit-safe.
# -----------------------------------------------------------------------------
COPY docker/patches/patch_pr47766_v024.py /tmp/patch_pr47766.py
RUN python3 /tmp/patch_pr47766.py && \
    ROCM_SPARSE=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py && \
    grep -q 'clamped_context_lens.tobytes()' "$ROCM_SPARSE" && \
    grep -q 'seg_lengths.tobytes()' "$ROCM_SPARSE" && \
    python3 -m py_compile "$ROCM_SPARSE" && \
    echo "PATCH: PR#47766 complete (6-field sparse-MLA metadata key)" >> /app/versions.txt && \
    rm -f /tmp/patch_pr47766.py

# -----------------------------------------------------------------------------
# 2. vllm-router (WideEP disagg proxy: DP-rank round-robin + MoRIIO). Not shipped in
#    the vllm image, so build it. openssl-sys needs libssl-dev headers; archive.ubuntu.com
#    is unreachable here, so libssl-dev is vendored as a local .deb (jammy/amd64
#    3.0.2-0ubuntu1.23, matches the base's libssl3) and installed offline via dpkg -i.
#    pkg-config/gcc/git/curl are already in the base; rustup + crates.io are reachable.
#    Set --build-arg WITH_ROUTER=0 to skip (falls back to PROXY_TYPE=moriio_toy).
# -----------------------------------------------------------------------------
ARG WITH_ROUTER=1
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
ARG ROUTER_REF=ravgupta/discovery-dp-rank-roundrobin
ARG RUST_TOOLCHAIN=1.88.0
COPY docker/debs/libssl-dev_3.0.2-0ubuntu1.23_amd64.deb /tmp/libssl-dev.deb
RUN if [ "${WITH_ROUTER}" != "1" ]; then \
      rm -f /tmp/libssl-dev.deb; \
      echo "WITH_ROUTER=0: skipping vllm-router" | tee -a /app/versions.txt; \
    else set -e && \
      dpkg -i /tmp/libssl-dev.deb && rm -f /tmp/libssl-dev.deb && \
      if ! command -v cargo >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
      fi && \
      export PATH="/root/.cargo/bin:${PATH}" && \
      rm -rf /tmp/vllm-router-src && \
      git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src && \
      cd /tmp/vllm-router-src && git checkout "${ROUTER_REF}" && \
      cargo build --release && \
      install -m 755 target/release/vllm-router /usr/local/bin/vllm-router && \
      vllm-router --help 2>&1 | grep -q moriio && \
      echo "VLLM_ROUTER_REF=${ROUTER_REF}@$(git -C /tmp/vllm-router-src rev-parse HEAD)" >> /app/versions.txt && \
      rm -rf /tmp/vllm-router-src; \
    fi

# -----------------------------------------------------------------------------
# 3. Boot-robustness patch for WideEP: vllm/platforms/rocm.py resolves the GCN arch
#    at module load via amdsmi; on failure it calls logger.warning_once(), which
#    imports vllm.platforms.current_platform WHILE vllm.platforms is still initializing
#    -> circular ImportError -> crash on concurrent EP-worker boot. Downgrade
#    warning_once->warning + return ${VLLM_GCN_ARCH:-gfx942} instead of torch.cuda.
# -----------------------------------------------------------------------------
RUN ROCM_PY=/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py && \
    test -f "$ROCM_PY" && \
    sed -i 's/logger\.warning_once(/logger.warning(/g' "$ROCM_PY" && \
    sed -i 's#return torch\.cuda\.get_device_properties("cuda")\.gcnArchName#return __import__("os").environ.get("VLLM_GCN_ARCH", "gfx942")#' "$ROCM_PY" && \
    if grep -q 'logger.warning_once(' "$ROCM_PY"; then echo "PATCH FAILED: warning_once remains" >&2; exit 1; fi && \
    if grep -q 'torch.cuda.get_device_properties("cuda").gcnArchName' "$ROCM_PY"; then echo "PATCH FAILED: torch.cuda fallback remains" >&2; exit 1; fi && \
    python3 -m py_compile "$ROCM_PY" && \
    echo "PATCH: rocm.py GCN-arch circular-import fix" >> /app/versions.txt

# -----------------------------------------------------------------------------
# 4. Cache locations + MoRI JIT scrub (stale build-time .hsaco.lock -> runtime deadlock).
# -----------------------------------------------------------------------------
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr
RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED" >> /app/versions.txt
RUN cat /app/versions.txt 2>/dev/null | tail -20 || true
