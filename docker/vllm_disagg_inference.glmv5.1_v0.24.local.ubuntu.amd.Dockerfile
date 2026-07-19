# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# =============================================================================
# LOCAL build variant of vllm_disagg_inference.glmv5.1_v0.24 — installs vLLM from a
# LOCAL rebased source tree (the glm5.1-dsa-wideEP_on_v0.24.0 branch that is NOT
# pushed anywhere) via a BuildKit named build-context, instead of git-cloning it.
#
# Build (context = MAD repo root; vLLM source passed as a named context):
#   DOCKER_BUILDKIT=1 docker build \
#     -f docker/vllm_disagg_inference.glmv5.1_v0.24.local.ubuntu.amd.Dockerfile \
#     --build-context vllmsrc=/home/mdeopuja/cohere/vllm-rebase \
#     --build-arg REBUILD_AITER=0 --build-arg REBUILD_MORI=0 --build-arg WITH_NIXL=0 \
#     -t vllm-disagg:glmv5.1-v0.24-local .
#
# Fast-first-build defaults: reuse the v0.24.0 image's bundled AITER/MoRI and skip
# NIXL (MoRI-EP + toy proxy is enough for 1P1D/2P2D NIAH). Flip REBUILD_AITER/MORI=1
# and WITH_NIXL=1 later if NIAH needs the validated aiter e03fa6040 / mori pins.
# =============================================================================
ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.24.0
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app

ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
ARG MAX_JOBS=32
ARG WITH_NIXL=0
ARG NIC_COMPILATION_ARCH="cx7"
ARG REBUILD_AITER=0
ARG REBUILD_MORI=0
RUN mkdir -p /app && echo "BASE_IMAGE=${BASE_IMAGE} (v0.24.0 local build)" >> /app/versions.txt

# -----------------------------------------------------------------------------
# 1. MoRI (optional rebuild; default reuse base bundled)
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=42e895472b08
ENV MORI_GPU_ARCHS=gfx942 BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF
RUN if [ "${REBUILD_MORI}" != "1" ]; then \
      echo "REBUILD_MORI=0: keeping base bundled MoRI" | tee -a /app/versions.txt; \
    else set -e && \
      apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates && \
      pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
      (pip uninstall -y amd_mori amd-mori amd-mori-nightly mori 2>/dev/null || true) && \
      rm -rf /tmp/mori-src && git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
      cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
      BUILD_UMBP=OFF pip install . && \
      python3 -c "import mori, mori.io, mori.ops; print('MoRI OK')" && \
      echo "MORI_REF=${MORI_REF}@$(git -C /tmp/mori-src rev-parse HEAD)" >> /app/versions.txt && \
      rm -rf /tmp/mori-src; \
    fi

# -----------------------------------------------------------------------------
# 2. AITER (optional rebuild; default reuse base bundled)
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REF=e03fa6040
RUN if [ "${REBUILD_AITER}" != "1" ]; then \
      echo "REBUILD_AITER=0: keeping base bundled AITER" | tee -a /app/versions.txt; \
    else set -e && \
      rm -rf /tmp/aiter-src && git clone --recursive "${AITER_REPO}" /tmp/aiter-src && \
      cd /tmp/aiter-src && git checkout "${AITER_REF}" && git submodule update --init --recursive && \
      (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
      pip install --no-build-isolation --no-deps -v . && \
      pip install --no-deps -U "flydsl>=0.1.7,<0.1.9" && \
      echo "AITER_REF=${AITER_REF}@$(git -C /tmp/aiter-src rev-parse HEAD)" >> /app/versions.txt && \
      rm -rf /tmp/aiter-src /opt/vllm_cache/aiter_jit /root/.aiter; \
    fi

# -----------------------------------------------------------------------------
# 3. vLLM: install from the LOCAL rebased source (named build-context `vllmsrc`).
#    SETUPTOOLS_SCM pretend-version avoids any git-describe dependency in the copy.
# -----------------------------------------------------------------------------
ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    MAX_JOBS=${MAX_JOBS} \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.24.0+glm5.1dsa
COPY --from=vllmsrc . /tmp/vllm-src
RUN cd /tmp/vllm-src && \
    echo "VLLM_LOCAL=$(git rev-parse HEAD 2>/dev/null || echo nogit)" >> /app/versions.txt && \
    pip uninstall -y vllm 2>/dev/null || true && \
    pip install --no-deps --no-build-isolation -v . && \
    python3 -c "import vllm; print('vLLM', vllm.__version__, 'from', vllm.__file__)" && \
    cd / && rm -rf /tmp/vllm-src

RUN python3 -c "import mori, mori.io, mori.ops; import vllm; print('post-vLLM import OK: vllm', vllm.__version__, '+ MoRI')"

# -----------------------------------------------------------------------------
# 4. vllm-router (DP-rank round-robin + MoRIIO + 2P2D KV-notify dpfix)
# -----------------------------------------------------------------------------
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
ARG ROUTER_REF=ravgupta/discovery-dp-rank-roundrobin
ARG RUST_TOOLCHAIN=1.88.0
# openssl-sys needs libssl-dev headers + openssl.pc. archive.ubuntu.com is unreachable
# from this build network, so libssl-dev is VENDORED as a local .deb (jammy/amd64
# 3.0.2-0ubuntu1.23 -- exactly matches the base image's installed libssl3, so its only
# Depends is already satisfied) and installed offline with dpkg -i. pkg-config/gcc/git/
# curl are already in the base image; rustup + crates.io + the router repo are reachable.
# Default ON now that the dep is satisfied; --build-arg WITH_ROUTER=0 falls back to the
# toy proxy (PROXY_TYPE=moriio_toy, no router binary).
ARG WITH_ROUTER=1
COPY docker/debs/libssl-dev_3.0.2-0ubuntu1.23_amd64.deb /tmp/libssl-dev.deb
RUN if [ "${WITH_ROUTER}" != "1" ]; then \
      rm -f /tmp/libssl-dev.deb; \
      echo "WITH_ROUTER=0: skipping vllm-router (use PROXY_TYPE=moriio_toy for 1P1D)" | tee -a /app/versions.txt; \
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
# 4c. Robustness patch: vllm/platforms/rocm.py resolves the GCN arch at MODULE
#     LOAD via amdsmi; on failure it calls logger.warning_once(), which imports
#     vllm.distributed.parallel_state -> vllm.platforms.current_platform WHILE
#     vllm.platforms is still initializing -> circular ImportError -> hard crash
#     (seen on concurrent decode-rank startup when amdsmi_get_gpu_asic_info races
#     with AMDSMI_STATUS_FILE_ERROR). Downgrading warning_once -> warning in this
#     one file breaks the import-time cycle so the intended torch.cuda fallback runs.
#     Placed after the router stage so it doesn't invalidate the cached cargo build.
# -----------------------------------------------------------------------------
# Note: verified via grep only -- importing rocm.py at build time forces the arch
# resolver, but the BuildKit sandbox has no GPU (amdsmi + torch.cuda both fail), so
# a live import here is not representative. Two edits:
#   (1) warning_once -> warning     : breaks the import-time circular dependency.
#   (2) torch.cuda fallback -> env  : returns ${VLLM_GCN_ARCH:-gfx942} instead of
#       initializing CUDA (which can race/mis-place devices at concurrent DP startup).
RUN ROCM_PY=/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py && \
    test -f "$ROCM_PY" && \
    sed -i 's/logger\.warning_once(/logger.warning(/g' "$ROCM_PY" && \
    sed -i 's#return torch\.cuda\.get_device_properties("cuda")\.gcnArchName#return __import__("os").environ.get("VLLM_GCN_ARCH", "gfx942")#' "$ROCM_PY" && \
    if grep -q 'logger.warning_once(' "$ROCM_PY"; then echo "PATCH FAILED: warning_once still present" >&2; exit 1; fi && \
    if grep -q 'torch.cuda.get_device_properties("cuda").gcnArchName' "$ROCM_PY"; then echo "PATCH FAILED: torch.cuda fallback still present" >&2; exit 1; fi && \
    grep -q 'VLLM_GCN_ARCH' "$ROCM_PY" && \
    python3 -m py_compile "$ROCM_PY" && \
    echo "PATCH OK: rocm.py GCN-arch circular-import + torch.cuda fallback fixed" && \
    echo "PATCH: rocm.py warning_once->warning + torch.cuda->env(VLLM_GCN_ARCH:-gfx942)" >> /app/versions.txt

# -----------------------------------------------------------------------------
# 5. caches + MoRI JIT scrub
# -----------------------------------------------------------------------------
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr
RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED" >> /app/versions.txt
RUN cat /app/versions.txt 2>/dev/null | tail -20 || true
