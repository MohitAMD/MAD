# syntax=docker/dockerfile:1.7
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# =============================================================================
# GLM-5.1-FP8 — Disaggregated (WideEP) serving on AMD MI300X (gfx942)
# FULLY SELF-CONTAINED, FROM-SCRATCH build (no prebuilt vllm-rocm base needed).
#
# This is the from-scratch sibling of GLM5.1-FP8.disagg.MI300X.share.Dockerfile:
# it builds the entire pinned stack itself, starting only from the public ROCm
# OS image, then applies the two GLM disagg patches. Build context MUST be the
# MAD repo root (so the COPY docker/patches/... and scripts/... resolve):
#
#   cd /home/mdeopuja/cohere/MAD
#   DOCKER_BUILDKIT=1 docker build \
#     -f docker/GLM5.1-FP8.disagg.MI300X.from-scratch.Dockerfile \
#     -t glm5.1-fp8-disagg:mi300x-fromscratch .
#
# Pinned components (identical to the recipe):
#   Base OS         rocm/dev-ubuntu-22.04:7.2.3-complete   (ROCm 7.2.3)
#   Python          3.12
#   PyTorch         2.11.0  ROCm fork @ d0c8b1f3   (+ vision v0.24.1, audio v2.9.0)
#   Triton          ROCm triton @ 0f380657  (+ cherry-pick 555d04f = triton#8991)
#   FlashAttention  @ 0e60e394
#   AITER           v0.1.18 (d6de776...)   + flydsl 0.2.4
#   MoRI (amd_mori) 1.1.2.dev43+g42e895472  (built from ROCm/mori @ 42e895472b08)
#   vLLM            0.25.1 (built from source) + PR#47766 (6-field sparse-MLA key)
#   vllm-router     raviguptaamd/router @ ravgupta/discovery-dp-rank-roundrobin
#
# GLM disagg patches (why: DP=8/TP=1 => gqa_ratio=64, the opposite of the TP=8
#   gqa=8 regime that vLLM#36855 / aiter#2821 target):
#   Patch A  aiter/mla.py: gfx942 gqa64 fp8 qseqlen1 native-qh64 (#3188) GPU-faults
#            at page_size=1 -> fold to qh16 (docker/patches/patch_aiter_mla_qh64_fold.py)
#   Patch B  vLLM rocm_aiter_mla_sparse.py: force use_persistent=True (gqa64 fp8 has
#            no non-persistent kernel; PR#47766 makes persistent correct across
#            chunked prefill) (docker/patches/patch_glm_dsa_force_persistent.py)
#
# NOTE: the official vLLM ROCm pipeline (docker/Dockerfile.rocm) additionally
# builds RIXL/DeepEP/UCX. This recipe uses MoRI (mori.io) as the KV connector and
# WideEP all2all, so those are intentionally omitted to keep the from-scratch
# build tractable. If you need DeepEP/RIXL, build via the upstream two-stage
# Dockerfile.rocm_base + Dockerfile.rocm instead.
# =============================================================================

ARG BASE_IMAGE=rocm/dev-ubuntu-22.04:7.2.3-complete
# ---- pinned refs (overridable) ----------------------------------------------
ARG TRITON_BRANCH="0f380657"
ARG TRITON_REPO="https://github.com/ROCm/triton.git"
ARG PYTORCH_BRANCH="d0c8b1f3"
ARG PYTORCH_REPO="https://github.com/ROCm/pytorch.git"
ARG PYTORCH_VISION_BRANCH="v0.24.1"
ARG PYTORCH_VISION_REPO="https://github.com/pytorch/vision.git"
ARG PYTORCH_AUDIO_BRANCH="v2.9.0"
ARG PYTORCH_AUDIO_REPO="https://github.com/pytorch/audio.git"
ARG FA_BRANCH="0e60e394"
ARG FA_REPO="https://github.com/Dao-AILab/flash-attention.git"
ARG AITER_BRANCH="v0.1.18"
ARG AITER_REPO="https://github.com/ROCm/aiter.git"
ARG MORI_REF="42e895472b08"
ARG MORI_REPO="https://github.com/ROCm/mori.git"
ARG VLLM_BRANCH="v0.25.1"
ARG VLLM_REPO="https://github.com/vllm-project/vllm.git"

# =============================================================================
# STAGE 0 — base: ROCm 7.2.3 + Python 3.12 + build tooling
# (from vllm-shiksha/docker/Dockerfile.rocm_base "base" stage)
# =============================================================================
FROM ${BASE_IMAGE} AS base
ENV PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:
# GLM disagg targets MI300X only; narrow arch to gfx942 to cut build time.
ARG PYTORCH_ROCM_ARCH=gfx942
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV AITER_ROCM_ARCH=gfx942
ENV MORI_GPU_ARCHS=gfx942
ENV HSA_NO_SCRATCH_RECLAIM=1
ARG PYTHON_VERSION=3.12
ENV PYTHON_VERSION=${PYTHON_VERSION}
RUN mkdir -p /app
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
# Some build environments block outbound HTTP (port 80) but allow HTTPS (443).
# Point apt at HTTPS mirrors so the from-scratch base can install its packages.
RUN sed -i -e 's|http://archive.ubuntu.com/ubuntu|https://iad-ad-1.clouds.archive.ubuntu.com/ubuntu|g' \
           -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
           /etc/apt/sources.list
RUN apt-get update -y \
    && apt-get install -y software-properties-common git curl sudo vim less libgfortran5 libopenmpi-dev libpci-dev liblzma-dev pkg-config \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && sed -i 's|http://ppa.launchpad|https://ppa.launchpad|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true; \
    apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
       python${PYTHON_VERSION}-lib2to3 python-is-python3  \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version
RUN pip install -U packaging 'cmake<4' ninja wheel 'setuptools<80' pybind11 Cython
RUN apt-get update && apt-get install -y libjpeg-dev libsox-dev libsox-fmt-all sox && rm -rf /var/lib/apt/lists/*

# =============================================================================
# STAGE 1 — build_triton  (ROCm triton @ 0f380657 + cherry-pick 555d04f)
# =============================================================================
FROM base AS build_triton
ARG TRITON_BRANCH
ARG TRITON_REPO
RUN git clone ${TRITON_REPO}
RUN cd triton \
    && git checkout ${TRITON_BRANCH} \
    && git config --global user.email "you@example.com" && git config --global user.name "Your Name" \
    && git cherry-pick 555d04f \
    && if [ ! -f setup.py ]; then cd python; fi \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && mkdir -p /app/install && cp dist/*.whl /app/install
RUN if [ -d triton/python/triton_kernels ]; then pip install build && cd triton/python/triton_kernels \
    && python3 -m build --wheel && cp dist/*.whl /app/install; fi

# =============================================================================
# STAGE 2 — build_amdsmi
# =============================================================================
FROM base AS build_amdsmi
RUN cd /opt/rocm/share/amd_smi && pip wheel . --wheel-dir=dist
RUN mkdir -p /app/install && cp /opt/rocm/share/amd_smi/dist/*.whl /app/install

# =============================================================================
# STAGE 3 — build_pytorch  (ROCm fork @ d0c8b1f3 + vision v0.24.1 + audio v2.9.0)
# =============================================================================
FROM base AS build_pytorch
ARG PYTORCH_BRANCH
ARG PYTORCH_VISION_BRANCH
ARG PYTORCH_AUDIO_BRANCH
ARG PYTORCH_REPO
ARG PYTORCH_VISION_REPO
ARG PYTORCH_AUDIO_REPO
RUN apt-get update && apt-get install -y pkg-config liblzma-dev
RUN git clone ${PYTORCH_REPO} pytorch
RUN cd pytorch && git checkout ${PYTORCH_BRANCH}
RUN cd pytorch && pip install -r requirements.txt && git submodule update --init --recursive
RUN cd pytorch && python3 tools/amd_build/build_amd.py \
    && CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl
RUN git clone ${PYTORCH_VISION_REPO} vision
RUN cd vision && git checkout ${PYTORCH_VISION_BRANCH} \
    && python3 setup.py bdist_wheel --dist-dir=dist && pip install dist/*.whl
RUN git clone ${PYTORCH_AUDIO_REPO} audio
RUN cd audio && git checkout ${PYTORCH_AUDIO_BRANCH} \
    && git submodule update --init --recursive && pip install -r requirements.txt \
    && python3 setup.py bdist_wheel --dist-dir=dist && pip install dist/*.whl
RUN mkdir -p /app/install && cp /app/pytorch/dist/*.whl /app/install \
    && cp /app/vision/dist/*.whl /app/install \
    && cp /app/audio/dist/*.whl /app/install

# =============================================================================
# STAGE 4 — build_fa  (FlashAttention @ 0e60e394)
# =============================================================================
FROM base AS build_fa
ARG FA_BRANCH
ARG FA_REPO
RUN --mount=type=bind,from=build_pytorch,src=/app/install/,target=/install \
    pip install /install/*.whl
RUN git clone ${FA_REPO}
RUN cd flash-attention \
    && git checkout ${FA_BRANCH} \
    && git submodule update --init \
    && GPU_ARCHS=$(echo ${PYTORCH_ROCM_ARCH} | sed -e 's/;gfx1[0-9]\{3\}//g') python3 setup.py bdist_wheel --dist-dir=dist
RUN mkdir -p /app/install && cp /app/flash-attention/dist/*.whl /app/install

# =============================================================================
# STAGE 5 — build_aiter  (v0.1.18 + flydsl 0.2.4, prebuilt kernels)
# =============================================================================
FROM base AS build_aiter
ARG AITER_BRANCH
ARG AITER_REPO
RUN --mount=type=bind,from=build_pytorch,src=/app/install/,target=/install \
    pip install /install/*.whl
RUN pip install --no-deps -U "flydsl==0.2.4"
RUN git clone --recursive --branch ${AITER_BRANCH} ${AITER_REPO}
RUN cd aiter \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt
RUN pip install pyyaml && cd aiter \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.1.18 PREBUILD_KERNELS=1 AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=${AITER_ROCM_ARCH} \
       python3 setup.py bdist_wheel --dist-dir=dist \
    && ls /app/aiter/dist/*.whl
RUN mkdir -p /app/install && cp /app/aiter/dist/*.whl /app/install

# =============================================================================
# STAGE 6 — build_mori  (amd_mori @ 42e895472b08)
# =============================================================================
FROM base AS build_mori
ARG MORI_REF
ARG MORI_REPO
RUN --mount=type=bind,from=build_pytorch,src=/app/install/,target=/install \
    pip install /install/*.whl
ENV BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF
# mori's pyproject declares its version dynamically via setuptools_scm (build
# requirement). `python3 setup.py bdist_wheel` bypasses PEP 517 isolation, so scm
# must already be present or the wheel silently versions as 0.0.0. Install it and
# keep git tags so scm resolves the real version (e.g. 1.1.2.dev43+g42e895472).
RUN pip install "setuptools_scm[toml]>=6.2"
RUN git clone --recursive ${MORI_REPO}
RUN cd mori \
    && git checkout ${MORI_REF} \
    && git fetch --tags --force \
    && git submodule update --init --recursive \
    && python3 setup.py bdist_wheel --dist-dir=dist && ls /app/mori/dist/*.whl \
    && case "$(ls dist/*.whl)" in *-0.0.0-*|*-0.1.0-*) echo "ERROR: mori scm version unresolved ($(ls dist/*.whl))" >&2; exit 1;; esac
RUN mkdir -p /app/install && cp /app/mori/dist/*.whl /app/install

# =============================================================================
# STAGE 7 — rocm_final: ROCm stack = base + all wheels installed
# =============================================================================
FROM base AS rocm_final
RUN --mount=type=bind,from=build_triton,src=/app/install/,target=/install cp /install/*.whl /tmp/ && pip install /tmp/*.whl && rm -f /tmp/*.whl
RUN --mount=type=bind,from=build_amdsmi,src=/app/install/,target=/install cp /install/*.whl /tmp/ && pip install /tmp/*.whl && rm -f /tmp/*.whl
RUN --mount=type=bind,from=build_pytorch,src=/app/install/,target=/install cp /install/*.whl /tmp/ && pip install /tmp/*.whl && rm -f /tmp/*.whl
RUN --mount=type=bind,from=build_fa,src=/app/install/,target=/install cp /install/*.whl /tmp/ && pip install /tmp/*.whl && rm -f /tmp/*.whl
RUN --mount=type=bind,from=build_aiter,src=/app/install/,target=/install cp /install/*.whl /tmp/ && pip install /tmp/*.whl && rm -f /tmp/*.whl
RUN --mount=type=bind,from=build_mori,src=/app/install/,target=/install cp /install/*.whl /tmp/ && pip install /tmp/*.whl && rm -f /tmp/*.whl
# aiter's arch detection (chip_info.py get_gfx_custom_op) reads GPU_ARCHS and only
# falls back to rocminfo when it is "native"/unset. On a GPU-less build node rocminfo
# fails, so pin GPU_ARCHS=gfx942 for all import-time codegen. Runtime dispatch uses
# get_gfx_runtime() which always re-detects the live GPU, so this is safe at runtime.
ENV GPU_ARCHS=gfx942
# aiter also calls get_gfx_runtime() at import time (aiter/utility/dtypes.py), which
# ALWAYS shells out to rocminfo and ignores GPU_ARCHS. That aborts every `import aiter`
# on a GPU-less build node. Install a transparent rocminfo wrapper: it delegates to the
# real binary (correct on real GPU nodes at runtime) and only emits a canned gfx942
# stanza when the real one fails (i.e. no GPU present, build time only).
COPY docker/patches/rocminfo_buildshim.sh /tmp/rocminfo_buildshim.sh
RUN RI="$(readlink -f "$(command -v rocminfo)")" \
    && if [ ! -e "${RI}.real" ]; then cp -a "$RI" "${RI}.real"; fi \
    && install -m 0755 /tmp/rocminfo_buildshim.sh "$RI" \
    && rm -f /tmp/rocminfo_buildshim.sh \
    && rocminfo | grep -qi gfx942
# aiter arch_info.py resolves the arch via triton's active driver at import time and
# falls back to jax when triton has no active target (GPU-less node). Rewrite that
# fallback to read GPU_ARCHS; real GPU nodes take the primary triton path unchanged.
COPY docker/patches/patch_aiter_gpuless_import.py /tmp/patch_aiter_gpuless_import.py
RUN python3 /tmp/patch_aiter_gpuless_import.py && rm -f /tmp/patch_aiter_gpuless_import.py
RUN python3 -c "import torch, aiter, mori, mori.io, mori.ops; print('rocm stack OK: torch', torch.__version__)"

# =============================================================================
# STAGE 8 — vllm_build: build vLLM 0.25.1 from source (rust frontend + ROCm ext)
# =============================================================================
FROM rocm_final AS vllm_build
ARG VLLM_BRANCH
ARG VLLM_REPO
ENV VLLM_TARGET_DEVICE=rocm
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH=/root/.cargo/bin:${PATH}
RUN apt-get update -q -y && apt-get install -q -y --no-install-recommends \
        ca-certificates curl unzip ccache mold libnuma-dev git \
    && rm -rf /var/lib/apt/lists/*
# uv for fast dependency resolution (as upstream Dockerfile.rocm)
RUN curl -LsSf --retry 3 --retry-delay 5 https://astral.sh/uv/install.sh -o /tmp/uv-install.sh \
    && env UV_INSTALL_DIR="/usr/local/bin" sh /tmp/uv-install.sh && rm -f /tmp/uv-install.sh && uv --version
ENV UV_HTTP_TIMEOUT=500 UV_INDEX_STRATEGY="unsafe-best-match" UV_LINK_MODE=copy
# Fetch vLLM 0.25.1 source
RUN git clone ${VLLM_REPO} vllm \
    && cd vllm && git fetch -v --prune -- origin ${VLLM_BRANCH} && git checkout FETCH_HEAD
# Rust toolchain + protoc for the vllm-rs frontend
RUN cd vllm \
    && ./tools/install_protoc.sh \
    && TOOLCHAIN="$(grep '^channel' rust-toolchain.toml | sed 's/.*= *"\(.*\)"/\1/')" \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${TOOLCHAIN}" \
    && rustc --version && cargo --version
# Build the rust frontend (vllm-rs) then the ROCm wheel
RUN cd vllm \
    && uv pip install --system -r requirements/build/rust.txt \
    && CARGO_BUILD_JOBS=4 bash build_rust.sh \
    && test -x vllm/vllm-rs
RUN cd vllm \
    && uv pip install --system -r requirements/rocm.txt \
    && export CCACHE_BASEDIR="$PWD" \
    && LDFLAGS="-fuse-ld=mold" MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
       python3 setup.py bdist_wheel --dist-dir=dist \
    && ls dist/*.whl
RUN cd vllm && (pip uninstall -y vllm 2>/dev/null || true) && pip install --no-deps dist/*.whl
RUN python3 -c "import vllm; print('vLLM', vllm.__version__)"

# =============================================================================
# STAGE 9 — final: vLLM disagg overlay (PR#47766 + csfix + router + rocm.py)
#           then GLM disagg Patch A (aiter qh16 fold) + Patch B (force persistent)
# =============================================================================
FROM vllm_build AS final
ENTRYPOINT []
WORKDIR /app
ARG PYTORCH_ROCM_ARCH=gfx942
RUN mkdir -p /app && echo "GLM-5.1-FP8 disagg MI300X — FROM-SCRATCH (PT d0c8b1f3, Triton 0f380657, FA 0e60e394, aiter v0.1.18, mori 42e895472, vLLM 0.25.1)" >> /app/versions.txt

# --- 0) MoRIIO KV-connector + WideEP proxy runtime deps -----------------------
# vLLM's MoRIIO connector does `import msgpack` at module import time, and the
# WideEP proxy path uses quart/aiohttp/pyzmq/blinker. The upstream Dockerfile.rocm
# mori_base stage installs these; the from-scratch base doesn't carry them, so add
# them here (else every engine dies in create_engine_config: ModuleNotFoundError:
# No module named 'msgpack').
RUN pip install --no-cache-dir --ignore-installed blinker \
    && pip install --no-cache-dir msgpack quart aiohttp pyzmq \
    && python3 -c "import msgpack, quart, aiohttp, zmq; print('MoRIIO proxy deps OK')" \
    && echo "PATCH: MoRIIO/WideEP runtime deps (msgpack quart aiohttp pyzmq blinker)" >> /app/versions.txt

# --- 1) PR#47766: 6-field sparse-MLA metadata key ----------------------------
COPY docker/patches/patch_pr47766_v024.py /tmp/patch_pr47766.py
RUN python3 /tmp/patch_pr47766.py && \
    ROCM_SPARSE=$(python3 -c "import importlib.util,os;print(os.path.join(os.path.dirname(importlib.util.find_spec('vllm').origin),'v1/attention/backends/mla/rocm_aiter_mla_sparse.py'))") && \
    grep -q 'clamped_context_lens.tobytes()' "$ROCM_SPARSE" && \
    grep -q 'seg_lengths.tobytes()' "$ROCM_SPARSE" && \
    python3 -m py_compile "$ROCM_SPARSE" && \
    echo "PATCH: PR#47766 (6-field sparse-MLA metadata key)" >> /app/versions.txt && \
    rm -f /tmp/patch_pr47766.py

# --- 1b) cold-start fixes: DSA indexer boot-warmup + scheduler stale-req guard -
COPY scripts/vllm_dissag/apply_glm_dsa_indexer_warmup_fix.py /tmp/patch_dsa_warmup.py
RUN VLLM_DIR=$(python3 -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('vllm').origin))") && \
    python3 /tmp/patch_dsa_warmup.py "$VLLM_DIR" \
      && echo "PATCH: GLM DSA indexer boot-warmup" >> /app/versions.txt \
      || echo "WARN: DSA indexer warmup patch did not apply" >> /app/versions.txt; \
    rm -f /tmp/patch_dsa_warmup.py
COPY docker/patches/patch_glm_sched_kv_xfer_stale_guard.py /tmp/patch_sched_guard.py
RUN python3 /tmp/patch_sched_guard.py \
      && echo "PATCH: scheduler KV-xfer stale-req guard" >> /app/versions.txt \
      || echo "WARN: scheduler stale-req guard did not apply" >> /app/versions.txt; \
    rm -f /tmp/patch_sched_guard.py

# --- 2) vllm-router (WideEP disagg proxy, DP-rank round-robin) ----------------
ARG WITH_ROUTER=1
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
ARG ROUTER_REF=ravgupta/discovery-dp-rank-roundrobin
ARG RUST_TOOLCHAIN=1.88.0
RUN if [ "${WITH_ROUTER}" != "1" ]; then \
      echo "WITH_ROUTER=0: skipping vllm-router" | tee -a /app/versions.txt; \
    else set -e && \
      (apt-get update -y && apt-get install -y --no-install-recommends libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*) && \
      export PATH="/root/.cargo/bin:${PATH}" && \
      if ! command -v cargo >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
      fi && \
      rm -rf /tmp/vllm-router-src && \
      git clone --filter=blob:none "${ROUTER_REPO}" /tmp/vllm-router-src && \
      cd /tmp/vllm-router-src && git checkout "${ROUTER_REF}" && \
      cargo build --release && \
      install -m 755 target/release/vllm-router /usr/local/bin/vllm-router && \
      vllm-router --help 2>&1 | grep -q moriio && \
      echo "VLLM_ROUTER_REF=${ROUTER_REF}@$(git -C /tmp/vllm-router-src rev-parse HEAD)" >> /app/versions.txt && \
      rm -rf /tmp/vllm-router-src; \
    fi

# --- 3) rocm.py GCN-arch circular-import boot fix (best-effort) ---------------
RUN ROCM_PY=$(python3 -c "import importlib.util,os;print(os.path.join(os.path.dirname(importlib.util.find_spec('vllm').origin),'platforms/rocm.py'))") && \
    test -f "$ROCM_PY" && \
    sed -i 's/logger\.warning_once(/logger.warning(/g' "$ROCM_PY" && \
    sed -i 's#return torch\.cuda\.get_device_properties("cuda")\.gcnArchName#return __import__("os").environ.get("VLLM_GCN_ARCH", "gfx942")#' "$ROCM_PY" && \
    python3 -m py_compile "$ROCM_PY" && \
    echo "PATCH: rocm.py GCN-arch circular-import fix (best-effort)" >> /app/versions.txt

# --- 4) PATCH A — aiter MLA dispatch: fold gfx942 gqa64 fp8 decode qh64 -> qh16
COPY docker/patches/patch_aiter_mla_qh64_fold.py /tmp/patch_aiter_qh64.py
RUN python3 /tmp/patch_aiter_qh64.py && \
    echo "PATCH A: aiter gfx942 gqa64 fp8 decode folds to qh16 (avoids #3188 qh64 page_size=1 OOB)" >> /app/versions.txt && \
    rm -f /tmp/patch_aiter_qh64.py

# --- 5) PATCH B — vLLM sparse-MLA: force use_persistent=True for gqa64 --------
COPY docker/patches/patch_glm_dsa_force_persistent.py /tmp/patch_force_persistent.py
RUN python3 /tmp/patch_force_persistent.py && \
    echo "PATCH B: vLLM sparse-MLA use_persistent forced True (gqa64 fp8 needs persistent; rely on PR#47766)" >> /app/versions.txt && \
    rm -f /tmp/patch_force_persistent.py

# --- 6) cache locations + JIT scrub so kernels recompile against patched dispatch
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr
RUN rm -rf /opt/vllm_cache/aiter_jit /root/.aiter /root/.mori /tmp/mori_jit_* /tmp/vllm_cache*/aiter_jit 2>/dev/null || true; \
    mkdir -p /root/.mori && echo "JIT_SCRUBBED" >> /app/versions.txt
RUN python3 -c "import vllm, mori, mori.io, mori.ops; print('import OK: vllm', vllm.__version__, '+ MoRI (aiter@runtime)')"
RUN cat /app/versions.txt 2>/dev/null | tail -30 || true

# Runtime (set by the launcher):
#   VLLM_ROCM_USE_AITER=1  VLLM_GCN_ARCH=gfx942
#   Disagg: prefill DP=8 TP=1 + decode DP=8 TP=1 (EP8 per role; EP16 for 2P2D)
#   KV connector: MoRIIO (mori.io)   Router: vllm-router (DP-rank round-robin)
#   Serve containers must use --network host.
# =============================================================================
