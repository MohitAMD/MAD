# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
# =============================================================================
# vllm_disagg_inference.glmv5.1_v0.24.ubuntu.amd.Dockerfile
#   GLM-5.1-FP8 (MLA + DeepSeek Sparse Attention) MoRI-EP WideEP disagg image,
#   REBASED onto the OFFICIAL vLLM v0.24.0 ROCm release image.
#
#   This is the v0.24 counterpart of vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile
#   (which builds on the rocm/vllm-dev ci_base-0fcd9b99 nightly + the shik-latest fork,
#   vLLM 0.16-dev line). Here the base is the clean upstream release
#   `vllm/vllm-openai-rocm:v0.24.0` (ROCm 7.2.3 / py3.12 / gfx942+gfx950; already bundles
#   PyTorch + AITER + MoRI + FlashAttention), and only the GLM-DSA + MoRI-EP WideEP
#   *delta* is re-applied on top.
#
# -----------------------------------------------------------------------------
# PREREQUISITE (build-blocking): a vLLM branch rebased onto v0.24.0
# -----------------------------------------------------------------------------
#   VLLM_REF below (default `glm5.1-dsa-wideEP_on_v0.24.0`) MUST be a branch =
#   `git checkout v0.24.0` + the GLM-DSA / MoRI-WideEP delta re-applied. Against
#   v0.24.0 (released 2026-06-29) the following are NOT yet in the release and must
#   be cherry-picked / carried on that branch:
#     - vLLM #47766  sparse-MLA persistent metadata keyed per-request ctx/query len
#                    (merged upstream 2026-07-08 > 0.24 cut) -- long-context accuracy. CRITICAL.
#     - vLLM #45324  DSA invalid-token kernel returns -1 (open) -- garbage-token fix.
#     - vLLM #39276  engine_id collision + MoRIIO multi-node disagg DP robustness (open).
#     - vLLM #45043  MoRI-EP inter-node WideEP + MoRIIO write-mode for 2p2d dp=ep=16 (open).
#     - GLM-5.1 DSA dual-KV MoRIIO connector work (per-layer geometry, completion-gate
#       on transfer layers, indexer-KV pairing/transfer) -- the 9 DSA patchers' logic,
#       baked into source (so SKIP_RUNTIME_PATCH=1, no runtime patcher).
#   ALREADY in v0.24.0 (do NOT re-apply): #41751 (mori InterNodeV1LL, merged 2026-05-27).
#   Also verify GlmMoeDsaForCausalLM + sparse_attn_indexer + ROCM_AITER_MLA_SPARSE are
#   present on the branch (port the model + sparse backend if 0.24 lacks GLM-5.1 DSA).
#
# -----------------------------------------------------------------------------
# AITER / MoRI: the v0.24.0 image already bundles both. GLM-5.1 was validated on the
# pinned aiter e03fa6040 (persistent gqa64 fold under #47766) + mori 42e895472b08
# (large-transfer notify at high EP). Defaults REBUILD those pins for correctness; set
# --build-arg REBUILD_AITER=0 / REBUILD_MORI=0 to reuse the base image's bundled builds
# (faster, but only if their versions are GLM-compatible -- verify NIAH before trusting).
# -----------------------------------------------------------------------------
#
#   docker build -f docker/vllm_disagg_inference.glmv5.1_v0.24.ubuntu.amd.Dockerfile \
#     -t <your-registry>/vllm-disagg:glmv5.1-v0.24 .
#   export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg:glmv5.1-v0.24
#
#   WITH_NIXL=1 (default) => builds UCX + RIXL(+nixlbench) + rocSHMEM + DeepEP from
#     source (rixl connector). WITH_NIXL=0 => MoRI-EP only (lean, faster).
#
# STATUS: port target. Validate 1P/1D EP8 + 2P/2D EP16 NIAH 2k-35k before trusting;
# 4P/4D EP32 is a KNOWN OPEN DEFECT on the 0.16 stack (all2all combine at EP32) and
# must be re-checked on 0.24. Use 1P/1D and 2P/2D.
# =============================================================================

# Official vLLM v0.24.0 ROCm release image (pin the digest for reproducibility):
#   vllm/vllm-openai-rocm:v0.24.0
#   @sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28
ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.24.0
FROM ${BASE_IMAGE}

# The upstream vllm-openai image sets ENTRYPOINT ["vllm","serve"]; the disagg launcher
# runs its own recipe, so clear it (matches the sibling GLM Dockerfile).
ENTRYPOINT []
WORKDIR /app

ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
ARG MAX_JOBS=32
ARG WITH_NIXL=1
ARG NIC_COMPILATION_ARCH="cx7"
# Reuse-vs-rebuild toggles for the base image's bundled AITER / MoRI.
ARG REBUILD_AITER=1
ARG REBUILD_MORI=1

RUN mkdir -p /app && echo "BASE_IMAGE=${BASE_IMAGE} (vLLM v0.24.0 ROCm release)" >> /app/versions.txt

# -----------------------------------------------------------------------------
# 1. MoRI: (default) rebuild ROCm/mori @ 42e895472b08 -- MoRI main tip past v1.2.1
#    validated for GLM-5.1 DSA WideEP disagg (v1.2.1 large-transfer notify was
#    insufficient at high EP). Set REBUILD_MORI=0 to keep the v0.24.0 image's bundled MoRI.
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=42e895472b08
ENV MORI_GPU_ARCHS=gfx942
ENV BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF
RUN if [ "${REBUILD_MORI}" != "1" ]; then \
      echo "REBUILD_MORI=${REBUILD_MORI}: keeping base image bundled MoRI" | tee -a /app/versions.txt; \
    else set -e && \
      sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
      sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
      apt-get update && apt-get install -y --no-install-recommends \
          git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates && \
      pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
      (pip uninstall -y amd_mori amd-mori amd-mori-nightly mori 2>/dev/null || true) && \
      rm -rf /tmp/mori-src && \
      git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
      cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
      BUILD_UMBP=OFF pip install . && \
      python3 -c "import mori, mori.io, mori.ops; print('MoRI OK at', mori.__path__[0])" && \
      echo "MORI_REF=${MORI_REF}@$(git -C /tmp/mori-src rev-parse HEAD)" >> /app/versions.txt && \
      rm -rf /tmp/mori-src; \
    fi

# -----------------------------------------------------------------------------
# 2. AITER: (default) rebuild STOCK ROCm/aiter @ e03fa6040 from source. Under vLLM
#    #47766 the sparse-MLA persistent path stays ON, so GLM's gqa=64 decode hits
#    aiter's pre-existing persistent gqa64->16 fold. Set REBUILD_AITER=0 to keep the
#    v0.24.0 image's bundled AITER.
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REF=e03fa6040
# Single guarded RUN (no heredoc inside if/else -> BuildKit-safe). The mla.py check
# is a python3 -c one-liner; stale AITER JIT is cleared in the same layer.
RUN if [ "${REBUILD_AITER}" != "1" ]; then \
      echo "REBUILD_AITER=${REBUILD_AITER}: keeping base image bundled AITER" | tee -a /app/versions.txt; \
    else set -e && \
      echo "Compiling STOCK AITER (no fork) from ${AITER_REPO}@${AITER_REF}" && \
      rm -rf /tmp/aiter-src && \
      git clone --recursive "${AITER_REPO}" /tmp/aiter-src && \
      cd /tmp/aiter-src && git checkout "${AITER_REF}" && git submodule update --init --recursive && \
      (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
      pip install --no-build-isolation --no-deps -v . && \
      pip install --no-deps -U "flydsl>=0.1.7,<0.1.9" && \
      echo "AITER_REF=${AITER_REF}@$(git -C /tmp/aiter-src rev-parse HEAD) (stock ROCm/aiter, no fork)" >> /app/versions.txt && \
      python3 -c "import glob,pathlib; c=glob.glob('/usr/local/lib/python*/dist-packages/aiter/mla.py')+glob.glob('/usr/lib/python*/dist-packages/aiter/mla.py'); assert c,'aiter/mla.py not found after install'; assert 'persistent_mode' in pathlib.Path(c[0]).read_text(),'AITER persistent gqa64 fold MISSING'; print('STOCK AITER OK (persistent fold present):',c[0])" && \
      rm -rf /tmp/aiter-src && \
      rm -rf /opt/vllm_cache/aiter_jit /root/.aiter && echo "cleared stale AITER JIT cache"; \
    fi

# -----------------------------------------------------------------------------
# 3. vLLM: compile from the v0.24.0-rebased GLM-DSA WideEP branch (see PREREQUISITE
#    header). Full source compile: it re-applies the GLM/DSA/MoRIIO delta + the
#    cherry-picked PRs (#47766/#45324/#39276/#45043) on top of v0.24.0, so a .py-only
#    overlay would be ABI-mismatched against the base's compiled vLLM. Fixes are native
#    on this branch -> SKIP_RUNTIME_PATCH=1 (no runtime patcher). Override VLLM_REF to
#    rebuild a different commit; build only committed commits (no working-tree edits).
# -----------------------------------------------------------------------------
ARG VLLM_REPO=https://github.com/raviguptaamd/vllm.git
ARG VLLM_REF=glm5.1-dsa-wideEP_on_v0.24.0
ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    MAX_JOBS=${MAX_JOBS}
RUN rm -rf /tmp/vllm-src && \
    git clone "${VLLM_REPO}" /tmp/vllm-src && \
    cd /tmp/vllm-src && git checkout "${VLLM_REF}" && \
    echo "VLLM_REF=${VLLM_REF}@$(git rev-parse HEAD)" >> /app/versions.txt && \
    pip uninstall -y vllm 2>/dev/null || true && \
    pip install --no-deps --no-build-isolation -v . && \
    python3 -c "import vllm; print('vLLM', vllm.__version__, 'from', vllm.__file__)" && \
    rm -rf /tmp/vllm-src

# Cross-check MoRI + AITER survived the vLLM install (no silent downgrade). When the
# pins were rebuilt, assert the exact commit tag; when reused-from-base, just import.
RUN REBUILD_AITER=${REBUILD_AITER} python3 - <<'PYEOF'
from importlib.metadata import version as v, PackageNotFoundError
import os
def get(names):
    for n in names:
        try: return v(n)
        except PackageNotFoundError: pass
    return None
av = get(("amd-aiter", "amd_aiter", "aiter"))
if os.environ.get("REBUILD_AITER", "1") == "1":
    assert av and "e03fa6040" in av, f"AITER missing/downgraded (want e03fa6040 build): {av!r}"
else:
    assert av, "AITER not importable"
import mori, mori.io, mori.ops
print("Post-vLLM check OK: AITER", av, "+ MoRI importable")
PYEOF

# -----------------------------------------------------------------------------
# 4. vllm-router (DP-rank round-robin + MoRIIO + 2P2D KV-notify dpfix) built in.
#    Source = vllm-project/router PR #181 branch. Pinned Rust toolchain (>=1.88).
# -----------------------------------------------------------------------------
ARG ROUTER_REPO=https://github.com/raviguptaamd/router.git
ARG ROUTER_REF=ravgupta/discovery-dp-rank-roundrobin
ARG RUST_TOOLCHAIN=1.88.0
# openssl-sys needs libssl-dev headers + openssl.pc. archive.ubuntu.com is unreachable
# from this build network, so libssl-dev is VENDORED as a local .deb (jammy/amd64
# 3.0.2-0ubuntu1.23 -- exactly matches the base image's installed libssl3) and installed
# offline with dpkg -i. pkg-config/gcc/git/curl are already in the base image.
COPY docker/debs/libssl-dev_3.0.2-0ubuntu1.23_amd64.deb /tmp/libssl-dev.deb
RUN dpkg -i /tmp/libssl-dev.deb && rm -f /tmp/libssl-dev.deb && \
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
    echo "VLLM_ROUTER_REF=${ROUTER_REPO}@${ROUTER_REF}@$(git -C /tmp/vllm-router-src rev-parse HEAD)" >> /app/versions.txt && \
    rm -rf /tmp/vllm-router-src

# -----------------------------------------------------------------------------
# 4b. WITH_NIXL=1 (default): UCX + RIXL(+nixlbench) + rocSHMEM + DeepEP from source
#     for the rixl connector. Single guarded RUN so WITH_NIXL=0 skips it entirely.
# -----------------------------------------------------------------------------
ENV _ROCM_DIR=/opt/rocm \
    _UCX_SOURCE=https://github.com/ROCm/ucx.git \
    _UCX_BRANCH=da3fac2a \
    _UCX_INSTALL_DIR=/usr/local/ucx/ \
    _RIXL_SOURCE=https://github.com/ROCm/RIXL.git \
    _RIXL_BRANCH=f33a5599 \
    _RIXL_INSTALL_DIR=/usr/local/RIXL/install \
    _NIXLBENCH_INSTALL_DIR=/usr/local/RIXL
RUN if [ "${WITH_NIXL}" != "1" ]; then \
      echo "WITH_NIXL=${WITH_NIXL}: skipping UCX/RIXL/rocSHMEM/DeepEP (MoRI-EP + base DeepEP only)"; \
    else set -e && \
      echo "WITH_NIXL=1: building UCX + RIXL + rocSHMEM + DeepEP" && \
      apt-get update && apt-get install -y \
        autoconf automake libtool autogen pkg-config m4 gcc make \
        librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
        libibverbs-dev rdma-core strace libgflags-dev \
        libaio-dev liburing-dev libcpprest-dev libgrpc-dev libgrpc++-dev \
        libprotobuf-dev protobuf-compiler-grpc wget && \
      pip install meson==0.64.0 "pybind11[global]" pyyaml && \
      cd /tmp && git clone "${_UCX_SOURCE}" && cd ucx && git checkout "${_UCX_BRANCH}" && \
        ./autogen.sh && mkdir -p build && cd build && \
        ../configure --prefix="${_UCX_INSTALL_DIR}" --with-rocm="${_ROCM_DIR}" \
          --disable-go --disable-java --disable-assertions --enable-mt && \
        make -j && make install && \
      cd /tmp && wget -q https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz && \
        tar -xzf v1.14.0.tar.gz && cd googletest-1.14.0 && mkdir -p build && cd build && \
        cmake -DBUILD_SHARED_LIBS=on .. && make -j && make install && \
      cd /tmp && git clone "${_RIXL_SOURCE}" && cd RIXL && git checkout "${_RIXL_BRANCH}" && \
        meson setup build/ --prefix="${_RIXL_INSTALL_DIR}" -Ducx_path="${_UCX_INSTALL_DIR}" \
          -Ddisable_gds_backend=true -Dcudapath_inc="${_ROCM_DIR}/include" -Dcudapath_lib="${_ROCM_DIR}/lib" && \
        cd build && ninja && ninja install && cd /tmp/RIXL && \
        pip install --config-settings=setup-args="-Dcudapath_inc=${_ROCM_DIR}/include" \
                    --config-settings=setup-args="-Dcudapath_lib=${_ROCM_DIR}/lib" \
                    --config-settings=setup-args="-Ducx_path=${_UCX_INSTALL_DIR}" \
                    --config-settings=setup-args="-Ddisable_gds_backend=true" . && \
      cd /tmp && git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git && \
        cd rocm-systems && git sparse-checkout set --cone projects/rocshmem && git checkout develop && \
        mkdir -p /tmp/rocshmem-build && cd /tmp/rocshmem-build && \
        /tmp/rocm-systems/projects/rocshmem/scripts/build_configs/all_backends \
          -DUSE_EXTERNAL_MPI=OFF -DGPU_TARGETS="${GFX_COMPILATION_ARCH}" && \
      cd /tmp && git clone https://github.com/ROCm/DeepEP.git && cd DeepEP && \
        PYTORCH_ROCM_ARCH="${GFX_COMPILATION_ARCH}" CFLAGS="-O3 -fPIC" \
          CXXFLAGS="-O3 -fPIC --offload-arch=${GFX_COMPILATION_ARCH}" HIP_CXX_FLAGS="-O3 -fPIC" \
          python3 setup.py --variant rocm --nic "${NIC_COMPILATION_ARCH}" build develop && \
      echo "WITH_NIXL build complete" >> /app/versions.txt && \
      rm -rf /tmp/ucx /tmp/googletest-1.14.0 /tmp/v1.14.0.tar.gz /tmp/rocm-systems /tmp/rocshmem-build; \
    fi
ENV LD_LIBRARY_PATH="/usr/local/ucx/lib:/usr/local/lib:/usr/local/RIXL/install/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/ucx/bin:${PATH}"

# -----------------------------------------------------------------------------
# 5. Cache locations (structural). No runtime recipe/tuning ENV baked -- the launcher
#    forwards models.yaml + connectors/<connector>.env via `docker -e` at launch, so
#    this image stays a clean binary artifact reusable across models/clusters.
# -----------------------------------------------------------------------------
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

# -----------------------------------------------------------------------------
# 6. CRITICAL: scrub build-time MoRI JIT state (stale .hsaco.lock -> runtime deadlock
#    at ep:0 init). A clean image ships /root/.mori empty -> runtime compiles fresh.
# -----------------------------------------------------------------------------
RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED: /root/.mori + /tmp/mori_jit_* cleared at build end" >> /app/versions.txt

RUN cat /app/versions.txt 2>/dev/null | tail -20 || true
