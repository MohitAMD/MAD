#!/bin/bash
# =============================================================================
# CONTROL for the disagg determinism study: single-node, TP=8, NO WideEP, NO KV
# transfer (colocated) -- matches the user's baseline (QMpp.log). Same image as the
# disagg runs (vllm-disagg:glmv5.1-v0.24-local-batonfix) so kernels/patches are
# identical; the ONLY variable vs 1P1D disagg is colocated-vs-disaggregated.
# Runs the NIAH determinism probe: same prompt x6 per size (8k/20k/96k), /v1/completions.
#
# Self-contained (does NOT use the disagg recipe): it `vllm serve --tp 8` in one
# container on one node, waits for readiness, runs benchmark_niah.py, tears down.
#
# Submit:
#   sbatch -p amd-rccl -N 1 --gres=gpu:8 --time=8:00:00 --requeue \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-singlenode-tp8-determ --export=ALL \
#          glm5.1_notes/sbatch_singlenode_tp8_determinism_probe.sh
# =============================================================================
#SBATCH --job-name=glm_singlenode_tp8_determ
#SBATCH --partition=amd-rccl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
IMAGE=vllm-disagg:glmv5.1-v0.24-pr47766
WANT_ID=5f0f3679b47e
TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-pr47766.tar
MODEL=/mnt/m2m_nobackup/models_blog/GLM-5.1-FP8
[ -f "$MODEL/config.json" ] || MODEL=/shared_inference/models_blog/GLM-5.1-FP8
CTR="glm_sn_tp8_${SLURM_JOB_ID}"
LOGDIR="/shared_inference/mdeopuja/model_blog_logs/${SLURM_JOB_ID}"
JITCACHE="/mnt/m2m_nobackup/mdeopuja/vllm_jit_cache_sn_tp8"   # node-local, persists across retries
mkdir -p "$LOGDIR"
echo "node=$(hostname) model=$MODEL image=$IMAGE"

# --- stage image (by id) ------------------------------------------------------
have=$(docker images -q "$IMAGE" 2>/dev/null | head -c12)
if [ "$have" != "$WANT_ID" ]; then
  echo "loading image from tar..."; docker load -i "$TAR" >/dev/null 2>&1 && echo "loaded" || { echo "LOAD_FAILED"; exit 1; }
fi

# --- cleanup: stale container + root-owned stale aiter locks -------------------
docker rm -f "$CTR" >/dev/null 2>&1 || true
mkdir -p "$JITCACHE"
docker run --rm -v /mnt/m2m_nobackup:/mnt/m2m_nobackup --entrypoint bash "$IMAGE" -c \
  'for L in /mnt/m2m_nobackup/*/vllm_jit_cache*/*/aiter_jit/build/lock_* /mnt/m2m_nobackup/*/vllm_jit_cache*/aiter_jit/build/lock_*; do rm -f "$L" 2>/dev/null; done; true' 2>/dev/null || true

# --- launch vllm serve (TP=8, colocated, no EP, no KV transfer) ---------------
# Matches the baseline: /v1/completions, fp8 KV. Persistent jit cache mounted at
# /opt/vllm_cache so retries are warm. AITER baton self-heal is baked in the image.
docker run -d --name "$CTR" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 128g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /shared_inference:/shared_inference -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  -v "$REPO":/repo -v "$JITCACHE":/opt/vllm_cache \
  -e VLLM_GCN_ARCH=gfx942 -e AITER_BATON_TIMEOUT=300 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 -e VLLM_ROCM_USE_AITER_MLA=1 \
  -e VLLM_ROCM_USE_AITER_RMSNORM=1 -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 \
  -e VLLM_ROCM_USE_AITER_PAGED_ATTN=0 -e VLLM_USE_AITER_TRITON_SILU_MUL=0 \
  -e AITER_JIT_DIR=/opt/vllm_cache/aiter_jit -e VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
  -e TRITON_CACHE_DIR=/opt/vllm_cache/triton -e COMGR_CACHE_DIR=/opt/vllm_cache/comgr \
  --entrypoint bash "$IMAGE" -c "
    mkdir -p /opt/vllm_cache/aiter_jit /opt/vllm_cache/vllm /opt/vllm_cache/triton /opt/vllm_cache/comgr
    vllm serve '$MODEL' \
      --tensor-parallel-size 8 \
      --trust-remote-code \
      --kv-cache-dtype fp8 \
      --max-model-len 202752 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.9 \
      --port 30000 2>&1 | tee /shared_inference/mdeopuja/model_blog_logs/${SLURM_JOB_ID}/serve.log
  "

# --- wait for server readiness (cold bring-up: weights + JIT, up to ~50 min) ---
echo "waiting for server readiness..."
ready=0
for i in $(seq 1 300); do
  if grep -q "Application startup complete" "$LOGDIR/serve.log" 2>/dev/null; then ready=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CTR}$"; then echo "serve container exited early"; break; fi
  sleep 20
done
if [ "$ready" != "1" ]; then
  echo "SERVER NOT READY -- tail serve.log:"; tail -30 "$LOGDIR/serve.log" 2>/dev/null
  docker rm -f "$CTR" >/dev/null 2>&1 || true
  exit 1
fi
echo "server ready; sanity check:"; curl -s -m 30 http://127.0.0.1:30000/v1/models 2>&1 | head -c 200; echo

# --- run NIAH determinism probe (same prompt x6) inside the container ----------
docker exec \
  -e NIAH_URL=http://127.0.0.1:30000/v1/completions \
  -e NIAH_MODEL="$MODEL" \
  -e NIAH_ENDPOINT=completions \
  -e NIAH_COMBOS="8000/2048,20000/2048,96000/2048" \
  -e NIAH_SEEDS=0 -e NIAH_REPEAT=6 -e NIAH_TIMEOUT=5400 -e NIAH_WARMUP=1 \
  -e NIAH_TOPO="MI300X TP=8 single-node (colocated, no WideEP) [DETERMINISM PROBE same prompt x6]" \
  "$CTR" python3 /repo/benchmark_niah.py 2>&1 | tee "$LOGDIR/niah_singlenode_tp8.log"

echo "=== teardown ==="
docker rm -f "$CTR" >/dev/null 2>&1 || true
echo "DONE_SN_PROBE rc=$?"
